#!/usr/bin/env python3
"""Roll a reorient checkpoint out in plain CPU MuJoCo, rebuilding its observation from raw state.

    uv run --extra rl python scripts/policy_cpu_rollout.py \
        --policy results/rl/<run>/tensorboard/model_270.pt \
        --morphology-run results/phase1/real_v1/20260916-sv1_u0308_b050_cal \
        [--scene <other frozen_scene.xml>] --steps 250 --out <dir>/<name>.json [--video <name>.mp4]

WHY. Everything the policy has been evaluated on runs in MuJoCo-Warp on the GPU. Deploying it
anywhere else -- the CPU chain environment with the UR5e, the bench -- means rebuilding the
66-dim observation from raw state and applying the same scripted palm lift, finger closing
lerp and residual schedule outside mjlab. This script is that reconstruction, on the policy's
own frozen scene first (where it must reproduce the GPU eval: D6 grip finetune cos 0.97 held),
then on any scene passed with --scene. It writes the same per-step trace as rl_render_rollout.py
(step, cos, z, per-pad N) plus clearance and the raw action, and prints held / aligned at the end.

Observation (obs_mode full, enable_target_axis): joint_pos_rel(15) joint_vel_rel(15)
object_pos(3, tool centre minus palm site in the palm frame) object_pose_actual(7, tool pose in
the palm frame) ref_finger_qpos(9) ref_object_pose(7) actions(9) target_axis_misalign(1).
Schedule (sim steps of 2 ms, policy every 10): palm z ramps by lift_delta_z over lift_ramp_steps
after settle_steps; fingers lerp from the keyframe's qpos (open) to its ctrl (grip) over
settle_steps with the run's easing; the residual (raw action x finger_residual_scale, clipped to
+-clip_actions if the run had one) is added from finger_residual_active_from_step x 10.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
FINGER_JOINTS = ("thumb_yaw", "thumb_mcp", "thumb_pip", "index_yaw", "index_mcp", "index_pip",
                 "middle_yaw", "middle_mcp", "middle_pip")
PALM_JOINTS = ("palm_px", "palm_py", "palm_pz", "palm_rx", "palm_ry", "palm_rz")
CHAIN_BODIES = ("{f}_yaw_frame", "{f}_mcp_frame", "{f}_pip_frame", "{f}_tip")
CAP_R = 0.01055


def load_actor(ckpt_path: Path) -> torch.nn.Module:
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = ck["actor_state_dict"]
    dims = [sd["mlp.0.weight"].shape[1]]
    layers = []
    i = 0
    while f"mlp.{i}.weight" in sd:
        layers.append(torch.nn.Linear(sd[f"mlp.{i}.weight"].shape[1], sd[f"mlp.{i}.weight"].shape[0]))
        dims.append(sd[f"mlp.{i}.weight"].shape[0])
        if f"mlp.{i + 2}.weight" in sd:
            layers.append(torch.nn.ELU())
        i += 2
    mlp = torch.nn.Sequential(*layers)
    mlp.load_state_dict({k.replace("mlp.", ""): v for k, v in sd.items() if k.startswith("mlp.")})
    mlp.eval()
    return mlp


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1; w2, x2, y2, z2 = q2
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def quat_rotate(q, v):
    qw, qvec = q[0], q[1:]
    t = 2.0 * np.cross(qvec, v)
    return v + qw * t + np.cross(qvec, t)


def quat_inv(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def seg_seg(p1, q1, p2, q2):
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    c, b = d1 @ r, d1 @ d2
    denom = a * e - b * b
    s = np.clip((b * f - c * e) / denom, 0, 1) if denom > 1e-12 else 0.0
    t = (b * s + f) / e
    if t < 0:
        t, s = 0.0, np.clip(-c / a, 0, 1)
    elif t > 1:
        t, s = 1.0, np.clip((b - c) / a, 0, 1)
    return np.linalg.norm((p1 + d1 * s) - (p2 + d2 * t))


class CpuPolicy:
    """The checkpoint's observation builder and actor over a CPU MjModel/MjData.

    `obs(k, last_action, palm_dz)` rebuilds the 66-dim vector at policy step k (the benchmark's
    step count, so the reference-trajectory terms line up: the residual is active from the run's
    finger_residual_active_from_step); `act(obs)` is the deterministic mean action, clipped as
    the run was; `finger_target(act, anchor)` is the position set-point per finger joint.
    Scenes without palm joints (the chain's arm scene) report the palm terms as the benchmark
    would see them: pz = palm_dz, everything else 0."""

    def __init__(self, policy: Path, morph_run: Path, m, d, palm_body: str = "palm_pose"):
        import mujoco
        import yaml
        sys.path.insert(0, str(ROOT / "src"))
        from morphohand.rl.reference_trajectory import ReferenceTrajectory
        run_cfg = yaml.safe_load(open(Path(policy).resolve().parent.parent / "config.yaml"))
        env_c, ppo_c = run_cfg["env"], run_cfg.get("ppo", {})
        self.settle = int(env_c.get("settle_steps", 240)); self.ramp = int(env_c.get("lift_ramp_steps", 80))
        self.lift_dz = float(env_c.get("lift_delta_z", 0.1)); self.rscale = float(env_c.get("finger_residual_scale", 0.5))
        self.easing = env_c.get("finger_close_easing", "ease_out_quad"); self.decim = int(env_c.get("decimation", 10))
        self.active_from = int(env_c.get("finger_residual_active_from_step", 0)) * self.decim
        self.clip = ppo_c.get("clip_actions")
        self.blind_terms = tuple(env_c.get("actor_blind_terms", ()) or ())
        self.m, self.d = m, d
        morph = Path(morph_run).resolve()
        self.summ = json.load(open(morph / "summary.json"))
        ref_model = mujoco.MjModel.from_xml_path(str(morph / "frozen_scene.xml"))
        self.ref = ReferenceTrajectory.from_run_dir(morph, ref_model)
        self.jid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in FINGER_JOINTS}
        self.aid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"a_{n}") for n in FINGER_JOINTS}
        palm_j = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in PALM_JOINTS}
        self.have_palm = all(v >= 0 for v in palm_j.values())
        self.qadr = {n: int(m.jnt_qposadr[self.jid[n]]) for n in FINGER_JOINTS}
        self.vadr = {n: int(m.jnt_dofadr[self.jid[n]]) for n in FINGER_JOINTS}
        if self.have_palm:
            self.qadr.update({n: int(m.jnt_qposadr[palm_j[n]]) for n in PALM_JOINTS})
            self.vadr.update({n: int(m.jnt_dofadr[palm_j[n]]) for n in PALM_JOINTS})
        # defaults = the benchmark keyframe (the run's own frozen scene), not the current scene's state
        key = mujoco.mj_name2id(ref_model, mujoco.mjtObj.mjOBJ_KEY, self.summ["keyframe"])
        self.default_q = {}
        for n in list(PALM_JOINTS) + list(FINGER_JOINTS):
            j = mujoco.mj_name2id(ref_model, mujoco.mjtObj.mjOBJ_JOINT, n)
            if j >= 0:
                self.default_q[n] = float(ref_model.key_qpos[key, int(ref_model.jnt_qposadr[j])])
        self.tool = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "screwdriver_medium")
        self.palm_b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, palm_body)
        self.actor = load_actor(Path(policy))
        self.step_dt = float(m.opt.timestep) * self.decim
        self.robot_joints = list(PALM_JOINTS) + list(FINGER_JOINTS)

    def tool_cos(self):
        q = self.d.xquat[self.tool]
        return 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2])

    def pad_forces(self):
        """Contact force magnitude (N) between each fingertip body's geoms and the tool."""
        import mujoco
        m, d = self.m, self.d
        if not hasattr(self, "_tip_geoms"):
            self._tool_geoms = set(np.where(m.geom_bodyid == self.tool)[0].tolist())
            self._tip_geoms = {f: set(np.where(m.geom_bodyid == mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{f}_tip"))[0].tolist())
                               for f in ("thumb", "index", "middle")}
        out = {f: 0.0 for f in self._tip_geoms}
        fbuf = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            for f, gs in self._tip_geoms.items():
                if (c.geom1 in gs and c.geom2 in self._tool_geoms) or (c.geom2 in gs and c.geom1 in self._tool_geoms):
                    mujoco.mj_contactForce(m, d, i, fbuf)
                    out[f] += float(np.linalg.norm(fbuf[:3]))
        return out

    def obs(self, k: int, last_action, palm_dz: float | None = None):
        d = self.d
        jp = np.array([d.qpos[self.qadr[n]] - self.default_q[n] if n in self.qadr else 0.0 for n in self.robot_joints])
        jv = np.array([d.qvel[self.vadr[n]] if n in self.vadr else 0.0 for n in self.robot_joints])
        if not self.have_palm:
            s = k * self.decim
            jp[2] = palm_dz if palm_dz is not None else np.clip((s - self.settle + 1) / self.ramp, 0, 1) * self.lift_dz
        palm_pos, palm_quat = d.xpos[self.palm_b].copy(), d.xquat[self.palm_b].copy()
        tpos, tquat = d.xpos[self.tool].copy(), d.xquat[self.tool].copy()
        pq_inv = quat_inv(palm_quat)
        obj_pos = quat_rotate(pq_inv, tpos - palm_pos)
        rq = quat_mul(pq_inv, tquat)
        # q and -q are one rotation but two input vectors. The benchmark palm is the identity and
        # the tool's quaternion keeps w > 0 through the turn, so that is what the policy saw; the
        # chain's arm scene reports the palm body as (-1, 0, 0, 0) and flips every component.
        if rq[0] < 0:
            rq = -rq
        rel = np.concatenate([obj_pos, rq])
        rb = self.ref.batch_at([k * self.step_dt])
        ref_fq = rb["finger_qpos"][0]
        ref_op = np.concatenate([rb["object_pos"][0], rb["object_quat"][0]])
        mis = np.array([np.arccos(np.clip(self.tool_cos(), -1, 1))])
        o = np.concatenate([jp, jv, obj_pos, rel, ref_fq, ref_op, last_action, mis]).astype(np.float32)
        return self.blind(o)

    # slot layout of the 66-dim vector
    SLOTS = {"joint_pos": (0, 15), "joint_vel": (15, 30), "object_pos": (30, 33), "object_pose_actual": (33, 40),
             "ref_finger_qpos": (40, 49), "ref_object_pose": (49, 56), "actions": (56, 65), "target_axis_misalign": (65, 66)}

    def blind(self, o):
        for t in self.blind_terms:
            lo, hi = self.SLOTS[t]
            o[lo:hi] = 0.0
        return o

    def act(self, o):
        with torch.no_grad():
            a = self.actor(torch.as_tensor(o).unsqueeze(0)).squeeze(0).numpy()
        if self.clip is not None:
            a = np.clip(a, -float(self.clip), float(self.clip))
        return a.astype(np.float32)

    def finger_target(self, act, anchor):
        """anchor: (9,) grip set-point in FINGER_JOINTS order -> (9,) commanded positions."""
        return np.asarray(anchor, dtype=float) + act * self.rscale


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--morphology-run", type=Path, required=True)
    ap.add_argument("--scene", type=Path, default=None, help="frozen scene to roll out in (default: the run's)")
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--out", type=Path, required=True, help="JSON summary; a CSV trace lands beside it")
    ap.add_argument("--video", type=Path, default=None)
    ap.add_argument("--cam", default="0.32,120,-12")
    ap.add_argument("--palm-body", default="palm_pose")
    ap.add_argument("--no-palm-drive", action="store_true", help="do not drive the palm joints (scene has none)")
    ap.add_argument("--jitter-xy", type=float, default=0.0, help="uniform tool spawn jitter, mm, seeded by --seed")
    ap.add_argument("--blind", action="store_true",
                    help="zero the object terms (object_pos, object_pose_actual, target_axis_misalign) as a blinded "
                         "actor would see them: does the policy need the tool pose, or is it acting open loop?")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import mujoco
    import yaml
    torch.set_num_threads(2)   # a 66x512 MLP once per policy step; keep the box's load flat
    sys.path.insert(0, str(ROOT / "src"))
    from morphohand.rl.reference_trajectory import ReferenceTrajectory

    run_cfg = yaml.safe_load(open(a.policy.resolve().parent.parent / "config.yaml"))
    env_c, ppo_c = run_cfg["env"], run_cfg.get("ppo", {})
    settle = int(env_c.get("settle_steps", 240)); ramp = int(env_c.get("lift_ramp_steps", 80))
    lift_dz = float(env_c.get("lift_delta_z", 0.1)); rscale = float(env_c.get("finger_residual_scale", 0.5))
    easing = env_c.get("finger_close_easing", "ease_out_quad"); decim = int(env_c.get("decimation", 10))
    active_from = int(env_c.get("finger_residual_active_from_step", 0)) * decim
    clip = ppo_c.get("clip_actions")
    morph = a.morphology_run.resolve()
    summ = json.load(open(morph / "summary.json"))
    scene = a.scene or (morph / "frozen_scene.xml")
    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)
    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, summ["keyframe"])
    mujoco.mj_resetDataKeyframe(m, d, key)
    if a.jitter_xy > 0:
        rng = np.random.default_rng(a.seed)
        tb_ = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "screwdriver_medium")
        adr = int(m.jnt_qposadr[int(m.body_jntadr[tb_])])
        d.qpos[adr:adr + 2] += rng.uniform(-a.jitter_xy, a.jitter_xy, 2) * 1e-3
    mujoco.mj_forward(m, d)
    ref_model = mujoco.MjModel.from_xml_path(str(morph / "frozen_scene.xml"))
    ref = ReferenceTrajectory.from_run_dir(morph, ref_model)

    jid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in FINGER_JOINTS}
    aid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"a_{n}") for n in FINGER_JOINTS}
    palm_j = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in PALM_JOINTS}
    palm_a = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"a_{n}") for n in PALM_JOINTS}
    have_palm = all(v >= 0 for v in palm_j.values()) and not a.no_palm_drive
    robot_joints = list(PALM_JOINTS) + list(FINGER_JOINTS)   # mjlab entity order = spec order
    qadr = {n: int(m.jnt_qposadr[jid[n]]) for n in FINGER_JOINTS}
    vadr = {n: int(m.jnt_dofadr[jid[n]]) for n in FINGER_JOINTS}
    if have_palm:
        qadr.update({n: int(m.jnt_qposadr[palm_j[n]]) for n in PALM_JOINTS})
        vadr.update({n: int(m.jnt_dofadr[palm_j[n]]) for n in PALM_JOINTS})
    default_q = {n: float(d.qpos[qadr[n]]) for n in qadr}
    open_ctrl = np.array([d.qpos[qadr[n]] for n in FINGER_JOINTS])          # keyframe qpos = open
    grip_ctrl = np.array([m.key_ctrl[key, aid[n]] for n in FINGER_JOINTS])   # keyframe ctrl = grip
    palm_default = np.array([m.key_ctrl[key, palm_a[n]] for n in PALM_JOINTS]) if have_palm else None
    tool = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "screwdriver_medium")
    palm_b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, a.palm_body)
    tool_geoms = set(np.where(m.geom_bodyid == tool)[0].tolist())
    tip_geoms = {f: set(np.where(m.geom_bodyid == mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{f}_tip"))[0].tolist())
                 for f in ("thumb", "index", "middle")}
    chain_b = {f: [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n.format(f=f)) for n in CHAIN_BODIES] for f in ("index", "middle")}
    actor = load_actor(a.policy)
    step_dt = float(m.opt.timestep) * decim

    def pad_forces():
        out = {f: 0.0 for f in tip_geoms}
        fbuf = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            for f, gs in tip_geoms.items():
                if (c.geom1 in gs and c.geom2 in tool_geoms) or (c.geom2 in gs and c.geom1 in tool_geoms):
                    mujoco.mj_contactForce(m, d, i, fbuf)
                    out[f] += float(np.linalg.norm(fbuf[:3]))
        return out

    def clearance():
        P = {f: np.array([d.xpos[b] for b in chain_b[f]]) for f in chain_b}
        best = 1e9
        for i in range(3):
            for j in range(3):
                best = min(best, seg_seg(P["index"][i], P["index"][i + 1], P["middle"][j], P["middle"][j + 1]))
        return best - 2 * CAP_R

    def tool_cos():
        q = d.xquat[tool]
        return 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2])

    def obs(k, last_action):
        jp = np.array([d.qpos[qadr[n]] - default_q[n] if n in qadr else 0.0 for n in robot_joints])
        jv = np.array([d.qvel[vadr[n]] if n in vadr else 0.0 for n in robot_joints])
        if not have_palm:
            # the benchmark palm rises lift_dz over the ramp; report that as the palm_pz joint
            s = k * decim
            jp[2] = np.clip((s - settle + 1) / ramp, 0, 1) * lift_dz
        palm_pos, palm_quat = d.xpos[palm_b].copy(), d.xquat[palm_b].copy()
        tpos, tquat = d.xpos[tool].copy(), d.xquat[tool].copy()
        pq_inv = quat_inv(palm_quat)
        obj_pos = quat_rotate(pq_inv, tpos - palm_pos)          # palm site sits at the palm body origin
        rel = np.concatenate([quat_rotate(pq_inv, tpos - palm_pos), quat_mul(pq_inv, tquat)])
        rb = ref.batch_at([k * step_dt])
        ref_fq = rb["finger_qpos"][0]
        ref_op = np.concatenate([rb["object_pos"][0], rb["object_quat"][0]])
        mis = np.array([np.arccos(np.clip(tool_cos(), -1, 1))])
        return np.concatenate([jp, jv, obj_pos, rel, ref_fq, ref_op, last_action, mis]).astype(np.float32)

    renderer = None
    frames = []
    if a.video:
        import imageio
        dist, az, el = (float(v) for v in a.cam.split(","))
        renderer = mujoco.Renderer(m, 720, 960)
        cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance, cam.azimuth, cam.elevation = dist, az, el
    rows = []
    last_action = np.zeros(len(FINGER_JOINTS), dtype=np.float32)
    with torch.no_grad():
        for k in range(a.steps):
            o = obs(k, last_action)
            if a.blind:
                o[30:40] = 0.0    # object_pos (3) + object_pose_actual (7)
                o[65] = 0.0       # target_axis_misalign
            act = actor(torch.as_tensor(o).unsqueeze(0)).squeeze(0).numpy()
            if clip is not None:
                act = np.clip(act, -float(clip), float(clip))
            for sub in range(decim):
                s = k * decim + sub
                t = min(1.0, s / max(1.0, settle))
                alpha = t if easing == "linear" else 1 - (1 - t) ** 2 if easing == "ease_out_quad" else 1 - (1 - t) ** 3
                offset = (1 - alpha) * open_ctrl + alpha * grip_ctrl
                active = 1.0 if s >= active_from else 0.0
                target = act * rscale * active + offset
                for i, n in enumerate(FINGER_JOINTS):
                    d.ctrl[aid[n]] = target[i]
                if have_palm:
                    pz = palm_default[2] + np.clip((s - settle + 1) / max(1.0, ramp), 0, 1) * lift_dz
                    for i, n in enumerate(PALM_JOINTS):
                        d.ctrl[palm_a[n]] = palm_default[i] if n != "palm_pz" else pz
                mujoco.mj_step(m, d)
            last_action = act.astype(np.float32)
            if renderer is not None:
                cam.lookat[:] = d.xpos[tool]
                renderer.update_scene(d, camera=cam); frames.append(renderer.render().copy())
            pf = pad_forces()
            rows.append([k, round(tool_cos(), 4), round(float(d.xpos[tool][2]), 4), round(pf["thumb"], 3), round(pf["index"], 3),
                         round(pf["middle"], 3), round(clearance() * 1000, 2), round(float(np.abs(act).max()), 3)])
    if renderer is not None:
        a.video.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(str(a.video), frames, fps=25, codec="libx264", quality=7)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out.with_suffix(".csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["step", "cos", "z", "f_thumb", "f_index", "f_middle", "clearance_mm", "act_absmax"]); w.writerows(rows)
    arr = np.array(rows, dtype=float)
    af = active_from // decim
    held = bool(arr[-1, 2] > 0.06 and arr[-1, 3:6].sum() >= 0.5)
    aligned_steps = np.where(arr[:, 1] >= 0.9)[0]
    summary = {"policy": str(a.policy), "scene": str(scene), "steps": a.steps, "held_end": held,
               "final_cos": float(arr[-1, 1]), "peak_cos": float(arr[:, 1].max()), "final_z": float(arr[-1, 2]),
               "t_align": int(aligned_steps[0]) if len(aligned_steps) else None,
               "force_active": [float(v) for v in arr[af:, 3:6].mean(axis=0)], "pad_peak": [float(v) for v in arr[af:, 3:6].max(axis=0)],
               "clearance_min_mm": float(arr[af:, 6].min()), "clearance_end_mm": float(arr[-1, 6]),
               "act_absmax": float(arr[af:, 7].max()), "clip_actions": clip, "residual_active_from_step": af}
    json.dump(summary, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("policy", "scene")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
