#!/usr/bin/env python3
"""Evaluation of the RL contact-model comparisons.

`legacy` (2026-10-06): held cosine of every checkpoint of the TPU block mesh and 1 mm pad runs of
scripts/rl_contact_train_queue.sh, through scripts/policy_eval_suite.py; rows in
docs/experiments/20261006-rl_contact/train_eval.jsonl.

The other subcommands serve the 2026-10-08 study (docs/handoff/20261007-contact_model_policy_training.md; runs of
scripts/rl_contact_train_queue_20261008.sh, tags 20261008-d6_work_<arm>_40M_s<seed>). Each evaluation rolls the
deterministic policy (action = the policy mean) in 64 parallel envs for 250 policy steps (5 s) with the training's
timing read from the run's config.yaml (residual and reorientation from step 58), the scripted grasp and lift, no
early termination and no randomisation unless a perturbation asks for one. One env is built per scene and the
checkpoints are loaded into its actor in turn. Per rollout and policy step it records the tool's signed cosine with
vertical (its own +z against world +z; +1 is tip down), its height, each fingertip's net contact force and contact count
(the env's fingertip sensor), and the action; with `final` also the deepest tip-tool penetration, the tool's drift
against the palm over the last second and the peak fingertip force.

Held (the load test of feedback_reorientation_needs_a_load_test): at least two fingertips each pressing >= 0.24 N (the
tool's weight) and the tool above 60 mm.

    PY="uv run --extra rl --extra gpu python"
    $PY scripts/rl_contact_eval.py ckpts --all [--every 82]        # checkpoint series of every finished run
    $PY scripts/rl_contact_eval.py final --all                     # final checkpoints with the physics records
    $PY scripts/rl_contact_eval.py transfer --scene tpu27meshc4 --all
    $PY scripts/rl_contact_eval.py robust --all --perturb friction=0.7 friction=1.3 mass=0.8 mass=1.2 kp=2 kp=6 kp=10 \\
        dt=0.001 noise=2,5
    uv run --extra rl python scripts/rl_contact_eval.py tb --all   # training curves from the event files (CPU)
Rows (fsynced): docs/experiments/20261008-contact_model_policies/{ckpt_eval,final_eval,transfer,robust,tb_dynamics}.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MORPH = ROOT / "results/phase1/real_v1"
RL = ROOT / "results/rl"
OUT8 = ROOT / "docs/experiments/20261008-contact_model_policies"
JOBS8 = ROOT / "logs/20261008-contact_model_policies/jobs.txt"
STEPS_PER_IT = 2048 * 24
FINGERS = ("thumb", "index", "middle")
ARMS = {
    "box": MORPH / "20261006-sv1_u0308_b050_work_tip",
    "tpu27mesh": MORPH / "20261006-sv1_u0308_b050_work_tip_tpu2.7mesh",
    "tpu27pads1": MORPH / "20261006-sv1_u0308_b050_work_tip_tpu2.7pads1",
    "tpu27meshc4": MORPH / "20261008-sv1_u0308_b050_work_tip_tpu2.7mesh_c4",
    "tpu27skin": MORPH / "20261008-sv1_u0308_b050_work_tip_tpu2.7skin",
}
HELD_N, FLOOR_Z, ALIGN = 0.24, 0.06, 0.9


# ---------------------------------------------------------------------------------- legacy (2026-10-06)

def legacy():
    out = ROOT / "docs/experiments/20261006-rl_contact/train_eval.jsonl"
    js_dir = ROOT / "logs/20261006-rl_contact/eval"
    done = set()
    if out.exists():
        for line in open(out):
            r = json.loads(line)
            done.add((r["tag"], r["iteration"]))
    js_dir.mkdir(parents=True, exist_ok=True)
    for variant in ("tpu2.7mesh", "tpu2.7pads1"):
        for seed in (0, 1):
            tag = f"20261006-d6_work_{variant.replace('.', '')}_20M_s{seed}"
            rd, morph = RL / tag, MORPH / f"20261006-sv1_u0308_b050_work_tip_{variant}"
            for ck in sorted((rd / "tensorboard").glob("model_*.pt"), key=_it):
                it = _it(ck)
                if (tag, it) in done:
                    continue
                js = js_dir / f"{tag}_{it}.json"
                cmd = [sys.executable, str(ROOT / "scripts/policy_eval_suite.py"), "--policy", str(ck),
                       "--morphology-run", str(morph), "--n", "64", "--steps", "250", "--align-thresh", "0.9",
                       "--lift-delta", "0.1", "--finger-residual-scale", "0.5", "--open-finger-from-keyframe",
                       "--closed-ctrl-from-keyframe", "open_ik", "--json-out", str(js), "--label", f"{tag} it {it}"]
                w0 = time.perf_counter()
                p = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, MUJOCO_GL="egl"))
                row = {"tag": tag, "variant": variant, "seed": seed, "iteration": it,
                       "env_steps": (it + 1) * STEPS_PER_IT, "checkpoint": str(ck.relative_to(ROOT)),
                       "wall_s": round(time.perf_counter() - w0, 1), "when": time.strftime("%Y-%m-%d %H:%M")}
                if p.returncode == 0 and js.exists():
                    row.update(json.loads(js.read_text()), status="ok")
                else:
                    row.update(status="error", error=(p.stderr or p.stdout)[-1500:])
                append(out, row)


# ---------------------------------------------------------------------------------- helpers

def _it(p: Path) -> int:
    return int(re.findall(r"\d+", p.stem)[0])


def append(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def done_keys(path: Path, keys: tuple[str, ...]) -> set:
    out = set()
    if path.exists():
        for line in open(path):
            r = json.loads(line)
            if r.get("status") == "ok":
                out.add(tuple(r.get(k) for k in keys))
    return out


def jobs(path: Path = JOBS8):
    """(tag, arm, seed, timesteps) of the 2026-10-08 queue, in queue order."""
    out = []
    for line in open(path):
        line = line.split("#")[0].split()
        if len(line) == 4:
            tag = line[0]
            out.append((tag, re.match(r"20261008-d6_work_(.+)_40M_s\d+", tag).group(1), int(line[2]), int(line[3])))
    return out


def final_ckpt(tag: str, steps: int) -> Path:
    return RL / tag / "tensorboard" / f"model_{steps // STEPS_PER_IT - 1}.pt"


def finished(tags=None):
    """Jobs whose final checkpoint exists (optionally restricted to `tags`)."""
    return [j for j in jobs() if final_ckpt(j[0], j[3]).exists() and (not tags or j[0] in tags)]


def pick(args):
    if args.all:
        return finished()
    return [j for j in jobs() if j[0] in args.tags]


# ---------------------------------------------------------------------------------- evaluator

class Evaluator:
    """One env of `scene_dir` with the actor of `ckpt`; `load` swaps checkpoints, `perturb` edits model fields in
    place (the captured CUDA graphs read the same arrays), `rollout` runs the 64 deterministic rollouts."""

    def __init__(self, scene_dir: Path, ckpt: Path, n: int = 64, steps: int = 250, dt: float | None = None,
                 noise: tuple[float, float] | None = None, physics: bool = False):
        import torch
        from morphohand.rl.deploy import build_actor, finger_ctrl_from_keyframe, make_env_cfg, run_env_overrides
        from morphohand.tools.video_paths import tmp_dir
        self.torch, self.n, self.steps, self.physics = torch, n, steps, physics
        trained = run_env_overrides(ckpt)
        self.residual_from = int(trained.get("finger_residual_active_from_step", 58))
        frozen = scene_dir / "frozen_scene.xml"
        summ = json.loads((scene_dir / "summary.json").read_text())
        bfc = finger_ctrl_from_keyframe(frozen, "open_ik")
        extra = dict(scene_floor=bool(trained.get("scene_floor", False)))
        if dt is not None:      # same seconds of scripted grasp and lift, same 50 Hz policy
            k = 0.002 / dt
            extra.update(sim_timestep=dt, decimation=int(round(10 * k)), settle_steps=int(round(240 * k)),
                         lift_ramp_steps=int(round(80 * k)), finger_close_sim_steps=int(round(80 * k)))
        if noise is not None:   # grasp-pose noise: uniform tool spawn offset (mm) and yaw (deg)
            extra.update(cube_spawn_x_jitter=noise[0] * 1e-3, cube_spawn_y_jitter=noise[0] * 1e-3,
                         cube_spawn_yaw_jitter=float(np.radians(noise[1])))
        cfg = make_env_cfg(frozen, summ["keyframe"], scene_dir, bfc, enable_target_axis=True, num_steps=steps,
                           finger_residual_scale=float(trained.get("finger_residual_scale", 0.5)),
                           finger_close_easing=trained.get("finger_close_easing", "ease_out_quad"),
                           lift_delta=0.1, open_finger_from_keyframe=True, num_envs=n,
                           finger_residual_active_from_step=self.residual_from,
                           reorient_start_step=int(trained.get("reorient_start_step", 58)),
                           lift_phase_start_step=trained.get("lift_phase_start_step"),
                           actor_blind_terms=tuple(trained.get("actor_blind_terms", ()) or ()), extra_cfg=extra)
        self.env, self.wrapped, self.actor = build_actor(cfg, ckpt, tmp_dir("contact_policies_eval"))
        u = self.u = self.env.unwrapped
        self.robot, self.cube = u.scene["robot"], u.scene["cube"]
        self.sensor = u.scene.sensors["fingertip_cube_contact"]
        self.palm = self.robot.body_names.index("palm_pose")
        self.finger_j = [i for i, nm in enumerate(self.robot.joint_names) if nm.split("_")[0] in FINGERS
                         and "_skin_" not in nm]
        mjm = self.mjm = u.sim.mj_model
        bname = [mjm.body(i).name.split("/")[-1] for i in range(mjm.nbody)]
        self.tool_body = next(i for i, b in enumerate(bname) if b in ("cube", "screwdriver_medium"))
        self.geom_finger = np.full(mjm.ngeom, -1)
        for g in range(mjm.ngeom):
            b = bname[mjm.geom_bodyid[g]]
            for k, f in enumerate(FINGERS):
                if b in (f"{f}_tip", f"{f}_tipskin"):
                    self.geom_finger[g] = k
        self.tool_geom = np.array([mjm.geom_bodyid[g] == self.tool_body for g in range(mjm.ngeom)])
        self.fing_act = [a for a in range(mjm.nu) if mjm.joint(int(mjm.actuator_trnid[a, 0])).name.split("/")[-1]
                         .split("_")[0] in FINGERS]
        self.nominal = {k: self._raw(k).clone() for k in ("geom_friction", "body_mass", "body_inertia",
                                                           "actuator_gainprm", "actuator_biasprm")}
        self.kp0 = float(self.nominal["actuator_gainprm"][..., self.fing_act[0], 0].flatten()[0])

    def _raw(self, name: str):
        """The model field's own torch view: mjlab shows a field shared by all worlds as a stride-0 expansion,
        which torch cannot write in place."""
        import warp as wp
        return wp.to_torch(getattr(self.u.sim.model, name).wp_array)

    def load(self, ckpt: Path):
        sd = self.torch.load(str(ckpt), map_location="cpu", weights_only=False)["actor_state_dict"]
        self.actor.load_state_dict(sd, strict=True)
        self.actor.eval()

    def perturb(self, kind: str | None = None, value: float | None = None):
        """Restore the nominal model, then scale one quantity: friction (every geom's sliding coefficient), mass (the
        tool's mass and inertia), kp (the finger servos' position gain, N m/rad)."""
        raw = {k: self._raw(k) for k in self.nominal}
        for k, v in self.nominal.items():
            raw[k].copy_(v)
        if kind == "friction":
            raw["geom_friction"][..., 0] = self.nominal["geom_friction"][..., 0] * value
        elif kind == "mass":
            raw["body_mass"][..., self.tool_body] = self.nominal["body_mass"][..., self.tool_body] * value
            raw["body_inertia"][..., self.tool_body, :] = self.nominal["body_inertia"][..., self.tool_body, :] * value
        elif kind == "kp":
            for a in self.fing_act:
                raw["actuator_gainprm"][..., a, 0] = value
                raw["actuator_biasprm"][..., a, 1] = -value
        elif kind is not None:
            raise ValueError(kind)

    def rollout(self) -> dict:
        torch = self.torch
        from morphohand.rl.deploy import act_b
        T, N = self.steps, self.n
        cos, z = np.zeros((T, N), np.float32), np.zeros((T, N), np.float32)
        force, found = np.zeros((T, N, 3), np.float32), np.zeros((T, N, 3), np.float32)
        acts = np.zeros((T, N, 9), np.float32)
        rel = np.zeros((T, N, 3), np.float32)        # tool position minus palm position, m
        axis = np.zeros((T, N, 3), np.float32)       # tool +z in world
        fvel = np.zeros((T, N), np.float32)          # fastest finger joint, rad/s
        pen = np.zeros((T, N), np.float32)           # deepest tip-tool penetration, m (physics only)
        wd = self.u.sim.wp_data
        obs_td, _ = self.wrapped.reset()
        with torch.no_grad():
            for s in range(T):
                a = act_b(self.actor, obs_td, False)
                obs_td, *_ = self.wrapped.step(a)
                pose = self.cube.data.root_link_pose_w
                w, x, y, zq = pose[:, 3], pose[:, 4], pose[:, 5], pose[:, 6]
                cos[s] = (1.0 - 2.0 * (x * x + y * y)).cpu().numpy()
                z[s] = pose[:, 2].cpu().numpy()
                axis[s] = torch.stack([2 * (x * zq + w * y), 2 * (y * zq - w * x), 1 - 2 * (x * x + y * y)], -1).cpu().numpy()
                rel[s] = (pose[:, :3] - self.robot.data.body_link_pose_w[:, self.palm, :3]).cpu().numpy()
                d = self.sensor.data
                force[s] = d.force.norm(dim=-1).reshape(N, -1)[:, :3].cpu().numpy()
                found[s] = d.found.reshape(N, -1)[:, :3].float().cpu().numpy()
                acts[s] = a.cpu().numpy()
                fvel[s] = self.robot.data.joint_vel[:, self.finger_j].abs().amax(-1).cpu().numpy()
                if self.physics:
                    nc = int(wd.nacon.numpy()[0])
                    if nc:
                        g = wd.contact.geom.numpy()[:nc]
                        dist = wd.contact.dist.numpy()[:nc]
                        wid = wd.contact.worldid.numpy()[:nc]
                        tip = ((self.geom_finger[g[:, 0]] >= 0) & self.tool_geom[g[:, 1]]) | \
                              ((self.geom_finger[g[:, 1]] >= 0) & self.tool_geom[g[:, 0]])
                        if tip.any():
                            p = np.zeros(N, np.float32)
                            np.maximum.at(p, wid[tip], np.maximum(-dist[tip], 0.0))
                            pen[s] = p
        return dict(cos=cos, z=z, force=force, found=found, acts=acts, rel=rel, axis=axis, fvel=fvel, pen=pen)

    def close(self):
        self.env.close()


def metrics(r: dict, residual_from: int = 58, physics: bool = False) -> dict:
    """Rates and spreads over the 64 rollouts of one evaluation."""
    cos, z, f, found, acts = r["cos"], r["z"], r["force"], r["found"], r["acts"]
    T, N = cos.shape
    a0 = max(1, residual_from)
    held_t = (z > FLOOR_Z) & ((f >= HELD_N).sum(-1) >= 2)
    held = held_t[-1]
    aligned_held = (cos >= ALIGN) & held_t
    t09 = np.where(aligned_held[a0:].any(0), aligned_held[a0:].argmax(0) + a0, -1)
    lost_step = np.full(N, -1)
    for e in np.nonzero(~held & held_t[a0])[0]:
        lost_step[e] = int(np.nonzero(held_t[:, e])[0].max()) + 1
    act = held_t[a0:]
    grip = f[a0:].sum(-1)
    dact = np.abs(np.diff(acts[a0 - 1:], axis=0)).mean(-1)               # (T - a0, N), action units
    cont = found[a0:].sum(-1)
    fc = cos[-1]

    def m_(x, mask=None):
        x = x[mask] if mask is not None else x.ravel()
        return float(np.mean(x)) if x.size else None

    out = dict(
        n=int(N), steps=int(T), n_held=int(held.sum()), held_frac=float(held.mean()),
        cos_mean=float(fc.mean()), cos_sd=float(fc.std()),
        held_cos_mean=float(fc[held].mean()) if held.any() else None,
        held_cos_median=float(np.median(fc[held])) if held.any() else None,
        held_cos_sd=float(fc[held].std()) if held.any() else None,
        cos_q10=float(np.percentile(fc, 10)), cos_q90=float(np.percentile(fc, 90)),
        peak_cos_mean=float(cos[a0:].max(0).mean()),
        n_reach09_held=int((t09 >= 0).sum()), t09_median=float(np.median(t09[t09 >= 0])) if (t09 >= 0).any() else None,
        n_lost=int((lost_step >= 0).sum()), lost_step_median=float(np.median(lost_step[lost_step >= 0])) if (lost_step >= 0).any() else None,
        grip_N=m_(grip, act), grip_thumb_N=m_(f[a0:, :, 0], act), grip_index_N=m_(f[a0:, :, 1], act),
        grip_middle_N=m_(f[a0:, :, 2], act),
        three_finger_share=m_(((f[a0:] >= HELD_N).sum(-1) == 3).astype(float), act),
        dact_mean=float(dact.mean()), dact_held=m_(dact, act),
        contacts_per_step=m_(cont, act), contacts_thumb=m_(found[a0:, :, 0], act),
        z_final_mm=float(1e3 * np.median(z[-1])),
        final_cos=[round(float(v), 4) for v in fc], held_final=[bool(v) for v in held],
    )
    if physics:
        pen = r["pen"][a0:]
        pmax = np.where(act, pen, 0).max(0) * 1e3                        # mm per rollout over its held steps
        fpk = np.where(act[..., None], f[a0:], 0).max((0, 2))              # N per rollout
        k = min(50, T - a0)                                                 # last second
        span = (k - 1) * 0.02
        drift = np.linalg.norm(r["rel"][-1] - r["rel"][-k], axis=-1) * 1e3 / span
        ax0, ax1 = r["axis"][-k], r["axis"][-1]
        ang = np.degrees(np.arccos(np.clip((ax0 * ax1).sum(-1), -1, 1))) / span
        h = held & held_t[-k]
        out.update(
            pen_max_mm_mean=float(pmax[held].mean()) if held.any() else None,
            pen_max_mm_worst=float(pmax.max()), pen_p95_mm=float(np.percentile(np.where(act, pen, np.nan)[act] * 1e3, 95)) if act.any() else None,
            peak_force_N_mean=float(fpk[held].mean()) if held.any() else None, peak_force_N_worst=float(fpk.max()),
            creep_mm_s_median=float(np.median(drift[h])) if h.any() else None,
            creep_deg_s_median=float(np.median(ang[h])) if h.any() else None,
            creep_mm_s_q90=float(np.percentile(drift[h], 90)) if h.any() else None,
            finger_speed_last_s_deg_s=float(np.degrees(np.median(r["fvel"][-k:][:, h]))) if h.any() else None,
        )
    return out


def save_traces(tag: str, r: dict, sub: str = "final_traces"):
    """Compact per-step traces of one evaluation for the page's figures (float16)."""
    d = OUT8 / sub
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / f"{tag}.npz", cos=r["cos"].astype(np.float16), z=r["z"].astype(np.float16),
                        force=r["force"].astype(np.float16), found=r["found"].astype(np.uint8))


# ---------------------------------------------------------------------------------- subcommands

def cmd_ckpts(args):
    out = OUT8 / "ckpt_eval.jsonl"
    have = done_keys(out, ("tag", "iteration"))
    for tag, arm, seed, steps in pick(args):
        cks = sorted((RL / tag / "tensorboard").glob("model_*.pt"), key=_it)
        last = _it(final_ckpt(tag, steps))
        todo = [c for c in cks if (_it(c) % args.every == 0 or _it(c) == last) and (tag, _it(c)) not in have]
        if not todo:
            continue
        ev = Evaluator(ARMS[arm], todo[0], n=args.n, steps=args.steps)
        for ck in todo:
            t0 = time.perf_counter()
            ev.load(ck)
            row = dict(tag=tag, arm=arm, seed=seed, iteration=_it(ck), env_steps=(_it(ck) + 1) * STEPS_PER_IT,
                       checkpoint=str(ck.relative_to(ROOT)), scene=arm)
            try:
                row.update(metrics(ev.rollout(), ev.residual_from), status="ok")
            except Exception as e:  # noqa: BLE001
                row.update(status="error", error=repr(e)[-500:])
            row.update(wall_s=round(time.perf_counter() - t0, 1), when=time.strftime("%Y-%m-%d %H:%M"))
            append(out, row)
            print(f"{tag} it {row['iteration']}: held {row.get('n_held')}/64, cos {row.get('cos_mean', float('nan')):+.3f}, "
                  f"held cos {row.get('held_cos_mean')}", flush=True)
        ev.close()


def cmd_final(args):
    out = OUT8 / "final_eval.jsonl"
    have = done_keys(out, ("tag",))
    for tag, arm, seed, steps in pick(args):
        if (tag,) in have:
            continue
        ck = final_ckpt(tag, steps)
        ev = Evaluator(ARMS[arm], ck, n=args.n, steps=args.steps, physics=True)
        t0 = time.perf_counter()
        r = ev.rollout()
        row = dict(tag=tag, arm=arm, seed=seed, iteration=_it(ck), checkpoint=str(ck.relative_to(ROOT)), scene=arm,
                   **metrics(r, ev.residual_from, physics=True), status="ok", wall_s=round(time.perf_counter() - t0, 1),
                   when=time.strftime("%Y-%m-%d %H:%M"))
        save_traces(tag, r)
        append(out, row)
        print(f"{tag} final: held {row['n_held']}/64, held cos {row['held_cos_mean']}, pen {row['pen_max_mm_mean']} mm, "
              f"creep {row['creep_mm_s_median']} mm/s", flush=True)
        ev.close()


def cmd_transfer(args):
    """Final checkpoints evaluated on scene `--scene` (every arm's own scene gives the matrix's diagonal)."""
    out = OUT8 / "transfer.jsonl"
    have = done_keys(out, ("tag", "scene"))
    todo = [j for j in pick(args) if (j[0], args.scene) not in have]
    if not todo:
        return
    ev = Evaluator(ARMS[args.scene], final_ckpt(todo[0][0], todo[0][3]), n=args.n, steps=args.steps)
    for tag, arm, seed, steps in todo:
        ck = final_ckpt(tag, steps)
        ev.load(ck)
        t0 = time.perf_counter()
        row = dict(tag=tag, arm=arm, seed=seed, iteration=_it(ck), scene=args.scene,
                   **metrics(ev.rollout(), ev.residual_from), status="ok", wall_s=round(time.perf_counter() - t0, 1),
                   when=time.strftime("%Y-%m-%d %H:%M"))
        append(out, row)
        print(f"{tag} on {args.scene}: held {row['n_held']}/64, held cos {row['held_cos_mean']}", flush=True)
    ev.close()


def _parse_perturb(s: str):
    k, v = s.split("=")
    if k == "noise":
        a, b = v.split(",")
        return k, (float(a), float(b))
    return k, float(v)


def cmd_robust(args):
    """Each final checkpoint on its own arm's scene under one perturbation at a time (and the nominal)."""
    out = OUT8 / "robust.jsonl"
    have = done_keys(out, ("tag", "perturb"))
    perts = [("nominal", None)] + [_parse_perturb(p) for p in args.perturb]
    by_arm = {}
    for j in pick(args):
        by_arm.setdefault(j[1], []).append(j)
    for arm, js in by_arm.items():
        groups = {"runtime": [p for p in perts if p[0] in ("nominal", "friction", "mass", "kp")],
                  **{f"{k}={v}": [(k, v)] for k, v in perts if k in ("dt", "noise")}}
        for gname, plist in groups.items():
            need = [(j, p) for j in js for p in plist if (j[0], _pname(p)) not in have]
            if not need:
                continue
            dt = plist[0][1] if plist[0][0] == "dt" else None
            noise = plist[0][1] if plist[0][0] == "noise" else None
            ev = Evaluator(ARMS[arm], final_ckpt(js[0][0], js[0][3]), n=args.n, steps=args.steps, dt=dt, noise=noise)
            for (tag, _, seed, steps), p in need:
                ev.load(final_ckpt(tag, steps))
                ev.perturb(p[0] if p[0] in ("friction", "mass", "kp") else None, p[1])
                t0 = time.perf_counter()
                row = dict(tag=tag, arm=arm, seed=seed, perturb=_pname(p), kind=p[0],
                           value=list(p[1]) if isinstance(p[1], tuple) else p[1], kp_nominal=ev.kp0,
                           **metrics(ev.rollout(), ev.residual_from), status="ok",
                           wall_s=round(time.perf_counter() - t0, 1), when=time.strftime("%Y-%m-%d %H:%M"))
                append(out, row)
                print(f"{tag} {row['perturb']}: held {row['n_held']}/64, held cos {row['held_cos_mean']}", flush=True)
            ev.close()


def _pname(p) -> str:
    k, v = p
    if k == "nominal":
        return "nominal"
    return f"{k}={v[0]:g},{v[1]:g}" if isinstance(v, tuple) else f"{k}={v:g}"


def cmd_tb(args):
    """Training curves from each run's event file: every scalar per iteration, with wall time; rewritten per run."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    out = OUT8 / "tb_dynamics.jsonl"
    rows = {}
    if out.exists():
        for line in open(out):
            r = json.loads(line)
            rows.setdefault(r["tag"], []).append(r)
    for tag, arm, seed, steps in (finished() if args.all else [j for j in jobs() if j[0] in args.tags]):
        ea = EventAccumulator(str(RL / tag / "tensorboard"), size_guidance={"scalars": 0})
        ea.Reload()
        by_it = {}
        for k in ea.Tags()["scalars"]:
            if k.endswith("/time"):          # the same scalars logged against elapsed seconds
                continue
            for e in ea.Scalars(k):
                d = by_it.setdefault(int(e.step), {"tag": tag, "arm": arm, "seed": seed, "iteration": int(e.step)})
                d[k] = float(e.value)
                d["wall_time"] = max(d.get("wall_time", 0.0), float(e.wall_time))
        rows[tag] = [by_it[i] for i in sorted(by_it)]
    tmp = out.with_suffix(".tmp")
    with open(tmp, "w") as fh:
        for tag in rows:
            for r in rows[tag]:
                fh.write(json.dumps(r) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(out)
    print(f"{out}: {sum(len(v) for v in rows.values())} rows, {len(rows)} runs")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("legacy")
    for name in ("ckpts", "final", "transfer", "robust", "tb"):
        s = sub.add_parser(name)
        s.add_argument("--tags", nargs="*", default=[])
        s.add_argument("--all", action="store_true", help="every finished run of the 2026-10-08 queue")
        s.add_argument("--n", type=int, default=64)
        s.add_argument("--steps", type=int, default=250)
        if name == "ckpts":
            s.add_argument("--every", type=int, default=82, help="iterations between evaluated checkpoints")
        if name == "transfer":
            s.add_argument("--scene", required=True, choices=sorted(ARMS))
        if name == "robust":
            s.add_argument("--perturb", nargs="+", default=[])
    a = ap.parse_args()
    if a.cmd in (None, "legacy"):
        legacy()
        return 0
    {"ckpts": cmd_ckpts, "final": cmd_final, "transfer": cmd_transfer, "robust": cmd_robust, "tb": cmd_tb}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
