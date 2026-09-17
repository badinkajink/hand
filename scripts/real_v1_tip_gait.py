#!/usr/bin/env python3
"""Two-tip gait on the bench maneuver: tip, rest, regrip, tip again.

    python3 scripts/real_v1_tip_gait.py --out docs/experiments/20260916-tip_gait --video

The deployed plan's turn on the calibrated plant is a cradle tip: the grip opens, the
pads fall to the tool's weight and the tool tips 20-60 deg until it rests on the fingers
(sim cos 0.638 vs bench 0.593 on rv05, 2026-09-02). That rest is a stable state. This
script treats it as the first step of a gait: after the tip the fingers are re-clamped
from their COMMANDED pose (each finger's flexion joints close until its pads carry
`--regrip-N`), then the plan's turn deltas are applied again from the new command, and
so on for `--tips` steps. Same setting as the 251 hardware runs: palm fixed, tool on the
100 mm post, the plan's own grip and trajectory.

Per tip the trace records the tool's signed cos, height, pad forces per finger, post and
palm contact, and the commanded-minus-achieved joint error. A tip is `carried` when the
tool ends higher than 20 mm below its grip height on >= 2 pads.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from real_v1_turn_mechanism import FINGERS, JOINTS, OBJECT, make_scene  # noqa: E402
from real_v1_turn_probe import contact_variant  # noqa: E402

HANDS = {
    "D1": ("sv1_w6689_b060", "20260902-residual-bench"),
    "D2": ("sv1_w2360_b075", "20260902-residual-bench"),
    "D3": ("sv1_u1364_b080", "20260902-residual-bench"),
    "D4": ("g12_b095", "20260829-real_v1_deploy"),
    "D5": ("sv1_u0060_b75", "20260830-real_v1-sobol128"),
    "D6": ("sv1_u0308_b050", "20260829-real_v1_deploy"),
    "D7": ("rv05_manual_b85", "20260902-residual-bench"),
    "D8": ("sv1_w0099_b100", "20260829-real_v1_deploy"),
}
PLANTS = {
    # the plant that matched the bench (pyramidal template contact, kp 0.5)
    "fast": dict(kp=0.5, kv=0.02, cone=None, impratio=None, mu=None),
    # Coulomb-like contact, mu assumed 1.0
    "cal": dict(kp=0.5, kv=0.02, cone="elliptic", impratio=10.0, mu=1.0),
    "cal25": dict(kp=0.25, kv=0.02, cone="elliptic", impratio=10.0, mu=1.0),
}
TIPS = {"thumb": "thumb_tip", "index": "index_tip", "middle": "middle_tip"}
W = 0.240


def plant_scene(base: Path, scenes: Path, tag: str, pname: str) -> Path:
    p = PLANTS[pname]
    s = make_scene(base, scenes / f"{tag}__kp{p['kp']:g}_kv{p['kv']:g}.xml", p["kp"], 0.35, 0.0035,
                   mass=True, kv=p["kv"])
    return contact_variant(s, p["cone"], p["impratio"], p["mu"])


def rollout(scene: Path, plan: dict, traj: Path | None, *, tips: int, regrip_N: float,
            regrip_step_deg: float, rest_s: float, seed: int = 0, jitter_xy: float = 0.0,
            video: Path | None = None, fps: int = 30, width: int = 640, height: int = 480,
            cam_dist: float = 0.24, cam_az: float = 110.0, cam_el: float = -5.0) -> dict:
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)
    poses = {p["name"]: p["joints"] for p in plan["poses"]}
    aid = {(f, j): mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"a_{f}_{j}")
           for f in FINGERS for j in JOINTS}
    jid = {(f, j): mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{f}_{j}")
           for f in FINGERS for j in JOINTS}
    lim = {}
    for k, a in aid.items():
        if m.actuator_ctrllimited[a]:
            lim[k] = tuple(m.actuator_ctrlrange[a])
        else:
            lim[k] = tuple(m.jnt_range[jid[k]]) if m.jnt_limited[jid[k]] else (-np.pi, np.pi)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, OBJECT)
    tip_bid = {f: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, TIPS[f]) for f in FINGERS}
    post_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "tool_post")
    obj_geoms = {i for i in range(m.ngeom) if m.geom_bodyid[i] == bid}
    pad_geoms: dict[str, set[int]] = {}
    palm_geoms: set[int] = set()
    for i in range(m.ngeom):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[i]) or ""
        for f in FINGERS:
            if bn.startswith(f):
                pad_geoms.setdefault(f, set()).add(i)
        if "palm" in n or "plate" in n or bn == "palm":
            palm_geoms.add(i)
    qadr = m.jnt_qposadr[m.body_jntadr[bid]]

    mujoco.mj_resetData(m, d)
    meta = plan["meta"]
    d.qpos[:] = np.asarray(meta["replay_initial_qpos"], dtype=float)
    d.ctrl[:] = np.asarray(meta["replay_base_ctrl"], dtype=float)
    mujoco.mj_forward(m, d)
    if jitter_xy:
        rng = np.random.default_rng(seed)
        d.qpos[qadr + 0] += rng.normal(0, jitter_xy)
        d.qpos[qadr + 1] += rng.normal(0, jitter_xy)

    renderer = None
    frames: list[np.ndarray] = []
    if video is not None:
        renderer = mujoco.Renderer(m, height=height, width=width)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = d.xpos[bid]
        cam.distance = cam_dist
        cam.azimuth = cam_az
        cam.elevation = cam_el
    frame_every = max(1, int(round(1.0 / (fps * m.opt.timestep))))
    nstep = [0]

    def cmd() -> dict[str, dict[str, float]]:
        return {f: {j: float(np.degrees(d.ctrl[aid[(f, j)]])) for j in JOINTS} for f in FINGERS}

    def set_cmd(pose: dict[str, dict[str, float]]) -> None:
        for f in FINGERS:
            for j in JOINTS:
                lo, hi = lim[(f, j)]
                d.ctrl[aid[(f, j)]] = float(np.clip(np.deg2rad(pose[f][j]), lo, hi))

    def run(seconds: float) -> None:
        for _ in range(max(1, int(seconds / m.opt.timestep))):
            mujoco.mj_step(m, d)
            nstep[0] += 1
            if renderer is not None and nstep[0] % frame_every == 0:
                cam.lookat[:] = d.xpos[bid]
                renderer.update_scene(d, camera=cam)
                frames.append(renderer.render().copy())

    def contacts() -> dict:
        f_pad = {f: 0.0 for f in FINGERS}
        n_pad = {f: 0 for f in FINGERS}
        f_post = f_palm = 0.0
        c6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            if not (g1 in obj_geoms or g2 in obj_geoms):
                continue
            other = g2 if g1 in obj_geoms else g1
            mujoco.mj_contactForce(m, d, i, c6)
            fn = float(abs(c6[0]))
            if other == post_gid:
                f_post += fn
            elif other in palm_geoms:
                f_palm += fn
            for f, gs in pad_geoms.items():
                if other in gs:
                    f_pad[f] += fn
                    n_pad[f] += 1
        return {"f_pad": {f: round(v, 3) for f, v in f_pad.items()}, "n_pad": n_pad,
                "f_post": round(f_post, 3), "f_palm": round(f_palm, 3),
                "pads_loaded": sum(1 for f in FINGERS if f_pad[f] >= W)}

    def snap(phase: str, k: int) -> dict:
        R = d.xmat[bid].reshape(3, 3)
        c = contacts()
        return {"phase": phase, "k": k, "t": round(float(d.time), 4),
                "cos": round(float(R[2, 2]), 4),
                "z": round(float(d.xpos[bid][2]), 5), "x": round(float(d.xpos[bid][0]), 5),
                "y": round(float(d.xpos[bid][1]), 5), **c,
                "q_err_deg": {f"{f}_{j}": round(float(np.degrees(
                    d.ctrl[aid[(f, j)]] - d.qpos[m.jnt_qposadr[jid[(f, j)]]])), 2)
                    for f in FINGERS for j in JOINTS},
                "cmd_deg": {f"{f}_{j}": round(float(np.degrees(d.ctrl[aid[(f, j)]])), 2)
                            for f in FINGERS for j in JOINTS}}


    # the plan's turn as per-row deltas from its grip pose, re-based on whatever the
    # current command is when a tip starts
    grip = poses["grip"]
    if traj is not None:
        rows = list(csv.DictReader(open(traj)))
        span = float(rows[-1]["t_s"]) or 1.6
        deltas = [({f: {j: float(r[f"{f}_{j}_deg"]) - grip[f][j] for j in JOINTS} for f in FINGERS},
                   span / len(rows)) for r in rows]
    else:
        end = poses["turn_end"]
        deltas = [({f: {j: (end[f][j] - grip[f][j]) * i / 60 for j in JOINTS} for f in FINGERS},
                   1.6 / 60) for i in range(61)]

    trace = []
    set_cmd(grip)
    run(0.8)
    trace.append(snap("grip", 0))
    z_grip = trace[0]["z"]
    tips_out = []
    for t in range(1, tips + 1):
        base = cmd()
        for k, (dl, dt) in enumerate(deltas):
            set_cmd({f: {j: base[f][j] + dl[f][j] for j in JOINTS} for f in FINGERS})
            run(dt)
            if k % 5 == 0 or k == len(deltas) - 1:
                trace.append(snap(f"tip{t}", k))
        run(rest_s)
        rest = snap(f"rest{t}", 0)
        trace.append(rest)
        # regrip: from the COMMANDED pose, close each finger's flexion joints in the
        # direction that brings its tip toward the shaft, until its pads carry regrip_N
        # closing direction = the plan's own open -> grip direction on mcp (pip is at its
        # stop in every grip pose, so it stays where the tip left it)
        sign = {f: float(np.sign(grip[f]["mcp"] - poses["open"][f]["mcp"]) or 1.0) for f in FINGERS}
        steps = {f: 0 for f in FINGERS}
        for _ in range(int(30 / regrip_step_deg)):
            c = contacts()
            open_f = [f for f in FINGERS if c["f_pad"][f] < regrip_N]
            if not open_f:
                break
            cur = cmd()
            for f in open_f:
                cur[f]["mcp"] += sign[f] * regrip_step_deg
                steps[f] += 1
            set_cmd(cur)
            run(0.1)
        run(0.4)
        rg = snap(f"regrip{t}", 0)
        rg["closure_deg"] = {f: round(steps[f] * regrip_step_deg * sign[f], 1) for f in FINGERS}
        trace.append(rg)
        tips_out.append({"tip": t, "cos_rest": rest["cos"], "z_rest": rest["z"],
                         "pads_rest": rest["pads_loaded"], "f_pad_rest": rest["f_pad"],
                         "f_post_rest": rest["f_post"], "f_palm_rest": rest["f_palm"],
                         "cos_regrip": rg["cos"], "z_regrip": rg["z"], "pads_regrip": rg["pads_loaded"],
                         "f_pad_regrip": rg["f_pad"], "closure_deg": rg["closure_deg"],
                         "carried": bool(rg["z"] > z_grip - 0.020 and rg["pads_loaded"] >= 2)})
    run(0.5)
    trace.append(snap("end", 0))

    if renderer is not None and frames:
        import imageio
        video.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(str(video), frames, fps=fps, codec="libx264", quality=7)
        renderer.close()
    end = trace[-1]
    return {"cos_grip": trace[0]["cos"], "z_grip": z_grip, "tips": tips_out,
            "cos_end": end["cos"], "z_end": end["z"], "pads_end": end["pads_loaded"],
            "f_pad_end": end["f_pad"], "dropped": bool(end["z"] < z_grip - 0.020),
            "turn_deg": round(float(np.degrees(np.arccos(np.clip(trace[0]["cos"], -1, 1))
                                               - np.arccos(np.clip(end["cos"], -1, 1)))), 1),
            "trace": trace}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hands", default="D7,D8,D6,D2,D4,D3")
    ap.add_argument("--plants", default="fast,cal")
    ap.add_argument("--tips", type=int, default=2)
    ap.add_argument("--regrip-N", type=float, default=0.8, help="per-finger pad force the regrip closes to")
    ap.add_argument("--regrip-step-deg", type=float, default=1.0)
    ap.add_argument("--rest-s", type=float, default=1.5)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--jitter-xy", type=float, default=0.0)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--cam", default="0.24,110,-5", help="distance m, azimuth deg, elevation deg")
    a = ap.parse_args()
    cam_dist, cam_az, cam_el = (float(v) for v in a.cam.split(","))
    a.out.mkdir(parents=True, exist_ok=True)
    scenes = a.out / "scenes"
    scenes.mkdir(exist_ok=True)
    rows = []
    out_json = a.out / "tip_gait.json"
    for dn in a.hands.split(","):
        tag, folder = HANDS[dn]
        plan_p = ROOT / "docs/experiments" / folder / "deploy" / f"{tag}_plan.json"
        plan = json.loads(plan_p.read_text())
        traj = plan_p.with_name(f"{tag}_traj.csv")
        base = Path(plan["meta"]["scene"])
        for pname in a.plants.split(","):
            scene = plant_scene(base, scenes, tag, pname)
            for s in range(a.seeds):
                cell = f"{dn}_{tag}__{pname}__s{s}"
                vid = (a.out / "videos" / f"{cell}.mp4") if a.video else None
                r = rollout(scene, plan, traj if traj.exists() else None, tips=a.tips,
                            regrip_N=a.regrip_N, regrip_step_deg=a.regrip_step_deg, rest_s=a.rest_s,
                            seed=1000 + s, jitter_xy=a.jitter_xy, video=vid,
                            cam_dist=cam_dist, cam_az=cam_az, cam_el=cam_el)
                r.update({"hand": dn, "tag": tag, "plant": pname, "seed": s, "cell": cell})
                rows.append(r)
                tips_s = "  ".join(f"tip{t['tip']}: rest {t['cos_rest']:+.2f} ({t['pads_rest']} pads, post {t['f_post_rest']:.1f} N)"
                                   f" -> regrip {t['cos_regrip']:+.2f} ({t['pads_regrip']} pads, "
                                   f"{'carried' if t['carried'] else 'LOST'})" for t in r["tips"])
                print(f"{dn} {pname} s{s}: grip {r['cos_grip']:+.2f}  {tips_s}  end {r['cos_end']:+.2f} "
                      f"{r['pads_end']} pads  turn {r['turn_deg']:+.0f} deg", flush=True)
                out_json.write_text(json.dumps(rows, indent=1))
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
