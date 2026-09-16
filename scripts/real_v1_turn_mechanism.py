#!/usr/bin/env python3
"""What turns the tool in the bench maneuver: the fingers, the post, or gravity?

    python3 scripts/real_v1_turn_mechanism.py --out docs/experiments/20260916-turn_mechanism

The deployed plans (the maneuver the hardware ran 251 times and the plant was calibrated on)
replay a tool lying on a 100 mm post, palm fixed, grip -> per-step CSV turn -> hold. The chain
turns a lifted tool in the air. This runs the SAME plan replay under plant {shipped, corrected}
x post {kept, removed after the grip} x gravity {on, off}, and records per step the tool's
signed cosine, centre height, post contact force, pad contacts and force, and every finger's
commanded-minus-achieved angle. If the turn needs the post or gravity, the chain's mid-air turn
is a different task from the one the hardware demonstrated.

`--kp` sweeps the corrected plant's gain with the post kept and removed, so the plant band that
matches the bench can be compared with the band the mid-air turn needs.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FINGERS = ("thumb", "index", "middle")
JOINTS = ("yaw", "mcp", "pip")
OBJECT = "screwdriver_medium"
DEFAULT_PLAN = ROOT / "docs/experiments/20260902-residual-bench/deploy/rv05_manual_b85_plan.json"


def make_scene(base: Path, out: Path, kp: float, forcerange: float, frictionloss: float,
               mass: bool, kv: float | None = None) -> Path:
    if out.exists():
        return out
    flags = ["--kp", str(kp), "--forcerange", str(forcerange), "--frictionloss", str(frictionloss)]
    if kv is not None:
        flags += ["--kv", str(kv)]
    if not mass:
        flags.append("--no-mass")
    subprocess.run([sys.executable, str(ROOT / "scripts/apply_measured_plant.py"),
                    "--scene", str(base), "--out", str(out)] + flags,
                   check=True, capture_output=True)
    return out


def replay(scene: Path, plan: dict, traj: Path | None, *, post: bool, gravity: bool,
           seed: int = 0, jitter_xy: float = 0.0, jitter_yaw: float = 0.0,
           video: Path | None = None, fps: int = 30, width: int = 640, height: int = 480) -> dict:
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(scene))
    if not gravity:
        m.opt.gravity[:] = 0.0
    d = mujoco.MjData(m)
    poses = {p["name"]: p["joints"] for p in plan["poses"]}
    aid = {(f, j): mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"a_{f}_{j}")
           for f in FINGERS for j in JOINTS}
    jid = {(f, j): mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{f}_{j}")
           for f in FINGERS for j in JOINTS}
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, OBJECT)
    post_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "tool_post")
    obj_geoms = {i for i in range(m.ngeom) if m.geom_bodyid[i] == bid}
    pad_geoms = {}
    for i in range(m.ngeom):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        for f in FINGERS:
            if n.startswith(f) and ("pad" in n or "tip" in n or "dist" in n):
                pad_geoms.setdefault(f, set()).add(i)
    # fall back: every geom on a body whose name starts with the finger
    if not pad_geoms:
        for i in range(m.ngeom):
            bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[i]) or ""
            for f in FINGERS:
                if bn.startswith(f):
                    pad_geoms.setdefault(f, set()).add(i)
    qadr = m.jnt_qposadr[m.body_jntadr[bid]]

    mujoco.mj_resetData(m, d)
    meta = plan["meta"]
    d.qpos[:] = np.asarray(meta["replay_initial_qpos"], dtype=float)
    d.ctrl[:] = np.asarray(meta["replay_base_ctrl"], dtype=float)
    base_ctrl = d.ctrl.copy()
    mujoco.mj_forward(m, d)
    if jitter_xy or jitter_yaw:
        rng = np.random.default_rng(seed)
        d.qpos[qadr + 0] += rng.normal(0, jitter_xy)
        d.qpos[qadr + 1] += rng.normal(0, jitter_xy)

    renderer = None
    frames = []
    if video is not None:
        renderer = mujoco.Renderer(m, height=height, width=width)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = d.xpos[bid]
        cam.distance = 0.35
        cam.azimuth = 150
        cam.elevation = -20
    frame_every = max(1, int(round(1.0 / (fps * m.opt.timestep))))
    step_count = [0]

    def snap(phase: str, k: int) -> dict:
        R = d.xmat[bid].reshape(3, 3)
        f_post = 0.0
        f_pad = {f: 0.0 for f in FINGERS}
        n_pad = {f: 0 for f in FINGERS}
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
            for f, gs in pad_geoms.items():
                if other in gs:
                    f_pad[f] += fn
                    n_pad[f] += 1
        return {
            "phase": phase, "k": k, "t": round(float(d.time), 4),
            "cos": round(float(R[2, 2]), 4),
            "z": round(float(d.xpos[bid][2]), 5),
            "y": round(float(d.xpos[bid][1]), 5),
            "x": round(float(d.xpos[bid][0]), 5),
            "f_post": round(f_post, 3),
            "f_pad": {f: round(v, 3) for f, v in f_pad.items()},
            "n_pad": n_pad,
            "q_err_deg": {f"{f}_{j}": round(float(np.degrees(
                d.ctrl[aid[(f, j)]] - d.qpos[m.jnt_qposadr[jid[(f, j)]]])), 2)
                for f in FINGERS for j in JOINTS},
            "cmd_deg": {f"{f}_{j}": round(float(np.degrees(d.ctrl[aid[(f, j)]])), 2)
                        for f in FINGERS for j in JOINTS},
            "q_deg": {f"{f}_{j}": round(float(np.degrees(d.qpos[m.jnt_qposadr[jid[(f, j)]]])), 2)
                      for f in FINGERS for j in JOINTS},
        }

    def hold(pose, seconds):
        d.ctrl[:] = base_ctrl
        for f in FINGERS:
            for j in JOINTS:
                d.ctrl[aid[(f, j)]] = np.deg2rad(pose[f][j])
        for _ in range(max(1, int(seconds / m.opt.timestep))):
            mujoco.mj_step(m, d)
            step_count[0] += 1
            if renderer is not None and step_count[0] % frame_every == 0:
                cam.lookat[:] = d.xpos[bid]
                renderer.update_scene(d, camera=cam)
                frames.append(renderer.render().copy())

    trace = []
    hold(poses["grip"], 0.8)
    trace.append(snap("grip", 0))
    if not post and post_gid >= 0:
        m.geom_contype[post_gid] = 0
        m.geom_conaffinity[post_gid] = 0
        hold(poses["grip"], 0.5)
        trace.append(snap("grip_nopost", 0))
    if traj is not None:
        rows = list(csv.DictReader(open(traj)))
        span = float(rows[-1]["t_s"]) or 1.6
        for k, r in enumerate(rows):
            hold({f: {j: float(r[f"{f}_{j}_deg"]) for j in JOINTS} for f in FINGERS},
                 span / len(rows))
            trace.append(snap("turn", k))
        last = {f: {j: float(rows[-1][f"{f}_{j}_deg"]) for j in JOINTS} for f in FINGERS}
    else:
        a_, b_ = poses["grip"], poses["turn_end"]
        for i in range(61):
            u = i / 60
            hold({f: {j: a_[f][j] + (b_[f][j] - a_[f][j]) * u for j in JOINTS} for f in FINGERS},
                 1.6 / 60)
            trace.append(snap("turn", i))
        last = b_
    for k in range(6):
        hold(last, 0.25)
        trace.append(snap("hold", k))

    if renderer is not None and frames:
        import imageio
        video.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(str(video), frames, fps=fps, codec="libx264", quality=7)
        renderer.close()

    turn = [s for s in trace if s["phase"] == "turn"]
    g = trace[0]
    end = trace[-1]
    return {
        "cos_grip": g["cos"], "cos_turn_end": turn[-1]["cos"] if turn else None,
        "cos_end": end["cos"],
        "turn_deg": round(float(np.degrees(np.arccos(np.clip(g["cos"], -1, 1))
                                          - np.arccos(np.clip(end["cos"], -1, 1)))), 1),
        "z_grip": g["z"], "z_end": end["z"], "z_min": min(s["z"] for s in trace),
        "dropped": bool(end["z"] < g["z"] - 0.020),
        "f_post_grip": g["f_post"], "f_post_end": end["f_post"],
        "f_post_max_turn": max((s["f_post"] for s in turn), default=0.0),
        "f_pad_grip": round(sum(g["f_pad"].values()), 3),
        "f_pad_end": round(sum(end["f_pad"].values()), 3),
        "n_pad_end": sum(1 for f in FINGERS if end["n_pad"][f] > 0),
        "q_err_end": end["q_err_deg"],
        "trace": trace,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--kp", type=float, nargs="*", default=[],
                    help="extra corrected-plant gains to sweep (post kept and removed)")
    ap.add_argument("--kv", type=float, nargs="*", default=[],
                    help="extra corrected-plant dampings at kp 0.5 (apply_measured_plant defaults to 0.6)")
    ap.add_argument("--skip-base", action="store_true", help="only the --kp/--kv variants")
    ap.add_argument("--variant-kv", type=float, default=None,
                    help="kv for the --kp variants (default: apply_measured_plant's 0.6)")
    ap.add_argument("--forcerange", type=float, default=0.35)
    ap.add_argument("--frictionloss", type=float, default=0.0035)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--chord", action="store_true", help="interpolate grip->turn_end instead of the CSV")
    a = ap.parse_args()

    plan = json.loads(a.plan.read_text())
    base = Path(plan["meta"]["scene"])
    traj = None if a.chord else a.plan.with_name(a.plan.name.replace("_plan.json", "_traj.csv"))
    if traj is not None and not traj.exists():
        print(f"no {traj}; using the chord")
        traj = None
    a.out.mkdir(parents=True, exist_ok=True)
    scenes = a.out / "scenes"
    scenes.mkdir(exist_ok=True)
    plants = {
        "shipped": make_scene(base, scenes / "shipped.xml", 30, 10, 0, mass=False),
        "corrected": make_scene(base, scenes / "corrected_kp0.5.xml", 0.5, a.forcerange,
                                a.frictionloss, mass=True),
    }
    if a.skip_base:
        plants = {}
    for kp in a.kp:
        vk = "" if a.variant_kv is None else f"_kv{a.variant_kv:g}"
        plants[f"kp{kp:g}{vk}"] = make_scene(base, scenes / f"corrected_kp{kp:g}{vk}.xml", kp,
                                             a.forcerange, a.frictionloss, mass=True, kv=a.variant_kv)
    for kv in a.kv:
        plants[f"kv{kv:g}"] = make_scene(base, scenes / f"corrected_kp0.5_kv{kv:g}.xml", 0.5,
                                         a.forcerange, a.frictionloss, mass=True, kv=kv)

    rows = []
    tag = a.plan.name.replace("_plan.json", "")
    for pname, scene in plants.items():
        for post in ((True, False) if pname in ("shipped", "corrected") else (True,)):
            for grav in ((True, False) if pname in ("shipped", "corrected") else (True,)):
                cell = f"{tag}__{pname}__post{int(post)}__g{int(grav)}"
                vid = (a.out / f"{cell}.mp4") if a.video else None
                r = replay(scene, plan, traj, post=post, gravity=grav, video=vid)
                r.update({"cell": cell, "plant": pname, "post": post, "gravity": grav, "tag": tag})
                rows.append(r)
                print(f"{cell:<48} cos {r['cos_grip']:+.3f} -> {r['cos_end']:+.3f}  "
                      f"turn {r['turn_deg']:+6.1f} deg  z {r['z_grip']*1e3:.1f}->{r['z_end']*1e3:.1f} mm  "
                      f"post {r['f_post_grip']:.2f}->{r['f_post_end']:.2f} N (max turn {r['f_post_max_turn']:.2f})  "
                      f"pads {r['f_pad_grip']:.2f}->{r['f_pad_end']:.2f} N/{r['n_pad_end']}  "
                      f"{'DROP' if r['dropped'] else 'held'}")
    name = f"{tag}_mechanism{'_' + a.suffix if a.suffix else ''}.json"
    (a.out / name).write_text(json.dumps(rows, indent=1))
    print(f"wrote {a.out / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
