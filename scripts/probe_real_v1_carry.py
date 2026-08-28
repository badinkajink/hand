"""Can this hand CARRY the shaft to vertical without slipping in the grip?

The reorient can happen two ways and they need different fixes, so they need separating before
any more GPU is spent:

  SLIP    the shaft rotates INSIDE a static grasp. `probe_real_v1_pivot.py` says no: on
          rv04_mid the centred tripod does not move under 0.005 Nm and loses the shaft entirely
          at 0.010, and the largest gravitational torque the geometry affords (grip offset
          30 mm on a 24.5 g shaft) is 0.0072 Nm. Between locked and dropped there is no
          band where it turns.
  CARRY   the fingers move WITH the shaft, contacts fixed on the same material points, and the
          whole grip rotates. This is what both reference policies actually do — r4's contacts
          are parked at s = +42 mm and slide ~5 mm over the entire reorient
          (docs/experiments/REORIENT_PRIMITIVE.txt §2).

This probe drives the carry OPEN-LOOP. It rotates the object's pose about the pinch axis through
the contact centroid, moves each fingertip to where that rigid rotation puts its own contact
point, and asks whether the shaft follows. No policy, no reward, no seed — if the carry works
here then the four flat B runs were an exploration and reward-shaping failure, and if it does not
then no amount of PPO was going to find it.

TWO RULES IT OBEYS, both from REORIENT_PRIMITIVE.txt, both of which produced convincing wrong
answers when they were broken:

  * The IK solution is applied as a RESIDUAL ON THE GRASP ANCHOR, never as the anchor. These are
    position servos and the grip force IS the commanded-minus-actual error; commanding the tip
    exactly onto the surface zeroes that error and the hand lets go.
  * peak_cos is not a result. A dropped shaft lands upright and reads cos 1.000, which is
    exactly what the torque sweep in probe_real_v1_pivot.py turned out to be measuring. Every
    row here carries final object z and contact count, and `ok` requires both.

    MUJOCO_GL=egl uv run python scripts/probe_real_v1_carry.py \
        --morph-run results/phase1/real_v1/rv04_mid_sp30 --lift 0.10 --turn-steps 400
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

sys.path.insert(0, str(ROOT / "scripts"))

import fit_real_v1_pose as fp  # noqa: E402
from morphohand.tools.keyframe_ik import (  # noqa: E402
    FINGERS, actuator_ctrl_from_qpos, ik_finger,
)

TIPS = {f: f"{f}_tip" for f in FINGERS}


def workspace(morph_run: Path, obj: str = "screwdriver_medium") -> dict:
    """Per-finger RADIAL slack at the grasp: how much reach is left to give, and to take back.

    This is the scalar that predicts whether a design can carry the shaft round at all. A
    fixed-contact rotation of theta about the pinch axis sends the descending pair contact down
    by straddle * sin(theta), which its finger pays for by EXTENDING. `fit_real_v1_pose` picks
    the DEEPEST reachable palm, so every real_v1 design grasps within a few millimetres of full
    extension and the ceiling is asin(extend / straddle):

        design         extend   straddle   ceiling      trained peak
        rv04_mid        1.3 mm    30 mm     2.5 deg     0.015  (0.9 deg)
        rv05_manual     4.1        --         --        0.030
        rv00_wide       4.5        30         8.6       0.019  (1.1 deg)
        rv03_narrowy    7.1        40        10.2       0.069  (4.0 deg)

    Raising the pivot above the contact plane converts the demand from extension into
    RETRACTION, of which there is 38-47 mm, and rv03_narrowy -- the design with the most
    extension left -- is the only one of the four whose open-loop carry then holds.
    """
    import itertools
    m = mujoco.MjModel.from_xml_path(str(morph_run / "frozen_scene.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("open_ik").id)
    mujoco.mj_forward(m, d)
    out = {}
    for f, joints in FINGERS.items():
        adr = [m.jnt_qposadr[m.joint(j).id] for j in joints]
        rng = [m.jnt_range[m.joint(j).id] for j in joints]
        root = d.body(f"{f}_yaw_frame").xpos.copy()
        now = float(np.linalg.norm(d.body(TIPS[f]).xpos - root))
        q0 = d.qpos.copy()
        lo = hi = None
        for a, b in itertools.product(np.linspace(*rng[1], 40), np.linspace(*rng[2], 40)):
            d.qpos[adr[1]], d.qpos[adr[2]] = a, b
            mujoco.mj_forward(m, d)
            r = float(np.linalg.norm(d.body(TIPS[f]).xpos - d.body(f"{f}_yaw_frame").xpos))
            lo = r if lo is None else min(lo, r)
            hi = r if hi is None else max(hi, r)
        d.qpos[:] = q0
        mujoco.mj_forward(m, d)
        out[f] = {"reach_mm": round(now * 1000, 1), "min_mm": round(lo * 1000, 1),
                  "max_mm": round(hi * 1000, 1), "extend_mm": round((hi - now) * 1000, 1),
                  "retract_mm": round((now - lo) * 1000, 1)}
    return {"run": morph_run.name, "fingers": out,
            "extend_mm": min(v["extend_mm"] for v in out.values())}


def _rotx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _contacts(m, d, obj: str):
    n, tot = 0, 0.0
    f6 = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        b = {m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name}
        if obj in b and b & set(TIPS.values()):
            mujoco.mj_contactForce(m, d, i, f6)
            n += 1
            tot += float(abs(f6[0]))
    return n, tot


def _per_finger_contact(m, d, obj: str) -> dict:
    """Per finger: normal force, tangential force, and FRICTION-CONE UTILISATION |f_t|/(mu*f_n).

    Utilisation is the number that separates the two explanations for a finger that fails to
    turn the shaft. At ~1.0 the contact is on the edge of the cone and is sliding, so that finger
    needs more NORMAL force (or the others need less, since they set how hard the object is
    pinned). Well below 1.0 it is not slipping at all and the finger simply is not being
    commanded far enough. `mj_contactForce` returns the wrench in the CONTACT frame, so f[0] is
    the normal and f[1:3] are the two tangential components directly."""
    out = {f: {"fn": 0.0, "ft": 0.0, "util": 0.0, "mu": 0.0} for f in FINGERS}
    f6 = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        names = {m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name}
        if obj not in names:
            continue
        for f in FINGERS:
            if TIPS[f] in names:
                mujoco.mj_contactForce(m, d, i, f6)
                fn = abs(float(f6[0]))
                ft = float(np.linalg.norm(f6[1:3]))
                mu = float(c.friction[0])
                out[f]["fn"] += fn
                out[f]["ft"] += ft
                out[f]["mu"] = mu
                out[f]["util"] = ft / (mu * fn) if fn > 1e-6 else 0.0
    return out


def _cos(m, d, obj: str) -> float:
    """SIGNED alignment of the shaft's local +Z with world +Z.

    Signed, not |cos|: `terms_reward.target_axis_alignment` rewards cos -> +1 only, so a carry
    that stands the shaft up the OTHER way scores -0.8 and is punished. The first training
    smoke of this schedule read target_axis_progress -0.93 for exactly that reason."""
    return float(d.body(obj).xmat[8])


def _finger_act(m):
    return {j: next(k for k in range(m.nu) if m.actuator(k).name == f"a_{j}")
            for joints in FINGERS.values() for j in joints}


def _grip_from_fit(scene: Path, straddle: float, offset: float, squeeze: float, obj: str,
                   depth: float | None = None):
    """Author a grasp at a PINNED straddle, so the carry can be swept over it.

    `fit_real_v1_pose` normally picks the straddle by scoring close-lift-hold rollouts, which
    selects for resisting exactly the rotation this probe is trying to produce. Here the
    straddle is the independent variable, so it is pinned and the hold is reported, not gated.
    """
    # GRIP DEPTH IS A VARIABLE, NOT AN OPTIMUM. `fit` normally takes the DEEPEST reachable
    # palm height, for clearance over the shaft's upper half once it stands up. On this hand
    # that lands every design at 52-65 mm of a 68.11 mm mount-to-pad chain -- 95% extension, the
    # boundary of the reach shell, where the fingertip has no radial workspace left and so
    # cannot carry the object anywhere. Capping the palm height trades clearance for workspace,
    # and an axial grip offset buys the clearance back (the stub above the grip is half_len -
    # offset, not half_len).
    m0 = mujoco.MjModel.from_xml_path(str(scene))
    d0 = mujoco.MjData(m0)
    mujoco.mj_resetDataKeyframe(m0, d0, fp._seed_key(m0, "open"))
    mujoco.mj_forward(m0, d0)
    obj_z = float(fp._object_geometry(m0, d0, obj)[0][2])
    pz_ref = float(m0.body("palm_pose").pos[2])
    if depth is None:
        pz_hi, pz_lo = 0.060, -0.030
    else:
        pz_hi = obj_z + depth - pz_ref
        pz_lo = pz_hi - 0.008
    out = fp.fit(scene, 0.001, straddle, 0.0, obj, pz_lo, pz_hi, 0.0025, verbose=False,
                 spreads=(straddle,), squeeze=squeeze, hold_min=-1.0, axial_offset=offset)
    if out is None:
        return None
    rep = out[0]
    px, py, pz = rep["palm"]["px"], rep["palm"]["py"], rep["palm"]["pz"]
    depth_mm = rep["grip_depth_mm"]
    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)
    seed = fp._seed_key(m, "open")
    mujoco.mj_resetDataKeyframe(m, d, seed)
    mujoco.mj_forward(m, d)
    centre, radius, _ = fp._object_geometry(m, d, obj)
    fp.solve(m, d, fp.tip_targets(centre, radius, 0.001, straddle, 0.0, offset),
             px, py, pz, seed, iters=600)
    open_qpos = d.qpos.copy()
    fp.solve(m, d, fp.tip_targets(centre, radius, 0.001 - squeeze, straddle, 0.0, offset),
             px, py, pz, seed, iters=600)
    return m, open_qpos, np.array(actuator_ctrl_from_qpos(m, d)), depth_mm


def carry(scene: Path, lift: float, turn_steps: int, hold_steps: int, angle: float,
          axis_shift: float, obj: str, budget: float, trace: bool,
          morph_run: Path | None = None, straddle: float | None = None,
          offset: float = 0.0, squeeze: float = 0.004, label: str = "",
          depth: float | None = None, axis_k: float = 0.0,
          record: Path | None = None, linear_anchor: bool = False,
          write_hold: bool = False, film: Path | None = None, film_frames: int = 8,
          contact_trace: list | None = None):
    if morph_run is not None:
        m = mujoco.MjModel.from_xml_path(str(scene))
        d = mujoco.MjData(m)
        depth_mm = float("nan")
        closed = np.load(morph_run / "best_rollout.npz")["best_finger_ctrl"]
        key = m.key("open_ik").id
        mujoco.mj_resetDataKeyframe(m, d, key)
        d.ctrl[:] = m.key_ctrl[key]
        anchor = {j: float(closed[i * 3 + k])
                  for i, (f, js) in enumerate(FINGERS.items()) for k, j in enumerate(js)}
    else:
        built = _grip_from_fit(scene, straddle, offset, squeeze, obj, depth)
        if built is None:
            return None
        m, open_qpos, grip, depth_mm = built
        d = mujoco.MjData(m)
        d.qpos[:] = open_qpos
        d.qvel[:] = 0.0
        d.ctrl[:] = grip
        acts0 = _finger_act(m)
        anchor = {j: float(grip[a]) for j, a in acts0.items()}
    mik = mujoco.MjModel.from_xml_path(str(scene))   # scratch pair for the IK, so the live
    dik = mujoco.MjData(mik)                         # sim state is never disturbed

    acts = _finger_act(m)
    pz_a = next(k for k in range(m.nu) if m.actuator(k).name == "a_palm_pz")
    for j, a in acts.items():
        d.ctrl[a] = anchor[j]
    mujoco.mj_forward(m, d)

    # close -> lift -> settle, the schedule Policy A's env runs
    for _ in range(250):
        mujoco.mj_step(m, d)
    pz0 = float(d.ctrl[pz_a])
    for k in range(200):
        d.ctrl[pz_a] = pz0 + lift * (k + 1) / 200
        mujoco.mj_step(m, d)
    for _ in range(200):
        mujoco.mj_step(m, d)

    tip0 = {f: d.body(TIPS[f]).xpos.copy() for f in FINGERS}
    centroid = np.mean([tip0[f] for f in FINGERS], axis=0)
    centroid[1] += axis_shift          # slide the pivot along the shaft
    # RAISE THE PIVOT. Rotating about an axis through the contacts makes the descending pair
    # finger travel S*sin(theta) DOWNWARD, i.e. it must EXTEND by that much -- and the measured
    # extension budget on these grasps is 1.3-9.6 mm against a 30-40 mm straddle, a ceiling of
    # 2.5-10 degrees which is exactly the 0.9-4 deg the four trained policies reached. Putting
    # the axis h above the contact plane adds h*(1-cos theta) to BOTH pair contacts, so the
    # descending one needs no extension once h >= S*cot(theta/2) (h = S at 90 deg) and the
    # ascending one spends retraction instead, of which there is 38-47 mm. Physically this is
    # the object descending relative to the palm as it stands up -- the palm has to come down
    # by about one straddle over the turn, and nothing in Policy B's action space can do that.
    span = abs(tip0["index"][1] - tip0["middle"][1]) / 2.0
    centroid[2] += axis_k * span
    q0 = {j: float(d.qpos[m.jnt_qposadr[m.joint(j).id]]) for j in acts}
    start = {"cos": _cos(m, d, obj), "z": float(d.body(obj).xpos[2]),
             "contacts": _contacts(m, d, obj)}

    rows = []
    ik_miss = 0.0
    travel = {f: 0.0 for f in FINGERS}
    ref, ref_every = [], 10          # 10 sim steps = one 0.02 s control step
    _prev_obj: dict = {}
    shots, renderer = [], None
    if film is not None:
        renderer = mujoco.Renderer(m, height=480, width=640)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(m, cam)
        cam.azimuth, cam.elevation, cam.distance = 90, -10, 0.45
        cam.lookat[:] = d.body(obj).xpos
    final_ctrl = dict(anchor)

    if linear_anchor:
        # What `LerpFingerActionCfg.hold_target_ctrl` actually does: a straight line in JOINT
        # space from the grasp anchor to one end pose. If that reproduces the IK'd carry then
        # the whole reorient needs no new controller, only a second keyframe and a switch.
        dik.qpos[:] = d.qpos
        dik.qvel[:] = 0.0
        mujoco.mj_forward(mik, dik)
        R = _rotx(angle)
        for f in FINGERS:
            ik_finger(mik, dik, f, centroid + R @ (tip0[f] - centroid), iters=400)
        end = {j: float(dik.qpos[mik.jnt_qposadr[mik.joint(j).id]]) for j in acts}
    for k in range(1, turn_steps + 1):
        th = angle * k / turn_steps
        if linear_anchor:
            u = k / turn_steps
            for j, a in acts.items():
                delta = (end[j] - q0[j]) * u
                d.ctrl[a] = anchor[j] + float(np.clip(delta, -budget, budget))
                final_ctrl[j] = float(d.ctrl[a])
        else:
            R = _rotx(th)
            dik.qpos[:] = d.qpos
            dik.qvel[:] = 0.0
            mujoco.mj_forward(mik, dik)
            for f in FINGERS:
                tgt = centroid + R @ (tip0[f] - centroid)
                travel[f] = max(travel[f], float(np.linalg.norm(tgt - tip0[f])))
                ik_miss = max(ik_miss, ik_finger(mik, dik, f, tgt, iters=60))
            for j, a in acts.items():
                delta = float(dik.qpos[mik.jnt_qposadr[mik.joint(j).id]]) - q0[j]
                d.ctrl[a] = anchor[j] + float(np.clip(delta, -budget, budget))
                final_ctrl[j] = float(d.ctrl[a])
        mujoco.mj_step(m, d)
        if film is not None and k % max(1, turn_steps // film_frames) == 0:
            renderer.update_scene(d, cam)
            shots.append(renderer.render())
        if record is not None and k % ref_every == 0:
            op = d.body(obj).xpos.copy()
            R_o = d.body(obj).xmat.reshape(3, 3)
            ref.append([(R_o.T @ (d.body(TIPS[f]).xpos - op)) for f in FINGERS])
        if contact_trace is not None and k % max(1, turn_steps // 20) == 0:
            op = d.body(obj).xpos.copy()
            R_o = d.body(obj).xmat.reshape(3, 3)
            here = {f: (R_o.T @ (d.body(TIPS[f]).xpos - op)) for f in FINGERS}
            row = {"step": k, "cmd_deg": round(float(np.degrees(th)), 1),
                   "cos": round(_cos(m, d, obj), 3),
                   "z": round(float(d.body(obj).xpos[2]), 4), "fingers": {}}
            pf = _per_finger_contact(m, d, obj)
            for f in FINGERS:
                slip = (0.0 if not contact_trace or not _prev_obj.get(f) is not None
                        else float(np.linalg.norm(here[f] - _prev_obj[f])))
                _prev_obj[f] = here[f]
                row["fingers"][f] = {"fn_N": round(pf[f]["fn"], 2),
                                     "ft_N": round(pf[f]["ft"], 2),
                                     "cone_util": round(pf[f]["util"], 3),
                                     # mm the pad has moved across the shaft's SURFACE since the
                                     # last sample: the direct measure of pad slip, in the frame
                                     # that makes it meaningful (the object's own)
                                     "slip_mm": round(slip * 1000, 2)}
            contact_trace.append(row)
        if trace and k % max(1, turn_steps // 10) == 0:
            n, fo = _contacts(m, d, obj)
            rows.append({"step": k, "cmd_deg": round(np.degrees(th), 1),
                         "cos": round(_cos(m, d, obj), 3),
                         "z": round(float(d.body(obj).xpos[2]), 4),
                         "contacts": n, "force_N": round(fo, 2)})

    peak = _cos(m, d, obj)
    for _ in range(hold_steps):
        mujoco.mj_step(m, d)
        peak = peak if abs(peak) >= abs(_cos(m, d, obj)) else _cos(m, d, obj)
    n, fo = _contacts(m, d, obj)
    z = float(d.body(obj).xpos[2])
    if film is not None:
        renderer.update_scene(d, cam)
        shots.append(renderer.render())
        import PIL.Image
        w = min(4, len(shots))
        rows_ = [np.hstack(shots[i:i + w]) for i in range(0, len(shots), w)]
        wmax = max(r.shape[1] for r in rows_)
        rows_ = [np.pad(r, ((0, 0), (0, wmax - r.shape[1]), (0, 0))) for r in rows_]
        film.parent.mkdir(parents=True, exist_ok=True)
        PIL.Image.fromarray(np.vstack(rows_)).save(film)
    if write_hold:
        from morphohand.tools.keyframe_ik import inject_keyframe
        inject_keyframe(scene, "hold_ik",
                        " ".join(f"{v:.6g}" for v in d.qpos),
                        " ".join(f"{v:.6g}" for v in d.ctrl))
    if record is not None:
        record.parent.mkdir(parents=True, exist_ok=True)
        np.savez(record, fingertip_obj=np.asarray(ref, dtype=np.float32),
                 dt=np.float64(ref_every * m.opt.timestep), t0=np.float64(0.0),
                 source=np.str_(f"probe_real_v1_carry {label} axis_k={axis_k} "
                                f"steps={turn_steps} peak={peak:.3f}"),
                 hold_ctrl=np.asarray([final_ctrl[j] for js in FINGERS.values() for j in js],
                                      dtype=np.float32),
                 hold_qpos=d.qpos.copy(), hold_full_ctrl=d.ctrl.copy())
    return {
        "run": label or (morph_run.name if morph_run else scene.stem[:18]),
        "lift": lift, "turn_steps": turn_steps,
        "angle_deg": round(np.degrees(angle), 1), "axis_shift_mm": axis_shift * 1000,
        "budget_rad": budget, "axis_k": axis_k,
        "axis_height_mm": round(axis_k * span * 1000, 1),
        "start_cos": round(start["cos"], 3), "start_z": round(start["z"], 4),
        "peak_cos": round(peak, 3), "final_cos": round(_cos(m, d, obj), 3),
        "final_z": round(z, 4), "contacts": n, "force_N": round(fo, 2),
        "max_ik_residual_mm": round(ik_miss * 1000, 2),
        "straddle_mm": None if straddle is None else straddle * 1000,
        "grip_depth_mm": round(depth_mm, 1),
        "offset_mm": offset * 1000,
        "commanded_tip_travel_mm": {f: round(v * 1000, 1) for f, v in travel.items()},
        # a shaft standing on the table also reads cos 1.0; require it to still be HELD
        "ok": bool(n >= 2 and z > start["z"] - 0.02),
        "trace": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--morph-run", type=Path, default=None, action="append",
                    help="a CEM morphology run; uses its frozen scene and its own grip")
    ap.add_argument("--scene", type=Path, default=None,
                    help="a generated scene; the grip is authored at --straddle / --offset")
    ap.add_argument("--straddle", default="0.010,0.015,0.020,0.030",
                    help="metres, half-separation of index/middle along the shaft")
    ap.add_argument("--offset", default="0.0", help="metres, grip offset along the shaft")
    ap.add_argument("--squeeze", type=float, default=0.004)
    ap.add_argument("--depth", default="",
                    help="metres of grip depth below the mounting plane, comma list. Empty = "
                         "the fitter's DEEPEST reachable palm, which is 95%% finger extension")
    ap.add_argument("--object-body", default="screwdriver_medium")
    ap.add_argument("--lift", type=float, default=0.10)
    ap.add_argument("--turn-steps", default="200,400,800",
                    help="sim steps over which the 90 deg is commanded (speed axis)")
    ap.add_argument("--angle-deg", type=float, default=90.0)
    ap.add_argument("--axis-k", default="0.0",
                    help="height of the rotation axis above the contact plane, in units of the "
                         "half-straddle. 1.0 removes the descending finger's extension demand "
                         "at 90 deg; comma list")
    ap.add_argument("--axis-shift", default="0.0",
                    help="metres to slide the rotation pivot along the shaft, comma list")
    ap.add_argument("--budget", type=float, default=0.5,
                    help="per-joint clip on the anchor move; 0.5 = the trainer's residual scale")
    ap.add_argument("--hold-steps", type=int, default=500)
    ap.add_argument("--linear-anchor", action="store_true",
                    help="sweep the anchor in a straight line in joint space instead of IK-ing "
                         "each step -- exactly what LerpFingerActionCfg.hold_target_ctrl does")
    ap.add_argument("--record-ref", type=Path, default=None,
                    help="write the object-relative fingertip schedule + the end anchor pose "
                         "here (the imitation-reference npz format)")
    ap.add_argument("--write-hold-keyframe", action="store_true",
                    help="write the end-of-carry pose into the scene as `hold_ik`; that is the "
                         "second anchor --hold-ctrl-from-keyframe reads")
    ap.add_argument("--film", type=Path, default=None, help="tile frames of the carry here")
    ap.add_argument("--contact-trace", type=Path, default=None,
                    help="write per-finger normal / tangential force, friction-cone utilisation "
                         "and pad slip across the shaft surface, through the turn")
    ap.add_argument("--workspace", action="store_true",
                    help="report per-finger radial slack for each --morph-run and exit")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.workspace:
        print(f"{'run':22} {'finger':7} {'reach':>7} {'min':>7} {'max':>7} "
              f"{'extend':>7} {'retract':>8}")
        ws = []
        for r in (args.morph_run or []):
            w = workspace(r, args.object_body)
            ws.append(w)
            for f, v in w["fingers"].items():
                print(f"{r.name:22} {f:7} {v['reach_mm']:6.1f}m {v['min_mm']:6.1f}m "
                      f"{v['max_mm']:6.1f}m {v['extend_mm']:+6.1f}m {v['retract_mm']:+7.1f}m")
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(ws, indent=2))
        return 0

    steps = [int(v) for v in str(args.turn_steps).split(",")]
    shifts = [float(v) for v in str(args.axis_shift).split(",")]
    ks = [float(v) for v in str(args.axis_k).split(",")]
    rows, ctrace = [], []
    if args.morph_run:
        cases = [(r / "frozen_scene.xml", r, None, 0.0, r.name, None)
                 for r in args.morph_run]
    else:
        if args.scene is None:
            ap.error("pass --morph-run or --scene")
        depths = ([None] if not args.depth
                  else [float(v) for v in str(args.depth).split(",")])
        cases = [(args.scene, None, st, off,
                  f"sp{st*1000:.0f}_off{off*1000:.0f}"
                  + ("" if dp is None else f"_d{dp*1000:.0f}"), dp)
                 for dp in depths
                 for off in (float(v) for v in str(args.offset).split(","))
                 for st in (float(v) for v in str(args.straddle).split(","))]
    print(f"{'case':22} {'axisH':>6} {'steps':>6} {'peak':>6} {'final':>6} {'z':>7} "
          f"{'con':>4} {'N':>7} {'ikRes':>7} {'travel':>7}  ok")
    for scene, run, st_m, off, label, dp in cases:
        for sh in shifts:
          for kk in ks:
            for st in steps:
                r = carry(scene, args.lift, st, args.hold_steps,
                          np.radians(args.angle_deg), sh, args.object_body,
                          args.budget, args.trace, morph_run=run, straddle=st_m,
                          offset=off, squeeze=args.squeeze, label=label, depth=dp,
                          axis_k=kk, record=args.record_ref,
                          linear_anchor=args.linear_anchor,
                          write_hold=args.write_hold_keyframe, film=args.film,
                          contact_trace=(ctrace if args.contact_trace else None))
                if r is None:
                    print(f"{label:22}   -- no pose --")
                    continue
                rows.append(r)
                tv = max(r["commanded_tip_travel_mm"].values())
                print(f"{label:22} {r['axis_height_mm']:6.1f} {st:6d} {r['peak_cos']:6.3f} "
                      f"{r['final_cos']:6.3f} {r['final_z']:7.4f} {r['contacts']:4d} "
                      f"{r['force_N']:7.2f} {r['max_ik_residual_mm']:7.2f} {tv:7.1f}  "
                      f"{'OK' if r['ok'] else 'dropped'}")
                if args.trace:
                    for t in r["trace"]:
                        print(f"       {t}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2))
    if args.contact_trace:
        args.contact_trace.parent.mkdir(parents=True, exist_ok=True)
        args.contact_trace.write_text(json.dumps(ctrace, indent=2))
        print(f"\n{'step':>5} {'deg':>6} {'cos':>6} "
              + "  ".join(f"{f[:3]}:fn/ft/util/slip" for f in FINGERS))
        for r in ctrace:
            cells = "  ".join(
                f"{v['fn_N']:5.1f}/{v['ft_N']:4.1f}/{v['cone_util']:4.2f}/{v['slip_mm']:4.1f}"
                for v in r["fingers"].values())
            print(f"{r['step']:5d} {r['cmd_deg']:6.1f} {r['cos']:6.3f}  {cells}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
