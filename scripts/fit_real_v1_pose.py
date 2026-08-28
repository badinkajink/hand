"""Author the grasp keyframe for one `real_v1` design — palm pose AND finger angles together.

WHY THIS EXISTS RATHER THAN `retarget_keyframe_ik.py`

The retargeter transfers a known-good grasp across morphologies by IK-ing the fingertips onto
the SAME world positions, holding the palm where the source keyframe put it. That is the right
tool when two hands share a grasp geometry. It is the wrong tool here for two reasons:

  1. There is no known-good pose to transfer. `real_v1` is a new topology — 78.66 mm of flexing
     reach where m05 has 117 mm, overlapping links, a ROM taken from CAD. `morphology-scenes`
     non-negotiable #3 says author from fingertip targets instead.
  2. Palm height is not shared across designs. A hand whose mounts sit 100 mm apart has to hang
     lower to reach the shaft than one whose mounts sit 40 mm apart, and inheriting one palm
     height for all of them is trap #1 in docs/experiments/8-27_LINK_LENGTH_GATE — it produced
     a spurious "2-FINGER, ungraspable" CEM verdict on a design that grasps fine. So palm z is
     SOLVED per design here, and palm x/y re-centre the pinch over the shaft.

Palm pose is legitimately free: it is the arm's pose, not a hardware parameter. Link lengths,
ROM, mounts and workspace all come from the user's CAD model untouched.

WHAT IT SOLVES

Fingertip targets ring the shaft axis at pad-contact distance (shaft radius + pad radius + a
approach gap), thumb from -x and index/middle from +x, lifted `--elevation` degrees above the
shaft's mid-height so the pads press down-and-in rather than scooping toward the table (a pad
21 mm across cannot get under a 25 mm shaft lying on a table without going through the floor).

Palm z is chosen as the DEEPEST height at which all three fingers still reach their target with
every joint clear of its limit — deepest because grip depth below the mounting plane is what
buys clearance for the shaft's upper half when it stands up (LINK_LENGTH_GATE §3), and the
`ceiling` column reports exactly that: depth / (tool length / 2), capped at 1.

    MUJOCO_GL=egl uv run python scripts/fit_real_v1_pose.py \
        --scene assets/mjcf/real_v1/scenes/scene_screwdriver_medium.xml --write

Writes `open_ik` (the CEM seed and the RL reset pose — the keyframe `--open-finger-from-keyframe`
reads) and, with `--also-open`, an `open` approach pose backed off by `--open-gap`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from morphohand.tools.keyframe_ik import (  # noqa: E402
    FINGERS,
    PALM_JOINTS,
    TIPS,
    actuator_ctrl_from_qpos,
    has_joint,
    ik_finger,
    inject_keyframe,
)

PAD_RADIUS = 0.010550          # <f>_tip sphere, real_v1 CAD half-width
LIMIT_MARGIN = 0.05            # rad; a joint closer than this to its stop counts as "parked"


def _object_geometry(m: mujoco.MjModel, d: mujoco.MjData, body: str) -> tuple[np.ndarray, float, float]:
    """(axis midpoint world xyz, radius, half-length) of the scene's cylinder object."""
    bid = m.body(body).id
    gids = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid]
    if not gids:
        raise ValueError(f"body '{body}' has no geom")
    g = gids[0]
    return d.body(bid).xpos.copy(), float(m.geom_size[g, 0]), float(m.geom_size[g, 1])


def tip_targets(center: np.ndarray, radius: float, gap: float, spread: float,
                elevation_deg: float) -> dict[str, np.ndarray]:
    """Pad-CENTRE targets ringing the shaft. The shaft lies along world Y.

    KEEP `elevation_deg` AT OR BELOW ZERO. It was +10 for one day and it cost a training run.
    Above the shaft's equator the contact normals tilt up and outward, so the wedge drives the
    shaft DOWN out of the pinch and only friction opposes it — and that friction is against a
    normal force which decays as the shaft creeps down and the pads ride toward the top of the
    cylinder, where the surface curves away. Positive feedback: measured on rv00_wide, the grip
    survives the entire 10 cm lift and then relaxes 1.75 N -> 0.00 N over 0.6 s of just holding,
    with the shaft's x/y unchanged to 1 mm. It falls straight out of the bottom.

    +10 was chosen because a 21 mm pad cannot get under a 25 mm shaft lying on a table without
    going through the floor. At 0 the pad's underside sits 2.0 mm clear of the floor, which is
    enough, and the user's own hand-authored grasp independently lands at +1.9 / -3.2 / +7.5 deg
    — on the equator. Blaming friction or pad compliance for this reads as plausible and is
    wrong: neither torsional friction, nor mu 2.4 -> 4.0, nor TPU-soft pads changes the outcome
    (scripts/probe_real_v1_slip.py). Wedge sign beats friction.
    """
    r = radius + PAD_RADIUS + gap
    el = np.deg2rad(elevation_deg)
    dz = r * np.sin(el)
    dx = r * np.cos(el)
    return {
        "thumb": center + np.array([-dx, 0.0, dz]),
        "index": center + np.array([dx, spread, dz]),
        "middle": center + np.array([dx, -spread, dz]),
    }


def _set_palm(m, d, px: float, py: float, pz: float) -> None:
    for name, val in (("palm_px", px), ("palm_py", py), ("palm_pz", pz)):
        if has_joint(m, name):
            d.qpos[m.jnt_qposadr[m.joint(name).id]] = val


def _mount_centre(m, d) -> tuple[float, float]:
    """Palm-frame XY midpoint of the thumb mount and the index/middle midpoint.

    The pinch closes along X between the thumb and the pair, so re-centring that midpoint over
    the shaft is what keeps a lopsided design from having to reach further with one side than
    the other. Read off the compiled model, so it works on a rigid scene whose morphology is
    already baked into the mount transforms.
    """
    palm = d.body("palm_pose").xpos
    t = d.body("thumb_mount").xpos - palm
    i = d.body("index_mount").xpos - palm
    mid = d.body("middle_mount").xpos - palm
    pair = 0.5 * (i + mid)
    return float(0.5 * (t[0] + pair[0])), float((t[1] + i[1] + mid[1]) / 3.0)


def _joint_margins(m, d) -> dict[str, float]:
    out = {}
    for f, joints in FINGERS.items():
        for j in joints:
            jid = m.joint(j).id
            q = float(d.qpos[m.jnt_qposadr[jid]])
            lo, hi = m.jnt_range[jid]
            out[j] = float(min(q - lo, hi - q))
    return out


def _seed_key(m: mujoco.MjModel, name: str) -> int:
    """Index of the keyframe the IK starts from.

    NOT key 0: a scene that has already been fitted once has `open_ik` sitting there, so seeding
    by index silently restarts from the previous solve — and on a generated scene that previous
    solve belongs to the BASE design's mounts and palm height. Seed by name, and fall back to
    key 0 only when the named key is genuinely absent.
    """
    for i in range(m.nkey):
        if m.key(i).name == name:
            return i
    return 0


# Damped-least-squares IK is local, and this finger has two qualitatively different ways to put
# its pad on the shaft: a PINCH (positive MCP, the tip swings inward under the mount) and a HOOK
# (negative MCP so the middle link leans outward, then strong PIP flexion to bring the pad back
# in). A design whose mounts sit INBOARD of the contact ring can only do it by hooking -- the
# 40 mm-separation designs need their pads 5 mm outboard of their own mounts, and flexion only
# moves a tip inboard. Seeded from a pinch, the solver reports them unreachable at every palm
# height, which is a false "ungraspable" verdict of exactly the kind this whole file exists to
# avoid. The user's own hand-authored grasp is a hook (MCP -0.22, PIP +0.96), which is what
# pointed at this.
SEED_POSES: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.55, 0.55),      # pinch, the shipped `open` placeholder
    (0.0, -0.22, 0.96),     # hook, the user's authored grasp
    (0.0, 0.0, 0.0),        # straight
)


def solve(m, d, targets: dict[str, np.ndarray], px: float, py: float, pz: float,
          seed_key: int = 0, iters: int = 300):
    """IK all three fingers at a given palm height, multi-started. Returns (residuals, margins).

    Per finger, every seed in SEED_POSES is tried plus the seed keyframe's own angles, and the
    solution with the smallest residual wins (ties broken toward the larger joint margin, since a
    pose that reaches by parking a joint on its stop has no squeeze authority).
    """
    base = m.key_qpos[seed_key].copy()
    best_q: dict[str, list[float]] = {}
    res: dict[str, float] = {}
    for f, joints in FINGERS.items():
        adr = [m.jnt_qposadr[m.joint(j).id] for j in joints]
        starts = [tuple(base[a] for a in adr)] + list(SEED_POSES)
        best = None
        for st in starts:
            d.qpos[:] = base
            _set_palm(m, d, px, py, pz)
            for a, v in zip(adr, st):
                d.qpos[a] = v
            mujoco.mj_forward(m, d)
            r = ik_finger(m, d, f, targets[f], iters=iters)
            q = [float(d.qpos[a]) for a in adr]
            rng = [m.jnt_range[m.joint(j).id] for j in joints]
            marg = min(min(qq - lo, hi - qq) for qq, (lo, hi) in zip(q, rng))
            key = (round(r, 5), -marg)
            if best is None or key < best[0]:
                best = (key, q, r)
        best_q[f], res[f] = best[1], best[2]

    d.qpos[:] = base
    _set_palm(m, d, px, py, pz)
    for f, joints in FINGERS.items():
        for j, v in zip(joints, best_q[f]):
            d.qpos[m.jnt_qposadr[m.joint(j).id]] = v
    mujoco.mj_forward(m, d)
    return res, _joint_margins(m, d)


def hold_probe(m, d, open_qpos: np.ndarray, grip_ctrl: np.ndarray, lift: float,
               obj_z0: float, object_body: str,
               settle: int = 250, ramp: int = 200, hold: int = 600) -> float:
    """Close on the shaft from `open_qpos` toward `grip_ctrl`, lift, hold; report lift retained.

    `grip_ctrl` is the IK solution for fingertip targets placed INSIDE the shaft's surface, so
    driving the position actuators to it presses each pad radially inward — the squeeze is
    per-finger and along each finger's own approach direction. A uniform "+0.15 rad at MCP and
    PIP" was tried first and dropped the shaft on every candidate of every design, including
    ones CEM grasps perfectly: flexion is not "close", it curls the tip past the object
    (mujoco-eyes gotcha #1), so a uniform flexion offset is not a squeeze.

    This is `phase1_optimize_grasp`'s schedule with a geometric grip instead of a CEM-optimised
    one — deliberately, because the question is whether the POSE is any good, not how good CEM
    can make it. Returns metres of lift still held; near zero means the shaft is back down.

    The hold is 600 steps (1.2 s), not the 150 it started at. The failure this probe now exists
    to catch takes ~0.6 s to develop: an above-equator grip holds through the whole lift and then
    the normal force decays monotonically to zero while the shaft creeps down through the pads.
    A 0.3 s hold reports that grasp as fine.
    """
    pz_a = next(k for k in range(m.nu) if m.actuator(k).name == "a_palm_pz")
    pz0 = float(grip_ctrl[pz_a])

    d.qpos[:] = open_qpos
    d.qvel[:] = 0.0
    d.ctrl[:] = grip_ctrl
    mujoco.mj_forward(m, d)
    for _ in range(settle):
        mujoco.mj_step(m, d)
    for k in range(ramp):
        d.ctrl[pz_a] = pz0 + lift * (k + 1) / ramp
        mujoco.mj_step(m, d)
    for _ in range(hold):
        mujoco.mj_step(m, d)
    return float(d.body(object_body).xpos[2] - obj_z0)


def _feasible(res: dict, marg: dict, res_tol: float) -> bool:
    return max(res.values()) < res_tol and min(marg.values()) > LIMIT_MARGIN


def _deepest(m, d, targets, px, py, seed, res_tol, pz_lo, pz_hi, pz_step):
    """Deepest palm height at which the tripod is reachable with every joint off its stop.

    Coarse scan then a fine pass around the coarse hit. Straight fine scanning over the whole
    range is 5x the IK calls for the same answer, and the answer is only ever near the bottom
    edge of the feasible band.
    """
    coarse = None
    pz = pz_hi
    while pz >= pz_lo - 1e-9:
        res, marg = solve(m, d, targets, px, py, pz, seed, iters=300)
        if _feasible(res, marg, res_tol):
            coarse = pz
            break
        pz -= pz_step
    if coarse is None:
        return None
    best = coarse
    pz = coarse + pz_step - pz_step / 5.0
    while pz > coarse:
        res, marg = solve(m, d, targets, px, py, pz, seed, iters=300)
        if _feasible(res, marg, res_tol):
            best = pz
            break
        pz -= pz_step / 5.0
    return best


def fit(scene: Path, gap: float, spread: float, elevation: float, object_body: str,
        pz_lo: float, pz_hi: float, pz_step: float, verbose: bool = True,
        seed_keyframe: str = "open", res_tol: float = 0.004,
        spreads: tuple[float, ...] | None = None,
        spread_max: float | None = None, spread_min: float = 0.020,
        spread_frac: float = 0.85, squeeze: float = 0.004, lift_probe: float = 0.05,
        hold_min: float = 0.020):
    """Solve palm pose + finger angles for one design.

    Two things are fitted rather than fixed, both for the same reason: they are choices about
    HOW to grasp, not hardware parameters, and pinning them turns a reachable design into a
    false "ungraspable" -- the same class of error as inheriting another hand's palm height.

      palm z   the deepest height at which all three fingers reach with every joint off its
               stop. Deeper = more clearance for the shaft's upper half when it stands up.
      spread   how far apart index and middle sit along the shaft. Candidates are scored by an
               actual close-lift-hold rollout -- see the block comment below for the two
               reachability heuristics that were tried first and what each of them cost.

    `res_tol` is 4 mm, not sub-millimetre. The pad is 10.55 mm in radius and the targets carry a
    1 mm approach gap, so a few millimetres off the intended ring position is still on the pad and
    CEM closes it -- and the residual was only ever a PROXY for "is this pose any good", which the
    hold probe now answers directly. At 2 mm the most compact designs (thumb and pair 40 mm apart,
    fingers at 0.91 of their reach) missed by 2.1-3.6 mm and were reported unreachable at every
    palm height, which is a false "ungraspable" verdict. The binding condition is the joint
    MARGIN -- a pose that reaches its target by parking a joint on its stop has no squeeze
    authority in that direction and is not a grasp -- and then the probe.
    """
    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)
    seed = _seed_key(m, seed_keyframe)
    mujoco.mj_resetDataKeyframe(m, d, seed)
    mujoco.mj_forward(m, d)

    centre, radius, half_len = _object_geometry(m, d, object_body)
    # Re-centre the pinch over the shaft: the mount midpoint should sit above the object.
    cx, cy = _mount_centre(m, d)
    px, py = float(centre[0] - cx), float(centre[1] - cy)

    # WHICH (spread, palm height) PAIR IS CHOSEN IS DECIDED BY A ROLLOUT, NOT BY REACHABILITY.
    #
    # Two heuristics were tried first and both produced false verdicts on 2026-08-27:
    #
    #   "nominal 30 mm spread, first reachable palm height"  -> rv01_compact's CEM grasp lifted
    #      the shaft and DROPPED it (held lift -1.4 mm, persistence 0.86). Re-fitting the same
    #      hand at 40 mm: held +47.0 mm, persistence 1.00. The design was never the problem.
    #   "widest reachable spread"                            -> fixed rv01 and BROKE rv00_wide,
    #      which had been holding at 30 mm (held +48.2 mm) and dropped at ~42 mm.
    #
    # Reachability cannot rank poses because it does not know about the pitch-out failure: the
    # thumb sits at y=0 and the pair straddles it, and whether that tripod resists the shaft
    # levering about the pinch axis depends on the straddle AND on how far the fingers are
    # extended to hold it, which trade against each other. So each candidate is now driven
    # through a scripted close -> lift -> hold and scored on what it is still holding at the
    # end. It costs ~1 s of CPU per design and it is the same open-loop-probe-before-RL rule
    # the rest of this program runs on.
    if spreads is None:
        hi = min(spread_max if spread_max else spread_frac * half_len, half_len - 0.005)
        n = max(1, int(round((hi - spread_min) / 0.005)))
        spreads = tuple(hi - k * 0.005 for k in range(n + 1))
    spreads = tuple(s for s in spreads if 0.005 <= s <= half_len - 0.005)

    obj_z0 = float(centre[2])
    cands = []
    for sp in spreads:
        targets = tip_targets(centre, radius, gap, sp, elevation)
        pz_top = _deepest(m, d, targets, px, py, seed, res_tol, pz_lo, pz_hi, pz_step)
        if pz_top is None:
            if verbose:
                print(f"  spread {sp*1000:.0f}mm: unreachable at every palm height")
            continue
        for drop in (0.0, 0.005, 0.010):
            pz = pz_top - drop
            res, marg = solve(m, d, targets, px, py, pz, seed, iters=600)
            if not _feasible(res, marg, res_tol):
                continue
            open_qpos = d.qpos.copy()
            # The squeeze: the same targets pulled `squeeze` metres INSIDE the shaft surface.
            solve(m, d, tip_targets(centre, radius, gap - squeeze, sp, elevation),
                  px, py, pz, seed, iters=600)
            grip_ctrl = np.array(actuator_ctrl_from_qpos(m, d))
            held = hold_probe(m, d, open_qpos, grip_ctrl, lift_probe, obj_z0, object_body)
            cands.append({"spread": sp, "pz": pz, "held": held,
                          "res": max(res.values()), "marg": min(marg.values())})
            if verbose:
                print(f"  spread {sp*1000:4.0f}mm  palm_z {pz:+.4f}  held {held*1000:+7.2f}mm"
                      f"  res {max(res.values())*1000:.2f}mm  marg {min(marg.values()):.2f}")

    # A pose only counts if it actually HOLDS. Returning the best of a set of poses that all
    # drop the shaft is how a design gets handed to a 90-minute training run on a grasp that
    # cannot work; the caller needs to hear "no viable pose", not a ranking of failures.
    cands = [c for c in cands if c["held"] >= hold_min]
    if not cands:
        return None
    # Best hold wins. Ties (within 1 mm of the best) go to the DEEPER palm -- grip depth below
    # the mounting plane is what clears the shaft's upper half when it stands up -- then to the
    # wider straddle.
    top = max(c["held"] for c in cands)
    best_c = max((c for c in cands if c["held"] >= top - 0.001),
                 key=lambda c: (c["pz"], c["spread"]))
    best, spread = best_c["pz"], best_c["spread"]

    if verbose:
        print(f"scene   {scene}")
        print(f"object  {object_body} centre {np.round(centre, 4)} r={radius*1000:.1f}mm "
              f"half-len={half_len*1000:.1f}mm")
        print(f"palm xy re-centre  px={px*1000:+.1f}mm py={py*1000:+.1f}mm")

    if best is None:
        return None

    targets = tip_targets(centre, radius, gap, spread, elevation)
    res, marg = solve(m, d, targets, px, py, best, seed, iters=600)
    depth = float(d.body("palm_pose").xpos[2] - centre[2])
    ceiling = min(1.0, depth / half_len) if half_len > 0 else float("nan")
    report = {
        "scene": str(scene),
        "palm": {"px": px, "py": py, "pz": best},
        "palm_z_world": float(d.body("palm_pose").xpos[2]),
        "grip_depth_mm": depth * 1000.0,
        "clearance_ceiling": ceiling,
        "tip_residual_mm": {f: v * 1000.0 for f, v in res.items()},
        "joint_margin_rad": marg,
        "targets": {f: t.tolist() for f, t in targets.items()},
        "gap_mm": gap * 1000.0,
        "spread_mm": spread * 1000.0,
        "elevation_deg": elevation,
        "probe_held_lift_mm": best_c["held"] * 1000.0,
        "probe_candidates": [
            {"spread_mm": c["spread"] * 1000.0, "pz": c["pz"], "held_mm": c["held"] * 1000.0}
            for c in cands
        ],
    }
    qpos = " ".join(f"{v:.6g}" for v in d.qpos)
    ctrl = " ".join(f"{v:.6g}" for v in actuator_ctrl_from_qpos(m, d))

    # A pose is only good if it is also non-interpenetrating. The by-construction overlaps are
    # already <exclude>d in the scene, so anything here other than the pads on the shaft is a
    # real collision the design cannot be evaluated through.
    mujoco.mj_forward(m, d)
    bad = []
    for c in d.contact[: d.ncon]:
        b1 = m.body(m.geom_bodyid[c.geom1]).name
        b2 = m.body(m.geom_bodyid[c.geom2]).name
        pair = {b1, b2}
        if pair & {"thumb_tip", "index_tip", "middle_tip"} and object_body in pair:
            continue
        if c.dist > -1e-4:
            continue
        bad.append((b1, b2, float(c.dist)))
    report["self_collisions"] = bad
    if verbose:
        print(f"chosen palm_z {best:.4f}  (world {report['palm_z_world']:.4f})  "
              f"grip depth {report['grip_depth_mm']:.1f}mm  ceiling {ceiling:.2f}  "
              f"spread {spread*1000:.0f}mm")
        for f in FINGERS:
            print(f"  {f:7} residual {res[f]*1000:5.2f}mm   "
                  + "  ".join(f"{j.split('_')[-1]} {marg[j]:+.2f}" for j in FINGERS[f]))
        print(f"  self-collisions: {bad if bad else 'none'}")
    return report, qpos, ctrl


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True, type=Path)
    ap.add_argument("--object-body", default="screwdriver_medium")
    ap.add_argument("--seed-keyframe", default="open",
                    help="keyframe the IK starts from (never by index -- see _seed_key)")
    ap.add_argument("--keyframe", default="open_ik", help="keyframe name to write")
    ap.add_argument("--gap", type=float, default=0.001,
                    help="pad-surface clearance from the shaft surface, metres (CEM closes it)")
    ap.add_argument("--spread", type=float, default=None,
                    help="pin the index/middle straddle instead of fitting it, metres")
    ap.add_argument("--spread-frac", type=float, default=0.85,
                    help="widest straddle to try, as a fraction of the shaft's half-length")
    ap.add_argument("--spread-min", type=float, default=0.020)
    ap.add_argument("--squeeze", type=float, default=0.004,
                    help="metres the hold probe drives each pad inside the shaft's surface")
    ap.add_argument("--lift-probe", type=float, default=0.05,
                    help="metres the hold probe raises the palm by")
    ap.add_argument("--hold-min", type=float, default=0.020,
                    help="metres of lift the probe must still hold for a pose to count")
    ap.add_argument("--elevation", type=float, default=0.0,
                    help="degrees above the shaft's equator to place the pads; see tip_targets")
    ap.add_argument("--pz-lo", type=float, default=-0.030)
    ap.add_argument("--pz-hi", type=float, default=0.060)
    ap.add_argument("--pz-step", type=float, default=0.0025)
    ap.add_argument("--res-tol", type=float, default=0.004,
                    help="max fingertip IK residual, metres; CEM closes anything under the pad")
    ap.add_argument("--also-open", action="store_true",
                    help="additionally write an `open` approach pose at --open-gap")
    ap.add_argument("--open-gap", type=float, default=0.008)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--json", type=Path, default=None, help="write the report here")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    out = fit(args.scene, args.gap, args.spread or 0.030, args.elevation, args.object_body,
              args.pz_lo, args.pz_hi, args.pz_step, verbose=not args.quiet,
              seed_keyframe=args.seed_keyframe, res_tol=args.res_tol,
              spreads=(args.spread,) if args.spread else None,
              spread_min=args.spread_min, spread_frac=args.spread_frac,
              squeeze=args.squeeze, lift_probe=args.lift_probe, hold_min=args.hold_min)
    if out is None:
        print(f"FAIL {args.scene.name}: no pose reaches the shaft AND holds it "
              f"(needs >= {args.hold_min*1000:.0f} mm of lift retained)")
        return 2
    report, qpos, ctrl = out
    if args.write:
        inject_keyframe(args.scene, args.keyframe, qpos, ctrl)
        print(f"wrote keyframe '{args.keyframe}' into {args.scene}")
    if args.also_open:
        # The approach pose shares the solved palm height so that closing is a pure finger
        # motion — a policy that has to move the palm to close is not the grasp we are studying.
        m = mujoco.MjModel.from_xml_path(str(args.scene))
        d = mujoco.MjData(m)
        seed = _seed_key(m, args.seed_keyframe)
        mujoco.mj_resetDataKeyframe(m, d, seed)
        mujoco.mj_forward(m, d)
        centre, radius, _ = _object_geometry(m, d, args.object_body)
        # Same spread the fit SETTLED on, not the requested one -- an approach pose at a
        # different spread than the grasp makes closing a lateral move, not a squeeze.
        tg = tip_targets(centre, radius, args.open_gap, report["spread_mm"] / 1000.0,
                         args.elevation)
        p = report["palm"]
        solve(m, d, tg, p["px"], p["py"], p["pz"], seed)
        if args.write:
            inject_keyframe(args.scene, "open",
                            " ".join(f"{v:.6g}" for v in d.qpos),
                            " ".join(f"{v:.6g}" for v in actuator_ctrl_from_qpos(m, d)))
            print(f"wrote keyframe 'open' into {args.scene}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
