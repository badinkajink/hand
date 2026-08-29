"""Sample `real_v1` hands, score each one, and measure the reorient it actually produces.

WHY THIS EXISTS

`real_v1_reorient_landscape.py` swept nine cells of the two knobs the 3-parameter compact family
has and got 9/9 agreement with `ceiling = asin(extension_left / straddle)`. Nine cells of one
family is not a landscape, the family cannot express an asymmetric hand (the user's own
`rv05_manual` is asymmetric and is the best design in the study), and the ceiling was derived for
a FIXED-CONTACT rotation -- which the successful carries turn out not to be. Measured on
rv05_manual, the pads slide 12-28 mm across the shaft over the turn against a 19.6 mm surface
arc, i.e. the index pad is a near-stationary bearing and the middle pad over-travels. So the
turn is a sliding pivot, and a metric built on contacts that do not move cannot be the whole
story.

WHAT IT MEASURES

Per (design, grasp) it records a set of CANDIDATE cheap scores computed from the settled grasp
alone -- no policy, no seed, one rollout of 650 steps -- and then the open-loop carry those
scores claim to stand in for, over a sweep of pivot heights. Cheap scores:

    extend_mm      radial reach left before the finger is straight  (the old ceiling's input)
    ceiling_deg    asin(extend / straddle), the fixed-contact bound
    arm_*_mm       |r_yz| of each pad from the shaft's centre = its moment arm about the PINCH
                   AXIS. A pad at the shaft's mid-length has arm 0 and cannot torque the shaft
                   upright however hard it presses -- which is exactly what the thumb does.
    tau_cap_Nmm    mu * sum_f f_n,f * arm_f, the friction-limited couple the grasp can apply
    sweep_mm       first-order tangential authority: how far each pad can move ALONG the turn
                   inside the +-0.5 rad residual budget, 0.5 * sum_j |u . J[:,j]|
    sweep_ratio    sweep_mm / (arm * theta), demand-normalised

and a STYLE vector from the winning cell's contact trace (carry fraction per finger, drive
share, object descent, contact count), because the designs do not all reorient the same way.

    MUJOCO_GL=egl uv run python scripts/real_v1_design_search.py --set axes --stage generate
    MUJOCO_GL=egl uv run python scripts/real_v1_design_search.py --set axes --shard 0 --shards 10
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import mujoco  # noqa: E402

import probe_real_v1_carry as pc  # noqa: E402
from morphohand.sampling.morphology import (  # noqa: E402
    REAL_V1_MOUNTS, REAL_V1_WORKSPACE, morph_to_array, real_v1_compact_design,
)
from morphohand.tools.keyframe_ik import FINGERS  # noqa: E402
from morphohand.tools.morphology_xml import MorphologyValues  # noqa: E402

BASE_HAND = ROOT / "assets/mjcf/real_v1/real_hand.xml"
BASE_SCENE = ROOT / "assets/mjcf/real_v1/scenes/scene_screwdriver_medium.xml"
DATE = "20260828"
GEN = ROOT / f"assets/mjcf/experimental/{DATE}-real_v1_search"
OBJ = "screwdriver_medium"
TIPS = {f: f"{f}_tip" for f in FINGERS}
CHAIN_MM = 68.11          # straight mount-to-pad reach, yaw 20.75 + 20.75 + pad 26.61


# --------------------------------------------------------------------------------------------
# design sets
# --------------------------------------------------------------------------------------------
def _v(tx=0.0, ty=0.0, ix=0.0, iy=0.0, mx=0.0, my=0.0) -> tuple:
    """A 9-vector from the six XY slides, clipped to the gantry workspace."""
    b = REAL_V1_WORKSPACE
    c = lambda v, lo, hi: float(min(max(v, lo), hi))  # noqa: E731
    return (c(tx, b.thumb.x_min, b.thumb.x_max), c(ty, b.thumb.y_min, b.thumb.y_max), 0.0,
            c(ix, b.index.x_min, b.index.x_max), c(iy, b.index.y_min, b.index.y_max), 0.0,
            c(mx, b.middle.x_min, b.middle.x_max), c(my, b.middle.y_min, b.middle.y_max), 0.0)


def design_set(name: str) -> dict[str, tuple]:
    """Named design sets. Every one includes the four studied hands so the sweep is anchored."""
    known = {
        # the five that went through the A->B pipeline, so every sweep can be read against them
        "rv00_wide": tuple(morph_to_array(real_v1_compact_design(0.0, 0.0, 0.0))),
        "rv01_compact": tuple(morph_to_array(real_v1_compact_design(1.0, 1.0, 1.0))),
        "rv03_narrowy": tuple(morph_to_array(real_v1_compact_design(0.0, 0.0, 1.0))),
        "rv04_mid": tuple(morph_to_array(real_v1_compact_design(0.5, 0.5, 0.5))),
        "rv05_manual": (0.011, 0.0, 0.0, -0.015, -0.03, 0.0, -0.0015, 0.02, 0.0),
    }
    out = dict(known)
    if name == "known":
        return out
    if name == "axes":
        # One knob at a time off the CAD-nominal hand, so each axis is separable. thumb_y and an
        # asymmetric pair are in here because NOTHING has ever sampled them: every design in the
        # study to date has thumb_y = 0 and index_y = -middle_y.
        for v in (-0.030, -0.020, -0.010, 0.010, 0.020, 0.030):
            out[f"ax_tx{v*1000:+.0f}"] = _v(tx=v)
            out[f"ax_px{v*1000:+.0f}"] = _v(ix=v, mx=v)
        for v in (-0.030, -0.020, -0.010, 0.010, 0.020, 0.030):
            out[f"ax_py{v*1000:+.0f}"] = _v(iy=v, my=-v)          # symmetric pair-Y
        for v in (-0.055, -0.040, -0.025, 0.025, 0.040, 0.055):
            out[f"ax_ty{v*1000:+.0f}"] = _v(ty=v)                 # thumb ALONG the shaft
        for v in (-0.030, -0.020, -0.010, 0.010, 0.020, 0.030):
            out[f"ax_asym{v*1000:+.0f}"] = _v(iy=v, my=v)         # pair slid together along Y
        return out
    if name == "grid":
        # Xsep x Ysep, five levels each, the plane the earlier 3x3 landscape sampled at 3.
        for xi, xt in enumerate(np.linspace(0.0, 1.0, 5)):
            for yi, yt in enumerate(np.linspace(0.0, 1.0, 5)):
                out[f"g{xi}{yi}"] = tuple(morph_to_array(real_v1_compact_design(xt, xt, yt)))
        return out
    if name == "random":
        rng = np.random.default_rng(20260828)
        b = REAL_V1_WORKSPACE
        for i in range(48):
            out[f"r{i:02d}"] = _v(
                tx=rng.uniform(b.thumb.x_min, b.thumb.x_max),
                ty=rng.uniform(b.thumb.y_min, b.thumb.y_max),
                ix=rng.uniform(b.index.x_min, b.index.x_max),
                iy=rng.uniform(b.index.y_min, b.index.y_max),
                mx=rng.uniform(b.middle.x_min, b.middle.x_max),
                my=rng.uniform(b.middle.y_min, b.middle.y_max))
        return out
    if name == "all":
        for sub in ("axes", "grid", "random"):
            out.update(design_set(sub))
        return out
    raise ValueError(f"unknown design set {name!r}")


def scene_for(vec) -> Path:
    """Generate (or find) the rigid scene for a design. Morph joints baked out."""
    GEN.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([sys.executable, str(ROOT / "scripts/generate_morphology_xml.py"),
                        "--base-hand-xml", str(BASE_HAND), "--base-scene-xml", str(BASE_SCENE),
                        "--output-dir", str(GEN),
                        "--thumb", *map(str, vec[0:3]), "--index", *map(str, vec[3:6]),
                        "--middle", *map(str, vec[6:9])],
                       check=True, capture_output=True, text=True, timeout=180)
    for line in r.stdout.splitlines():
        if "scene_" in line and line.strip().endswith(".xml"):
            return Path(line.split()[-1])
    return sorted(GEN.glob("scene_*.xml"), key=lambda p: p.stat().st_mtime)[-1]


# --------------------------------------------------------------------------------------------
# the cheap scores, all read off ONE settled grasp
# --------------------------------------------------------------------------------------------
def _settle(m, open_qpos, grip, lift: float):
    """Policy A's schedule, open-loop: close 250, ramp the palm up over 200, settle 200."""
    d = mujoco.MjData(m)
    d.qpos[:] = open_qpos
    d.qvel[:] = 0.0
    d.ctrl[:] = grip
    mujoco.mj_forward(m, d)
    pz_a = next(k for k in range(m.nu) if m.actuator(k).name == "a_palm_pz")
    for _ in range(250):
        mujoco.mj_step(m, d)
    pz0 = float(d.ctrl[pz_a])
    for k in range(200):
        d.ctrl[pz_a] = pz0 + lift * (k + 1) / 200
        mujoco.mj_step(m, d)
    for _ in range(200):
        mujoco.mj_step(m, d)
    return d


def grasp_scores(m, d, sign: float = -1.0) -> dict:
    """Candidate design scores at a settled grasp. FK + one Jacobian per finger.

    `sign` is the turn direction: -1 matches the -90 deg carry, which is the pole
    `target_axis_alignment` pays for.
    """
    o = d.body(OBJ).xpos.copy()
    pf = pc._per_finger_contact(m, d, OBJ)
    out, arms, sweeps, taus = {}, {}, {}, []
    jacp = np.zeros((3, m.nv))
    for f in FINGERS:
        tip = d.body(TIPS[f]).xpos.copy()
        root = d.body(f"{f}_yaw_frame").xpos.copy()
        r = tip - o
        arm = float(np.hypot(r[1], r[2]))                     # moment arm about world X
        reach = float(np.linalg.norm(tip - root))
        # tangential direction of the commanded turn at this pad
        u = np.cross(np.array([sign, 0.0, 0.0]), r)
        nu = float(np.linalg.norm(u))
        u = u / nu if nu > 1e-9 else np.zeros(3)
        mujoco.mj_jacBody(m, d, jacp, None, m.body(TIPS[f]).id)
        cols = [m.jnt_dofadr[m.joint(j).id] for j in FINGERS[f]]
        # +-0.5 rad per joint is the trainer's finger_residual_scale and the carry's clip; the
        # reachable tip displacement inside that box is a zonotope, and its extent along `u` is
        # the L1 sum of the projected Jacobian columns.
        sweep = 0.5 * float(sum(abs(float(u @ jacp[:, c])) for c in cols))
        mu, fn = pf[f]["mu"], pf[f]["fn"]
        taus.append(mu * fn * arm)
        arms[f], sweeps[f] = arm, sweep
        out[f] = {"arm_mm": round(arm * 1000, 2), "reach_mm": round(reach * 1000, 2),
                  "extend_mm": round(CHAIN_MM - reach * 1000, 2),
                  "fn_N": round(fn, 2), "sweep_mm": round(sweep * 1000, 2),
                  "sweep_ratio": round(sweep / (arm * np.pi / 2), 3) if arm > 1e-4 else None}
    straddle = 0.5 * abs(d.body(TIPS["index"]).xpos[1] - d.body(TIPS["middle"]).xpos[1])
    extend = min(v["extend_mm"] for v in out.values())
    return {
        "fingers": out,
        "extend_mm": round(extend, 2),
        "straddle_mm": round(straddle * 1000, 2),
        "ceiling_deg": round(float(np.degrees(np.arcsin(
            min(1.0, max(0.0, extend / 1000.0) / max(straddle, 1e-6))))), 2),
        # the couple the grasp can apply about the pinch axis before a pad slides
        "tau_cap_Nmm": round(float(sum(taus)) * 1000, 2),
        "tau_pair_Nmm": round(float(sum(pf[f]["mu"] * pf[f]["fn"] * arms[f]
                                        for f in ("index", "middle"))) * 1000, 2),
        "tau_thumb_Nmm": round(float(pf["thumb"]["mu"] * pf["thumb"]["fn"]
                                     * arms["thumb"]) * 1000, 2),
        "sweep_min_mm": round(min(sweeps.values()) * 1000, 2),
        "sweep_sum_mm": round(sum(sweeps.values()) * 1000, 2),
        "grip_N": round(sum(v["fn_N"] for v in out.values()), 2),
        "grip_depth_mm": round(float(d.body("palm_pose").xpos[2] - d.body(OBJ).xpos[2]) * 1000, 1),
        "grasp_z": round(float(d.body(OBJ).xpos[2]), 4),
        "grasp_contacts": pc._contacts(m, d, OBJ)[0],
    }


def geometry(vec) -> dict:
    """Mount separations, in millimetres — the readable form of the design vector."""
    t = (REAL_V1_MOUNTS["thumb"][0] + vec[0], REAL_V1_MOUNTS["thumb"][1] + vec[1])
    i = (REAL_V1_MOUNTS["index"][0] + vec[3], REAL_V1_MOUNTS["index"][1] + vec[4])
    mi = (REAL_V1_MOUNTS["middle"][0] + vec[6], REAL_V1_MOUNTS["middle"][1] + vec[7])
    return {"x_sep_mm": round((0.5 * (i[0] + mi[0]) - t[0]) * 1000, 1),
            "y_sep_mm": round(abs(i[1] - mi[1]) * 1000, 1),
            "thumb_y_mm": round(t[1] * 1000, 1),
            "pair_y_mid_mm": round(0.5 * (i[1] + mi[1]) * 1000, 1),
            "mounts_mm": {k: [round(v[0] * 1000, 1), round(v[1] * 1000, 1)]
                          for k, v in (("thumb", t), ("index", i), ("middle", mi))}}


# --------------------------------------------------------------------------------------------
# style: HOW the design turns the shaft, from the contact trace
# --------------------------------------------------------------------------------------------
def style(trace: list, r: dict, radius: float = 0.0125) -> dict:
    """Carry vs spin, per finger, plus who drives.

    `carry_frac` = 1 - (pad travel across the shaft's surface) / (surface arc the turn sweeps).
    1.0 = the pad rides the object with the contact fixed (a true carry). 0.0 = the pad stands
    still in the palm and the object spins under it (a bearing). Negative = the pad over-travels,
    i.e. it is driving the rotation and outrunning it.
    """
    if not trace:
        return {}
    turned = abs(float(np.degrees(np.arccos(np.clip(r["final_cos"], -1, 1)))
                       - np.degrees(np.arccos(np.clip(r["start_cos"], -1, 1)))))
    arc = radius * np.radians(max(turned, 1.0))
    # SLIP ONLY COUNTS WHILE THE PAD IS ON THE SHAFT. The trace measures the tip in the object's
    # frame whether or not it is touching, so a finger that lets go and swings away books
    # hundreds of millimetres of "slip" on a 78 mm circumference -- rv00_wide's index read
    # 485.6 mm and a carry fraction of -42.8. Gate on normal force and normalise the arc by how
    # much of the turn the pad was actually there for.
    on = {f: [row["fingers"][f]["fn_N"] > 0.1 for row in trace] for f in FINGERS}
    touch = {f: float(np.mean(v)) for f, v in on.items()}
    slip = {f: sum(row["fingers"][f]["slip_mm"] for row, k in zip(trace, on[f]) if k) / 1000.0
            for f in FINGERS}
    ft = {f: sum(row["fingers"][f]["ft_N"] for row in trace) for f in FINGERS}
    fn = {f: sum(row["fingers"][f]["fn_N"] for row in trace) for f in FINGERS}
    tot = sum(ft.values()) or 1.0
    return {
        "turned_deg": round(turned, 1),
        "arc_mm": round(arc * 1000, 2),
        "touch_frac": {f: round(v, 2) for f, v in touch.items()},
        "slip_mm": {f: round(v * 1000, 1) for f, v in slip.items()},
        # 1.0 = the pad rides the shaft with the contact fixed (a carry); 0.0 = the pad stands
        # still and the shaft spins under it (a bearing); negative = the pad outruns the
        # rotation, i.e. it is driving. None when the pad was never on the shaft.
        "carry_frac": {f: (None if touch[f] < 0.05
                           else round(float(np.clip(1.0 - slip[f] / (arc * touch[f]),
                                                    -3.0, 1.5)), 2))
                       for f in FINGERS},
        "drive_share": {f: round(v / tot, 2) for f, v in ft.items()},
        "mean_fn_N": {f: round(v / len(trace), 2) for f, v in fn.items()},
        "max_util": {f: round(max(row["fingers"][f]["cone_util"] for row in trace), 2)
                     for f in FINGERS},
        "mean_contacts": round(float(np.mean([row["contacts"] for row in trace])), 2),
        "obj_dz_mm": round((trace[-1]["z"] - trace[0]["z"]) * 1000, 1),
        "driver": max(ft, key=ft.get),
        # How much of the alignment arrives AFTER the command stops. On the linear anchor the
        # shaft goes on settling into vertical against a still-loaded grip -- rv05_manual runs
        # 0.837 at the end of the turn to 0.999 half a second later -- so a schedule judged at
        # the end of its own sweep is judged early.
        "cos_turn_end": trace[-1]["cos"],
        "settle_frac": round(float((r["final_cos"] - trace[-1]["cos"])
                                   / max(1e-6, abs(r["final_cos"] - r["start_cos"]))), 2),
    }


# --------------------------------------------------------------------------------------------
def evaluate(tag: str, vec, args) -> dict:
    scene = scene_for(vec)
    row = {"design": tag, "vector": [round(float(v), 4) for v in vec],
           "scene": str(scene.relative_to(ROOT)), **geometry(vec), "grasps": []}
    for st, dp, ta, sq in itertools.product(args.straddles, args.depths,
                                            args.thumb_axials, args.squeezes):
        # GRIP DEPTH IS THE KNOB THAT BUYS EXTENSION BUDGET, and it is the one the fitter
        # normally spends: it takes the DEEPEST reachable palm, for clearance over the shaft's
        # upper half, which parks the fingers at ~95% extension with nothing left to give. A
        # wide hand cannot go deep (its fingers run out of reach first) and so keeps its budget
        # by accident; a compact hand can, and loses it. Sweeping depth is what separates the
        # design from the fitter's choice.
        built = pc._grip_from_fit(scene, st, 0.0, sq, OBJ, dp, ta)
        label = {"straddle_mm": round(st * 1000, 1), "thumb_axial_mm": round(ta * 1000, 1),
                 "depth_req_mm": None if dp is None else round(dp * 1000, 1),
                 "squeeze_mm": round(sq * 1000, 1)}
        if built is None:
            row["grasps"].append({**label, "pose": False})
            continue
        m, open_qpos, grip, depth_mm = built
        sc = grasp_scores(m, _settle(m, open_qpos, grip, args.lift))
        sc["depth_fit_mm"] = round(float(depth_mm), 1)
        g = {**label, "pose": True, "scores": sc, "cells": []}
        for k, mode, ang in itertools.product(args.axis_ks, args.modes, args.angles):
            reps, trace = [], []
            for rep in range(args.repeats):
                # The style trace is taken from repeat 0 only; it is a description of the
                # behaviour, and averaging pad-slip curves across draws that end differently
                # would describe none of them.
                tr = trace if rep == 0 else None
                r = pc.carry(scene, args.lift, args.turn_steps, args.hold_steps,
                             np.radians(ang), 0.0, OBJ, args.budget, False,
                             straddle=st, label=tag, axis_k=k,
                             linear_anchor=(mode == "linear"), built=built, contact_trace=tr,
                             jitter=(args.jitter if args.repeats > 1 else 0.0), seed=rep)
                if r is None:
                    break
                reps.append(r)
            if not reps:
                continue
            fin = [x["final_cos"] for x in reps]
            r0 = reps[0]
            g["cells"].append({
                "axis_k": k, "mode": mode, "angle_deg": ang, "n": len(reps),
                "mean_cos": round(float(np.mean(fin)), 3),
                "sd_cos": round(float(np.std(fin)), 3),
                "kept": sum(1 for x in reps if x["ok"]),
                "peak_cos": r0["peak_cos"], "final_cos": r0["final_cos"],
                "final_z": r0["final_z"], "min_z_hold": r0["min_z_hold"],
                "contacts": r0["contacts"], "contacts_hand": r0["contacts_hand"],
                "force_N": r0["force_N"], "ok": r0["ok"], "style": style(trace, r0)})
        row["grasps"].append(g)
    return row


def best_cell(row: dict) -> dict | None:
    """The design's best shot: highest held final cos over every grasp and pivot height."""
    best = None
    for g in row["grasps"]:
        for c in g.get("cells", []):
            # Rank on the REPEATED mean, gated on the cell keeping the shaft in most draws.
            # Ranking on a single rollout's final cos ranks the luckiest draw of a schedule
            # whose good cells sit in narrow resonances -- rv05_manual reads 0.99 and 0.00 at
            # neighbouring pivot heights.
            key = (c["kept"] * 2 >= c["n"], c["mean_cos"])
            if best is None or key > (best["kept"] * 2 >= best["n"], best["mean_cos"]):
                best = {**c, "straddle_mm": g["straddle_mm"],
                        "thumb_axial_mm": g["thumb_axial_mm"],
                        "depth_req_mm": g["depth_req_mm"],
                        "squeeze_mm": g["squeeze_mm"], "scores": g["scores"]}
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default="axes", help="known | axes | grid | random")
    ap.add_argument("--only", default="", help="comma list of design tags")
    ap.add_argument("--stage", default="all", choices=("all", "generate"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--straddle", default="0.025,0.032,0.040")
    ap.add_argument("--thumb-axial", default="0.0")
    ap.add_argument("--axis-k", default="0.15,0.25,0.35,0.5")
    ap.add_argument("--modes", default="ik", help="ik,linear")
    ap.add_argument("--depth", default="auto,0.058,0.061,0.064",
                    help="comma list of grip depths in metres; `auto` = the fitter's deepest "
                         "reachable palm, which is the choice that spends the whole budget")
    ap.add_argument("--squeeze", default="0.004",
                    help="comma list, metres the pad targets are pulled INSIDE the shaft")
    ap.add_argument("--lift", type=float, default=0.10)
    ap.add_argument("--turn-steps", type=int, default=550)
    ap.add_argument("--hold-steps", type=int, default=800,
                    help="1.6 s. At 400 the grip's slow release after the turn is invisible.")
    ap.add_argument("--angle-deg", default="-90",
                    help="comma list of commanded turn angles in degrees; negative is the pole "
                         "target_axis_alignment pays for. Over-rotating past -90 drives the pads "
                         "back DOWN the far side of the shaft instead of parking them at the top "
                         "where the grip self-extinguishes.")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--jitter", type=float, default=0.0005)
    ap.add_argument("--budget", type=float, default=0.5)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.straddles = [float(v) for v in args.straddle.split(",")]
    args.thumb_axials = [float(v) for v in args.thumb_axial.split(",")]
    args.axis_ks = [float(v) for v in args.axis_k.split(",")]
    args.depths = [None if v.strip() == "auto" else float(v) for v in args.depth.split(",")]
    args.squeezes = [float(v) for v in args.squeeze.split(",")]
    args.angles = [float(v) for v in str(args.angle_deg).split(",")]
    args.modes = [v for v in args.modes.split(",") if v]

    designs = design_set(args.set)
    if args.only:
        keep = set(args.only.split(","))
        designs = {k: v for k, v in designs.items() if k in keep}

    if args.stage == "generate":
        for tag, vec in designs.items():
            print(f"{tag:16} {scene_for(vec).name}")
        return 0

    items = [(t, v) for i, (t, v) in enumerate(sorted(designs.items()))
             if i % args.shards == args.shard]
    rows = []
    print(f"{'design':16} {'Xsep':>6} {'Ysep':>6} {'Ty':>5} {'ext':>6} {'ceil':>6} "
          f"{'tau':>7} {'swp':>6} {'dep':>5} {'mean':>6} {'sd':>6} {'z':>7} "
          f"{'con':>4}  ok")
    for tag, vec in items:
        try:
            row = evaluate(tag, vec, args)
        except Exception as exc:                       # a design that will not compile is data
            print(f"{tag:16} FAILED {type(exc).__name__}: {exc}")
            rows.append({"design": tag, "error": f"{type(exc).__name__}: {exc}"})
            continue
        rows.append(row)
        b = best_cell(row)
        if b is None:
            print(f"{tag:16} {row['x_sep_mm']:6.0f} {row['y_sep_mm']:6.0f} "
                  f"{row['thumb_y_mm']:5.0f}   -- no grasp --")
        else:
            s = b["scores"]
            print(f"{tag:16} {row['x_sep_mm']:6.0f} {row['y_sep_mm']:6.0f} "
                  f"{row['thumb_y_mm']:5.0f} {s['extend_mm']:6.1f} {s['ceiling_deg']:6.1f} "
                  f"{s['tau_cap_Nmm']:7.1f} {s['sweep_min_mm']:6.1f} "
                  f"{s['depth_fit_mm']:5.0f} {b['mean_cos']:6.3f} "
                  f"{b['sd_cos']:6.3f} {b['final_z']:7.4f} {b['contacts']:4d}  "
                  f"k={b['axis_k']:.2f} a={b['angle_deg']:.0f} "
                  f"{b['kept']}/{b['n']}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
