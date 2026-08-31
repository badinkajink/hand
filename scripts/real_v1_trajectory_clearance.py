#!/usr/bin/env python3
"""Finger-to-finger clearance along a real_v1 deploy trajectory.

`morph_selfcollision_gate.py` answers a different question: does a mount rail
run through the palm, at one static pose.  Nothing checked whether the FINGERS
sweep through each other while the turn is executing, and on 2026-08-29 the
bench run drove index and middle into contact -- visible to the operator, and
reproducible in the sim's own geometry.

Two paths are checked because the exported plan has two of them and they are
not the same path:

  chord  the three set-points in <design>_plan.json, linearly interpolated --
         this is what `plan.run_trajectory` on the control station replays
  csv    <design>_traj.csv, the per-step schedule the carry actually produced,
         where per-joint timing differs (in g12 middle finishes its roll by
         u~0.4 while index tracks the chord)

Clearance is a lower bound on the SIM's geometry, whose link capsules and tip
boxes are thinner than the printed parts, so a positive number here is
necessary and not sufficient.  A negative number means the geoms interpenetrate.

  python3 scripts/real_v1_trajectory_clearance.py --plan g12 --plan g23
  python3 scripts/real_v1_trajectory_clearance.py --all --substeps 8
"""
from __future__ import annotations
import argparse, csv, itertools, json, os, sys
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parent.parent
DEPLOY = REPO / "docs/experiments/20260829-real_v1_deploy/deploy"
FINGERS = ("thumb", "index", "middle")
JOINTS = ("yaw", "mcp", "pip")
DISTMAX = 0.25  # metres; beyond this mj_geomDistance stops caring and returns distmax


def finger_geoms(m: mujoco.MjModel) -> dict[str, list[int]]:
    """Group every geom under the body subtree of each finger's root joint."""
    out: dict[str, list[int]] = {f: [] for f in FINGERS}
    for g in range(m.ngeom):
        b = m.geom_bodyid[g]
        # walk up to a body whose name starts with a finger prefix
        while b > 0:
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
            hit = next((f for f in FINGERS if name.startswith(f)), None)
            if hit:
                out[hit].append(g)
                break
            b = m.body_parentid[b]
    return out


def qadr(m: mujoco.MjModel) -> dict[tuple[str, str], int]:
    adr = {}
    for f in FINGERS:
        for j in JOINTS:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{f}_{j}")
            if jid < 0:
                raise SystemExit(f"scene has no joint {f}_{j}")
            adr[(f, j)] = m.jnt_qposadr[jid]
    return adr


def _body_of(m: mujoco.MjModel, g: int) -> str:
    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or f"geom{g}"


def min_clearance(m, d, groups, pairs, owner) -> tuple[float, str, int]:
    """Smallest finger-to-finger distance at the current (already mj_forward'ed) pose.

    `mj_geomDistance` is not trustworthy on its own here: on box-box pairs it
    returns exactly 0.0 for configurations that are demonstrably far apart --
    g12's thumb_tip/index_tip traces a smooth 8.8 -> 20.9mm over the turn with a
    single 0.0 sample dropped in the middle of it.  So the COLLISION PIPELINE is
    the authority on whether anything is actually touching (mj_forward has filled
    d.contact), and mj_geomDistance is used only to measure the margin when it
    is not.  Exact zeros are discarded as the artifact they are.
    """
    for i in range(d.ncon):
        c = d.contact[i]
        fa, fb = owner.get(c.geom1), owner.get(c.geom2)
        if fa and fb and fa != fb:
            return float(c.dist), f"{fa}<->{fb} CONTACT", 0
    worst, who, dropped = DISTMAX, "", 0
    for fa, fb in pairs:
        for ga, gb in itertools.product(groups[fa], groups[fb]):
            dist = mujoco.mj_geomDistance(m, d, ga, gb, DISTMAX, None)
            if dist == 0.0:
                dropped += 1
                continue
            if dist < worst:
                # Name the two BODIES, not just the two fingers.  The sim's proximal
                # link is a 21.2 mm capsule on the mount axis and the printed part is
                # a servo body and bracket around it, so a margin held between two
                # PROXIMAL segments is worth less than the same margin between two
                # tips -- and only the body names say which one this is.
                worst, who = dist, f"{_body_of(m, ga)}<->{_body_of(m, gb)}"
    return worst, who, dropped


def lerp(a, b, u):
    return {f: {j: a[f][j] + (b[f][j] - a[f][j]) * u for j in JOINTS} for f in FINGERS}


def chord_path(plan, n):
    poses = {p["name"]: p["joints"] for p in plan["poses"]}
    a, b = poses["grip"], poses["turn_end"]
    return [lerp(a, b, i / n) for i in range(n + 1)]


def csv_path(plan, path: Path):
    rows = list(csv.DictReader(open(path)))
    t_grip = plan["poses"][0]["ramp_s"] + plan["poses"][1]["ramp_s"] + plan["poses"][1]["hold_s"]
    t_end = t_grip + plan["poses"][2]["ramp_s"]
    return [{f: {j: float(r[f"{f}_{j}_deg"]) for j in JOINTS} for f in FINGERS}
            for r in rows if t_grip - 1e-9 <= float(r["t_s"]) <= t_end + 1e-9]


def densify(path, substeps):
    """The servos ramp between commanded set-points; check between them too."""
    if substeps <= 1:
        return path
    out = []
    for a, b in zip(path, path[1:]):
        for k in range(substeps):
            out.append(lerp(a, b, k / substeps))
    out.append(path[-1])
    return out


def scan(design: str, substeps: int, steps: int, verbose: bool):
    plan = json.loads((DEPLOY / f"{design}_plan.json").read_text())
    m = mujoco.MjModel.from_xml_path(plan["meta"]["scene"])
    d = mujoco.MjData(m)
    groups, adr = finger_geoms(m), qadr(m)
    pairs = list(itertools.combinations(FINGERS, 2))
    owner = {gi: f for f, gs in groups.items() for gi in gs}

    results = {}
    for label, path in (("chord", chord_path(plan, steps)),
                        ("csv", csv_path(plan, DEPLOY / f"{design}_traj.csv"))):
        dense = densify(path, substeps)
        worst, who, at, dropped = DISTMAX, "", 0.0, 0
        trace = []
        for i, pose in enumerate(dense):
            for f in FINGERS:
                for j in JOINTS:
                    d.qpos[adr[(f, j)]] = np.deg2rad(pose[f][j])
            mujoco.mj_forward(m, d)
            c, w, drop = min_clearance(m, d, groups, pairs, owner)
            dropped += drop
            trace.append(round(c * 1000.0, 2))
            if c < worst:
                worst, who, at = c, w, i / (len(dense) - 1)
        results[label] = {"min_mm": worst * 1000.0, "pair": who, "at_u": at,
                          "n_setpoints": len(path), "n_checked": len(dense),
                          "artifact_zeros": dropped, "trace_mm": trace}
        if verbose:
            print(f"    {label:6s} {worst*1000:+7.1f} mm  {who:40s} at u={at:.2f}"
                  f"  ({len(path)} set-points, {len(dense)} checked"
                  + (f", {dropped} artifact zeros discarded)" if dropped else ")"))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="append", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--substeps", type=int, default=4,
                    help="interpolated checks between consecutive set-points")
    ap.add_argument("--steps", type=int, default=55,
                    help="chord resolution; 55 matches the 50Hz service over the 1.1s ramp")
    ap.add_argument("--min-mm", type=float, default=5.0,
                    help="PASS threshold; the sim's links are thinner than the printed parts")
    ap.add_argument("--deploy-dir", default=None,
                    help="scan plans from another export dir (e.g. a --budget sweep) instead\n                         of the shipped deploy folder")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    global DEPLOY
    if args.deploy_dir:
        DEPLOY = Path(args.deploy_dir)

    designs = args.plan
    if args.all or not designs:
        designs = sorted({p.name.removesuffix("_plan.json")
                          for p in DEPLOY.glob("*_plan.json")})

    out, worst_overall = {}, {}
    print(f"finger<->finger clearance, PASS >= {args.min_mm:.1f} mm on sim geometry\n")
    for design in designs:
        print(f"  {design}")
        out[design] = scan(design, args.substeps, args.steps, verbose=True)
        print()

    print(f"  {'design':10s} {'chord':>10s} {'csv':>10s}   verdict")
    for design, r in out.items():
        c, v = r["chord"]["min_mm"], r["csv"]["min_mm"]
        ok = [lbl for lbl, val in (("chord", c), ("csv", v)) if val >= args.min_mm]
        print(f"  {design:10s} {c:+9.1f}  {v:+9.1f}   "
              + (f"run on: {', '.join(ok)}" if ok else "NO SAFE PATH"))

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"\n  wrote {args.json}")
    return 0 if any(min(r["chord"]["min_mm"], r["csv"]["min_mm"]) >= args.min_mm
                    or max(r["chord"]["min_mm"], r["csv"]["min_mm"]) >= args.min_mm
                    for r in out.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
