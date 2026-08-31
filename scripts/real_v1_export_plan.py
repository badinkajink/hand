#!/usr/bin/env python3
"""Turn a planned carry into something a bench can run: mount positions and a servo trajectory.

    uv run python scripts/real_v1_export_plan.py --design g03 --out docs/experiments/.../deploy

WHAT COMES OUT
    <design>_build.txt   where to put the three finger mounts, and the grasp the plan assumes
    <design>_traj.csv    t_s, then the nine finger joints in DEGREES, then palm_z_mm
    <design>_poses.txt   the same trajectory as the four set-points it actually is
    <design>_plan.json   the machine-readable form, for manta_hand.plan.HandPlan -- mounts plus
                         the set-points INCLUDING the open pose the sheets above leave implicit

The trajectory is open loop by construction -- it is a list of joint angles against time, and
nothing in it depends on knowing where the object is once the hand has closed. That is the whole
reason it is the thing that can go on hardware first.

SIGN AND UNITS. Angles are the MuJoCo joint values in degrees, in the scene's own convention
(`assets/mjcf/real_v1/real_hand.xml`): +mcp/+pip curl the finger toward the palm, yaw is the
roll about the finger's own mount axis. A servo whose zero or direction differs needs the
mapping applied here, once, not in the middle of the schedule -- and the envelope sweep says a
1 degree per-joint zero error already costs most of the success rate, so the mapping is worth
measuring against a physical pose rather than assumed from the CAD.
"""
from __future__ import annotations

import argparse, pathlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import mujoco  # noqa: E402

import real_v1_design_search as ds  # noqa: E402
from morphohand.tools.keyframe_ik import FINGERS  # noqa: E402
from real_v1_deploy_envelope import (  # noqa: E402
    OBJECTS, TABLE, make_plan, object_scene,
)

JOINTS = [j for js in FINGERS.values() for j in js]
CTRL_HZ = 50.0            # the repo's control rate: 10 sim steps at dt = 0.002 s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", required=True)
    ap.add_argument("--object", default="medium", choices=tuple(OBJECTS))
    ap.add_argument("--axis-k", type=float, default=None, help="override the table's pivot height")
    ap.add_argument("--angle-deg", type=float, default=None)
    ap.add_argument("--straddle-mm", type=float, default=None)
    ap.add_argument("--thumb-axial-mm", type=float, default=None)
    ap.add_argument("--squeeze-mm", type=float, default=10.0,
                    help="how far inside the tool's surface the pads are driven. 10, not the "
                         "fitter's 4: the sphere-based pad model leaves a flat face 2-5 mm short, "
                         "and paying it back takes the working region from 3% of cells to 15%.")
    ap.add_argument("--budget", type=float, default=0.5,
                    help="per-joint |delta| cap in RADIANS.  0.5 rad = 28.648 deg, which is "
                         "Policy B's residual action budget and has no business constraining "
                         "an OPEN-LOOP plan: it saturated middle_yaw and middle_pip on every "
                         "exported design and 5 of 9 joints on g24, against a hardware yaw "
                         "range of +-85 deg.  Raising it widens the commanded turn but also "
                         "the swept volume -- re-run real_v1_trajectory_clearance.py after.")
    ap.add_argument("--turn-steps", type=int, default=550)
    ap.add_argument("--hold-squeeze-mm", type=float, default=0.0)
    ap.add_argument("--lift", type=float, default=0.10)
    ap.add_argument("--bench-height", type=float, default=0.0,
                    help="the prototype bench: fixed palm, tool already at this height on a "
                         "platform, so the emitted trajectory has no palm ramp. METRES, but a "
                         "value above 1 is read as MILLIMETRES and said so -- every recorded "
                         "invocation in this repo passes 100, and taking that literally is what "
                         "wrote bench_height_mm 100000 into the g12w08/g12w11 plans. Nothing "
                         "downstream reads the number (the flag is used as a boolean and for "
                         "meta), so the bad value was invisible until two plans were diffed.")
    ap.add_argument("--post-y", type=float, default=-35.0, help="mm, where the support sits")
    ap.add_argument("--pad-width-mm", type=float, default=21.1,
                    help="flat pad width across the finger. 14.8 is the built part; 21.1 is a "
                         "free reprint and measured +0.021 +- 0.008 held cos better.")
    ap.add_argument("--flat-pads", action="store_true",
                    help="the BUILT finger cross-section rather than the shipped round capsules")
    ap.add_argument("--scene", default=None,
                    help="use an EXISTING scene xml instead of rebuilding one from the pad/bench "
                         "flags.  The shipped g12 plan was exported against a prebuilt "
                         "deploy_envelope scene; rebuilding it from flags produces a different "
                         "(and unstable) model, so a budget sweep has to reuse the same file.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--design-table", type=Path, default=TABLE,
                    help="where the operating point (straddle, thumb axial, depth, axis_k, "
                         "angle) is read from. Defaults to the 108-hand 20260828 search. A "
                         "later sampler writes its own table -- the sobol128 pilot's is "
                         "docs/experiments/20260830-real_v1-sobol128/pilot_table.json -- and "
                         "its designs do not exist in the default one.")
    args = ap.parse_args()
    if args.bench_height > 1.0:
        print(f"--bench-height {args.bench_height:g} read as {args.bench_height:g} mm "
              f"(> 1 m is not a bench); pass {args.bench_height / 1000:g} for metres.")
        args.bench_height /= 1000.0

    table = json.loads(args.design_table.read_text())
    if args.design not in {r["design"] for r in table}:
        print(f"{args.design}: not one of the {len(table)} designs in {args.design_table}. "
              f"A design from another sampler needs that sampler's table via --design-table.")
        return 1
    row = {r["design"]: r for r in table}[args.design]
    scene = pathlib.Path(args.scene) if args.scene else object_scene(
        ds.scene_for(ds.design_set("all")[args.design]), args.object,
                         args.bench_height, args.post_y / 1000.0, args.flat_pads,
                         pad_width=args.pad_width_mm / 1000.0)
    if args.bench_height > 0 and row.get("depth_req_mm") is None and row.get("depth_fit_mm"):
        row = dict(row, depth_req_mm=row["depth_fit_mm"])
    st = (args.straddle_mm or row["straddle_mm"]) / 1000.0
    ta = (row["thumb_axial_mm"] if args.thumb_axial_mm is None else args.thumb_axial_mm) / 1000.0
    k = row["axis_k"] if args.axis_k is None else args.axis_k
    ang = row["angle_deg"] if args.angle_deg is None else args.angle_deg
    plan = make_plan(scene, straddle=st,
                     depth=None if row["depth_req_mm"] is None else row["depth_req_mm"] / 1000.0,
                     thumb_axial=ta, squeeze=args.squeeze_mm / 1000.0, axis_k=k, angle_deg=ang, lift=args.lift,
                     budget=args.budget, turn_steps=args.turn_steps,
                     hold_squeeze=args.hold_squeeze_mm / 1000.0,
                     bench=args.bench_height > 0)
    if plan is None:
        print(f"{args.design}: no reachable pose at straddle {st*1000:.0f} mm")
        return 1

    m = mujoco.MjModel.from_xml_path(str(scene))
    args.out.mkdir(parents=True, exist_ok=True)

    # ---- what to build ----------------------------------------------------------------------
    lines = [f"design         {args.design}",
             f"object         {args.object}",
             f"scene          {scene}",
             "",
             "FINGER MOUNTS (palm frame, mm) -- set the three gantry blocks to these",
             f"  {'finger':8} {'x':>8} {'y':>8}"]
    for f in FINGERS:
        pos = m.body(f"{f}_mount").pos
        lines.append(f"  {f:8} {pos[0]*1000:8.1f} {pos[1]*1000:8.1f}")
    lines += ["",
              "GRASP the plan assumes",
              f"  straddle (each pair pad, from the tool's mid-length)  {st*1000:6.1f} mm",
              f"  thumb slid along the tool                             {ta*1000:6.1f} mm",
              f"  grip depth below the mounting plane                   {plan['grip_depth_mm']:6.1f} mm",
              f"  squeeze (pads driven inside the surface)              {args.squeeze_mm:6.1f} mm",
              "",
              "TURN",
              f"  pivot height   axis_k {k:.3f}  (that many x the half-straddle above the pads)",
              f"  commanded turn {ang:+.0f} deg about the pinch axis",
              f"  over           {args.turn_steps / (CTRL_HZ * 10):.1f} s",
              f"  re-squeeze     {args.hold_squeeze_mm:.1f} mm at the top",
              (f"  palm lift      {args.lift*1000:.0f} mm before the turn"
               if args.bench_height <= 0 else
               f"  palm           FIXED; tool starts at {args.bench_height*1000:.0f} mm on a "
               f"support at y {args.post_y:+.0f} mm")]
    (args.out / f"{args.design}_build.txt").write_text("\n".join(lines) + "\n")

    # ---- the trajectory ---------------------------------------------------------------------
    rows, t = [], 0.0
    dt = 1.0 / CTRL_HZ
    anchor = plan["anchor"]

    def emit(vals, pz_mm):
        nonlocal t
        rows.append([round(t, 3)] + [round(float(np.degrees(vals[j])), 3) for j in JOINTS]
                    + [round(pz_mm, 2)])
        t += dt

    for _ in range(25):                       # close, 0.5 s
        emit(anchor, 0.0)
    z_hold = 0.0 if args.bench_height > 0 else args.lift * 1000
    if args.bench_height > 0:
        for _ in range(40):                   # fixed palm: settle in place, no lift
            emit(anchor, 0.0)
    else:
        for i in range(20):                   # palm lift, 0.4 s
            emit(anchor, args.lift * 1000 * (i + 1) / 20)
        for _ in range(20):                   # settle
            emit(anchor, args.lift * 1000)
    n_turn = max(1, args.turn_steps // 10)
    for i in range(1, n_turn + 1):
        u = i / n_turn
        emit({j: anchor[j] + float(np.clip(plan["delta"][j] * u, -args.budget, args.budget)) for j in JOINTS},
             z_hold)
    if plan.get("squeeze_delta"):
        start = {j: anchor[j] + float(np.clip(plan["delta"][j], -args.budget, args.budget)) for j in JOINTS}
        n_sq = max(1, int(plan["squeeze_steps"]) // 10)
        for i in range(1, n_sq + 1):
            u = i / n_sq
            emit({j: start[j] + (anchor[j] + plan["squeeze_delta"][j] - start[j]) * u
                  for j in JOINTS}, z_hold)
    for _ in range(80):                       # hold 1.6 s
        rows.append([round(t, 3)] + rows[-1][1:])
        t += dt
    csv = args.out / f"{args.design}_traj.csv"
    csv.write_text("t_s," + ",".join(f"{j}_deg" for j in JOINTS) + ",palm_z_mm\n"
                   + "\n".join(",".join(str(v) for v in r) for r in rows) + "\n")

    # ---- the same thing as the four set-points it really is ---------------------------------
    poses = ["The trajectory is four set-points and three ramps; the CSV is only that, sampled.",
             "", f"  {'joint':12} {'open/grip':>10} {'end of turn':>12} {'re-squeeze':>11}"]
    for j in JOINTS:
        a = np.degrees(anchor[j])
        e = np.degrees(anchor[j] + float(np.clip(plan["delta"][j], -args.budget, args.budget)))
        q = (np.degrees(anchor[j] + plan["squeeze_delta"][j])
             if plan.get("squeeze_delta") else float("nan"))
        poses.append(f"  {j:12} {a:10.2f} {e:12.2f} {q:11.2f}")
    poses += ["", "  ramp 1  close to the grip pose            0.5 s",
              ("  ramp 2  settle, palm held still         0.8 s" if args.bench_height > 0
               else f"  ramp 2  palm up {args.lift*1000:.0f} mm                     0.4 s"),
              f"  ramp 3  grip pose -> end of turn         {args.turn_steps/500:.1f} s",
              f"  ramp 4  end of turn -> re-squeeze        "
              f"{(plan['squeeze_steps']/500 if plan.get('squeeze_delta') else 0):.1f} s"]
    (args.out / f"{args.design}_poses.txt").write_text("\n".join(poses) + "\n")

    # ---- the same thing again, for the driver ------------------------------------------------
    # The .txt sheets are for a person; this is for manta_hand.plan.HandPlan, which turns it into
    # MOVEMM/servo commands. It carries the OPEN pose the .txt sheets leave out: the CSV starts
    # already at the grip set-point (the close is `ramp 1`, implied), and a servo told to jump
    # there from wherever it happens to be sitting closes onto the tool at full speed.
    def _split(j):
        f, _, sj = j.rpartition("_")
        return f, sj

    open_deg = {j: float(np.degrees(np.asarray(plan["open_qpos"])[m.joint(j).qposadr[0]]))
                for j in JOINTS}

    def _pose(name, ramp_s, hold_s, vals):
        out = {"name": name, "ramp_s": round(ramp_s, 3), "hold_s": round(hold_s, 3), "joints": {}}
        for j in JOINTS:
            f, sj = _split(j)
            out["joints"].setdefault(f, {})[sj] = round(float(vals[j]), 3)
        return out

    grip_deg = {j: float(np.degrees(anchor[j])) for j in JOINTS}
    end_deg = {j: float(np.degrees(anchor[j] + float(np.clip(plan["delta"][j], -args.budget, args.budget))))
               for j in JOINTS}
    settle_s = 0.8 if args.bench_height > 0 else 0.8   # lift = 0.4 ramp + 0.4 settle
    turn_s = args.turn_steps / 500.0
    sq_s = plan["squeeze_steps"] / 500.0 if plan.get("squeeze_delta") else 0.0
    jplan = {
        "design": args.design,
        "mounts_palm_mm": {f: [round(float(m.body(f"{f}_mount").pos[0]) * 1000, 2),
                               round(float(m.body(f"{f}_mount").pos[1]) * 1000, 2)]
                           for f in FINGERS},
        "poses": [
            _pose("open", 0.0, 0.0, open_deg),
            _pose("grip", 0.5, settle_s, grip_deg),
            _pose("turn_end", turn_s, 0.0 if sq_s else 1.6, end_deg),
        ],
        "meta": {"object": args.object, "scene": str(scene),
                 "design_table": str(args.design_table),
                 "source": "scripts/real_v1_export_plan.py",
                 # The nine hardware joints are enough to command the hand, but not enough to
                 # reconstruct the fitted MuJoCo initial state: the scene also has the object's
                 # free joint and six virtual palm joints. Preserve both here so a hardware-log
                 # replay cannot silently start from an unrelated scene keyframe.
                 "replay_initial_qpos": [float(v) for v in plan["open_qpos"]],
                 "replay_base_ctrl": [float(v) for v in plan["grip_ctrl"]],
                 "straddle_mm": st * 1000, "thumb_axial_mm": ta * 1000,
                 "squeeze_mm": args.squeeze_mm, "grip_depth_mm": plan["grip_depth_mm"],
                 "axis_k": k, "angle_deg": ang, "turn_steps": args.turn_steps,
                 "pad_width_mm": args.pad_width_mm, "flat_pads": bool(args.flat_pads),
                 "bench_height_mm": args.bench_height * 1000, "post_y_mm": args.post_y,
                 "palm_lift_mm": 0.0 if args.bench_height > 0 else args.lift * 1000},
    }
    if sq_s:
        sq_deg = {j: float(np.degrees(anchor[j] + plan["squeeze_delta"][j])) for j in JOINTS}
        jplan["poses"].append(_pose("resqueeze", sq_s, 1.6, sq_deg))
    (args.out / f"{args.design}_plan.json").write_text(json.dumps(jplan, indent=2) + "\n")

    print("\n".join(lines))
    print(f"\nwrote {csv} ({len(rows)} rows at {CTRL_HZ:.0f} Hz), the build/poses sheets, "
          f"and {args.design}_plan.json for manta_hand.plan")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
