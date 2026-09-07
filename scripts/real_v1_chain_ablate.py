#!/usr/bin/env python3
"""Why the chain that worked on 2026-09-03 does not work on the deployed plans.

ONE configuration has ever run this task end to end: `rv05_manual_stored` on 2026-09-03,
169/169, reproduced 2026-09-06 -- fingers turn the shaft from horizontal to 4.8 deg off
vertical IN THE AIR on three pads, tip down, then set down and gaited 8/8. Every chain run
since has gone through `real_v1_chain_hands.py` on the deployed plans and drops the tool.

Four things differ, and none of them was ever varied alone:

    tips     SPHERE r 10.55 mm            vs BOX 5.275 x 10.55 x 7.5 (set_finger_flat_pads)
    grasp    stored CEM best_finger_ctrl  vs _grip_from_fit, the bench fitter
    axis_k   0.25                         vs the plans' 0.05-0.15
    turn     angle -90 deg, clip 0.50     vs the plans' -60 deg, clip 0.85

The arm is NOT one of them: `docs/experiments/20260904-real_v1_bench` runs the same
`rv05_manual_stored` on a UR5e and delivers the turn at 8.03 deg tilt, so the gantry/arm axis
is already known to be clean.

Scored on the SIGNED cosine at `reoriented` -- the tool's own +z against world +z, so tip down
is +1 and handle down is -1. `tilt_deg` is folded to arccos|cos| whenever `tip_len` is 0 and
cannot tell the two apart; 111 of 123 "stands" in the 2026-09-06 table-stand sweeps were
handle down and scored as perfect.

    uv run --extra rl --extra arm python scripts/real_v1_chain_ablate.py --out <dir>
"""
from __future__ import annotations

import argparse, json, sys, xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

RV05 = ROOT / "results/phase1/real_v1/rv05_manual_stored"
OUTS = ROOT / "assets/mjcf/experimental/20260906-ablate"
OBJ = "screwdriver_medium"
PAD = dict(pad_len=0.015, width=0.0211, links=False)
# rv05_manual_b85_plan.json, the deployed operating point for this same hand.
PLAN = dict(straddle=0.040, thumb_axial=0.010, depth=0.066, squeeze=0.002,
            axis_k=0.05, angle_deg=-60.0, budget=0.85)
# The 2026-09-03 cell, reproduced 2026-09-06 as `ok` with cos +0.9965 at `reoriented`.
BASE = dict(obj=OBJ, lift=0.10, angle_deg=-90.0, axis_k=0.25, turn_steps=550, budget=0.5,
            hold_steps=500, gap=0.002, press_mm=2.0, grip_depth=0.050, squeeze=0.002,
            reindex="full", stand_order="ground", airgrip="cradle", cycles=8)


def flat_scene() -> Path:
    """The stored scene with its spherical pads replaced by the chain's rectangular ones."""
    from morphohand.studies.scene_mutate import Scene
    OUTS.mkdir(parents=True, exist_ok=True)
    p = OUTS / "rv05_flatpads.xml"
    if not p.exists():
        sc = Scene(RV05 / "frozen_scene.xml")
        sc.set_finger_flat_pads(**PAD)
        sc.write(p)
    return p


def fitted(scene: Path, tag: str, squeeze: float | None = None,
           depth: float | None = None) -> tuple[Path, dict] | None:
    """`_grip_from_fit` on the same scene, with its open pose written into `open_ik`.

    The fit's open pose has to land in the keyframe as well as the anchor: the chain resets
    from `open_ik` and takes the finger anchor separately, so handing it a fitted anchor over
    the CEM run's open pose would be a third grasp, neither of the two being compared.
    """
    import numpy as np
    import probe_real_v1_carry as pc
    from morphohand.tools.keyframe_ik import FINGERS
    OUTS.mkdir(parents=True, exist_ok=True)
    out = OUTS / f"{tag}.xml"
    meta = OUTS / f"{tag}.json"
    if out.exists() and meta.exists():
        return out, json.loads(meta.read_text())
    built = pc._grip_from_fit(scene, PLAN["straddle"], 0.0,
                              PLAN["squeeze"] if squeeze is None else squeeze, OBJ,
                              PLAN["depth"] if depth is None else depth,
                              PLAN["thumb_axial"])
    if built is None:
        return None
    m, open_qpos, grip, _ = built
    acts = {j: next(k for k in range(m.nu) if m.actuator(k).name == f"a_{j}")
            for js in FINGERS.values() for j in js}
    root = ET.parse(scene).getroot()
    for k in root.findall("keyframe"):
        root.remove(k)
    kf = ET.SubElement(root, "keyframe")
    q = " ".join(f"{v:.9g}" for v in open_qpos)
    ET.SubElement(kf, "key", {"name": "open_ik", "qpos": q,
                              "ctrl": " ".join(f"{v:.9g}" for v in grip)})
    ET.SubElement(kf, "key", {"name": "open", "qpos": q})
    ET.ElementTree(root).write(out, encoding="unicode")
    anchor = {j: float(grip[a]) for j, a in acts.items()}
    meta.write_text(json.dumps(anchor))
    return out, anchor


def _cell(kw):
    import probe_real_v1_chain as C
    arm, seed = kw.pop("_arm"), kw["seed"]
    scene = kw.pop("_scene", None)
    anchor = kw.pop("_anchor", None)
    cell = dict(BASE)
    cell.update(kw)
    try:
        r = C.chain(RV05, scene_path=scene, anchor_ctrl=anchor, **cell)
    except Exception as exc:
        return {"arm": arm, "seed": seed, "error": repr(exc), "ok": False}
    s = {x["phase"]: x for x in r["seams"]}
    ro = s.get("reoriented", {})
    up = s.get("upright", s.get("staged", {}))
    r["seams"] = [{k: v for k, v in x.items()
                   if k in ("phase", "t", "cos", "tilt_deg", "z", "pad_contacts",
                            "pad_force_N")} for x in r["seams"]]
    r.pop("cycles", None)
    r["arm"] = arm
    # THE ONLY HONEST SCORE. +1 is tip down, -1 is handle down, and the folded `tilt_deg`
    # reports both as 0.00.
    r["cos_reoriented"] = ro.get("cos")
    r["pads_reoriented"] = ro.get("pad_contacts")
    r["force_reoriented_N"] = ro.get("pad_force_N")
    r["cos_upright"] = up.get("cos")
    r["reorient_ok"] = bool((ro.get("cos") or -1.0) > 0.90
                            and (ro.get("pad_contacts") or 0) >= 2
                            and (ro.get("z") or 0.0) > 0.08)
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--depths", default=None,
                    help="comma list of fitted-grasp grip depths in mm. THE FITTER MAXIMISES "
                         "THIS BY CONSTRUCTION and its own docstring says so: it takes the "
                         "DEEPEST reachable palm height, which lands the fingers at 96-99%% of "
                         "the 68.11 mm mount-to-pad chain with 0.44-1.26 mm of headroom. The "
                         "turn is paid for in EXTENSION, and the reference CEM grasp sits at "
                         "92-96%% with 2.74 mm. A lower palm trades tip clearance for the "
                         "travel the turn needs.")
    ap.add_argument("--squeezes", default=None,
                    help="comma list of fitted-grasp squeezes in mm. THE FITTER HAS NO FORCE "
                         "TARGET: `_grip_from_fit` places the pad CENTRES geometrically at "
                         "r_obj + r_pad + gap - squeeze and never asks what force that "
                         "produces, so squeeze is the only knob between a grip that holds a "
                         "static tool (0.24 N, its weight) and one that holds it through a "
                         "turn (the CEM grasp's 10.47 N). Runs on the REFERENCE hand with "
                         "sphere pads at reference axis_k and clip, so the fitter is the only "
                         "thing that differs.")
    args = ap.parse_args()

    sph = RV05 / "frozen_scene.xml"
    box = flat_scene()
    f_sph = fitted(sph, "rv05_fit_sphere")
    f_box = fitted(box, "rv05_fit_box")

    # One factor at a time off the reproduced baseline, then all four together.
    arms: list[tuple[str, dict]] = [
        ("baseline", {}),
        ("tips=box", {"_scene": box}),
        ("grasp=fit", {"_scene": f_sph[0], "_anchor": f_sph[1]} if f_sph else None),
        ("axis_k=0.05", {"axis_k": PLAN["axis_k"]}),
        ("angle=-60", {"angle_deg": PLAN["angle_deg"]}),
        ("clip=0.85", {"budget": PLAN["budget"]}),
        ("turn=plan", {"angle_deg": PLAN["angle_deg"], "budget": PLAN["budget"]}),
        ("all=deployed", ({"_scene": f_box[0], "_anchor": f_box[1], "axis_k": PLAN["axis_k"],
                           "angle_deg": PLAN["angle_deg"], "budget": PLAN["budget"]}
                          if f_box else None)),
    ]
    if args.depths:
        arms = [("baseline", {})]
        for dp in (float(v) for v in args.depths.split(",")):
            f = fitted(sph, f"rv05_fit_sphere_d{dp:g}", None, dp / 1000.0)
            arms.append((f"fit_d{dp:g}mm",
                         {"_scene": f[0], "_anchor": f[1]} if f else None))
    elif args.squeezes:
        arms = [("baseline", {})]
        for sq in (float(v) for v in args.squeezes.split(",")):
            f = fitted(sph, f"rv05_fit_sphere_sq{sq:g}", sq / 1000.0)
            arms.append((f"fit_sq{sq:g}mm",
                         {"_scene": f[0], "_anchor": f[1]} if f else None))
    jobs = []
    for name, kw in arms:
        if kw is None:
            print(f"  {name}: NO FIT", flush=True)
            continue
        for rep in range(args.reps):
            jobs.append({"_arm": name, "seed": rep, "jitter": 0.0005, **kw})
    print(f"{len(jobs)} cells on {args.workers} workers", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(_cell, jobs, chunksize=1)):
            rows.append(r)
            if (i + 1) % 8 == 0:
                print(f"  {i + 1}/{len(jobs)}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "ablate.json").write_text(json.dumps({"rows": rows, "base": BASE,
                                                      "plan": PLAN}, separators=(",", ":")))
    import statistics as st
    print(f"\n   {'arm':16} {'reorient':>9} {'cos_reo':>8} {'pads':>5} {'F_N':>7} "
          f"{'cos_up':>7} {'chain':>6} {'cyc':>5}")
    for name, _ in arms:
        g = [r for r in rows if r.get("arm") == name]
        if not g:
            continue
        f = lambda k: [x for x in (r.get(k) for r in g) if x is not None]
        m = lambda v: st.mean(v) if v else float("nan")
        print(f"   {name:16} {sum(1 for r in g if r.get('reorient_ok')):2}/{len(g):<2}     "
              f"{m(f('cos_reoriented')):+8.4f} {m(f('pads_reoriented')):5.1f} "
              f"{m(f('force_reoriented_N')):7.2f} {m(f('cos_upright')):+7.4f} "
              f"{sum(1 for r in g if r.get('ok')):2}/{len(g):<2} "
              f"{m(f('cycles_run')):5.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
