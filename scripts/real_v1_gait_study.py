"""The ground-supported gaiting study: six sweeps over `probe_real_v1_gait.gait`, one JSON.

Each arm answers one question and carries its own control, because every cell is ~1.3 s of CPU
and there is therefore never a compute reason to run a sweep at one setting
(docs/experiments/20260830-real_v1-sobol128 cost this program a whole ranking that way).

  premise   ground vs no-ground, every design            is the floor doing the work?
  press     how hard the grip pushes the tool down       "some contact, but not too much"
  stroke    degrees of azimuth per cycle                 where is the kinematic ceiling?
  schedule  all-release vs relay, release duration       does it need the shaft to stand alone?
  squeeze   commanded interference = grip force          how little grip does this need?
  endurance 40 cycles at the best cell                   does it accumulate, or drift out?

    uv run python scripts/real_v1_gait_study.py --out docs/experiments/20260902-real_v1_gait
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# Defaults established by the reachability scan of 2026-09-02: grip_depth 0.050 and a ring centre
# 4 mm +x of the palm origin solve every GRIP target to 0.1-0.7 mm, where the palm-centred ring at
# the vertical-hold probe's 0.0615 depth left 1.8-11.0 mm. The mounts are not symmetric about the
# palm origin and the ring must not be either.
BASE = dict(squeeze=0.002, release_mm=6.0, grip_z=0.075, grip_depth=0.050,
            centre_x=0.004, obj="screwdriver_medium", twist_steps=120, move_steps=60)


def _cell(kw):
    import probe_real_v1_gait as G
    tag = kw.pop("_tag")
    design = kw.pop("_design")
    scene = Path(kw.pop("_scene"))
    try:
        r = G.gait(scene, **kw)
    except Exception as exc:                      # a cell that raises is data, not a stop
        return {"arm": tag, "design": design, "error": repr(exc), "ok": False, **{
            k: v for k, v in kw.items() if isinstance(v, (int, float, bool, str))}}
    r["arm"] = tag
    r["design"] = design
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--designs", type=Path, default=ROOT / "results/phase1/real_v1")
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--cycles", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--arms",
                    default="premise,press,stroke,schedule,squeeze,padr,endurance")
    args = ap.parse_args()

    designs = sorted(p for p in args.designs.iterdir()
                     if (p / "frozen_scene.xml").exists())
    if not designs:
        ap.error(f"no design dirs with frozen_scene.xml under {args.designs}")
    lead = next((p for p in designs if p.name.startswith("rv05_manual")), designs[0])
    print(f"{len(designs)} designs, lead = {lead.name}")
    arms = set(args.arms.split(","))

    jobs: list = []

    def add(tag, design: Path, reps: int, **kw):
        cycles = kw.pop("cycles", args.cycles)
        for rep in range(reps):
            jobs.append({"_tag": tag, "_design": design.name,
                         "_scene": str(design / "frozen_scene.xml"),
                         "seed": rep, "jitter": 0.0005 if reps > 1 else 0.0,
                         "cycles": cycles, **BASE, **kw})

    if "premise" in arms:
        for dsg in designs:
            add("premise_ground", dsg, args.reps, press_mm=0.0, stroke_deg=30.0)
            add("premise_air", dsg, args.reps, press_mm=0.0, stroke_deg=30.0, no_floor=True)
    if "press" in arms:
        for pr in (-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0):
            add("press", lead, args.reps, press_mm=pr, stroke_deg=30.0)
    if "stroke" in arms:
        for st in (10.0, 20.0, 30.0, 45.0, 60.0, 75.0):
            add("stroke", lead, args.reps, press_mm=0.0, stroke_deg=st)
    if "schedule" in arms:
        for relay in (False, True):
            for mv in (30, 60, 120):
                add("schedule", lead, args.reps, press_mm=0.0, stroke_deg=30.0,
                    relay=relay, move_steps=mv)
    if "squeeze" in arms:
        for sq in (0.0005, 0.001, 0.002, 0.003, 0.004, 0.006):
            add("squeeze", lead, args.reps, press_mm=0.0, stroke_deg=30.0, squeeze=sq)
    if "padr" in arms:
        # The pad radius IS the gear ratio, so this arm is a fingertip co-design sweep that the
        # gaiting task can actually resolve -- unlike the reorient, where shape set absolute
        # grip and the hold-per-turn ratio was invariant across every shape tried.
        for pr in (0.005, 0.0075, 0.01055, 0.013, 0.016, 0.020):
            add("padr", lead, args.reps, press_mm=0.0, stroke_deg=30.0, pad_radius=pr)
    if "endurance" in arms:
        for dsg in designs:
            add("endurance", dsg, 3, press_mm=0.0, stroke_deg=30.0, cycles=40)

    print(f"{len(jobs)} cells on {args.workers} workers")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(_cell, jobs, chunksize=2))

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "gait_study.json").write_text(json.dumps(rows, indent=2))
    print(f"wrote {args.out/'gait_study.json'}  ({len(rows)} cells, "
          f"{sum(1 for r in rows if r.get('error'))} errored)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
