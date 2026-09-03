"""grasp -> lift -> reorient -> stand -> gait, swept: what does the seam actually cost?

Each arm asks one question about the join between the two halves of the task, and carries the
control that makes its answer falsifiable. A cell is ~1 s of CPU, so there is never a compute
reason to run a single setting.

  chain     every design, end to end                which hands can do the whole task?
  reindex   let go and retake vs keep the grip      is the floor's real gift the RE-GRASP?
  airgrip   change grasp in mid-air, or on the floor  can the hand re-grip what it is holding?
  hold      200..900 settle steps after the turn    what tilt does the seam actually inherit?
  repose    1..8 corrections bringing it upright    how much object feedback does the tilt need?
  descend   1..8 corrections on the way down        and how much does the HEIGHT need? (none)
  press     -6..+8 mm through the grip              the gait window, re-measured after a carry
  floor     ground vs no ground during the gait     the premise, inside the full chain
  stroke    10..60 deg of azimuth per cycle         does the chained grip change the ceiling?
  endurance 40 cycles from a real carry             does it accumulate, or walk out?

    uv run --extra rl python scripts/real_v1_chain_study.py --out docs/experiments/20260903-real_v1_chain
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

# The gait study's validated cell (docs/experiments/20260902-real_v1_gait), plus the carry's
# published raised-pivot settings (docs/experiments/20260830-carry_mass_envelope). Nothing here
# is new tuning: the point of the chain is to run both halves AS PUBLISHED and pay only the
# seam's cost.
BASE = dict(obj="screwdriver_medium", lift=0.10, angle_deg=-90.0, axis_k=0.25, turn_steps=550,
            budget=0.5, hold_steps=500, gap=0.002, carry_squeeze=0.0,
            descend_iters=1, descend_steps=400, airgrip="cradle",
            stand_order="ground",
            centre_x=0.004, grip_depth=0.050, squeeze=0.002, release_mm=6.0,
            twist_steps=120, move_steps=60)


def _cell(kw):
    import probe_real_v1_chain as C
    tag = kw.pop("_tag")
    run = Path(kw.pop("_run"))
    try:
        r = C.chain(run, **kw)
    except Exception as exc:                      # a cell that raises is data, not a stop
        return {"arm": tag, "run": run.name, "error": repr(exc), "ok": False,
                **{k: v for k, v in kw.items() if isinstance(v, (int, float, bool, str))}}
    r["arm"] = tag
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
                    default="chain,reindex,airgrip,hold,repose,descend,press,floor,stroke,endurance")
    args = ap.parse_args()

    designs = sorted(p for p in args.designs.iterdir() if (p / "frozen_scene.xml").exists())
    if not designs:
        ap.error(f"no design dirs with frozen_scene.xml under {args.designs}")
    lead = next((p for p in designs if p.name.startswith("rv05_manual")), designs[0])
    print(f"{len(designs)} designs, lead = {lead.name}")
    arms = set(args.arms.split(","))
    jobs: list = []

    def add(tag, run: Path, reps: int, **kw):
        cycles = kw.pop("cycles", args.cycles)
        for rep in range(reps):
            jobs.append({"_tag": tag, "_run": str(run), "seed": rep,
                         # The carry is chaotic at the 1e-6 level -- its exit tilt on this hand
                         # ranges 1.7-18.9 deg over spawn jitter alone -- so a chain measured at
                         # one seed measures one carry, not the seam.
                         "jitter": 0.0005 if reps > 1 else 0.0,
                         "cycles": cycles, **BASE, **kw})

    if "chain" in arms:
        for dsn in designs:
            add("chain", dsn, args.reps, reindex="full")
    if "reindex" in arms:
        for idx in ("full", "none"):
            add("reindex", lead, args.reps, reindex=idx)
    if "repose" in arms:
        for it in (1, 2, 4, 8):
            add("repose", lead, args.reps, reindex="full", repose_iters=it)
    if "descend" in arms:
        for it in (1, 2, 4, 8):
            add("descend", lead, args.reps, reindex="full", descend_iters=it,
                descend_steps=400 * it)
    if "airgrip" in arms:
        for ag in ("cradle", "ring"):
            add("airgrip", lead, args.reps, reindex="full", airgrip=ag)
    if "hold" in arms:
        for hs in (200, 300, 400, 500, 700, 900):
            add("hold", lead, args.reps, reindex="full", hold_steps=hs)
    if "press" in arms:
        for pr in (-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0):
            add("press", lead, args.reps, reindex="full", press_mm=pr)
    if "floor" in arms:
        for nf in (False, True):
            add("floor", lead, args.reps, reindex="full", no_floor_gait=nf)
    if "stroke" in arms:
        for st in (10.0, 20.0, 30.0, 45.0, 60.0):
            add("stroke", lead, args.reps, reindex="full", stroke_deg=st)
    if "endurance" in arms:
        add("endurance", lead, 3, reindex="full", cycles=40)

    print(f"{len(jobs)} cells on {args.workers} workers")
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(_cell, jobs, chunksize=1)):
            rows.append(r)
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(jobs)}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "chain_study.json").write_text(json.dumps(rows, indent=2))
    print(f"-> {args.out / 'chain_study.json'}")

    import statistics as st
    for arm in sorted({r["arm"] for r in rows}):
        sub = [r for r in rows if r["arm"] == arm]
        print(f"\n== {arm}  ({len(sub)} cells)")
        keys = sorted({(r.get("run", "?"), r.get("reindex"), r.get("repose_iters"),
                        r.get("descend_iters"), r.get("press_mm"), r.get("no_floor_gait"),
                        r.get("stroke_deg"), r.get("airgrip"), r.get("hold_steps"),
                        r.get("cycles_asked")) for r in sub})
        for k in keys:
            g = [r for r in sub
                 if (r.get("run", "?"), r.get("reindex"), r.get("repose_iters"),
                     r.get("descend_iters"), r.get("press_mm"), r.get("no_floor_gait"),
                     r.get("stroke_deg"), r.get("airgrip"), r.get("hold_steps"),
                     r.get("cycles_asked")) == k]
            ok = sum(1 for r in g if r.get("ok"))
            good = [r for r in g if r.get("ok")]
            turns = st.mean([r["turns"] for r in good]) if good else 0.0
            tilt = st.mean([r["seams"][1]["tilt_deg"] for r in g if r.get("seams")]) if g else 0.0
            fin = st.mean([r["final_tilt_deg"] for r in good]) if good else 0.0
            print(f"   {str(k):78} ok {ok}/{len(g)}  turns {turns:6.2f}  "
                  f"carry_tilt {tilt:5.1f}  final_tilt {fin:5.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
