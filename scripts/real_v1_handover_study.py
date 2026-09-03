"""Does the hand have to LET GO to change grasp? Sweeping the handover and the gait schedule.

The chained task published in docs/experiments/20260903-real_v1_chain gets from the carry's
grasp to the gait's by opening the hand completely, flying the palm somewhere else, and closing
again -- and it works, because a 100 x 25 mm cylinder standing on its end face is statically
stable to atan(12.5/50) = 14.0 deg and simply waits where it was put. That is a real property of
that object on that floor, and it is doing a great deal of load-bearing work: for 55% of the time
between the tool being set down and the gait finishing, NOTHING IS TOUCHING IT. A tool that is
seated in a countersink, or slightly heavier, or on a bench that is not perfectly level, or that
anything at all brushes against, does not offer that.

This study removes the assumption in two directions at once.

  handover  how the carry's grasp becomes the gait's, from a full release to a per-finger walk
  gait      release all three pads per cycle, or one at a time
  load      a steady lateral load on the tool, as the table tilt that would produce it
  floor     delete the floor after the grasp: is the HAND holding it, or the ground?
  track     how much of the wrist move the fingers can absorb with their contacts pinned
  screw     the same, with the tool seated in a 45 deg countersink
  endurance 40 cycles of the mode that never lets go

Every arm carries the shipped `full` + synchronous-gait chain as its control, so the cost of
continuous support is always read against the thing it replaces rather than in isolation.

    uv run --extra rl python scripts/real_v1_handover_study.py \
        --out docs/experiments/20260903-real_v1_handover
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# Identical to real_v1_chain_study.BASE: the published carry and the published gait, so the only
# thing this study changes is how the two are joined.
BASE = dict(obj="screwdriver_medium", lift=0.10, angle_deg=-90.0, axis_k=0.25, turn_steps=550,
            budget=0.5, hold_steps=500, gap=0.002, carry_squeeze=0.0,
            descend_iters=1, descend_steps=400, airgrip="cradle", stand_order="ground",
            centre_x=0.004, grip_depth=0.050, squeeze=0.002, release_mm=6.0,
            twist_steps=120, move_steps=60)

SCREW = ROOT / "docs/experiments/20260903-real_v1_screw/screw_a45_x40_y-11.json"

# An insertion is not a set-down with a different floor: it needs the 10 mm press that seats the
# cone (below ~8 mm the tip stops proud), the 0.3 mm re-grip that buys the time to cross to the
# socket, and a shorter transport. These are real_v1_screw_study.BASE verbatim.
SEAT_BASE = dict(carry_squeeze=0.0003, press_mm=10.0, transport_steps=300)


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
    ap.add_argument("--run", type=Path, default=ROOT / "results/phase1/real_v1/rv05_manual_stored")
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--cycles", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--arms", default="handover,gait,grip,load,floor,track,screw,dir,nonerelay,endurance")
    args = ap.parse_args()

    lead = args.run
    arms = set(args.arms.split(","))
    screw = json.loads(SCREW.read_text()) if arms & {"screw", "dir"} else None
    jobs: list = []

    def add(tag, reps: int, **kw):
        cycles = kw.pop("cycles", args.cycles)
        for rep in range(reps):
            jobs.append({"_tag": tag, "_run": str(lead), "seed": rep,
                         # The carry is chaotic at the 1e-6 level, so a handover measured at one
                         # seed measures one carry.
                         "jitter": 0.0005 if reps > 1 else 0.0,
                         "cycles": cycles, **BASE, **kw})

    def seat(**kw):
        return dict(SEAT_BASE, **kw, scene_path=Path(screw["scene"]),
                    place_xy=screw["socket_xy"], seat_z=screw["seat_z"],
                    tip_len=screw["tip_len"])

    if "handover" in arms:
        # full and none both pass through zero contacts; relay and track do not try to.
        for idx in ("full", "none", "relay", "track", "slide"):
            add("handover", args.reps, reindex=idx)
    if "gait" in arms:
        for idx in ("full", "relay"):
            for rg in (False, True):
                add("gait", args.reps, reindex=idx, relay_gait=rg)
    if "load" in arms:
        # 14.0 deg is where a free-standing cylinder topples on its own. A mode that keeps two
        # pads on the tool should not care about that number; one that lets go should.
        for tilt in (0.0, 4.0, 8.0, 12.0, 14.0, 16.0, 20.0, 30.0):
            add("load", args.reps, reindex="full", tilt_deg=tilt)
            add("load", args.reps, reindex="relay", relay_gait=True, tilt_deg=tilt)
    if "floor" in arms:
        for nf in (False, True):
            add("floor", args.reps, reindex="full", no_floor_gait=nf)
            add("floor", args.reps, reindex="relay", relay_gait=True, no_floor_gait=nf)
    if "track" in arms:
        for tf in (0.0, 0.25, 0.5, 0.75, 1.0):
            add("track", args.reps, reindex="track", track_frac=tf)
    if "screw" in arms:
        for idx in ("full", "relay"):
            add("screw", args.reps, **seat(reindex=idx, relay_gait=idx == "relay"))
        for tilt in (0.0, 8.0, 16.0, 30.0):
            add("screw_load", args.reps, **seat(reindex="full", tilt_deg=tilt))
            add("screw_load", args.reps,
                **seat(reindex="relay", relay_gait=True, tilt_deg=tilt))
    if "dir" in arms:
        # The load has been coming from +x, away from the palm, which is the direction the hand
        # is least able to resist and so the fair worst case -- but only if it IS the worst
        # case. Four azimuths at one magnitude, on the seat, says whether the result is a
        # property of the seat or of one lucky direction.
        for az in (0.0, 90.0, 180.0, 270.0):
            add("dir", args.reps, **seat(reindex="full", tilt_deg=16.0, tilt_dir=az))
            add("dir", args.reps,
                **seat(reindex="relay", relay_gait=True, tilt_deg=16.0, tilt_dir=az))
    if "nonerelay" in arms:
        # The user's other option: never take the gait's grasp at all, and gait from whatever
        # the carry left. `none` still opens all three fingers together to reach its ring, so
        # pairing it with the relay gait tests the second half of the idea on its own.
        for rg in (False, True):
            add("nonerelay", args.reps, reindex="none", relay_gait=rg)
    if "grip" in arms:
        # The relay sets its grip with one commanded squeeze. Too little and the tool rocks in
        # the hand; too much and the gait stalls, which is the same over-clamp window the gait
        # study found at 75 N and the reorient found at every value it tried.
        for q in (0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003):
            add("grip", args.reps, reindex="relay", relay_gait=True, relay_squeeze=q)
    if "endurance" in arms:
        add("endurance", 3, reindex="full", cycles=40)
        add("endurance", 3, reindex="relay", relay_gait=True, cycles=40)

    print(f"{len(jobs)} cells on {args.workers} workers")
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(_cell, jobs, chunksize=1)):
            rows.append(r)
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(jobs)}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "handover_study.json").write_text(json.dumps(rows, indent=2))
    print(f"-> {args.out / 'handover_study.json'}  "
          f"({sum(1 for r in rows if r.get('error'))} errored)")

    def key(r):
        return (r["arm"], r.get("reindex"), r.get("relay_gait"), r.get("tilt_deg"),
                r.get("no_floor_gait"), r.get("track_frac"), r.get("relay_squeeze_mm"),
                r.get("tilt_dir"))

    groups: dict = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r)
    print(f"\n{'arm':11} {'idx':6} {'rg':3} {'tilt':>5} {'nofl':4} {'trk':>4} "
          f"{'sq':>4} {'ok':>5} {'deg/cy':>7} {'free':>5} {'gnd':>5} {'slip':>6} """
          f"{'tilt':>6} {'drift':>7}")
    for k in sorted(groups, key=lambda x: [str(v) for v in x]):
        v = [r for r in groups[k] if not r.get("error")]
        if not v:
            continue
        okv = [r for r in v if r["ok"]]
        print(f"{k[0]:11} {str(k[1]):6} {str(k[2])[:3]:3} {k[3] if k[3] is not None else 0:5.0f} "
              f"{str(k[4])[:4]:4} {k[5] if k[5] is not None else 1:4.2f} "
              f"{k[6] if k[6] is not None else 0:4.1f} "
              f"{sum(r['ok'] for r in v):2d}/{len(v):<2d} "
              f"{st.mean(r['gain_mean_deg'] for r in okv) if okv else 0:7.2f} "
              f"{st.mean(r['free_frac'] for r in v if r.get('free_frac') is not None):6.3f} "
              f"{st.mean(r['ground_frac'] for r in v if r.get('ground_frac') is not None):5.2f} "
              f"{st.mean(r['slip_mm_per_cycle'] for r in v if 'slip_mm_per_cycle' in r):6.2f} "
              f"{st.mean(r['final_tilt_deg'] for r in v):6.2f} "
              f"{st.mean(r['drift_mm'] for r in v):7.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
