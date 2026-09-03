"""Make the tool a screw and the floor its seat, then ask what changes.

The gaiting result stands a flat-ended rod on a flat plane. That makes the ground a pure
friction brake and leaves the tool free to walk, which is why `drift_mm` had to be reported.
A screw sits in a countersink, and the seat centres, wedges and constrains. So:

  press     how hard the tip has to be driven home       is insertion a placement or a press?
  capture   lateral error in the insertion target        how accurately must the seat be found?
  seat      countersink vs the flat plane, matched       does the gait still turn it?
  distance  how far the seat is from the pick-up         how much transport does the grip afford?
  angle     30 / 45 / 60 deg countersink                 which seat geometry is kindest?
  endurance 40 cycles in the seat                        does it stay seated?

Every arm runs the WHOLE chain -- grasp, lift, reorient, stage, insert, release, re-index, gait
-- because the seat changes the set-down as much as it changes the turn, and a gait-only
comparison would miss the insertion entirely.

    uv run --extra rl python scripts/real_v1_screw_study.py --out docs/experiments/20260903-real_v1_screw
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

RUN = ROOT / "results/phase1/real_v1/rv05_manual_stored"
# Settled by the press sweep and the staging trials of 2026-09-03. The two that are not
# arbitrary: press 10 mm, because 6 leaves the tip 1.5 mm proud and the gait then has nothing to
# push against, and transport 300 steps, because the carry's grasp is gone in about 1.6 s.
BASE = dict(obj="screwdriver_medium", lift=0.10, angle_deg=-90.0, axis_k=0.25, turn_steps=550,
            budget=0.5, hold_steps=500, gap=0.002, carry_squeeze=0.0005, press_mm=10.0,
            transport_steps=300, descend_iters=1, descend_steps=400, airgrip="cradle",
            centre_x=0.004, grip_depth=0.050, squeeze=0.002, release_mm=6.0,
            twist_steps=120, move_steps=60)


def _cell(kw):
    import probe_real_v1_chain as C
    tag = kw.pop("_tag")
    try:
        r = C.chain(RUN, **kw)
    except Exception as exc:
        return {"arm": tag, "error": repr(exc), "ok": False,
                **{k: v for k, v in kw.items() if isinstance(v, (int, float, bool, str))}}
    r["arm"] = tag
    return r


def _build(socket_xy, angle, out_dir: Path) -> dict:
    """One screw scene per (seat position, cone angle); the JSON carries every coupled number."""
    tag = f"a{angle:g}_x{socket_xy[0] * 1000:.0f}_y{socket_xy[1] * 1000:.0f}"
    scene = RUN / f"screw_{tag}.xml"
    js = out_dir / f"screw_{tag}.json"
    subprocess.run([sys.executable, str(ROOT / "scripts/build_screw_scene.py"),
                    "--socket-xy", f"{socket_xy[0]},{socket_xy[1]}",
                    "--half-angle", str(angle), "--out", str(scene), "--out-json", str(js)],
                   check=True, capture_output=True)
    return json.loads(js.read_text())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--cycles", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--arms", default="press,capture,seat,distance,angle,endurance")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    arms = set(args.arms.split(","))

    seat = _build((0.04, -0.011), 45.0, args.out)
    SEAT = dict(scene_path=seat["scene"], place_xy=seat["socket_xy"],
                seat_z=seat["seat_z"], tip_len=seat["tip_len"])
    jobs: list = []

    def add(tag, reps, **kw):
        cycles = kw.pop("cycles", args.cycles)
        for rep in range(reps):
            jobs.append({"_tag": tag, "seed": rep, "jitter": 0.0005 if reps > 1 else 0.0,
                         "cycles": cycles, **BASE, **kw})

    if "press" in arms:
        for pr in (2.0, 4.0, 6.0, 8.0, 10.0, 14.0, 18.0):
            add("press", args.reps, press_mm=pr, **SEAT)
    if "capture" in arms:
        for e in (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0):
            add("capture", args.reps, place_err=(e / 1000.0, 0.0), **SEAT)
    if "seat" in arms:
        # The control the whole arm exists for: the same tool, the same press, the same gait, on
        # a flat plane instead of in a countersink. The flat scene is the ORIGINAL frozen scene,
        # so nothing but the seat differs.
        add("seat", args.reps, **SEAT)
        add("seat", args.reps, press_mm=2.0, **SEAT)
        add("seat", args.reps, press_mm=2.0)
        add("seat", args.reps, press_mm=10.0)
    if "distance" in arms:
        for x in (0.03, 0.04, 0.06, 0.09):
            s = _build((x, -0.011), 45.0, args.out)
            add("distance", args.reps, scene_path=s["scene"], place_xy=s["socket_xy"],
                seat_z=s["seat_z"], tip_len=s["tip_len"])
    if "angle" in arms:
        for a in (30.0, 45.0, 60.0):
            s = _build((0.04, -0.011), a, args.out)
            add("angle", args.reps, scene_path=s["scene"], place_xy=s["socket_xy"],
                seat_z=s["seat_z"], tip_len=s["tip_len"])
    if "endurance" in arms:
        add("endurance", 3, cycles=40, **SEAT)

    print(f"{len(jobs)} cells on {args.workers} workers")
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(_cell, jobs, chunksize=1)):
            rows.append(r)
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(jobs)}")
    (args.out / "screw_study.json").write_text(json.dumps(rows, indent=2))
    print(f"-> {args.out / 'screw_study.json'}")

    import statistics as st
    for arm in sorted({r["arm"] for r in rows}):
        sub = [r for r in rows if r["arm"] == arm]
        print(f"\n== {arm} ({len(sub)})")
        keys = sorted({(r.get("press_mm"), tuple(r.get("place_err_mm") or ()),
                        r.get("seat_z"), tuple(r.get("place_xy") or ()),
                        r.get("tip_len_mm"), r.get("cycles_asked")) for r in sub},
                      key=str)
        for k in keys:
            g = [r for r in sub
                 if (r.get("press_mm"), tuple(r.get("place_err_mm") or ()), r.get("seat_z"),
                     tuple(r.get("place_xy") or ()), r.get("tip_len_mm"),
                     r.get("cycles_asked")) == k]
            good = [r for r in g if r.get("ok")]
            f = lambda key: (st.mean([r[key] for r in good]) if good else 0.0)  # noqa: E731
            print(f"   {str(k):62} ok {len(good)}/{len(g)}  turns {f('turns'):6.2f}  "
                  f"deg/cy {f('gain_mean_deg'):6.2f}  drift {f('drift_mm'):6.2f}mm  "
                  f"tilt {f('final_tilt_deg'):5.2f}  seat_off "
                  f"{(st.mean([r['seat_offset_mm'] for r in good if r.get('seat_offset_mm') is not None]) if any(r.get('seat_offset_mm') is not None for r in good) else float('nan')):6.2f}mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
