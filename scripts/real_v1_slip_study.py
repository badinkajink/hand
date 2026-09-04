"""Does the reorientation's residual tilt behave the same way on every hand?

The bench study (docs/experiments/20260904-real_v1_bench) measured the turn's end tilt on ONE
hand, rv05_manual, and reported that a payload-compensated arm delivers the tool 8.03 deg off
vertical with a seed spread of 0.42 deg, that holding it there makes the tilt worse rather than
better, and that squeezing through the turn destroys it. Three claims, one morphology. A number
measured on one hand is a property of that hand until a second hand has it too.

Every real_v1 design with a CEM grasp gets the SAME commanded chain -- same turn schedule, same
budget, same carry grip, same seat -- and the questions are:

  settle    hold 0..900 steps, every design      is the 8 deg a property of the PRIMITIVE?
  squeeze   turn squeeze 0..2 mm, every design   is "just squeeze it" wrong everywhere?
  wrist     payload declared vs not              is the sag correction hand-independent?

The turn seam is read whether or not the rest of the chain survives: a hand that turns the tool
and then loses it on the way down still measured a turn. `held_turn` is the gate (two pads on the
tool at the `turned` seam), `ok` is the whole chain.

    uv run --extra rl --extra arm python scripts/real_v1_slip_study.py \
        --out docs/experiments/20260904-real_v1_slip
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

DESIGNS = ROOT / "results/phase1/real_v1"
SEAT = dict(half_angle=45.0, socket_xy="0.04,-0.011")     # the published seat, unchanged
STEM = "screw_a45_x40_y-11"

# The published chain cell (docs/experiments/20260903-real_v1_chain + the seat's own carry), with
# NOTHING retuned per design. Retuning it per hand would answer a different question.
BASE = dict(obj="screwdriver_medium", lift=0.10, angle_deg=-90.0, axis_k=0.25, turn_steps=550,
            budget=0.5, hold_steps=500, gap=0.002, descend_iters=1, descend_steps=400,
            airgrip="cradle", stand_order="ground", centre_x=0.004, grip_depth=0.050,
            squeeze=0.002, release_mm=6.0, twist_steps=120, move_steps=60,
            carry_squeeze=0.0003, press_mm=10.0, transport_steps=300,
            reindex="relay", relay_gait=True)


def seat_for(run: Path) -> dict | None:
    """The design's own countersink scene, built once and cached in its run dir."""
    js = run / f"{STEM}.json"
    if not js.exists():
        p = subprocess.run(
            [sys.executable, str(ROOT / "scripts/build_screw_scene.py"),
             "--scene", str(run / "frozen_scene.xml"),
             "--half-angle", str(SEAT["half_angle"]), "--socket-xy", SEAT["socket_xy"],
             "--out", str(run / f"{STEM}.xml"), "--out-json", str(js)],
            capture_output=True, text=True)
        if p.returncode != 0 or not js.exists():
            return None
    d = json.loads(js.read_text())
    d["scene"] = str((ROOT / d["scene"]).resolve() if not Path(d["scene"]).is_absolute()
                     else Path(d["scene"]))
    return d


def arm_for(cache: Path, run: Path, seat: dict, stack: float, pgc: bool) -> tuple:
    """The UR5e scene for one design, at one wrist stack, with or without a declared payload."""
    cache.mkdir(parents=True, exist_ok=True)
    t = f"{run.name}_s{stack * 1000:.0f}{'_pgc' if pgc else ''}"
    sc, ik = cache / f"{t}.xml", cache / f"{t}_ik.xml"
    if sc.exists() and ik.exists():
        return sc, ik
    cmd = [sys.executable, str(ROOT / "scripts/build_real_v1_arm_scene.py"),
           "--scene", seat["scene"], "--base=-0.50,0,0", "--wrist-stack", str(stack),
           "--out", str(sc), "--ik-out", str(ik)]
    if pgc:
        cmd.append("--payload-gravcomp")
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        for f in (sc, ik):
            f.unlink(missing_ok=True)
        return None, (p.stdout + p.stderr).strip().splitlines()[-1] if p.stderr else "build failed"
    return sc, ik


def _cell(kw):
    import probe_real_v1_chain as C
    tag, run, meta = kw.pop("_tag"), Path(kw.pop("_run")), kw.pop("_meta")
    try:
        r = C.chain(run, **kw)
    except Exception as exc:
        return {"arm": tag, "design": run.name, "error": repr(exc), "ok": False, **meta}
    seam = {s["phase"]: s for s in r["seams"]}
    t, g = seam.get("turned", {}), seam.get("lifted", {})
    r["seams"] = [{k: s[k] for k in ("phase", "t", "cos", "tilt_deg", "z", "pad_contacts",
                                     "pad_force_N", "ground_contacts", "spin_deg") if k in s}
                  for s in r["seams"]]
    r.pop("cycles", None)
    # The turn is measured on its own terms: a hand still holding the tool at the last commanded
    # turn step has turned it, whatever happens to the tool afterwards.
    r["pads_turned"] = t.get("pad_contacts")
    r["force_turned_N"] = t.get("pad_force_N")
    r["z_turned"] = t.get("z")
    r["pads_lifted"] = g.get("pad_contacts")
    r["held_turn"] = bool((t.get("pad_contacts") or 0) >= 2 and (t.get("z") or 0.0) > 0.08)
    r["arm"], r["design"] = tag, run.name
    r.update(meta)
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=DESIGNS / "slip_scenes")
    ap.add_argument("--designs", type=Path, default=DESIGNS)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--cycles", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--arms", default="settle,squeeze,wrist")
    args = ap.parse_args()
    arms = set(args.arms.split(","))

    runs = sorted(p for p in args.designs.iterdir()
                  if (p / "frozen_scene.xml").exists() and (p / "best_rollout.npz").exists())
    if not runs:
        ap.error(f"no design dirs with a frozen scene and a CEM grasp under {args.designs}")
    jobs, skipped = [], []
    ready = {}
    for run in runs:
        seat = seat_for(run)
        if seat is None:
            skipped.append({"design": run.name, "why": "no seat scene"})
            continue
        ready[run] = seat
    print(f"{len(ready)}/{len(runs)} designs with a seat scene")

    def add(tag, run, sc, ik, meta, reps=None, **kw):
        for rep in range(reps or args.reps):
            jobs.append({"_tag": tag, "_run": str(run), "_meta": dict(meta), "seed": rep,
                         "jitter": 0.0005, "cycles": args.cycles, "arm_ik": ik,
                         "scene_path": sc, **BASE,
                         "place_xy": ready[run]["socket_xy"], "seat_z": ready[run]["seat_z"],
                         "tip_len": ready[run]["tip_len"], **kw})

    for run, seat in ready.items():
        sc, ik = arm_for(args.cache, run, seat, 0.100, True)
        if sc is None:
            skipped.append({"design": run.name, "why": ik})
            print(f"  {run.name}: NO HOME POSE ({ik})")
            continue
        if "settle" in arms:
            for hs in (0, 150, 300, 500, 900):
                add("settle", run, sc, ik, {"pgc": True, "hold_steps_m": hs}, hold_steps=hs)
        if "squeeze" in arms:
            for tsq in (0.0005, 0.0010, 0.0020):
                add("squeeze", run, sc, ik, {"pgc": True, "turn_squeeze_m": tsq},
                    turn_squeeze=tsq, hold_steps=150)
        if "wrist" in arms:
            sc0, ik0 = arm_for(args.cache, run, seat, 0.100, False)
            if sc0 is None:
                skipped.append({"design": run.name, "why": f"no-pgc: {ik0}"})
                continue
            for hs in (0, 300, 900):
                add("wrist", run, sc0, ik0, {"pgc": False, "hold_steps_m": hs}, hold_steps=hs)

    print(f"{len(jobs)} cells on {args.workers} workers, {len(skipped)} designs skipped")
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(_cell, jobs, chunksize=1)):
            rows.append(r)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(jobs)}")

    args.out.mkdir(parents=True, exist_ok=True)
    dst = args.out / "slip_study.json"
    dst.write_text(json.dumps({"rows": rows, "skipped": skipped, "base": BASE,
                               "seat": SEAT}, separators=(",", ":")))
    print(f"-> {dst}")

    import statistics as st
    for arm in sorted({r["arm"] for r in rows}):
        sub = [r for r in rows if r["arm"] == arm]
        var = "turn_squeeze_m" if arm == "squeeze" else "hold_steps_m"
        print(f"\n== {arm}  ({len(sub)} cells)   held = holding at the last turn step")
        print(f"   {'design':22} {var:>14} {'held':>6} {'turned':>8} {'sd':>6} "
              f"{'settled':>8} {'sd':>6} {'chain':>6} {'padF':>6}")
        for dsn in sorted({r["design"] for r in sub}):
            for v in sorted({r[var] for r in sub if r["design"] == dsn}):
                g = [r for r in sub if r["design"] == dsn and r[var] == v]
                h = [r for r in g if r.get("held_turn")]
                f = lambda k: [float(r.get(k) or 0.0) for r in h]
                sd = lambda x: st.pstdev(x) if len(x) > 1 else 0.0
                print(f"   {dsn:22} {v:14g} {len(h):3}/{len(g):<2} "
                      f"{(st.mean(f('tilt_turned_deg')) if h else float('nan')):8.2f} "
                      f"{sd(f('tilt_turned_deg')):6.2f} "
                      f"{(st.mean(f('tilt_settled_deg')) if h else float('nan')):8.2f} "
                      f"{sd(f('tilt_settled_deg')):6.2f} "
                      f"{sum(1 for r in g if r.get('ok')):3}/{len(g):<2} "
                      f"{(st.mean(f('force_turned_N')) if h else float('nan')):6.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
