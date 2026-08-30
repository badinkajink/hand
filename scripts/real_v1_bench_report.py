#!/usr/bin/env python3
"""Read a pile of bench sessions and print the one table that matters.

  python3 scripts/real_v1_bench_report.py                 # every session in the suite
  python3 scripts/real_v1_bench_report.py --design g12
  python3 scripts/real_v1_bench_report.py --json out.json

The comparison that carries the result is the DRIVER FINGER'S YAW AT THE LAST STEP,
loaded vs free-air on the same day: free air is what the servo can do against
nothing, so the difference is grip load and nothing else (g12, 2026-08-29: 25.8
free, 12.6 clamped, a repeatable 13 deg of missing turn).  A raw loaded number
alone cannot distinguish a stalled finger from a droopy one.
"""
import argparse, json, statistics, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SUITE = REPO / "docs/experiments/20260830-real_v1_bench_suite"
FINGERS = ("thumb", "index", "middle")


def read_log(p: Path):
    """-> (last step row, abort record or None, hold rows)"""
    steps, abort, holds = [], None, []
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("kind") == "step":
            steps.append(r)
        elif r.get("kind") == "abort":
            abort = r
        elif r.get("kind") == "hold":
            holds.append(r)
    return (steps[-1] if steps else None), abort, holds, len(steps)


def yaw_at_end(row, finger):
    return row["joints"][finger]["got"]["yaw"] if row else None


def peak_load(p: Path, finger):
    best = 0.0
    for line in p.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("kind") in ("step", "hold"):
            v = r["joints"][finger].get("yaw_load")
            if v is not None:
                best = max(best, abs(v))
    return best


def plateau(p: Path, finger):
    """Longest run of an identical load value, and its value -- the overload
    protection signature is a value held for tens of steps without varying."""
    vals = []
    for line in p.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("kind") == "step":
            v = r["joints"][finger].get("yaw_load")
            vals.append(None if v is None else abs(v))
    best_n, best_v, n, prev = 0, None, 0, object()
    for v in vals:
        n = n + 1 if v == prev else 1
        prev = v
        if v is not None and n > best_n:
            best_n, best_v = n, v
    return best_v, best_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", action="append", default=[])
    ap.add_argument("--suite", default=str(SUITE))
    ap.add_argument("--json", default=None)
    ap.add_argument("--include-excluded", action="store_true",
                    help="also read sessions carrying an EXCLUDED.txt")
    a = ap.parse_args()

    sessions = sorted(Path(a.suite).glob("*/MANIFEST.json"))
    if not sessions:
        print(f"no sessions under {a.suite}"); return 1

    out, dropped = [], []
    for mp in sessions:
        man = json.loads(mp.read_text())
        if a.design and man["design"] not in a.design:
            continue
        d = mp.parent
        # A session can be retired without deleting it: drop an EXCLUDED.txt in it
        # saying why.  Session 20260830-1359-g12 is the reason this exists -- its
        # repeats did not return to the staged pose, so they are not repeats.
        exc = d / "EXCLUDED.txt"
        if exc.exists() and not a.include_excluded:
            dropped.append((d.name, exc.read_text().strip().splitlines()[0]))
            continue
        drivers = man["plan_facts"].get("drivers") or [man["plan_facts"]["driver"]]
        drv = man["plan_facts"]["driver"]
        free = None
        fp = d / "freeair.jsonl"
        if fp.exists():
            free = yaw_at_end(read_log(fp)[0], drv)
        obs = {o["kind"]: o for o in man.get("observations", [])}
        for r in man.get("runs", []):
            if r.get("arm") != "loaded" or not r.get("log"):
                continue
            lp = d / r["log"]
            if not lp.exists():
                continue
            last, abort, holds, nstep = read_log(lp)
            pv, pn = plateau(lp, drv)
            o = obs.get(f"loaded_{r.get('repeat')}", {})
            out.append({
                "session": d.name, "design": man["design"], "repeat": r.get("repeat"),
                "servo_speed": r.get("servo_speed"),
                "path": man["args"].get("path"), "max_u": man["args"].get("max_u"),
                "driver": drv, "drivers": drivers,
                "holders": man["plan_facts"].get("holders", []),
                "steps": nstep, "aborted_at": (abort or {}).get("i"),
                "free_air_yaw": free,
                "loaded_yaw": yaw_at_end(last, drv),
                "deficit": (None if free is None or last is None
                            else round(free - yaw_at_end(last, drv), 2)),
                "peak_driver_load": peak_load(lp, drv),
                "plateau_load": pv, "plateau_steps": pn,
                "hold_samples": len(holds),
                "held": o.get("held"), "rotation_deg": o.get("rotation_deg"),
                "rotation_source": o.get("rotation_source"),
                "fingers_touched": o.get("fingers_touched"), "slipped": o.get("slipped"),
                "notes": o.get("notes", ""),
            })

    for name, why in dropped:
        print(f"  (excluded {name}: {why})")
    if dropped:
        print()
    if not out:
        print("no loaded runs found"); return 1

    def s(v, w, f="{}"):
        return (f.format(v) if v is not None else "-").rjust(w)

    print(f"{'design':9s} {'rep':>3s} {'path':5s} {'maxu':>5s} {'drv':6s} "
          f"{'free':>6s} {'load':>6s} {'defc':>6s} {'peakL':>6s} {'plateau':>9s} "
          f"{'spd':>4s} {'held':>5s} {'rot':>6s} {'slip':>5s} {'touch':>5s}")
    for r in out:
        print(f"{r['design']:9s} {s(r['repeat'],3)} {(r['path'] or '-'):5s} "
              f"{s(r['max_u'],5,'{:.2f}')} {r['driver'][:6]:6s} "
              f"{s(r['free_air_yaw'],6,'{:+.1f}')} {s(r['loaded_yaw'],6,'{:+.1f}')} "
              f"{s(r['deficit'],6,'{:+.1f}')} {s(r['peak_driver_load'],6,'{:.0f}')} "
              f"{s(r['plateau_load'],4,'{:.0f}')}x{s(r['plateau_steps'],4,'{:d}')} "
              f"{s(r['servo_speed'],4)} {s(r['held'],5)} {s(r['rotation_deg'],6,'{:.0f}')} "
              f"{s(r['slipped'],5)} {s(r['fingers_touched'],5)}")

    print()
    for design in sorted({r["design"] for r in out}):
        rows = [r for r in out if r["design"] == design]
        # A run the operator flagged as SLIPPED is not a reorientation measurement:
        # the shaft rotated because the grasp let go, not because the fingers turned
        # it.  Averaging the two together is what turned g12 into "36.7 +- 30.1 deg".
        turned = [r for r in rows if not r.get("slipped")]
        slipped = [r for r in rows if r.get("slipped")]
        rot = [r["rotation_deg"] for r in turned if r["rotation_deg"] is not None]
        defc = [r["deficit"] for r in rows if r["deficit"] is not None]
        held = [r["held"] for r in rows if r["held"] is not None]
        print(f"  {design:9s} n={len(rows)}"
              + (f"  rotation {statistics.mean(rot):.1f}"
                 + (f" +-{statistics.stdev(rot):.1f}" if len(rot) > 1 else "")
                 + f" deg (n={len(rot)} turned)" if rot else "")
              + (f"  grip deficit {statistics.mean(defc):+.1f} deg" if defc else "")
              + (f"  held {sum(held)}/{len(held)}" if held else ""))
        if slipped:
            sr = [f"rep {r['repeat']} {r['rotation_deg']:.0f} deg" for r in slipped
                  if r["rotation_deg"] is not None]
            print(f"             !! {len(slipped)}/{len(rows)} runs SLIPPED and are excluded "
                  f"from the mean ({', '.join(sr)}) -- the shaft turned because the grasp "
                  f"released, which is the opposite result")
        eye = {r["rotation_source"] for r in turned if r.get("rotation_source")}
        if rot and eye <= {"eye"}:
            print(f"             .. rotation_source is the operator's eye on every run; "
                  f"treat +-5 deg as the floor and do not rank designs inside it")
        prot = [r for r in rows if r["plateau_load"] == 200 and r["plateau_steps"] >= 10]
        if prot:
            print(f"             !! {len(prot)}/{len(rows)} runs show the load-200 overload "
                  f"plateau (protective_torque = 20%), not a mechanical stall")

    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=2))
        print(f"\n  wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
