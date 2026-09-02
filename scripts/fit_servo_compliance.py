#!/usr/bin/env python3
"""Fit the SCS0009's finite stiffness from the bench's own run logs.

    python3 scripts/fit_servo_compliance.py \
        --logs docs/experiments/20260902-cb1-log-archive/logs \
        --out docs/experiments/20260902-servo-compliance/fit.json

WHY THIS EXISTS.  Every `real_v1` scene actuates its fingers with `<position kp="4000">`, and
no MJCF in the repo carries a `frictionloss`.  That is a servo about two orders of magnitude
stiffer than the one bolted to the hand, and it is the reason the exported plans are open loop
in COMMANDED units while the bench answers in achieved ones.  This module turns 240-odd logged
runs into the two numbers MuJoCo needs to stop lying about the plant.

WHAT IS MEASURED, AND WHY IT IS NOT `real_v1_bench_arrival.py`.  That script reports an
`achieved_fraction` -- a ratio of two DELTAS, (after - before) / (cmd_end - cmd_start).  It is
the right quantity for "how much of the turn happened", and the wrong one for a stiffness: a
joint already deflected at `before` has part of its sag cancel out of the delta, and a ratio
over a small commanded travel is dominated by whatever the denominator does.  A position
servo's compliance is an ABSOLUTE angle, so that is what is fitted here:

    deflection = commanded - achieved      at the settled terminal hold ('after')

Reported per joint and against `servo_load`, which is the SCS0009's PWM-duty proxy for torque
-- not torque in N*m, so the fitted slope is deg per load unit and becomes a MuJoCo `kp` only
after the sim's own actuator force at the same pose supplies the missing scale.  The intercept
is stiction: real, unmodelled, and independent of load.

CAVEAT THE FIT CANNOT SEE.  `after` is one sample.  If the servo is still creeping when it is
taken, some tracking lag is scored as deflection; `--report-settling` compares the last two
`during` samples against `after` so that assumption is checked rather than assumed.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/morphohand/driver/manta/host"))

from manta_hand.plan import FINGER_ID, JOINT_SIGN, SIM_JOINT_TO_SERVO  # noqa: E402

FINGERS = ("thumb", "index", "middle")
JOINTS = ("yaw", "mcp", "pip")
TRIP_LOAD = 200.0   # protective_torque = 20 percent of the 0-1000 scale
LOAD_QUANTUM = 15.0  # every other reported load is a multiple of this

RUN_RE = re.compile(r"^(\d{8})-(\d{6})-(.+)-([0-9a-f]{6})\.jsonl$")


def read_run(path: Path) -> dict | None:
    """One run log -> {design, commands, telemetry}. Returns None if it carries no servo
    feedback, which 1 of 244 archived runs does not."""
    m = RUN_RE.match(path.name)
    if not m:
        return None
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                return None
    cmds = [r for r in rows if r.get("kind") == "command" and "sim_joint_deg" in r]
    tel = [r for r in rows if r.get("kind") == "telemetry" and "servos" in r.get("data", {})]
    if not cmds or not tel:
        return None
    return {"design": m.group(3), "date": m.group(1), "run": path.stem,
            "commands": cmds, "telemetry": tel}


def achieved_sim_deg(servos: dict, finger: str, joint: str) -> float | None:
    """The servo's own zero-relative degrees back into sim degrees. `servo_deg` is sign-only,
    so the inverse is the same multiply."""
    rec = servos.get(str(FINGER_ID[finger]))
    if not rec:
        return None
    v = rec.get(SIM_JOINT_TO_SERVO[joint])
    return None if v is None else v * JOINT_SIGN[(finger, joint)]


def deflections(run: dict, phase: str = "after") -> list[dict]:
    """Absolute commanded-minus-achieved at the settled hold, one record per joint."""
    tel = [t for t in run["telemetry"] if t.get("phase") == phase]
    if not tel:
        return []
    sample = tel[-1]
    cmd = run["commands"][-1]["sim_joint_deg"]
    servos = sample["data"]["servos"]
    loads = sample["data"].get("servo_load") or {}
    out = []
    for f in FINGERS:
        for j in JOINTS:
            c = cmd.get(f, {}).get(j)
            a = achieved_sim_deg(servos, f, j)
            if c is None or a is None:
                continue
            sid = str(FINGER_ID[f] * 3 + JOINTS.index(j))
            load = loads.get(sid)
            out.append({"design": run["design"], "run": run["run"], "joint": f"{f}_{j}",
                        "commanded_deg": c, "achieved_deg": a, "deflection_deg": c - a,
                        "load": None if load is None else abs(float(load))})
    return out


def ols(xs: list[float], ys: list[float]) -> tuple[float, float, float, float]:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return float("nan"), my, float("nan"), float("nan")
    k = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    b = my - k * mx
    ss = sum((y - my) ** 2 for y in ys)
    rss = sum((y - (k * x + b)) ** 2 for x, y in zip(xs, ys))
    return k, b, (1 - rss / ss if ss else float("nan")), math.sqrt(rss / n)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--report-settling", action="store_true",
                    help="how much the last two 'during' samples still move before 'after'")
    a = ap.parse_args()

    runs, skipped = [], 0
    for p in sorted(a.logs.glob("*.jsonl")):
        r = read_run(p)
        runs.append(r) if r else None
        skipped += 0 if r else 1
    print(f"{len(runs)} runs with servo feedback ({skipped} skipped) from {a.logs}")

    recs = [d for r in runs for d in deflections(r)]
    by_design = defaultdict(int)
    for r in runs:
        by_design[r["design"]] += 1
    print("designs: " + ", ".join(f"{k}={v}" for k, v in sorted(by_design.items())))

    print(f"\nABSOLUTE DEFLECTION AT THE HOLD (commanded - achieved), n={len(recs)}")
    print(f"{'joint':<13} {'n':>4} {'|cmd| med':>10} {'defl mean':>10} {'sd':>7} "
          f"{'p10':>7} {'p90':>7} {'n load':>7}")
    per_joint = {}
    for f in FINGERS:
        for j in JOINTS:
            key = f"{f}_{j}"
            s = [r for r in recs if r["joint"] == key]
            if not s:
                continue
            d = sorted(r["deflection_deg"] for r in s)
            nl = sum(1 for r in s if r["load"] is not None)
            per_joint[key] = {"n": len(s), "mean": st.mean(d), "sd": st.pstdev(d),
                              "p10": d[int(0.1 * len(d))], "p90": d[int(0.9 * len(d))]}
            print(f"{key:<13} {len(s):>4} "
                  f"{st.median([abs(r['commanded_deg']) for r in s]):>10.1f} "
                  f"{st.mean(d):>10.2f} {st.pstdev(d):>7.2f} "
                  f"{d[int(0.1*len(d))]:>7.2f} {d[int(0.9*len(d))]:>7.2f} {nl:>7}")

    withload = [r for r in recs if r["load"] is not None]
    fit = {"per_joint": {}, "trip_value": TRIP_LOAD}
    if withload:
        loads = sorted({r["load"] for r in withload})
        odd = [v for v in loads if v % LOAD_QUANTUM != 0]
        print(f"\nTHE TRIP.  Reported load takes {len(loads)} distinct values; all are multiples "
              f"of {LOAD_QUANTUM:g}\nexcept {odd}.  A value off the measurement quantum is a "
              f"REGISTER, not a reading:\n`protective_torque` = 20 percent of the 0-1000 scale.  A servo "
              f"there has unloaded itself,\nso its deflection is set by whatever pushes the joint "
              f"and is not a stiffness at all.")

        print(f"\nPER-JOINT STIFFNESS   |deflection| = k * load + b, TRIPPED SAMPLES EXCLUDED")
        print(f"{'joint':<13} {'n ok':>5} {'n trip':>7} {'trip%':>6} {'k':>9} {'b(deg)':>8} "
              f"{'R2':>7} {'rms':>6} | {'R2 if pooled':>13}")
        slopes = []
        for f in FINGERS:
            for j in JOINTS:
                key = f"{f}_{j}"
                sj = [r for r in recs if r["joint"] == key and r["load"] is not None]
                if not sj:
                    continue
                trip = [r for r in sj if r["load"] == TRIP_LOAD]
                ok = [r for r in sj if r["load"] != TRIP_LOAD]
                if len(ok) < 6:
                    print(f"{key:<13} {len(ok):>5} {len(trip):>7}   (too few untripped)")
                    continue
                k, b, r2, rms = ols([r["load"] for r in ok],
                                    [abs(r["deflection_deg"]) for r in ok])
                _, _, r2all, _ = ols([r["load"] for r in sj],
                                     [abs(r["deflection_deg"]) for r in sj])
                fit["per_joint"][key] = {"k_deg_per_load": k, "b_deg": b, "r2": r2, "rms_deg": rms,
                                         "n_ok": len(ok), "n_trip": len(trip),
                                         "trip_rate": len(trip) / len(sj),
                                         "r2_pooled_with_trip": r2all}
                if r2 > 0.5:
                    slopes.append(k)
                print(f"{key:<13} {len(ok):>5} {len(trip):>7} {len(trip)/len(sj):>5.0%} "
                      f"{k:>9.5f} {b:>8.2f} {r2:>7.3f} {rms:>6.2f} | {r2all:>13.3f}")
        if slopes:
            fit["slope_deg_per_load"] = st.median(slopes)
            print(f"\n  median slope over the {len(slopes)} joints that load enough to fit "
                  f"(R2>0.5): {st.median(slopes):.5f} deg/load-unit")
            print(f"  intercepts are {min(v['b_deg'] for v in fit['per_joint'].values()):+.2f} to "
                  f"{max(v['b_deg'] for v in fit['per_joint'].values()):+.2f} deg -- "
                  f"there is no stiction term to model.")

        print(f"\nWHAT THE TRIP COSTS  (same joint, tripped vs not)")
        print(f"{'joint':<13} {'trip%':>6} | {'|defl| ok':>10} {'|defl| trip':>12} "
              f"{'sd ok':>7} {'sd trip':>8}")
        for key, v in fit["per_joint"].items():
            sj = [r for r in recs if r["joint"] == key and r["load"] is not None]
            trip = [abs(r["deflection_deg"]) for r in sj if r["load"] == TRIP_LOAD]
            ok = [abs(r["deflection_deg"]) for r in sj if r["load"] != TRIP_LOAD]
            if len(trip) < 3:
                continue
            print(f"{key:<13} {v['trip_rate']:>5.0%} | {st.mean(ok):>10.2f} {st.mean(trip):>12.2f} "
                  f"{st.pstdev(ok):>7.2f} {st.pstdev(trip):>8.2f}")

    print(f"\nWHERE THE VARIANCE LIVES  (all {len(runs)} runs; load not required)")
    print("  between-design -> a per-design CALIBRATION removes it")
    print("  within-design  -> only FEEDBACK can")
    print(f"{'joint':<13} {'n':>4} {'sd total':>9} {'sd within':>10} {'sd between':>11} {'% within':>9}")
    for key in per_joint:
        s = [r for r in recs if r["joint"] == key]
        bydes = defaultdict(list)
        for r in s:
            bydes[r["design"]].append(r["deflection_deg"])
        bydes = {k: v for k, v in bydes.items() if len(v) >= 3}
        if len(bydes) < 3:
            continue
        within = [x - st.mean(v) for v in bydes.values() for x in v]
        sw, stot = st.pstdev(within), st.pstdev([r["deflection_deg"] for r in s])
        if stot == 0:
            continue
        sb = math.sqrt(max(stot ** 2 - sw ** 2, 0.0))
        per_joint[key].update({"sd_total": stot, "sd_within_design": sw, "sd_between_design": sb})
        print(f"{key:<13} {len(s):>4} {stot:>9.2f} {sw:>10.2f} {sb:>11.2f} "
              f"{100*sw**2/stot**2:>8.0f}%")

    if a.report_settling:
        print("\nSETTLING: |after - last during| per joint, deg "
              "(large => 'after' is still moving and the fit above includes tracking lag)")
        moved = []
        for r in runs:
            dur = [t for t in r["telemetry"] if t.get("phase") == "during"]
            aft = [t for t in r["telemetry"] if t.get("phase") == "after"]
            if not dur or not aft:
                continue
            for f in FINGERS:
                for j in JOINTS:
                    x = achieved_sim_deg(dur[-1]["data"]["servos"], f, j)
                    y = achieved_sim_deg(aft[-1]["data"]["servos"], f, j)
                    if x is not None and y is not None:
                        moved.append(abs(y - x))
        if moved:
            moved.sort()
            print(f"  n={len(moved)}  median={st.median(moved):.2f}  "
                  f"p90={moved[int(0.9*len(moved))]:.2f}  max={max(moved):.2f}")

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(
            {"n_runs": len(runs), "per_joint": per_joint, "fit": fit,
             "records": recs}, indent=1))
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
