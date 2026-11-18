"""Step responses of step_test.py's records (7.5 V supply, 2026-11-17) beside the yaw holds of 2026-11-06 at 5 V
(docs/experiments/20261106-sr2_servo_sysid/data/holds_c.tsv: steps of 20, 30 and 40 deg at speed 80, read by the
station's telemetry at about 8.6 Hz). Per step: the times to 10, 50 and 90 % of the step from its command, the peak
slew between samples, the overshoot and the error at the end of the hold. For the 5 V holds the times come from
samples 0.1 s apart, so they are bounds to within that; the 7.5 V records are sampled every 20 ms.

    /usr/bin/python3 docs/experiments/20261117-sr2_servo_7v5/lab/step_analysis.py LOG.jsonl [LOG.jsonl ...] --out data/X.json
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
HOLDS = ROOT / "docs/experiments/20261106-sr2_servo_sysid/data/holds_c.tsv"


def response(t, q, q0, target):
    """t10, t50, t90 (s, linear between samples), peak slew (deg/s), overshoot (deg past the target), final error."""
    t, q = np.asarray(t, float), np.asarray(q, float)
    d = target - q0
    if abs(d) < 1e-6 or len(t) < 3:
        return None
    u = (q - q0) / d
    out = {}
    for frac in (0.1, 0.5, 0.9):
        k = np.nonzero(u >= frac)[0]
        if not len(k):
            out[f"t{int(frac * 100)}"] = None
            continue
        k = int(k[0])
        if k == 0:
            out[f"t{int(frac * 100)}"] = float(t[0])
        else:
            out[f"t{int(frac * 100)}"] = float(t[k - 1] + (frac - u[k - 1]) / (u[k] - u[k - 1]) * (t[k] - t[k - 1]))
    dt = np.diff(t)
    ok = dt > 1e-3
    out["slew"] = float(np.max(np.abs(np.diff(q)[ok] / dt[ok]))) if ok.any() else None
    out["overshoot"] = float(max(0.0, (np.max(u) - 1.0) * abs(d)))
    out["final_err"] = float(q[-1] - target) * float(np.sign(d))
    return out


def steps_7v5(path):
    rows = [json.loads(l) for l in open(path)]
    hdr = rows[0]
    seq = hdr["sequence"]
    by = defaultdict(list)
    for r in rows[1:]:
        if r.get("q") is None or "step" not in r:
            continue
        by[(r["step"], r["phase"])].append(r)
    res = []
    for i, (f, j, d) in enumerate(seq):
        for ph, name in ((1, "out"), (2, "back")):
            rr = by.get((i, ph))
            prev = by.get((i, ph - 1))
            if not rr or not prev:
                continue
            q0 = prev[-1]["q"][f][j]
            target = rr[0]["target"][f][j]
            t = [r["t_cmd"] for r in rr]
            q = [r["q"][f][j] for r in rr]
            x = response(t, q, q0, target)
            if x:
                res.append({"joint": f"{f}.{j}", "dir": name, "step_deg": round(target - q0, 2), "speed": hdr["speed"],
                            "first_sample_s": round(t[0], 4), **{k: None if v is None else round(v, 4) for k, v in x.items()}})
    return res


def holds_5v():
    rows = list(csv.DictReader(open(HOLDS), delimiter="\t"))
    groups = defaultdict(list)
    for r in rows:
        groups[(r["block"], r["step"], r["finger"], r["dir"], r["cmd_plan_deg"])].append(r)
    out = []
    keys = list(groups)
    for n, key in enumerate(keys):
        block, step, finger, dr, cmd = key
        rr = sorted(groups[key], key=lambda r: float(r["t_from_cmd_s"]))
        # the angle before this command: the last sample of the previous command of the same finger and block
        prev = None
        for k2 in reversed(keys[:n]):
            if k2[0] == block and k2[2] == finger:
                prev = sorted(groups[k2], key=lambda r: float(r["t_from_cmd_s"]))[-1]
                break
        if prev is None:
            continue
        q0, target = float(prev["deg"]), float(rr[0]["cmd_servo_deg"])
        if abs(target - q0) < 5:
            continue
        t = [float(r["t_from_cmd_s"]) for r in rr]
        q = [float(r["deg"]) for r in rr]
        x = response(t, q, q0, target)
        if x:
            out.append({"block": block, "joint": f"{finger}.yaw", "dir": dr, "step_deg": round(target - q0, 2),
                        "first_sample_s": round(t[0], 4), **{k: None if v is None else round(v, 4) for k, v in x.items()}})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = []
    for p in a.logs:
        res += steps_7v5(p)
    old = holds_5v()
    print("7.5 V, free air, 50 Hz read-back")
    print(f"{'joint':12s} {'dir':4s} {'spd':>3s} {'step':>6s} {'t10':>6s} {'t50':>6s} {'t90':>6s} {'slew':>6s} {'over':>5s} {'err':>5s}")
    fmt = lambda v, w=6, p=3: f"{v:{w}.{p}f}" if v is not None else " " * (w - 1) + "-"
    for r in res:
        print(f"{r['joint']:12s} {r['dir']:4s} {r['speed']:3d} {r['step_deg']:6.1f} {fmt(r['t10'])} {fmt(r['t50'])} "
              f"{fmt(r['t90'])} {fmt(r['slew'], 6, 0)} {fmt(r['overshoot'], 5, 1)} {fmt(r['final_err'], 5, 1)}")
    print("\n5 V, 2026-11-06 yaw holds (speed 80, ~8.6 Hz samples), steps of 15-25 deg")
    sel = [r for r in old if 15 <= abs(r["step_deg"]) <= 25 and r["block"] == "c"]
    for r in sel:
        print(f"{r['joint']:12s} {r['dir']:4s}  80 {r['step_deg']:6.1f} {fmt(r['t10'])} {fmt(r['t50'])} {fmt(r['t90'])} "
              f"{fmt(r['slew'], 6, 0)} {fmt(r['overshoot'], 5, 1)} {fmt(r['final_err'], 5, 1)}  (first sample {r['first_sample_s']:.3f} s)")
    if a.out:
        Path(a.out).write_text(json.dumps({"v7_5": res, "v5_holds_c": old}, indent=1))
        print(f"-> {a.out}")


if __name__ == "__main__":
    main()
