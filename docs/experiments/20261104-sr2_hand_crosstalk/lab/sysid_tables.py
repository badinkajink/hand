"""The coordinator's tables from the free-space sysid's run folders (lab/sysid_chain.sh), as plain .tsv, since GitHub
refuses LFS transfers from 21:37 on 11-05 (the repository's LFS budget):
- dwells.tsv: one row per step. Columns: block, pass, step, the tested finger, joint and direction, and the command in
  the plan's degrees and in the servo's. Then the runner's times and its error, clearance and blocked flag. Last, the
  mean and sd of each of the nine servos' angle and load over the dwell (t_settled to t_end), with the sample count.
- holds_c.tsv: for blocks c and c40, the moving yaw's telemetry samples from each yaw step's command to its end (the
  10 s hold), each with t from the command, the angle and the load.
Angles are the station's servo degrees from each calibrated zero; the index's and the middle's yaw are minus the
plan's yaw. Loads are the servos' present load (sign and magnitude, per mille of full duty). The telemetry runs at about
8.6 Hz. Each sample's time is t_sample = t_recv minus its age, on corvus's clock.

    /usr/bin/python3 sysid_tables.py LOG_DIR OUT_DIR
"""
import bisect
import json
import statistics
import sys
from pathlib import Path

LOG, OUT = Path(sys.argv[1]), Path(sys.argv[2])
BLOCKS = ("a1", "b1", "a2", "b2", "c", "c40")
PASS = {"a1": 1, "b1": 1, "a2": 2, "b2": 2, "c": 1, "c40": 1}
FID = {"thumb": "0", "index": "1", "middle": "2"}
KEY = {"yaw": "aa", "mcp": "fe1", "pip": "fe2"}
SID = {(f, j): str(3 * int(FID[f]) + k) for f in FID for k, j in enumerate(("yaw", "mcp", "pip"))}
YAW_SIGN = {"thumb": 1.0, "index": -1.0, "middle": -1.0}
NINE = [(f, j) for f in FID for j in ("yaw", "mcp", "pip")]


def servo_cmd(f, j, v):
    return YAW_SIGN[f] * v if j == "yaw" else v


def r(v, n):
    return "" if v is None else f"{v:.{n}f}"


def stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    return statistics.fmean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


OUT.mkdir(parents=True, exist_ok=True)
head = ["block", "pass", "step", "finger", "joint", "dir", "cmd_plan_deg", "cmd_servo_deg", "t_cmd", "settle_s",
        "t_settled", "t_end", "err_deg", "err_joint", "clear_mm", "blocked", "n"]
for f, j in NINE:
    head += [f"{f}_{j}_deg_mean", f"{f}_{j}_deg_sd", f"{f}_{j}_load_mean", f"{f}_{j}_load_sd"]
dwell_rows, hold_rows = ["\t".join(head)], ["\t".join(["block", "step", "finger", "dir", "cmd_plan_deg", "cmd_servo_deg",
                                                        "t_from_cmd_s", "deg", "load"])]
for b in BLOCKS:
    d = LOG / b
    if not (d / "events.jsonl").exists():
        continue
    plan = json.load(open(LOG / "plan" / f"20261106-sysid_{b}.json"))["steps"]
    ev = [json.loads(x) for x in open(d / "events.jsonl")]
    tl = sorted((json.loads(x) for x in open(d / "telemetry.jsonl")), key=lambda x: x["t_sample"])
    ts = [x["t_sample"] for x in tl]
    for e in ev:
        if e.get("kind") != "step":
            continue
        s = plan[e["step"]]
        f, j = s["finger"], s["joint"]
        lo, hi = (e.get("t_settled") or e["t_end"]), e["t_end"]
        win = tl[bisect.bisect_left(ts, lo):bisect.bisect_right(ts, hi)]
        row = [b, PASS[b], e["step"], f, j, s["dir"], s["cmd"], servo_cmd(f, j, s["cmd"]), r(e["t_cmd"], 3),
               r(e.get("settle_s"), 2), r(e.get("t_settled"), 3), r(e["t_end"], 3), e.get("err_deg", ""),
               e.get("err_joint") or "", e.get("clear_mm", ""), int(bool(e.get("blocked"))), len(win)]
        for ff, jj in NINE:
            m, sd = stats([(x["servos"] or {}).get(FID[ff], {}).get(KEY[jj]) for x in win])
            lm, lsd = stats([(x["load"] or {}).get(SID[ff, jj]) for x in win])
            row += [r(m, 3), r(sd, 3), r(lm, 1), r(lsd, 1)]
        dwell_rows.append("\t".join(str(v) for v in row))
        if b in ("c", "c40") and j == "yaw":
            for x in tl[bisect.bisect_left(ts, e["t_cmd"]):bisect.bisect_right(ts, e["t_end"])]:
                hold_rows.append("\t".join(str(v) for v in (
                    b, e["step"], f, s["dir"], s["cmd"], servo_cmd(f, j, s["cmd"]), r(x["t_sample"] - e["t_cmd"], 3),
                    r((x["servos"] or {}).get(FID[f], {}).get("aa"), 3), (x["load"] or {}).get(SID[f, "yaw"], ""))))
(OUT / "dwells.tsv").write_text("\n".join(dwell_rows) + "\n")
(OUT / "holds_c.tsv").write_text("\n".join(hold_rows) + "\n")
print(f"dwells.tsv {len(dwell_rows) - 1} rows, holds_c.tsv {len(hold_rows) - 1} rows -> {OUT}")
