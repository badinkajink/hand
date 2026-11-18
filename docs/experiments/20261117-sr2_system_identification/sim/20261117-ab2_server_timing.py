"""The HOM server's compute time and record lengths over the night's two A/B blocks of the g0 twist at (70, 110) (W3 of
the identification queue, 2026-11-17): the first block (19:42-20:08) with the 2 ms filter sub-steps (--filter-step-s
0.002), the second (21:06-21:25) with --filter-exact in their place.

Per server record in docs/experiments/20261115-hom_bench_trials/runs/ since 19:40: the arm from the header (tips, filter_exact,
filter_step_s, contact_force_off_N), the periods, the end reason, and compute_ms (the server's time from a state's arrival
to its targets, per 20 ms period) as median, 95th percentile and largest. A record of one period is a fragment: corvus's
last state arriving after the server had closed the trial (W1c addendum). The lab's tracking scores are in
data/20261117-night_repeats_tracking.json of the bench study and are not recomputed here.

    python3 sim/20261117-ab2_server_timing.py [--out data/20261117-ab2_server_timing.json]
"""
import argparse
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RUNS = ROOT / "docs/experiments/20261115-hom_bench_trials/runs"


def arm(h):
    a = "on" if h.get("tips") not in (None, "off") else "off"
    a += "_exact" if h.get("filter_exact") else ("_fstep" if h.get("filter_step_s") else "_base")
    return a + ("_hyst" if h.get("contact_force_off_N") else "")


def pctl(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(p / 100 * (len(v) - 1))))]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--since", default="20261117-194000")
    ap.add_argument("--out", default=str(HERE.parent / "data/20261117-ab2_server_timing.json"))
    a = ap.parse_args()
    recs = []
    for p in sorted(RUNS.glob("20261117-*_twist_d40_trial*.jsonl")):
        if p.name[:15] < a.since:
            continue
        lines = p.read_text().splitlines()
        h = json.loads(lines[0])
        per = [json.loads(x) for x in lines[1:] if x.startswith('{"t"')]
        end = json.loads(lines[-1]) if '"kind": "end"' in lines[-1] else {}
        c = [r["compute_ms"] for r in per if r.get("compute_ms") is not None]
        recs.append(dict(record=p.name, block=1 if p.name[:15] < "20261117-210000" else 2, arm=arm(h), git=h.get("git"),
                         periods=len(per), fragment=len(per) <= 1, ended=end.get("reason"),
                         compute_ms=dict(median=round(statistics.median(c), 2), p95=round(pctl(c, 95), 2),
                                         max=round(max(c), 2)) if c else None))
    by = {}
    for r in recs:
        if r["fragment"]:
            continue
        k = f"block {r['block']} {r['arm']}"
        by.setdefault(k, []).append(r["compute_ms"]["median"])
    out = dict(script=str(Path(__file__).relative_to(ROOT)), runs=str(RUNS.relative_to(ROOT)), since=a.since,
               records=recs, compute_median_ms_by_block_arm={k: [min(v), max(v)] for k, v in sorted(by.items())},
               fragments=[r["record"] for r in recs if r["fragment"]])
    Path(a.out).write_text(json.dumps(out, indent=1))
    for r in recs:
        cm = r["compute_ms"]
        print(f"{r['record'][:15]} block {r['block']} {r['arm']:14s} {r['periods']:4d} {str(r['ended']):18s} "
              f"{cm['median']:6.2f} {cm['p95']:6.2f} {cm['max']:6.2f}")
    print(json.dumps(out["compute_median_ms_by_block_arm"], indent=1))
    print("fragments:", out["fragments"])
    print("->", a.out)


if __name__ == "__main__":
    main()
