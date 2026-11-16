"""Summary of rev_rescore.py's rows (hand-4a, 2026-11-16): per grasp the mean over seeds of J (a run that failed in HOM's
least squares counts as J 0 and not held), tracking 1 - err/ref, held runs, the largest rise along z_P (mm) and the
largest tilt (deg), at each friction; the grasps ranked by J at each friction, Spearman's rho between the frictions, and
each grasp's J in the grid's own rows (rows/grid2dff_v4b_sv25r and the gr grid's, track_score, seeds 0-2) for reference.

    .venv/bin/python rev_rescore_report.py rescore_grids_talk2.jsonl [--json OUT.json] [--top 8]
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path.home() / "Programs/hand"
XY = {329: (105, 70), 162: (70, 110)}


def load(path):
    rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    by = defaultdict(list)
    for r in rows:
        by[(r["grasp"], r["params"].get("mu"))].append(r)
    return by


def agg(rs):
    ok = [r for r in rs if "error" not in r]
    J = [r["J"] if "error" not in r else 0.0 for r in rs]
    tr = [(r["track"] or 0.0) if "error" not in r else 0.0 for r in rs]
    return {"n": len(rs), "failed": len(rs) - len(ok), "J": float(np.mean(J)), "track": float(np.mean(tr)),
            "held": sum(bool(r.get("held")) for r in ok),
            "rise_max": float(np.mean([r["rise_max"] for r in ok])) if ok else None,
            "tilt_max": float(np.mean([r["tilt_max"] for r in ok])) if ok else None,
            "contact_all": float(np.mean([r.get("contact_all") or 0.0 for r in ok])) if ok else None,
            "index_pad": float(np.mean([r["pad_frac"]["index"] for r in ok])) if ok else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows")
    ap.add_argument("--json", default=None)
    ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args()
    by = load(a.rows)
    grasps = sorted({g for g, _m in by})
    mus = sorted({m for _g, m in by})
    S = {m: {g: agg(by[(g, m)]) for g in grasps if (g, m) in by} for m in mus}
    out = {"rows": a.rows, "mus": mus, "grasps": {g: {str(m): S[m].get(g) for m in mus} for g in grasps}}
    if len(mus) >= 2:
        common = [g for g in grasps if all(g in S[m] for m in mus)]
        rho = spearmanr([S[mus[0]][g]["J"] for g in common], [S[mus[-1]][g]["J"] for g in common]).statistic
        out["rho_J_between_mu"] = float(rho)
        print(f"Spearman rho of J between mu {mus[0]} and {mus[-1]} over {len(common)} grasps: {rho:.2f}")
    for m in mus:
        print(f"\nmu {m}: grasps by J (J / track / held / rise mm / tilt deg / index pad share)")
        for call in sorted({int(g.split('_c')[1].split('_')[0]) for g in grasps if '_c' in g}, reverse=True):
            gs = sorted([g for g in S[m] if f"_c{call}_" in g], key=lambda g: -S[m][g]["J"])
            print(f"  layout {XY.get(call, call)}:")
            for g in gs[:a.top]:
                s = S[m][g]
                print(f"    {g:22s} J {s['J']:.3f} track {s['track']:.2f} held {s['held']}/{s['n']} rise "
                      f"{(s['rise_max'] or 0):5.1f} tilt {(s['tilt_max'] or 0):5.1f} index {(s['index_pad'] or 0):.2f}"
                      + (f" failed {s['failed']}" if s["failed"] else ""))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    sys.exit(main())
