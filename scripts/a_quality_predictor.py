"""A-side predictor of B fate — pair Policy-A trajectory-health scorecards with B outcomes.

Standing CPU task from docs/rl/morph_sweep_STATUS.md (2026-07-11): the P1/P2 probes showed the
per-design verdict is dominated by the A draw, and the A health gate can't see the difference
between draws that yield cos 0.49 vs -0.16. This script asks whether any FINE-GRAINED A metric
(from the kept A checkpoint's .health.json) predicts the downstream B held-cos / B collapse —
a usable predictor would restore single-draw evaluation and halve landscape cost.

Data: every record in docs/experiments/MORPH_PIPELINE_<tag>.json whose A run still has
results/rl/<run>/tensorboard/*.health.json on disk (results/ is gitignored — rerun while runs
are still present, and re-run after P4 with --tags ... global12x2).

Findings as of 2026-07-11 (n=26 pairs): see docs/notes/a_quality_predictor.md.

Usage:
  uv run python scripts/a_quality_predictor.py [--tags confirm large16 rescue avar global12x2]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

A_METRICS = ["minz", "jerk", "drift", "slide", "tipF", "spread", "fmin", "fmax", "tfmin", "ccount"]


def a_health(run: str):
    tb = ROOT / "results/rl" / run / "tensorboard"
    if not tb.exists():
        return None
    hs = sorted(tb.glob("*.health.json"))
    return json.loads(hs[-1].read_text()) if hs else None


def collect(tags):
    rows = []
    for tag in tags:
        p = ROOT / f"docs/experiments/MORPH_PIPELINE_{tag}.json"
        if not p.exists():
            print(f"[skip] {p} missing")
            continue
        for rec in json.load(open(p)):
            arun = (rec.get("A") or {}).get("run")
            h = a_health(arun) if arun else None
            if not h:
                continue
            m = h.get("metrics", {})
            fm = m.get("force_mean") or []
            tf = m.get("touch_frac") or []
            ho = rec.get("handoff") or {}
            hm = (ho.get("health") or {}).get("metrics") or {}
            rows.append(
                dict(
                    tag=tag,
                    id=rec["id"],
                    arun=arun,
                    a_verdict=h.get("verdict"),
                    minz=m.get("min_z_hold"),
                    jerk=m.get("ang_jerk"),
                    drift=m.get("net_drift_cm"),
                    slide=m.get("slide_ratio"),
                    tipF=m.get("tip_force"),
                    spread=m.get("contact_spread"),
                    fmin=min(fm) if fm else None,
                    fmax=max(fm) if fm else None,
                    tfmin=min(tf) if tf else None,
                    ccount=m.get("contact_count"),
                    b_cos=hm.get("held_cos_tail"),
                    b_minz=ho.get("min_z_post"),
                    b_collapsed=bool((rec.get("B") or {}).get("aborted")),
                )
            )
    return rows


def spearman(x, y):
    n = len(x)
    rkx, rky = [0] * n, [0] * n
    for k, i in enumerate(sorted(range(n), key=lambda i: x[i])):
        rkx[i] = k
    for k, i in enumerate(sorted(range(n), key=lambda i: y[i])):
        rky[i] = k
    mx, my = sum(rkx) / n, sum(rky) / n
    cov = sum((rkx[i] - mx) * (rky[i] - my) for i in range(n))
    vx = sum((v - mx) ** 2 for v in rkx)
    vy = sum((v - my) ** 2 for v in rky)
    return cov / math.sqrt(vx * vy) if vx * vy > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["confirm", "large16", "rescue", "avar"])
    args = ap.parse_args()
    rows = collect(args.tags)

    hdr = ["a_verdict"] + A_METRICS + ["b_cos", "b_collapsed"]
    print(f"{'id':16s}" + "".join(f"{k.replace('a_',''):>9s}" for k in hdr))
    for r in rows:
        print(f"{r['id']:16s}" + "".join(f"{str(r[k])[:8]:>9s}" for k in hdr))

    done = [r for r in rows if r["b_cos"] is not None and not r["b_collapsed"]]
    print(f"\npairs with A scorecard: {len(rows)}  completed-B: {len(done)}  "
          f"B-collapsed: {sum(r['b_collapsed'] for r in rows)}")

    print("Spearman(A metric, B held-cos) on completed Bs:")
    for k in A_METRICS:
        sub = [(r[k], r["b_cos"]) for r in done if r[k] is not None]
        if len(sub) >= 8:
            rho = spearman([a for a, _ in sub], [b for _, b in sub])
            print(f"  {k:8s} n={len(sub):2d}  rho={rho:+.2f}")

    print("\nA metrics, B-collapsed vs B-completed (median):")
    for k in A_METRICS:
        c = [r[k] for r in rows if r["b_collapsed"] and r[k] is not None]
        n = [r[k] for r in rows if not r["b_collapsed"] and r[k] is not None]
        if c and n:
            print(f"  {k:8s} collapsed={st.median(c):.3f} (n={len(c)})  ok={st.median(n):.3f} (n={len(n)})")

    art = [r for r in rows if r["minz"] is not None and r["minz"] < 0.05]
    if art:
        print("\nA scorecards with min_z_hold < 0.05 (pre-lift-window artifact — spurious drop-FAIL):")
        for r in art:
            print(f"  {r['id']:16s} minz={r['minz']:.4f} verdict={r['a_verdict']}  -> B cos {r['b_cos']}")

    idle = [r for r in rows if r["fmin"] is not None and r["fmin"] < 0.5]
    if idle:
        print("\nA deliveries with an idle finger (min force < 0.5 N — candidate veto):")
        for r in idle:
            print(f"  {r['id']:16s} fmin={r['fmin']:.1f} tfmin={r['tfmin']}  -> B cos {r['b_cos']} "
                  f"collapsed={r['b_collapsed']}")


if __name__ == "__main__":
    main()
