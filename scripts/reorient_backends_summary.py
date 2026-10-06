#!/usr/bin/env python3
"""Tables of the reorientation grid (reorient_backends*.py): per hand and condition, the fraction of seeds holding
the tool at the end of the bench hold and after 4 s more, the net turn and its spread over seeds, and the
bench's own outcome (paper/figures/ranking.json).

    python3 scripts/reorient_backends_summary.py [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "docs/experiments/20261006-fingertip_backends"
HANDS = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8"]


def load_rows():
    rows = {}
    for f in sorted(D.glob("reorient_*.jsonl")):
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "case_id" not in r or r.get("status") == "failed" and r.get("seed") is None:
                continue
            rows[r["case_id"]] = r                  # last write wins (re-runs are identical)
    return list(rows.values())


def cond_key(r):
    sim = r["sim"]
    if sim == "drake":
        return f"drake|{r['tip']}|{r['plant']}|mu{r['mu']}"
    return f"{sim}|{r['tip']}|{r['model']}|{r['plant']}|{r['numerics']}|ir{r.get('impratio'):g}|mu{r['mu']}"


def bench():
    rk = json.load(open(ROOT / "paper/figures/ranking.json"))["rows"]
    return {r["did"]: r for r in rk}


def summarize(rows):
    by = defaultdict(list)
    for r in rows:
        if r.get("status") not in ("complete", "ejected"):
            continue
        by[(cond_key(r), r["hand"])].append(r)
    out = {}
    for (c, h), rs in by.items():
        ok = [r for r in rs if r.get("status") == "complete"]
        th = [r["turn_hold_deg"] for r in ok]
        te = [r["turn_end_deg"] for r in ok]
        held_t = [r["turn_hold_deg"] for r in ok if r.get("held_hold")]
        out[(c, h)] = {
            "n": len(rs), "held_hold": sum(1 for r in ok if r.get("held_hold")),
            "held_end": sum(1 for r in ok if r.get("held_end")),
            "turn_hold_mean": float(np.mean(th)) if th else None, "turn_hold_sd": float(np.std(th)) if th else None,
            "turn_held_mean": float(np.mean(held_t)) if held_t else None,
            "turn_held_sd": float(np.std(held_t)) if held_t else None,
            "cos_hold_mean": float(np.mean([r["cos_hold"] for r in ok])) if ok else None,
            "F_hold_mean": float(np.mean([r["F_pads_hold_N"] for r in ok])) if ok else None,
            "drift_mean_mm": float(np.mean([r["drift_hold_to_end_mm"] for r in ok if r.get("held_end")]))
            if any(r.get("held_end") for r in ok) else None,
            "us_step": float(np.median([r.get("us_per_step_median") or r.get("us_per_world_step") or 0 for r in rs])),
            "seed0": next((r for r in ok if r["seed"] == 0), None) and {
                k: next(r for r in ok if r["seed"] == 0).get(k) for k in ("turn_hold_deg", "held_hold", "cos_hold")},
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    rows = load_rows()
    S = summarize(rows)
    B = bench()
    conds = sorted({c for c, _ in S})
    print(f"{len(rows)} rows, {len(conds)} conditions")
    print("bench:  " + "  ".join(f"{h} {B[h]['n_held']}/{B[h]['n']} {B[h]['held_deg']:4.0f}" for h in HANDS))
    for c in conds:
        cells = []
        for h in HANDS:
            s = S.get((c, h))
            if not s:
                cells.append(f"{h}   --      ")
                continue
            t = s["turn_held_mean"]
            cells.append(f"{h} {s['held_hold']}/{s['n']} {t:+4.0f}" if t is not None else f"{h} {s['held_hold']}/{s['n']}  -- ")
        print(f"{c:52s} " + "  ".join(cells))
    if a.json:
        a.json.write_text(json.dumps({f"{c}||{h}": v for (c, h), v in S.items()}, indent=1, default=float))


if __name__ == "__main__":
    main()
