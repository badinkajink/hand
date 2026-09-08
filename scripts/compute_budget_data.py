#!/usr/bin/env python3
"""Collect training-throughput data for the compute-budget page.

Reads every trainer log in logs/ that got past five PPO iterations, and writes the
per-iteration timing population to compute_budget.json. The DeltaAI side is not
re-derivable from this machine; its numbers are transcribed from the measured tables in
docs/notes/20260820-deltaai_bulk_training_runbook.md (jobs 2989123 / 2989164) and marked
with their provenance.

    python3 scripts/compute_budget_data.py
"""
from __future__ import annotations

import collections
import datetime as dt
import json
import re
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
OUT = ROOT / "docs/experiments/20260908-compute_budget/compute_budget.json"

NUM_STEPS_PER_ENV = 24          # PPO rollout length; one iteration is num_envs * this

RE = {k: re.compile(p) for k, p in {
    "sps": r"Steps per second:\s*([\d.]+)",
    "col": r"Collection time:\s*([\d.]+)s",
    "lea": r"Learning time:\s*([\d.]+)s",
    "itt": r"Iteration time:\s*([\d.]+)s",
    "tot": r"Total steps:\s*(\d+)",
    "run": r"Run name:\s*(\S+)",
    "el":  r"Time elapsed:\s*(\d+):(\d+):(\d+)",
}.items()}
RE_IT = re.compile(r"Learning iteration (\d+)/(\d+)")


def parse(p: Path) -> dict | None:
    txt = p.read_text(errors="ignore")
    sps = [float(x) for x in RE["sps"].findall(txt)]
    if len(sps) < 5:
        return None
    col = [float(x) for x in RE["col"].findall(txt)]
    lea = [float(x) for x in RE["lea"].findall(txt)]
    itt = [float(x) for x in RE["itt"].findall(txt)]
    tot = [int(x) for x in RE["tot"].findall(txt)]
    its = RE_IT.findall(txt)
    el = RE["el"].findall(txt)

    # num_envs is not logged; recover it from the step delta between iterations.
    envs = None
    deltas = [b - a for a, b in zip(tot, tot[1:]) if b > a]
    if deltas:
        d = st.mode(deltas)
        if d % NUM_STEPS_PER_ENV == 0:
            envs = d // NUM_STEPS_PER_ENV

    # Iteration 0 carries kernel compile and CUDA-graph capture; it is not a steady-state
    # sample and is dropped from every median below.
    def body(v):
        return v[1:] if len(v) > 1 else v

    elapsed = None
    if el:
        h, m, s = el[-1]
        elapsed = int(h) * 3600 + int(m) * 60 + int(s)

    return {
        "log": p.name,
        "run": RE["run"].findall(txt)[0] if RE["run"].findall(txt) else None,
        "envs": envs,
        # "Learning iteration N/TOTAL" is 0-indexed against TOTAL total iterations.
        "iters_done": int(its[-1][0]) + 1 if its else len(sps),
        "iters_planned": int(its[0][1]) if its else None,
        "sps_med": round(st.median(body(sps)), 1),
        "sps_min": round(min(body(sps)), 1),
        "sps_max": round(max(body(sps)), 1),
        "col_med": round(st.median(body(col)), 4) if col else None,
        "lea_med": round(st.median(body(lea)), 4) if lea else None,
        "itt_med": round(st.median(body(itt)), 4) if len(itt) > 1 else None,
        "elapsed_s": elapsed,
        "day": dt.datetime.fromtimestamp(p.stat().st_mtime).date().isoformat(),
    }


def quart(v):
    v = sorted(v)
    return {"n": len(v), "min": round(v[0], 1), "q1": round(v[len(v) // 4], 1),
            "med": round(st.median(v), 1), "q3": round(v[3 * len(v) // 4], 1),
            "max": round(v[-1], 1)}


def main() -> None:
    rows = [r for r in (parse(p) for p in sorted(LOGS.glob("*.log"))) if r]

    # ---- throughput population, split by batch width
    by_env = {}
    for e in sorted({r["envs"] for r in rows if r["envs"]}):
        sub = [r for r in rows if r["envs"] == e]
        cols = [r["col_med"] for r in sub if r["col_med"]]
        by_env[e] = {
            **quart([r["sps_med"] for r in sub]),
            "col_med": round(st.median(cols), 4) if cols else None,
            # collection seconds per simulated env-step: the batch-width-free unit
            "us_per_env_step": round(1e6 * st.median(cols) / (e * NUM_STEPS_PER_ENV), 3)
            if cols else None,
        }

    # ---- where an iteration goes
    both = [r for r in rows if r["col_med"] and r["lea_med"]]
    learn_share = [r["lea_med"] / (r["col_med"] + r["lea_med"]) for r in both]
    acc = [r for r in both if r["itt_med"]]
    unacc = [(r["itt_med"] - r["col_med"] - r["lea_med"]) / r["itt_med"] for r in acc]

    # ---- duty cycle: training wall clock against the calendar it was spread over
    days = collections.Counter()
    for r in rows:
        days[r["day"]] += (r["elapsed_s"] or 0) / 3600.0
    d0, d1 = min(days), max(days)
    span_h = ((dt.date.fromisoformat(d1) - dt.date.fromisoformat(d0)).days + 1) * 24
    train_h = sum(days.values())

    # ---- what budgets were actually requested
    budgets = collections.Counter()
    for r in rows:
        if r["iters_planned"] and r["envs"]:
            budgets[(r["iters_planned"], r["envs"])] += 1

    total_steps = sum((r["envs"] or 0) * NUM_STEPS_PER_ENV * r["iters_done"] for r in rows)
    med_sps = by_env[3072]["med"]

    doc = {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "local": {
            "gpu": "NVIDIA GeForce RTX 4070 Ti SUPER, 16 GB",
            "cpu": "Intel Core Ultra 7 265KF, 20 threads",
            "sm_clock_mhz_under_load": 2820,       # nvidia-smi clocks.sm, 2026-09-08
            "sm_clock_max_mhz": 3120,
            "n_logs": len(rows),
            "n_iterations": sum(r["iters_done"] for r in rows),
            "total_env_steps": total_steps,
            "train_hours": round(train_h, 1),
            "span_days": span_h // 24,
            "active_days": len([d for d, h in days.items() if h > 0]),
            "duty_cycle": round(100 * train_h / span_h, 1),
            "first_day": d0, "last_day": d1,
            "by_env": by_env,
            "learn_share_pct": round(100 * st.median(learn_share), 2),
            "learn_share_max_pct": round(100 * max(learn_share), 2),
            "unaccounted_pct": round(100 * st.median(unacc), 2),
            "per_day": [{"day": d, "hours": round(h, 2)} for d, h in sorted(days.items())],
            "budgets": [{"iters": k[0], "envs": k[1], "logs": v,
                         "timesteps_m": round(k[0] * k[1] * NUM_STEPS_PER_ENV / 1e6, 1)}
                        for k, v in budgets.most_common(6)],
        },
        # Transcribed from the measured tables in the runbook; jobs 2989123 / 2989164,
        # 2026-08-20. Not reproducible from this machine.
        "deltaai": {
            "gpu": "NVIDIA GH200 120 GB (Grace-Hopper, aarch64)",
            "sm_clock_mhz_spec": 1980,             # NOT measured on the node
            "source": "docs/notes/20260820-deltaai_bulk_training_runbook.md",
            "sweep": [
                {"envs": 3072, "s_per_iter": 10.85, "note": "10.3-11.4 s range"},
                {"envs": 8192, "s_per_iter": 27.80, "note": ""},
                {"envs": 16384, "s_per_iter": 55.70, "note": ""},
            ],
            "headline_sps": 6950,
            "balance_su": 92, "deposited_su": 100,
            "credits_per_gpu_hour": 143,
        },
        # The reference number the published 0.66x ratio was computed against.
        "published_local_reference_sps": 10530,
        # B33_SEED_BAND.txt, 2026-08-21 — the one study the cluster has actually run.
        "seed_band": {
            "n_draws": 16, "gpu_hours": 12.8, "su_billed": 15,
            "assumed_sd": 0.35, "measured_sd_core": 0.032, "measured_sd_all": 0.110,
            "within_draw_sd": 0.080, "reference_cos": 0.891, "rank": "11/16 at or below",
        },
        # docs/rl/reorientation.md — b33's own 20M run, the one measured convergence point.
        "convergence": {"converged_iter": 26, "budget_iters": 271,
                        "align_it26": 55.9, "align_it134": 55.1, "align_it270": 60.5},
    }

    for e, v in by_env.items():
        v["envs"] = e
    doc["local"]["gh200_ratio_vs_median"] = round(
        doc["deltaai"]["headline_sps"] / med_sps, 3)
    doc["local"]["gh200_ratio_vs_published"] = round(
        doc["deltaai"]["headline_sps"] / doc["published_local_reference_sps"], 3)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1))
    print(f"{len(rows)} logs, {doc['local']['n_iterations']} iterations, "
          f"{total_steps/1e9:.2f} G env-steps, {train_h:.1f} h -> {OUT}")


if __name__ == "__main__":
    main()
