#!/usr/bin/env python3
"""What each SHIPPED plan does as its residual clip moves -- the band, on its own cell.

    python3 scripts/real_v1_plan_band.py \
        --deploy-dir docs/experiments/20260829-real_v1_deploy/deploy \
        --budgets 0.40,2.00,0.05 --reps 4 --out band.json

A design does not have a clip that works and clips that do not.  It has a contiguous BAND of
clips inside which it keeps the tool, and it drops on BOTH sides: below, the clipped turn stops
while the shaft is still rotating and the fingers -- which pay for the turn in extension -- run
out of commanded travel; above, the turn overdrives and ejects.  Alignment falls monotonically
across the band, so the best plan sits at its LOWER edge, plus enough margin to absorb the 4-6
deg of yaw droop the bench measured under load.

This differs from `probe_hold_convergence.py`, which re-clips a saved *carry* plan: here the
trajectory is rebuilt from each deploy plan's own metadata, so the band is measured on the cell
the bench is actually running, and the rebuild is checked against the shipped set-points before
a single rollout is scored.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from morphohand.tools.keyframe_ik import FINGERS  # noqa: E402

JOINTS = [j for js in FINGERS.values() for j in js]


def _depth_req(meta: dict, design: str) -> float | None:
    """What the export ASKED the fitter for -- not what it got.

    `meta.grip_depth_mm` is the achieved depth, and the fitter snaps: asking sv1_u0060 for its
    table depth of 64.5 mm lands at 66.5, and asking for 66.5 lands at 67.5, a different grasp
    half a degree away in every joint. The request lives in the design table, exactly where
    `real_v1_export_plan.py` reads it.
    """
    # Plans exported before 2026-08-30 do not record their table; they all came from the
    # default one, which is what `real_v1_export_plan.py` still falls back to.
    table = Path(meta.get("design_table")
                 or "docs/experiments/20260828-real_v1_search/table.json")
    if not table.is_absolute():
        table = ROOT / table
    row = {r["design"]: r for r in json.loads(table.read_text())}[design]
    depth = row.get("depth_req_mm")
    if depth is None and meta.get("bench_height_mm", 0.0) > 0:
        depth = row.get("depth_fit_mm")
    return None if depth is None else depth / 1000.0


def _build(meta: dict, budget: float, design: str):
    import real_v1_deploy_envelope as de
    return de.make_plan(
        Path(meta["scene"]), straddle=meta["straddle_mm"] / 1000.0,
        depth=_depth_req(meta, design), thumb_axial=meta["thumb_axial_mm"] / 1000.0,
        squeeze=meta["squeeze_mm"] / 1000.0, axis_k=meta["axis_k"],
        angle_deg=meta["angle_deg"], lift=meta.get("palm_lift_mm", 0.0) / 1000.0,
        budget=budget, turn_steps=meta["turn_steps"], hold_squeeze=0.0,
        bench=meta.get("bench_height_mm", 0.0) > 0)


def _job(a):
    import real_v1_deploy_envelope as de
    tag, meta, design, budget, rep, hold = a
    plan = _build(meta, budget, design)
    if plan is None:
        return {"plan": tag, "budget": budget, "rep": rep, "ok": False, "final_cos": 0.0,
                "peak_cos": 0.0, "error": "no reachable pose"}
    short, worst = de.servo_shortfall(Path(meta["scene"]), plan, budget)
    r = de.execute(Path(meta["scene"]), plan, hold_steps=hold, seed=rep)
    return {"plan": tag, "budget": budget, "rep": rep, "ok": bool(r["ok"]),
            "final_cos": r["final_cos"], "peak_cos": r["peak_cos"],
            "final_z": r["final_z"], "lifted_z": r["lifted_z"],
            "contacts": r["contacts_hand"], "force_N": r["force_hand_N"],
            "servo_short_deg": short, "servo_worst": worst}


def _check_rebuild(tag: str, shipped: dict, meta: dict) -> str:
    """The rebuild has to land on the SHIPPED set-points, or the band is another hand's."""
    plan = _build(meta, meta["budget_rad"], shipped["design"])
    if plan is None:
        return "REBUILD FAILED"
    grip = {f: {j.rpartition("_")[2]: np.degrees(plan["anchor"][j]) for j in js}
            for f, js in FINGERS.items()}
    want = next(p for p in shipped["poses"] if p["name"] == "grip")["joints"]
    err = max(abs(grip[f][j] - want[f][j]) for f in want for j in want[f])
    return f"grip matches to {err:.3f} deg" if err < 0.05 else f"GRIP DIFFERS by {err:.2f} deg"


def _servo_job(a):
    import real_v1_deploy_envelope as de
    tag, meta, design, budget = a
    plan = _build(meta, budget, design)
    if plan is None:
        return tag, budget, 0.0, ""
    short, worst = de.servo_shortfall(Path(meta["scene"]), plan, budget)
    return tag, budget, short, worst


def _servo_only(a, metas, shipped, budgets) -> int:
    """Rewrite servo_short_deg in place. Everything else in the file is a rollout result and
    is left exactly as it was measured."""
    if not a.out or not a.out.exists():
        print("--servo-only needs an existing --out to update"); return 1
    doc = json.loads(a.out.read_text())
    jobs = [(t, metas[t], shipped[t]["design"], b) for t in metas for b in budgets]
    print(f"{len(jobs)} plan rebuilds, no rollouts", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        out = list(ex.map(_servo_job, jobs))
    short = {(t, b): (s, w) for t, b, s, w in out}
    for row in doc["rows"]:
        key = (row["plan"], row["budget"])
        if key in short:
            row["servo_short_deg"], row["servo_worst"] = short[key]
    print(f"\n{'plan':22s} {'was blocked at':>32s} -> {'now blocked at':>32s}")
    for tag, s in doc["summary"].items():
        was = [b for b in budgets if s["servo_short_deg"].get(str(b), 0.0) > 0.0]
        s["servo_short_deg"] = {str(b): short.get((tag, b), (0.0, ""))[0] for b in budgets}
        now = [b for b in budgets if s["servo_short_deg"][str(b)] > 0.0]
        if was != now:
            fmt = lambda v: (f"{min(v):.2f}+" if v else "nothing")
            print(f"{tag:22s} {fmt(was):>32s} -> {fmt(now):>32s}")
    a.out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"\nrewrote {a.out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deploy-dir", type=Path, required=True)
    ap.add_argument("--plans", default=None, help="comma list of tags; default every plan")
    ap.add_argument("--budgets", default="0.40,2.00,0.05", help="lo,hi,step in radians")
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--hold-steps", type=int, default=4800,
                    help="9.6 s. The screen's 800 (1.6 s) is a snapshot part-way through a "
                         "fall: every Sobol-128 finalist that 'held' at 800 was on its way out.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--servo-only", action="store_true",
                    help="recompute only servo_short_deg into an EXISTING --out, no rollouts. "
                         "The shortfall is kinematics against servos.FINGER_JOINTS, so when "
                         "that table moves (the aa cap went +-70 -> +-85 on 2026-08-31) the "
                         "recorded band is still valid and only this column is stale.")
    a = ap.parse_args()

    lo, hi, step = (float(v) for v in a.budgets.split(","))
    budgets = [round(lo + step * i, 3) for i in range(int(round((hi - lo) / step)) + 1)]
    wanted = set(a.plans.split(",")) if a.plans else None

    metas, shipped = {}, {}
    for p in sorted(a.deploy_dir.glob("*_plan.json")):
        tag = p.name.removesuffix("_plan.json")
        if wanted and tag not in wanted:
            continue
        d = json.loads(p.read_text())
        if "budget_rad" not in d["meta"]:
            print(f"{tag:22s} skipped: no meta.budget_rad, so its own clip is unknown")
            continue
        metas[tag], shipped[tag] = d["meta"], d

    for tag in metas:
        print(f"{tag:22s} b{metas[tag]['budget_rad']:<5} {_check_rebuild(tag, shipped[tag], metas[tag])}",
              flush=True)

    if a.servo_only:
        return _servo_only(a, metas, shipped, budgets)

    jobs = [(t, metas[t], shipped[t]["design"], b, r, a.hold_steps)
            for t in metas for b in budgets for r in range(a.reps)]
    print(f"\n{len(jobs)} rollouts: {len(metas)} plans x {len(budgets)} budgets x {a.reps} reps",
          flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(_job, jobs))

    print(f"\n{'plan':22s} {'own':>5s}  " + "".join(f"{b:5.2f}" for b in budgets))
    for tag in metas:
        line = f"{tag:22s} {metas[tag]['budget_rad']:5.2f}  "
        for b in budgets:
            g = [r for r in rows if r["plan"] == tag and r["budget"] == b]
            kept = sum(1 for r in g if r["ok"])
            line += f"{kept if kept else '.':>5}"
        print(line)
    print("\n(cells are rollouts KEPT out of "
          f"{a.reps}; '.' = every rep dropped the tool)")

    print(f"\n{'plan':22s} {'band (rad)':>14s} {'best cos':>9s} {'at':>6s} "
          f"{'own-clip cos':>13s} {'own-clip kept':>14s}")
    summary = {}
    for tag in metas:
        held = [b for b in budgets
                if all(r["ok"] for r in rows if r["plan"] == tag and r["budget"] == b)]
        band = None
        if held:                       # the LONGEST contiguous run, not merely the extremes
            runs, cur = [], [held[0]]
            for b in held[1:]:
                if abs(b - cur[-1] - step) < 1e-6:
                    cur.append(b)
                else:
                    runs.append(cur)
                    cur = [b]
            runs.append(cur)
            band = max(runs, key=len)
        cos = {b: float(np.mean([r["final_cos"] for r in rows
                                 if r["plan"] == tag and r["budget"] == b])) for b in budgets}
        best = max(band, key=lambda b: cos[b]) if band else None
        own = metas[tag]["budget_rad"]
        own_rows = [r for r in rows if r["plan"] == tag and abs(r["budget"] - own) < 1e-9]
        summary[tag] = {
            "budget_rad": own,
            "band_rad": [band[0], band[-1]] if band else None,
            "best_cos": round(cos[best], 3) if best else None,
            "best_at": best,
            "own_cos": round(float(np.mean([r["final_cos"] for r in own_rows])), 3)
            if own_rows else None,
            "own_kept": [sum(1 for r in own_rows if r["ok"]), len(own_rows)],
            "cos_by_budget": {str(b): round(cos[b], 3) for b in budgets},
            "servo_short_deg": {str(b): max((r["servo_short_deg"] for r in rows
                                             if r["plan"] == tag and r["budget"] == b),
                                            default=0.0) for b in budgets},
        }
        s = summary[tag]
        band_s = f"{band[0]:.2f}-{band[-1]:.2f}" if band else "none"
        best_s = f"{s['best_cos']:.3f}" if best else "--"
        at_s = f"{best:.2f}" if best else "--"
        own_s = f"{s['own_cos']:.3f}" if s["own_cos"] is not None else "--"
        kept_s = f"{s['own_kept'][0]}/{s['own_kept'][1]}"
        print(f"{tag:22s} {band_s:>14s} {best_s:>9s} {at_s:>6s} {own_s:>13s} {kept_s:>14s}")

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1) + "\n")
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
