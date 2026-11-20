"""The entries of a series over several plans (2026-11-19 night; the ranked layouts of
docs/experiments/20261119-twist_layout_ranking, the peer's instructions: "Run the 'ranked' and 'anchor' plans first,
three trials each, layouts in a random order").

Reads the trial list (data/20261119-trial_plans.json of the ranking study: {"plans": [{"plan", "role", "xy", "k",
"rank_among_deployable", "prediction", ["prm"]}, ...]}) or plan files given directly, keeps the rows of --roles and
--forces (default: the plans with HOM's default forces; 'refined' the stage-E plans that carry meta.hom.prm), gives each
plan a port from --port0 upward, shuffles the plans with --seed (printed, so the order can be repeated) and repeats each
--trials times in a row (one layout change, one lift and one free-air sweep per plan). Writes the entries to
series_entries.txt and prints the hom_series.py command, which starts each plan's server on the workstation before its
first trial and stops it after its last (--serve: --task twist, --d 40, the sphere's 28.8 g, the torque budget 0.56 of
the 7.5 V servos, tips off and no filter flags).

    /usr/bin/python3 series_plan.py [--list ../../20261119-twist_layout_ranking/data/20261119-trial_plans.json]
        [--roles ranked anchor] [--forces default|refined|both] [--trials 3] [--port0 47102] [--seed N]
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
LIST = ROOT / "docs/experiments/20261119-twist_layout_ranking/data/20261119-trial_plans.json"


def rows_of(path):
    d = json.loads(Path(path).read_text())
    rows = d if isinstance(d, list) else d.get("plans") or next(
        v for v in d.values() if isinstance(v, list) and v and isinstance(v[0], dict))
    out = []
    for r in rows:
        if r.get("error"):
            print(f"   skipped (no plan built): {r.get('plan')}: {r['error'][-120:]}", flush=True)
            continue
        plan = r.get("plan") or r.get("plan_file") or r.get("file") or r.get("path")
        rank = r.get("rank") if r.get("rank") is not None else r.get("rank_among_deployable")
        xy, k = r.get("xy"), r.get("k")
        label = r.get("label") or (f"x{xy[0]:g}y{xy[1]:g}g{k}" if xy and k is not None else None)
        if r.get("prm"):
            label = (label or "") + "f"
        out.append(dict(plan=plan, role=str(r.get("role", "")), rank=rank, label=label, refined=bool(r.get("prm")),
                        row=r))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("plans", nargs="*", help="plan files instead of --list")
    ap.add_argument("--list", default=str(LIST))
    ap.add_argument("--roles", nargs="+", default=["ranked", "anchor"])
    ap.add_argument("--forces", choices=("default", "refined", "both"), default="default")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--port0", type=int, default=47102)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--serve", default="twist 40 28.8 0.56", help="hom_series.py --serve: TASK D MASS_G BUDGET")
    ap.add_argument("--video-dir", default=None)
    a = ap.parse_args()
    if a.plans:
        rows = [dict(plan=str(Path(p).resolve().relative_to(ROOT)), role="given", rank=None, label=Path(p).stem,
                     refined=False, row={}) for p in a.plans]
    else:
        rows = [r for r in rows_of(a.list) if any(r["role"].startswith(x) for x in a.roles)
                and (a.forces == "both" or r["refined"] == (a.forces == "refined"))]
    seed = a.seed if a.seed is not None else int(time.strftime("%H%M%S"))
    random.Random(seed).shuffle(rows)
    print(f"{len(rows)} plans, seed {seed}, {a.trials} trials each", flush=True)
    entries = []
    for k, r in enumerate(rows):
        plan = str(r["plan"])
        rel = plan if plan.startswith("docs/") else f"docs/experiments/20261115-hom_bench_trials/plans/{Path(plan).name}"
        r["rel"], r["port"] = rel, a.port0 + k
        lab = (r["label"] or Path(rel).stem)[:24].replace(":", "_").replace(" ", "_")
        tag = f"{r['role'][:6]}{r['rank'] if r['rank'] is not None else ''}"
        for t in range(1, a.trials + 1):
            entries.append(f"../plans/{Path(rel).name}:{r['port']}:{tag}_{lab}_t{t}")
        pred = (r["row"].get("prediction") or {})
        tr = [((pred.get(m) or {}).get("track_seeds012")) for m in ("point", "pads")]
        print(f"   {k + 1:2d}. port {r['port']} {r['role']:7s} rank {str(r['rank']):>4s} {lab:14s} predicted "
              f"{tr} {Path(rel).name}", flush=True)
    missing = [r["rel"] for r in rows if not (ROOT / r["rel"]).exists()]
    if missing:
        sys.exit(f"missing on corvus: {missing} (git pull first)")
    Path(HERE / "series_entries.txt").write_text(" ".join(entries) + "\n")
    vd = a.video_dir or f"/home/will/workspace/hand/logs/20261116-hom_trials/{time.strftime('%Y%m%d-%H%M')}_session"
    print(f"\n{len(entries)} entries -> series_entries.txt\n\n/usr/bin/python3 hom_series.py --object sphere --map "
          f"../maps/20261116-sphere_fitted_map_O.json --task twist --video-dir {vd} --out "
          f"{ROOT}/logs/20261116-hom_trials/{time.strftime('%Y%m%d')}-series_rank_seed{seed}.jsonl --sweep --pickup "
          f"--serve '{a.serve}' --ready-args='--watch-mm 3' --putdown-args='--level-frac 1 --level-max 35' "
          f"$(cat series_entries.txt) --go")
    return 0


if __name__ == "__main__":
    sys.exit(main())
