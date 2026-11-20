"""The entries and servers of a series over several plans (2026-11-19 night; the ranked layouts of
docs/experiments/20261119-twist_layout_ranking, the peer's instructions: "Run the 'ranked' and 'anchor' plans first,
three trials each, layouts in a random order").

Reads the trial list (data/20261119-trial_plans.json of the ranking study: each row a plan with its role and rank) or
plan files given directly, keeps the rows of --roles, gives each plan a port from --port0 upward, shuffles the plans with
--seed (printed, so the order can be repeated) and repeats each --trials times in a row (one layout change, one lift and
one free-air sweep per plan). Prints the servers' start commands (hom_serve_budget.sh on the workstation: --task twist,
--d 40, the sphere's 28.8 g, the torque budget 0.56 of the 7.5 V servos, tips off and no filter flags) and the
hom_series.py command; with --go it starts the servers and checks each answered with its "HOM server:" line.

    /usr/bin/python3 series_plan.py [--list ../../20261119-twist_layout_ranking/data/20261119-trial_plans.json]
        [--roles ranked anchor] [--trials 3] [--port0 47102] [--seed N] [--go]
"""
import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
WS = "humanoid@100.65.73.59"
KEY = str(Path.home() / ".ssh/id_wxie")
LIST = ROOT / "docs/experiments/20261119-twist_layout_ranking/data/20261119-trial_plans.json"


def rows_of(path):
    d = json.loads(Path(path).read_text())
    rows = d if isinstance(d, list) else next(v for k, v in d.items() if isinstance(v, list) and v and isinstance(v[0], dict))
    out = []
    for r in rows:
        plan = r.get("plan") or r.get("plan_file") or r.get("file") or r.get("path")
        out.append(dict(plan=plan, role=str(r.get("role", "")), rank=r.get("rank"), label=r.get("label") or r.get("design"),
                        row=r))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("plans", nargs="*", help="plan files instead of --list")
    ap.add_argument("--list", default=str(LIST))
    ap.add_argument("--roles", nargs="+", default=["ranked", "anchor"])
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--port0", type=int, default=47102)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--budget", type=float, default=0.56)
    ap.add_argument("--task", default="twist")
    ap.add_argument("--d", default="40")
    ap.add_argument("--mass-g", default="28.8")
    ap.add_argument("--go", action="store_true", help="start the servers")
    a = ap.parse_args()
    if a.plans:
        rows = [dict(plan=str(Path(p).resolve().relative_to(ROOT)), role="given", rank=None, label=Path(p).stem, row={})
                for p in a.plans]
    else:
        rows = [r for r in rows_of(a.list) if any(r["role"].startswith(x) for x in a.roles)]
    seed = a.seed if a.seed is not None else int(time.strftime("%H%M%S"))
    random.Random(seed).shuffle(rows)
    print(f"{len(rows)} plans, seed {seed}, {a.trials} trials each", flush=True)
    entries = []
    for k, r in enumerate(rows):
        plan = r["plan"] if not str(r["plan"]).startswith("docs/") else str(r["plan"])
        rel = plan if plan.startswith("docs/") else f"docs/experiments/20261115-hom_bench_trials/plans/{Path(plan).name}"
        r["rel"], r["port"] = rel, a.port0 + k
        tag = f"1119rank_{Path(rel).stem.replace('20261119-', '')[:40]}"
        r["cmd"] = (f"bash ~/Programs/hand/logs/20261116-hom_setup/hom_serve_budget.sh {a.task} {a.d} {rel} {r['port']} "
                    f"{tag} {a.mass_g} {a.budget}")
        lab = (r["label"] or Path(rel).stem)[:24].replace(":", "_").replace(" ", "_")
        for t in range(1, a.trials + 1):
            entries.append(f"../plans/{Path(rel).name}:{r['port']}:{r['role'][:6]}{r['rank'] if r['rank'] is not None else ''}_{lab}_t{t}")
        print(f"   {k + 1:2d}. port {r['port']} {r['role']:10s} rank {r['rank']} {rel}", flush=True)
    for r in rows:
        if not (ROOT / r["rel"]).exists():
            sys.exit(f"missing on corvus: {r['rel']} (git pull first)")
    if a.go:
        for r in rows:
            out = subprocess.run(["ssh", "-o", "BatchMode=yes", "-i", KEY, WS, r["cmd"]], capture_output=True, text=True,
                                 timeout=90).stdout
            ok = "HOM server:" in out
            print(f"   port {r['port']}: {'up' if ok else 'NOT UP'} {out.strip().splitlines()[-1][:160] if out.strip() else ''}",
                  flush=True)
            if not ok:
                sys.exit("a server did not start")
    else:
        for r in rows:
            print(f"   ssh {WS} '{r['cmd']}'")
    print("\nentries:\n" + " ".join(entries))
    Path(HERE / "series_entries.txt").write_text(" ".join(entries) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
