#!/usr/bin/env python3
"""A seed registration folder for a new tip's pass on an indenter that is already registered (the lab, 2026-10-28 14:24:
the program resumes a folder that holds only a seed registration.json, with --stages preflight,pad,pivot,register, and
its touch-offs then aim at the indenter where the seed puts it).

The seed takes T_EM, T_MP, K, Kt, T_ES, grav and q_home from the new tip's own registration (--tool: its sphere pass 1),
T_BI from the indenter's registration (--fixture: for the rod, 102701-rodv3id1), and T_BE_home from --fixture as well:
the program's fit adds the homing pose as a soft term, and the seating places the mount, so it holds for the new tip
only while the mount stays on the flange during the swap. --no-home writes T_BE_home as null, and the fit then runs on
the touches alone. The folder must not hold touches.json, pivots.json or events.jsonl: the fit takes every touch in the
folder as one on the current indenter, and events mark actions as done.

    .venv/bin/python scripts/ur5_tip_calibration_seed_registration.py --tool lab/<sphere pass 1>/registration.json \
        --fixture lab/20261028-102701-rodv3id1/registration.json --out lab/<date>-rod-seed-<tip> [--no-home]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

TOOL_KEYS = ("T_EM", "T_MP", "K", "Kt", "T_ES", "grav", "q_home")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tool", required=True, help="the new tip's registration.json (its sphere pass 1)")
    ap.add_argument("--fixture", required=True, help="the indenter's registration.json (its T_BI and T_BE_home)")
    ap.add_argument("--out", required=True, help="the seed folder; it gets registration.json only")
    ap.add_argument("--no-home", action="store_true", help="T_BE_home null: the mount came off the flange during the swap")
    a = ap.parse_args()
    out = Path(a.out)
    bad = [f for f in ("touches.json", "pivots.json", "events.jsonl") if (out / f).exists()]
    if bad:
        sys.exit(f"{out} holds {', '.join(bad)}; a seed folder holds registration.json only")
    tool, fix = json.loads(Path(a.tool).read_text()), json.loads(Path(a.fixture).read_text())
    missing = [k for k in TOOL_KEYS if k not in tool] + [k for k in ("T_BI", "T_BE_home") if k not in fix]
    if missing:
        sys.exit(f"missing keys: {missing}")
    seed = {k: tool[k] for k in TOOL_KEYS}
    seed["T_BI"] = fix["T_BI"]
    seed["T_BE_home"] = None if a.no_home else fix["T_BE_home"]
    seed["notes"] = dict(seed=f"{time.strftime('%Y-%m-%d %H:%M')}, scripts/ur5_tip_calibration_seed_registration.py: "
                         f"T_EM, T_MP, K, Kt, T_ES, grav and q_home from {a.tool}; T_BI from {a.fixture}; T_BE_home "
                         + ("null (--no-home)" if a.no_home else f"from {a.fixture}"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "registration.json").write_text(json.dumps(seed, indent=1) + "\n")
    print("->", out / "registration.json")


if __name__ == "__main__":
    main()
