#!/usr/bin/env python3
"""Simulated ATI calibration sessions for the plan page: the session program on the plan (scripts/ati_calibration_session.py
--sim) for several seeds of the simulated sensor, each analysed (scripts/ati_calibration_analysis.py), the HEX-E reference
comparison on the first (scripts/ati_reference_compare.py) and a summary in
docs/experiments/20261023-ati_ft_calibration/20261023-ati_sim_sessions.json. The sessions' streams stay in logs/.

    .venv/bin/python scripts/ati_sim_sessions.py [--seeds 5] [--mass-g 500.3 --tag "" --rest-s S --line-mm 0.3 --sim-line-mm D --plan P]

With --tag the summary is 20261023-ati_sim_sessions_<tag>.json, the streams go to logs/20261023-ati_sim_<tag>/ and the
HEX-E comparison is left out.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ati_calibration_analysis as AN  # noqa: E402
import ati_reference_compare as RC  # noqa: E402

D = ROOT / "docs/experiments/20261023-ati_ft_calibration"
PLAN = D / "plan/20261023-ati_plan_mass.json"
LOGS = ROOT / "logs/20261023-ati_sim"


def main(seeds, mass_g=500.3, tag="", rest_s=None, line_mm=0.3, sim_line_mm=None, plan=PLAN):
    logs = LOGS.with_name(f"{LOGS.name}_{tag}") if tag else LOGS
    if logs.exists():
        shutil.rmtree(logs)
    out = dict(plan=str(Path(plan).resolve().relative_to(ROOT)), mass_g=mass_g, rest_s=rest_s, line_mm=line_mm, sim_line_mm=sim_line_mm,
               sessions=[])
    first = None
    for s in range(seeds):
        cmd = [sys.executable, str(ROOT / "scripts/ati_calibration_session.py"), "--plan", str(plan), "--mass-g", str(mass_g),
               "--sim", "--sim-seed", str(s), "--logs", str(logs), "--name", f"seed{s}", "--line-mm", str(line_mm)]
        cmd += (["--rest-s", str(rest_s)] if rest_s else []) + (["--sim-line-mm", str(sim_line_mm)] if sim_line_mm else [])
        r = subprocess.run(cmd, capture_output=True, text=True, check=True)
        folder = Path(r.stdout.strip().splitlines()[-1].split("session in ")[-1])
        first = first or folder
        rec = AN.main(folder)
        out["sessions"].append(dict(seed=s, folder=str(folder.relative_to(ROOT)), rests=rec["rests"], pairs=rec["pairs"],
                                    design=rec["design_singular_values"], leave_eye_out=rec["leave_eye_out"],
                                    truth=rec["truth_score"], settle_s=rec["settle_s"]))
    if not tag:
        out["reference_compare"] = RC.run(first)
    (D / f"20261023-ati_sim_sessions{'_' + tag if tag else ''}.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--mass-g", type=float, default=500.3)
    ap.add_argument("--tag", default="")
    ap.add_argument("--rest-s", type=float, default=None)
    ap.add_argument("--line-mm", type=float, default=0.3, help="the line's diameter (a polyester thread or floss)")
    ap.add_argument("--sim-line-mm", type=float, default=None, help="the truth's line in place of --line-mm (a stiff line)")
    ap.add_argument("--plan", default=str(PLAN), help="the plan (the Mini40's: plan/20261023-ati_plan_mass_mini40.json)")
    a = ap.parse_args()
    main(a.seeds, a.mass_g, a.tag, a.rest_s, a.line_mm, a.sim_line_mm, a.plan)
