#!/usr/bin/env python3
"""Simulated ATI calibration sessions for the plan page: the session program on the plan (scripts/ati_calibration_session.py
--sim) for several seeds of the simulated sensor, each analysed (scripts/ati_calibration_analysis.py), the HEX-E reference
comparison on the first (scripts/ati_reference_compare.py) and a summary in
docs/experiments/20261023-ati_ft_calibration/20261023-ati_sim_sessions.json. The sessions' streams stay in logs/.

    .venv/bin/python scripts/ati_sim_sessions.py [--seeds 5]
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


def main(seeds):
    if LOGS.exists():
        shutil.rmtree(LOGS)
    out = dict(plan=str(PLAN.relative_to(ROOT)), sessions=[])
    first = None
    for s in range(seeds):
        r = subprocess.run([sys.executable, str(ROOT / "scripts/ati_calibration_session.py"), "--plan", str(PLAN),
                            "--mass-g", "500.3", "--sim", "--sim-seed", str(s), "--logs", str(LOGS), "--name", f"seed{s}"],
                           capture_output=True, text=True, check=True)
        folder = Path(r.stdout.strip().splitlines()[-1].split("session in ")[-1])
        first = first or folder
        rec = AN.main(folder)
        out["sessions"].append(dict(seed=s, folder=str(folder.relative_to(ROOT)), rests=rec["rests"], pairs=rec["pairs"],
                                    design=rec["design_singular_values"], leave_eye_out=rec["leave_eye_out"],
                                    truth=rec["truth_score"], settle_s=rec["settle_s"]))
    out["reference_compare"] = RC.run(first)
    (D / "20261023-ati_sim_sessions.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=5)
    main(ap.parse_args().seeds)
