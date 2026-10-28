#!/usr/bin/env python3
"""A fixture's frame I in the UR5's base (T_BI) from a hand seating on its key: T_BI = T_BE_seat T_EM T_IM_home^-1, with
T_IM_home from a plan of that fixture and T_EM from a registration of the tool. Writes a folder with a registration.json
holding that T_BI, which ur5_tip_calibration_holder_plan.py takes as an obstacle (--add-other KIND:FOLDER). The seating
comes from the lab's poses.jsonl in the repository, or from corvus's working copy (read-only, over Tailscale with the
password in .dotenv) when it has not been pushed yet.

    .venv/bin/python scripts/ur5_fixture_from_seating.py rod_indenter_seated \\
        --plan docs/experiments/20261022-ur5_tip_calibration/lab/20261027-plan_rod_lab_r27_pass1_v2id1.json \\
        --out docs/experiments/20261022-ur5_tip_calibration/lab/20261027-rod-seed
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "docs/experiments/20261022-ur5_tip_calibration/lab"
POSES = "docs/experiments/20261022-ur5_tip_calibration/lab/poses.jsonl"


def seating(label):
    rows = [json.loads(s) for s in (ROOT / POSES).read_text().splitlines() if s.strip()]
    hit = [r for r in rows if r.get("label") == label]
    if hit:
        return hit[-1], "repository"
    env = dict(os.environ, SSH_ASKPASS=str(Path.home() / ".local/bin/corvus-askpass"), SSH_ASKPASS_REQUIRE="force")
    out = subprocess.run(["ssh", "-o", "PubkeyAuthentication=no", "-o", "PreferredAuthentications=password,keyboard-interactive",
                          "-o", "ConnectTimeout=10", "will@100.93.255.76", f"cat ~/workspace/hand/{POSES}"],
                         env=env, capture_output=True, text=True, timeout=30, stdin=subprocess.DEVNULL).stdout
    hit = [json.loads(s) for s in out.splitlines() if s.strip() and f'"{label}"' in s]
    if not hit:
        raise SystemExit(f"no seating labelled {label!r} in the repository or on corvus")
    return hit[-1], "corvus"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("label")
    ap.add_argument("--plan", required=True, help="a plan of the fixture: its geometry.T_IM_home")
    ap.add_argument("--registration", default=str(LAB / "20261027-holder-seed-seat2/registration.json"),
                    help="the tool's registration: its T_EM")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    r, src = seating(a.label)
    T_BE = np.array(r["T_tcp_mm"], float)
    T_EM = np.array(json.loads(Path(a.registration).read_text())["T_EM"], float)
    T_IM = np.array(json.loads(Path(a.plan).read_text())["geometry"]["T_IM_home"], float)
    T_BI = T_BE @ T_EM @ np.linalg.inv(T_IM)
    yaw = np.degrees(np.arctan2(T_BI[1, 0], T_BI[0, 0]))
    tilt = np.degrees(np.arccos(np.clip(T_BI[2, 2], -1.0, 1.0)))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "registration.json").write_text(json.dumps(dict(
        T_BI=T_BI.tolist(), notes=dict(seating=dict(label=a.label, time=r.get("time"), source=src,
                                                       plan=str(Path(a.plan).name), registration=str(a.registration))),
    ), indent=1) + "\n")
    print(f"{a.label} ({r.get('time')}, {src}): origin {T_BI[:3, 3].round(1)} mm, yaw {yaw:.2f} deg, tilt {tilt:.2f} deg"
          f" -> {out}")


if __name__ == "__main__":
    sys.exit(main())
