"""S2's data for the workstation's fits (W2), in the layout of 2026-11-06's identification
(docs/experiments/20261106-sr2_servo_sysid/data/), so scripts/hardware/sr2_servo_sysid_fit.py reads them unchanged.

    /usr/bin/python3 docs/experiments/20261117-sr2_system_identification/lab/s2_package.py [OUT]

From OUT (default logs/20261117-sr2_sysid_7v5, s2_chain.sh's folder) into data/s2/ of this study:
  dwells.tsv, holds_c.tsv            sysid_tables.py's tables (one row per dwell; the c blocks' holds)
  20261117-sysid_plan_steps.tsv      every step of the six plans: block, step, family, finger, joint, dir, command,
                                     dwell, speed, and the CAD clearance at the pose and in the move's box (mm)
  20261117-sysid_clearance_at_mounts.json   make_sysid_plan_at.py's check at the station's mounts
  supply_diag.txt                    hom_setup.py diag's lines around each block (each servo's voltage, temperature,
                                     torque and status read through the station), from chain.out
  <block>_events.jsonl               run_plan.py's step events, when under 2 MB (the telemetry stays in OUT)
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "logs/20261117-sr2_sysid_7v5"
DST = Path(__file__).resolve().parents[1] / "data/s2"
BLOCKS = ("a1", "b1", "a2", "b2", "c", "c40")
DST.mkdir(parents=True, exist_ok=True)
for name in ("dwells.tsv", "holds_c.tsv"):
    src = OUT / "tables" / name
    if src.exists():
        shutil.copy(src, DST / name)
        print(f"{name}: {sum(1 for _ in src.open()) - 1} rows")
rows = ["block\tstep\tfamily\tfinger\tjoint\tdir\tcmd_deg\tdwell_s\tspeed\tclear_pose_mm\tclear_box_mm"]
for b in BLOCKS:
    p = OUT / "plan" / f"20261106-sysid_{b}.json"
    if not p.exists():
        continue
    for i, s in enumerate(json.loads(p.read_text())["steps"]):
        rows.append("\t".join(str(v) for v in (b, i, s["family"], s["finger"], s["joint"], s["dir"], s["cmd"],
                                                s["dwell_s"], s["speed"], s.get("clearance_mm"),
                                                s.get("clearance_box_mm"))))
(DST / "20261117-sysid_plan_steps.tsv").write_text("\n".join(rows) + "\n")
print(f"plan steps: {len(rows) - 1}")
chk = OUT / "plan" / "20261117-sysid_clearance_at_mounts.json"
if chk.exists():
    shutil.copy(chk, DST / chk.name)
lines = [x for x in (OUT / "chain.out").read_text().splitlines()
         if any(k in x for k in ("block", "volts", "temperature", "torque_enable", "rebooted", "S2", "waiting", "cooled"))]
(DST / "supply_diag.txt").write_text("\n".join(lines) + "\n")
print(f"supply_diag.txt: {len(lines)} lines")
for b in BLOCKS:
    ev = OUT / b / "events.jsonl"
    if ev.exists() and ev.stat().st_size < 2e6:
        shutil.copy(ev, DST / f"{b}_events.jsonl")
        print(f"{b}_events.jsonl: {ev.stat().st_size / 1e3:.0f} kB")
    elif ev.exists():
        print(f"{b}: events.jsonl {ev.stat().st_size / 1e6:.1f} MB, left in {OUT}")
