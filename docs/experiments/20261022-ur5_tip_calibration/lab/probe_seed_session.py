"""A session folder for the rod (2026-11-03), seeded from the 2 mm sphere's session of the same seat: its registration
(the strip's frame T_BI, the pad's fitted T_AP, T_EM, the homing pose and joints) with the probe's centre c_M set to the
plan's (the rod's axis point under the tool's axis), no touches, and the homing marked done, so that a run with
--resume OUT --stages preflight,pad,register,grid keeps the pads where the 2 mm sphere's pass 1 found them and fits only
the rod across its axis (the program's fit_rod_registration). The tips must stay seated through the tool change.

    /usr/bin/python3 docs/experiments/20261022-ur5_tip_calibration/lab/probe_seed_session.py SRC_SESSION PLAN OUT
"""
import json
import sys
import time
from pathlib import Path

src, plan_f, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
plan = json.loads(plan_f.read_text())
probe = plan["geometry"]["probe"]
reg = json.loads((src / "registration.json").read_text())
if reg.get("T_AP") is None or reg.get("T_BI") is None:
    sys.exit(f"{src}: its registration has no T_AP or T_BI")
seat_src = json.loads((src / "plan.json").read_text())["geometry"]["seat"]["index"]
if seat_src != plan["geometry"]["seat"]["index"]:
    sys.exit(f"{src} is seat {seat_src + 1}'s session; the plan is seat {plan['geometry']['seat']['index'] + 1}'s")
out.mkdir(parents=True, exist_ok=False)
reg["c_M"] = list(probe["centre_M"])
reg["notes"] = dict(reg.get("notes", {}), seeded=dict(from_session=str(src), at=time.strftime("%Y-%m-%d %H:%M:%S"),
                                                      c_M_from_plan=list(probe["centre_M"]), probe=probe.get("tool")))
(out / "registration.json").write_text(json.dumps(reg, indent=1))
(out / "touches.json").write_text("[]")
(out / "plan.json").write_text(json.dumps(plan, indent=1))
(out / "events.jsonl").write_text(json.dumps(dict(t=time.time(), event="end", id="homing", status="ok",
                                                  rec=dict(seeded_from=str(src)))) + "\n")
print(f"{out}: seeded from {src} (seat {seat_src + 1}), c_M {probe['centre_M']} ({probe.get('kind')}, R {probe['radius']})")
