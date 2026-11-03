"""A session folder for the rod (2026-11-03), seeded from the 2 mm sphere's session of the same seat: its registration
(the strip's frame T_BI, the pad's fitted T_AP, T_EM, the homing pose and joints) with the probe's centre c_M set to the
plan's (the rod's axis point under the tool's axis), no touches, and the homing marked done, so that a run with
--resume OUT --stages preflight,pad,register,grid keeps the pads where the 2 mm sphere's pass 1 found them and fits only
the rod across its axis (the program's fit_rod_registration). The tips must stay seated through the tool change.

    /usr/bin/python3 docs/experiments/20261022-ur5_tip_calibration/lab/probe_seed_session.py SRC_SESSION PLAN OUT [--keep-probe]

--keep-probe (11-03, the 2 mm sphere's sweep test): the same tool as the source session's, so its fitted c_M is kept
too, and the run needs no pass 1 (--stages preflight,grid). The tools must match.
"""
import json
import sys
import time
from pathlib import Path

keep = "--keep-probe" in sys.argv[1:]
args = [x for x in sys.argv[1:] if x != "--keep-probe"]
src, plan_f, out = Path(args[0]), Path(args[1]), Path(args[2])
plan = json.loads(plan_f.read_text())
probe = plan["geometry"]["probe"]
reg = json.loads((src / "registration.json").read_text())
if reg.get("T_AP") is None or reg.get("T_BI") is None:
    sys.exit(f"{src}: its registration has no T_AP or T_BI")
plan_src = json.loads((src / "plan.json").read_text())
seat_src = plan_src["geometry"]["seat"]["index"]
if seat_src != plan["geometry"]["seat"]["index"]:
    sys.exit(f"{src} is seat {seat_src + 1}'s session; the plan is seat {plan['geometry']['seat']['index'] + 1}'s")
if keep and plan_src["geometry"]["probe"].get("tool") != probe.get("tool"):
    sys.exit(f"--keep-probe: {src} used {plan_src['geometry']['probe'].get('tool')}, the plan {probe.get('tool')}")
out.mkdir(parents=True, exist_ok=False)
if not keep:
    reg["c_M"] = list(probe["centre_M"])
reg["notes"] = dict(reg.get("notes", {}), seeded=dict(from_session=str(src), at=time.strftime("%Y-%m-%d %H:%M:%S"),
                                                      c_M_from_plan=None if keep else list(probe["centre_M"]),
                                                      c_M_kept=reg["c_M"] if keep else None, probe=probe.get("tool")))
(out / "registration.json").write_text(json.dumps(reg, indent=1))
(out / "touches.json").write_text("[]")
(out / "plan.json").write_text(json.dumps(plan, indent=1))
(out / "events.jsonl").write_text(json.dumps(dict(t=time.time(), event="end", id="homing", status="ok",
                                                  rec=dict(seeded_from=str(src)))) + "\n")
print(f"{out}: seeded from {src} (seat {seat_src + 1}), c_M {reg['c_M']} ({'kept, ' if keep else ''}{probe.get('kind')}, "
      f"R {probe['radius']})")
