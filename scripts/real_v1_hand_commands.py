#!/usr/bin/env python3
"""Check the sim's hand design space against what the real gantries reach, and emit the firmware
commands that put the machine into a given design.

    uv run python scripts/real_v1_hand_commands.py --audit
    uv run python scripts/real_v1_hand_commands.py --audit --travel 1:59.8,3:59.8,5:59.8
    uv run python scripts/real_v1_hand_commands.py --plan docs/.../deploy/g12_plan.json

WHY

`morphohand.sampling.morphology.REAL_V1_WORKSPACE` and
`manta_hand.kinematics.FULL_EXTENSION_MM` are two independent statements about the same three
rails, written months apart from the same drawing, and nothing had ever compared them. The sim
declares a +-30 mm box in x for every finger. The rails do not deliver it: the three firmware
"y" axes (J1/J3/J5, which drive local x) measure 56.2 / 56.0 / 54.1 mm of travel against a
60 mm nominal, and the shortfall lands on exactly the axis the compactness knobs push against.

    --audit    prints both boxes, what the gap costs in designs, and the joint-name map
    --travel   re-runs the same audit under a HYPOTHETICAL travel, e.g. if a ruler check says
               J1/J3/J5 really do reach 59.8 mm and the current numbers are a scale error
    --plan     turns an exported plan into MOVEMM lines and a servo set-point table

This script is the workstation half; it needs mujoco + morphohand. The half that runs on the
CB1 is `manta_hand.plan`, which needs neither.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src/morphohand/driver/manta/host"))

import mujoco  # noqa: E402

from manta_hand.kinematics import FINGER_GEOMETRY, FULL_EXTENSION_MM, STEPPER_JOINTS  # noqa: E402
from manta_hand.plan import (  # noqa: E402
    FINGER_ID, SIM_JOINT_TO_SERVO, HandPlan, local_envelope, mount_violations, travel_audit,
)
from manta_hand.servos import FINGER_JOINTS  # noqa: E402
from morphohand.sampling.morphology import REAL_V1_MOUNTS, REAL_V1_WORKSPACE  # noqa: E402

BASE_HAND = ROOT / "assets/mjcf/real_v1/real_hand.xml"
FINGERS = ("thumb", "index", "middle")


def _box(finger: str):
    return getattr(REAL_V1_WORKSPACE, finger)


def check_origins() -> list[str]:
    """The one thing that makes the two frames the same frame: the mount origins."""
    out = ["MOUNT ORIGINS -- sim REAL_V1_MOUNTS vs firmware FINGER_GEOMETRY (mm)",
           f"  {'finger':7} {'sim':>18} {'firmware':>18}   verdict"]
    ok = True
    for f in FINGERS:
        sx, sy = (v * 1000 for v in REAL_V1_MOUNTS[f])
        fx, fy = FINGER_GEOMETRY[FINGER_ID[f]]["origin"]
        same = abs(sx - fx) < 1e-6 and abs(sy - fy) < 1e-6
        ok &= same
        out.append(f"  {f:7} ({sx:7.1f},{sy:7.1f}) ({fx:7.1f},{fy:7.1f})   "
                   f"{'identical' if same else 'MISMATCH'}")
    out.append("  => the sim's palm frame IS {P}: no rotation, no flip, mm for mm."
               if ok else "  => THE FRAMES DISAGREE. Nothing below is trustworthy.")
    return out


def check_joint_map() -> list[str]:
    """Identify each sim joint with a servo by its range, which is a unique fingerprint here."""
    m = mujoco.MjModel.from_xml_path(str(BASE_HAND))
    out = ["JOINT MAP -- sim limits vs the servo contract they match",
           f"  {'sim joint':13} {'sim range (deg)':>18}  {'servo':>10} {'declared':>16} "
           f"{'measured':>18}"]
    declared = {"aa": (-85.0, 85.0), "fe1": (-15.0, 92.0), "fe2": (-18.0, 92.0)}
    for f in FINGERS:
        for sj, servo in SIM_JOINT_TO_SERVO.items():
            jid = m.joint(f"{f}_{sj}").id
            lo, hi = (math.degrees(v) for v in m.jnt_range[jid])
            sid, _zero, (rlo, rhi) = FINGER_JOINTS[FINGER_ID[f]][servo]
            dlo, dhi = declared[servo]
            mark = "" if abs(lo - dlo) < 0.01 and abs(hi - dhi) < 0.01 else "   <-- NOT the contract"
            out.append(f"  {f + '_' + sj:13} [{lo:7.2f},{hi:7.2f}]  {servo:>6}(id{sid}) "
                       f"[{dlo:6.1f},{dhi:6.1f}] [{rlo:7.2f},{rhi:7.2f}]{mark}")
    out += ["  => three distinct declared ranges, three exact matches: the identification is",
            "     forced. SIGNS are a separate question -- see manta_hand.plan's docstring."]
    return out


def check_designs(travel: dict[int, float] | None) -> list[str]:
    import real_v1_design_search as ds

    designs = ds.design_set("all")
    idx = {"thumb": (0, 1), "index": (3, 4), "middle": (6, 7)}
    bad = []
    for name, vec in designs.items():
        v = []
        for f, (xi, yi) in idx.items():
            v += mount_violations(f, vec[xi] * 1000, vec[yi] * 1000,
                                  frame="local", travel_mm=travel)
        if v:
            bad.append((max(x.short for x in v), name, v))
    out = [f"DESIGN REACHABILITY -- {len(designs) - len(bad)}/{len(designs)} of the search set fits"]
    if not bad:
        out.append("  every sampled design is reachable")
    for short, name, v in sorted(bad, reverse=True):
        out.append(f"  {name:14} needs {short:4.1f} mm more travel   "
                   + "; ".join(f"{x.finger}_{x.axis} {x.value:+.1f}" for x in v))
    return out


def audit(travel: dict[int, float] | None) -> str:
    blocks = [check_origins(), [], check_joint_map(), []]
    blocks += [["TRAVEL ENVELOPE -- the sim's declared design box vs the rails"] + travel_audit(travel)]
    if travel:
        blocks += [[""],
                   ["  (hypothetical travel applied: "
                    + ", ".join(f"J{j}={v}mm" for j, v in sorted(travel.items())) + ")"]]
    rails = ["  firmware axis      measured  nominal  drives"]
    for f in FINGERS:
        jx, jy = STEPPER_JOINTS[FINGER_ID[f]]
        nom_x, nom_y = FINGER_GEOMETRY[FINGER_ID[f]]["box"][1], FINGER_GEOMETRY[FINGER_ID[f]]["box"][0]
        for j, axis, nom, drives in ((jx, "x", nom_x, "y"), (jy, "y", nom_y, "x")):
            got = FULL_EXTENSION_MM[j]
            flag = "   <-- short of nominal" if got < nom - 0.5 else ""
            rails.append(f"  J{j} ({f:6} '{axis}')  {got:6.1f} mm  {nom:5.1f} mm  "
                         f"local {f} {drives}{flag}")
    blocks += [[], rails]
    blocks += [[], check_designs(travel)]
    return "\n".join(line for b in blocks for line in ([""] if not b else b))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit", action="store_true", help="frame + envelope + design report")
    ap.add_argument("--plan", type=Path, nargs="*", default=[],
                    help="exported <design>_plan.json files to turn into commands")
    ap.add_argument("--travel", default=None,
                    help="hypothetical stepper travel, 'J:mm,J:mm' e.g. 1:59.8,3:59.8,5:59.8")
    ap.add_argument("--no-home", action="store_true",
                    help="emit MOVEMM only, for a hand already homed this session")
    args = ap.parse_args()

    travel = None
    if args.travel:
        travel = {int(k): float(v) for k, v in (p.split(":") for p in args.travel.split(","))}

    if args.audit or not args.plan:
        print(audit(travel))
    for path in args.plan:
        plan = HandPlan.from_json(path)
        print("\n" + "=" * 92)
        print(plan.describe(travel))
        print("\nFIRMWARE COMMANDS (docs/protocol.md, in order)")
        for line in plan.stepper_commands(home=not args.no_home):
            print(f"  {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
