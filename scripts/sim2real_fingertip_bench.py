#!/usr/bin/env python3
"""Bench procedure and first-bench package of the fingertip deployment-readiness study (docs/handoff/
20261016-sim2real_fingertip_readiness.md, steps 8 and 9).

  procedure   the assembly and bench measurements that the models need, in order, each with the model's prediction, the
              file it comes from and its status (every one pending: no printed 85A tip has been measured):
              docs/experiments/20261016-sim2real_fingertip_readiness/bench/procedure.json

Indentation predictions come from the 85A finite elements (solid modulus 12 MPa, NinjaFlex's) and are inverted for
Siraya's modulus: in linear elastic contact the response at force F and solid modulus E equals the 12 MPa response at
F x 12 MPa / E, so a measured approach d at force F gives E = 12 MPa x F / F12(d), with F12(d) the 12 MPa model's force at
approach d (log-log interpolation over 0.5, 1 and 3 N).

    python3 scripts/sim2real_fingertip_bench.py procedure
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
D = ROOT / "docs/experiments/20261016-sim2real_fingertip_readiness"
OUT = D / "bench"
E0 = 12.0


def fe_rows():
    return {(r["layout"], r["recipe"]): r for r in (json.loads(l) for l in open(D / "fe/fem.jsonl"))}


def _loglog(x, xs, ys):
    """Piecewise power law through (xs, ys), extended beyond the ends with the nearest segment's exponent."""
    lx, lxs, lys = math.log(x), np.log(xs), np.log(ys)
    k = int(np.clip(np.searchsorted(lxs, lx) - 1, 0, len(xs) - 2))
    s = (lys[k + 1] - lys[k]) / (lxs[k + 1] - lxs[k])
    return float(math.exp(lys[k] + s * (lx - lxs[k])))


def approach_at(row, F, E_s):
    """Approach (um) under the 25 mm tool at force F (N) for solid modulus E_s (MPa), from the 12 MPa rows."""
    Fs = np.array([0.5, 1.0, 3.0])
    ds = np.array([row["per_force"][f"{f:g}"]["delta_um"] for f in Fs])
    return _loglog(F * E0 / E_s, Fs, ds)


def modulus_from(row, F, d_um):
    """Siraya's solid modulus (MPa) from a measured approach d at force F, by the scaling of the module docstring."""
    Fs = np.array([0.5, 1.0, 3.0])
    ds = np.array([row["per_force"][f"{f:g}"]["delta_um"] for f in Fs])
    return E0 * F / _loglog(d_um, ds, Fs)


def cmd_procedure(a):
    fe = fe_rows()
    S = json.loads((D / "summary.json").read_text()) if (D / "summary.json").exists() else {}
    sens = json.loads((D / "sensing/sensing.json").read_text())["rows"]
    bed = S.get("bed", {})
    var = S.get("variants", {})

    def sense(layout, recipe, plane="plate"):
        return next(r for r in sens if r["layout"] == layout and r["recipe"] == recipe and r["plane"] == plane)
    indent = {}
    for lay, lab in (("four_d3x2_d5", "block, four magnets"), ("five_d3x2_d5", "block, five magnets")):
        for rec in ("w85_30", "w85_50"):
            r = fe[(lay, rec)]
            indent[f"{lab}, {rec[-2:]} % gyroid"] = {
                f"{F:g} N": {f"E_s {E:g} MPa": round(approach_at(r, F, E), 1) for E in (6, 9, 12, 18, 24)} for F in (0.5, 1.0, 3.0)}
    s5, s4 = sense("five_d3x2_d5", "w85_30"), sense("four_d3x2_d5", "w85_30")
    steps = [
        dict(id="seat", what="Seat coupon in 85A (print/20261016-siraya85a_seat_coupon_*): press a magnet into each 2.7, 2.8 and "
                             "2.9 mm round seat and slot; keep the narrowest seat that takes the magnet to its floor without "
                             "splitting the wall and holds it when the coupon is turned over and tapped",
             predicts="no model; sets the production seat (2.7 mm in every print file now)", status="pending"),
        dict(id="retention", what="Magnet retention: after the print, a 3 x 2 mm steel-free pull on each magnet through the bridge "
                                  "is not possible; instead read the board's zero-load field (step 'board') before and after 50 "
                                  "presses at 3 N; a moved magnet shifts the field at its nearest chip",
             predicts="field change under 2 % between the two zero-load readings", status="pending"),
        dict(id="mass", what="Weigh each printed tip with its magnets, and one assembled finger",
             predicts={v: var[v]["tip_g"] for v in var if v != "block_e10"} | {"finger_model_g": 85.2, "owner_estimate_g": 120},
             status="pending"),
        dict(id="indentation", what="Indentation of the printed block under a 25 mm diameter rod at the contact point (the "
                                    "bench tool's radius), at 0.5, 1 and 3 N on the load cell, tip on a rigid back; three "
                                    "repeats; the approach from first touch",
             predicts=indent, inverts="E_s = 12 MPa x F / F12(d): modulus_from() in this script", status="pending"),
        dict(id="pull", what="Friction pull: two printed tips pinch the 25 mm rod at 0.5, 1 and 2 N per tip; pull along the "
                             "rod at 2 N/s (scripts/tpu_tip_calibrate.py fit --pull); onset force over twice the pinch force "
                             "is the friction coefficient",
             predicts=dict(model_mu=1.0, note="every simulation uses mu 1.0 (studies vary 0.5-1.0); the pads' onset/(2N) is "
                                              f"{(bed.get('block4_g30|v2') or {}).get('pull_mu_eff')} at mu 1"),
             status="pending"),
        dict(id="twist", what="Twist onset: the pinched rod (1 N per tip) under a slow torque ramp about the pinch axis; the "
                              "torque at which it starts to turn over (2 mu N) is the onset arm",
             predicts={k: (bed.get(f"{k}|v2") or {}).get("onset_arm_mm") for k in ("block4_g30", "block4_g50", "cap_g30",
                                                                                  "ell16_g30", "ell11_g30")} |
             {"drake_block4_g30": (bed.get("block4_g30|drake") or {}).get("onset_arm_mm")}, status="pending"),
        dict(id="servo_step", what="Servo free-air step: each finger servo commanded 10 deg at the station's 50 Hz with the "
                                   "finger in air; position readback at 111 Hz; then a stall against the load cell at the "
                                   "thumb MCP (servo identification protocol, 20261013-servo_identification)",
             predicts=dict(time_constant_s=0.02, kp_Nm_per_rad=0.72, torque_limit_Nm=0.18, datasheet_stall_Nm=0.23),
             status="pending"),
        dict(id="board", what="Board zero-load field: the five chips at GAIN_SEL 7, RES 0, OSR 0, DIG_FILT 5 (141 Hz), tip "
                              "unloaded, board on the dorsal face of the DIP plate, chips toward the plate; compare each "
                              "chip's field with the prediction; a reversed magnet shows its signature",
             predicts=dict(five_magnets_plate_mT=s5["B_chips_mT"], four_magnets_plate_mT=s4["B_chips_mT"],
                           gain_sel=dict(five=s5["gain_sel"], four=s4["gain_sel"]),
                           one_magnet_reversed_dB_uT=dict(five=s5["polarity_flip_dB_uT"], four=s4["polarity_flip_dB_uT"]),
                           force_bound_N=dict(five=s5["centre_force_N"], four=s4["centre_force_N"])),
             status="pending"),
        dict(id="calibration", what="Force calibration of the board: press the contact point at 0.25-3 N on the load cell "
                                    "with the rod, 15 positions over the face; fit the field change per newton per chip",
             predicts=dict(travel_um_per_N=dict(five=s5["magnet_travel_1N_um"], four=s4["magnet_travel_1N_um"]),
                           signal_U1_uT_per_N=dict(five=s5["signal_U1_uT_per_N"], four=s4["signal_U1_uT_per_N"])),
             status="pending"),
        dict(id="tags", what="AprilTag tracker: scripts/real_v1_tag_tracker.py --probe with the station's BENCH_DEFAULTS; for "
                             "the screwdriver remove --symmetric-object from BENCH_DEFAULTS (docs/hardware_control_station.md)",
             predicts="static tag id6 latched, object tag id0 sampled before any motion", status="pending"),
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "procedure.json").write_text(json.dumps(dict(steps=steps, E0_MPa=E0, script="scripts/sim2real_fingertip_bench.py",
                                                        when=time.strftime("%Y-%m-%d %H:%M")), indent=1, default=float))
    print(f"{(OUT / 'procedure.json').relative_to(ROOT)}: {len(steps)} steps, all pending")
    r = fe[("four_d3x2_d5", "w85_30")]
    for d in (40.0, 60.8, 120.0):
        print(f"block four 30 %: approach {d} um at 1 N -> E_s {modulus_from(r, 1.0, d):.1f} MPa")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("procedure")
    a = ap.parse_args()
    {"procedure": cmd_procedure}[a.cmd](a)


if __name__ == "__main__":
    main()
