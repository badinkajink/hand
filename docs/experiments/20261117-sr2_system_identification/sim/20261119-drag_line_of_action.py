"""Where the forces of the S4(b) drags act, from the HEX-E's moments (identification queue, 2026-11-19): whether the
middle's friction of 1.8-2 times the index's comes from its contact or from the measurement.

The HEX-E reads all six components (the raw files' "hexe" rows: Fx Fy Fz Tx Ty Tz, in the sensor's frame); the
header's R_PH turns both the force and the moment into the palm frame P. Per leg the plateau medians of the 0.3 s means
are taken as in sim/20261118-s4b_drag_friction.py, and the free-air controls' medians at the same speed and direction
are subtracted. A single contact at c gives T = (c - o) x F + T0 in every leg, with o the HEX-E's tool face in P (the
cad_v3 chain) and T0 the moments' constant offset. A least-squares fit over a run's 36 legs gives c and T0; c is
compared with the contact the approach planned (lead_P: the pad's apex, or the DIP link's end for the sides). The legs
span normal forces of 0.5-2 N and frictions of both signs, so the fit sees the normal force's moment and the friction's
moment separately.

Two levers along y_P test the measurement. The friction's lever comes from the half-differences of the down and up legs
(dT_x + c_z dF_y) / dF_z; the normal force's from the change of the leg means' moment with the normal force across
the load levels. A moment-into-force crosstalk that scales the measured friction by (1 + k y) shortens one lever
against the other, with opposite signs for the index and the middle, which sit on opposite sides of the sensor.

    python3 sim/20261119-drag_line_of_action.py [--out data/s3/20261119-drag_line_of_action.json]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[2]
sys.path.insert(0, str(ROOT / "src"))
from sr2hand.geometry.rotations import skew  # noqa: E402
from sr2hand.robot.tool_chain import SR2Hand  # noqa: E402

RUNS = {  # loaded run: (approach plan, free-air controls, what touches, normal axis in P)
    "20261118-191748_index_protocol": ("20261118-191629_approach_index", ["20261119-134601_index_protocol"],
                                       "index pad", 0),
    "20261118-200305_middle_protocol": ("20261118-200220_approach_middle", ["20261119-135018_middle_protocol"],
                                        "middle pad", 0),
    "20261119-123219_index_protocol": ("20261119-122743_approach_index_yaw_pad",
                                       ["20261119-122906_index_protocol", "20261119-130711_index_protocol"],
                                       "index PLA side", 1),
    "20261119-131412_middle_protocol": ("20261119-131019_approach_middle_yaw_pad",
                                        ["20261119-131125_middle_protocol", "20261119-134049_middle_protocol"],
                                        "middle PLA side", 1),
}


def logs(stem):
    return ROOT / f"logs/{stem[:8]}-sr2_s3_press"


def legs_wrench(stem):
    """Per leg: (N, speed, rest, dir) -> plateau medians of F (N) and T (N mm) in P."""
    rec = STUDY / "data/s3" / f"{stem}.jsonl"
    rec = rec if rec.exists() else logs(stem) / f"{stem}.jsonl"
    L = [json.loads(x) for x in rec.read_text().splitlines() if x.strip()]
    h = L[0]
    raw = [json.loads(x) for x in (logs(stem) / f"{stem}_raw.jsonl").read_text().splitlines() if x.strip()]
    th = np.concatenate([r["hexe_t"] for r in raw])
    W = np.concatenate([np.array(r["hexe"], float)[:, :6] for r in raw])
    o = np.argsort(th)
    th, W = th[o], W[o]
    tt = np.concatenate([[q[0] for q in r["tcp"]] for r in raw])
    zt = np.concatenate([[q[1][2] for q in r["tcp"]] for r in raw]) * 1e3
    o = np.argsort(tt)
    z = np.interp(th, tt[o], zt[o])
    R = np.array(h["R_PH"], float)
    F, T = W[:, :3] @ R.T, W[:, 3:] @ R.T * 1e3
    n = max(1, int(round(0.3 / np.median(np.diff(th)))))
    k = np.ones(n) / n
    F = np.column_stack([np.convolve(F[:, i], k, "same") for i in range(3)])
    T = np.column_stack([np.convolve(T[:, i], k, "same") for i in range(3)])
    out = {}
    for lg in [x for x in L if x.get("phase") == "leg"]:
        s = (th >= lg["t0"]) & (th <= lg["t1"])
        pl = np.abs(z[s] - z[s][0]) >= 2.5
        key = (lg["N"], lg["speed_mm_s"], lg["rest_s"], "up" if lg["to_mm"] > 0 else "down")
        out[key] = (np.median(F[s][pl], axis=0), np.median(T[s][pl], axis=0))
    return out


def corrected(stem, controls):
    legs = legs_wrench(stem)
    cl = {}
    for c in controls:
        for (N, v, rs, d), w in legs_wrench(c).items():
            cl.setdefault((v, d), []).append(w)
    bias = {key: (np.mean([w[0] for w in ws], axis=0), np.mean([w[1] for w in ws], axis=0)) for key, ws in cl.items()}
    return {key: (f - bias[(key[1], key[3])][0], t - bias[(key[1], key[3])][1]) for key, (f, t) in legs.items()}


def fit_contact(legs, o):
    """Least squares T = (c - o) x F + T0 over the legs: c (mm, P), T0 (N mm) and the moments' RMS residual."""
    A = np.vstack([np.hstack([-skew(f), np.eye(3)]) for f, _ in legs.values()])
    b = np.concatenate([t for _, t in legs.values()])
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    return o + x[:3], x[3:], float(np.sqrt(np.mean((A @ x - b) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(STUDY / "data/s3/20261119-drag_line_of_action.json"))
    a = ap.parse_args()
    hand = SR2Hand.load()
    T_HP = np.eye(4)
    for ln in hand.palm_links()[1:]:
        T_HP = T_HP @ ln.T
    o_P = np.linalg.inv(T_HP)[:3, 3]
    print("HEX-E tool face in P (mm):", np.round(o_P, 1))
    res = []
    for stem, (plan, controls, what, nax) in RUNS.items():
        lead = np.array(json.loads((logs(plan) / f"{plan}.json").read_text())["lead_P"], float)
        legs = corrected(stem, controls)
        c, T0, rms = fit_contact(legs, o_P)
        r = c - o_P
        fr = []
        for (N, v, rs, d), (f, t) in legs.items():
            if d == "down" and (N, v, rs, "up") in legs:
                fu, tu = legs[(N, v, rs, "up")]
                dF, dT = (f - fu) / 2, (t - tu) / 2
                fr.append((dT[0] + r[2] * dF[1]) / dF[2])
        lev = np.array([[N, (f[nax] + legs[(N, v, rs, "up")][0][nax]) / 2, (t + legs[(N, v, rs, "up")][1]) / 2]
                        for (N, v, rs, d), (f, t) in legs.items() if d == "down" and (N, v, rs, "up") in legs],
                       dtype=object)
        Fn = np.array([x[1] for x in lev], float)
        Tm = np.array([x[2] for x in lev], float)
        # normal along x_P: T_z = r_x F_y - r_y F_x; along y_P: T_x = r_y F_z - r_z F_y, so use T_z = r_x F_y - r_y F_x
        # with F_y the normal: dT_z/dF_y = r_x; the y lever then comes from the fit only
        if nax == 0:
            slope = np.polyfit(Fn, Tm[:, 2], 1)[0]
            y_normal = -slope
        else:
            y_normal = None
        rec = {"run": stem, "what": what, "plan": plan, "controls": controls, "lead_P_mm": np.round(lead, 1).tolist(),
               "contact_fit_P_mm": np.round(c, 1).tolist(), "fit_minus_lead_mm": np.round(c - lead, 1).tolist(),
               "T0_Nmm": np.round(T0, 1).tolist(), "moment_rms_Nmm": round(rms, 2), "legs": len(legs),
               "y_lever_friction_mm": round(float(np.median(fr)) + float(o_P[1]), 1),
               "y_lever_friction_range_mm": [round(float(np.min(fr)) + float(o_P[1]), 1),
                                             round(float(np.max(fr)) + float(o_P[1]), 1)],
               "y_lever_normal_mm": None if y_normal is None else round(float(y_normal) + float(o_P[1]), 1)}
        res.append(rec)
        print(f"{what:16s} lead {rec['lead_P_mm']} fit {rec['contact_fit_P_mm']} (fit - lead {rec['fit_minus_lead_mm']}),"
              f" moment rms {rec['moment_rms_Nmm']} N mm; y of the friction's line {rec['y_lever_friction_mm']} "
              f"{rec['y_lever_friction_range_mm']}, of the normal force's {rec['y_lever_normal_mm']}")
    Path(a.out).write_text(json.dumps({"what": __doc__.split("\n\n")[0].replace("\n", " "),
                                       "script": "docs/experiments/20261117-sr2_system_identification/sim/"
                                                 "20261119-drag_line_of_action.py",
                                       "hexe_tool_face_P_mm": np.round(o_P, 2).tolist(), "runs": res}, indent=1) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
