"""S3's fourteen head-on yaw presses of 2026-11-18 (lab, 16:33-17:17; data/s3/*_yaw_head_on.jsonl) against the
prediction of W3 (QUEUE, "The block as built" and "S3's index presses against the replay plant"), and the numbers a
replay plant takes from them.

Per press, over the press steps (each row: the target, the yaw's reading and the error target - reading, in simulator
degrees; the servo's duty ("load", 0.1 % units); the HEX-E's force change in P, of which only x_P and y_P are used):
  contact      the first step whose |F_xy| reached 0.15 N (lab/s3_yaw_summary.py's rule): its error and reading
  F vs reading the least-squares slope of |F_xy| against the reading's advance from contact, in the press's direction,
               over the steps in contact before the overload flag: N per deg of reading at the tip, and the stiffness
               past the potentiometer about the yaw axis, slope x 180/pi x L (N m/rad)
  F vs error   the force at an error of 5 and 10 deg (linear between steps), and the largest force with its error;
               torques at L
  slips        steps in contact where the reading advanced in the press's direction: the reading's jump, the force
               before and after, and the duty at the step before
  stall        the first run of four steps (2 deg of target at the slow rate) without a force rise of 0.05 N: its force
  overload     the duty at the step before the servo's overload flag (status 32), when there was one
L = 88 mm, the yaw axis to the pad's centre on the straight finger (1.53 mm per deg); the -y presses met the DIP link's
flat side, whose lever is not measured.

    python3 sim/20261118-s3_yaw_head_on_fits.py [--out data/s3/20261118-yaw_head_on_fits.json]
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
L_M = 0.088
F_CONTACT = 0.15
PREDICTION = {"slope_N_per_deg_error": 0.19, "force_at_14_15_deg_N": 2.8, "reading_deg_per_N": 1.3,
              "step_back": "the force holds until the target passes the reading"}


def per_press(path):
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    h, end = rows[0], rows[-1]
    st = [r for r in rows if r.get("phase") == "press"]
    s = float(h["sign"])
    e = np.array([r["offset"] for r in st])
    rd = np.array([r["reading"] for r in st])
    F = np.array([math.hypot(*r["dF_P"][:2]) for r in st])
    duty = np.array([abs(r["load"]) for r in st])
    flag = next((i for i, r in enumerate(st) if r["status"]["yaw"]), None)
    n = flag if flag is not None else len(st)
    c = next((i for i in range(n) if F[i] >= F_CONTACT), None)
    out = {"run": path.name, "finger": h["finger"], "sign": int(h["sign"]),
           "plate": "tag" if "tag" in (h.get("note") or "") else "taped",
           "rate": "fast" if h["step_deg"] > 1.0 else "slow", "steps": len(st)}
    if c is None:
        return {**out, "contact": None}
    adv = s * (rd - rd[c - 1 if c else 0])               # the reading's advance from the step before contact
    k = np.arange(c, n)
    a, b = np.polyfit(adv[k], F[k], 1)
    slips = []
    for i in range(c, n):
        jump = s * (rd[i] - rd[i - 1])
        if jump > 0.1:
            slips.append({"step": int(st[i]["k"]), "jump_deg": round(float(jump), 2), "F_before_N": round(float(F[i - 1]), 2),
                          "F_after_N": round(float(F[i]), 2), "duty_before": int(duty[i - 1])})
    stall = None
    for i in range(c, n - 4):
        if F[i + 4] - F[i] < 0.05 and all(F[i:i + 5] - F[i] < 0.05):
            stall = {"step": int(st[i]["k"]), "F_N": round(float(F[i]), 2), "error_deg": round(float(e[i]), 2)}
            break
    imax = int(np.argmax(F[:n + 1] if flag is not None else F))
    at = lambda x: round(float(np.interp(x, e[c:n], F[c:n])), 2) if e[c:n].min() <= x <= e[c:n].max() else None
    return {**out,
            "contact": {"step": int(st[c]["k"]), "error_deg": round(float(e[c]), 2), "reading_deg": round(float(rd[c]), 2),
                        "F_N": round(float(F[c]), 2)},
            "F_vs_reading": {"N_per_deg": round(float(a), 3), "offset_N": round(float(b), 2),
                             "deg_per_N": round(1.0 / a, 2) if a > 0 else None,
                             "k_past_pot_Nm_per_rad": round(float(a) * 180.0 / math.pi * L_M, 2), "n": int(len(k))},
            "F_at_error_5_N": at(5.0), "F_at_error_10_N": at(10.0),
            "F_max_N": round(float(F[imax]), 2), "error_at_max_deg": round(float(e[imax]), 2),
            "tau_max_Nm": round(float(F[imax]) * L_M, 3), "stall": stall, "slips": slips,
            "overload_duty": int(duty[flag - 1]) if flag else None, "stop": str(end.get("reason"))[:40]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(STUDY / "data/s3/20261118-yaw_head_on_fits.json"))
    a = ap.parse_args()
    presses = [per_press(p) for p in sorted((STUDY / "data/s3").glob("20261118-*_yaw_head_on.jsonl"))]
    for p in presses:
        fr = p["F_vs_reading"]
        print(f"{p['run'][:15]} {p['finger']:6s} {p['sign']:+d} {p['plate']:5s} {p['rate']:4s} contact e "
              f"{p['contact']['error_deg']:5.2f} | F/reading {fr['N_per_deg']:.3f} N/deg ({fr['deg_per_N']} deg/N, "
              f"{fr['k_past_pot_Nm_per_rad']} N m/rad) | F at e5 {p['F_at_error_5_N']} e10 {p['F_at_error_10_N']} | stall "
              f"{p['stall'] and p['stall']['F_N']} | max {p['F_max_N']} at e {p['error_at_max_deg']} ({p['tau_max_Nm']} N m) | "
              f"slips {len(p['slips'])}, largest {max([s['jump_deg'] for s in p['slips']] or [0])} deg | overload duty "
              f"{p['overload_duty']}")
    med = lambda v: round(float(np.median([x for x in v if x is not None])), 3)
    pooled = {"F_per_reading_N_per_deg": med([p["F_vs_reading"]["N_per_deg"] for p in presses]),
              "k_past_pot_Nm_per_rad": med([p["F_vs_reading"]["k_past_pot_Nm_per_rad"] for p in presses]),
              "F_at_error_5_N": med([p["F_at_error_5_N"] for p in presses]),
              "F_at_error_10_N": med([p["F_at_error_10_N"] for p in presses]),
              "stall_N": med([p["stall"] and p["stall"]["F_N"] for p in presses]),
              "F_max_N": med([p["F_max_N"] for p in presses]), "tau_max_Nm": med([p["tau_max_Nm"] for p in presses]),
              "overload_duty": med([p["overload_duty"] for p in presses])}
    print("medians", json.dumps(pooled))
    res = {"what": __doc__.split("\n\n")[0].replace("\n", " "),
           "script": "docs/experiments/20261117-sr2_system_identification/sim/20261118-s3_yaw_head_on_fits.py",
           "L_m": L_M, "prediction": PREDICTION, "medians": pooled, "presses": presses}
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
