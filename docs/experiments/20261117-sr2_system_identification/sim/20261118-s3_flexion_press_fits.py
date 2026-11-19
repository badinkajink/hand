"""S3's flexion presses of 2026-11-18 (lab; data/s3/*_mcp_press.jsonl, *_pip_press.jsonl, *_mcp_pip_prebend.jsonl): the
static torque of each MCP and PIP servo against its error, the give of the joints held beside it, and the hold on the
step back.

Each press stepped one joint's target 0.5 deg every 2 s past contact with a fixed plate (the taped plate for the index
and middle, the tag plate for the thumb), until 3 N, an overload flag, 55 C or the target's travel, then stepped it back
the same way. Per step, over the dwell's last second: the HEX-E's force change in P (dF_P; the press of 15:54 predates
it, so its dF_N turns into P with the next press's R_PH), the three joints' readings (the station's samples in the raw
file, median) and the held joints' targets (the header's start joints).

  torque       about each joint axis, from the finger's kinematics at the measured readings (sr2hand.robot.tool_chain
               SR2Hand, the cad_v3 chain): a . ((c - o) x F), with o a point on the axis, a its direction (positive
               flexes) and c the pad's apex, where the chain puts the contact. dF_P is the force of the finger on the
               plate, so the pressed joint's torque is positive. With the tip turned 20-25 deg the point of the pad
               that faces the plate lies a few mm further along the tip than the apex; the PIP's lever then changes by
               up to about 10 %.
  contact      the first press step with |F_xy| >= 0.15 N
  gain         the pressed joint's torque against its error (target - reading): the least-squares slope through zero
               over the steps in contact with an error up to 6 deg (N m/rad), and the torque at 5 deg (linear between
               steps)
  largest      the largest torque before the stop, with its error, and the servo's duty there ("load", in 0.1 % of full
               PWM); torque / duty extrapolates to the torque at full duty
  advance      the pressed joint's reading past contact at the largest torque, and the held joints' change of reading
  held joints  the torque on each held joint against its reading's change from the step before contact (N m per deg
               of give), at the largest torque and over the press
  step back    the error at which the force fell below half its value at the step back's start
  slips        press steps in contact whose reading advanced 1 deg or more over the step before

    python3 sim/20261118-s3_flexion_press_fits.py [--out data/s3/20261118-flexion_press_fits.json]
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[2]
sys.path.insert(0, str(ROOT / "src"))
from sr2hand.robot.tool_chain import SR2Hand  # noqa: E402

LOGS = ROOT / "logs/20261118-sr2_s3_press"
JOINTS = ("yaw", "mcp", "pip")
F_CONTACT = 0.15
GAIN_MAX_ERR = 6.0
R_PH_FROM = "20261118-160047_index_yaw"


def press_files():
    s3 = STUDY / "data/s3"
    return sorted([*s3.glob("20261118-*_mcp_press.jsonl"), *s3.glob("20261118-*_pip_press.jsonl"),
                   *s3.glob("20261118-*_mcp_pip_prebend.jsonl")])


def station(path):
    """The raw file's station samples: times, readings (n, 3) and loads (n, 3)."""
    stamp = re.match(r"(\d{8}-\d{6})", path.name).group(1)
    raw = next(LOGS.glob(f"{stamp}_*_raw.jsonl"))
    S = []
    for x in raw.read_text().splitlines():
        if x.strip():
            S += [(s[0], s[1], s[2]) for s in json.loads(x)["station"]]
    S.sort(key=lambda s: s[0])
    return (np.array([s[0] for s in S]), np.array([[s[1][j] for j in JOINTS] for s in S]),
            np.array([[s[2][j] for j in JOINTS] for s in S]))


def axes(hand, finger, q_deg):
    """{joint: (point on the axis, unit direction)} in P (mm) and the pad's apex in P, at readings q_deg."""
    ch = hand.finger_links(finger, 0.0, 0.0, *[math.radians(v) for v in q_deg])
    T, fr = np.eye(4), {}
    for ln in ch:
        T = T @ ln.T
        fr[ln.name] = T
    g = hand.fingers[finger]
    out = {"yaw": (fr[f"{finger}_yaw_frame"][:3, 3], fr[f"{finger}_yaw_frame"][:3, :3] @ [1.0, 0.0, 0.0])}
    for j, name in (("mcp", f"{finger}_mcp_frame"), ("pip", f"{finger}_pip_frame")):
        out[j] = (fr[name][:3, 3], fr[name][:3, :3] @ [0.0, g.flex, 0.0])
    return out, fr["contact"][:3, 3]


def per_press(path, hand, R_PH):
    L = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    h, end = L[0], L[-1]
    finger, joint = h["finger"], h["joint"]
    ji = JOINTS.index(joint)
    held = h["start_joints"]
    ts, Q, LD = station(path)
    rows = []
    for r in [x for x in L if x.get("phase") in ("press", "back")]:
        m = (ts > r["t"] - 1.0) & (ts <= r["t"])
        q = np.median(Q[m], 0) if m.any() else np.array([held[j] for j in JOINTS])
        F = np.asarray(r["dF_P"] if "dF_P" in r else R_PH @ np.asarray(r["dF_N"], float), float)
        ax, c = axes(hand, finger, q)
        tau = {j: float(np.dot(a, np.cross(c - o, F))) / 1000.0 for j, (o, a) in ax.items()}
        rows.append({"k": r["k"], "phase": r["phase"], "target": r["target"], "reading": r["reading"], "q": q, "F": F,
                     "Fxy": float(np.hypot(F[0], F[1])), "tau": tau, "duty": abs(float(r["load"])),
                     "flag": any(v for v in r["status"].values()), "temp": r["temp"][joint]})
    pr = [r for r in rows if r["phase"] == "press"]
    bk = [r for r in rows if r["phase"] == "back"]
    sign = 1.0 if np.median([r["tau"][joint] for r in pr if r["Fxy"] >= 1.0] or [1.0]) >= 0 else -1.0
    err = np.array([r["target"] - r["reading"] for r in pr])
    tq = np.array([sign * r["tau"][joint] for r in pr])
    Fxy = np.array([r["Fxy"] for r in pr])
    flag = next((i for i, r in enumerate(pr) if r["flag"]), None)
    n = flag if flag is not None else len(pr)
    c = next((i for i in range(n) if Fxy[i] >= F_CONTACT), None)
    out = {"run": path.name, "finger": finger, "joint": joint, "plate": "tag" if "tag" in (h.get("note") or "") else "taped",
           "held_targets_deg": {j: round(held[j], 2) for j in JOINTS if j != joint},
           "stop": str(end.get("reason"))[:60], "temp_C": [pr[0]["temp"], max(r["temp"] for r in rows)], "steps": len(pr)}
    if c is None:
        return {**out, "contact": None}, rows
    k = np.arange(c, n)
    g = k[err[k] <= GAIN_MAX_ERR]
    kp = float(np.sum(tq[g] * np.radians(err[g])) / np.sum(np.radians(err[g]) ** 2)) if len(g) >= 3 else None
    order = np.argsort(err[k])
    e5 = float(np.interp(5.0, err[k][order], tq[k][order])) if err[k].min() <= 5.0 <= err[k].max() else None
    im = int(k[np.argmax(tq[k])])
    q0 = pr[c - 1]["q"] if c else pr[0]["q"]
    give = {}
    for j in JOINTS:
        if j == joint:
            continue
        jj = JOINTS.index(j)
        tj = np.array([sign * r["tau"][j] for r in pr])
        dq = np.array([r["q"][jj] for r in pr]) - q0[jj]
        sel = k[np.abs(tj[k]) > 0.02]
        slope = float(np.polyfit(tj[sel], dq[sel], 1)[0]) if len(sel) >= 4 else None
        give[j] = {"torque_at_largest_Nm": round(float(tj[im]), 3), "reading_change_deg": round(float(dq[im]), 2),
                   "deg_per_Nm": round(slope, 1) if slope is not None else None}
    slips = [{"step": int(pr[i]["k"]), "jump_deg": round(float(pr[i]["reading"] - pr[i - 1]["reading"]), 2),
              "tau_before_Nm": round(float(tq[i - 1]), 3), "tau_after_Nm": round(float(tq[i]), 3)}
             for i in range(c + 1, n) if pr[i]["reading"] - pr[i - 1]["reading"] >= 1.0]
    half = None
    if bk:
        f0 = bk[0]["Fxy"]
        i = next((i for i, r in enumerate(bk) if r["Fxy"] < 0.5 * f0), None)
        if i is not None:
            half = round(float(bk[i]["target"] - bk[i]["reading"]), 2)
    duty = pr[im]["duty"]
    return {**out,
            "contact": {"step": int(pr[c]["k"]), "reading_deg": round(pr[c]["reading"], 2),
                        "error_deg": round(float(err[c]), 2)},
            "gain_Nm_per_rad": round(kp, 3) if kp is not None else None, "torque_at_5deg_Nm": round(e5, 3) if e5 else None,
            "largest": {"torque_Nm": round(float(tq[im]), 3), "error_deg": round(float(err[im]), 2),
                        "F_P_N": np.round(pr[im]["F"], 2).tolist(), "duty_permille": duty,
                        "torque_at_full_duty_Nm": round(float(tq[im]) * 1000.0 / duty, 3) if duty else None,
                        "lever_mm": round(float(tq[im]) / pr[im]["Fxy"] * 1000.0, 1)},
            "advance_past_contact_deg": round(float(pr[im]["reading"] - pr[c]["reading"]), 2),
            "held": give, "step_back_half_force_error_deg": half, "slips": slips,
            "overload": {"step": int(pr[flag]["k"]), "duty_before": pr[flag - 1]["duty"],
                         "torque_before_Nm": round(float(tq[flag - 1]), 3)} if flag is not None else None,
            "tangential_over_normal_at_largest": round(abs(float(pr[im]["F"][2])) / max(pr[im]["Fxy"], 1e-9), 2)}, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(STUDY / "data/s3/20261118-flexion_press_fits.json"))
    a = ap.parse_args()
    hand = SR2Hand.load()
    R_PH = np.asarray(json.loads((LOGS / f"{R_PH_FROM}.jsonl").read_text().splitlines()[0])["R_PH"], float)
    presses, steps = [], {}
    for p in press_files():
        res, rows = per_press(p, hand, R_PH)
        presses.append(res)
        steps[p.name] = [[r["k"], r["phase"], round(r["target"], 2), *np.round(r["q"], 2).tolist(), *np.round(r["F"], 3).tolist(),
                          *[round(r["tau"][j], 4) for j in JOINTS], r["duty"], r["temp"]] for r in rows]
        if res["contact"] is None:
            print(p.name, "no contact")
            continue
        lg = res["largest"]
        print(f"{p.name[:15]} {res['finger']:6s} {res['joint']} {res['plate']:5s} contact at {res['contact']['reading_deg']:5.1f} | "
              f"gain {res['gain_Nm_per_rad']} N m/rad, at 5 deg {res['torque_at_5deg_Nm']} | largest {lg['torque_Nm']} N m at "
              f"{lg['error_deg']} deg, duty {lg['duty_permille']}, full duty {lg['torque_at_full_duty_Nm']}, lever "
              f"{lg['lever_mm']} mm | advance {res['advance_past_contact_deg']} | held {json.dumps(res['held'])} | back half at "
              f"{res['step_back_half_force_error_deg']} | slips {len(res['slips'])} | overload {res['overload']} | t/n "
              f"{res['tangential_over_normal_at_largest']} | {res['temp_C']} C | {res['stop']}")
    by = {}
    for r in presses:
        if r["contact"] is not None and "prebend" not in r["run"]:
            by.setdefault(f"{r['finger']} {r['joint']}", []).append(r)
    med = lambda v: round(float(np.median([x for x in v if x is not None])), 3) if any(x is not None for x in v) else None
    summary = {key: {"n": len(v), "gain_Nm_per_rad": med([r["gain_Nm_per_rad"] for r in v]),
                     "torque_at_5deg_Nm": med([r["torque_at_5deg_Nm"] for r in v]),
                     "largest_torque_Nm": med([r["largest"]["torque_Nm"] for r in v]),
                     "torque_at_full_duty_Nm": med([r["largest"]["torque_at_full_duty_Nm"] for r in v]),
                     "advance_deg": med([r["advance_past_contact_deg"] for r in v]),
                     "step_back_half_force_error_deg": med([r["step_back_half_force_error_deg"] for r in v])}
               for key, v in by.items()}
    for key, s in summary.items():
        print("  ", key, json.dumps(s))
    Path(a.out).write_text(json.dumps({
        "what": __doc__.split("\n\n")[0].replace("\n", " "),
        "script": "docs/experiments/20261117-sr2_system_identification/sim/20261118-s3_flexion_press_fits.py",
        "summary": summary, "presses": presses,
        "step_cols": ["k", "phase", "target_deg", "yaw_deg", "mcp_deg", "pip_deg", "F_x_P_N", "F_y_P_N", "F_z_P_N",
                      "tau_yaw_Nm", "tau_mcp_Nm", "tau_pip_Nm", "duty_permille", "temp_C"],
        "steps": steps}, separators=(",", ":")) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
