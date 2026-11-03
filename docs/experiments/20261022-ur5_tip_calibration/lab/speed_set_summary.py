"""The speed sets of 11-03 (speed1, speed2 and speed3: 12 points a seat pressed at 0.05, 0.15, 0.6 and 1.2 mm/s, the
plans' v 0.00015 to 0.004 under the 0.30 slider), summarized per press from the sessions' own streams on corvus, for the
question whether presses (and the return to the rest) may run faster: per level hold, the push's mean and its fall over
the hold, the travel along d from the rest, and the held board's change per chip and axis against its zero interpolated
between the rests before and after the press (t_rest0, t_rest1), as ur5_tip_calibration_analysis.py zeroes it; per press,
the time from the release (the push under 0.05 N after the last hold) to t_rest1, and the board's drift within t_rest1
(its second half against its first). Reads only the presses' windows (memory maps); run it at low priority beside a run:

    nice -n 19 ionice -c3 /usr/bin/python3 docs/experiments/20261022-ur5_tip_calibration/lab/speed_set_summary.py OUT.json
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
LOGS = ROOT / "logs/20261102-tip_calibration"
SEATS = {1: ("145010-s1v6b", 3), 2: ("182710-s2v5", 2), 3: ("220337-s3v4", 0)}   # session, held board's mux channel
HELD = np.dtype([("t", "<f8"), ("b", "u1"), ("r", "<f4", (20,))])
MAG = [i for c in range(5) for i in (4 * c + 1, 4 * c + 2, 4 * c + 3)]          # the 15 field values (x, y, z of 5 chips)
SPEED = re.compile(r"^(?P<pt>.+)-v(?P<v>005|015|060|120)$")


def window(t, a, b):
    return np.searchsorted(t, a), np.searchsorted(t, b)


def apply(T, p):
    return T[:3, :3] @ p + T[:3, 3]


def press_summary(rec, ft, ft_t, rob, rob_t, held, held_t, ch, K, d_B, R_BE):
    def push(rows, tare):
        f_E = np.linalg.solve(K, (rows[:, 3:6] - tare[:3]).T).T
        return -(f_E @ R_BE.T) @ d_B

    def board(a, b):
        i, j = window(held_t, a, b)
        h = held[i:j]
        h = h[h["b"] == ch]
        return h["r"][:, MAG].astype(float), h["t"]

    tare = np.asarray(rec["tare"], float)
    r0a, r0b = rec["t_rest0"]
    r1a, r1b = rec["t_rest1"]
    z0, _ = board(r0a, r0b)
    z1, _ = board(r1a, r1b)
    if not len(z0) or not len(z1):
        return None
    m0, m1, tm0, tm1 = z0.mean(0), z1.mean(0), 0.5 * (r0a + r0b), 0.5 * (r1a + r1b)
    i, j = window(rob_t, r0a, r0b)
    p_rest = rob[i:j, 8:11].mean(0) * 1e3
    out = dict(levels=[])
    last_end = None
    for lv in rec["levels"]:
        a, b = lv["t"]
        i, j = window(ft_t, a, b)
        f = push(ft[i:j], tare)
        k = max(1, len(f) // 5)
        bv, bt = board(a, b)
        zero = m0 + (m1 - m0) * ((0.5 * (a + b) - tm0) / (tm1 - tm0))
        i, j = window(rob_t, a, b)
        p = rob[i:j, 8:11].mean(0) * 1e3
        out["levels"].append(dict(N=lv["N"], push=round(float(f.mean()), 4), fall=round(float(f[:k].mean() - f[-k:].mean()), 4),
                                  travel_mm=round(float((p - p_rest) @ d_B), 4),
                                  board=np.round(bv.mean(0) - zero, 3).tolist(),
                                  board_se=round(float(np.linalg.norm(bv.std(0)) / np.sqrt(max(len(bv), 1))), 4)))
        last_end = b
    if last_end is not None:                   # the release: the push under 0.05 N after the last hold
        i, j = window(ft_t, last_end, r1a)
        f = push(ft[i:j], tare)
        below = np.nonzero(f < 0.05)[0]
        if len(below):
            out["release_to_rest1_s"] = round(float(r1a - ft_t[i + below[0]]), 3)
    h = len(z1) // 2
    out["rest1_drift"] = round(float(np.linalg.norm(z1[h:].mean(0) - z1[:h].mean(0))), 4)
    out["rest1_noise"] = round(float(np.linalg.norm(z1.std(0)) / np.sqrt(max(h, 1))), 4)
    out["rest_change"] = round(float(np.linalg.norm(m1 - m0)), 4)
    return out


def main():
    dst = Path(sys.argv[1])
    res = dict(what=__doc__.split("\n\n")[0], seats={})
    for seat, (ses, ch) in SEATS.items():
        S = LOGS / ses
        reg = json.loads((S / "registration.json").read_text())
        K = np.asarray(reg["K"], float)
        T_BP = np.asarray(reg["T_BI"], float) @ np.asarray(reg["T_AP"], float)
        T_EM = np.asarray(reg["T_EM"], float)
        acts = {}
        for f in sorted(HERE.glob(f"20261102-plan_probe_table_speed_seat{seat}_v3id*.json")):     # the resumed runs' plans
            for a in json.loads(f.read_text())["actions"]:
                acts.setdefault(a["id"], a)
        ft = np.memmap(S / "ft.bin", "<f8", "r").reshape(-1, 9)
        rob = np.memmap(S / "robot.bin", "<f8", "r").reshape(-1, 15)
        held = np.memmap(S / "held.bin", HELD, "r")
        ft_t, rob_t, held_t = ft[:, 0], rob[:, 0], held["t"]
        pts = {}
        for line in open(S / "events.jsonl"):
            e = json.loads(line)
            m = SPEED.match(str(e.get("id", "")))
            if not (m and e.get("event") == "end" and e.get("status") == "ok"):
                continue
            rec, a = e["rec"], acts.get(e["id"])
            if a is None or not rec.get("levels"):
                continue
            tg = a["target"]
            d_B = T_BP[:3, :3] @ (np.asarray(tg["d"], float) / np.linalg.norm(tg["d"]))
            R_BE = np.asarray(rec["T_rest"], float)[:3, :3]
            s = press_summary(rec, ft, ft_t, rob, rob_t, held, held_t, ch, K, d_B, R_BE)
            if s is not None:
                s["v_mm_s"] = round(float(a.get("v", 0.0005)) * 0.3e3, 3)
                pts.setdefault(m.group("pt"), {})[m.group("v")] = s
        res["seats"][str(seat)] = pts
        print(f"seat {seat}: {sum(len(v) for v in pts.values())} presses at {len(pts)} points", flush=True)
    dst.write_text(json.dumps(res, indent=1))
    print("->", dst)


if __name__ == "__main__":
    main()
