"""How hard the pad pressed the indenter at each touch-off's rest pose, from the last vertical descent onto that pose.
The program tares the HEX-E at the rest pose, so a pad that already touches there reads zero, and its onset fit rests on
the start of the approach (rod1, 2026-10-23: 19 of 22 onsets within 0.1 mm of the start). The descent before the rest
keeps the orientation, so the HEX-E's change along it is the contact alone: the force at rest against the force 6 to 10
mm higher (the preload), and the height above the rest pose where the push first passes 0.1 N (the true contact).
Reads the session's events, registration, ft.bin and robot.bin (memory-mapped: only the descents' pages); moves nothing.

    /usr/bin/python3 docs/experiments/20261022-ur5_tip_calibration/lab/descent_check.py SESSION
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src/morphohand/driver/reskin5x/host"))
import ur5_tip_calibration as C  # noqa: E402

PRE_N = 0.3        # sphere2's clean touches (2026-10-22) read a preload of 0.06 N median, 0.17 N at most
sd = Path(sys.argv[1])
reg = json.loads((sd / "registration.json").read_text())
K = np.array(reg.get("K", np.eye(3).tolist()))
up = np.array(reg["T_BI"])[:3, 2]
f = np.memmap(sd / "ft.bin", "<f8", mode="r").reshape(-1, 9)
r = np.memmap(sd / "robot.bin", "<f8", mode="r").reshape(-1, 15)
plan = {a["id"]: a for a in json.loads((sd / "plan.json").read_text())["actions"]}
s0 = {t["id"]: t["s0"] for t in json.loads((sd / "touches.json").read_text())}


def window(x, t0, t1):
    i, j = np.searchsorted(x[:, 0], [t0, t1])
    return np.array(x[i:j])


rows = []
for s in open(sd / "events.jsonl"):
    e = json.loads(s)
    if e.get("event") != "end" or e.get("status") != "ok" or not str(e.get("id", "")).startswith("pad-"):
        continue
    rec = e["rec"]
    st = float(plan[e["id"]].get("standoff", 1.5))
    t_r = rec["t_rest0"][0]
    rr = window(r, t_r - 8.0, t_r + 0.5)
    T_rest = C.ur_to_T(rr[np.searchsorted(rr[:, 0], t_r)][8:14])
    T = [C.ur_to_T(v[8:14]) for v in rr]
    h = np.array([(Ti[:3, 3] - T_rest[:3, 3]) @ up for Ti in T])
    side = np.array([np.linalg.norm((Ti[:3, 3] - T_rest[:3, 3]) - h_i * up) for Ti, h_i in zip(T, h)])
    turn = np.array([C.pose_error(Ti, T_rest)[1] for Ti in T])
    ok = (turn < 0.05) & (side < 0.5) & (h > -0.05) & (h < 15.0) & (rr[:, 0] <= t_r)
    j1 = int(np.flatnonzero(ok)[-1])                     # the last vertical descent: the block of ok samples up to the rest
    k = j1
    while k > 0 and ok[k - 1]:
        k -= 1
    if j1 - k < 10 or h[k] < 6.0:
        rows.append((e["id"], st, s0.get(e["id"]), None, None, float(h[k])))
        continue
    ff = window(f, rr[k, 0], t_r)
    hf = np.interp(ff[:, 0], rr[k:j1 + 1, 0], h[k:j1 + 1])
    y = np.array([-(T_rest[:3, :3] @ np.linalg.solve(K, v[3:6] - np.array(rec["tare"])[:3])) @ (-up) for v in ff])
    base = y[(hf >= 6.0) & (hf <= 10.0)].mean()
    ys = np.convolve(y - base, np.ones(9) / 9, mode="same")
    valid = np.arange(4, len(ys) - 4)                    # clear of the moving mean's edges
    below = valid[ys[valid] <= 0.1]
    if ys[valid[-1]] <= 0.1:
        h_c = 0.0                                        # no contact at the rest
    else:
        h_c = float(hf[below[-1] + 1]) if len(below) else float(hf[valid[0]])
    rows.append((e["id"], st, s0.get(e["id"]), float(-base), h_c, float(h[k])))

print(f"{'touch':14s} {'s0 mm':>7s} {'preload N':>9s} {'contact mm above rest':>21s} {'onset from expected':>19s}")
for i, st, s, pre, h_c, h0 in rows:
    if pre is None:
        print(f"{i:14s} {s:+7.3f}   no vertical descent of 6 mm before the rest (from {h0} mm)")
        continue
    on = -(st + h_c) if pre > PRE_N else s - st         # a contact height without a preload is the noise's crossing
    print(f"{i:14s} {s:+7.3f} {pre:9.3f} {h_c:21.2f} {on:+19.2f}{'' if pre > PRE_N else '  (onset fit: no preload)'}")
pre = np.array([x[3] for x in rows if x[3] is not None])
hc = np.array([x[4] for x in rows if x[3] is not None and x[3] > PRE_N])
print(f"{len(pre)} touches: preload median {np.median(pre):.3f} N (max {pre.max():.3f}); a preload over {PRE_N} N in "
      f"{len(hc)}" + (f", their contact {hc.min():.2f} to {hc.max():.2f} mm above the rest" if len(hc) else ""))
