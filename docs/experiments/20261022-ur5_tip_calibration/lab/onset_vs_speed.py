#!/usr/bin/env python3
"""Contact onset against approach speed, for the owner's question of faster touch-offs (workstation's design, 10-29 04:04).

For each press of a stage (default grid) the push along the press against the flange's travel from its rest pose, over
the approach to the first level, fitted by the program's fit_onset as the touch-offs are. The onset pose is the rest
pose moved along the press by the fitted onset; its residual is the registration fit's own, the pad's signed distance to
the sphere's centre less R at that pose (registration.json of the session), with its sign turned so that a later contact
is positive. The TCP's speed at the onset is measured from robot.bin (a line through the flange's last 0.2 mm of travel
before it), since the slider scales the plan's v. The session's touch-offs (touches.json) go through the same residual,
as fitted and refitted on their own window at 0.4 N and at the presses' y_max (the window's own effect).

The reference press stops deeper over a session's first minutes, so each press off the common speed is compared with
the common-speed presses run within 10 min of it, and a fit of lag x speed plus a drift linear between knots 20 min
apart gives the effective lag over the speeds where fit_onset holds (v up to 0.002 m/s): faster, its 25-sample average
spans so much travel that s_hi falls before the rise and the fitted onset comes early (s_hi - s0 and where the push
passes 0.15 N are printed per speed). Only each press's window of ft.bin and robot.bin is read (memmap, bisection on the
time column).
   onset_vs_speed.py SESSION PLAN [--stage grid] [--ymax 0.25] [--out JSON]"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/will/workspace/hand")
sys.path.insert(0, str(ROOT / "src/morphohand/driver/reskin5x/host"))
import ur5_tip_calibration as C  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("session")
ap.add_argument("plan")
ap.add_argument("--stage", default="grid")
ap.add_argument("--ymax", type=float, default=0.25)
ap.add_argument("--out")
a = ap.parse_args()
sd = Path(a.session)
plan = json.loads(Path(a.plan).read_text())
acts = {x["id"]: x for x in plan["actions"] if x.get("stage") == a.stage}
slider = float(plan.get("speed_slider", 1.0))
geo = C.Geometry(plan)
reg = C.Registration(geo).load(json.loads((sd / "registration.json").read_text()))
T_EP = reg.T_EP()
c_B = C.apply(reg.T_BI, geo.c_I)
Kinv = np.linalg.inv(np.asarray(reg.K, float))


def residual(T_BE):
    """The registration fit's pad residual at a flange pose (mm), later contact positive."""
    q = C.apply(C.inv(np.asarray(T_BE, float) @ T_EP), c_B)
    return -(geo.surface_distance(q)[0] - geo.R)


def opened(name, ncol):
    n = (sd / name).stat().st_size // (8 * ncol)
    return np.memmap(sd / name, "<f8", "r", shape=(n, ncol))


def first_at(mm, t):
    lo, hi = 0, len(mm)
    while lo < hi:
        mid = (lo + hi) // 2
        if mm[mid, 0] < t:
            lo = mid + 1
        else:
            hi = mid
    return lo


def window(mm, t0, t1):
    return np.array(mm[first_at(mm, t0):first_at(mm, t1)])


ft_mm, rb_mm = opened("ft.bin", 9), opened("robot.bin", 15)
ends = {}
for line in open(sd / "events.jsonl"):
    e = json.loads(line)
    if e.get("event") == "end" and e.get("id") in acts and e.get("status") == "ok":
        ends[e["id"]] = e
rows = []
for aid, e in ends.items():
    act, rec = acts[aid], e["rec"]
    if not rec.get("t_move0") or not rec.get("levels") or "T_rest" not in rec or "T_stop" not in rec:
        continue
    T_rest, T_stop = np.array(rec["T_rest"]), np.array(rec["T_stop"])
    d = T_stop[:3, 3] - T_rest[:3, 3]
    d = d / np.linalg.norm(d)
    t0, t1 = rec["t_move0"], rec["levels"][0]["t"][0]
    fa = rec.get("fast")
    if fa and fa["t"][1] and not fa.get("contact"):          # from the fast approach on: the fast presses' slow part
        t0 = fa["t"][0]                                      # holds too few samples for fit_onset's 50
    ft, rb = window(ft_mm, t0, t1), window(rb_mm, t0 - 0.2, t1 + 0.2)
    v = float(act.get("v", 0.0005))
    if len(ft) < 50 or len(rb) < 5:
        rows.append(dict(id=aid, v=v, resid=None, why=f"{len(ft)} HEX-E samples"))
        continue
    P = np.array([C.ur_to_T(r[8:14])[:3, 3] for r in rb])
    s_r = (P - T_rest[:3, 3]) @ d
    s = np.interp(ft[:, 0], rb[:, 0], s_r)
    j = np.clip(np.searchsorted(rb[:, 0], ft[:, 0]), 0, len(rb) - 1)
    R = np.array([C.ur_to_T(r[8:14])[:3, :3] for r in rb[j]])
    f_E = (Kinv @ (ft[:, 3:6] - np.array(rec["tare"])[:3]).T).T
    push = -np.einsum("kij,kj->ki", R, f_E) @ d
    on = C.fit_onset(s, push, a.ymax)
    if on is None:
        rows.append(dict(id=aid, v=v, resid=None, why="no onset fit"))
        continue
    s0 = on[0]
    # the speed over the flange's last 0.2 mm before the onset (free travel at the press's speed); at 0.045 mm/s the
    # 0.2 s around the onset hold 9 um of travel, which the pose's noise swamps
    k = (s_r > s0 - 0.2) & (s_r <= s0)
    if k.sum() < 3:
        t_on = float(rb[np.argmin(np.abs(s_r - s0)), 0])
        k = (rb[:, 0] > t_on - 0.1) & (rb[:, 0] < t_on + 0.1)
    speed = float(np.polyfit(rb[k, 0], s_r[k], 1)[0]) if k.sum() >= 3 else float("nan")
    # fit_onset's own s_hi (the last sample whose 25-sample average lies under a quarter of y_max) and where a 9-sample
    # average of the push first passes 0.15 N: a fit whose s_hi falls before the rise cannot place s0 on it
    ys = np.convolve(push, np.ones(25) / 25, mode="same")
    above = np.nonzero(ys > a.ymax)[0]
    end = int(above[0]) if len(above) else len(push)
    below = np.nonzero(ys[:max(1, end - 12)] < 0.25 * a.ymax)[0]
    s_hi = float(s[below[-1]]) if len(below) else float(s[0])
    y9 = np.convolve(push, np.ones(9) / 9, mode="same")
    c15 = np.nonzero(y9[:len(y9) - 4] > 0.15)[0]
    cross15 = float(s[c15[0]]) if len(c15) else float("nan")
    T_on = T_rest.copy()
    T_on[:3, 3] = T_rest[:3, 3] + d * s0
    rows.append(dict(id=aid, v=v, speed_mm_s=speed, resid=residual(T_on), onset_minus_standoff=s0 - float(act.get("standoff", 1.5)),
                     p=on[1], fit_mN=on[2] * 1e3, n_contact=on[3], p_P=act["target"]["p"], t=t0,
                     s_hi_minus_s0=s_hi - s0, cross015_minus_s0=cross15 - s0))
touch = []
if (sd / "touches.json").exists():
    tends = {}
    for line in open(sd / "events.jsonl"):
        e = json.loads(line)
        if e.get("event") == "end" and e.get("status") == "ok":
            tends[e["id"]] = e
    for t in json.load(open(sd / "touches.json")):
        if t.get("kind") != "pad":
            continue
        row = dict(id=t["id"], resid=residual(t["T_BE"]))
        rec = (tends.get(t["id"]) or {}).get("rec") or {}
        if rec.get("t_move0") and rec.get("t_stop") and "T_stop" in rec:
            # the touch-off refitted on the program's own window (t_move0 to the stop) at its fit_N 0.4 N, which should
            # give back its onset, and at the presses' 0.25 N: the window's own effect on the residual
            T_on, T_stop = np.array(t["T_BE"]), np.array(rec["T_stop"])
            d = T_stop[:3, 3] - T_on[:3, 3]
            d = d / np.linalg.norm(d)
            p_rest = T_on[:3, 3] - d * t["s0"]
            ft, rb = window(ft_mm, rec["t_move0"], rec["t_stop"]), window(rb_mm, rec["t_move0"] - 0.2, rec["t_stop"] + 0.2)
            P = np.array([C.ur_to_T(r[8:14])[:3, 3] for r in rb])
            s_r = (P - p_rest) @ d
            s = np.interp(ft[:, 0], rb[:, 0], s_r)
            j = np.clip(np.searchsorted(rb[:, 0], ft[:, 0]), 0, len(rb) - 1)
            R = np.array([C.ur_to_T(r[8:14])[:3, :3] for r in rb[j]])
            push = -np.einsum("kij,kj->ki", R, (Kinv @ (ft[:, 3:6] - np.array(rec["tare"])[:3]).T).T) @ d
            for ym, key in ((0.4, "refit_0.40"), (a.ymax, f"refit_{a.ymax:.2f}")):
                on = C.fit_onset(s, push, ym, p_set=(1.0, 1.25, 1.5, 1.75, 2.0))
                if on is not None:
                    T2 = T_on.copy()
                    T2[:3, 3] = p_rest + d * on[0]
                    row[key] = residual(T2)
                    row[key + "_s0_change"] = on[0] - t["s0"]
        touch.append(row)


def line(label, x):
    x = np.asarray(x, float)
    sdv = x.std(ddof=1) if len(x) > 1 else 0.0
    return (f"  {label}: n {len(x):3d}, mean {x.mean():+.3f}, sd {sdv:.3f}, SE {sdv / np.sqrt(len(x)):.3f}, "
            f"rms {np.sqrt(np.mean(x ** 2)):.3f} mm")


ok = [r for r in rows if r["resid"] is not None]
print(f"{sd.name}: {len(ok)} of {len(rows)} {a.stage} presses fitted (y_max {a.ymax} N), slider {slider}; residual "
      "against registration.json, later contact positive")
if touch:
    print(line("touch-offs as fitted (fit_N 0.4 N)", [t["resid"] for t in touch]))
    for key in ("refit_0.40", f"refit_{a.ymax:.2f}"):
        rr = [t for t in touch if key in t]
        if rr:
            ch = np.array([t[key + "_s0_change"] for t in rr])
            print(line(f"touch-offs refitted, {key}", [t[key] for t in rr]) + f"; onset moved {ch.mean():+.4f} mm "
                  f"(max |{np.abs(ch).max():.4f}|)")
groups = {}
for v in sorted({r["v"] for r in ok}):
    g = [r for r in ok if r["v"] == v]
    sp = np.array([r["speed_mm_s"] for r in g])
    x = np.array([r["resid"] for r in g])
    groups[v] = (x.mean(), x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan, np.nanmean(sp))
    hi = np.nanmean([r["s_hi_minus_s0"] for r in g])
    cr = np.nanmean([r["cross015_minus_s0"] for r in g])
    print(line(f"v {v:.5f} m/s, measured {np.nanmean(sp):.3f} mm/s", x) + f"; s_hi - s0 {hi:+.3f}, 0.15 N at s0 {cr:+.3f} mm")
common = max(groups, key=lambda v: sum(r["v"] == v for r in ok))
print(f"  each press against the v {common} presses run within 10 min of it (the drift: the reference press stopped "
      "0.10 mm deeper over block 2's first 20 min)")
tm = {}
for v in sorted(set(groups) - {common}):
    dd = []
    for r in (r for r in ok if r["v"] == v):
        nb = [q["resid"] for q in ok if q["v"] == common and abs(q["t"] - r["t"]) < 600]
        if nb:
            dd.append(r["resid"] - np.mean(nb))
    if dd:
        dd = np.array(dd)
        se_ = dd.std(ddof=1) / np.sqrt(len(dd)) if len(dd) > 1 else np.nan
        tm[v] = (dd.mean(), se_)
        print(f"    v {v:.5f} ({groups[v][2]:.3f} mm/s): {len(dd)} presses, minus their time neighbours at v {common} "
              f"({groups[common][2]:.3f} mm/s): {dd.mean():+.3f} mm (SE {se_:.3f})")
# residual = lag x speed + a drift linear between knots 20 min apart, over the presses whose speed keeps fit_onset's
# 25-sample average under 0.25 mm of travel (v up to 0.002, 0.6 mm/s): above it s_hi falls before the rise
use = [r for r in ok if r["v"] <= 0.002 + 1e-12 and np.isfinite(r["speed_mm_s"])]
tt = np.array([r["t"] for r in use])
knots = np.arange(tt.min(), tt.max() + 1200, 1200)
H = np.clip(1 - np.abs((tt[:, None] - knots[None, :]) / 1200.0), 0, None)
H = H[:, H.sum(axis=0) > 0]
x = np.array([r["speed_mm_s"] for r in use])
y = np.array([r["resid"] for r in use])
A = np.c_[x, H]
coef, *_ = np.linalg.lstsq(A, y, rcond=None)
dof = max(1, len(y) - A.shape[1])
s2 = float(np.sum((A @ coef - y) ** 2) / dof)
se = np.sqrt(np.diag(s2 * np.linalg.pinv(A.T @ A)))
dsp = groups[0.0005][2] - groups[0.00015][2] if 0.0005 in groups and 0.00015 in groups else 0.105
print(f"  lag x speed + drift over {len(y)} presses at v <= 0.002: lag {coef[0] * 1e3:+.1f} ms (SE {se[0] * 1e3:.1f}); "
      f"residual sd {np.sqrt(s2):.3f} mm; over the {dsp:.3f} mm/s from v 0.00015 to 0.0005 that is {coef[0] * dsp:+.4f} mm "
      f"(SE {se[0] * dsp:.4f})")
if a.out:
    Path(a.out).write_text(json.dumps(dict(rows=rows, touches=touch, lag_s=coef[0], lag_se_s=se[0],
                                           time_matched={str(k): v for k, v in tm.items()},
                                           groups={str(k): v for k, v in groups.items()}), indent=1))
