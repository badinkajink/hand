#!/usr/bin/env python3
"""One rod for several probe sessions: the joint fit of the rod's axis (its point c_M across the axis, its turn about z_M
and across) and each session's pad pose T_AP over the rod sessions' pass 1 touches (workstation, 23:4x on 11-03: with
the same rod on the same boss, seat 3's own fit turned the rod +10.75 deg about z_M, seats 2 and 1's -5.40 and -2.90).

The residuals and priors are the program's (ur5_tip_calibration.fit_rod_registration): a pad touch's least distance of
the pad's surface from the rod's axis at the onset, less R, over REG_SIGMA_PROBE's pad_mm; each session's pad about its
seed (the 2 mm sphere session named in registration.json's notes.seeded) with the seat priors; the rod's axis point
across the axis about the plan's c_M (1 mm) and its turn about z_M and across (4 and 2 deg). --share turn shares only
the turn and fits c_M per session. --seat-scale widens the pad priors of the sessions named by --loose (all when
none is named): seat 3's rod session was seeded from a 5 mm sphere session (s3v4), seats 1 and 2's from 2 mm ones, and
on seats 1 and 2 the 5 mm and 2 mm sessions' pads stand 1.67 and 1.34 mm apart along x_P (the pad's normal) and 1.2 and
1.3 deg turned, where the seat prior is 0.5 mm.

Per session, printed and written: the own fit as the program makes it (against its registration.json), the joint fit's
T_AP against the own, the touches' rms under both, and the contact shift at the grid's presses. A press is aimed by the
own fit; at the rest moved standoff along d (the onset under the own fit), the pad's point nearest the rod's axis under
the joint fit is compared with the one under the own fit (P): its shift along the surface and the change in the rod's
gap from the pad (positive: the onset comes later under the joint fit).

--grid-level N adds the grid's sweeps (events.jsonl): at level N each sweep holds the rod at offsets of r_mm (0.06 mm
at 0.5 N) around its rest, and their mean flange pose is a contact at N. Its residual is the axis's least distance from
the pad less R plus the indentation at N, one fitted depth per region class (face, edge, flank), over --grid-sd. Pass 1
of 11-03 leaves the rod's turn about z_M unidentified: the shared turn fits the 60 touches at +9.3 or -10.9 deg, both
below the own fits' chi2; the sweeps press with the rod turned 0 and 90 deg about the stem.

--more SESSION ...: sessions that ran on a rod session's registration (the night of 11-03: each seat's second half and
extras ran on its first half's fitted registration); their sweeps join the grid of the rod session on the same seat.
Their registration's T_BI, T_EM and T_AP must equal that session's.

    python3 scripts/ur5_rod_joint_registration.py SESSION [SESSION ...] [--more SESSION ...] [--share both|turn]
        [--seat-scale K [--loose NAME ...]] [--grid-level N [--grid-sd MM]] [--drop ID ...] [--out OUT]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/morphohand/driver/reskin5x/host"))
import ur5_tip_calibration as U  # noqa: E402

LAB = ROOT / "docs/experiments/20261022-ur5_tip_calibration/lab"
OUT = ROOT / "docs/experiments/20261030-inverted_rig_seat_block/data/20261103-strip_table_pads_rod_joint_registration.json"


def seed_dir(reg_json, seeds_root):
    """The lab folder of the session the rod session was seeded from: logs/2026MMDD-tip_calibration/HHMMSS-name ->
    2026MMDD-HHMMSS-name."""
    src = Path(reg_json["notes"]["seeded"]["from_session"])
    return Path(seeds_root) / f"{src.parent.name.split('-')[0]}-{src.name}"


CLASSES = ("face", "edge", "flank")


def grid_poses(sd, level, T_EM):
    """(T_BM, region class, sweep id) for each completed sweep: the mean flange pose of its offsets at the level's
    smallest ring."""
    f = Path(sd) / "events.jsonl"
    if level is None or not f.exists():
        return []
    out = []
    for line in f.read_text().splitlines():
        e = json.loads(line)
        rec = e.get("rec") or {}
        if e.get("event") != "end" or not str(e.get("id", "")).startswith("sweep-") or "offsets" not in rec:
            continue
        o = [x for x in rec["offsets"] if abs(x["N"] - level) < 1e-6 and "T_BE" in x]
        if not o:
            continue
        r0 = min(x["r_mm"] for x in o)
        T = [np.array(x["T_BE"], float) for x in o if x["r_mm"] == r0]
        Tb = T[0].copy()
        Tb[:3, 3] = np.mean([t[:3, 3] for t in T], axis=0)
        out.append((Tb @ T_EM, CLASSES.index(e["id"].split("-")[1]), e["id"]))
    return out


def load(sd, drop, seeds_root, level=None):
    sd = Path(sd)
    plan = json.loads((sd / "plan.json").read_text())
    geo = U.Geometry(plan)
    rj = json.loads((sd / "registration.json").read_text())
    reg = U.Registration(geo)
    reg.load(rj)
    seed = U.Registration(geo)
    sdir = seed_dir(rj, seeds_root)
    seed.load(json.loads((sdir / "registration.json").read_text()))
    pad = [t for t in json.loads((sd / "touches.json").read_text()) if t["kind"] == "pad" and t["id"] not in drop]
    return dict(name=sd.name, plan=plan, geo=geo, reg=reg, rj=rj, seed_name=sdir.name, T_AP0=seed.T_AP, pad=pad,
                Ts=[np.array(t["T_BE"], float) @ reg.T_EM for t in pad], seat=int(plan["geometry"]["seat"]["index"]),
                grid=grid_poses(sd, level, reg.T_EM))


def grid_raw(s, T_AP, c_M, a_M):
    """Each sweep's axis distance from the pad less R at the grid level (mm; minus the indentation)."""
    T_PB = U.inv(s["reg"].T_BI @ T_AP)
    return np.array([s["geo"].line_distance(U.apply(T_PB @ T, c_M), (T_PB @ T)[:3, :3] @ a_M) - s["geo"].R
                     for T, _, _ in s["grid"]])


def class_spread(raw, cls):
    """Per region class: n, mean and sd of the grid residuals (the mean is the class's indentation)."""
    out = {}
    for k, name in enumerate(CLASSES):
        v = raw[cls == k]
        if len(v):
            out[name] = dict(n=int(len(v)), mean_mm=round(float(v.mean()), 4), sd_mm=round(float(v.std()), 4))
    return out


def own_fit(s):
    """The program's fit from the seed (its reg.T_AP and the plan's c_M at the time of the fit)."""
    reg = U.Registration(s["geo"])
    reg.load(s["rj"])
    reg.T_AP, reg.c_M = s["T_AP0"], s["geo"].c_M
    return U.fit_rod_registration(s["geo"], reg, s["pad"])


def basis(a_M0):
    e1 = np.cross(a_M0, [0.0, 0.0, 1.0] if abs(a_M0[2]) < 0.9 else [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(a_M0, e1)


def rod(geo, xc, xt):
    """The rod's axis point and direction (M) for the changes xc across the axis and the turns xt (rad)."""
    a0 = geo.axis_M
    e1, e2 = basis(a0)
    a = U.rot(U.EZ, math.degrees(xt[0])) @ U.rot(e1, math.degrees(xt[1])) @ a0
    return geo.c_M + xc[0] * e1 + xc[1] * e2, a / np.linalg.norm(a)


def touch_rms(s, T_AP, c_M, a_M):
    T_PB = U.inv(s["reg"].T_BI @ T_AP)
    r = [s["geo"].line_distance(U.apply(T_PB @ T, c_M), (T_PB @ T)[:3, :3] @ a_M) - s["geo"].R for T in s["Ts"]]
    return float(np.sqrt(np.mean(np.square(r)))), r


def nearest(geo, c, a):
    """The pad's point nearest the rod's axis (through c along a, P) within half_len, and the axis's distance from it."""
    r = geo.dense - c
    y = r @ a
    rp = r - y[:, None] * a[None]
    dist = np.where(np.abs(y) <= geo.half_len, np.linalg.norm(rp, axis=1), np.inf)
    k = int(np.argmin(dist))
    return geo.dense[k], float(dist[k])


def contact_shift(s, own, joint):
    """At each grid press aimed by the own fit, the pad's point nearest the rod under the joint fit against the own
    (mm along the surface) and the change in the rod's gap from the pad at the own fit's onset (mm)."""
    T_BI, T_EM, R = s["reg"].T_BI, s["reg"].T_EM, s["geo"].R
    T_BP_o, T_BP_j = T_BI @ own["T_AP"], T_BI @ joint["T_AP"]
    shift, gap = [], []
    for a in s["plan"]["actions"]:
        if a.get("stage") != "grid" or "target" not in a:
            continue
        tg = a["target"]
        n = np.asarray(tg["n"], float) / np.linalg.norm(tg["n"])
        c_P = np.asarray(tg["p"], float) + R * n          # the onset: the axis point R out from p (the rest less standoff)
        R_BM = T_BP_o[:3, :3] @ np.asarray(tg["R_PM"], float)
        T_BM = U.Tm(R_BM, U.apply(T_BP_o, c_P) - R_BM @ own["c_M"])
        To, Tj = U.inv(T_BP_o) @ T_BM, U.inv(T_BP_j) @ T_BM
        p_o, d_o = nearest(s["geo"], U.apply(To, own["c_M"]), To[:3, :3] @ own["axis_M"])
        p_j, d_j = nearest(s["geo"], U.apply(Tj, joint["c_M"]), Tj[:3, :3] @ joint["axis_M"])
        shift.append(float(np.linalg.norm(p_j - p_o)))
        gap.append(d_j - d_o)
    return np.array(shift), np.array(gap)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--share", choices=("both", "turn"), default="both", help="share c_M and the turn, or the turn only")
    ap.add_argument("--seat-scale", type=float, default=1.0, help="the pad priors widened by this factor")
    ap.add_argument("--loose", nargs="*", default=[], help="sessions (folder names or parts) whose pad priors are widened")
    ap.add_argument("--grid-level", type=float, default=None, help="add the sweeps' contacts at this level (N)")
    ap.add_argument("--grid-sd", type=float, default=0.05, help="the grid contacts' sd (mm)")
    ap.add_argument("--drop", nargs="*", default=[], help="touch ids left out")
    ap.add_argument("--more", nargs="*", default=[], help="sessions on a rod session's registration: their sweeps join "
                    "that seat's grid")
    ap.add_argument("--seeds", default=str(LAB), help="where the seed sessions' folders are")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    sig = U.REG_SIGMA_PROBE
    S = [load(sd, set(a.drop), a.seeds, a.grid_level) for sd in a.sessions]
    for sd in a.more:
        sd = Path(sd)
        seat = int(json.loads((sd / "plan.json").read_text())["geometry"]["seat"]["index"])
        base = [s for s in S if s["seat"] == seat]
        if len(base) != 1:
            raise SystemExit(f"{sd.name}: {len(base)} rod sessions on seat {seat + 1}")
        s, rj = base[0], json.loads((sd / "registration.json").read_text())
        for k in ("T_BI", "T_EM", "T_AP"):
            if np.abs(np.asarray(rj[k], float) - np.asarray(s["rj"][k], float)).max() > 1e-6:
                raise SystemExit(f"{sd.name}: its {k} is not {s['name']}'s")
        g = grid_poses(sd, a.grid_level, s["reg"].T_EM)
        s["grid"] += g
        s.setdefault("more", []).append(f"{sd.name} ({len(g)} sweeps)")
    nG = sum(len(s["grid"]) for s in S)
    g0 = S[0]["geo"]
    for s in S[1:]:
        g = s["geo"]
        if np.abs(g.c_M - g0.c_M).max() > 1e-9 or np.abs(g.axis_M - g0.axis_M).max() > 1e-9 or abs(g.R - g0.R) > 1e-9:
            raise SystemExit(f"{s['name']}: another rod (c_M0 {g.c_M}, axis_M0 {g.axis_M}, R {g.R})")
    own = []
    for s in S:
        f = own_fit(s)
        e = U.pose_error(s["reg"].T_AP, f["T_AP"])
        own.append(dict(T_AP=f["T_AP"], c_M=f["c_M"], axis_M=f["axis_M"], turn_deg=f["axis_turn_deg"],
                        turn_sd_deg=f["axis_turn_sd_deg"], rms=f["rms_pad_mm"], chi2=f["chi2"],
                        vs_session_mm_deg=[round(float(v), 4) for v in e],
                        vs_session_c_mm=float(np.linalg.norm(f["c_M"] - s["reg"].c_M)),
                        seed_change_mm_deg=[round(float(v), 4) for v in f["seat_change_mm_deg"]]))
        if s["grid"]:
            cls = np.array([g[1] for g in s["grid"]])
            own[-1]["grid"] = class_spread(grid_raw(s, f["T_AP"], f["c_M"], f["axis_M"]), cls)
    nS = len(S)
    nc = 2 if a.share == "both" else 2 * nS               # c_M across the axis: one, or one per session
    pri_c, pri_t = [1.0, 1.0], np.radians([4.0, 2.0])
    seat_sd = np.r_[np.radians(sig["seat_rot_deg"]), sig["seat_t_mm"]]
    wide = [a.seat_scale if not a.loose or any(n in s["name"] for n in a.loose) else 1.0 for s in S]
    sd = np.r_[np.tile(pri_c, nc // 2), pri_t, np.concatenate([seat_sd * w for w in wide]), [1.0] * (3 if nG else 0)]
    i_d = nc + 2 + 6 * nS                                 # the indentations at the grid level, per class

    def unpack(x):
        xt = x[nc:nc + 2]
        out = []
        for k, s in enumerate(S):
            xc = x[0:2] if a.share == "both" else x[2 * k:2 * k + 2]
            c, ax = rod(s["geo"], xc, xt)
            out.append((U.perturb(s["T_AP0"], x[nc + 2 + 6 * k:nc + 8 + 6 * k]), c, ax))
        return out

    def resid(x):
        r = []
        for s, (T_AP, c, ax) in zip(S, unpack(x)):
            T_PB = U.inv(s["reg"].T_BI @ T_AP)
            r += [(s["geo"].line_distance(U.apply(T_PB @ T, c), (T_PB @ T)[:3, :3] @ ax) - s["geo"].R) / sig["pad_mm"]
                  for T in s["Ts"]]
            if s["grid"]:
                cls = np.array([g[1] for g in s["grid"]])
                r += list((grid_raw(s, T_AP, c, ax) + x[i_d + cls]) / a.grid_sd)
        return np.r_[r, x / sd]

    x, r, cov = U.lm(resid, np.zeros(len(sd)))
    fits = unpack(x)
    sx = np.sqrt(np.clip(np.diag(cov), 0, None))
    turn, turn_sd = np.degrees(x[nc:nc + 2]), np.degrees(sx[nc:nc + 2])
    chi2_j, chi2_o = float(r @ r), float(sum(o["chi2"] for o in own))
    more = (2 + 2) * (nS - 1) if a.share == "both" else 2 * (nS - 1)
    if nG:
        print(f"grid at {a.grid_level} N: {nG} sweeps; the indentation per class (face, edge, flank) "
              f"{np.round(x[i_d:i_d + 3], 3).tolist()} mm (sd {np.round(sx[i_d:i_d + 3], 3).tolist()})")
    print(f"joint ({a.share}, pad priors x{wide}): the rod turned {turn[0]:+.2f} deg about z_M (sd {turn_sd[0]:.2f}) and {turn[1]:+.2f} across "
          f"(sd {turn_sd[1]:.2f}); c_M {np.round(fits[0][1], 3).tolist()}; chi2 joint {chi2_j:.1f}, sum of own "
          f"{chi2_o:.1f}, for {more} more constraints; {sum(len(s['pad']) for s in S)} touches", flush=True)
    rows = []
    for k, (s, o, (T_AP, c, ax)) in enumerate(zip(S, own, fits)):
        j = dict(T_AP=T_AP, c_M=c, axis_M=ax)
        rms_j, _ = touch_rms(s, T_AP, c, ax)
        sh, gp = contact_shift(s, o, j)
        e = U.pose_error(o["T_AP"], T_AP)
        q = lambda v: round(float(v), 4) if v is not None else None  # noqa: E731
        row = dict(session=s["name"], seed=s["seed_name"], seat=s["seat"] + 1, n_touch=len(s["pad"]),
                   more=s.get("more", []), n_grid=len(s["grid"]),
                   own=dict(c_M=np.round(o["c_M"], 4).tolist(), turn_deg=np.round(o["turn_deg"], 3).tolist(),
                            turn_sd_deg=np.round(o["turn_sd_deg"], 3).tolist(), rms_mm=q(o["rms"]),
                            pad_from_seed_mm_deg=o["seed_change_mm_deg"], vs_session_mm_deg=o["vs_session_mm_deg"],
                            vs_session_c_mm=q(o["vs_session_c_mm"])),
                   joint=dict(T_AP_vs_own_mm_deg=[q(v) for v in e], rms_mm=q(rms_j), c_M=np.round(c, 4).tolist(),
                              axis_M=np.round(ax, 6).tolist(),
                              c_M_vs_own_mm=np.round(c - o["c_M"], 4).tolist(),
                              pad_from_seed_mm_deg=[q(v) for v in U.pose_error(s["T_AP0"], T_AP)]),
                   contact_shift=dict(n_press=len(sh), median_mm=q(np.median(sh)) if len(sh) else None,
                                      p90_mm=q(np.percentile(sh, 90)) if len(sh) else None,
                                      max_mm=q(sh.max()) if len(sh) else None,
                                      gap_mean_mm=q(gp.mean()) if len(gp) else None,
                                      gap_absmax_mm=q(np.abs(gp).max()) if len(gp) else None),
                   T_AP_joint=np.round(T_AP, 6).tolist())
        if s["grid"]:
            cls = np.array([g[1] for g in s["grid"]])
            row["own"]["grid"] = o["grid"]
            row["joint"]["grid"] = class_spread(grid_raw(s, T_AP, c, ax) + x[i_d + cls], cls)
            print(f"  {s['name']} grid ({len(cls)}): own fit per class {o['grid']}; joint (less the indentation) "
                  f"{row['joint']['grid']}", flush=True)
        rows.append(row)
        print(f"{s['name']} (seat {s['seat'] + 1}, {len(s['pad'])} touches, seed {s['seed_name']}): own turn "
              f"{o['turn_deg'][0]:+.2f}/{o['turn_deg'][1]:+.2f} deg, rms {o['rms']:.3f} mm (registration.json: "
              f"{o['vs_session_mm_deg']} mm, deg); joint: T_AP {row['joint']['T_AP_vs_own_mm_deg']} mm, deg off the own, "
              f"rms {rms_j:.3f} mm; at {len(sh)} grid presses the contact moves median {row['contact_shift']['median_mm']}, "
              f"p90 {row['contact_shift']['p90_mm']}, max {row['contact_shift']['max_mm']} mm, gap mean "
              f"{row['contact_shift']['gap_mean_mm']} max {row['contact_shift']['gap_absmax_mm']} mm", flush=True)
    res = dict(what="the joint fit of one rod (its axis point c_M across the axis and its turn about z_M and across, or the "
                    "turn only) and each rod session's pad pose T_AP over the pass 1 touches "
                    "(scripts/ur5_rod_joint_registration.py), the program's residuals and priors (fit_rod_registration)",
               share=a.share, seat_scale=wide, sessions=list(a.sessions), more=list(a.more), dropped=list(a.drop),
               grid_level_N=a.grid_level, grid_sd_mm=a.grid_sd, n_grid=nG,
               indentation_mm=dict(zip(CLASSES, np.round(x[i_d:i_d + 3], 4).tolist())) if nG else None,
               turn_deg=np.round(turn, 3).tolist(), turn_sd_deg=np.round(turn_sd, 3).tolist(),
               chi2_joint=round(chi2_j, 2), chi2_own_sum=round(chi2_o, 2), more_constraints=more, rows=rows)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print("->", a.out)


if __name__ == "__main__":
    main()
