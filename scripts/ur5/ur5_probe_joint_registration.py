#!/usr/bin/env python3
"""One sphere centre for several probe sessions with the same tool: the joint fit of the probe's c_M and each session's
seat pose T_AP over their pass 1 touches (the lab, 15:12 on 11-03, after s2sph2's c_M stood about 0.5 mm across from
s1sph2b's, at sd 0.1 to 0.19, with the same 2 mm sphere and no swap).

Each session keeps its own registration (T_BI, the pad-fitted frame by --frame, and T_EM) and plan (the seat's CAD T_AP0,
the probe's c_M0 and the pad's surface). The residuals are the program's (ur5_tip_calibration.fit_probe_registration):
a pad touch's sphere centre in P against the pad's surface, less R, over REG_SIGMA_PROBE's pad_mm; each session's seat
pose about its CAD with the seat priors; the one c_M about the plans' with probe_mm. Two sessions of one seat keep a
pose each, so their difference is the seat's repeat under one c_M.

Per session, printed and written: the own fit as the program makes it (against its registration.json), the joint fit's
T_AP against the own, the touches' rms under both, and the aim change at the grid's presses. The aim change is where the
sphere's centre stands in P at a press's rest when the program aims by its own fit and the joint fit is right, against
the target: along n (positive: farther out, contact later) and across n.

    python3 scripts/ur5/ur5_probe_joint_registration.py SESSION [SESSION ...] [--drop ID ...] [--out OUT]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/sr2hand/driver/reskin5x/host"))
import ur5_tip_calibration as U  # noqa: E402

OUT = ROOT / "docs/experiments/20261030-inverted_rig_seat_block/data/20261103-strip_table_pads_joint_registration.json"


def load(sd, drop):
    sd = Path(sd)
    plan = json.loads((sd / "plan.json").read_text())
    geo = U.Geometry(plan)
    reg = U.Registration(geo)
    reg.load(json.loads((sd / "registration.json").read_text()))
    pad = [t for t in json.loads((sd / "touches.json").read_text()) if t["kind"] == "pad" and t["id"] not in drop]
    return dict(name=sd.name, plan=plan, geo=geo, reg=reg, pad=pad,
                Ts=[np.array(t["T_BE"], float) @ reg.T_EM for t in pad], seat=int(plan["geometry"]["seat"]["index"]))


def touch_rms(s, T_AP, c_M):
    T_PB = U.inv(s["reg"].T_BI @ T_AP)
    r = [s["geo"].surface_distance(U.apply(T_PB @ T, c_M))[0] - s["geo"].R for T in s["Ts"]]
    return float(np.sqrt(np.mean(np.square(r)))), r


def aim_change(s, own, joint):
    """The sphere's centre at each grid press's rest (P, joint fit) against its target, aimed by the own fit."""
    T_BI, T_EM, R = s["reg"].T_BI, s["reg"].T_EM, s["geo"].R
    T_BP_o, T_BP_j = T_BI @ own["T_AP"], T_BI @ joint["T_AP"]
    along, across = [], []
    for a in s["plan"]["actions"]:
        if a.get("stage") != "grid":
            continue
        tg = a["target"]
        n = np.asarray(tg["n"], float) / np.linalg.norm(tg["n"])
        d = np.asarray(tg["d"], float) / np.linalg.norm(tg["d"])
        c_P = np.asarray(tg["p"], float) + R * n - float(a.get("standoff", 1.5)) * d
        R_BM = T_BP_o[:3, :3] @ np.asarray(tg["R_PM"], float)
        T_BE = U.Tm(R_BM, U.apply(T_BP_o, c_P) - R_BM @ own["c_M"]) @ U.inv(T_EM)
        miss = U.apply(U.inv(T_BP_j) @ T_BE @ T_EM, joint["c_M"]) - c_P
        along.append(float(miss @ n))
        across.append(float(np.linalg.norm(miss - (miss @ n) * n)))
    return np.array(along), np.array(across)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--drop", nargs="*", default=[], help="touch ids left out (s1sph2's pad-tip-349-o35-a270 met the body)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    sig = U.REG_SIGMA_PROBE
    S = [load(sd, set(a.drop)) for sd in a.sessions]
    c_M0 = S[0]["geo"].c_M
    for s in S[1:]:
        if np.abs(s["geo"].c_M - c_M0).max() > 1e-9 or abs(s["geo"].R - S[0]["geo"].R) > 1e-9:
            raise SystemExit(f"{s['name']}: another probe (c_M0 {s['geo'].c_M}, R {s['geo'].R})")
    own = []
    for s in S:                                         # each session alone, as the program fits it
        f = U.fit_probe_registration(s["geo"], s["reg"], s["pad"])
        e = U.pose_error(s["reg"].T_AP, f["T_AP"])
        own.append(dict(T_AP=f["T_AP"], c_M=f["c_M"], rms=f["rms_pad_mm"], chi2=f["chi2"], sd_c=f["sd"][6:9],
                        vs_session_mm_deg=[round(float(v), 4) for v in e],
                        vs_session_c_mm=float(np.linalg.norm(f["c_M"] - s["reg"].c_M))))
    sd = np.r_[sig["probe_mm"], np.tile(np.r_[np.radians(sig["seat_rot_deg"]), sig["seat_t_mm"]], len(S))]

    def unpack(x):
        return c_M0 + x[0:3], [U.perturb(s["geo"].T_AP0, x[3 + 6 * k:9 + 6 * k]) for k, s in enumerate(S)]

    def resid(x):
        c_M, T_APs = unpack(x)
        r = []
        for s, T_AP in zip(S, T_APs):
            T_PB = U.inv(s["reg"].T_BI @ T_AP)
            r += [(s["geo"].surface_distance(U.apply(T_PB @ T, c_M))[0] - s["geo"].R) / sig["pad_mm"] for T in s["Ts"]]
        return np.r_[r, x / sd]

    x, r, cov = U.lm(resid, np.zeros(3 + 6 * len(S)))
    c_M, T_APs = unpack(x)
    sdx = np.sqrt(np.clip(np.diag(cov), 0, None))
    chi2_j, chi2_o = float(r @ r), float(sum(o["chi2"] for o in own))
    print(f"c_M joint {np.round(c_M, 3).tolist()} mm (sd {np.round(sdx[0:3], 3).tolist()}); chi2 joint {chi2_j:.1f}, "
          f"sum of own {chi2_o:.1f}, for {3 * (len(S) - 1)} more constraints; {sum(len(s['pad']) for s in S)} touches")
    rows = []
    for k, (s, o) in enumerate(zip(S, own)):
        j = dict(T_AP=T_APs[k], c_M=c_M)
        rms_j, res_j = touch_rms(s, T_APs[k], c_M)
        al, ac = aim_change(s, o, j)
        e = U.pose_error(o["T_AP"], T_APs[k])
        row = dict(session=s["name"], seat=s["seat"] + 1, n_touch=len(s["pad"]),
                   own=dict(c_M=np.round(o["c_M"], 4).tolist(), sd_c=np.round(o["sd_c"], 4).tolist(),
                            rms_mm=round(o["rms"], 4), vs_session_mm_deg=o["vs_session_mm_deg"],
                            vs_session_c_mm=round(o["vs_session_c_mm"], 4)),
                   joint=dict(T_AP_vs_own_mm_deg=[round(float(v), 4) for v in e], rms_mm=round(rms_j, 4),
                              c_M_vs_own_mm=np.round(c_M - o["c_M"], 4).tolist()),
                   aim_change=dict(n_press=len(al), along_mean_mm=round(float(al.mean()), 4) if len(al) else None,
                                   along_absmax_mm=round(float(np.abs(al).max()), 4) if len(al) else None,
                                   across_rms_mm=round(float(np.sqrt(np.mean(ac ** 2))), 4) if len(ac) else None,
                                   across_max_mm=round(float(ac.max()), 4) if len(ac) else None),
                   T_AP_joint=np.round(T_APs[k], 6).tolist())
        rows.append(row)
        print(f"{s['name']} (seat {s['seat'] + 1}, {len(s['pad'])} touches): own c_M {row['own']['c_M']} sd "
              f"{row['own']['sd_c']}, rms {o['rms']:.3f} mm (registration.json: {o['vs_session_mm_deg']} mm, deg; c_M "
              f"{o['vs_session_c_mm']:.3f} mm); joint: T_AP {row['joint']['T_AP_vs_own_mm_deg']} mm, deg off the own, rms "
              f"{rms_j:.3f} mm; aim at {len(al)} presses: along n mean {row['aim_change']['along_mean_mm']} max "
              f"{row['aim_change']['along_absmax_mm']}, across rms {row['aim_change']['across_rms_mm']} max "
              f"{row['aim_change']['across_max_mm']} mm", flush=True)
    seats = {}
    for k, s in enumerate(S):
        seats.setdefault(s["seat"], []).append(k)
    repeat = {f"seat {i + 1}": [round(float(v), 4) for v in U.pose_error(T_APs[ks[0]], T_APs[ks[1]])]
              for i, ks in seats.items() if len(ks) == 2}
    if repeat:
        print("one seat's two sessions under the joint c_M (mm, deg):", repeat)
    res = dict(what="the joint fit of one sphere centre c_M and each session's seat pose T_AP over the pass 1 touches "
                    "(scripts/ur5/ur5_probe_joint_registration.py), the program's residuals and priors (REG_SIGMA_PROBE)",
               sessions=list(a.sessions), dropped=list(a.drop), c_M_joint=np.round(c_M, 4).tolist(),
               sd_c_M_joint=np.round(sdx[0:3], 4).tolist(), chi2_joint=round(chi2_j, 2), chi2_own_sum=round(chi2_o, 2),
               more_constraints=3 * (len(S) - 1), seat_repeat_mm_deg=repeat, rows=rows)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print("->", a.out)


if __name__ == "__main__":
    main()
