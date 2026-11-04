"""The sweep test of 11-03 (s2sweep: the 2 mm sphere's face sweeps on seat 2, v3 ID 5), summarized per sweep from the
session's own streams. The HEX-E's force is tared at the press's rest and turned into B by the flange's pose, then split
along the pad's normal n and across it. Per level hold: the push along n and across. Per offset's hold: the push along
and across, the across push's direction against the offset's (-1 when the pad pulls the sphere back), the flange's step
from the level's stop across n, and the across push over that step (N/mm). Per slide: the push across over the push
along in the move's second half and at its hold, the friction coefficient once the contact slips. Per reference press:
the push and the flange's travel along d at each level, to compare the test's first and last. Reads only the actions'
windows (memory maps); run it at low priority beside a run:

    nice -n 19 ionice -c3 /usr/bin/python3 docs/experiments/20261022-ur5_tip_calibration/lab/sweep_test_summary.py SESSION OUT.json [PLAN...]

A resumed run does not rewrite the session's plan.json, so the plans of its later runs (the night of 11-03's halves
and extras) come as PLAN; each overrides plan.json's actions of the same id, so an action run under an earlier plan
(seat 2's full sweeps before the switch) reports the later plan's planned counts.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np


LOADED_N = 0.1                                  # the push along n under which a ratio means nothing


def window(t, a, b):
    return np.searchsorted(t, a), np.searchsorted(t, b)


def main():
    S, dst = Path(sys.argv[1]), Path(sys.argv[2])
    reg = json.loads((S / "registration.json").read_text())
    K = np.asarray(reg["K"], float)
    R_BP = (np.asarray(reg["T_BI"], float) @ np.asarray(reg["T_AP"], float))[:3, :3]
    acts = {a["id"]: a for a in json.loads((S / "plan.json").read_text())["actions"]}
    for f in sys.argv[3:]:
        acts.update({a["id"]: a for a in json.loads(Path(f).read_text())["actions"]})
    ft = np.memmap(S / "ft.bin", "<f8", "r").reshape(-1, 9)
    rob = np.memmap(S / "robot.bin", "<f8", "r").reshape(-1, 15)
    ft_t, rob_t = ft[:, 0], rob[:, 0]

    def f_B(a, b, tare, R_BE):                  # the force on the sphere in B (N), rows in [a, b]
        i, j = window(ft_t, a, b)
        return np.linalg.solve(K, (ft[i:j, 3:6] - tare[:3]).T).T @ R_BE.T

    def p_B(a, b):                              # the flange's mean place in B (mm) in [a, b]
        i, j = window(rob_t, a, b)
        return rob[i:j, 8:11].mean(0) * 1e3

    def split(f, n_B):
        fn = f @ n_B
        ft_v = f - np.outer(fn, n_B)
        return fn, ft_v

    res = dict(what=__doc__.split("\n\n")[0], session=str(S), sweeps={}, refs={})
    for line in open(S / "events.jsonl"):
        e = json.loads(line)
        if e.get("event") != "end" or e.get("status") != "ok" or e.get("id") not in acts:
            continue
        a, rec = acts[e["id"]], e.get("rec") or {}
        if not rec.get("levels"):
            continue
        tg = a["target"]
        n_P = np.asarray(tg["n"], float) / np.linalg.norm(tg["n"])
        n_B = R_BP @ n_P
        tare = np.asarray(rec["tare"], float)
        R_BE = np.asarray(rec["T_rest"], float)[:3, :3]
        levels, stop = [], {}
        for lv in rec["levels"]:
            fn, ft_v = split(f_B(*lv["t"], tare, R_BE), n_B)
            stop[lv["N"]] = p_B(*lv["t"])
            levels.append(dict(N=lv["N"], along=round(float(fn.mean()), 3),
                               across=round(float(np.linalg.norm(ft_v, axis=1).mean()), 3)))
        if a["kind"] == "press":
            d_B = R_BP @ (np.asarray(tg["d"], float) / np.linalg.norm(tg["d"]))
            p0 = p_B(*rec["t_rest0"])
            for lv in levels:
                lv["travel_mm"] = round(float((stop[lv["N"]] - p0) @ d_B), 4)
            res["refs"][e["id"]] = dict(levels=levels)
            continue
        if a["kind"] != "sweep":                    # twists and the rest: not summarized here
            continue
        t1_P = np.asarray(tg["t1"], float)
        t1_P = (t1_P - (t1_P @ n_P) * n_P) / np.linalg.norm(t1_P - (t1_P @ n_P) * n_P)
        t1_B, t2_B = R_BP @ t1_P, R_BP @ np.cross(n_P, t1_P)

        def u_of(az):
            return math.cos(math.radians(az)) * t1_B + math.sin(math.radians(az)) * t2_B

        offs = []
        for o in rec.get("offsets", []):
            u = u_of(o["az_deg"])
            R = np.asarray(o["T_BE"], float)[:3, :3]
            fn, ft_v = split(f_B(*o["t"], tare, R), n_B)
            fm = ft_v.mean(0)
            step = p_B(*o["t"]) - stop.get(o["N"], np.full(3, np.nan))
            step_t = float(np.linalg.norm(step - (step @ n_B) * n_B))
            offs.append(dict(N=o["N"], r_mm=o["r_mm"], az_deg=o["az_deg"], along=round(float(fn.mean()), 3),
                             across=round(float(np.linalg.norm(fm)), 3),
                             cos_u=round(float(fm @ u / max(np.linalg.norm(fm), 1e-9)), 3),
                             step_mm=round(step_t, 4), step_n_mm=round(float(step @ n_B), 4),
                             k_N_mm=round(float(np.linalg.norm(fm) / step_t), 2) if step_t > 0.01 else None))
        slides = []
        for s in rec.get("slides", []):
            u = u_of(s["az_deg"])
            R = np.asarray(s["T_end"], float)[:3, :3]
            a0, b0 = s["t_out"]
            fn, ft_v = split(f_B(0.5 * (a0 + b0), b0, tare, R), n_B)
            hn, ht_v = split(f_B(*s["t_hold"], tare, R), n_B)
            on = fn > LOADED_N                          # a slide may leave the pad where the face curves away
            ratio = np.linalg.norm(ft_v[on], axis=1) / fn[on]
            hm = ht_v.mean(0)
            slides.append(dict(az_deg=s["az_deg"], mm=s["mm"], at_N=s["at_N"],
                               along_move=round(float(fn.mean()), 3),
                               mu_move=round(float(np.median(ratio)), 3) if on.mean() > 0.5 else None,
                               along_hold=round(float(hn.mean()), 3), across_hold=round(float(np.linalg.norm(hm)), 3),
                               mu_hold=round(float(np.linalg.norm(hm) / hn.mean()), 3) if hn.mean() > LOADED_N else None,
                               cos_u=round(float(hm @ u / max(np.linalg.norm(hm), 1e-9)), 3),
                               travel_mm=round(float(np.linalg.norm(p_B(*s["t_hold"]) - stop.get(s["at_N"], np.nan))), 3)))
        n_off = sum(len(x) for x in a.get("offsets") or [])
        res["sweeps"][e["id"]] = dict(levels=levels, offsets=offs, slides=slides, skips=rec.get("skips", []),
                                      planned=dict(offsets=n_off, slides=len(a.get("slides") or [])))
    dst.write_text(json.dumps(res, indent=1))

    for k, r in res["refs"].items():
        print(f"{k}: " + ", ".join(f"{lv['N']:g} N at {lv['travel_mm']:.3f} mm" for lv in r["levels"]))
    for k, r in res["sweeps"].items():
        o, sl = r["offsets"], r["slides"]
        print(f"{k}: offsets {len(o)} of {r['planned']['offsets']}, slides {len(sl)} of {r['planned']['slides']}, "
              f"skips {len(r['skips'])}; levels " + ", ".join(f"{lv['N']:g} N {lv['along']:.2f}/{lv['across']:.2f}"
                                                            for lv in r["levels"]))
        for N in sorted({x["N"] for x in o}):
            for rr in sorted({x["r_mm"] for x in o if x["N"] == N}):
                g = [x for x in o if x["N"] == N and x["r_mm"] == rr]
                ks = [x["k_N_mm"] for x in g if x["k_N_mm"] is not None]
                print(f"   {N:g} N r {rr:g}: across {np.mean([x['across'] for x in g]):.2f} N "
                      f"(along {np.mean([x['along'] for x in g]):.2f}), cos_u {np.mean([x['cos_u'] for x in g]):+.2f}, "
                      f"step {np.mean([x['step_mm'] for x in g]):.3f} mm" + (f", {np.median(ks):.1f} N/mm" if ks else ""))
        for x in sl:
            mu = [f"{m:.2f}" if m is not None else "-" for m in (x["mu_move"], x["mu_hold"])]
            print(f"   slide az {x['az_deg']:g} at {x['at_N']:g} N: mu {mu[0]} moving, {mu[1]} held "
                  f"(along {x['along_move']:.2f}/{x['along_hold']:.2f} N), cos_u {x['cos_u']:+.2f}")
        for x in r["skips"]:
            print(f"   skip: {x}")
    print("->", dst)


if __name__ == "__main__":
    main()
