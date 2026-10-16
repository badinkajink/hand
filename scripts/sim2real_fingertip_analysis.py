#!/usr/bin/env python3
"""Summaries of the fingertip deployment-readiness study (docs/handoff/20261016-sim2real_fingertip_readiness.md) from its
rows, for the result page (scripts/sim2real_fingertip_readiness_page.py).

  prints     print/*.json sheets: per project the recipe, pauses, time and the G-code checks
  fe         fe/fem.jsonl: pad modulus, approach under the 25 mm tool at 0.5, 1, 3 N, face compliance over 15 positions,
             magnet travel per newton
  variants   assets/.../<variant>/README.json: shape, modulus, printed mass
  interference  scenes/interference.json: least clearance of the board and its tab to each CAD part over the joint ranges
  bed        bed/*.jsonl: per variant and arm the approach, twist-onset arm and creep, spin arm, pull onset and creep
  cell10     cell10/<variant>.jsonl: per arm and candidate grasp the seeds held of six, the cell score (the grasp with the
             best held turn on seed 0, its mean held turn on seeds 1-2) and the median RMS turn-angle difference to Drake
  replays    replays/traces.jsonl: per variant the held replays of 24 and, pairing each placement across arms
             (curved_tips_traj_analysis.compare), the RMS tool-angle, normal-force and friction-force differences
  cells      cells/hom.jsonl, cells/hom_noboard.jsonl, cells/icem.jsonl: per variant, cell and board arm the cell score
             (median held turn over seeds 0-2 at the grasp with the largest median), held and three-pad counts, peak joint
             torque, command rate and the runs in which the board touched the cylinder
  policies   policies/transfer.jsonl: per scene and policy the held rollouts of 64, held cosine and grip
  sensing    sensing/sensing.json

    python3 scripts/sim2real_fingertip_analysis.py
Out: docs/experiments/20261016-sim2real_fingertip_readiness/summary.json
"""
from __future__ import annotations

import json
import statistics as st
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import curved_tips_traj_analysis as CTA  # noqa: E402

D = ROOT / "docs/experiments/20261016-sim2real_fingertip_readiness"
ASSETS = ROOT / "assets/mjcf/experimental/20261016-sim2real_fingertip_readiness"
VARIANTS = ["block_e10", "block4_g30", "block5_g30", "block4w1_g30", "cap_g30", "ell16_g30", "ell11_g30",
            "block4_g50", "block5_g50", "block4w1_g50", "cap_g50", "ell16_g50", "ell11_g50"]


def rows(path):
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in open(p):
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def med(x):
    x = [v for v in x if v is not None and np.isfinite(v)]
    return round(float(np.median(x)), 4) if x else None


def q(x, p):
    x = [v for v in x if v is not None and np.isfinite(v)]
    return round(float(np.percentile(x, p)), 4) if x else None


def prints():
    out = []
    for f in sorted((D / "print").glob("20261016-siraya85a_*.json")):
        r = json.loads(f.read_text())
        ck = r["slice"]["check"]
        out.append(dict(name=r["name"], tip=r.get("tip"), layout=r.get("layout"), infill=r.get("infill_percent"),
                        walls=r.get("pocket_walls"), magnets=(r.get("magnet") or {}).get("count"),
                        depth_mm=(r.get("magnet") or {}).get("top_depth_mm"), wall_min_mm=r.get("wall_min_mm"),
                        pauses=[p["print_z"] for p in r.get("pauses", [])], print_time=ck["print_time"],
                        filament_g=ck.get("filament_g"), islands_max=r["slice"].get("islands_max"),
                        nozzle_C=r["slice"]["material"]["part_nozzle_setpoints_C"],
                        bed_C=[b for b in r["slice"]["material"]["bed_setpoints_C"] if b > 0],
                        flow_max=r["slice"]["material"]["volumetric_flow_mm3_s"]["max"],
                        project=Path(r["project"]).name))
    return out


def fe():
    out = []
    for r in rows(D / "fe/fem.jsonl"):
        pf = r["per_force"]
        face = [p["normal_face_um_per_N"] for p in r["patches"]]
        sy = [p["shear_y_face_um_per_N"] for p in r["patches"]]
        sz = [p["shear_z_face_um_per_N"] for p in r["patches"]]
        tr = [float(np.linalg.norm(t)) for t, _ in pf["1"].get("magnets", [])]
        out.append(dict(layout=r["layout"], recipe=r["recipe"], E_star_MPa=round(r["E_best_MPa"], 3),
                        approach_um={k: round(v["delta_um"], 2) for k, v in pf.items()},
                        face_normal_um_per_N=[round(min(face), 1), round(float(np.median(face)), 1), round(max(face), 1)],
                        face_shear_y_um_per_N=[round(min(sy), 1), round(float(np.median(sy)), 1), round(max(sy), 1)],
                        face_shear_z_um_per_N=[round(min(sz), 1), round(float(np.median(sz)), 1), round(max(sz), 1)],
                        magnet_travel_um_per_N=round(float(np.mean(tr)), 2) if tr else None))
    return out


def variants():
    out = {}
    for v in VARIANTS:
        f = ASSETS / v / "README.json"
        if f.exists():
            r = json.loads(f.read_text())
            out[v] = dict(key=r["key"], shape=r["shape"], E_MPa=round(r["pad_E_Pa"] / 1e6, 3), E_src=r["E_src"],
                          infill=r.get("infill"), magnets=r.get("magnets"), walls=r.get("walls"), board=r.get("board"),
                          tip_g=r["tip_mass"]["total_g"], tpu_g=r["tip_mass"]["tpu_g"], magnets_g=r["tip_mass"]["magnets_g"])
    return out


def bed():
    out = defaultdict(dict)
    for case in ("static", "T2", "spin", "T1"):
        for r in rows(D / "bed" / f"{case}.jsonl"):
            if r.get("status") != "complete":
                continue
            k = f"{r['variant']}|{r['arm']}"
            o = out[k]
            o.update(variant=r["variant"], arm=r["arm"], E_MPa=round(r["E_Pa"] / 1e6, 3))
            if case == "static":
                o.setdefault("approach_um", {})[f"{r['N']:g}"] = round(r["approach_um"], 2)
                o.setdefault("touching", {})[f"{r['N']:g}"] = r.get("touching_per_side")
            elif case == "T2":
                o.setdefault("onset_arm_mm", {})[f"{r['N']:g}"] = round(r.get("rbar_onset_mm", float("nan")), 3)
                o.setdefault("twist_creep_deg_s", {})[f"{r['N']:g}"] = round(r.get("creep_deg_s", float("nan")), 4)
            elif case == "spin":
                o["spin_arm_mm"] = round(r.get("rbar_spin_mm", float("nan")), 3)
            else:
                o["pull_mu_eff"] = round(r.get("mu_eff", float("nan")), 3)
                o["pull_creep_mm_s"] = round(r.get("creep_mm_s", float("nan")), 4)
    return dict(out)


def cell10():
    out = {}
    for v in VARIANTS:
        R = [r for r in rows(D / "cell10" / f"{v}.jsonl") if r.get("status") == "ok"]
        if not R:
            continue
        arms = {}
        by = defaultdict(lambda: defaultdict(dict))
        for r in R:
            by[r["arm"]][r["cand"]][r["seed"]] = r
        for arm, cc in by.items():
            held = {c: sum(1 for s in ss.values() if s["held"]) for c, ss in cc.items()}
            s0 = {c: ss[0]["held_turn_deg"] for c, ss in cc.items() if 0 in ss}
            best = max(s0, key=lambda c: (s0[c], -c)) if s0 else None
            score = (float(np.mean([cc[best][s]["held_turn_deg"] for s in (1, 2) if s in cc[best]]))
                     if best is not None else None)
            arms[arm] = dict(held_by_cand={str(c): held[c] for c in sorted(held)}, held_total=sum(held.values()),
                             runs=sum(len(ss) for ss in cc.values()), held_seed0=sum(1 for c in cc if 0 in cc[c] and cc[c][0]["held"]),
                             grasp=best, cell_score_deg=None if score is None else round(score, 1))
        # turn traces against Drake on the same candidate and seed
        d = {}
        for arm in ("v1", "v2"):
            diffs = []
            for c, ss in by.get(arm, {}).items():
                for s, r in ss.items():
                    rd = by.get("drake", {}).get(c, {}).get(s)
                    if rd is None or not r.get("trace") or not rd.get("trace"):
                        continue
                    ta = {round(x["t"], 2): x["th"] for x in r["trace"]}
                    tb = {round(x["t"], 2): x["th"] for x in rd["trace"]}
                    common = sorted(set(ta) & set(tb))
                    if len(common) > 5:
                        diffs.append(float(np.sqrt(np.mean([(ta[t] - tb[t]) ** 2 for t in common]))))
            d[arm] = med(diffs)
        out[v] = dict(arms=arms, rms_turn_vs_drake_deg=d)
    return out


def replays():
    """Per variant: held replays per arm, and per pair of arms the trajectory differences on the same placement."""
    R = defaultdict(dict)
    meta = defaultdict(lambda: defaultdict(list))
    with open(D / "replays/traces.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("status") not in ("complete", "ejected"):
                continue
            R[(r["variant"], r["hand"], r["seed"])][r["arm"]] = r
            meta[r["variant"]][r["arm"]].append(r)
    out = {}
    for v, arms in meta.items():
        o = {"held": {a: f"{sum(1 for r in rs if r.get('held_end'))}/{len(rs)}" for a, rs in arms.items()},
             "turn_end_median_deg": {a: med([r.get("turn_end_deg") for r in rs if r.get("held_end")]) for a, rs in arms.items()},
             "us_per_step": {a: med([r.get("us_per_step_median") for r in rs]) for a, rs in arms.items()}}
        for a, b in (("plan", "drake"), ("rl", "plan"), ("rl", "drake")):
            cs = []
            for (vv, h, s), aa in R.items():
                if vv == v and a in aa and b in aa:
                    c = CTA.compare(aa[a], aa[b])
                    if c:
                        cs.append(c)
            if cs:
                o[f"{a}-{b}"] = dict(n=len(cs), rms_dang_deg=med([c["rms_dang_deg"] for c in cs]),
                                     rms_dang_q=[q([c["rms_dang_deg"] for c in cs], 25), q([c["rms_dang_deg"] for c in cs], 75)],
                                     rms_dN_N=med([c["rms_dN_N"] for c in cs]), rms_dT_N=med([c["rms_dT_N"] for c in cs]),
                                     held_agree=sum(c["held_agree"] for c in cs), turn_within_3=sum(c["turn_within_3"] for c in cs))
        # spread between placements: the plan arm, every pair of seeds of one hand
        sp = []
        for h in {k[1] for k in R if k[0] == v}:
            for s1, s2 in ((0, 1), (0, 2), (1, 2)):
                a1, a2 = R.get((v, h, s1), {}).get("plan"), R.get((v, h, s2), {}).get("plan")
                if a1 and a2:
                    c = CTA.compare(a1, a2)
                    if c:
                        sp.append(c["rms_dang_deg"])
        o["plan_seed_spread_deg"] = med(sp)
        out[v] = o
    return out


def cells():
    out = {}
    for tag in ("hom", "hom_noboard"):
        by = defaultdict(lambda: defaultdict(list))
        for r in rows(D / "cells" / f"{tag}.jsonl"):
            if r.get("status") == "ok":
                by[(r["variant"], f"{r['shape']}:{r['d_mm']:g}:{r['t']:g}")][r["cand"]].append(r)
        for (v, cell), cc in by.items():
            best = max(sorted(cc), key=lambda c: float(np.median([x["held_turn_deg"] for x in cc[c]])))
            rs = cc[best]
            allr = [x for c in cc.values() for x in c]
            out[f"{tag}|{v}|{cell}"] = dict(
                arm=tag, variant=v, cell=cell, grasp=best, seeds=len(rs),
                cell_score_deg=round(float(np.median([x["held_turn_deg"] for x in rs])), 1),
                held=sum(x["held"] for x in rs), three_pads_end=sum(1 for x in rs if x["pads_loaded_end"] == 3),
                turn_end_deg=[round(x["turn_end_deg"], 1) for x in rs], peak_tau_Nm=round(max(x["peak_tau_Nm"] for x in rs), 3),
                tau_at_limit=med([x["tau_at_limit_frac"] for x in rs]), cmd_rate_rms=med([x["cmd_rate_rms"] for x in rs]),
                cmd_rate_peak=round(max(x["cmd_rate_peak"] for x in rs), 2), three_pad_frac=med([x["three_pad_frac"] for x in rs]),
                board_runs=sum(1 for x in allr if x["nonpad_contact_ticks"].get("board")), runs=len(allr),
                board_runs_best=sum(1 for x in rs if x["nonpad_contact_ticks"].get("board")),
                held_all=sum(x["held"] for x in allr))
    for r in rows(D / "cells/icem.jsonl"):
        if r.get("status") == "ok":
            out[f"icem|{r['variant']}|{r['shape']}:{r['d_mm']:g}:{r['t']:g}|s{r['seed']}"] = dict(
                arm="icem", variant=r["variant"], cell=f"{r['shape']}:{r['d_mm']:g}:{r['t']:g}", seed=r["seed"],
                grasp=r["cand"], held=r["held"], turn_end_deg=r["turn_end_deg"], held_turn_deg=r["held_turn_deg"],
                fingers_loaded_end=r["fingers_loaded_end"], three_frac=r.get("three_frac"), peak_tau_Nm=r["peak_tau_Nm"],
                cmd_rate_rms=r.get("cmd_rate_rms"), board_ticks=r["nonpad_contact_ticks"].get("board", 0), wall_s=r["wall_s"])
    return out


def policies():
    out = {}
    for r in rows(D / "policies/transfer.jsonl"):
        if r.get("status") == "ok":
            out[f"{r['scene']}|{r['tag']}"] = {k: r.get(k) for k in ("scene", "tag", "arm", "seed", "n_held", "held_cos_mean",
                                                                     "grip_mean", "grip_N", "cos_final_median")}
    return out


def main():
    S = dict(prints=prints(), fe=fe(), variants=variants(), bed=bed(), cell10=cell10(), cells=cells(), policies=policies(),
             when=time.strftime("%Y-%m-%d %H:%M"), script="scripts/sim2real_fingertip_analysis.py")
    if (D / "replays/traces.jsonl").exists():
        S["replays"] = replays()
    for name in ("scenes/interference.json", "sensing/sensing.json"):
        if (D / name).exists():
            S[name.split("/")[0]] = json.loads((D / name).read_text())
    (D / "summary.json").write_text(json.dumps(S, indent=1, default=float))
    print(f"summary.json: {len(S['prints'])} prints, {len(S['fe'])} FE rows, {len(S['bed'])} bed arms, {len(S['cell10'])} cell10, "
          f"{len(S['cells'])} cell rows, {len(S.get('replays', {}))} replay variants, {len(S['policies'])} policy rows")


if __name__ == "__main__":
    main()
