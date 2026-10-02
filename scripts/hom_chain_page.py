#!/usr/bin/env python3
"""Build docs/experiments/20261002-hom_chain/20261002-hom_screwdriver_chain.html from the study's jsonl files.

    python3 scripts/hom_chain_page.py

Reads chain.jsonl, exp1.jsonl and perf.jsonl written by scripts/hom_chain_study.py; tables reuse
scripts/hom_contact_patch_page.py's helpers. Every number in the prose is computed here.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import hom_contact_patch_page as R  # noqa: E402

D = os.path.join(ROOT, "docs/experiments/20261002-hom_chain")
OUT = os.path.join(D, "20261002-hom_screwdriver_chain.html")
TPL = os.path.join(ROOT, "scripts/hom_chain_page.template.html")
LINKS = {"RIG_URL": "https://claude.ai/artifact/Rvfw1yQFgfWV8PdTvv7jsc",
         "RIG_PATH": "docs/experiments/20261001-hom_contact_patch/20261001-hom_pinch_contact_models.html",
         "BRAKE_URL": "https://claude.ai/artifact/DJLxG63vLZXs2Ey6hiCBuc",
         "BRAKE_PATH": "docs/experiments/20261001-hom_hand_brake/20261001-hom_hand_brake.html"}
LBL = {"mj:point3": "MuJoCo point contact, condim&#160;3", "mj:point4s": "MuJoCo condim&#160;4, &#956;<sub>t</sub> rescheduled",
       "mj:spheres:s2:rs0.75:tr0.02": "MuJoCo 2&#8202;mm sphere fingertips (51 per pad)",
       "mj:spheres:s1:rs0.75:tr0.02": "MuJoCo 1&#8202;mm sphere fingertips (205 per pad)",
       "mj:spheres:s0.5:rs0.75:tr0.02": "MuJoCo 0.5&#8202;mm sphere fingertips (819 per pad)",
       "drake:point": "Drake point contact", "drake:hydro:rt0.01": "Drake hydroelastic, 0.01&#8202;s relaxation"}
NOM_ORDER = ["dhy_closed", "s1_closed", "p4s_closed", "p4s_open", "dpt_closed", "mp3_closed"]
load = lambda f: [json.loads(l) for l in open(os.path.join(D, f)) if l.strip()]  # noqa: E731
CH, E1, PF = load("chain.jsonl"), load("exp1.jsonl"), load("perf.jsonl")
NOM = {r["key"]: r for r in CH if r["trial"]["seed"] == 0}
BAT = [r for r in CH if r["trial"]["seed"] != 0]


def yn(v):
    return ("yes", "cell c0") if v else ("no", "cell c3")


def deg(v, nd=1):
    return "&#8211;" if v is None else f"{v:.{nd}f}&#176;"


def nominal_table():
    head = ["model", "brake", "pick", "swing ends", "peak rate", "after squeeze", "inserted", "lower end tilt",
            "axial slip", "chain"]
    body = []
    for k in NOM_ORDER:
        r = NOM[k]
        live = r["pick_ok"] and r["phi_end"] is not None and 0 < r["phi_end"] < 180
        body.append([LBL[r["spec"]], {"open": "open loop", "closed": "closed loop"}[r["brake"]], yn(r["pick_ok"]),
                     deg(r["phi_end"]) if live else "swung free at lift",
                     f"{r['rate_max_dps']:.0f}&#176;/s" if live else "&#8211;",
                     deg(r["phi_hold_end"]) if live else "&#8211;",
                     f"{r['end_depth_mm']:.1f}&#8202;mm" if r["insert_ok"] else "missed",
                     deg(r["end_tilt_deg"]) if r["insert_ok"] else "&#8211;",
                     f"{r['slip_mm']:.2f}&#8202;mm" if live else "&#8211;", yn(r["chain_ok"])])
    return R.table(head, body, cls_num={3, 4, 5, 6, 7, 8}) + (
        '<p class="note" style="font-size:13.5px;color:var(--ink3)">Swing ends = angle about the pinch axis when the brake '
        'phase ends (90&#176; = hanging). After squeeze = the same angle once the pad force is at the hold value. Inserted = depth of '
        'the lower end below the block top when the palm stops (goal 20&#8202;mm, hole 30&#8202;mm deep). Lower end tilt = tool axis from '
        'vertical, in and out of the swing plane. Axial slip = travel of the tool along its axis through the pinch from the brake '
        'onward. Chain = pick, swing within 5&#176;, hold within 5&#176;, and inserted 15&#8202;mm or more on two pads.</p>')


def batch_table():
    head = ["model", "brake", "pick", "swing within 5&#176;", "hold within 5&#176;", "inserted", "chain", "swing end, completed"]
    body = []
    for spec in ("mj:point4s", "mj:spheres:s1:rs0.75:tr0.02"):
        for brake in ("open", "closed"):
            rs = [r for r in BAT if r["spec"] == spec and r["brake"] == brake]
            c = lambda key: f"{sum(bool(r[key]) for r in rs)}/{len(rs)}"  # noqa: E731
            ends = [r["phi_end"] for r in rs if r["chain_ok"]]
            body.append([LBL[spec], {"open": "open loop", "closed": "closed loop"}[brake], c("pick_ok"), c("brake_ok"),
                         c("hold_ok"), c("insert_ok"), c("chain_ok"),
                         ", ".join(f"{v:.1f}&#176;" for v in ends) or "&#8211;"])
    return R.table(head, body, cls_num={2, 3, 4, 5, 6, 7})


def batch_prose():
    def get(spec, brake, seed):
        return next(r for r in BAT if r["spec"] == spec and r["brake"] == brake and r["trial"]["seed"] == seed)
    clean = sorted({r["trial"]["seed"] for r in BAT if r["pick_ok"] and r["spec"] == "mj:point4s" and r["brake"] == "closed"})
    out = [f"Seeds {', '.join(map(str, clean))} pick cleanly."]
    for seed in clean:
        parts = []
        for brake in ("closed", "open"):
            rs = [get(s, brake, seed) for s in ("mj:point4s", "mj:spheres:s1:rs0.75:tr0.02")]
            if all(r["chain_ok"] for r in rs):
                parts.append(f"the {brake}-loop brake completes the chain in both MuJoCo models "
                             f"({rs[0]['phi_end']:.1f}&#176; and {rs[1]['phi_end']:.1f}&#176;)")
            elif all(r["slip_mm"] is not None and r["slip_mm"] > 20 for r in rs):
                parts.append(f"the {brake}-loop brake drops the tool during the swing in both")
            else:
                parts.append(f"the {brake}-loop brake ends the swing at {rs[0]['phi_end']:.1f}&#176; and {rs[1]['phi_end']:.1f}&#176; "
                             f"(chain {'complete' if rs[0]['chain_ok'] else 'failed'} in condim&#160;4, "
                             f"{'complete' if rs[1]['chain_ok'] else 'failed'} with spheres)")
        t = get("mj:point4s", "closed", seed)["trial"]
        out.append(f"On seed {seed} (&#956;&#8202;{t['mu']:.2f}, mass &#215;{t['mscale']:.2f}) {parts[0]}, and {parts[1]}.")
    return " ".join(out)


def exp1_table():
    comps = ["sep", "slide_y", "slide_z", "spin", "roll_y", "roll_z"]
    head = ["servo"] + [f"{c.replace('_', ' ')} gain" for c in comps] + ["RMSE angular", "RMSE linear", "closest approach"]
    body = []
    for r in E1:
        ps = r["per_step"]
        body.append([{"cal": "calibrated, kp&#8202;0.5", "stiff": "template, kp&#8202;30"}[r["plant"]]]
                    + [f"{ps[c]['commanded_gain']:.2f}" for c in comps]
                    + [f"{r['rmse_ang_dps']:.1f}&#8202;&#176;/s", f"{r['rmse_lin_mmps']:.1f}&#8202;mm/s", f"{r['min_gap_mm']:.1f}&#8202;mm"])
    return R.table(head, body, cls_num=set(range(1, 10))) + (
        '<p class="note" style="font-size:13.5px;color:var(--ink3)">Gain = median achieved/commanded on the stepped component '
        'while it is stepped. RMSE over the whole 7.8&#8202;s sequence, all three angular or all three linear components. '
        'Closest approach = smallest tip-to-cylinder distance; contact-free when positive. 500&#8202;Hz control as in the paper.</p>')


def exp1_prose():
    g = {r["plant"]: r["per_step"] for r in E1}
    lin = {p: np.mean([g[p][c]["commanded_gain"] for c in ("sep", "slide_y", "slide_z")]) for p in g}
    ang = {p: np.mean([g[p][c]["commanded_gain"] for c in ("spin", "roll_y", "roll_z")]) for p in g}
    return (f"The mean gain on the linear steps is {lin['cal']:.2f} on the calibrated servo and {lin['stiff']:.2f} on the "
            f"template servo; on the angular steps {ang['cal']:.2f} and {ang['stiff']:.2f}. The two servos agree, so the "
            f"shortfall is the finger's kinematics, not servo lag.")


def perf_table():
    head = ["model", "wall time", "simulated", "ms per simulated s", "real-time factor", "chain"]
    body = []
    for r in PF:
        if "error" in r:
            continue
        ms = r["wall_s"] / r["sim_s"] * 1e3
        body.append([LBL[r["spec"]], f"{r['wall_s']:.2f}&#8202;s", f"{r['sim_s']:.1f}&#8202;s", f"{ms:.0f}",
                     f"{r['sim_s'] / r['wall_s']:.1f}&#215;", yn(r["chain_ok"])])
    return R.table(head, body, cls_num={1, 2, 3, 4})


def ms(spec):
    r = next(r for r in PF if r["spec"] == spec and "error" not in r)
    return f"{r['wall_s'] / r['sim_s'] * 1e3:.0f}"


def main():
    t = open(TPL).read()
    v = dict(LINKS)
    v.update(BUILT=time.strftime("%Y-%m-%d %H:%M"), DRAKE_V=R.DRAKE_V, MUJOCO_V=R.MUJOCO_V, N_ROWS=str(len(CH)), F_HOLD="3")
    ok = [NOM[k] for k in ("dhy_closed", "s1_closed", "p4s_closed")]
    v["DEPTH_RANGE"] = f"{min(r['end_depth_mm'] for r in ok):.0f}&#8211;{max(r['end_depth_mm'] for r in ok):.0f}"
    v["PHI_DHY"], v["PHI_S1"], v["PHI_P4S"] = (f"{NOM[k]['phi_end']:.1f}" for k in ("dhy_closed", "s1_closed", "p4s_closed"))
    v["PHI_OPEN"] = f"{NOM['p4s_open']['phi_end']:.1f}"
    pc = [r for r in BAT if r["spec"] == "mj:point4s" and r["brake"] == "closed"]
    v["PICK_FAIL"] = str(sum(not r["pick_ok"] for r in pc))
    v["TIMEOUT_N"] = str(sum(bool(r.get("close_timeout")) for r in pc))
    v["DROP_N"] = str(sum((not r["pick_ok"]) and not r.get("close_timeout") for r in pc))
    v["SEED9_NOTE"] = ""
    v["MS_P4S"], v["MS_S1"], v["MS_DHY"] = ms("mj:point4s"), ms("mj:spheres:s1:rs0.75:tr0.02"), ms("drake:hydro:rt0.01")
    v["NOMINAL_TABLE"], v["BATCH_TABLE"], v["BATCH_PROSE"] = nominal_table(), batch_table(), batch_prose()
    v["EXP1_TABLE"], v["EXP1_PROSE"], v["PERF_TABLE"] = exp1_table(), exp1_prose(), perf_table()
    m = os.path.join(D, "media")
    v["FILM"] = R.data_uri(os.path.join(m, "20261002-chain_six_models.mp4"), "video/mp4")
    v["POSTER"] = R.data_uri(os.path.join(m, "20261002-chain_six_models_poster.png"), "image/png")
    v["FILM_PATH"] = "docs/experiments/20261002-hom_chain/media/20261002-chain_six_models.mp4"
    v["EXP1_FILM_CAL"] = R.data_uri(os.path.join(m, "20261002-exp1_cal.mp4"), "video/mp4")
    v["EXP1_FILM_STIFF"] = R.data_uri(os.path.join(m, "20261002-exp1_stiff.mp4"), "video/mp4")
    v["EXP1_PATHS"] = "media/20261002-exp1_cal.mp4, media/20261002-exp1_stiff.mp4"
    v["CLOSEUP"] = R.data_uri(os.path.join(m, "20261002-sphere_pad_closeup.png"), "image/png")
    v["CLOSEUP_PATH"] = "docs/experiments/20261002-hom_chain/media/20261002-sphere_pad_closeup.png"
    for k, val in v.items():
        t = t.replace("{{" + k + "}}", val)
    left = sorted(set(x.split("}}")[0] for x in t.split("{{")[1:]))
    if left:
        raise SystemExit(f"unfilled placeholders: {left}")
    open(OUT, "w").write(t)
    print(f"wrote {OUT} ({os.path.getsize(OUT) / 1e6:.2f} MB)")
    print("EXP1", v["EXP1_PROSE"])
    print("BATCH", v["BATCH_PROSE"])


if __name__ == "__main__":
    main()
