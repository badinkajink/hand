#!/usr/bin/env python3
r"""Build docs/experiments/20261002-hom_chain/20261002-hom_screwdriver_chain.html from the study's jsonl files.

    python3 scripts/hom_chain_page.py

Reads films.jsonl (filmed runs: nominal per contact model and the perturbed films), chain.jsonl (perturbed batch),
bench.jsonl (timing), bench_check.jsonl (bare-physics timing check) and exp1.jsonl, written by
scripts/hom_chain_study.py, and paper_tasks.jsonl (Exp 2, Exp 3, wield) from scripts/hom_paper_tasks.py. Figures 1, 2
and 4 come from scripts/hom_chain_figures.py; tables reuse scripts/hom_contact_patch_page.py's helpers. Math in the
template, \( \) inline and \[ \] display, is LaTeX rendered to inline SVG by scripts/texsvg.py (cached in
texsvg_cache.json beside the data, so a rebuild without TeX works while no formula changes). Every number in the prose
is computed here.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import hom_chain_figures as G  # noqa: E402
import hom_contact_patch_page as R  # noqa: E402
import texsvg  # noqa: E402
import retro_style  # noqa: E402

D = os.path.join(ROOT, "docs/experiments/20261002-hom_chain")
M = os.path.join(D, "media")
OUT = os.path.join(D, "20261002-hom_screwdriver_chain.html")
TPL = os.path.join(ROOT, "scripts/hom_chain_page.template.html")
LINKS = {"RIG_URL": "https://claude.ai/artifact/Rvfw1yQFgfWV8PdTvv7jsc",
         "RIG_PATH": "docs/experiments/20261001-hom_contact_patch/20261001-hom_pinch_contact_models.html",
         "BRAKE_URL": "https://claude.ai/artifact/DJLxG63vLZXs2Ey6hiCBuc",
         "BRAKE_PATH": "docs/experiments/20261001-hom_hand_brake/20261001-hom_hand_brake.html"}
LBL = {"mj:point3": ("MuJoCo point contact, condim&#160;3", "point"),
       "mj:point4s": ("MuJoCo condim&#160;4, &#956;<sub>t</sub> rescheduled", "point"),
       "mj:spheres:s2:rs0.75:tr0.02": ("MuJoCo 2&#8202;mm sphere pad (51 spheres)", "spheres"),
       "mj:spheres:s1:rs0.75:tr0.02": ("MuJoCo 1&#8202;mm sphere pad (205 spheres)", "spheres"),
       "mj:spheres:s0.5:rs0.75:tr0.03": ("MuJoCo 0.5&#8202;mm sphere pad (819 spheres)", "spheres"),
       "drake:point": ("Drake point contact", "drake"),
       "drake:hydro:rt0.01:hr2": ("Drake hydroelastic, 2&#8202;mm mesh", "drake"),
       "drake:hydro:rt0.01": ("Drake hydroelastic, 1&#8202;mm mesh", "drake"),
       "drake:hydro:rt0.01:hr0.5": ("Drake hydroelastic, 0.5&#8202;mm mesh", "drake")}
SVG_LBL = {k: v[0].replace("&#160;", " ").replace("&#8202;", " ").replace("<sub>t</sub>", "t").replace("&#956;", "&#956;")
           for k, v in LBL.items()}
NOM = ["dhy_closed", "s05_closed", "s1_closed", "s2_closed", "p4s_closed", "p4s_open", "dpt_closed", "mp3_closed"]
load = lambda f: [json.loads(l) for l in open(os.path.join(D, f)) if l.strip()]  # noqa: E731
FL = {r["key"]: r for r in load("films.jsonl")}
CH = load("chain.jsonl")
BAT = [r for r in CH if r["trial"]["seed"] != 0]
BN = load("bench.jsonl")
E1 = load("exp1.jsonl")
BC = load("bench_check.jsonl")
PT = {(r["task"], r["key"]): r for r in load("paper_tasks.jsonl")}
PK = ["s1", "p4s", "mp3", "dhy"]
PLBL = {"s1": "MuJoCo 1&#8202;mm sphere pad", "p4s": "MuJoCo condim&#160;4, &#956;<sub>t</sub> rescheduled",
        "mp3": "MuJoCo point contact, condim&#160;3", "dhy": "Drake hydroelastic, 1&#8202;mm mesh",
        "s1spin": "MuJoCo 1&#8202;mm sphere pad, spin rows held"}
TEX_CACHE = os.path.join(D, "texsvg_cache.json")
TEX_SCALE = 1.1                     # Computer Modern x-height to Source Serif 4 at 17 px


def num(x, fmt=".1f"):
    """A number with a typographic minus sign."""
    return format(x, fmt).replace("-", "&#8722;")


US = '<span style="text-transform:none">&#181;s</span>'      # th is upper-cased; keep the micro sign a micro sign


def yn(v):
    return ("yes", "cell c3") if v else ("no", "cell c0")          # house classes: c3 good, c0 bad


def deg(v, nd=1):
    return "&#8211;" if v is None else f"{v:.1f}&#176;"


def brake_name(b):
    return {"open": "open loop", "closed": "closed loop"}[b]


def vid(path, poster=None):
    p = f' poster="{R.data_uri(poster, "image/png")}"' if poster else ""
    return (f'<video src="{R.data_uri(path, "video/mp4")}"{p} controls muted loop playsinline preload="metadata"></video>')


# ------------------------------------------------------------------------------------------ chain

def nominal_table():
    head = ["contact model", "brake", "pick", "swing ends", "peak rate", "after squeeze", "inserted", "lower end tilt",
            "axial slip", "chain"]
    body = []
    for k in NOM:
        r = FL[k]
        live = r["pick_ok"] and r["phi_end"] is not None and 0 < r["phi_end"] < 180
        body.append([LBL[r["spec"]][0], brake_name(r["brake"]), yn(r["pick_ok"]),
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
        'onward. Chain = pick, swing within 5&#176;, hold within 5&#176;, and inserted 15&#8202;mm or more on two pads. The Drake rows were '
        're-run on 2026-10-02 after the hand&#8217;s collision exclusions were restored (Known issues): the hydroelastic swing end moved '
        'from 92.4&#176; to its value here, the insertion depth from 22.1&#8202;mm by less than 0.1&#8202;mm, and point contact still swings '
        'free.</p>')


def caption(r):
    if not r["pick_ok"]:
        what = "the pick fails" + (" (one finger never reaches the tool)" if r.get("close_timeout") else
                                   " (the tool leaves the pinch at the lift)")
    elif r["chain_ok"]:
        what = f"swing ends at {r['phi_end']:.1f}&#176;, inserted {r['end_depth_mm']:.1f}&#8202;mm"
    elif r["phi_end"] is not None and 0 < r["phi_end"] < 180:
        what = f"swing ends at {r['phi_end']:.1f}&#176;; " + ("inserted, outside the 5&#176; gates" if r["insert_ok"] else "insert missed")
    else:
        what = "tool swings free at the lift" if r["pick_ok"] else "tool lost"
    return what


def films_grid(keys, lead=None):
    out = []
    for i in range(0, len(keys), 2):
        figs = []
        for k in keys[i:i + 2]:
            r = FL[k]
            t = r["trial"]
            seed = "" if t["seed"] == 0 else (f" Seed {t['seed']}: &#956;&#8202;{t['mu']:.2f}, mass &#215;{t['mscale']:.2f}, tool "
                                             f"{t['dx'] * 1e3:+.1f}/{t['dy'] * 1e3:+.1f}&#8202;mm, {np.degrees(t['dyaw']):+.1f}&#176;.")
            figs.append(f'<figure>{vid(os.path.join(M, f"20261002-chain_{k}.mp4"))}<figcaption><b>{LBL[r["spec"]][0]}, '
                        f'{brake_name(r["brake"])} brake.</b>{seed} {caption(r).capitalize()}. '
                        f'<code>media/20261002-chain_{k}.mp4</code></figcaption></figure>')
        out.append('<div class="duo">' + "".join(figs) + "</div>")
    return "\n".join(out)


def batch_table():
    head = ["contact model", "brake", "pick", "swing within 5&#176;", "hold within 5&#176;", "inserted", "chain", "swing end, completed"]
    body = []
    for spec in ("mj:point4s", "mj:spheres:s1:rs0.75:tr0.02"):
        for brake in ("open", "closed"):
            rs = [r for r in BAT if r["spec"] == spec and r["brake"] == brake]
            c = lambda key: f"{sum(bool(r[key]) for r in rs)}/{len(rs)}"  # noqa: E731
            ends = [r["phi_end"] for r in rs if r["chain_ok"]]
            body.append([LBL[spec][0], brake_name(brake), c("pick_ok"), c("brake_ok"), c("hold_ok"), c("insert_ok"),
                         c("chain_ok"), ", ".join(f"{v:.1f}&#176;" for v in ends) or "&#8211;"])
    return R.table(head, body, cls_num={2, 3, 4, 5, 6, 7})


def batch_prose():
    def get(spec, brake, seed):
        return next(r for r in BAT if r["spec"] == spec and r["brake"] == brake and r["trial"]["seed"] == seed)
    clean = sorted({r["trial"]["seed"] for r in BAT if r["pick_ok"] and r["spec"] == "mj:point4s" and r["brake"] == "closed"})
    out = [f"Seeds {' and '.join(map(str, clean))} pick cleanly."]
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
                parts.append(f"the {brake}-loop brake ends the swing at {rs[0]['phi_end']:.1f}&#176; and {rs[1]['phi_end']:.1f}&#176;, "
                             f"outside the 5&#176; gate")
        t = get("mj:point4s", "closed", seed)["trial"]
        out.append(f"On seed {seed} (&#956;&#8202;{t['mu']:.2f}, mass &#215;{t['mscale']:.2f}) {parts[0]}; {parts[1]}.")
    return " ".join(out)


# ------------------------------------------------------------------------------------------ Exp 1

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
            f"template servo; on the angular steps it is {ang['cal']:.2f} and {ang['stiff']:.2f}. The two servos agree, so the "
            f"shortfall is the finger&#8217;s kinematics, not servo lag.")


# ------------------------------------------------------------------------------------------ cost

def bench_rows():
    out = {}
    for spec in [s for s in LBL if any(r["spec"] == s for r in BN)]:
        rs = [r for r in BN if r["spec"] == spec]
        ph = np.array([sum(r["perf"]["phys"].values()) / r["sim_s"] * 1e3 for r in rs])
        ct = np.array([r["perf"]["ctrl"] / r["sim_s"] * 1e3 for r in rs])
        med = rs[int(np.argsort(ph)[len(ph) // 2])]
        out[spec] = {"name": SVG_LBL[spec], "family": LBL[spec][1], "reps": len(rs), "phys_med": float(np.median(ph)),
                     "phys_min": float(ph.min()), "phys_max": float(ph.max()), "ctrl_med": float(np.median(ct)),
                     "wall": float(np.median([r["wall_s"] for r in rs])), "sim": float(np.median([r["sim_s"] for r in rs])),
                     "nc_hold": float(np.mean([r["ncon"].get("hold", 0) for r in rs])),
                     "nc_brake": float(np.mean([r["ncon"].get("brake", 0) for r in rs])),
                     "phi": rs[0]["phi_end"], "ok": rs[0]["chain_ok"],
                     "stage": {k: med["perf"]["phys"][k] / med["perf"]["sim"][k] * 1e3 for k in med["perf"]["phys"]
                               if med["perf"]["sim"][k] > 0.05}}
    return out


def bench_table(b):
    head = ["contact model", "physics ms/s", "range (runs)", "controller ms/s", "real time", "hold contacts", "chain"]
    body = []
    for spec, r in sorted(b.items(), key=lambda kv: kv[1]["phys_med"]):
        unit = "faces" if r["family"] == "drake" and "hydro" in spec else ("spheres" if r["family"] == "spheres" else "points")
        body.append([LBL[spec][0], f"{r['phys_med']:.1f}", f"{r['phys_min']:.1f}&#8211;{r['phys_max']:.1f} ({r['reps']})",
                     f"{r['ctrl_med']:.0f}", f"{1e3 / (r['phys_med'] + r['ctrl_med']):.1f}&#215;",
                     f"{r['nc_hold']:.0f} {unit}", yn(r["ok"])])
    return R.table(head, body, cls_num={1, 2, 3, 4, 5}) + (
        '<p class="note" style="font-size:13.5px;color:var(--ink3)">Physics and controller in ms of one core per simulated '
        'second; range over the repeats (their number in brackets). Real time = simulated seconds per second of one core, '
        'physics and controller together (logging excluded). Contacts in the hold = thumb and index contacts with the tool, '
        'averaged over the hold: contact points, touching spheres, or contact-surface faces for Drake hydroelastic. '
        'Chain = the nominal trial completes; point contact cannot hold the pinch torque in either simulator.</p>')


def stage_table(b):
    stages = ["approach", "close", "lift", "brake", "hold", "transport", "insert"]
    head = ["contact model"] + stages
    body = []
    for spec, r in sorted(b.items(), key=lambda kv: kv[1]["phys_med"]):
        body.append([LBL[spec][0]] + [f"{r['stage'].get(s, float('nan')):.0f}" for s in stages])
    return R.table(head, body, cls_num=set(range(1, 8))) + (
        '<p class="note" style="font-size:13.5px;color:var(--ink3)">Physics, ms per simulated second, within each stage of the '
        'median run. The pinch force is 4&#8202;N in the lift, 3&#8202;N from the hold onward, and falls from 1.6 to about 0.3&#8202;N in '
        'the brake.</p>')


# ------------------------------------------------------------------------------------------ paper tasks

def wield_table():
    head = ["contact model", "turned, 6 cycles", "twist phase per cycle", "while released per cycle", "twist gain", "pad slide",
            "max tilt", "re-closes"]
    body = []
    for k in PK:
        r = PT[("wield", k)]
        pc = r["per_cycle"]
        tw = [c["twist_deg"] for c in pc]
        rel = [c["release_open_deg"] + c["return_close_deg"] for c in pc]
        body.append([PLBL[k], f"{r['turned_deg']:.1f}&#176;", f"{np.mean(tw):.1f}&#176; ({min(tw):.1f}&#8211;{max(tw):.1f})",
                     f"{num(np.mean(rel), '+.1f')}&#176; ({num(min(rel), '+.1f')} to {num(max(rel), '+.1f')})", f"{r['gain_w']:.2f}",
                     f"{r['slide_rms_mmps']:.2f}&#8202;mm/s", f"{r['tilt_max_deg']:.1f}&#176;", f"{sum(c['closed'] for c in pc)}/{len(pc)}"])
    return R.table(head, body, cls_num={1, 2, 3, 4, 5, 6, 7}) + (
        '<p class="note" style="font-size:13.5px;color:var(--ink3)">Twist phase = the tool&#8217;s rotation about its own axis '
        'while the pads roll it, 20&#176; commanded per cycle; mean and range over the six cycles. While released = rotation '
        'during release, opening, return and re-close; positive continues the twist. Twist gain = median achieved over '
        'commanded angular velocity in the twist. Pad slide = RMS slide speed of the faster pad relative to the tool '
        'in the twist. Max tilt = tool axis from vertical.</p>')


def exp3_table():
    head = ["contact model", "pusher reaches", "pinch-velocity gain", "RMSE while stepped", "angle range", "angle error",
            "pusher slide", "pusher roll", "tripod holds", "middle in the tripod", "drift in the lift"]
    body = []
    for k in PK:
        r = PT[("exp3", k)]
        lo, hi = r["phi_range_deg"]
        if not r["pusher_closed"]:
            body.append([PLBL[k], yn(False), "&#8211;", "&#8211;", f"{num(lo, '+.0f')} to {num(hi, '+.0f')}&#176;", "&#8211;", "&#8211;", "&#8211;",
                         yn(False), "&#8211;", "&#8211;"])
            continue
        body.append([PLBL[k], yn(True), f"{r['gain_s']:.2f}", f"{r['rmse_s_on_dps']:.1f}&#8202;&#176;/s",
                     f"{num(lo, '+.1f')} to {num(hi, '+.1f')}&#176;", f"{r['phi_track_rmse_deg']:.2f}&#176;",
                     f"{r['pusher_slide_rms_mmps']:.2f}&#8202;mm/s", f"{r['pusher_roll_rms_dps']:.0f}&#8202;&#176;/s",
                     yn(r["tripod_held"]), f"{r['tripod_N']['middle']:.2f}&#8202;N",
                     f"{num(r['tripod_phi_drift_deg'], '+.2f')}&#176;"])
    return R.table(head, body, cls_num={2, 3, 4, 5, 6, 7, 9, 10}) + (
        '<p class="note" style="font-size:13.5px;color:var(--ink3)">Pinch velocity s = the tool&#8217;s angular velocity about '
        'the pinch axis; gain = median achieved over commanded while stepped (&#177;20&#8202;&#176;/s). Angle error = RMS of the angle '
        'about the pinch axis against the integrated reference over both cycles. Pusher slide and roll = RMS of the middle&#8217;s '
        'relative contact velocities while stepped (16). Middle in the tripod = the middle&#8217;s normal force with the pinch at 3&#8202;N: '
        'the pinch&#8217;s friction torque, 7.9&#215;10<sup>&#8722;3</sup>&#8202;N&#183;m by (11), carries the 2.4&#215;10<sup>&#8722;3</sup>&#8202;N&#183;m of gravity, '
        'so the third contact stays nearly unloaded. Drift = change of the angle while the palm lifts 30&#8202;mm on the tripod.</p>')


EXP2_COMPS = [("v_pinch", "along pinch axis"), ("v_up", "vertical"), ("v_tool", "along tool axis"),
              ("w_pinch", "about pinch axis"), ("w_up", "about vertical"), ("w_tool", "about tool axis")]


def exp2_table():
    head = ["contact model"] + [h for _, h in EXP2_COMPS] + ["RMSE linear", "RMSE angular", "pad slide", "net turn about pinch axis"]
    body = []
    for k in PK + ["s1spin"]:
        r = PT.get(("exp2", k))
        if r is None:
            continue
        pc = r["per_comp"]
        body.append([PLBL[k] + (" (tool hanging from the lift)" if k == "mp3" else "")]
                    + [num(pc[c]['gain'], '.2f') for c, _ in EXP2_COMPS]
                    + [f"{r['rmse_lin_mmps']:.1f}&#8202;mm/s", f"{r['rmse_ang_dps']:.1f}&#8202;&#176;/s",
                       f"{max(pc[c]['slide_rms_mmps'] for c, _ in EXP2_COMPS):.2f}&#8202;mm/s",
                       f"{num(r['pinch_angle_end_deg'], '+.0f')}&#176;"])
    return R.table(head, body, cls_num=set(range(1, 11))) + (
        '<p class="note" style="font-size:13.5px;color:var(--ink3)">Gain = median achieved over commanded on the stepped '
        'component of the tool&#8217;s twist (translations of the pinch midpoint at 20&#8202;mm/s, rotations at 40&#8202;&#176;/s). RMSE over '
        'the whole sequence, all three linear or all three angular components. Pad slide = the largest RMS slide speed of a pad '
        'relative to the tool during any step. Net turn = the tool&#8217;s angle about the pinch axis at the end of the 7.8&#8202;s '
        'sequence, from the start of the steps; every reference integrates to zero. Spin rows held: Eq.&#8202;(17) also holds each '
        'pad&#8217;s spin about its normal at the tool&#8217;s. With point contact the pinch cannot hold the tool horizontal, so it hangs '
        'from the lift on and its row is not comparable.</p>')


def paper_film(task, key, what):
    r = PT[(task, key)]
    path = os.path.join(M, f"20261002-paper_{task}_{key}.mp4")
    return (f'<figure>{vid(path)}<figcaption>{what} {PLBL[key]}, real time; left the wide shot, right the '
            f'{"middle" if task == "exp3" else "thumb"} pad with its collision spheres coloured by pressure (0&#8211;0.3&#8202;MPa) and '
            f'the centre of pressure in black. <code>media/20261002-paper_{task}_{key}.mp4</code></figcaption></figure>')


def paper_tile(task, what):
    path = os.path.join(M, f"20261002-paper_{task}_models.mp4")
    poster = R.data_uri(os.path.join(M, f"20261002-paper_{task}_models_poster.jpg"), "image/jpeg")
    v = (f'<video src="{R.data_uri(path, "video/mp4")}" poster="{poster}" controls muted loop playsinline '
         f'preload="metadata"></video>')
    return (f'<figure>{v}<figcaption>{what} in four contact models, wide shots, real time. Top: MuJoCo '
            f'1&#8202;mm sphere pad, MuJoCo condim&#160;4. Bottom: MuJoCo point contact, Drake hydroelastic (replayed through the MuJoCo '
            f'renderer). <code>media/20261002-paper_{task}_models.mp4</code></figcaption></figure>')


def paper_text():
    w, e3, e2 = PT[("wield", "s1")], PT[("exp3", "s1")], PT[("exp2", "s1")]
    tw = np.mean([c["twist_deg"] for c in w["per_cycle"]])
    rel = np.mean([c["release_open_deg"] + c["return_close_deg"] for c in w["per_cycle"]])
    tr = [e2["per_comp"][c]["gain"] for c in ("v_pinch", "v_up", "v_tool")]
    lead = (f"All three tasks ran with the 1&#8202;mm sphere pad on the first attempt. The setups below come from the reach scan "
            f"and the brake law, not from tuning runs. The wield turned the tool {w['turned_deg']:.0f}&#176; in six retract-turn "
            f"cycles: each twist delivered {tw:.1f}&#176; of the commanded 20&#176;, and each release gave back {abs(rel):.1f}&#176;. "
            f"The pusher tracked the pinch velocity at gain {e3['gain_s']:.2f}, with the angle within {e3['phi_track_rmse_deg']:.1f}&#176; RMS "
            f"over &#177;10&#176; cycles, and the tripod held through a 30&#8202;mm lift. The pinch carried the tool&#8217;s translations at "
            f"gains {min(tr):.2f}&#8211;{max(tr):.2f}. It does not turn the tool about the pinch axis (gain "
            f"{num(e2['per_comp']['w_pinch']['gain'], '.2f')}), the axis the paper also reports as worst tracked. Drake hydroelastic and "
            f"condim&#160;4 give the same tracking numbers within a few percent. Point contact fails Exp&#160;3 at the lift.")
    lede = (f"The paper&#8217;s other simulated tasks also run on the 1&#8202;mm sphere pad under the same controller. Six "
            f"retract-turn cycles of the wield turn the tool {w['turned_deg']:.0f}&#176; in the peg hole. A middle-finger pusher turns "
            f"the pinched tool about the pinch axis at gain {e3['gain_s']:.2f}, and a tripod then holds it. A derivation section "
            f"explains the pinch friction torque, the rescheduled torsional coefficient and the pad force the brake commands.")
    mp = PT[("wield", "mp3")]
    spin = max(c["release_open_deg"] + c["return_close_deg"] for k in ("p4s", "mp3", "dhy") for c in PT[("wield", k)]["per_cycle"])
    mp3_3 = PT[("exp3", "mp3")]
    cross = [e2["per_comp"][c]["cross_ang_dps"] for c in ("v_pinch", "v_up", "v_tool")]
    sp = PT.get(("exp2", "s1spin"))
    models = (f"Turning the tool about its own axis takes only the pads&#8217; tangential forces, so point contact completes the "
              f"wield as well ({np.mean([c['twist_deg'] for c in mp['per_cycle']]):.1f}&#176; per twist). It fails Exp&#160;3 for the "
              f"brake&#8217;s reason. With no friction torque the pinch is a free hinge, and the tool swings toward hanging during the "
              f"lift ({num(mp3_3['phi_range_deg'][0], '.0f')}&#176;) before the middle arrives. In every model except the sphere pad the "
              f"release sometimes spins the tool on, by up to {spin:.0f}&#176; in one cycle. The tool stands on a single support "
              f"contact with no torsional friction, so tangential load left in the pads at release turns it freely. The sphere pad "
              f"unloads without this. In Exp&#160;2 the translation steps also turn the tool, mostly about the pinch axis, at "
              f"{min(cross):.0f}&#8211;{max(cross):.0f}&#8202;&#176;/s RMS, and over the 7.8&#8202;s sequence they ratchet it "
              f"{abs(e2['pinch_angle_end_deg']):.0f}&#176; toward hanging. Equation&#8202;(17) leaves each pad&#8217;s spin about its normal "
              f"free. The friction torque that brakes the swing then drags the tool round with the pads, and gravity biases each "
              f"excursion downward. At 4&#8202;N the pinch holds the tool wherever it is left.")
    if sp is not None:
        spc = sp["per_comp"]
        models += (f" Holding the spin rows makes the rotation about the pinch axis follow at gain {spc['w_pinch']['gain']:.2f}. It cuts "
                   f"the cross-coupling to {min(spc[c]['cross_ang_dps'] for c in ('v_pinch', 'v_up', 'v_tool')):.0f}&#8211;"
                   f"{max(spc[c]['cross_ang_dps'] for c in ('v_pinch', 'v_up', 'v_tool')):.0f}&#8202;&#176;/s and the net turn to "
                   f"{num(sp['pinch_angle_end_deg'], '+.0f')}&#176;. The vertical translation "
                   f"pays for it (gain {e2['per_comp']['v_up']['gain']:.2f} to {spc['v_up']['gain']:.2f}): three joints per finger "
                   f"cannot hold all four rows.")
    return lead, lede, models


# ------------------------------------------------------------------------------------------ timing check

def check_tables(b):
    mj = {r["spec"]: r for r in BC if "spec" in r}
    one = {r["model"]: r for r in BC if "model" in r}
    head = ["contact model", f"bare physics, C ({US}/step)", f"plant step loop ({US}/step)", "contacts at the hold",
            f"bench, hold stage ({US}/step)", f"bench, whole chain ({US}/step)"]
    body = []
    for spec in [s for s in LBL if s in mj]:
        r = mj[spec]
        c = f"{np.median(r['c_us_per_step']):.1f}" if "c_us_per_step" in r else "&#8211;"
        unit = "faces" if "hydro" in spec else ("spheres" if "spheres" in spec else "points")
        n = r.get("ncon", r.get("contacts"))
        body.append([LBL[spec][0], c, f"{np.median(r['plant_us_per_step']):.1f}", f"{n:.0f} {unit}",
                     f"{b[spec]['stage'].get('hold', float('nan')):.1f}" if spec in b else "&#8211;",
                     f"{b[spec]['phys_med']:.1f}" if spec in b else "&#8211;"])
    t1 = R.table(head, body, cls_num={1, 2, 3, 4, 5}) + (
        '<p class="note" style="font-size:13.5px;color:var(--ink3)">&#181;s of one core per 1&#8202;ms step (= ms per simulated '
        'second). Bare physics: the chain stopped in the middle of its hold, servo targets frozen, the median of three 1&#8202;s '
        'windows; C = <code>mj_step(m, d, nstep=1000)</code> on a copy of the state; plant step loop = the plant&#8217;s own Python '
        'loop, which for condim&#160;4 includes rewriting &#956;<sub>t</sub> every step; Drake = <code>Simulator.AdvanceTo</code>. Contacts '
        'at the hold: all MuJoCo contacts, or Drake&#8217;s pad contact-surface faces or point pairs. Bench = the timing table '
        'above, with the controller running.</p>')
    names = {"mj:point3": "MuJoCo, condim&#160;3", "mj:point4": "MuJoCo, condim&#160;4",
             "mj:spheres1": "MuJoCo, 1&#8202;mm sphere pad on the fingertip", "drake:point": "Drake point contact",
             "drake:hydro": "Drake hydroelastic, 1&#8202;mm mesh"}
    body = []
    for k in names:
        r = one[k]
        us = np.median(r.get("c_us_per_step") or r.get("plant_us_per_step"))
        n = (f"{r['ncon']} contact{'s' if r['ncon'] != 1 else ''}" if "ncon" in r else
             (f"{r['faces']} faces" if r.get("faces") else f"{r['point_pairs']} point pair"))
        body.append([names[k], f"{us:.1f}", n])
    t2 = R.table(["one fingertip on the tool", f"{US} per step", "contacts"], body, cls_num={1}) + (
        '<p class="note" style="font-size:13.5px;color:var(--ink3)">A fingertip sphere (20&#8202;g) on a vertical slide resting '
        'by its weight on the fixed tool cylinder, nothing else in the scene; same solver settings as the chain.</p>')
    sp = lambda s: mj[s]  # noqa: E731
    c_mp3 = np.median(sp("mj:point3")["c_us_per_step"])
    loop_p4s = np.median(sp("mj:point4s")["plant_us_per_step"]) - np.median(sp("mj:point4s")["c_us_per_step"])
    hold_ratio = [np.median(mj[s]["plant_us_per_step"]) / b[s]["stage"]["hold"] for s in mj if s in b and "hold" in b[s]["stage"]]
    dhy = np.median(sp("drake:hydro:rt0.01")["plant_us_per_step"])
    dpt = np.median(sp("drake:point")["plant_us_per_step"])
    per_face = (dhy - dpt) / sp("drake:hydro:rt0.01")["contacts"]
    s1 = sp("mj:spheres:s1:rs0.75:tr0.02")
    per_sph = (np.median(s1["c_us_per_step"]) - c_mp3) / (s1["ncon"] - 2)
    lead = ("The timing table runs the whole chain with the controller in the loop. As a check, each model&#8217;s chain is stopped "
            "in the middle of its hold, with the tool pinched at 3&#8202;N. The bare physics is then timed from that state with the "
            "servo targets frozen. A one-contact scene in each simulator gives the cost of contact alone.")
    prose = (f"At the hold, MuJoCo point contact takes {c_mp3:.1f}&#8202;&#181;s per step in C, and a single fingertip on the tool takes "
             f"{np.median(one['mj:point3']['c_us_per_step']):.1f}&#8202;&#181;s. Most of the scene&#8217;s cost is therefore the 19-degree-of-freedom "
             f"multibody and the collision pass over its geometries, not the contact. Each touching sphere of the 1&#8202;mm pad adds "
             f"about {per_sph:.1f}&#8202;&#181;s. Rewriting &#956;<sub>t</sub> every step in Python adds {loop_p4s:.1f}&#8202;&#181;s to condim&#160;4, about half "
             f"of that model&#8217;s cost in the bench; done in C it would cost nothing measurable. The hold values are "
             f"{min(hold_ratio):.2f}&#8211;{max(hold_ratio):.2f} times the bench&#8217;s hold stage, where the controller changes the "
             f"targets every 10&#8202;ms. The whole-chain averages are higher than the hold because the lift and transport, where the "
             f"palm accelerates the pinched tool, cost two to three times as much. Drake&#8217;s point contact takes {dpt:.0f}&#8202;&#181;s per "
             f"step in the full scene and {np.median(one['drake:point']['plant_us_per_step']):.0f}&#8202;&#181;s with one contact. Its "
             f"hydroelastic contact adds about {per_face:.1f}&#8202;&#181;s per contact-surface face and step, so the "
             f"{sp('drake:hydro:rt0.01')['contacts']:.0f} faces of the hold make {dhy / 1e3:.1f}&#8202;ms per step.")
    return lead, t1, prose, t2


# ------------------------------------------------------------------------------------------ handoff

HANDOFF = """
<h3>Where things are</h3>
<ul>
<li><code>scripts/hom_control.py</code>: generalized contact frames for a sphere against a cylinder (closed form), the relative
contact twist, Eq.&#8202;2 (scipy bounded least squares), the kinematic <code>Mirror</code> that lets one controller drive MuJoCo
or Drake, and Exp&#160;1.</li>
<li><code>scripts/hom_chain.py</code>: the scene (four-axis palm stage, posts, 32-sector chamfered hole), <code>MjChainPlant</code>
and <code>DrakeChainPlant</code> behind one interface, <code>ClosedBrake</code>, <code>run_chain</code> (stage machine, scoring,
per-stage timing) and <code>ChainRenderer</code> (wide shot plus thumb close-up). One filmed run:
<pre>MUJOCO_GL=egl logs/20261001-hom_contact/venv/bin/python scripts/hom_chain.py run --spec mj:point4s --brake closed --seed 0 --film /tmp/chain.mp4</pre></li>
<li>Contact specs (parser <code>hom_hand_brake.parse_spec</code>): <code>mj:point3</code>; <code>mj:point4s</code> (condim&#160;4,
&#956;<sub>t</sub> rescheduled); <code>mj:spheres:s&lt;mm&gt;:rs0.75:tr&lt;s&gt;</code> (pad spacing, sphere radius, relaxation time;
0.5&#8202;mm needs tr&#8202;&#8805;&#8202;0.03); <code>drake:point</code>; <code>drake:hydro:rt0.01[:hr&lt;mm&gt;]</code> (relaxation time,
mesh resolution).</li>
<li><code>scripts/hom_paper_tasks.py</code>: the paper&#8217;s Exp&#160;2, Exp&#160;3 and wield on the chain&#8217;s scene (<code>Rollout</code> =
plant, mirror and joint references one 100&#8202;Hz tick at a time; <code>hom(f, vref, w, twist=...)</code> is one Eq.&#8202;(17) step with
the tool&#8217;s reference twist; <code>all</code> runs every task and model with films, <code>tiles</code> builds the four-model grids).</li>
<li><code>scripts/hom_chain_study.py</code> (default run, <code>films</code>, <code>bench</code>, <code>check</code>), <code>scripts/hom_chain_figures.py</code>
(Figures&#160;1, 2 and 4), <code>scripts/texsvg.py</code> (LaTeX to inline SVG; formulas cached in <code>texsvg_cache.json</code>), and
<code>scripts/hom_chain_page.py</code> with its template. The contact models were characterised in
<code>hom_contact_rig.py</code> (two-pad rig) and <code>hom_hand_brake.py</code> with <code>hom_hand_drake.py</code> (hand load step).</li>
<li>Environment: <code>logs/20261001-hom_contact/venv</code> (drake 1.57.0, mujoco 3.6.0; gitignored). A MuJoCo chain run takes
1&#8211;4&#8202;s and a Drake hydroelastic run about 25&#8202;s.</li>
</ul>
<h3>Decisions to keep</h3>
<ul>
<li><b>Pinch line:</b> the thumb&#8211;index mount line turned +45&#176;, 50&#8202;mm below the mounts (<code>PINCH_ROT_DEG</code>,
<code>PINCH_DEPTH</code>), with the middle curled to (0, 0.9, 0.9). The mount-midpoint line at 45&#8202;mm sits on the index joint limits.</li>
<li><b>Pad force:</b> it goes through the servo targets as q<sub>touch</sub>&#8202;+&#8202;J<sup>T</sup>(&#8722;F&#8202;n)/k<sub>p</sub>, with
gravity feedforward. q<sub>touch</sub> is corrected for the remaining gap at the switch. Without these corrections the pad force
ran 0.12&#8211;0.15&#8202;N under command at low force.</li>
<li><b>Simulator settings:</b> MuJoCo uses the elliptic cone, impratio 100 and Newton; impratio 10 makes the torsional
friction viscous (rig page). Drake uses a 0.01&#8202;s relaxation time; its default 0.1&#8202;s weakens the brake as the swing speeds up.</li>
<li><b>Sphere pads:</b> per-sphere stiffness from E, as in Figure&#160;1, with spacing no coarser than 1&#8202;mm (Figure&#160;2). The
controller runs at 100&#8202;Hz; the servo bus runs at 111&#8202;Hz.</li>
<li><b>Drake hand:</b> apply the MJCF&#8217;s <code>&lt;contact&gt;&lt;exclude&gt;</code> pairs as collision filters
(<code>DrakeChainPlant</code> does since 2026-10-02). Without them each distal link sits 23&#8202;mm inside its own yaw-link capsule at
up to 230&#8202;N, and friction locks mcp and pip. Other Drake ports (<code>hom_hand_drake.py</code>, <code>drake_sr2_hand.py</code>)
keep the <code>&lt;contact&gt;</code> element; check any new port with a contact dump at rest.</li>
</ul>
<h3>Known issues</h3>
<ul>
<li><b>Perturbed pick:</b> fails on 8 of 10 seeds (step&#160;1 below).</li>
<li><b>Closed-loop brake:</b> dropped the heavier tool on seed&#160;9 (step&#160;2 below).</li>
<li><b>Lift creep:</b> MuJoCo condim&#160;4 creeps about 3&#176; in the pinch during the 4&#8202;N lift, against 0.05&#176; in Drake.</li>
<li><b>Out-of-plane lean:</b> the pinch line tilts about 1.3&#176; during the closing, because the thumb&#8217;s and index&#8217;s
yaw axes are not mirror images about the pinch plane. The hanging tool then leans 1.5&#8211;2.3&#176; out of the swing plane in
MuJoCo and about 1&#176; in Drake.</li>
<li><b>Brake end band:</b> near vertical a friction brake cannot do better than about 2&#176;. The gravity torque vanishes
while the pads must still carry the weight (2&#956;N&#8202;&#8805;&#8202;W), so the brake accepts 2.5&#176;.</li>
<li><b>Drake results before the exclusion fix:</b> every Drake number on the first two versions of this page came from a hand
whose mcp and pip were locked by self-contact. The re-run moved the hydroelastic swing end from 92.4 to 91.3&#176; and Drake
point-contact physics from 143 to the value in the timing table; the wield did not turn at all before the fix. The old rows
are kept in <code>logs/20261002-hom_chain/pre_exclude_fix/</code>.</li>
<li><b>Wield release:</b> the tool&#8217;s end stands on one support contact with no torsional friction, so a release that leaves
tangential load in the pads spins it on (up to 23&#176; in a cycle with condim&#160;4). The sphere pad gives back 1&#176; or less.</li>
<li><b>condim&#160;4 cost:</b> half of its cost is the per-step &#956;<sub>t</sub> rewrite in Python (timing check), not MuJoCo.</li>
</ul>
<h3>Next steps, in order</h3>
<ol>
<li><strong>Make the pick hold under perturbation.</strong> For seeds 1, 2, 4&#8211;8 and 10 in <code>mj:point4s</code>, trace the
close phase per finger: tip-to-surface distance, least-squares bound activity and tool displacement on the posts. The
trace already carries <code>pinch_tilt_deg</code>; add the per-finger frame distance. Candidate fixes are position feedback on the
slide references, so both contacts land on the station at equal height; a V-cradle instead of flat posts; and a longer
close window. Accept when <code>pick_ok</code> holds on at least 9 of seeds 1&#8211;10 in both <code>mj:point4s</code> and
<code>mj:spheres:s1:rs0.75:tr0.02</code>.</li>
<li><strong>Find why the closed-loop brake dropped seed&#160;9.</strong> Read <code>floor</code>, <code>F_thumb</code>,
<code>sax_mm</code> and <code>phidot</code> in its trace (<code>films.jsonl</code>, key <code>p4s_closed_s9</code>). The brake inverts its
law with the nominal mass, and its floor 2W/2&#956; comes from that same nominal mass. Estimate the weight in hand during the lift, or
raise the floor at the first axial slip. Accept when seed&#160;9 completes and the nominal trial is unchanged.</li>
<li><strong>Measure reliability.</strong> Run 20 seeds per brake (open and closed) in condim&#160;4 and the 1&#8202;mm pad (about 2&#8202;min),
and 10 closed-loop seeds in Drake hydroelastic (about 5&#8202;min). First tune the open-loop end force on the nominal trial,
because the paper&#8217;s operator ended the force reduction by eye. Report success per stage with 95&#8202;% intervals, the
final-angle spread and the drops.</li>
<li><strong>Level the hanging tool during the hold.</strong> Move the thumb and index tips about 0.8&#8202;mm vertically in opposite
directions, sized from the measured out-of-plane tilt. Accept a tilt under 1&#176; before the insert.</li>
<li><strong>Build the three-pad hold.</strong> &#8220;Hold firm again&#8221; is a two-pad squeeze here. Build the version on D8&#8217;s
deploy-plan grasp (<code>docs/experiments/20260829-real_v1_deploy/deploy/sv1_w0099_b100_plan.json</code>: straddle 40&#8202;mm, thumb
axial 10&#8202;mm, grip depth 63&#8202;mm). The middle releases for the swing and re-closes on the hanging tool, as the 2026-09-16
open-loop swing did (<code>real_v1_chain_hands.py --pinch</code>).</li>
<li><strong>Load the wield.</strong> The wield runs (Exp&#160;2, Exp&#160;3 and the wield are in <code>paper_tasks.jsonl</code>), but against a
tool that spins freely on its support. Give the tool&#8217;s end a torsional resistance (condim&#160;4 on the tool&#8211;table pair, or a
screw-like seat with a set torque). Hold each pad&#8217;s tangential position while the force ramps down at release, and run
pick, load, insert and wield as one rollout. Accept when the net turn per cycle stays within 10&#8202;% of the twist phase over
20 cycles in the 1&#8202;mm pad and Drake hydroelastic, and compare it with the relay gait (23&#176; per cycle).</li>
<li><strong>Bench measurements.</strong> Measure the tag tracker&#8217;s rate and latency, which matter because the brake reads the swing angle at 100&#8202;Hz.
Measure the pad modulus behind E&#8202;=&#8202;10&#8202;MPa, and &#956;<sub>t</sub>/&#956; from a step-down of pinch force with the tag reading the
angle.</li>
</ol>
<h3>Conventions</h3>
<ul>
<li><b>Films:</b> film every run. <code>--film</code> on <code>hom_chain.py run</code> writes the wide shot with the thumb close-up.</li>
<li><b>Pages:</b> report results as a dated HTML page built from a builder and a template. Add a row to
<code>docs/experiments/INDEX.md</code> and give the local path beside the artifact URL. Republish to the same URL when the page
changes.</li>
<li><b>Compute:</b> run <code>~/.claude/bin/resguard.sh status</code> before long or multi-core jobs, and wrap them in
<code>resguard.sh run</code>.</li>
<li><b>Commits:</b> commit only your own files. Codex&#8217;s Drake-port files (<code>scripts/drake_sr2_*.py</code>,
<code>docs/experiments/20261001-drake-port/</code>) and its INDEX row are uncommitted and belong to Codex.</li>
</ul>
"""


# ------------------------------------------------------------------------------------------ law figure

def svg_law():
    """Figure 4: (a) the torsional coefficient the patch law asks for at each pad force, against a fixed condim-4 value
    set at the 3 N hold; (b) the two pads' torque capacity tau_cap(N) against the swing's demand m g d cos(phi)."""
    import math
    c, W, mgd = 0.996e-3, 0.024544 * 9.81, 0.024544 * 9.81 * 0.015
    W_, H_ = 980, 380
    out = [f'<svg viewBox="0 0 {W_} {H_}" role="img" aria-label="Left: the torsional coefficient mu_t that reproduces the '
           f'hydroelastic patch grows as N to the one quarter, so a fixed value set at 3 N is 57 percent too high at 0.5 N. '
           f'Right: the pinch torque capacity rises from zero at the 0.12 N slip limit and meets the gravity demand at '
           f'1.61 N horizontal, 1.22 N at 45 degrees and 0.41 N at 80 degrees." font-family="var(--f-mono)" font-size="15.5">']

    def panel(x0, y0, w, h, xs, ys, xlab, ylab, xt, yt, logx):
        fx = (lambda v: x0 + (math.log(v) - math.log(xs[0])) / (math.log(xs[1]) - math.log(xs[0])) * w) if logx else \
            (lambda v: x0 + (v - xs[0]) / (xs[1] - xs[0]) * w)
        fy = lambda v: y0 + h - (v - ys[0]) / (ys[1] - ys[0]) * h  # noqa: E731
        for v in yt:
            out.append(f'<line x1="{x0}" x2="{x0 + w}" y1="{fy(v):.1f}" y2="{fy(v):.1f}" style="stroke:var(--rule2)"/>'
                       f'<text x="{x0 - 8}" y="{fy(v) + 4:.1f}" text-anchor="end" style="fill:var(--ink3)">{v:g}</text>')
        for v in xt:
            out.append(f'<line x1="{fx(v):.1f}" x2="{fx(v):.1f}" y1="{y0}" y2="{y0 + h}" style="stroke:var(--rule2)"/>'
                       f'<text x="{fx(v):.1f}" y="{y0 + h + 22}" text-anchor="middle" style="fill:var(--ink3)">{v:g}</text>')
        xlab, ylab = xlab[0].upper() + xlab[1:], (ylab[:4] + ylab[4].upper() + ylab[5:]) if ylab.startswith("(") else ylab
        out.append(f'<line x1="{x0}" x2="{x0 + w}" y1="{y0 + h}" y2="{y0 + h}" style="stroke:var(--ink3)"/>'
                   f'<text x="{x0 + w / 2}" y="{y0 + h + 44}" text-anchor="middle" style="fill:var(--ink2)">{xlab}</text>'
                   f'<text x="{x0}" y="{y0 - 12}" style="fill:var(--ink2)">{ylab}</text>')
        return fx, fy

    def path(fx, fy, pts, style):
        d = "M" + " L".join(f"{fx(x):.1f},{fy(y):.1f}" for x, y in pts)
        out.append(f'<path d="{d}" style="fill:none;stroke-width:2.2;{style}"/>')

    # (a) torsional coefficient
    fx, fy = panel(70, 50, 360, 250, (0.1, 6.0), (0.0, 1.8), "pad force N (N), log scale", "(a) torsional coefficient \u03bc<tspan dy='3' font-size='12.5'>t</tspan><tspan dy='-3'> (mm), \u03bc = 1</tspan>",
                   (0.1, 0.3, 1, 3, 6), (0, 0.5, 1.0, 1.5), True)
    Ns = [0.1 * (60 ** (i / 80)) for i in range(81)]
    path(fx, fy, [(N, c * 1e3 * N ** 0.25) for N in Ns], "stroke:var(--c-c4)")
    path(fx, fy, [(0.1, c * 1e3 * 3 ** 0.25), (6.0, c * 1e3 * 3 ** 0.25)], "stroke:var(--c-ref);stroke-dasharray:6 5")
    for N in (0.5, 3.0):
        out.append(f'<circle cx="{fx(N):.1f}" cy="{fy(c * 1e3 * N ** 0.25):.1f}" r="4.5" style="fill:var(--c-c4);stroke:var(--card);stroke-width:2"/>')
    x5 = fx(0.5)
    out.append(f'<line x1="{x5:.1f}" x2="{x5:.1f}" y1="{fy(c * 1e3 * 0.5 ** 0.25):.1f}" y2="{fy(c * 1e3 * 3 ** 0.25):.1f}" '
               f'style="stroke:var(--bad);stroke-width:1.5"/>'
               f'<text x="{x5 + 6:.1f}" y="{fy(1.07):.1f}" style="fill:var(--bad)">fixed value 57 % high at 0.5 N</text>'
               f'<text x="{fx(0.11):.1f}" y="{fy(1.31) - 8:.1f}" style="fill:var(--ink3)">fixed \u03bc<tspan dy="3" font-size="12.5">t</tspan><tspan dy="-3"> set at the 3 N hold</tspan></text>'
               f'<text x="{fx(0.75):.1f}" y="{fy(0.62):.1f}" style="fill:var(--c-c4)">law: \u03bc<tspan dy="3" font-size="12.5">t</tspan><tspan dy="-3"> = \u03bc c N</tspan><tspan dy="-5" font-size="12.5">1/4</tspan></text>'
               f'<text x="{fx(0.75):.1f}" y="{fy(0.62) + 20:.1f}" style="fill:var(--ink3)">rescheduled each 1 ms step</text>')
    # (b) brake capacity against demand
    cap = lambda N: 0.0 if N <= W / 2 else 2 * N * c * N ** 0.25 * math.sqrt(1 - (W / (2 * N)) ** 2)  # noqa: E731
    fx, fy = panel(580, 50, 360, 250, (0.0, 2.0), (0.0, 5.0), "pad force N (N)", "(b) torque about the pinch axis (mN\u00b7m)",
                   (0, 0.5, 1.0, 1.5, 2.0), (0, 1, 2, 3, 4, 5), False)
    path(fx, fy, [(N, cap(N) * 1e3) for N in [W / 2 + i * (2.0 - W / 2) / 120 for i in range(121)]], "stroke:var(--c-c4)")
    out.append(f'<text x="{fx(0.06):.1f}" y="{fy(4.6):.1f}" style="fill:var(--c-c4)">\u03c4<tspan dy="3" font-size="12.5">cap</tspan><tspan dy="-3">(N), Eq. (11)</tspan></text>'
               f'<text x="{fx(W / 2) + 6:.1f}" y="{fy(0) - 7:.1f}" style="fill:var(--ink3)">← slip limit 0.12 N</text>')
    for phi, lab_dy in ((0, -7), (45, -7), (80, -7)):
        dem = mgd * math.cos(math.radians(phi)) * 1e3
        lo, hi = W / 2, 6.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if cap(mid) * 1e3 < dem else (lo, mid)
        xl, anchor = (fx(1.98), ' text-anchor="end"') if phi == 80 else (fx(0.03), '')
        out.append(f'<line x1="{fx(0):.1f}" x2="{fx(2.0):.1f}" y1="{fy(dem):.1f}" y2="{fy(dem):.1f}" '
                   f'style="stroke:var(--c-ref);stroke-dasharray:6 5"/>'
                   f'<circle cx="{fx(hi):.1f}" cy="{fy(dem):.1f}" r="4.5" style="fill:var(--c-sphere);stroke:var(--card);stroke-width:2"/>'
                   f'<text x="{xl:.1f}" y="{fy(dem) + lab_dy:.1f}"{anchor} style="fill:var(--ink3)">demand at {phi}\u00b0</text>'
                   f'<text x="{fx(hi) + (-9 if hi > 1.4 else 9):.1f}" y="{fy(dem) - 8:.1f}" '
                   f'text-anchor="{"end" if hi > 1.4 else "start"}" style="fill:var(--c-sphere)">N* = {hi:.2f} N</text>')
    out.append("</svg>")
    return "".join(out)


# ------------------------------------------------------------------------------------------ task traces

COL = {"s1": "var(--c-sphere)", "p4s": "var(--c-c4)", "mp3": "var(--ink3)", "dhy": "var(--c-drake)"}
SHORT = {"s1": "1 mm sphere pad", "p4s": "condim 4", "mp3": "point contact", "dhy": "Drake hydroelastic"}


def _axes(out, x0, y0, w, h, xs, ys, xt, yt, xlab, ylab):
    fx = lambda v: x0 + (v - xs[0]) / (xs[1] - xs[0]) * w  # noqa: E731
    fy = lambda v: y0 + h - (v - ys[0]) / (ys[1] - ys[0]) * h  # noqa: E731
    for v in yt:
        out.append(f'<line x1="{x0}" x2="{x0 + w}" y1="{fy(v):.1f}" y2="{fy(v):.1f}" style="stroke:var(--rule2)"/>'
                   f'<text x="{x0 - 8}" y="{fy(v) + 4:.1f}" text-anchor="end" style="fill:var(--ink3)">{num(v, "g")}</text>')
    for v in xt:
        out.append(f'<text x="{fx(v):.1f}" y="{y0 + h + 17}" text-anchor="middle" style="fill:var(--ink3)">{v:g}</text>')
    out.append(f'<line x1="{x0}" x2="{x0 + w}" y1="{y0 + h}" y2="{y0 + h}" style="stroke:var(--ink3)"/>'
               f'<text x="{x0 + w / 2}" y="{y0 + h + 38}" text-anchor="middle" style="fill:var(--ink2)">{xlab}</text>'
               f'<text x="{x0}" y="{y0 - 12}" style="fill:var(--ink2)">{ylab}</text>')
    return fx, fy


def _legend(out, x, y, items):
    for name, col, dash in items:
        out.append(f'<line x1="{x}" x2="{x + 22}" y1="{y - 4}" y2="{y - 4}" style="stroke:{col};stroke-width:2.2'
                   f'{";stroke-dasharray:5 4" if dash else ""}"/><text x="{x + 28}" y="{y}" style="fill:var(--ink2)">{name}</text>')
        x += 44 + 9.4 * len(name)                     # 15.5-unit mono text


def svg_exp3_trace():
    out = ['<svg viewBox="0 0 980 340" role="img" aria-label="Exp 3: the tool angle about the pinch axis follows the integrated '
           'pinch-velocity reference through two plus and minus 10 degree cycles in the sphere-pad, condim 4 and Drake models." '
           'font-family="var(--f-mono)" font-size="15.5">']
    rs = {k: PT[("exp3", k)] for k in ("s1", "p4s", "dhy")}
    T = max(r["trace"][-1][0] - r["trace"][0][0] for r in rs.values())
    fx, fy = _axes(out, 70, 60, 880, 220, (0, T), (-12, 12), [i for i in range(0, int(T) + 1)], (-10, -5, 0, 5, 10),
                   "time from the first pinch-velocity step (s)", "angle about the pinch axis (deg)")
    r0 = rs["s1"]["trace"]
    out.append('<path d="M' + " L".join(f"{fx(row[0] - r0[0][0]):.1f},{fy(row[4]):.1f}" for row in r0) +
               '" style="fill:none;stroke:var(--ink3);stroke-width:1.6;stroke-dasharray:6 5"/>')
    for k, r in rs.items():
        tr = r["trace"]
        out.append('<path d="M' + " L".join(f"{fx(row[0] - tr[0][0]):.1f},{fy(row[3]):.1f}" for row in tr) +
                   f'" style="fill:none;stroke:{COL[k]};stroke-width:2"/>')
    _legend(out, 300, 30, [("reference", "var(--ink3)", True)] + [(SHORT[k], COL[k], False) for k in rs])
    out.append("</svg>")
    return "".join(out)


def svg_wield_steps():
    out = ['<svg viewBox="0 0 980 384" role="img" aria-label="Wield: cumulative turn of the tool after each twist and each '
           'release over six cycles; the twists match across models, the condim 4, point-contact and Drake runs gain extra '
           'turn in some releases, the sphere pad gives back about one degree per release." font-family="var(--f-mono)" font-size="15.5">']
    fx, fy = _axes(out, 70, 84, 830, 240, (0, 6), (0, 140), range(0, 7), (0, 20, 40, 60, 80, 100, 120, 140),
                   "Cycle (twist, then release, open, return and close)", "Tool turned about its own axis (deg)")
    out.append('<path d="M' + " L".join(f"{fx(x):.1f},{fy(20 * x):.1f}" for x in range(7)) +
               '" style="fill:none;stroke:var(--ink3);stroke-width:1.6;stroke-dasharray:6 5"/>')
    ends = []
    for k in PK:
        cum, pts = 0.0, [(0, 0.0)]
        for i, c in enumerate(PT[("wield", k)]["per_cycle"]):
            cum += c["twist_deg"]
            pts.append((i + 0.5, cum))
            cum += c["release_open_deg"] + c["return_close_deg"]
            pts.append((i + 1, cum))
        out.append('<path d="M' + " L".join(f"{fx(x):.1f},{fy(y):.1f}" for x, y in pts) +
                   f'" style="fill:none;stroke:{COL[k]};stroke-width:2"/>')
        ends.append((fy(cum) + 5, f"{cum:.0f}°", COL[k]))
    prev = None
    for y, lab, col in sorted(ends):                  # end labels at least one line apart
        y = y if prev is None else max(y, prev + 17)
        out.append(f'<text x="{fx(6) + 8:.1f}" y="{y:.1f}" style="fill:{col}">{lab}</text>')
        prev = y
    _legend(out, 20, 20, [("Commanded, 20° per cycle", "var(--ink3)", True)])
    _legend(out, 20, 42, [(SHORT[k], COL[k], False) for k in PK])
    out.append("</svg>")
    return "".join(out)


def svg_exp2_drift():
    out = ['<svg viewBox="0 0 980 340" role="img" aria-label="Exp 2: the tool angle about the pinch axis, integrated over the '
           'twist steps; with spin free it ratchets about 50 degrees toward hanging in the sphere-pad, condim 4 and Drake '
           'models, and about half as far with the spin rows held." font-family="var(--f-mono)" font-size="15.5">']
    keys = [k for k in ("s1", "p4s", "dhy", "s1spin") if ("exp2", k) in PT]
    col = dict(COL, s1spin="var(--c-sphere)")
    T = max(PT[("exp2", k)]["trace"][-1][0] - PT[("exp2", k)]["trace"][0][0] for k in keys)
    fx, fy = _axes(out, 70, 60, 880, 220, (0, T), (-60, 10), [i for i in range(0, int(T) + 1)], (-60, -40, -20, 0),
                   "time from the first twist step (s)", "tool angle about the pinch axis (deg)")
    tr0 = PT[("exp2", "s1")]["trace"]
    comps = ["v_pinch", "v_up", "v_tool", "w_pinch", "w_up", "w_tool"]
    t_prev, c_prev = None, None
    for row in tr0:                                                      # step labels at each component's start
        if row[1] != c_prev:
            out.append(f'<text x="{fx(row[0] - tr0[0][0]) + 3:.1f}" y="{fy(8):.1f}" style="fill:var(--ink3)">{row[1].replace("_", " ")}</text>'
                       f'<line x1="{fx(row[0] - tr0[0][0]):.1f}" x2="{fx(row[0] - tr0[0][0]):.1f}" y1="{fy(10):.1f}" y2="{fy(-60):.1f}" style="stroke:var(--rule)"/>')
            c_prev = row[1]
    for k in keys:
        tr = PT[("exp2", k)]["trace"]
        ang, pts = 0.0, []
        for i, row in enumerate(tr):
            if i:
                ang += row[11] * (row[0] - tr[i - 1][0]) * 57.29578
            pts.append((row[0] - tr[0][0], ang))
        dash = ";stroke-dasharray:6 4" if k == "s1spin" else ""
        out.append('<path d="M' + " L".join(f"{fx(x):.1f},{fy(y):.1f}" for x, y in pts) +
                   f'" style="fill:none;stroke:{col[k]};stroke-width:2{dash}"/>')
    _legend(out, 20, 30, [(SHORT[k] if k != "s1spin" else "Sphere pad, spin rows held", col[k], k == "s1spin") for k in keys])
    out.append("</svg>")
    return "".join(out)


# ------------------------------------------------------------------------------------------ math

TEX_RE = re.compile(r"\\\[(.+?)\\\]|\\\((.+?)\\\)", re.S)


def render_tex(t):
    r"""Every \( \) and \[ \] in the page becomes an inline SVG (scripts/texsvg.py)."""
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None)
             for m in TEX_RE.finditer(t)]
    svgs = iter(texsvg.render(items, cache_path=TEX_CACHE, scale=TEX_SCALE))
    return TEX_RE.sub(lambda m: next(svgs), t), len(items)


# ------------------------------------------------------------------------------------------ main

def main():
    t = open(TPL).read()
    v = dict(LINKS)
    v.update(BUILT=time.strftime("%Y-%m-%d %H:%M"), DRAKE_V=R.DRAKE_V, MUJOCO_V=R.MUJOCO_V)
    for key, tag in (("dhy_closed", "DHY"), ("s05_closed", "S05"), ("s1_closed", "S1"), ("s2_closed", "S2"),
                     ("p4s_closed", "P4S")):
        v[f"PHI_{tag}"] = f"{FL[key]['phi_end']:.1f}"
    ok = [FL[k] for k in ("dhy_closed", "s05_closed", "s1_closed", "s2_closed", "p4s_closed")]
    v["DEPTH_RANGE"] = f"{min(r['end_depth_mm'] for r in ok):.0f}" if \
        round(min(r['end_depth_mm'] for r in ok)) == round(max(r['end_depth_mm'] for r in ok)) else \
        f"{min(r['end_depth_mm'] for r in ok):.0f}&#8211;{max(r['end_depth_mm'] for r in ok):.0f}"
    pc = [r for r in BAT if r["spec"] == "mj:point4s" and r["brake"] == "closed"]
    v["PICK_FAIL"] = str(sum(not r["pick_ok"] for r in pc))
    v["TIMEOUT_N"] = str(sum(bool(r.get("close_timeout")) for r in pc))
    v["DROP_N"] = str(sum((not r["pick_ok"]) and not r.get("close_timeout") for r in pc))
    # theory numbers
    hi, lo = G.compute(3.0), G.compute(0.5)
    v["HI_AREA"] = f"{hi['field']['area'] * 1e6:.1f}"
    v["HI_ARM"] = f"{hi['field']['arm'] * 1e3:.2f}"
    v["HI_RIG"] = f"{0.996 * 3.0 ** 0.25:.2f}"
    dev = lambda res, s: (res[s]["arm"] / res["field"]["arm"] - 1) * 100  # noqa: E731
    v["HI_DEV_MAX"] = f"{max(abs(dev(hi, s)) for s in (0.002, 0.001, 0.0005)):.0f}"
    v["LO_N2"] = str(lo[0.002]["n_in"])
    v["LO_DEV2"] = f"{dev(lo, 0.002):.0f}"
    v["LO_DEV_FINE"] = f"{max(abs(dev(lo, s)) for s in (0.001, 0.0005)):.0f}"
    v["FIG_LAW"] = svg_law()
    v["FIG_SECTION"] = G.svg_section(hi)
    v["FIG_PATCH"] = G.svg_patch(hi, lo)
    v["CLOSEUP_FILM"] = R.data_uri(os.path.join(M, "20261002-contact_closeups.mp4"), "video/mp4")
    v["CLOSEUP_POSTER"] = R.data_uri(os.path.join(M, "20261002-contact_closeups_poster.png"), "image/png")
    v["CLOSEUP_PATH"] = "docs/experiments/20261002-hom_chain/media/20261002-contact_closeups.mp4"
    # chain
    v["NOMINAL_TABLE"] = nominal_table()
    v["FILM"] = R.data_uri(os.path.join(M, "20261002-chain_six_models.mp4"), "video/mp4")
    v["POSTER"] = R.data_uri(os.path.join(M, "20261002-chain_six_models_poster.png"), "image/png")
    v["FILM_PATH"] = "docs/experiments/20261002-hom_chain/media/20261002-chain_six_models.mp4"
    v["MODEL_FILMS"] = films_grid(NOM)
    v["BATCH_TABLE"], v["BATCH_PROSE"] = batch_table(), batch_prose()
    v["PERTURBED_FILMS"] = films_grid(["p4s_closed_s3", "p4s_open_s3", "p4s_closed_s9", "p4s_open_s9",
                                       "p4s_closed_s1", "p4s_closed_s4"])
    v["EXP1_TABLE"], v["EXP1_PROSE"] = exp1_table(), exp1_prose()
    v["EXP1_FILM_CAL"] = R.data_uri(os.path.join(M, "20261002-exp1_cal.mp4"), "video/mp4")
    v["EXP1_FILM_STIFF"] = R.data_uri(os.path.join(M, "20261002-exp1_stiff.mp4"), "video/mp4")
    v["EXP1_PATHS"] = "media/20261002-exp1_cal.mp4, media/20261002-exp1_stiff.mp4"
    # cost
    b = bench_rows()
    v["FIG_BENCH"] = G.svg_bench(list(b.values()))
    v["BENCH_TABLE"], v["STAGE_TABLE"] = bench_table(b), stage_table(b)
    ms = lambda s: b[s]["phys_med"]  # noqa: E731
    v["MS_P4S"], v["MS_S1"], v["MS_DHY"] = f"{ms('mj:point4s'):.0f}", f"{ms('mj:spheres:s1:rs0.75:tr0.02'):.0f}", \
        f"{ms('drake:hydro:rt0.01'):.0f}"
    reps = sorted({r["reps"] for r in b.values()})
    v["REP_RANGE"] = f"{reps[0]}&#8211;{reps[-1]}" if len(reps) > 1 else str(reps[0])
    mj = [r["ctrl_med"] for s, r in b.items() if s.startswith("mj:")]
    dk = [r["ctrl_med"] for s, r in b.items() if s.startswith("drake:")]
    v["CTRL_MJ"] = f"{min(mj):.0f}&#8211;{max(mj):.0f}"
    v["CTRL_DK"] = f"{min(dk):.0f}&#8211;{max(dk):.0f}"
    v["REP_SPREAD"] = f"{max((r['phys_max'] - r['phys_min']) / r['phys_med'] * 100 for r in b.values()):.0f}"
    v["X_S1_P4S"] = f"{ms('mj:spheres:s1:rs0.75:tr0.02') / ms('mj:point4s'):.1f}"
    v["X_DHY_S1"] = f"{ms('drake:hydro:rt0.01') / ms('mj:spheres:s1:rs0.75:tr0.02'):.0f}"
    v["X_DHY_S05"] = f"{ms('drake:hydro:rt0.01') / ms('mj:spheres:s0.5:rs0.75:tr0.03'):.0f}"
    v["NC_S1"] = f"{b['mj:spheres:s1:rs0.75:tr0.02']['nc_hold']:.0f}"
    v["NC_S05"] = f"{b['mj:spheres:s0.5:rs0.75:tr0.03']['nc_hold']:.0f}"
    v["NC_DHY"] = f"{b['drake:hydro:rt0.01']['nc_hold']:.0f}"
    # the paper's other tasks
    v["PAPER_LEAD"], _, v["PAPER_MODELS"] = paper_text()      # the paper lede no longer goes in the page lede
    v["WIELD_TABLE"], v["EXP3_TABLE"], v["EXP2_TABLE"] = wield_table(), exp3_table(), exp2_table()
    v["WIELD_FILMS"] = paper_film("wield", "s1", "Wield, six retract-turn cycles in the peg hole:") + paper_tile("wield", "The wield")
    v["EXP3_FILMS"] = paper_film("exp3", "s1", "Exp&#160;3, pusher cycles and the tripod lift:") + paper_tile("exp3", "Exp&#160;3")
    v["EXP2_FILMS"] = paper_film("exp2", "s1", "Exp&#160;2, the six twist steps:") + paper_tile("exp2", "Exp&#160;2")
    v["CHECK_LEAD"], v["CHECK_TABLE"], v["CHECK_PROSE"], v["ONE_TABLE"] = check_tables(b)
    v["FIG_EXP3"], v["FIG_WIELD"], v["FIG_EXP2"] = svg_exp3_trace(), svg_wield_steps(), svg_exp2_drift()
    v["HANDOFF"] = HANDOFF
    for k, val in v.items():
        t = t.replace("{{" + k + "}}", val)
    left = sorted(set(x.split("}}")[0] for x in t.split("{{")[1:]))
    if left:
        raise SystemExit(f"unfilled placeholders: {left}")
    t, n_tex = render_tex(t)
    print("formulas", n_tex)
    open(OUT, "w").write(retro_style.apply(t))  # plain page style (owner, 2026-10-09)
    print(f"wrote {OUT} ({os.path.getsize(OUT) / 1e6:.2f} MB)")
    for k in ("HI_AREA", "HI_ARM", "HI_RIG", "HI_DEV_MAX", "LO_N2", "LO_DEV2", "LO_DEV_FINE", "CTRL_MJ", "CTRL_DK", "REP_SPREAD",
              "X_S1_P4S", "X_DHY_S1", "X_DHY_S05", "NC_S1", "NC_S05", "NC_DHY", "DEPTH_RANGE", "PICK_FAIL"):
        print(k, v[k])
    print("BATCH", v["BATCH_PROSE"])


if __name__ == "__main__":
    main()
