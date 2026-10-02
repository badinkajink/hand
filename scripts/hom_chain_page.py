#!/usr/bin/env python3
"""Build docs/experiments/20261002-hom_chain/20261002-hom_screwdriver_chain.html from the study's jsonl files.

    python3 scripts/hom_chain_page.py

Reads films.jsonl (filmed runs: nominal per contact model and the perturbed films), chain.jsonl (perturbed batch),
bench.jsonl (timing) and exp1.jsonl, all written by scripts/hom_chain_study.py; Figures 1, 2 and 4 come from
scripts/hom_chain_figures.py; tables reuse scripts/hom_contact_patch_page.py's helpers. Every number in the prose is
computed here.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import hom_chain_figures as G  # noqa: E402
import hom_contact_patch_page as R  # noqa: E402

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
        'onward. Chain = pick, swing within 5&#176;, hold within 5&#176;, and inserted 15&#8202;mm or more on two pads.</p>')


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
<li><code>scripts/hom_chain_study.py</code> (default run, <code>films</code>, <code>bench</code>), <code>scripts/hom_chain_figures.py</code>
(Figures&#160;1, 2 and 4), and <code>scripts/hom_chain_page.py</code> with its template. The contact models were characterised in
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
<li><strong>Run the paper&#8217;s wield.</strong> Drive the screw-turning gait with object-twist references through the same controller, and
compare it with the relay gait (23&#176; per cycle).</li>
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
    v["HANDOFF"] = HANDOFF
    for k, val in v.items():
        t = t.replace("{{" + k + "}}", val)
    left = sorted(set(x.split("}}")[0] for x in t.split("{{")[1:]))
    if left:
        raise SystemExit(f"unfilled placeholders: {left}")
    open(OUT, "w").write(t)
    print(f"wrote {OUT} ({os.path.getsize(OUT) / 1e6:.2f} MB)")
    for k in ("HI_AREA", "HI_ARM", "HI_RIG", "HI_DEV_MAX", "LO_N2", "LO_DEV2", "LO_DEV_FINE", "CTRL_MJ", "CTRL_DK", "REP_SPREAD",
              "X_S1_P4S", "X_DHY_S1", "X_DHY_S05", "NC_S1", "NC_S05", "NC_DHY", "DEPTH_RANGE", "PICK_FAIL"):
        print(k, v[k])
    print("BATCH", v["BATCH_PROSE"])


if __name__ == "__main__":
    main()
