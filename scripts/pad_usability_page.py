#!/usr/bin/env python3
"""Page: cost and control outcomes of the sphere-pad fingertip on the real_v1 hand (experiments U1-U5, 2026-10-14).

Owner, 2026-10-14: "our cpu and gpu usability is a main selling point". The page reports what the 1 mm sphere pads cost
on the CPU, in MuJoCo-Warp and in PPO training on the real_v1 hand, against the cheaper fingertip models (2 mm pads,
the TPU block as one convex mesh with point contact, condim 4 or multi-CCD), and what the choice changes in control:
agreement with Drake on the servo-limited plant (U2), the usable physics step (U3), the rollout model of pinned iCEM
(U4) and the scene RL policies train and are scored on (U5). Plan: docs/notes/20261014-contact_model_assessment_plan.md.

Data: docs/experiments/20261014-pad_usability/{cost,agree,step,planner,rl}/ with one findings_U<k>.json per experiment
(tables, key numbers, figures, films, each with its row files and how it was computed).

    .venv/bin/python scripts/pad_usability_page.py
Out: docs/experiments/20261014-pad_usability/20261014-sphere_pad_cost_and_control.html (media inlined as data URIs).
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import contact_overview_page as P  # noqa: E402
import retro_style  # noqa: E402
import texsvg  # noqa: E402

D = os.path.join(ROOT, "docs/experiments/20261014-pad_usability")
OUT = os.path.join(D, "20261014-sphere_pad_cost_and_control.html")
TPL = os.path.join(ROOT, "scripts/pad_usability_page.template.html")
TEX_CACHE = os.path.join(D, "texsvg_cache.json")
OVERVIEW = "../../overviews/contact_model.html"
FIG, TAB, REFS = [0], [0], {}
EXTRA_CSS = """
.tw th{white-space:normal !important;vertical-align:bottom;text-transform:none}
.tw td:first-child{min-width:7em}
.col > figure,.col > .tw{width:auto !important;max-width:100% !important}
figure.films video{width:100%;height:auto}
figure.img img{width:100%;height:auto}
.eq{margin:.8em 0;text-align:center}
""" + texsvg.BLOCK_CSS + "\n"


# ------------------------------------------------------------------------------------------ helpers

def F(k):
    p = os.path.join(D, {1: "cost", 2: "agree", 3: "step", 4: "planner", 5: "rl"}[k], f"findings_U{k}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def data_uri(path, mime=None):
    if mime is None:
        mime = {".png": "image/png", ".jpg": "image/jpeg", ".mp4": "video/mp4"}[os.path.splitext(path)[1]]
    return f"data:{mime};base64," + base64.b64encode(open(path, "rb").read()).decode()


def jpeg_uri(path, quality=86):
    """A poster as an inline JPEG (the PNG posters are 0.5-0.6 MB)."""
    from PIL import Image
    buf = io.BytesIO()
    Image.open(path).convert("RGB").save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def figure(body, caption, key=None, cls="img"):
    FIG[0] += 1
    if key:
        REFS[f"fig:{key}"] = FIG[0]
    return f'<figure class="{cls}">{body}<figcaption>Figure&#160;{FIG[0]}. {caption}</figcaption></figure>'


def img(rel, caption, key=None):
    return figure(f'<img alt="" src="{data_uri(os.path.join(ROOT, rel))}">', caption, key)


def film(rel, poster_rel, caption, key=None):
    p = os.path.join(ROOT, rel)
    if not os.path.exists(p):
        return ""
    v = (f'<video controls muted loop playsinline preload="metadata" poster="{jpeg_uri(os.path.join(ROOT, poster_rel))}" '
         f'src="{data_uri(p)}"></video>')
    return figure(v, caption, key, cls="films")


def table(head, body, caption, note=None, text_cols=(0,), key=None):
    TAB[0] += 1
    if key:
        REFS[f"tab:{key}"] = TAB[0]
    cls = lambda i: "" if i in text_cols else " class=num"  # noqa: E731
    out = [f"<p class='tcap'><b>Table&#160;{TAB[0]}.</b> {caption}</p>",
           "<div class='tw'><table><thead><tr>" + "".join(f"<th{cls(i)}>{h}</th>" for i, h in enumerate(head))
           + "</tr></thead><tbody>"]
    for r in body:
        out.append("<tr>" + "".join(f"<td{cls(i)}>{c}</td>" for i, c in enumerate(r)) + "</tr>")
    out.append("</tbody></table></div>")
    if note:
        out.append(f"<p class='tnote'>{note}</p>")
    return "".join(out)


def tab_of(f, title_start):
    return next(t for t in f["tables"] if t["title"].startswith(title_start))


def render_tex(t):
    maths = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None)
             for m in P.TEX_RE.finditer(t)]
    svgs = texsvg.render(maths, cache_path=TEX_CACHE, scale=P.TEX_SCALE)
    it = iter(svgs)
    return P.TEX_RE.sub(lambda m: next(it), t), len(maths)


def sci(p):
    """A p-value as m&#215;10^e."""
    m, e = f"{p:.1e}".split("e")
    return f"{m}&#215;10<sup>{'&#8722;' if int(e) < 0 else ''}{abs(int(e))}</sup>"


def us(x):
    return f"{x}&#8202;&#181;s"


# ------------------------------------------------------------------------------------------ sections

GLOSSARY = [
    ("Fingertip models", "1&#8202;mm pads: the printed TPU block&#8217;s palmar face as 3,177 spheres per hand, 1&#8202;mm apart and "
     "0.75&#8202;mm in radius, each a soft MuJoCo contact calibrated to the hydroelastic pressure law (the default fingertip since "
     "13 October; <a href=\"" + OVERVIEW + "\">contact overview</a>). 2&#8202;mm pads: the same at 2&#8202;mm spacing and 1.5&#8202;mm "
     "radius, 870 spheres. Point contact: the block as one convex mesh, MuJoCo&#8217;s condim&#160;3 contact (time constant "
     "6&#8202;ms). Condim&#160;4: the mesh with torsional friction, coefficient \\(\\mu_t\\) = 2.8&#8202;mm (2.9&#8202;mm in the RL "
     "scenes), the pads&#8217; twist-onset arm at the grips. Multi-CCD: the mesh with MuJoCo&#8217;s multi-contact convex collision."),
    ("Servo-limited and working plant", "The finger servos as position actuators: servo-limited plant 0.72&#8202;N&#183;m/rad and "
     "0.18&#8202;N&#183;m torque limit (the STS3215 servos at their stall torque, the plant the RL policies and the planners use), "
     "working plant 4&#8202;N&#183;m/rad and 1&#8202;N&#183;m."),
    ("&#181;s per step, per world-step", "Wall time of one 1&#8202;ms physics step of the hand on one CPU core "
     "(<code>mj_step</code>), and MuJoCo-Warp time of a batch step divided by the number of worlds."),
    ("Replanning time", "Wall time of one replanning of pinned iCEM: 64 rollouts of 0.5&#8202;s on the planner&#8217;s model."),
    ("PPO iteration time", "The trainer&#8217;s <code>Iteration time</code>: collection of 2,048 environments &#215; 24 steps and "
     "the update, in seconds; median of iterations 3&#8211;12 for the probes."),
    ("Placement, kept", "Placement: one deployed hand D1&#8211;D8 and one jitter seed of the tool&#8217;s pose. Kept: at the end of "
     "the replay the tool is less than 20&#8202;mm below its grip height with at least two fingers on it."),
    ("Turn, within 3&#176; of Drake", "Turn: change of the tool axis&#8217; angle from the vertical between the grip and the end of "
     "the replay, in degrees. Within 3&#176; of Drake: the model keeps the tool and its turn is within 3&#176; of Drake "
     "hydroelastic&#8217;s on the same placement."),
    ("Usable step", "The largest physics step at which at least 36 of 40 placements keep the tool and end within 3&#176; of the same "
     "model&#8217;s 1&#8202;ms replay, with no ejection (the tool faster than 2&#8202;m/s or more than 0.30&#8202;m from its start)."),
    ("Spectral threshold", "The step \\(h\\) at which the spectral radius of MuJoCo&#8217;s one-step map at a rest state "
     "(<code>mjd_transitionFD</code>, reference clamp <code>refsafe</code> off) passes 1.001, bisected to 0.01&#8202;ms."),
    ("\\(t_c\\), \\(d_0\\), \\(n\\), \\(q\\)", "Contact time constant (solref) and impedance at rest (solimp), loaded tool contacts of "
     "one fingertip, and \\(q = d_0 + (1-d_0)/n\\)."),
    ("Impratio \\(\\kappa\\), creep", "Ratio of the friction rows&#8217; to the normal rows&#8217; impedance (100 in the CPU scenes, 10 "
     "in the RL trainer). Creep: sliding speed of a pinched tool held below its slip force, in &#181;m/s."),
    ("Held, held turn, three-finger ending", "Planner runs: held when the cylinder ends less than 20&#8202;mm below its post-grip height "
     "with two fingers each pressing at least its weight; held turn: the end turn if held, else 0; three-finger ending: all three "
     "fingertips loaded over at least 99&#8202;% of the last second."),
    ("Held (RL)", "An RL rollout keeps the tool when two fingertips each press at least 0.24&#8202;N and the tool is above 60&#8202;mm "
     "at 5&#8202;s; held cosine: the signed cosine of the tool axis with the vertical at the end, over held rollouts."),
]


def glossary():
    return '<dl class="glossary">' + "".join(f"<dt>{a}</dt><dd>{b}</dd>" for a, b in GLOSSARY) + "</dl>"


def question():
    return ("<p>The sphere pads run unmodified in CPU MuJoCo, where the hand&#8217;s planners roll out, and in MuJoCo-Warp, where "
            "its RL policies train. This page measures what that costs on the real_v1 hand and what the fingertip model changes "
            "in five uses: the cost per step on the CPU, the GPU and in PPO (U1), agreement with Drake hydroelastic on the "
            "servo-limited plant (U2), the longest usable physics step (U3), the rollout model of the in-hand planner (U4) and "
            "the scene RL policies train and are scored on (U5). Every run uses the deployed real_v1 hands (D1&#8211;D8 for the "
            "plan replays, D6 for RL, the scale study&#8217;s layouts for the planner) on the servo-limited plant unless a table "
            "says otherwise. The plan, its hypotheses and falsifiers are in "
            "<code>docs/notes/20261014-contact_model_assessment_plan.md</code>.</p>")


def cost_section():
    f = F(1)
    cpu = tab_of(f, "CPU cost")
    order = [("1 mm pads", "1 mm pads"), ("1 mm pads, cylinder only", "1 mm pads, cylinder only"),
             ("1 mm pads, GPU solver (10 line-search it.)", "1 mm pads, GPU solver settings"),
             ("1 mm pads, tol 1e-6, 20 it., 50 line-search it.", "1 mm pads, tolerance 10<sup>&#8722;6</sup>, 50 line searches"),
             ("2 mm pads", "2 mm pads"), ("Block mesh, multi-CCD", "Mesh, multi-CCD"),
             ("Block mesh, point contact", "Mesh, point contact"), ("Block mesh, condim 4", "Mesh, condim 4")]
    by = {(r[0], r[1]): r for r in cpu["rows"]}
    body = []
    for key, lab in order:
        a, b = by[("16 mm cylinder", key)], by[("25 mm cylinder", key)]
        body.append([lab, a[2], a[3].replace(" (grip lost)", " (reset)"), b[3].replace(" (grip lost)", " (reset)"), a[5], b[5],
                     a[8], b[8], a[9], f"{a[10]}, {b[10]}"])
    t1 = table(["Model", "Geoms", "Grip step, 16 mm (&#181;s)", "Grip step, 25 mm (&#181;s)", "Contacts, 16 mm",
                "Contacts, 25 mm", "Replanning, 16 mm (s)", "Replanning, 25 mm (s)", "Replanning over 1 mm pads, 16 mm",
                "Plans reset"], body,
               "CPU cost at the held three-finger grip of the scale study&#8217;s 16 and 25&#8202;mm cylinders, servo-limited plant, "
               "MuJoCo 3.6 on one E-core (Intel Core Ultra 7 265KF), 1&#8202;ms step. Replanning: 64 plans of 0.5&#8202;s on three "
               "threads from the pads&#8217; grip. Plans reset: rollouts that MuJoCo reset on a bad acceleration.", key="cpu")
    gpu = tab_of(f, "MuJoCo-Warp physics")
    ppo = tab_of(f, "12-iteration PPO probes at")
    g = {r[0]: r for r in gpu["rows"] if r[1] == "2048"}
    pp = {}
    for r in ppo["rows"]:
        pp.setdefault(r[0], []).append(r)
    mesh_it = sum(float(r[3]) for r in pp["Block mesh"]) / len(pp["Block mesh"])
    body2 = []
    for lab in ("1 mm pads", "2 mm pads", "Block mesh, condim 4", "Block mesh"):
        ge, gt = g[lab], g[lab + " (trainer's buffers)"]
        it = sum(float(r[3]) for r in pp[lab]) / len(pp[lab])
        body2.append([lab, ge[2].split(" / ")[0], gt[2], ge[3], ge[5], f"{it:.2f}", f"{it - mesh_it:+.2f}",
                      f"{max(int(r[8]) for r in pp[lab]) / 1000:.1f}"])
    t2 = table(["Model", "&#181;s per world-step, sized buffers", "&#181;s per world-step, trainer&#8217;s buffers",
                "Contacts per world", "Constraint rows per world", "PPO iteration (s)", "Over the mesh (s)",
                "Probe GPU memory (GB)"], body2,
               "MuJoCo-Warp at the held RL state of the servo-limited plant (D6 after the scripted grasp and lift, 2&#8202;ms step, "
               "impratio 10, 2,048 worlds), with buffers 1.5 times the state&#8217;s needs and with the trainer&#8217;s; and "
               "12-iteration PPO probes of the 2026-10-11 servo-plant trainer at 2,048 environments.", key="gpu")
    return (f"<p>At the servo plant&#8217;s three-finger grip the 1&#8202;mm pads cost {us('146.9')} per 1&#8202;ms step on one CPU "
            f"core (16&#8202;mm cylinder; {us('124.8')} on the 25&#8202;mm cylinder), 9.1 and 7.3 times point contact on the TPU "
            "block (Table&#160;[[tab:cpu]], Figure&#160;[[fig:cpu]]). Narrowphase collision and the constraint solve take 67&#8202;% "
            "of the pads&#8217; step. Restricting the pads to collide with the cylinder changes nothing, since every grip contact "
            "is already a pad&#8211;cylinder pair; the GPU planner&#8217;s solver settings (tolerance 10<sup>&#8722;6</sup>, 20 "
            "iterations, 10 line searches) make the CPU plant reset on a bad acceleration in 7 and 20 of 64 rollouts. Fewer pads "
            f"cut the cost: 2&#8202;mm pads step the grip in {us('56.2')} and {us('68.3')}, and a replanning of pinned iCEM takes "
            "0.75&#8202;s against the pads&#8217; 2.19&#8202;s on three threads, 0.34 of it. Point contact and condim&#160;4 "
            "replan in 0.20&#8211;0.26&#8202;s.</p>"
            + t1
            + img("docs/experiments/20261014-pad_usability/cost/media/cpu_step_cost.png", f["figures"][0]["caption"], "cpu")
            + "<p>On the GPU the 1&#8202;mm pads cost 2.37&#8202;&#181;s per world-step at the held RL state, 3.7 times the block "
            "mesh; 2&#8202;mm pads cost 0.95&#8202;&#181;s and condim&#160;4 0.68&#8202;&#181;s (Table&#160;[[tab:gpu]], "
            "Figure&#160;[[fig:gpu]]). In 12-iteration PPO probes the pads add 2.36&#8202;s per iteration to the mesh&#8217;s "
            "8.62&#8202;s; the 40&#8202;M-step training logs of 11 October give 2.32&#8202;s over 5.45&#8202;s, 1.47 times the "
            "mesh over the whole run. The probes ran pinned to E-cores, which added 3.2&#8202;s per iteration to both arms. 2&#8202;mm "
            "pads add 1.07&#8202;s and condim&#160;4 0.04&#8202;s.</p>"
            "<p>GPU memory follows the trainer&#8217;s buffer rule: every scene with more than 500 geoms gets 640 contacts, 3,072 "
            "constraint rows and 768 sensor matches per world, so the 2&#8202;mm pads take 10.9&#8202;GB against the 1&#8202;mm "
            "pads&#8217; 11.2&#8202;GB and the mesh&#8217;s 2.0&#8202;GB. Buffers sized from the held state overflow on the first "
            "PPO iteration, because the exploring policy presses more pads into the tool (up to 573 rows and 195 matches per world "
            "with 2&#8202;mm pads, 2,001 and 698 with 1&#8202;mm pads); sized to those requests (242 contacts, 768 rows, 256 "
            "matches) the 2&#8202;mm pads train 12 iterations without overflow in 4.7&#8202;GB.</p>"
            + t2
            + img("docs/experiments/20261014-pad_usability/cost/media/gpu_ppo_cost.png", f["figures"][1]["caption"], "gpu")
            + film(f["films"][0]["path"], f["films"][0]["poster"], f["films"][0]["caption"], "grip"))


def agree_section():
    f = F(2)
    t = f["tables"][0]
    keep = [r for r in t["rows"] if not r[0].startswith("Condim 4, 2.6")]
    t1 = table(["Model", "Plant", "Numerics", "Within 3&#176; of Drake (of 24)", "95&#8202;% interval", "Kept (of 40)",
                "Median |turn difference| (&#176;) [95&#8202;% interval]", "McNemar against the 1 mm pads",
                "CPU per step (&#181;s)"], [r for r in keep], t["title"] + ". Servo-limited plant: rows of this study; working "
               "plant: the 1&#8202;mm pads, point contact and Drake rows of 6 October. McNemar: discordant placements (pads "
               "within, model not : the reverse) and the exact two-sided p.", key="agree")
    # figures numbered in the order they appear: the two charts, then the two films
    figs = (img(f["figures"][0]["path"], f["figures"][0]["caption"], "agree")
            + img(f["figures"][1]["path"], f["figures"][1]["caption"]))
    films = "".join(film(x["path"], x["poster"], x["caption"]) for x in f["films"][:1] + f["films"][2:3])
    return ("<p>On the servo-limited plant the 1&#8202;mm pads end the eight deployed open-loop plans within 3&#176; of Drake "
            "hydroelastic, keeping the tool, on 16 of 24 placements, against 23 of 24 on the working plant "
            "(Table&#160;[[tab:agree]], Figure&#160;[[fig:agree]]). They keep the tool in 40 of 40 runs and differ from Drake by a "
            "median 2.1&#176;. Four of their eight misses are stalls: on D3 (seeds 0&#8211;2) and D8 seed 1 the pads leave the tool "
            "within 4&#176; of its grip angle where Drake turns it 40&#8211;44&#176;; the other four are overturns of 5&#8211;8&#176; "
            "on D5 and a 3.8&#176; under-turn on D6 seed 1. Drake is converged in its step: 16 placements at 0.5&#8202;ms move at "
            "most 0.52&#176;.</p>"
            "<p>No cheaper model reaches the pads&#8217; count on the servo plant: 2&#8202;mm pads 12, condim&#160;4 1, point contact 1 "
            "and multi-CCD 1 of 24, and the three single-contact models drop the tool in 11&#8211;26 of 40 runs. On the working "
            "plant 2&#8202;mm pads reach 21 of 24. At the RL trainer&#8217;s numerics (2&#8202;ms, impratio 10) the pads score 14 "
            "of 24. The HOM closed-loop turn agrees with Drake within 5&#176; and in the kept outcome on 21 of 24 servo-plant runs. "
            "On the two-pad block bed impratio 10 lengthens the pads&#8217; twist-onset arm by 7.1&#8202;% at 1&#8202;N and "
            "4.5&#8202;% at 3&#8202;N and raises creep 9.9-fold, to 107&#8202;&#181;m/s at 1&#8202;N.</p>"
            + t1
            + figs
            + films)


def step_section():
    f = F(3)
    S = json.load(open(os.path.join(D, "step/summary_U3.json")))
    lab = {"pads": "1 mm pads", "pads2": "2 mm pads", "pt": "Point contact", "pt4": "Condim 4"}
    body = []
    for m in ("pads", "pads2", "pt", "pt4"):
        for pl, pn in (("kp0.72_kv0_fr0.18_fl0_dp0.0144", "Servo"), ("kp4_kv0_fr1_fl0_dp0.08", "Working")):
            ss = [s for s in S["sweep"] if s["model"] == m and s["plant"] == pl]
            u = S["usable_step_ms"][f"{m}|{pl}"]
            cpu = {s["dt_ms"]: s["cpu_ms_per_sim_s"] for s in ss}
            body.append([lab[m], pn] + [str(s["within3_1ms"]) for s in ss] + [f"{u:g}" if u else "none",
                                                                               f"{cpu[1.0]:.1f}", f"{cpu[u]:.1f}" if u else "&#8211;"])
    t1 = table(["Model", "Plant", "1 ms", "2 ms", "4 ms", "5 ms", "10 ms", "Usable step (ms)", "CPU at 1 ms (ms/s)",
                "CPU at the usable step (ms/s)"], body,
               "Plan replays of D1&#8211;D8 at five physics steps with MuJoCo&#8217;s reference clamp on: placements (of 40) that "
               "keep the tool and end within 3&#176; of the same model&#8217;s 1&#8202;ms replay; at 1&#8202;ms the count is the "
               "kept placements. CPU: wall time per simulated second on one core.", key="sweep")
    hand = [r for r in S["spectral_hand"] if "postoff" in r["state"] or not r["state"].startswith("scale:")]
    nm = {"pads": "1 mm pads", "pads2": "2 mm pads", "pt": "Point contact", "pt4": "Condim 4", "mesh": "Block mesh"}

    def st(s):
        return (s.replace("scale:cylinder:", "Grip, cylinder ").replace(":postoff", "").replace(":0.625", " mm, t 0.625")
                .replace(":0.125", " mm, t 0.125").replace("bench:D6:servo", "D6 replay, servo plant")
                .replace("bench:D6:work", "D6 replay, working plant").replace("rl:servo_pads1", "RL held state")
                .replace("rl:servo_mesh", "RL held state"))
    body2 = [[st(r["state"]), nm[r["model"]], ", ".join(str(v) for v in r["n_by_finger"].values()), f"{r['d0']:.2f}",
              f"{r['tc_ms']:.0f}", f"{r['h_ms']:.2f}", f"{r['normal_ms']:.2f}", f"{r['friction_ms']:.2f}",
              f"{r['jointspace_ms']:.2f}", r["mode"]] for r in hand]
    t2 = table(["State", "Model", "Loaded contacts per finger", "\\(d_0\\)", "\\(t_c\\) (ms)", "Threshold (ms)",
                "Normal-row bound (ms)", "Friction-row bound (ms)", "Joint-space model (ms)", "Unstable mode"], body2,
               "Spectral stability thresholds with the reference clamp off. Grips: the scale study&#8217;s 16 and 25&#8202;mm cells "
               "0.8&#8202;s after the grip command, with the post&#8217;s collisions removed as in the planner&#8217;s model.",
               key="spectral")
    return ("<p>With MuJoCo&#8217;s reference clamp on (the contact time constant raised to twice the step when the step exceeds "
            "half of it), the 1&#8202;mm pads keep 37 of 40 placements within 3&#176; of their own 1&#8202;ms turn at 2&#8202;ms on "
            "the servo-limited plant and at 4&#8202;ms on the working plant (Table&#160;[[tab:sweep]], Figure&#160;[[fig:sweep]]). "
            "Point contact keeps the tool on only 29 of 40 servo-plant placements at 1&#8202;ms and reaches a usable 2&#8202;ms on "
            "the working plant. At their usable step the pads cost 56&#8202;ms of CPU per simulated second on the servo plant, "
            "against 114&#8202;ms at 1&#8202;ms and 16&#8202;ms for point contact at 1&#8202;ms.</p>"
            "<p>With the clamp off the pads&#8217; one-step map turns unstable at 7.0&#8211;7.2&#8202;ms on six hand states, "
            "2&#8211;4&#8202;% above the bound of the normal rows</p>"
            "<div class='eq'>\\[h^{*} = 2\\,t_c\\left(\\sqrt{1+q}-1\\right),\\qquad q = d_0 + \\frac{1-d_0}{n},\\]</div>"
            "<p>with the pad normals as the unstable mode (Table&#160;[[tab:spectral]], Figure&#160;[[fig:spectral]]). Point contact "
            "and condim&#160;4 on the block, whose time constant is 6&#8202;ms against the pads&#8217; 10&#8202;ms, fail at "
            "4.3&#8211;5.2&#8202;ms. When the tool hangs on friction across a two-pad pinch (the contact bed) the friction rows set "
            "the step, at \\(h^{*} = 2/(b\\,\\lambda')\\) with \\(b\\) the rows&#8217; damping and \\(\\lambda'\\) the largest "
            "eigenvalue of \\(A(A+R)^{-1}\\) over them, which is \\(d_0 t_c\\) at impratio 100; all 24 bed pairs fail there within "
            "0.15&#8202;%. A linear model of the step built from the rest state, with the servo springs, damping and every loaded "
            "contact row, gives all 18 hand thresholds within 3&#8202;%, with or without the servo springs.</p>"
            + t1
            + img(f["figures"][0]["path"], f["figures"][0]["caption"], "sweep")
            + t2
            + img(f["figures"][1]["path"], f["figures"][1]["caption"], "spectral")
            + film(f["films"][0]["path"], f["films"][0]["poster"], f["films"][0]["caption"], "steps"))


def pending(k, what):
    return (f"<p>The rows of U{k} were still being written when this page was built: {what} "
            f"(<code>docs/experiments/20261014-pad_usability/{'planner' if k == 4 else 'rl'}/</code>).</p>")


def planner_section():
    f = F(4)
    if f is None:
        return pending(4, "pinned iCEM rolling out point contact and condim&#160;4 on the 1&#8202;mm pad plant, four cells and four "
                          "seeds, against the pads planner")
    S = json.load(open(os.path.join(D, "planner/summary_U4.json")))
    R = {r["config"]: r for r in S["configs"]}
    pads, c4, pt = R["icem_pin"], R["icem_pin_pt4"], R["icem_pin_pt"]
    lab = {"icem_pin": "Pads planner", "icem_pin_pt4": "Condim-4 planner", "icem_pin_pt": "Point-contact planner"}

    def ci(r):
        if r.get("paired_change_median") is None:
            return "&#8211;"
        lo, hi = r["paired_change_ci95"] or (float("nan"), float("nan"))
        return f"{r['paired_change_median']:+.1f} [{lo:+.1f}, {hi:+.1f}]"
    body = [[lab[c], f"{R[c]['held_turn_median']:.1f}", ci(R[c]),
             "&#8211;" if R[c].get("wilcoxon_p") is None else sci(R[c]["wilcoxon_p"]),
             f"{R[c]['three_finger']} of {R[c]['n_pairs']}", f"{R[c]['contact_breaks_median']:.0f}",
             f"{R[c]['cmd_rate_rms_median']:.3f}", f"{R[c]['omega_jitter_median']:.2f}",
             "&#8211;" if R[c].get("replan_s_3threads") is None else f"{R[c]['replan_s_3threads']:.2f}"]
            for c in ("icem_pin", "icem_pin_pt4", "icem_pin_pt")]
    t1 = table(["Planner", "Held turn, median (&#176;)", "Change against the pads planner, median [95&#8202;% interval] (&#176;)",
                "Wilcoxon p", "Three-finger endings", "Contact breaks, median", "Command rate, RMS (rad/s)",
                "Turn-rate jitter (rad/s)", "Replanning, 3 threads (s)"], body,
               "Pinned iCEM on the 1&#8202;mm pad plant with three rollout models, four cells of the scale study &#215; seeds "
               "0&#8211;3. Interval: cell-cluster bootstrap; Wilcoxon: exact signed-rank test of the paired changes.", key="planner")
    fr = ""
    var = S.get("friction") or []
    if var:
        fr = ("<p>" + " ".join(re.sub(r"-(?=\d)", "&#8722;", v["sentence"]) for v in var) + "</p>")
    return (f"<p>Pinned iCEM (64 rollouts of 0.5&#8202;s every 20&#8202;ms in two iterations, the first knot pinned to the command "
            f"in force, the three-pad cost of 13 October) turns the cylinder on the 1&#8202;mm pad plant by a median "
            f"{pads['held_turn_median']:.1f}&#176; over {pads['n_pairs']} pairs of four cells and four seeds, at "
            f"{pads['replan_s_3threads']:.2f}&#8202;s per replanning on three threads; its seed-0 run on the 25&#8202;mm cell "
            f"reproduces the 13 October row ({S['reproduction']['new']:.2f}&#176;). Rolling out condim&#160;4 on the TPU block "
            f"instead cuts the replanning to {c4['replan_s_3threads']:.2f}&#8202;s and the held turn by a median "
            f"{-c4['paired_change_median']:.1f}&#176; (95&#8202;% interval {-c4['paired_change_ci95'][1]:.1f}&#8211;"
            f"{-c4['paired_change_ci95'][0]:.1f}&#176;, Wilcoxon p {sci(c4['wilcoxon_p'])}); point contact cuts the replanning to "
            f"{pt['replan_s_3threads']:.2f}&#8202;s and the held turn by {-pt['paired_change_median']:.1f}&#176;. The pads planner "
            f"ends {pads['three_finger']} of {pads['n_pairs']} runs with three fingertips loaded, the condim-4 and point-contact "
            f"planners {c4['three_finger']} and {pt['three_finger']} (Table&#160;[[tab:planner]], Figure&#160;[[fig:planner]]). "
            "Neither cheaper model passes the plan&#8217;s test (a median loss under 10&#176; and at least 15 of 16 three-finger "
            "endings), so the CPU planner keeps the pads as its rollout model.</p>"
            + fr + t1
            + img("docs/experiments/20261014-pad_usability/planner/20261014-u4_planner_models.png", f["figures"][0]["caption"],
                  "planner")
            + "".join(film(x["path"], x["poster"], x["caption"]) for x in f.get("films", [])))


def rl_section():
    f = F(5)
    if f is None:
        return pending(5, "the four servo-plant finals on the TPU-mesh, 1&#8202;mm pad, 2&#8202;mm pad and condim-4 plants under "
                          "four conditions, and their own-scene evaluation at impratio 100 and a 4&#8202;ms step")
    S = json.load(open(os.path.join(D, "rl/summary_U5.json")))
    C = {(c["final"], c["scene"], c["mode"], c["condition"]): c for c in S["cells"]}
    fin = ["Mesh s0", "Mesh s1", "Pads s0", "Pads s1"]
    scn = [("tpu27mesh", "TPU-mesh"), ("tpu27pads1", "1 mm pad"), ("tpu27pads2", "2 mm pad"), ("tpu27meshc4", "Condim-4")]
    cond = [("nominal", "Nominal"), ("kp=0.36", "Servo gain 0.36 N&#183;m/rad"), ("fr=0.12", "Torque limit 0.12 N&#183;m"),
            ("friction=0.5", "Friction &#215;0.5")]
    body = []
    for mode, mname in (("closed", "Closed loop (of 128)"), ("open", "Replay (of 64)")):
        for c, cn in cond:
            body.append([mname if c == "nominal" else "", cn] + [" / ".join(str(C[(x, sc, mode, c)]["n_held"]) for x in fin)
                                                                for sc, _ in scn])
    t1 = table(["Evaluation", "Condition"] + [f"{n} plant" for _, n in scn], body,
               "Held rollouts of the four servo-plant finals on each plant, in the order TPU-mesh policy seed 0, seed 1, "
               "1&#8202;mm pad policy seed 0, seed 1. Closed loop: 128 rollouts with grasp-pose noise of 2&#8202;mm and "
               "5&#176;; replay: the 64 action sequences each policy produced on its own plant, played open loop.",
               text_cols=(0, 1), key="rlcross")
    rows = [[x["tag"].replace("20261011-d6_servo_", "").replace("tpu27", "").replace("_40M_", " ").replace("pads1", "1 mm pads")
             .replace("mesh", "TPU mesh"), {"impratio100": "Impratio 100", "dt4ms": "4 ms step"}[x["numerics"]], str(x["n_held"]),
             f"{x['d_held']:+d}", f"{x['d_cos']:+.3f}"] for x in S["numerics"] if x["numerics"] != "nominal"]
    t2 = table(["Final", "Numerics", "Held (of 128)", "Change", "Held-cosine change"], rows,
               "Each final on its own plant, closed loop with grasp-pose noise, at impratio 100 (written into the model in place) "
               "and at a 4&#8202;ms step, against the trainer&#8217;s 2&#8202;ms and impratio 10.", key="rlnum")
    return ("<p>On the 1&#8202;mm pad plant every servo-plant final keeps the tool, whichever fingertip it trained on: at half servo "
            "gain (0.36&#8202;N&#183;m/rad) the TPU-mesh finals keep 125 and 123 of 128 closed-loop rollouts and the pad finals 119 "
            "and 126, and replays of their own actions at a 0.12&#8202;N&#183;m torque limit keep 61&#8211;64 of 64 "
            "(Table&#160;[[tab:rlcross]], Figure&#160;[[fig:rlcross]]). On the TPU-mesh plant the same conditions separate the "
            "policies: the mesh finals keep 96 and 117 of 128 at half gain, the pad finals 48 and 32, and every final&#8217;s replays "
            "keep 0&#8211;9 of 64.</p>"
            "<p>Training on the pads therefore gives no margin on the pad plant and costs 30&#8211;90 of 128 rollouts on the mesh "
            "plant under each of the three perturbations (all four seed pairs; Fisher p below 10<sup>&#8722;33</sup>). The RL "
            "trainer can train on the TPU mesh, at 1.47 times less time per PPO iteration and 2.0 against 11.2&#8202;GB of GPU "
            "memory, and score its policies on the 1&#8202;mm pads. The 2&#8202;mm pad plant reproduces the 1&#8202;mm pad plant "
            "within 5 of 128 held rollouts and 0.03 of held cosine on 12 of 16 final-condition pairs; condim&#160;4 on the block "
            "drops the tool (8&#8211;99 of 128 held at nominal) and matches on none. Which plant the hardware follows is the "
            "question of the torque-limited bench replays in Future work.</p>"
            "<p>On their own plants a 4&#8202;ms physics step changes three finals by 1&#8211;3 of 128 held rollouts and costs the "
            "TPU-mesh seed 1 22; impratio 100, written into the model in place, changes held counts by &#8722;1 to +4 and held "
            "cosine by &#8722;0.006 to &#8722;0.027 (Table&#160;[[tab:rlnum]]), so the trainer keeps its 2&#8202;ms step and its "
            "impratio of 10.</p>"
            + t1
            + img("docs/experiments/20261014-pad_usability/rl/20261014-u5_cross.png", f["figures"][0]["caption"], "rlcross")
            + t2
            + "".join(film(x["path"], x["poster"], x["caption"]) for x in f.get("films", [])))


def lede():
    return ("On the real_v1 hand the 1&#8202;mm sphere pads cost 147&#8202;&#181;s per grip step on one CPU core (9.1 times point "
            "contact on the TPU block), 2.37&#8202;&#181;s per MuJoCo-Warp world-step and 1.47 times the TPU mesh per PPO iteration. "
            "On the servo-limited plant they agree with Drake on 16 of 24 plan replays, against 1 for point contact, and their "
            "usable step is 2&#8202;ms. The CPU planner keeps them as its rollout model: condim&#160;4 and point contact replan 6 and "
            "9 times faster but lose a median 19 and 28&#176; of held turn. RL can train on the mesh and score on the pads: "
            "mesh-trained policies keep 123&#8211;125 of 128 rollouts on the pad plant at half servo gain.")


def limitations():
    return ("<p>Drake hydroelastic is the reference of U2 and shares the pads&#8217; pressure law and its unmeasured 10&#8202;MPa "
            "modulus; no fingertip has been measured on the bench. CPU times are E-core times on one workstation; the 2026-10-13 "
            "planner timing ran unpinned and gives 105.5&#8202;&#181;s for the same pad step. The PPO probes are 12 iterations "
            "from scratch, so their buffer requests are a lower bound for a full run. The plan replays are open loop and the "
            "spectral thresholds are rest states; a turning tool or an exploring policy loads other contact rows.</p>")


def open_items():
    items = [
        ("Friction of the printed tip", "pull test of two printed 95A gyroid tips on the bench cylinder at 0.5, 1 and 2&#8202;N per "
         "tip, 5 repeats, fit with <code>scripts/tpu_tip_calibrate.py fit --pull</code>; a coefficient below 0.55 lies outside the "
         "RL friction randomisation (&#215;0.55&#8211;1.15) and calls for retraining."),
        ("Servo stiffness and overload trip", "the bench plan&#8217;s push test (<code>docs/experiments/20261011-bench_plan/</code>) "
         "with the station stopped: 0.29&#8202;N/mm and a 4.1&#8202;N trip confirm the servo-limited plant; 1.6&#8202;N/mm "
         "and 22.6&#8202;N would move every result here to the working plant&#8217;s rows."),
        ("Twist-onset torque of the printed tip", "a hanging mass on a 50&#8202;mm lever at 1 and 3&#8202;N per tip; an arm of "
         "3.0&#8202;mm or more at 3&#8202;N means the pads under-predict torsion by 9&#8202;% or more (pads 2.74, finite elements "
         "3.12&#8202;mm)."),
        ("Pinch brake on D6", "thumb and index pinch the bench cylinder 15&#8202;mm off its centre of mass while the squeeze ramps "
         "down 0.5&#176;/s, tracked by the tag, 10 repeats, after <code>scripts/real_v1_trajectory_clearance.py</code>; point "
         "contact predicts a free swing at the ramp&#8217;s start, the pads 45&#176; at 0.34&#8202;N per pad."),
        ("Contact buffers of a full run", "log the largest contacts, rows and matches per world over a 40&#8202;M-step 2&#8202;mm pad "
         "run with <code>scripts/rl_train_cube.py</code>; per-scene buffers 1.3 times those peaks would cut its GPU memory from "
         "10.9&#8202;GB toward the probe&#8217;s 4.7&#8202;GB, and an overflow line falsifies the sizing."),
        ("Planner friction", "planning with friction 1 on a plant at 0.5 drops the cylinder in 8 of 8 runs (U4). Give "
         "<code>pad_usability_planner_model.ModelPlanner</code> a rollout friction (sampled from 0.4&#8211;1.0 per rollout) and "
         "rerun <code>run --configs icem_pin --variant mu0.5 --plant-set mu=0.5</code>; at least 7 of 8 three-finger endings at "
         "plant friction 0.5 and 0.7 make it the bench configuration, fewer keep the measured coefficient as a precondition."),
        ("Turn stalls of the pads on D3 and D8", "<code>scripts/pad_usability_backends.py run --dt 0.25 0.5</code> on D3, D5 and D8 "
         "for the pads and Drake, and the pads at impratio 1000; a stall that persists at impratio 1000 rules out creep as its "
         "cause."),
    ]
    return "<ol>" + "".join(f"<li><b>{a}.</b> {b}</li>" for a, b in items) + "</ol>"


def byline():
    return ("Scripts <code>scripts/pad_usability_timing.py</code>, <code>pad_usability_gpu_cost.py</code>, "
            "<code>pad_usability_rl.py</code>, <code>pad_usability_backends.py</code>, <code>pad_usability_bed.py</code>, "
            "<code>pad_usability_step.py</code>, <code>pad_usability_planner_model.py</code> &#183; builder "
            "<code>scripts/pad_usability_page.py</code> &#183; rows <code>docs/experiments/20261014-pad_usability/</code>")


def main():
    v = {"STYLE": P.style_block().replace("</style>", EXTRA_CSS + "</style>"), "BUILT": time.strftime("%Y-%m-%d %H:%M"),
         "BYLINE": byline(), "GLOSSARY": glossary(), "QUESTION": question(), "COST": cost_section(),
         "AGREE": agree_section(), "STEP": step_section(), "PLANNER": planner_section(), "RL": rl_section(),
         "LIMITS": limitations(), "OPEN": open_items(), "LEDE": lede()}
    v["FOOTER"] = ("<p>Rebuild: <code>.venv/bin/python scripts/pad_usability_page.py</code>. Rows and findings in "
                   "<code>docs/experiments/20261014-pad_usability/</code>: <code>cost/</code> (U1), <code>agree/</code> (U2), "
                   "<code>step/</code> (U3), <code>planner/</code> (U4), <code>rl/</code> (U5), each with "
                   "<code>findings_U&lt;k&gt;.json</code>; launchers and logs in <code>logs/20261014-pad_usability/</code>.</p>")
    t = open(TPL).read()
    for k, val in v.items():
        t = t.replace("{{" + k + "}}", val)
    left = sorted(set(x.split("}}")[0] for x in t.split("{{")[1:]))
    if left:
        raise SystemExit(f"unfilled placeholders: {left}")
    miss = sorted(set(re.findall(r"\[\[((?:fig|tab):[a-z0-9_]+)\]\]", t)) - set(REFS))
    if miss:
        raise SystemExit(f"unresolved references: {miss}")
    t = re.sub(r"\[\[((?:fig|tab):[a-z0-9_]+)\]\]", lambda m: str(REFS[m.group(1)]), t)
    t, n = render_tex(t)
    t = retro_style.apply(t)
    open(OUT, "w").write(t)
    print(f"formulas {n}; wrote {OUT} ({os.path.getsize(OUT) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
