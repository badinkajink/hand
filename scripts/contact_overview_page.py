#!/usr/bin/env python3
r"""Build docs/overviews/contact_model.html, the topic overview of the fingertip contact model.

    python3 scripts/contact_overview_page.py

Since 2026-10-09 the overview has a stable name and its revisions are the file's git history (owner, 2026-10-09); the
dated revisions of 2026-10-05 to 10-09 stay in docs/experiments/<date>-contact_overview/ as snapshots, and the page keeps
its artifact (artifact_url.txt in the 2026-10-09 folder).

Until 2026-10-09 each revision was a new dated file: 2026-10-05 (first), 10-07 (agreement of the pads with Drake
across MuJoCo, MuJoCo-Warp and Newton; effective mass; cost on one RL state), 10-08 (native-compliance candidates, Newton
hydroelastic tests, hand-object scale) and 10-09 (policies on four fingertip models; the plain page style).

The overview page of the fingertip contact work of 1-14 October 2026: the components of a fingertip contact model,
the contact mechanics (patch torque and the hinge balance, Winkler/hydroelastic and Hertz arms, lateral coupling), the
sphere-packed pad in MuJoCo (sampling, the soft-contact spring law, inverse-inertia calibration, friction creep as a
damper with cone saturation, the step bound of the semi-implicit step, condim 4), the reference and related models, and
the evidence: the comparison bed (docs/experiments/20261005-contact_bed/), the printed TPU block on the bed
(20261006-fingertip_backends/tip_T1, T2, T5.jsonl), the inverse-inertia study (20261005-pad_calibration/), the GPU batches
(20261005-gpu_scaling/), the hand-level costs (20261006-hom_turn3/plans_*.jsonl, 20261006-simulator_agreement/
plans_mujoco_pt.jsonl, 20261013-scale_turn_controllers/ and 20261014-scale_turn_cad_v1/ planners/timing.json,
20261006-rl_contact/same_state_timing.jsonl, the run_costs.jsonl of 20261008-contact_model_policies and
20261011-servo_plant_policies), the policy studies, Codex's step sweep (20261004-codex/data/task_timestep_results.json)
and the chain page (20261002-hom_chain/), whose style block, Figures 1-2 and films are reused. The derivation steps 1-9
follow deck 23 of the owner's contact tutorial (DECK below); their numbers are computed here from the constants and the
bed's rows (deck_values). Every number in the prose is computed here from those rows; figures whose rows are not yet
written render a short note instead. Math in the template, \( \) inline and \[ \] display, is LaTeX rendered to inline
SVG by scripts/texsvg.py, cached in docs/overviews/media/contact_texsvg_cache.json.
"""
from __future__ import annotations

import json
import math
import os
import re
import statistics
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import hom_chain_figures as G  # noqa: E402
import hom_chain_page as CP  # noqa: E402
import hom_contact_patch_page as R  # noqa: E402
import texsvg  # noqa: E402
import retro_style  # noqa: E402

EXP = os.path.join(ROOT, "docs/experiments")
D = os.path.join(EXP, "20261009-contact_overview")                 # last dated snapshot; its artifact_url.txt
OUT = os.path.join(ROOT, "docs", "overviews", "contact_model.html")
USE_DIR = os.path.join(ROOT, "docs/experiments/20261014-pad_usability")   # the 2026-10-14 usability study (U1-U5)
USE_PAGE = "docs/experiments/20261014-pad_usability/20261014-sphere_pad_cost_and_control.html"
TPL = os.path.join(ROOT, "scripts/contact_overview_page.template.html")
CHAIN_TPL = os.path.join(ROOT, "scripts/hom_chain_page.template.html")
BED = os.path.join(EXP, "20261005-contact_bed")
CAL = os.path.join(EXP, "20261005-pad_calibration/mass_calibration.jsonl")
GPU = os.path.join(EXP, "20261005-gpu_scaling")
CODEX = os.path.join(EXP, "20261004-codex")
CHAIN_D = os.path.join(EXP, "20261002-hom_chain")
MSC = os.path.join(EXP, "20261006-newton_mass_scaling")      # Newton hydroelastic with k_h divided by m_eff
SAD = os.path.join(EXP, "20261006-simulator_agreement")      # GPU pad rows of the bed, Newton brake
HT3 = os.path.join(EXP, "20261006-hom_turn3")
RLD = os.path.join(EXP, "20261006-rl_contact")
HT3_PATH = "docs/experiments/20261006-hom_turn3/20261006-servo_refit_hom_turn_pad_cost.html"
TEX_CACHE = os.path.join(ROOT, "docs", "overviews", "media", "contact_texsvg_cache.json")
TEX_SCALE = 1.1

CHAIN_PATH = "docs/experiments/20261002-hom_chain/20261002-hom_screwdriver_chain.html"
CHAIN_URL = "https://claude.ai/artifact/RN5kgsnDiKyuVSZYWrQexs"
BED_PATH = "docs/experiments/20261005-contact_bed/20261005-contact_model_bed.html"
BED_URL_FILE = os.path.join(BED, "artifact_url.txt")          # written after the bed page is published
ASIDES_PATH = "docs/experiments/20261004-codex/20261004-contact_research_asides.html"
LIT_NOTES = os.path.join(ROOT, "docs/notes/20261005-contact_literature_notes.md")

# Friction arm of the hydroelastic pressure law on the screwdriver: the field integrated over the sphere-cylinder overlap
# (scripts/hydroelastic_arm_integral.py: 0.820, 0.974, 1.281 mm at 0.5, 1, 3 N; c N^(1/4) to within 0.1 %). condim 4's
# mu_t schedule (mj:point4s:fit0.000996044x0.2498) keeps the earlier fit to Drake's torque, 2.3 % above it.
C_LAW, EXP_LAW = 0.974e-3, 0.25
MU = 1.0
M_TOOL = 0.024544
R_TOOL_MM = 12.5                           # screwdriver radius on the bed rig

# model key -> (label, colour variable, marker, dashed)
MODELS = {
    "mj_point3": ("MuJoCo point, condim 3", "var(--ink3)", "circle", False),
    "mj_point4s": ("MuJoCo condim 4, torsion rescheduled", "var(--c-c4)", "square", True),
    "mj_pads1": ("MuJoCo 1 mm sphere pad", "var(--c-sphere)", "circle", False),
    "drake_hydro": ("Drake hydroelastic", "var(--c-drake)", "circle", False),
    "newton_hydro_mc": ("Newton hydroelastic, mass-corrected", "var(--c-newton)", "diamond", True),
    "mjw_pads1": ("MuJoCo-Warp 1 mm sphere pad", "var(--c-sphere)", "square", True),
    "newton_pads1": ("Newton 1 mm sphere pad", "var(--c-sphere)", "diamond", True),
}
ORDER = ["mj_point3", "mj_point4s", "mj_pads1", "drake_hydro", "newton_hydro_mc"]
# Table 2 and Figure 9: the same pads stepped by MuJoCo-Warp and by Newton (bed rows of 20261006-simulator_agreement/)
TABLE_ORDER = ["mj_point3", "mj_point4s", "mj_pads1", "mjw_pads1", "newton_pads1", "drake_hydro", "newton_hydro_mc"]
GPU_MODELS = ("mjw_pads1", "newton_pads1", "newton_hydro_mc")
HTML_LBL = {
    "mj_point3": "MuJoCo point contact, condim&#160;3",
    "mj_point4s": "MuJoCo condim&#160;4, \\(\\mu_t\\) rescheduled",
    "mj_pads1": "MuJoCo 1&#8202;mm sphere pad",
    "mj_pads2": "MuJoCo 2&#8202;mm sphere pad",
    "mj_pads05": "MuJoCo 0.5&#8202;mm sphere pad",
    "drake_hydro": "Drake hydroelastic",
    "newton_hydro": "Newton hydroelastic",
    "newton_hydro_unreduced": "Newton hydroelastic, unreduced",
    "newton_hydro_mc": "Newton hydroelastic, \\(k_h/m_\\text{eff}\\)",
    "mjw_pads1": "MuJoCo-Warp 1&#8202;mm sphere pad",
    "newton_pads1": "Newton 1&#8202;mm sphere pad",
}
# chain specs (hom_chain.make_plant) and rig specs (hom_contact_rig.make_rig) -> bed model keys
SPEC_KEY = {
    "mj:point3": "mj_point3", "mj:point3:ir100": "mj_point3",
    "mj:point4s": "mj_point4s", "mj:point4s:fit0.000996044x0.2498:ir100": "mj_point4s",
    "mj:spheres:s1:rs0.75:tr0.02": "mj_pads1", "mj:spheres:s1:rs0.75:ir100:tr0.02": "mj_pads1",
    "drake:hydro:rt0.01": "drake_hydro", "drake:hydro:E1e7:r1:rt0.01": "drake_hydro",
}

EXTRA_CSS = """
:root{--c-newton:#7A4FA6}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--c-newton:#B58EDB}}
:root[data-theme="dark"]{--c-newton:#B58EDB}
nav.toc{padding:26px 0 6px;border-bottom:1px solid var(--rule2)}
nav.toc ol{margin:0;padding-left:22px;columns:2;column-gap:40px;font:400 15px/1.7 var(--f-display)}
nav.toc li{margin:0}
nav.toc a{color:var(--ink2);text-decoration:none}
nav.toc a:hover{color:var(--s2);text-decoration:underline}
@media (max-width:640px){nav.toc ol{columns:1}}
tr.grp td{text-transform:none}
.tnote{font:400 14px/1.55 var(--f-mono);color:var(--ink3);margin:-12px 0 24px;max-width:86ch}
table.cap td,table.agree td{font-size:15px}
table.cap th,table.agree th{white-space:normal}
table.agree td:first-child{min-width:15em}
td.ref{background:var(--sunk)}
table.cost td:first-child{min-width:18em}
table.cost td.num{white-space:nowrap}
td.cap-y{color:var(--good);font-weight:500}
td.cap-n{color:var(--ink3)}
td.cap-p{color:var(--s1)}
.pending{border:1px dashed var(--rule);border-radius:10px;padding:18px 20px;color:var(--ink3);font:400 14px/1.55 var(--f-mono);background:var(--card)}
.open li{margin:0 0 12px}
.open b{font-family:var(--f-display);font-weight:600}
dl.glossary{display:grid;grid-template-columns:minmax(9em,13em) 1fr;gap:10px 22px;margin:8px 0 0;font-size:15.5px}
dl.glossary dt{font:600 14px/1.5 var(--f-display);color:var(--ink)}
dl.glossary dd{margin:0;color:var(--ink2)}
@media (max-width:640px){dl.glossary{grid-template-columns:1fr}dl.glossary dd{margin-bottom:8px}}
.legendrow{display:flex;flex-wrap:wrap;justify-content:center;gap:6px 18px;margin:8px 0 0;font:400 15px/1.4 var(--f-mono);color:var(--ink2)}
"""


# ------------------------------------------------------------------------------------------ terms and metrics

GLOSSARY = [
    ("test bed", "The two-pad pinch rig of the 10-01 study. Two fingertip pads on rails pinch the screwdriver, and numbered "
            "tasks load it: 1 pull, 2 twist, 3 roll, 4 shake and 5 brake. Task&#160;6 varies the settings that set creep, "
            "task&#160;7 the time step, and task&#160;8 cycles a load below slip. The fingertip is the 10.55&#8202;mm sphere unless "
            "the printed block is named."),
    ("pad", "One fingertip&#8217;s contact with the tool. \\(N\\) is its normal force."),
    ("friction arm \\(\\bar r\\)", "The mean distance of the contact patch from its centre, weighted by pressure, in mm. A pad "
            "spins once the torque about its normal exceeds \\(\\mu N\\bar r\\)."),
    ("effective \\(\\mu\\)", "The pull force at the onset of slip (task&#160;1) divided by \\(2N\\). Rigid Coulomb friction "
            "gives 1."),
    ("creep", "Sliding under half the slip load, in &#181;m/s for a pull or &#176;/s for a twist. Rigid Coulomb friction "
            "gives 0."),
    ("slip speed", "The mean sliding speed in the 50&#8202;ms after slip starts, in mm/s."),
    ("arm at spin onset", "The twist torque at which the tool starts to spin (task&#160;2), divided by \\(2\\mu N\\), in mm per "
            "pad."),
    ("arm spinning at 1&#8202;rad/s", "The twist torque while the tool spins at 1&#8202;rad/s about the pinch axis "
            "(task&#160;2), divided by \\(2\\mu N\\), in mm per pad."),
    ("rolling ratio", "The tool&#8217;s rotation times its radius, divided by half the pad travel (task&#160;3). Rolling without "
            "slip gives 1.008."),
    ("drift per cycle", "The tool&#8217;s displacement per cycle of a 5&#8202;Hz shake below the slip load (task&#160;4), in "
            "&#181;m."),
    ("swing end", "The tool&#8217;s final angle as the pinch force falls from 6 to 0.2&#8202;N (task&#160;5). Hanging is "
            "90&#176;."),
    ("brake force at 45&#176;", "The pinch force per pad at which the braked tool passes 45&#176; below horizontal "
            "(task&#160;5), in N. More friction torque keeps the tool up down to a lower force."),
    ("tool turn", "The rotation of the tool&#8217;s axis toward vertical in a replayed three-finger turn. <i>Within 3&#176;</i> "
            "counts the replays that end within 3&#176; of Drake&#8217;s."),
    ("presliding", "The tool&#8217;s displacement under a tangential load below the slip force, which returns when the load is "
            "removed. In task&#160;8 it is read at half the slip force, in &#181;m (Mindlin&#8217;s solution at 1&#8202;N: 40.3)."),
    ("loop area", "The energy lost per cycle of a tangential load below slip, in &#181;J. A spring loses none, and partial slip "
            "traces a closed loop (Mindlin at 1&#8202;N: 6.13)."),
    ("&#181;s per step", "The wall time of one physics step of 1&#8202;ms on one CPU core, or of one world-step on the GPU."),
    ("\\(t_c\\), \\(d_0\\), \\(\\hat\\Lambda\\), \\(m_\\text{eff}\\)", "MuJoCo&#8217;s contact settings: the time constant of "
            "<code>solref</code>, the impedance of <code>solimp</code>, and the contact&#8217;s inverse inertia \\(\\hat\\Lambda\\), "
            "with the effective mass \\(m_\\text{eff} = 1/\\hat\\Lambda\\). Together they set the sphere stiffness, (10)."),
    ("\\(k_h\\), \\(k_f\\)", "Newton&#8217;s hydroelastic stiffness \\(E/h\\), in N/m&#179;, and its friction gain, a damping "
            "coefficient in N&#8202;s/m."),
    ("impratio, condim", "impratio is MuJoCo&#8217;s ratio of friction stiffness to normal stiffness. condim sets the friction "
            "directions of a contact: condim&#160;3 has two tangential directions, and condim&#160;4 adds torsion about the "
            "normal."),
    ("\\(d'\\), \\(q\\)", "The impedance of \\(n\\) contact rows of one pad that move together, (13), and \\(q=d_0/d'\\), which "
            "sets the step bound (14)."),
    ("relaxation time \\(t_r\\)", "Drake&#8217;s contact dissipation time, its damping over its stiffness: 10&#8202;ms. A pad "
            "sphere&#8217;s damping over its rest stiffness is \\(2t_c\\), (10). Released from a push, a sphere returns to within "
            "5&#8202;% of rest in \\(4.74\\,t_c\\)."),
]

# terms of the policy and cost paragraphs, shown on this page only (other pages import GLOSSARY)
GLOSSARY_RL = [
    ("world-step", "One physics step of one world in a GPU batch. Its cost is the wall time of the batch step divided by "
            "the number of worlds."),
    ("PPO iteration", "One iteration of the reorientation trainer, which uses proximal policy optimization (PPO) over "
            "2,048 envs: 24 env steps per env and the policy update, in s. Zero-action throughput is the number of env steps "
            "per second with the actions held at zero after the scripted grasp; it leaves out the update."),
    ("held, held cosine", "A rollout or replay holds the tool when it ends with at least two fingertips pressing on it and "
            "the tool above 60&#8202;mm. The held cosine is the cosine between the tool&#8217;s axis and the vertical at the end, "
            "averaged over the held rollouts; 1 is upright."),
    ("servo plant, 1&#8202;N&#8202;m plant", "Models of the simulated hand&#8217;s finger servos (the plant). The servo plant has "
            "the SCS0009&#8217;s gain of 0.72&#8202;N&#8202;m/rad and torque limit of 0.18&#8202;N&#8202;m. The 1&#8202;N&#8202;m plant, used in "
            "the studies of 2026-10-06 and 10-08, has a 1&#8202;N&#8202;m limit, 4.4 times the servo&#8217;s stall torque."),
]


def glossary_html(keys=None, extra=()):
    items = [g for g in list(GLOSSARY) + list(extra) if keys is None or g[0] in keys]
    return ('<dl class="glossary">' + "".join(f"<dt>{t}</dt><dd>{d}</dd>" for t, d in items) + "</dl>")


# ------------------------------------------------------------------------------------------ data

def load(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def bed(name):
    """Rows of one bed task, CPU models and Newton merged, with a `model` key on every row."""
    rows = load(os.path.join(BED, f"{name}.jsonl")) + load(os.path.join(BED, f"{name}_newton.jsonl"))
    rows += load(os.path.join(MSC, f"{name}_newton.jsonl")) + load(os.path.join(SAD, f"{name}_newton.jsonl"))
    for r in rows:
        if "model" not in r:
            r["model"] = SPEC_KEY.get(r.get("chain_spec")) or SPEC_KEY.get(r.get("spec")) or _pad_key(r.get("spec", ""))
    return rows


def _pad_key(spec):
    m = re.search(r"spheres:s([0-9.]+)", spec or "")
    if not m:
        return spec
    return {"1": "mj_pads1", "2": "mj_pads2", "0.5": "mj_pads05"}.get(m.group(1), spec)


def first(r, *keys):
    for k in keys:
        if r is not None and k in r and r[k] is not None:
            return r[k]
    return None


def pick(rows, model, **kw):
    out = [r for r in rows if r.get("model") == model and all(_eq(r.get(k), v) for k, v in kw.items())]
    return out[-1] if out else None


def _eq(a, b):
    if isinstance(b, float) or isinstance(a, float):
        try:
            return abs(float(a) - float(b)) < 1e-9
        except (TypeError, ValueError):
            return False
    return a == b


num = CP.num
US = CP.US


def fmt(x, nd=2):
    if x is None:
        return "&#8211;"
    if isinstance(x, str):
        return x
    return num(x, f".{nd}f")


# ------------------------------------------------------------------------------------------ svg helpers

def _sentence(label):
    """Capitalise an axis title that starts with a plain lower-case word (sentence case); symbols such as kp stay."""
    w = label.split(" ", 1)[0]
    return label[0].upper() + label[1:] if len(w) > 2 and w.isalpha() and w.islower() else label


def _panel(out, x0, y0, w, h, xs, ys, xt, yt, xlab, ylab, logx=False, logy=False, xfmt="{:g}", yfmt="{:g}"):
    xlab, ylab = _sentence(xlab), _sentence(ylab)
    tx = (lambda v: math.log10(v)) if logx else (lambda v: v)
    ty = (lambda v: math.log10(v)) if logy else (lambda v: v)
    fx = lambda v: x0 + (tx(v) - tx(xs[0])) / (tx(xs[1]) - tx(xs[0])) * w  # noqa: E731
    fy = lambda v: y0 + h - (ty(v) - ty(ys[0])) / (ty(ys[1]) - ty(ys[0])) * h  # noqa: E731
    for v in yt:
        out.append(f'<line x1="{x0}" x2="{x0 + w}" y1="{fy(v):.1f}" y2="{fy(v):.1f}" style="stroke:var(--rule2)"/>'
                   f'<text x="{x0 - 8}" y="{fy(v) + 4:.1f}" text-anchor="end" style="fill:var(--ink3)">{yfmt(v) if callable(yfmt) else yfmt.format(v)}</text>')
    for v in xt:
        out.append(f'<line x1="{fx(v):.1f}" x2="{fx(v):.1f}" y1="{y0}" y2="{y0 + h}" style="stroke:var(--rule2)"/>'
                   f'<text x="{fx(v):.1f}" y="{y0 + h + 17}" text-anchor="middle" style="fill:var(--ink3)">{xfmt.format(v)}</text>')
    out.append(f'<line x1="{x0}" x2="{x0 + w}" y1="{y0 + h}" y2="{y0 + h}" style="stroke:var(--ink3)"/>'
               f'<text x="{x0 + w / 2}" y="{y0 + h + 38}" text-anchor="middle" style="fill:var(--ink2)">{xlab}</text>'
               f'<text x="{x0}" y="{y0 - 12}" style="fill:var(--ink2)">{ylab}</text>')
    return fx, fy


def _path(out, fx, fy, pts, colour, dashed=False, width=2.2):
    pts = [(x, y) for x, y in pts if x is not None and y is not None]
    if len(pts) < 2:
        return
    d = "M" + " L".join(f"{fx(x):.1f},{fy(y):.1f}" for x, y in pts)
    dash = ";stroke-dasharray:6 5" if dashed else ""
    out.append(f'<path d="{d}" style="fill:none;stroke-width:{width};stroke:{colour}{dash};stroke-linejoin:round"/>')


def _marker(out, x, y, colour, shape="circle", hollow=False, r=4.6, title=None):
    fill = "var(--card)" if hollow else colour
    stroke = colour if hollow else "var(--card)"
    t = f"<title>{title}</title>" if title else ""
    if shape == "square":
        out.append(f'<rect x="{x - r:.1f}" y="{y - r:.1f}" width="{2 * r:.1f}" height="{2 * r:.1f}" '
                   f'style="fill:{fill};stroke:{stroke};stroke-width:2">{t}</rect>')
    elif shape == "diamond":
        q = r * 1.3
        out.append(f'<path d="M{x:.1f},{y - q:.1f} L{x + q:.1f},{y:.1f} L{x:.1f},{y + q:.1f} L{x - q:.1f},{y:.1f} Z" '
                   f'style="fill:{fill};stroke:{stroke};stroke-width:2">{t}</path>')
    elif shape == "cross":
        out.append(f'<path d="M{x - r:.1f},{y - r:.1f} L{x + r:.1f},{y + r:.1f} M{x - r:.1f},{y + r:.1f} L{x + r:.1f},{y - r:.1f}" '
                   f'style="fill:none;stroke:{colour};stroke-width:2.2">{t}</path>')
    else:
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" style="fill:{fill};stroke:{stroke};stroke-width:2">{t}</circle>')


COLUMN_PX = 828                     # width of the plain style's text column; charts scale to it


def _svg_open(w, h, label, fs=12):
    """fs: default text size in viewBox units, raised so that text shows at 13 px or more once the chart is scaled to
    the 828 px column (owner 2026-10-09: chart text at least 13 px as displayed)."""
    fs = max(fs, math.ceil(13.0 * w / COLUMN_PX * 10) / 10)
    return [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{label}" font-family="var(--f-mono)" font-size="{fs:g}">']


def _legend_html(items):
    """items: (label, colour, dashed, shape)"""
    out = ['<div class="legendrow">']
    for lab, col, dashed, shape in items:
        sym = {"circle": "&#9679;", "square": "&#9632;", "diamond": "&#9670;", "cross": "&#215;", None: ""}[shape]
        out.append(f'<span><span class="key{" dash" if dashed else ""}" style="border-color:{col}"></span>'
                   f'<span style="color:{col}">{sym}</span> {lab}</span>')
    out.append("</div>")
    return "".join(out)


def pending(text):
    return f'<div class="pending">{text}</div>'


# ------------------------------------------------------------------------------------------ Figure 1: representations

def svg_models():
    W, H = 990, 262
    out = _svg_open(W, H, "Five fingertip contact representations: point contact, condim 4 with a torsional coefficient, "
                          "a pad of small spheres, a hydroelastic pressure field over the overlap, and a lattice with "
                          "lateral springs.", fs=12)
    out.append('<defs><marker id="ovar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
               'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--ink2)"/></marker></defs>')
    titles = ["(a) point contact", "(b) condim 4, μt(N)", "(c) sphere pad", "(d) hydroelastic field",
              "(e) sphere lattice (CSLC)"]
    notes = [("no torque about n", "one Coulomb cone"),
             ("τ ≤ μt N", "μt = μ c N<tspan dy='-5' font-size='0.8em'>1/4</tspan><tspan dy='5'>, each step</tspan>"),
             ("a soft contact per sphere", "T = Σ ρᵢ fₜᵢ"),
             ("p = E·δ/R", "summed over the surface"),
             ("anchor + side springs", "spread past the overlap")]
    Rr, yl = 62.0, 150.0                     # fingertip radius (px) and tool surface
    for i in range(5):
        x0 = 8 + i * 197
        cx = x0 + 96
        soft = i >= 2
        cy = yl - Rr + (7.0 if soft else 0.0)
        out.append(f'<text x="{cx}" y="22" text-anchor="middle" style="fill:var(--ink);font:600 16.5px var(--f-display)">{titles[i]}</text>')
        # tool
        out.append(f'<rect x="{x0 + 6}" y="{yl}" width="180" height="34" rx="3" style="fill:var(--sunk);stroke:var(--ink3);stroke-width:1"/>')
        out.append(f'<text x="{x0 + 176}" y="{yl + 22}" text-anchor="end" style="fill:var(--ink3);font-size:16px">tool</text>')
        if i in (0, 1):
            out.append(f'<circle cx="{cx}" cy="{cy:.1f}" r="{Rr}" style="fill:var(--card);stroke:var(--ink2);stroke-width:1.6"/>')
            out.append(f'<line x1="{cx}" y1="{yl}" x2="{cx}" y2="{yl - 52}" style="stroke:var(--ink2);stroke-width:1.8" marker-end="url(#ovar)"/>'
                       f'<text x="{cx + 6}" y="{yl - 44}" style="fill:var(--ink2)">N</text>')
            for s in (-1, 1):
                out.append(f'<line x1="{cx}" y1="{yl}" x2="{cx + s * 34}" y2="{yl - 34}" style="stroke:var(--c-ref);stroke-dasharray:4 3"/>')
            out.append(f'<circle cx="{cx}" cy="{yl}" r="4.5" style="fill:var(--ink)"/>')
            if i == 1:
                out.append(f'<ellipse cx="{cx}" cy="{yl - 30}" rx="22" ry="7" style="fill:none;stroke:var(--c-c4);stroke-width:2"/>'
                           f'<path d="M{cx + 20},{yl - 33} l7,1 l-4,5" style="fill:none;stroke:var(--c-c4);stroke-width:2"/>'
                           f'<text x="{cx + 27}" y="{yl - 24}" style="fill:var(--c-c4)">τ</text>')
        elif i in (2, 4):
            out.append(f'<circle cx="{cx}" cy="{cy:.1f}" r="{Rr}" style="fill:none;stroke:var(--ink3);stroke-width:1;stroke-dasharray:3 3"/>')
            rs = 4.2
            core = Rr - 2 * rs - 7
            out.append(f'<circle cx="{cx}" cy="{cy:.1f}" r="{core:.1f}" style="fill:var(--card);stroke:var(--ink2);stroke-width:1.4"/>')
            pts = []
            for k in range(-8, 9):
                ang = math.radians(90 + k * 6.4)
                px, py = cx + (Rr - rs) * math.cos(ang), cy + (Rr - rs) * math.sin(ang)
                pts.append((px, py, ang))
            for j, (px, py, ang) in enumerate(pts):
                if i == 4:
                    ax, ay = cx + core * math.cos(ang), cy + core * math.sin(ang)
                    out.append(_zigzag(ax, ay, px - rs * math.cos(ang), py - rs * math.sin(ang), "var(--ink3)"))
                    if j + 1 < len(pts):
                        qx, qy, _ = pts[j + 1]
                        out.append(_zigzag(px, py, qx, qy, "var(--c-newton)", amp=2.0, n=3))
            for px, py, ang in pts:
                over = py + rs - yl
                if over > 0:
                    out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{rs}" style="fill:var(--c-sphere);stroke:var(--card);stroke-width:1"/>')
                    out.append(f'<rect x="{px - 1.8:.1f}" y="{yl + 2}" width="3.6" height="{over * 3.2:.1f}" style="fill:var(--c-sphere)"/>')
                else:
                    out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{rs}" style="fill:var(--card);stroke:var(--ink3);stroke-width:1"/>')
        else:   # hydroelastic
            out.append(f'<circle cx="{cx}" cy="{cy:.1f}" r="{Rr}" style="fill:var(--card);stroke:var(--ink2);stroke-width:1.6"/>')
            hx = math.sqrt(max(Rr ** 2 - (yl - cy) ** 2, 0.0))
            seg = (f'M{cx - hx:.1f},{yl} A{Rr},{Rr} 0 0 0 {cx + hx:.1f},{yl} Z')
            out.append(f'<path d="{seg}" style="fill:var(--c-drake);fill-opacity:.35;stroke:var(--c-drake);stroke-width:1"/>')
            prof = []
            for k in range(41):
                x = cx - hx + 2 * hx * k / 40
                depth = (cy + math.sqrt(max(Rr ** 2 - (x - cx) ** 2, 0.0))) - yl
                prof.append((x, yl + 4 + depth * 2.6))
            d = f"M{cx - hx:.1f},{yl + 4} " + " ".join(f"L{x:.1f},{y:.1f}" for x, y in prof) + f" L{cx + hx:.1f},{yl + 4} Z"
            out.append(f'<path d="{d}" style="fill:var(--c-drake);fill-opacity:.55;stroke:var(--c-drake);stroke-width:1.2"/>')
        out.append(f'<text x="{cx}" y="{yl + 58}" text-anchor="middle" style="fill:var(--ink2);font:16px var(--f-display)">{notes[i][0]}</text>'
                   f'<text x="{cx}" y="{yl + 78}" text-anchor="middle" style="fill:var(--ink3);font:16px var(--f-display)">{notes[i][1]}</text>')
    out.append("</svg>")
    return "".join(out)


def _zigzag(x1, y1, x2, y2, colour, amp=2.4, n=4):
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy)
    if L < 1e-6:
        return ""
    ux, uy = dx / L, dy / L
    nx, ny = -uy, ux
    pts = [(x1, y1)]
    for k in range(1, 2 * n):
        t = k / (2 * n)
        s = amp if k % 2 else -amp
        pts.append((x1 + dx * t + nx * s, y1 + dy * t + ny * s))
    pts.append((x2, y2))
    return ('<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts) +
            f'" style="fill:none;stroke:{colour};stroke-width:1"/>')


# ------------------------------------------------------------------------------------------ Figure 2: arm scaling

def twist_arm(r):
    """Per-pad friction arm at spin onset (mm) from a twist row."""
    v = first(r, "rbar_onset_mm")
    if v is not None:
        return v
    tau = first(r, "tau_onset_Nm", "tau_onset")
    N = first(r, "N")
    if tau is not None and N:
        return tau / (2 * MU * N) * 1e3
    return None


def svg_scaling(tw, tor):
    W, H = 900, 440
    out = _svg_open(W, H, "Friction arm against pad force on log axes: the foundation law, a Hertz law of the same value at "
                          "1 N, and the arm each simulator delivers.", fs=16)
    fx, fy = _panel(out, 80, 46, 480, 300, (0.2, 5.0), (0.5, 4.2), (0.25, 0.5, 1, 2, 4), (0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0),
                    "pad force N per pad (N), log scale", "friction arm r̄ per pad (mm), log scale", True, True)
    Ns = [0.2 * (25 ** (k / 60)) for k in range(61)]
    _path(out, fx, fy, [(N, C_LAW * 1e3 * N ** EXP_LAW) for N in Ns], "var(--c-ref)")
    _path(out, fx, fy, [(N, C_LAW * 1e3 * N ** (1 / 3)) for N in Ns], "var(--c-ref)", dashed=True)
    out.append(f'<text x="{fx(0.3):.1f}" y="{fy(C_LAW * 1e3 * 0.3 ** EXP_LAW) - 12:.1f}" text-anchor="middle" style="fill:var(--ink3)">'
               f'Foundation, c N<tspan dy="-6" font-size="0.8em">1/4</tspan><tspan dy="6"> </tspan></text>'
               f'<text x="{fx(4.4):.1f}" y="{fy(C_LAW * 1e3 * 4.4 ** (1 / 3)) - 9:.1f}" text-anchor="end" style="fill:var(--ink3)">'
               f'Hertz, N<tspan dy="-6" font-size="0.8em">1/3</tspan><tspan dy="6"> </tspan></text>')
    legend = []
    have_onset = False
    for k in ORDER:
        if k == "mj_point3":
            continue
        lab, col, shape, dashed = MODELS[k]
        pts = []
        for r in tw:
            if r.get("model") == k and _eq(r.get("dt_ms"), 1.0) and r.get("status", "complete") == "complete":
                a = twist_arm(r)
                if a and 0.5 <= a <= 4.2:
                    pts.append((r["N"], a))
        if pts:
            have_onset = True
            pts.sort()
            _path(out, fx, fy, pts, col, dashed=dashed, width=1.4)
            for N, a in pts:
                _marker(out, fx(N), fy(a), col, shape, title=f"{lab}: {a:.3f} mm at {N:g} N (onset)")
            legend.append((lab + " (onset)", col, dashed, shape))
    if not have_onset:
        # 10-01 rig: steady kinetic torque at 0.5 rad/s, per-pad arm
        for spec, k in (("drake:hydro:E1e7:r1:rt0.01", "drake_hydro"), ("mj:spheres:s1:rs0.75:ir100:tr0.02", "mj_pads1"),
                        ("mj:point4s:fit0.000996044x0.2498:ir100", "mj_point4s")):
            lab, col, shape, dashed = MODELS[k]
            pts = sorted((r["N"], r["rbar_per_pad_mm"]) for r in tor if r["spec"] == spec and r.get("rbar_per_pad_mm"))
            if pts:
                for N, a in pts:
                    _marker(out, fx(N), fy(a), col, shape, hollow=True, title=f"{lab}: {a:.3f} mm at {N:g} N (sliding)")
                legend.append((lab + " (sliding, 10-01 rig)", col, dashed, shape))
    # ratio panel
    x0 = 615
    out.append(f'<text x="{x0}" y="50" style="fill:var(--ink2)">Arm ratio, 3 N over 0.5 N</text>')
    rows = [("Foundation, 6<tspan dy='-6' font-size='0.8em'>1/4</tspan><tspan dy='6'> </tspan>", 6 ** 0.25, "var(--c-ref)"), ("Hertz, 6<tspan dy='-6' font-size='0.8em'>1/3</tspan><tspan dy='6'> </tspan>", 6 ** (1 / 3), "var(--c-ref)")]
    for k in ORDER:
        lo = pick(tw, k, N=0.5, dt_ms=1.0)
        hi = pick(tw, k, N=3.0, dt_ms=1.0)
        if lo and hi and twist_arm(lo) and twist_arm(hi) and 1.3 <= twist_arm(hi) / twist_arm(lo) <= 2.0:
            lab = MODELS[k][0].replace("MuJoCo ", "").replace(", torsion rescheduled", ", rescheduled").replace(", mass-corrected", "")
            rows.append((lab[:1].upper() + lab[1:], twist_arm(hi) / twist_arm(lo), MODELS[k][1]))
    bx = lambda v: x0 + (v - 1.3) / (2.0 - 1.3) * 180  # noqa: E731
    for j, (lab, val, col) in enumerate(rows):
        y = 82 + j * 40
        # label above its own line and the value past the line's right end, so no line crosses a number
        out.append(f'<text x="{x0}" y="{y}" style="fill:var(--ink2);font-size:15px">{lab}</text>'
                   f'<line x1="{x0}" x2="{x0 + 180}" y1="{y + 12}" y2="{y + 12}" style="stroke:var(--rule2)"/>'
                   f'<circle cx="{bx(val):.1f}" cy="{y + 12}" r="5" style="fill:{col};stroke:var(--card);stroke-width:2"/>'
                   f'<text x="{x0 + 190}" y="{y + 17}" style="fill:var(--ink);font-size:15px">{val:.3f}</text>')
    for v in (1.4, 1.6, 1.8, 2.0):
        out.append(f'<text x="{bx(v):.1f}" y="{82 + len(rows) * 40 + 4}" text-anchor="middle" style="fill:var(--ink3);font-size:14.5px">{v:g}</text>')
    out.append("</svg>")
    return "".join(out) + _legend_html([("foundation law (3)", "var(--c-ref)", False, None),
                                        ("Hertz, equal at 1 N", "var(--c-ref)", True, None)] + legend), have_onset


# ------------------------------------------------------------------------------------------ Figure 5: mass calibration

def svg_mass(cal):
    rows = [r for r in cal if r.get("exp") == "pinch" and _eq(r.get("dt_ms"), 1.0)]
    if not rows:
        return pending("pad_mass_calibration rows missing")
    W, H = 980, 380
    out = _svg_open(W, H, "Friction arm of the 1 mm pad against tool mass: with the inverse inertia fixed at its nominal value "
                          "the arm grows for a light tool and shrinks for a heavy one; calibrated at load it stays on the "
                          "hydroelastic law.", fs=16)
    fx, fy = _panel(out, 80, 46, 600, 260, (0.2, 5.0), (0.4, 2.0), (0.25, 0.5, 1, 2, 4), (0.5, 1.0, 1.5, 2.0),
                    "tool mass relative to the real screwdriver (24.5 g), log scale", "friction arm r̄ per pad (mm)",
                    True, False, xfmt="{:g}×")
    for N in (0.5, 3.0):
        law = C_LAW * 1e3 * N ** EXP_LAW
        out.append(f'<line x1="{fx(0.2):.1f}" x2="{fx(5.0):.1f}" y1="{fy(law):.1f}" y2="{fy(law):.1f}" '
                   f'style="stroke:var(--c-ref);stroke-width:1.6"/>'
                   f'<text x="{fx(5.0) + 8:.1f}" y="{fy(law) + 4:.1f}" style="fill:var(--ink3)">law, {N:g} N</text>')
        for calib, dashed, hollow in (("fixed", True, True), (None, False, False)):
            pts = sorted((r["mscale"], r["rbar_mm"]) for r in rows if _eq(r.get("N_cmd"), N) and
                         ((r.get("calib") == "fixed") == (calib == "fixed")))
            _path(out, fx, fy, pts, "var(--c-sphere)", dashed=dashed)
            for m, a in pts:
                _marker(out, fx(m), fy(a), "var(--c-sphere)", "circle", hollow=hollow,
                        title=f"{'fixed' if hollow else 'load-time'} inverse inertia, {m:g}x mass, {N:g} N: {a:.3f} mm")
    fixed = {(r["mscale"], r["N_cmd"]): r for r in rows if r.get("calib") == "fixed"}
    for (m, N), r in fixed.items():
        if m in (0.25, 4.0) and N == 3.0:
            ref = [q for q in rows if q.get("calib") != "fixed" and _eq(q["mscale"], m) and _eq(q["N_cmd"], N)]
            if ref:
                dv = (r["rbar_mm"] / ref[0]["rbar_mm"] - 1) * 100
                out.append(f'<text x="{fx(m) + (10 if m < 1 else -10):.1f}" y="{fy(r["rbar_mm"]) - 10:.1f}" '
                           f'text-anchor="{"start" if m < 1 else "end"}" style="fill:var(--c-sphere)">{dv:+.0f} %</text>')
    out.append("</svg>")
    return "".join(out) + _legend_html([("inverse inertia fixed at the nominal tool", "var(--c-sphere)", True, "circle"),
                                        ("inverse inertia taken at load", "var(--c-sphere)", False, "circle"),
                                        ("hydroelastic law", "var(--c-ref)", False, None)])


def _rng(v, nd=3):
    a, b = f"{min(v):.{nd}f}", f"{max(v):.{nd}f}"
    return a if a == b else f"{a}&#8211;{b}"


def mass_text(cal):
    rows = [r for r in cal if r.get("exp") == "pinch" and _eq(r.get("dt_ms"), 1.0)]
    if not rows:
        return ""
    f = {(r["mscale"], r["N_cmd"]): r for r in rows if r.get("calib") == "fixed"}
    lt = {(r["mscale"], r["N_cmd"]): r for r in rows if r.get("calib") != "fixed"}
    k4, k025 = f[(4.0, 3.0)]["k_ratio"], f[(0.25, 3.0)]["k_ratio"]
    d4 = (f[(4.0, 3.0)]["rbar_mm"] / lt[(4.0, 3.0)]["rbar_mm"] - 1) * 100
    d025 = (f[(0.25, 3.0)]["rbar_mm"] / lt[(0.25, 3.0)]["rbar_mm"] - 1) * 100
    lo = [lt[(m, 0.5)]["rbar_mm"] for m in (0.25, 1.0, 4.0)]
    hi = [lt[(m, 3.0)]["rbar_mm"] for m in (0.25, 1.0, 4.0)]
    lawlo, lawhi = C_LAW * 1e3 * 0.5 ** EXP_LAW, C_LAW * 1e3 * 3 ** EXP_LAW
    return (f"The calibration rig&#8217;s railed pads have the thumb&#8217;s inverse weight. A sphere calibrated there with the "
            f"nominal tool comes out {k4:.2f}&#215; too stiff when reused for a tool four times heavier, and {k025:.2f}&#215; for "
            f"a tool four times lighter. Computing \\(\\hat\\Lambda\\) from the compiled model for each pad and object at load, "
            f"and \\(d_0\\) from it, removes this dependence at no run-time cost. MuJoCo computes <code>body_invweight0</code> "
            f"once, at the reference pose <code>qpos0</code>, and does not update it as the bodies move. The value read after "
            f"compiling is therefore the one the solver uses. With it the arm is {_rng(lo)}&#8202;mm at 0.5&#8202;N and "
            f"{_rng(hi)}&#8202;mm at 3&#8202;N for all three masses, {abs(statistics.mean(lo) / lawlo - 1) * 100:.0f} and "
            f"{abs(statistics.mean(hi) / lawhi - 1) * 100:.0f}&#8202;% under the hydroelastic law. With the fixed value the arm "
            f"moves by {num(d025, '+.0f')} and {num(d4, '+.0f')}&#8202;% (Figure&#160;5).")


# ------------------------------------------------------------------------------------------ creep and stability text

def creep_t6_text(rows):
    """Bed task 6 at 1 N and a 1 ms step (part T1, the 1 mm pad): creep against impratio and with noslip iterations, one
    case per setting (the bed page quotes it; 2026-10-14: clean values in place of medians pooled over unlike variants)."""
    at = lambda kap, ns: next((r["creep_mm_s"] * 1e3 for r in rows if r.get("part") == "T1" and r.get("model") == "mj_pads1"  # noqa: E731
                               and _eq(r.get("N"), 1.0) and _eq(r.get("dt_ms"), 1.0) and _eq(r.get("impratio"), kap)
                               and r.get("noslip_iterations", 0) == ns and r.get("status") == "complete"), None)
    kaps = [k for k in (100.0, 1000.0, 3000.0, 10000.0) if at(k, 0) is not None]
    if len(kaps) < 2:
        return ""
    vals = [f"{at(k, 0):.2f}" for k in kaps]
    names = [f"{k:,.0f}" if k >= 10000 else f"{k:.0f}" for k in kaps]
    s = (f"At 1&#8202;N and a 1&#8202;ms step the 1&#8202;mm pad creeps {', '.join(vals[:-1])} and {vals[-1]}&#8202;&#181;m/s at impratio "
         f"{', '.join(names[:-1])} and {names[-1]}")
    ns = [at(100.0, n) for n in (3, 10)]
    if all(x is not None and x < 0 for x in ns):
        s += (f", and three or ten <code>noslip_iterations</code> turn the creep into a slide of {abs(ns[0]):.2f}&#8202;&#181;m/s "
              "against the load")
    return s + "."


def _tc_of(spec, default=0.01):
    """Normal-row time constant (s) of a pad spec string: t_c = tr / 2 (scripts/hom_contact_rig.py)."""
    m = re.search(r"tr([0-9.]+)", spec or "")
    return float(m.group(1)) / 2 if m else default


def _same4(a, b):
    return f"{a:.4g}" == f"{b:.4g}"


def creep_text(pull, creep, tip_pull=None):
    """Step 8 (deck 23, creep-mechanism): the friction damper c = 2 kappa k_s t_c of the test bed's 1 mm pad, the
    equal-share and cone-saturated predictions against the creep of task 6 (creep.jsonl part T1, which stores both), the
    factor over Drake per tip and step (pull_slip.jsonl, tip_T1.jsonl), the 1/(kappa t_c) law and noslip."""
    out = []
    t1 = [r for r in creep if r.get("part") == "T1" and _eq(r.get("dt_ms"), 1.0) and r.get("status") == "complete"]
    sel = lambda model, kap, ns=0: sorted((r for r in t1 if r.get("model") == model and _eq(r.get("impratio"), kap)  # noqa: E731
                                           and r.get("noslip_iterations", 0) == ns), key=lambda r: r["N"])
    base = sel("mj_pads1", 100.0)
    if not base:
        return ("By (12), creep falls with a larger <code>impratio</code> and a longer \\(t_c\\). Task&#160;6 on the test bed, "
                "which tests both, has not run.")
    r0 = base[0]
    d0, lam, tc, kap = r0["pred_d0"], r0["pred_Lambda"], r0["pred_tc_s"], r0["pred_impratio"]
    ks = 1.0 / ((1 - d0) * lam * tc ** 2)
    c = 2 * kap * ks * tc
    lst = lambda v, nd: ", ".join(f"{x:.{nd}f}" for x in v[:-1]) + " and " + f"{v[-1]:.{nd}f}"  # noqa: E731
    meas = [r["creep_mm_s"] * 1e3 for r in base]
    cone = [r["pred_v_cone_mm_s"] * 1e3 for r in base]
    same = all(_same4(a, b) for a, b in zip(meas, cone))
    out.append(f"For the test bed&#8217;s 1&#8202;mm pad (\\(d_0\\)&#8202;=&#8202;{d0:.3f}, "
               f"\\(\\hat\\Lambda\\)&#8202;=&#8202;{lam:.2f}&#8202;kg\\(^{{-1}}\\), \\(t_c\\)&#8202;=&#8202;{tc * 1e3:g}&#8202;ms, "
               f"\\(\\kappa\\)&#8202;=&#8202;{kap:g}), each touching sphere is a damper of \\(c_t\\)&#8202;=&#8202;{num(c, ',.1f')}&#8202;N&#8202;s/m. "
               f"In task&#160;1 the pad is held at half its slip force, with {lst([r['pred_n_contacts'] for r in base], 0)} "
               f"spheres touching at {_lst([r['N'] for r in base], '&#8202;N')}. Equal shares give "
               f"{lst([r['pred_v_formula_mm_s'] * 1e3 for r in base], 1)}&#8202;&#181;m/s. With "
               f"{lst([r['pred_n_saturated'] for r in base], 0)} spheres at their cones, the sum gives {lst(cone, 1)}&#8202;&#181;m/s, "
               + ("the measured creep to four digits." if same else f"against {lst(meas, 1)}&#8202;&#181;m/s measured."))
    out.append("</p><p>")
    # the 1/(kappa t_c) law: impratio and the time constant at 1 N
    at1 = lambda model, k: next((r["creep_mm_s"] * 1e3 for r in sel(model, k) if _eq(r["N"], 1.0)), None)  # noqa: E731
    kaps = [k for k in (100.0, 1000.0, 3000.0, 10000.0) if at1("mj_pads1", k) is not None]
    tcs = [(m, _tc_of(next((r.get("rig_spec") or r.get("spec") for r in t1 if r.get("model") == m), ""))) for m in
           ("mj_pads1_tr05", "mj_pads1_tr10") if at1(m, 100.0) is not None]
    if len(kaps) > 1:
        out.append(f"At 1&#8202;N the creep falls as \\(1/\\kappa t_c\\): it is {lst([at1('mj_pads1', k) for k in kaps], 2)}"
                   f"&#8202;&#181;m/s at impratio {lst([k for k in kaps], 0).replace('10000', '10,000')} (task&#160;6).")
        if tcs:
            out.append(f"At \\(t_c\\)&#8202;=&#8202;{_lst([t * 1e3 for _, t in tcs], '&#8202;ms')} and impratio 100 it is "
                       f"{lst([at1(m, 100.0) for m, _ in tcs], 2)}&#8202;&#181;m/s.")
        out.append(f"At the policy trainer&#8217;s impratio of 10 the test bed&#8217;s pad would therefore creep about "
                   f"{at1('mj_pads1', 100.0) * 10 * 1e-3:.2f}&#8202;mm/s.")
    ns = [at1x for at1x in (next((r["creep_mm_s"] * 1e3 for r in sel("mj_pads1", 100.0, n) if _eq(r["N"], 1.0)), None)
                            for n in (3, 10)) if at1x is not None]
    if ns and max(ns) < 0:
        out.append(f"Three or ten <code>noslip_iterations</code> re-solve the friction rows without \\(R_t\\) after the main "
                   f"solve. They turn the creep into a slide of {abs(ns[0]):.2f}&#8202;&#181;m/s against the load.")
    # a finer pad at its own t_c
    p05 = pick(pull, "mj_pads05", N=1.0, dt_ms=1.0)
    if p05:
        out.append(f"A finer pad creeps as fast at equal \\(t_c\\), because \\(nk_s\\) stays near \\(k_w\\) times the patch area. "
                   f"The test bed&#8217;s 0.5&#8202;mm pad runs at \\(t_c\\)&#8202;=&#8202;{_tc_of(p05.get('spec')) * 1e3:g}&#8202;ms "
                   f"(step&#160;7) and creeps {p05['creep_mm_s'] * 1e3:.1f}&#8202;&#181;m/s at 1&#8202;N.")
    # factor over Drake, one per tip and step
    fac = lambda rows, a, b, dt: [(pick(rows, a, N=N, dt_ms=dt) or {}).get("creep_mm_s", 0) /  # noqa: E731
                                  max((pick(rows, b, N=N, dt_ms=dt) or {}).get("creep_mm_s", 0), 1e-15) for N in (0.5, 1.0, 3.0)]
    f1, f5 = fac(pull, "mj_pads1", "drake_hydro", 1.0), fac(pull, "mj_pads1", "drake_hydro", 5.0)
    dk = [pick(pull, "drake_hydro", N=N, dt_ms=1.0) for N in (0.5, 1.0, 3.0)]
    if all(dk) and all(f1):
        out.append("</p><p>")
        out.append(f"Drake&#8217;s SAP solver creeps {dk[0]['creep_mm_s'] * 1e3:.2f}&#8211;{dk[-1]['creep_mm_s'] * 1e3:.2f}"
                   "&#8202;&#181;m/s under the same loads. On the 10.55&#8202;mm sphere at a 1&#8202;ms step the pad therefore creeps "
                   f"{lst([round(x) for x in f1], 0)} times faster than Drake at 0.5, 1 and 3&#8202;N.")
        if all(f5):
            out.append(f"At a 5&#8202;ms step, where Drake creeps more, the factor is {min(f5):.0f}&#8211;{max(f5):.0f}.")
        if tip_pull:
            tb = pick(tip_pull, "mj_tpu2p7_pads1", N=1.0, dt_ms=1.0)
            td = pick(tip_pull, "drake_tpu2p7", N=1.0, dt_ms=1.0)
            if tb and td:
                out.append(f"On the printed TPU block at 1&#8202;N and 1&#8202;ms the factor is "
                           f"{tb['creep_mm_s'] / td['creep_mm_s']:.0f} (Table&#160;3).")
        out.append("A damper moves the tool by the time integral of its load, so a held tool does not come back when the "
                   "load is removed (task&#160;8, Figure&#160;14).")
    return " ".join(out).replace(" </p><p> ", "</p><p>")


def stab_rows():
    return load(os.path.join(BED, "stability.jsonl"))


def stab_summaries(rows, refsafe=False, model=None):
    """Summary rows of bed task 7 (one per model, d0, tc and refsafe setting)."""
    return [r for r in rows if r.get("kind") == "summary" and bool(r.get("refsafe")) == refsafe
            and (model is None or r.get("model") == model)]


def stab_fail_kind(r):
    """'diverged' (non-finite state) or 'ejected' (the tool left the pinch) at the first failed step; None if none failed."""
    ff = r.get("dt_first_fail_ms")
    if ff is None:
        return None
    st = (r.get("status_by_dt") or {}).get(str(ff))
    return "diverged" if st == "failed" else "ejected"


def stab_bound(r):
    """Bound (14), 2 t_c (sqrt(1 + q) - 1) with q = d0 + (1 - d0) / n, for the summary row: n = spheres in contact on one
    pad at the end of the settle (deck 23, pad-step-bound: the spring term of a_ref is as explicit as the damper)."""
    n = r.get("n_L_ref") or 0
    return step_bound(r["tc_ms"], r["d0"], n) if n else None


def step_bound(tc, d0, n):
    """Largest stable step (unit of tc) of n contact rows of one pad moving together, (14)."""
    q = d0 + (1 - d0) / n
    return 2 * tc * (math.sqrt(1 + q) - 1)


def _brackets(r, b):
    held, ff = r.get("dt_max_held_ms") or 0.0, r.get("dt_first_fail_ms")
    return held <= b * 1.0001 and (ff is None or ff > b)


def svg_stab(rows):
    S = [r for r in stab_summaries(rows, model="mj_pads1") if stab_bound(r)]
    if not S:
        return pending("Bed task&#160;7 rows (<code>stability.jsonl</code>) are not written yet.")
    W, H = 980, 430
    n_in = sum(1 for r in S if _brackets(r, stab_bound(r)))
    out = _svg_open(W, H, "Largest stable step against the step bound (14) for the 1 mm pad at three time constants "
                          "and seven impedances; each bar spans the largest step that held and the first that failed, and the "
                          f"diagonal passes through {n_in} of the {len(S)} bars.", fs=16)
    lo, hi = 0.7, 25.0
    tk = (1, 2, 5, 10, 20)
    fx, fy = _panel(out, 80, 46, 620, 300, (lo, hi), (lo, hi), tk, tk,
                    "bound of equation (14), n spheres on one pad (ms), log scale",
                    "physics step (ms), log scale: largest held ●, first failed × or ○", True, True)
    out.append(f'<line x1="{fx(lo):.1f}" y1="{fy(lo):.1f}" x2="{fx(hi):.1f}" y2="{fy(hi):.1f}" style="stroke:var(--c-ref);stroke-dasharray:6 5"/>')
    cols = {5.0: "var(--c-c4)", 10.0: "var(--c-sphere)", 20.0: "var(--c-drake)"}
    for r in S:
        b, held, ff = stab_bound(r), r.get("dt_max_held_ms"), r.get("dt_first_fail_ms")
        point = "point" in str(r.get("model"))
        col = "var(--ink3)" if point else cols.get(round(r["tc_ms"], 1), "var(--ink3)")
        x = fx(b)
        tag = (f"{'point contact' if point else '1 mm pad'}, d0 {r['d0']:g}, tc {r['tc_ms']:g} ms, n {r.get('n_L_ref')}: "
               f"bound {b:.2f} ms, held to {held} ms, first failure {ff} ms ({stab_fail_kind(r) or 'none'})")
        if held and ff:
            out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{fy(held):.1f}" y2="{fy(ff):.1f}" style="stroke:{col};stroke-width:2"/>')
        elif held:
            out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{fy(held):.1f}" y2="{fy(hi):.1f}" style="stroke:{col};stroke-width:2;stroke-dasharray:2 3"/>')
        if held:
            _marker(out, x, fy(held), col, "square" if point else "circle", r=4.2, title=tag)
        if ff:
            if stab_fail_kind(r) == "diverged":
                _marker(out, x, fy(ff), col, "cross", r=4.2, title=tag)
            else:
                _marker(out, x, fy(ff), col, "circle", hollow=True, r=4.2, title=tag)
    out.append("</svg>")
    return "".join(out) + _legend_html([("1 mm pad, time constant 5 ms", "var(--c-c4)", False, "circle"),
                                        ("10 ms", "var(--c-sphere)", False, "circle"),
                                        ("20 ms", "var(--c-drake)", False, "circle"),
                                        ("step equal to the bound", "var(--c-ref)", True, None)])


def stab_stats(rows):
    """Counts and values for the stability text: pads with the clamp off against (14) and against the friction rows'
    damper bound (pred_dt_max_friction_ms, about d0 t_c), point contact, and the pads with MuJoCo's default clamp."""
    pads = [r for r in stab_summaries(rows, model="mj_pads1") if stab_bound(r)]
    if not pads:
        return None
    inside = [r for r in pads if _brackets(r, stab_bound(r))]
    outside = [r for r in pads if r not in inside]
    fric = lambda r: r.get("pred_dt_max_friction_ms")  # noqa: E731
    f_in = [r for r in pads if fric(r) and _brackets(r, fric(r))]
    pt = sorted(stab_summaries(rows, model="mj_point3"), key=lambda r: r["d0"])
    pt_f = [r for r in pt if fric(r) and _brackets(r, fric(r))]
    on = stab_summaries(rows, refsafe=True, model="mj_pads1")
    stiff = [r for r in on if r["d0"] >= 0.5]
    soft = [r for r in on if r["d0"] < 0.5]
    return dict(n=len(pads), inside=inside, outside=outside, f_in=f_in, point=pt, pt_f=pt_f, stiff=stiff, soft=soft,
                d0=sorted({r["d0"] for r in pads}), tc=sorted({r["tc_ms"] for r in pads}))


def _lst(v, unit=""):
    v = [f"{x:g}" for x in v]
    return (", ".join(v[:-1]) + " and " + v[-1] if len(v) > 1 else v[0]) + unit


def stab_text(rows, fig=6, tail=False):
    """Two paragraphs, joined by '</p><p>' for a caller that wraps the text in one <p>: task 7 on the test bed against (14),
    then the friction rows' bound, point contact and MuJoCo's clamp (review of 2026-10-14: one paragraph per bound)."""
    st = stab_stats(rows)
    if not st:
        return ""
    out = [f"Task&#160;7 on the test bed holds the tool against gravity at 1&#8202;N per pad for 1&#8202;s. It varies \\(d_0\\) from "
           f"{st['d0'][0]:g} to {st['d0'][-1]:g} and \\(t_c\\) over {_lst(st['tc'], '&#8202;ms')}. The step runs from 0.25 to "
           "15&#8202;ms, with MuJoCo&#8217;s clamp switched off so that \\(t_c\\) stays as set. For "
           f"{len(st['inside'])} of the {st['n']} pairs, the largest step that held and the first that failed bracket (14) "
           f"(Figure&#160;{fig})."]
    if st["outside"]:
        above = [r for r in st["outside"] if (r.get("dt_max_held_ms") or 0) > stab_bound(r) * 1.0001]
        below = sorted((r for r in st["outside"] if r not in above), key=lambda r: (r["tc_ms"], r["d0"]))
        out.append(f"In {len(above)} of the other {len(st['outside'])} the pinch held at steps above the bound.")
        if below:
            kinds = {stab_fail_kind(r) for r in below}
            how = "the pads ejected the tool" if kinds == {"ejected"} else "the run failed"
            out.append(f"At \\(d_0\\)&#8202;=&#8202;{_lst([r['d0'] for r in below])} with \\(t_c\\)&#8202;=&#8202;"
                       f"{_lst(sorted({r['tc_ms'] for r in below}), '&#8202;ms')}, {how} at "
                       f"{_lst(sorted({r['dt_first_fail_ms'] for r in below}), '&#8202;ms')}, below bounds of "
                       f"{min(stab_bound(r) for r in below):.1f}&#8211;{max(stab_bound(r) for r in below):.1f}&#8202;ms.")
    first, out = out, []
    if st["f_in"]:
        pt = st["point"]
        out.append("In the pinch the tool&#8217;s weight rests on the friction rows, which have no spring. Their damper alone gives "
                   "the bound \\(\\Delta t<t_c\\bigl(d_0+(1-d_0)\\hat\\Lambda/(\\kappa\\lambda_t)\\bigr)\\approx d_0t_c\\), with "
                   "\\(\\lambda_t\\) the largest eigenvalue of the friction rows&#8217; \\(A\\). This bound brackets "
                   f"{len(st['f_in'])} of the {st['n']} pad pairs.")
        if pt and len(st["pt_f"]) == len(pt):
            out.append(f"It also brackets the failures of point contact at {_lst([r['dt_first_fail_ms'] for r in pt], '&#8202;ms')} "
                       f"for \\(d_0\\)&#8202;=&#8202;{_lst([r['d0'] for r in pt])} at \\(t_c\\)&#8202;=&#8202;{pt[0]['tc_ms']:g}&#8202;ms, where "
                       f"(14) gives {step_bound(pt[0]['tc_ms'], pt[0]['d0'], 1):.1f}&#8202;ms for any \\(d_0\\).")
    if st["stiff"] and st["soft"]:
        out.append("With the clamp on, as MuJoCo runs by default, a row whose \\(t_c\\) is below \\(2\\Delta t\\) runs at "
                   "\\(t_c=2\\Delta t\\). The friction rows&#8217; bound then becomes \\(2d_0\\Delta t\\), which exceeds \\(\\Delta t\\) "
                   "once \\(d_0\\ge1/2\\). The pads with \\(d_0\\)&#8202;&#8805;&#8202;0.5 held at every step up to "
                   f"{min(r['dt_max_held_ms'] for r in st['stiff']):g}&#8202;ms at all three \\(t_c\\). Those with \\(d_0\\) of "
                   f"{_lst(sorted({r['d0'] for r in st['soft']}))} failed from {min(r['dt_first_fail_ms'] for r in st['soft']):g} "
                   f"to {max(r['dt_first_fail_ms'] for r in st['soft']):g}&#8202;ms.")
    if tail:
        out.append("Which rows set the largest stable step on the hand is open (Future work).")
    return " ".join(first) + ("</p><p>" + " ".join(out) if out else "")


# ------------------------------------------------------------------------------------------ Figure 7: step sweep

def svg_step(rows):
    if not rows:
        return pending("Codex&#8217;s task_timestep_results.json is missing.")
    W, H = 980, 400
    out = _svg_open(W, H, "Left: the tool angle at the end of the twist sequence stays within a few degrees for the sphere pad "
                          "and Drake from 50 microseconds to 10 milliseconds; the per-contact mapping diverges at every step "
                          "from 0.5 ms. Right: wall time per simulated second.", fs=16)
    series = {"legacy": ("1 mm sphere pad", "var(--c-sphere)", False, "circle"),
              "drake": ("Drake hydroelastic", "var(--c-drake)", False, "circle"),
              "runtime_exact": ("per-contact mapping, runtime, exact Λ", "var(--c-c4)", True, "square"),
              "physical": ("per-contact mapping, fixed", "var(--ink3)", True, "square")}
    fx, fy = _panel(out, 70, 46, 380, 280, (0.04, 13.0), (-72.0, -40.0), (0.05, 0.1, 0.5, 1, 2, 5, 10), (-70, -60, -50, -40),
                    "Physics step (ms), log scale", "(a) Tool angle about the pinch axis at the end (°)", True, False)
    for key, (lab, col, dashed, shape) in series.items():
        rs = sorted([r for r in rows if r["case"]["label"] == key], key=lambda r: r["case"]["physics_dt"])
        ok = [(r["case"]["physics_dt"] * 1e3, r["pinch_angle_end_deg"]) for r in rs if r["status"] == "complete"]
        bad = [r["case"]["physics_dt"] * 1e3 for r in rs if r["status"] != "complete"]
        _path(out, fx, fy, ok, col, dashed=dashed, width=1.6)
        for x, y in ok:
            _marker(out, fx(x), fy(y), col, shape, title=f"{lab}, {x:g} ms: {y:.1f} deg")
        for x in bad:
            _marker(out, fx(x), fy(-41.5), col, "cross", title=f"{lab}, {x:g} ms: diverged")
    out.append(f'<text x="{fx(0.5):.1f}" y="{fy(-43.6):.1f}" style="fill:var(--ink3)">× diverged</text>')
    fx, fy = _panel(out, 560, 46, 380, 280, (0.04, 13.0), (0.003, 100.0), (0.05, 0.1, 0.5, 1, 2, 5, 10), (0.01, 0.1, 1, 10, 100),
                    "Physics step (ms), log scale", "(b) Wall time per simulated second (s)", True, True)
    for key, (lab, col, dashed, shape) in series.items():
        rs = sorted([r for r in rows if r["case"]["label"] == key and r["status"] == "complete"], key=lambda r: r["case"]["physics_dt"])
        pts = [(r["case"]["physics_dt"] * 1e3, (r.get("timing") or {}).get("physics_seconds_per_sim_second")) for r in rs]
        _path(out, fx, fy, pts, col, dashed=dashed, width=1.6)
        for x, y in pts:
            if y:
                _marker(out, fx(x), fy(y), col, shape, title=f"{lab}, {x:g} ms: {y:.3f} s per simulated s")
    out.append("</svg>")
    return "".join(out) + _legend_html([(v[0], v[1], v[2], v[3]) for v in series.values()])


def ifig_step(rows):
    """Figure 8 as an interactive figure: (a) the tool angle at the end of the twist sequence and (b) the wall time per
    simulated second against the physics step, both on a log step axis; a run that diverged is an open marker at the
    top of panel (a)."""
    import interactive_figure as IF
    series_def = {"legacy": ("1 mm sphere pad", "var(--c-sphere)", ""), "drake": ("Drake hydroelastic", "var(--c-drake)", ""),
                  "runtime_exact": ("Per-contact mapping, runtime, exact &#923;", "var(--c-c4)", "7 4"),
                  "physical": ("Per-contact mapping, fixed", "var(--ink3)", "7 4")}
    series, legend = [], []
    for key, (lab, col, dash) in series_def.items():
        col = HEX.get(col, col)
        rs = sorted([r for r in rows if r["case"]["label"] == key], key=lambda r: r["case"]["physics_dt"])
        ok = [r for r in rs if r["status"] == "complete"]
        bad = [r["case"]["physics_dt"] * 1e3 for r in rs if r["status"] != "complete"]
        data = {"angle": {"x": [r["case"]["physics_dt"] * 1e3 for r in ok], "med": [r["pinch_angle_end_deg"] for r in ok]},
                "wall": {"x": [r["case"]["physics_dt"] * 1e3 for r in ok],
                         "med": [(r.get("timing") or {}).get("physics_seconds_per_sim_second") or float("nan") for r in ok]}}
        series.append(IF.Series(key, lab, col, data, dash=dash, mode="both", width=1.6, facets={"model": key}))
        ids = [key]
        if bad:
            series.append(IF.Series(key + "-div", lab + ", diverged", col, {"angle": {"x": bad, "med": [-41.5] * len(bad)}},
                                    mode="markers", open=True, facets={"model": key},
                                    point_labels={"angle": [f"diverged at {x:g} ms" for x in bad]}))
            ids.append(key + "-div")
        legend.append((lab, col, dash, ids))
    legend.append(("Open marker at the top of (a): the run diverged", "#000", "", [s.id for s in series if s.id.endswith("-div")]))
    panels = [IF.Panel("angle", "(a) Tool angle about the pinch axis at the end of the twist (&#176;)", (-72.0, -40.0), band=False,
                       dec=2, height=200),
              IF.Panel("wall", "(b) Wall time per simulated second (s)", band=False, ylog=True, height=200)]
    return IF.Figure("contact-step", None, "Physics step (ms)", panels, series, legend=legend, x_domain=(0.04, 13.0), xlog=True,
                     facets=[{"key": "model", "label": "Model", "values": [(k, v[0]) for k, v in series_def.items()]}])


def step_text(rows):
    if not rows:
        return ""
    g = lambda lab: [r for r in rows if r["case"]["label"] == lab]  # noqa: E731
    leg = [r for r in g("legacy") if r["status"] == "complete"]
    dk = [r for r in g("drake") if r["status"] == "complete"]
    mapped = [r for r in rows if r["case"]["label"] in ("physical", "physical_exact", "runtime_approx", "runtime_exact")]
    mfail = [r for r in mapped if r["status"] != "complete"]
    mok = [r for r in mapped if r["status"] == "complete"]
    cost = lambda rs, dt: next((r["timing"]["physics_seconds_per_sim_second"] for r in rs if abs(r["case"]["physics_dt"] - dt) < 1e-9), None)  # noqa: E731
    a = [r["pinch_angle_end_deg"] for r in leg]
    b = [r["pinch_angle_end_deg"] for r in dk]
    return (f"Codex ran the paper&#8217;s Exp&#160;2 twist sequence with a controller at 50&#8202;Hz and physics steps from "
            f"50&#8202;&#181;s to 10&#8202;ms. Over all six steps the 1&#8202;mm pad ends the sequence at {num(min(a), '.1f')} to "
            f"{num(max(a), '.1f')}&#176; about the pinch axis, and Drake hydroelastic at {num(min(b), '.1f')} to "
            f"{num(max(b), '.1f')}&#176;. Both include the same ratchet toward hanging, which the chain page traced to the pads "
            f"spinning freely. Codex&#8217;s four per-contact mappings diverged in {len(mfail)} of {len(mapped)} runs, at every step "
            f"from 0.5&#8202;ms up. The {len(mok)} that finished, at 50&#8202;&#181;s, ended at "
            f"{num(min(r['pinch_angle_end_deg'] for r in mok), '.0f')} to {num(max(r['pinch_angle_end_deg'] for r in mok), '.0f')}"
            f"&#176;. At equal step the pad costs {cost(dk, 0.001) / cost(leg, 0.001):.0f}&#215; less than Drake at 1&#8202;ms and "
            f"{cost(dk, 0.01) / cost(leg, 0.01):.0f}&#215; less at 10&#8202;ms (Figure&#160;8).")


# ------------------------------------------------------------------------------------------ Table 2: agreement

def metrics(T):
    """(group, label, unit, nd, model -> value) for every bed metric with data."""
    out = []

    def add(group, label, unit, nd, rows, fn, **kw):
        vals = {}
        for k in TABLE_ORDER:
            r = pick(rows, k, **kw)
            if r is not None and r.get("status", "complete") in ("complete", None):
                try:
                    v = fn(r)
                except (KeyError, TypeError, ValueError, ZeroDivisionError):
                    v = None
                if v is not None:
                    vals[k] = v
            elif r is not None:
                vals[k] = r.get("status")
        if any(isinstance(v, (int, float)) for v in vals.values()):
            out.append((group, label, unit, nd, vals))

    pull = T["pull"]
    add("1 pull", "effective \\(\\mu\\) at slip, 1&#8202;N", "", 3, pull, lambda r: r["mu_eff"], N=1.0, dt_ms=1.0)
    add("1 pull", "slip speed, first 50&#8202;ms, 1&#8202;N", "mm/s", 1, pull, lambda r: r["v_slip_mean50_mm_s"], N=1.0, dt_ms=1.0)
    add("1 pull", "creep: sliding speed at half the slip force, 1&#8202;N", "&#181;m/s", 2, pull, lambda r: r["creep_mm_s"] * 1e3, N=1.0, dt_ms=1.0)
    tw = T["twist"]
    add("2 twist", "arm at spin onset, 0.5&#8202;N", "mm", 3, tw, twist_arm, N=0.5, dt_ms=1.0)
    add("2 twist", "arm at spin onset, 3&#8202;N", "mm", 3, tw, twist_arm, N=3.0, dt_ms=1.0)
    add("2 twist", "creep: turning speed at half the onset torque, 1&#8202;N", "&#176;/s", 3, tw,
        lambda r: first(r, "creep_deg_s", "creep_rate_deg_s"), N=1.0, dt_ms=1.0)
    ro = T["roll"]
    add("3 roll", "rolling ratio, 1&#8202;N, 10&#8202;mm/s", "", 3, ro,
        lambda r: first(r, "rolling_ratio", "rho"), N=1.0, dt_ms=1.0, **_roll_kw(ro))
    sh = T["shake"]
    add("4 shake", "drift per cycle, 0.5&#8202;N, 2&#8202;g", "&#181;m", 2, sh,
        lambda r: abs(first(r, "drift_per_cycle_mm", "net_drift_per_cycle_mm")) * 1e3, N=0.5, dt_ms=1.0, **_shake_kw(sh, 2.0))
    br = T["brake"]
    add("5 brake", "swing end angle", "&#176;", 1, br, lambda r: first(r, "phi_end_deg", "phi_end", "end_angle_deg"), dt_ms=1.0)
    return out


def _roll_kw(rows):
    for k in ("v_mm_s", "speed_mm_s", "v_pad_mm_s"):
        if any(k in r for r in rows):
            return {k: 10.0}
    return {}


def _shake_kw(rows, g):
    for k in ("a_pk_g", "apk_g", "a_g"):
        if any(k in r for r in rows):
            return {k: g}
    return {}


def agree_class(v, ref):
    if not isinstance(v, (int, float)):
        return "cell c0"
    if not isinstance(ref, (int, float)) or ref == 0:
        return "cell"
    d = abs(v / ref - 1)
    return "cell c4" if d < 0.10 else "cell c3" if d < 0.25 else "cell c2" if d < 0.5 else "cell c0"


SHORT_TH = {"mj_point3": "point", "mj_point4s": "condim 4", "mj_pads1": "1 mm pad, MuJoCo", "mjw_pads1": "1 mm pad, MuJoCo-Warp",
            "newton_pads1": "1 mm pad, Newton", "drake_hydro": "Drake hydro&#173;elastic",
            "newton_hydro_mc": "Newton hydro&#173;elastic, mass-corrected"}
TASK_NAME = {"1 pull": "Task 1, pull to slip", "2 twist": "Task 2, twist to spin", "3 roll": "Task 3, roll between the pads",
             "4 shake": "Task 4, shake while held", "5 brake": "Task 5, brake to hanging"}


def agree_table(M):
    if not M:
        return pending("No bed rows yet.")
    head = "".join(f"<th class='num'>{SHORT_TH[k]}</th>" for k in TABLE_ORDER)
    out = [f"<table class='agree'><thead><tr><th>metric</th>{head}</tr></thead><tbody>"]
    last = None
    for group, label, unit, nd, vals in M:
        if group != last:
            out.append(f"<tr class='grp'><td colspan='{len(TABLE_ORDER) + 1}'>{TASK_NAME.get(group, group)}</td></tr>")
            last = group
        ref = vals.get("drake_hydro")
        cells = []
        for k in TABLE_ORDER:
            v = vals.get(k)
            if v is None:
                cells.append("<td class='cell'>&#8211;</td>")
            elif isinstance(v, str):
                cells.append(f"<td class='cell c0'>{v}</td>")
            else:
                cls = "cell ref" if k == "drake_hydro" else agree_class(v, ref)
                cells.append(f"<td class='{cls}'>{fmt(v, nd)}</td>")
        u = f" ({unit})" if unit else ""
        out.append(f"<tr><td>{label}{u}</td>{''.join(cells)}</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def deviations(M):
    dev = {k: [] for k in TABLE_ORDER}
    for _, _, _, _, vals in M:
        ref = vals.get("drake_hydro")
        if not isinstance(ref, (int, float)) or ref == 0:
            continue
        for k in TABLE_ORDER:
            v = vals.get(k)
            if isinstance(v, (int, float)) and k != "drake_hydro":
                dev[k].append(abs(v / ref - 1) * 100)
    return dev


# ------------------------------------------------------------------------------------------ cost

def step_cost(pull):
    """Median physics us/step at 1 ms per model from bed task 1: one CPU core for MuJoCo and Drake, one GPU world for
    MuJoCo-Warp and Newton (the bed rig steps the GPU from the host and copies the state back each step)."""
    c = {}
    for k in TABLE_ORDER:
        v = [r["us_per_step_median"] for r in pull if r.get("model") == k and _eq(r.get("dt_ms"), 1.0) and r.get("us_per_step_median")]
        if v:
            c[k] = statistics.median(v)
    return c


def gpu_curves(pads, newton):
    """World-steps per second by batch for each GPU series of Figure 10: the median over repeats of complete rows.
    Keys: MuJoCo-Warp pads ('legacy', 'compiled', 0.5 mm) and the Newton rows of the twist fixture by model."""
    out = {}
    for r in pads:
        key = r.get("key", "")
        m = re.match(r"(legacy|compiled)_s([0-9.]+)_dt([0-9.]+)_n(\d+)_", key)
        if not m or r.get("status") != "complete":
            continue
        name = {("legacy", "1.0", "0.001"): "mjw_pads1", ("compiled", "1.0", "0.001"): "mjw_pads1_compiled",
                ("legacy", "0.5", "0.001"): "mjw_pads05"}.get((m.group(1), m.group(2), m.group(3)))
        if name:
            out.setdefault(name, {}).setdefault(int(m.group(4)), []).append(r["world_steps_per_s"])
    for r in newton:
        if r.get("kind") != "timing" or r.get("status") != "complete" or r.get("fixture_name") != "twist" or r.get("contention"):
            continue
        out.setdefault(r["model"], {}).setdefault(int(r["nworld"]), []).append(r["world_steps_per_s"])
    return {k: {n: statistics.median(v) for n, v in sorted(d.items())} for k, d in out.items()}


def gpu_one_world(curves):
    """Wall time (us) of one world alone on the GPU per model, from the 1-world rows of the Figure 10 sweep: the SR2
    holding fixture with each 10-step block captured as a CUDA graph, in MuJoCo-Warp and Newton alike. The bed rig's
    own GPU timing is not comparable across simulators: its Newton rig steps without a CUDA graph."""
    return {k: 1e6 / curves[k][1] for k in GPU_MODELS if 1 in curves.get(k, {})}


def gpu_batched(curves):
    """Per-world-step cost (us) at the batch with the highest throughput, per series: {key: (us, nworld)}."""
    out = {}
    for k, d in curves.items():
        n = max(d, key=d.get)
        out[k] = (1e6 / d[n], n)
    return out


def spread(ys, gap):
    """Positions in the order of ys with neighbours at least `gap` apart, each group centred on its members' mean."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    groups = []
    for i in order:
        groups.append([i])
        while len(groups) > 1:
            g0, g1 = groups[-2], groups[-1]
            c0 = sum(ys[j] for j in g0) / len(g0)
            c1 = sum(ys[j] for j in g1) / len(g1)
            if (c1 - (len(g1) - 1) * gap / 2) - (c0 + (len(g0) - 1) * gap / 2) >= gap:
                break
            groups[-2:] = [g0 + g1]
    out = [0.0] * len(ys)
    for g in groups:
        c = sum(ys[j] for j in g) / len(g)
        for r, j in enumerate(g):
            out[j] = c + (r - (len(g) - 1) / 2) * gap
    return out


def svg_cost(M, cost, batched=None):
    dev = deviations(M)
    batched = batched or {}
    pts = [(cost[k], max(statistics.median(dev[k]), 0.1), k) for k in TABLE_ORDER if k in cost and dev.get(k)]
    if not pts:
        return pending("Needs task rows with Drake references and step costs.")
    W, H = 980, 400
    out = _svg_open(W, H, "Median deviation from Drake over the bed metrics against the cost of a physics step: one CPU core for "
                          "MuJoCo and Drake; for MuJoCo-Warp and Newton one GPU world (filled) joined to the batched cost per "
                          "world-step (hollow).", fs=14)
    ytop = 10 ** math.ceil(math.log10(max(p[1] for p in pts) * 1.5))
    yt = [v for v in (1, 10, 100, 1000) if v <= ytop]
    fx, fy = _panel(out, 80, 46, 540, 280, (0.5, 5000.0), (1.0, ytop), (1, 10, 100, 1000), yt,
                    "cost of one physics step at 1 ms (µs), log scale",
                    "median |deviation from Drake| over the metrics of Table 2 (%), log scale", True, True, yfmt="{:g}")
    if "drake_hydro" in cost:
        x = fx(cost["drake_hydro"])
        out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="46" y2="326" style="stroke:var(--c-drake);stroke-dasharray:6 5"/>'
                   f'<text x="{x - 8:.1f}" y="62" text-anchor="end" style="fill:var(--c-drake)">Drake hydroelastic, '
                   f'{cost["drake_hydro"]:.0f} µs (the reference)</text>')
    # models closer than one label height in deviation are spread apart in their order, so that every line shows
    ys = spread([fy(d) for _, d, _ in pts], 19.0)
    xr = fx(5000.0) + 16
    for (c, d, k), y in zip(pts, ys):
        lab, col, shape, _ = MODELS[k]
        xs = [fx(c)]
        if k in batched:
            b, n = batched[k]
            xs.append(fx(b))
            out.append(f'<line x1="{fx(b):.1f}" x2="{fx(c):.1f}" y1="{y:.1f}" y2="{y:.1f}" style="stroke:{col};stroke-width:1.6"/>')
            _marker(out, fx(b), y, col, shape, hollow=True, r=5.5,
                    title=f"{lab}: {b:.2f} us per world-step batched at {n} worlds, median deviation {d:.1f} %")
        out.append(f'<line x1="{fx(c) + 8:.1f}" x2="{xr - 4:.1f}" y1="{y:.1f}" y2="{y:.1f}" '
                   f'style="stroke:{col};stroke-width:1;stroke-dasharray:1 4;opacity:.7"/>')
        _marker(out, fx(c), y, col, shape, r=5.5,
                title=f"{lab}: {c:.1f} us per step ({'one GPU world' if k in GPU_MODELS else 'one CPU core'}), "
                      f"median deviation {d:.1f} %")
        short = lab.replace(', torsion rescheduled', ', rescheduled').replace(', mass-corrected', '')
        out.append(f'<text x="{xr:.1f}" y="{y + 4:.1f}" style="fill:{col}">{short}, {d:.1f} %</text>')
    out.append("</svg>")
    return "".join(out) + _legend_html([("filled: one CPU core (MuJoCo, Drake) or one GPU world (MuJoCo-Warp, Newton)", "var(--ink2)", False, "circle"),
                                        ("&#9675; hollow: per world-step in a GPU batch", "var(--ink2)", False, None)])


def _si(v):
    return f"{v / 1e6:g}M" if v >= 1e6 else f"{v / 1e3:g}k" if v >= 1e3 else f"{v:g}"


GPU_SERIES = [   # key, label, colour, dashed, marker
    ("mjw_pads1", "MuJoCo-Warp, 1 mm sphere pad", "var(--c-sphere)", False, "square"),
    ("mjw_pads1_compiled", "MuJoCo-Warp, 1 mm pad, Codex mapping", "var(--c-sphere)", True, "square"),
    ("mjw_pads05", "MuJoCo-Warp, 0.5 mm sphere pad", "var(--ink3)", False, "square"),
    ("newton_pads1", "Newton, 1 mm sphere pad", "var(--c-sphere)", False, "diamond"),
    ("newton_hydro_mc", "Newton hydroelastic, mass-corrected, contact reduction", "var(--c-newton)", False, "diamond"),
    ("newton_hydro_unreduced_mc", "Newton hydroelastic, mass-corrected, all faces", "var(--c-newton)", True, "diamond"),
]


HEX = {"var(--c-sphere)": "#BE7514", "var(--c-drake)": "#4A7FC4", "var(--c-ref)": "#79858E", "var(--c-newton)": "#7A4FA6",
       "var(--c-c4)": "#25875A", "var(--ink3)": "#444444", "var(--ink2)": "#000000"}   # the plain style's light values


def ifig_gpu(curves):
    """Figure 10 as an interactive figure (scripts/interactive_figure.py, 2026-10-11): the same series as svg_gpu, log
    axes, the cost per world-step in the readout. Two series that share a colour and dash on the static chart differ there
    by marker shape; here the Newton pad series is dotted."""
    import interactive_figure as IF
    dash = {"newton_pads1": "2 3"}
    series = []
    for key, lab, col, dashed, shape in GPU_SERIES:
        d = curves.get(key)
        if not d:
            continue
        pts = sorted(d.items())
        series.append(IF.Series(key, lab, HEX.get(col, col), {"gpu": {"x": [n for n, _ in pts], "med": [v for _, v in pts]}},
                                dash=dash.get(key, "7 4" if dashed else ""), mode="both", width=1.8, facets={"series": key},
                                point_labels={"gpu": [f"{n} worlds, {1e6 / v:.2f} &#181;s per world-step" for n, v in pts]}))
    panel = IF.Panel("gpu", "World-steps per second", band=False, ylog=True, height=300, x_label="Parallel worlds", xlog=True)
    return IF.Figure("contact-gpu", None, "Parallel worlds", [panel], series,
                     facets=[{"key": "series", "label": "Series", "values": [(x.id, x.label) for x in series]}])


def svg_gpu(curves):
    if not curves:
        return pending("GPU rows missing.")
    W, H = 980, 420
    out = _svg_open(W, H, "World-steps per second against the number of parallel worlds on one GPU, for the 1 mm sphere pads "
                          "in MuJoCo-Warp and Newton and for Newton's hydroelastic tip.", fs=16)
    ys = [v for d in curves.values() for v in d.values()]
    lo, hi = max(min(ys) / 2, 100.0), max(ys) * 2
    fx, fy = _panel(out, 90, 46, 600, 300, (0.8, 12000.0), (lo, hi), (1, 10, 100, 1000, 10000),
                    [v for v in (1e3, 1e4, 1e5, 1e6) if lo <= v <= hi],
                    "parallel worlds, log scale", "world-steps per second, log scale", True, True, yfmt=_si)
    series = []
    for key, lab, col, dashed, shape in GPU_SERIES:
        d = curves.get(key)
        if not d:
            continue
        pts = sorted(d.items())
        _path(out, fx, fy, pts, col, dashed=dashed, width=1.8)
        for n, v in pts:
            _marker(out, fx(n), fy(v), col, shape, title=f"{lab}: {n} worlds, {v / 1e3:.1f}k world-steps/s, {1e6 / v:.2f} us per world-step")
        series.append((lab, col, dashed, shape))
    out.append("</svg>")
    return "".join(out) + _legend_html(series)


# ------------------------------------------------------------------------------------------ tables and text

def cap_table(cost, batched=None, skin=None, hand=None, creep_x=None):
    """skin: {model: (CPU us per step in task 8, GPU multiple of the pads)} for the presliding rows (step 12).
    hand: {model: CPU us per step on the real_v1 hand's turn replay}; creep_x: (sphere, block) pad/Drake creep factors."""
    batched = batched or {}
    skin = skin or {}
    hand = hand or {}

    def sk(m):
        c, g = skin.get(m, (None, None))
        return (f"{num(c, ',.0f')} CPU core, task&#160;8<br>{g:.1f}&#215; the pads batched" if c and g else "&#8211;")

    def us(k, gpu=None):
        """Cost cell: one CPU core on the bed and on the hand, or one GPU world and the batched cost per world-step
        (gpu = the GPU model key)."""
        parts = []
        if cost.get(k):
            if k in GPU_MODELS:
                parts.append(f"{num(cost[k], ',.0f')} one GPU world")
            else:
                h = hand.get(k)
                parts.append(f"{num(cost[k], ',.0f')} bed" + (f"<br>{num(h, ',.0f')} hand" if h else ""))
        g = gpu or (k if k in GPU_MODELS else None)
        if g and cost.get(g) and g != k:
            parts.append(f"{num(cost[g], ',.0f')} one GPU world")
        if g and g in batched:
            b, n = batched[g]
            parts.append(f"{b:.2f} batched ({num(n, ',d')} worlds)")
        return "<br>".join(parts) or "&#8211;"
    Y, N_, P = "cap-y", "cap-n", "cap-p"
    rows = [
        ("Point contact, condim&#160;3", [(Y, "One spring"), (P, "Creeps"), (N_, "None"), (N_, "One point"), (N_, "No"), (N_, "No"), us("mj_point3"), (Y, "MuJoCo-Warp")]),
        ("Condim&#160;4, \\(\\mu_t\\) rescheduled", [(Y, "One spring"), (P, "Creeps"), (P, "Fitted law, rescheduled"), (N_, "One point"), (N_, "No"), (N_, "No"), us("mj_point4s"), (P, "Needs a per-step write")]),
        ("1&#8202;mm sphere pad", [(Y, "Sampled foundation"), (P, "Creeps"), (Y, "From the sphere spread"), (Y, "Sampled"), (N_, "No"), (N_, "No"), us("mj_pads1", "mjw_pads1"), (Y, "MuJoCo-Warp, Newton")]),
        ("1&#8202;mm pads on a sprung skin", [(Y, "Sampled foundation on springs"), (P, "Presliding, then creeps"), (Y, "From the sphere spread"),
                                          (Y, "Sampled"), (P, "At impratio 1000; creep swamps it at 10"), (N_, "No"), sk("mj_pads1_skin"), (Y, "MuJoCo-Warp")]),
        ("1&#8202;mm pads on rolling bristles", [(Y, "Sampled foundation"), (P, "Partial slip, then creeps"), (Y, "From the sphere spread"),
                                             (Y, "Sampled"), (Y, "With hysteresis"), (N_, "No"), sk("mj_pads1_bristle20a"), (Y, "MuJoCo-Warp")]),
        ("Drake hydroelastic", [(Y, "Pressure field"), (Y, (f"Creeps {creep_x[0]:.0f}&#215; slower than the pads (sphere), "
                                                            f"{creep_x[1]:.0f}&#215; (block)" if creep_x else "Creeps")),
                                (Y, "Integrated"), (Y, "Surface mesh"), (N_, "No"), (N_, "No"), us("drake_hydro"), (N_, "CPU only")]),
        ("Newton hydroelastic, \\(k_h/m_\\text{eff}\\)", [(Y, "Pressure field"), (P, "MuJoCo-Warp rows"), (P, "0.76&#8211;0.88 of Drake"), (Y, "Voxel surface"), (N_, "No"), (N_, "No"),
                                 us("newton_hydro_mc"), (Y, "Warp")]),
        ("Sphere lattice, CSLC", [(Y, "Anchor springs"), (P, "Presliding; no sliding dynamics"), (Y, "From the lattice"), (Y, "Lattice"), (P, "Presliding"), (Y, "Lateral springs"), "&#8211;", (P, "&#8211;")]),
    ]
    head = ["Contact model", "Compliance", "Stick, slip", "Friction torque", "Area, CoP",
            "Pre-slip shear", "Lateral spread", f"Cost ({US} per step)", "GPU batch"]
    out = ["<table class='cap'><thead><tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr></thead><tbody>"]
    for name, cells in rows:
        tds = []
        for c in cells:
            if isinstance(c, tuple):
                tds.append(f"<td class='{c[0]}'>{c[1]}</td>")
            else:
                tds.append(f"<td class='num'>{c}</td>")
        out.append(f"<tr><td>{name}</td>{''.join(tds)}</tr>")
    out.append("</tbody></table>")
    return "".join(out)


# ------------------------------------------------------------------------------------------ the printed TPU block
# (2026-10-14: the default fingertip since 2026-10-13 is the printed TPU block as 1 mm pads; its bed rows are the
# 2026-10-06 backends study's tasks 1, 2 and 5 on the block with 2.7 mm fillets, scripts/tpu_tip_rig.py)

TIP_D = os.path.join(EXP, "20261006-fingertip_backends")
TIP_PATH = "docs/experiments/20261006-fingertip_backends/20261006-fingertip_contact_backends.html"
BLOCK = ("mj_tpu2p7_pads1", "drake_tpu2p7")          # 1 mm pads and Drake hydroelastic on the 2.7 mm-fillet block


def tip_rows():
    """Bed tasks 1 (pull), 2 (twist) and 5 (brake) on the printed fingertips, keyed T1, T2, T5."""
    return {t: load(os.path.join(TIP_D, f"tip_{t}.jsonl")) for t in ("T1", "T2", "T5")}


def block_metrics(B):
    """Table 3: (label, unit, decimals, pads, Drake) on the 2.7 mm block, 1 ms step unless named."""
    g = lambda t, k, **kw: pick(B[t], k, **kw) or {}  # noqa: E731
    out = []
    for N in (1.0, 3.0):
        out.append((f"Arm at spin onset, {N:g}&#8202;N", "mm", 3,
                    *(g("T2", k, N=N, dt_ms=1.0).get("rbar_onset_mm") for k in BLOCK)))
    for N in (0.5, 1.0, 3.0):
        out.append((f"Arm spinning at 1&#8202;rad/s, {N:g}&#8202;N", "mm", 2,
                    *(g("T2", k, N=N, dt_ms=1.0).get("rbar_kin_mm") for k in BLOCK)))
    out.append(("Effective \\(\\mu\\) at slip, 1&#8202;N", "", 3, *(g("T1", k, N=1.0, dt_ms=1.0).get("mu_eff") for k in BLOCK)))
    out.append(("Brake force at 45&#176;", "N", 3, *(g("T5", k, dt_ms=1.0).get("N_at_45_N") for k in BLOCK)))
    for dt in (1.0, 5.0):
        out.append((f"Swing end, {dt:g}&#8202;ms step", "&#176;", 1, *(g("T5", k, dt_ms=dt).get("phi_end_deg") for k in BLOCK)))
    c = [g("T1", k, N=1.0, dt_ms=1.0).get("creep_mm_s") for k in BLOCK]
    out.append(("Creep at half the slip force, 1&#8202;N", "&#181;m/s", 2, *(x * 1e3 if x is not None else None for x in c)))
    out.append(("Twist creep at half the onset torque, 1&#8202;N", "&#176;/s", 3,
                *(g("T2", k, N=1.0, dt_ms=1.0).get("creep_deg_s") for k in BLOCK)))
    out.append((f"Cost of a 1&#8202;ms step, task&#160;1 at 1&#8202;N, one CPU core", US, 0,
                *(g("T1", k, N=1.0, dt_ms=1.0).get("us_per_step_median") for k in BLOCK)))
    return out


def block_table(B):
    M = block_metrics(B)
    if not any(p is not None for _, _, _, p, _ in M):
        return pending("The block rows (tip_T1, tip_T2, tip_T5.jsonl) are missing.")
    out = ["<table class='agree'><thead><tr><th>Metric</th><th class='num'>1&#8202;mm pads, MuJoCo</th>"
           "<th class='num'>Drake hydro&#173;elastic</th><th class='num'>Pads / Drake</th></tr></thead><tbody>"]
    for lab, unit, nd, p, d in M:
        r = f"{p / d:.2f}" if isinstance(p, (int, float)) and isinstance(d, (int, float)) and d else "&#8211;"
        u = f" ({unit})" if unit else ""
        cell = lambda x: fmt(x, nd) if nd or x is None else num(x, ",.0f")  # noqa: E731
        cls = "num" if unit == US else agree_class(p, d)          # cost is not shaded for agreement
        out.append(f"<tr><td>{lab}{u}</td><td class='{cls}'>{cell(p)}</td>"
                   f"<td class='{'num' if unit == US else 'cell ref'}'>{cell(d)}</td><td class='num'>{r}</td></tr>")
    out.append("</tbody></table>")
    return "".join(out)


# ------------------------------------------------------------------------------------------ cost on the hand and in RL

def hand_cost():
    """CPU us per 1 ms step on the real_v1 hand: the open-loop turn replay of D1-D8 on the working plant (TPU block as 1 mm
    pads, as one convex mesh with point contact, or Drake hydroelastic; median over the placements Drake also ran, paired
    by hand and seed, so the three medians cover one set) and the scale-study planner's three-finger grip of a 16 mm
    cylinder on the servo plant (real_v1 and the CAD hand)."""
    out = {}
    rows = {key: load(path) for key, path in (("mj_pads1", os.path.join(HT3, "plans_mujoco.jsonl")),
                                              ("mj_point3", os.path.join(SAD, "plans_mujoco_pt.jsonl")),
                                              ("drake_hydro", os.path.join(HT3, "plans_drake.jsonl")))}
    place = lambda r: (r.get("hand"), r.get("seed"))  # noqa: E731
    paired = {place(r) for r in rows["drake_hydro"] if r.get("us_per_step_median")}
    for key, rs in rows.items():
        v = [r["us_per_step_median"] for r in rs if r.get("us_per_step_median") and (not paired or place(r) in paired)]
        if v:
            out[key] = statistics.median(v)
            if key == "mj_pads1":
                out["n_pads"] = sum((rs[0].get("n_pads") or {}).values())
                out["d0"] = sorted({round(x, 3) for r in rs for x in (r.get("pad_d0") or {}).values()})
    out["n_paired"] = len(paired)
    for name, d in (("real_v1", "20261013-scale_turn_controllers"), ("cad", "20261014-scale_turn_cad_v1")):
        p = os.path.join(EXP, d, "planners/timing.json")
        if os.path.exists(p):
            m = json.load(open(p))["models"]
            out[name] = {k: m[k]["grip"]["step_us"] for k in ("pads", "pt") if k in m}
    return out


def rl_cost():
    """PPO iteration (s) and GPU memory (GB) of the D6 reorientation trainer, TPU mesh against 1 mm pads, from the
    run_costs.jsonl of the servo-plant (2026-10-11, whole-GPU peak) and working-plant (2026-10-08, training process)
    studies; the working plant's idle-GPU five-iteration probe from its page builder."""
    out = {}
    for plant, d, memkey in (("servo", "20261011-servo_plant_policies", "gpu_mem_peak_mb"),
                             ("work", "20261008-contact_model_policies", "gpu_mem_proc_mb")):
        rows = load(os.path.join(EXP, d, "run_costs.jsonl"))
        for arm in ("tpu27mesh", "tpu27pads1"):
            rs = [r for r in rows if r.get("arm") == arm and r.get("s_per_it_median")]
            mem = [r[memkey] for r in rs if r.get(memkey)]
            if rs:
                out[(plant, arm)] = (statistics.median(r["s_per_it_median"] for r in rs),
                                     statistics.median(mem) / 1e3 if mem else None)
    try:
        import contact_model_policies_page as CMP
        out["probe"] = {a: CMP.probe_spit(a) for a in ("tpu27mesh", "tpu27pads1")}
    except Exception:            # the probe logs are gitignored; the table then leaves the idle column out
        out["probe"] = {}
    return out


def state_cost():
    """One held D6 RL state stepped on the GPU (2026-10-06, same_state_timing.jsonl): {(engine, MuJoCo-Warp version,
    fingertip): {nworld: us per world-step}} and the fresh-process GPU memory (GB) at 1,024 worlds."""
    us, mem = {}, {}
    for r in load(os.path.join(RLD, "same_state_timing.jsonl")):
        if r.get("status") != "ok":
            continue
        k = (r["engine"], str(r.get("mujoco_warp"))[:4].rstrip("."), r["variant"])
        us.setdefault(k, {})[r["nworld"]] = r["us_per_world_step"]
        if r["nworld"] == 1024 and r.get("gpu_used_mb") and r.get("gpu_before_mb") is not None:
            mem[k] = (r["gpu_used_mb"] - r["gpu_before_mb"]) / 1e3
    return us, mem


def cost_table(cost, batched, hand, rl, state, B):
    """Table 4: cost of one physics step on the bed and on the hand (one CPU core), per world-step on the GPU, and of a
    PPO iteration, for point contact, the 1 mm pads, Drake and Newton hydroelastic."""
    f = lambda x, fm=",.1f": num(x, fm) if isinstance(x, (int, float)) else "&#8211;"  # noqa: E731
    pair = lambda d, fm=".2f": (f"{d[1024]:{fm}}, {d[2048]:{fm}}" if d and 1024 in d and 2048 in d  # noqa: E731
                                else (f"{d[1024]:{fm}}" if d and 1024 in d else "&#8211;"))
    us, mem = state
    blk = {k: (pick(B["T1"], k, N=1.0, dt_ms=1.0) or {}).get("us_per_step_median") for k in BLOCK}
    rv, cad = hand.get("real_v1", {}), hand.get("cad", {})
    sv = lambda arm, i: (rl.get(("servo", arm)) or (None, None))[i]  # noqa: E731
    wk = lambda arm, i: (rl.get(("work", arm)) or (None, None))[i]  # noqa: E731
    pr = rl.get("probe") or {}
    two = lambda a, b, fm: " / ".join(f(x, fm) for x in (a, b) if isinstance(x, (int, float))) or "&#8211;"  # noqa: E731
    bt = batched or {}
    b0 = lambda k: (bt.get(k) or (None,))[0]  # noqa: E731
    rows = [
        ("Bed, 10.55&#8202;mm sphere, task&#160;1, median over 0.5, 1 and 3&#8202;N",
         f(cost.get("mj_point3")), f(cost.get("mj_pads1")), f(cost.get("drake_hydro"), ",.0f"), "&#8211;"),
        ("Bed, printed TPU block, task&#160;1 at 1&#8202;N", "&#8211;", f(blk[BLOCK[0]]), f(blk[BLOCK[1]], ",.0f"), "&#8211;"),
        (f"real_v1 hand, open-loop turn replay D1&#8211;D8, median over the {hand.get('n_paired', 0)} placements that Drake also ran",
         f(hand.get("mj_point3")), f(hand.get("mj_pads1")), f(hand.get("drake_hydro"), ",.0f"), "&#8211;"),
        ("real_v1 hand, three-finger grip in the planner", f(rv.get("pt")), f(rv.get("pads")), "&#8211;", "&#8211;"),
        ("CAD hand, the same grip", f(cad.get("pt")), f(cad.get("pads")), "&#8211;", "&#8211;"),
        ("SR2 holding fixture, best batch, MuJoCo-Warp (GPU)", "&#8211;", f(b0("mjw_pads1"), ".2f"), "&#8211;", "&#8211;"),
        ("The same fixture in Newton; hydroelastic reduced / all faces (GPU)", "&#8211;", f(b0("newton_pads1"), ".2f"), "&#8211;",
         two(b0("newton_hydro_mc"), b0("newton_hydro_unreduced_mc"), ".2f")),
        ("D6 RL held state, 1,024 and 2,048 worlds, MuJoCo-Warp 3.6, the trainer&#8217;s build (GPU)",
         pair(us.get(("mjw", "3.6", "mesh"))), pair(us.get(("mjw", "3.6", "pads1"))), "&#8211;", "&#8211;"),
        ("The same state, MuJoCo-Warp 3.14 (GPU)", pair(us.get(("mjw", "3.14", "mesh"))),
         pair(us.get(("mjw", "3.14", "pads1"))), "&#8211;", "&#8211;"),
        ("The same state in Newton&#8217;s pipeline on MuJoCo-Warp 3.14 (GPU)", pair(us.get(("nt_pt", "3.14", "mesh"))),
         pair(us.get(("nt_pt", "3.14", "pads1"))), "&#8211;", pair(us.get(("nt_hydro", "3.14", "mesh")))),
        ("GPU memory of that state at 1,024 worlds, MuJoCo-Warp 3.6 / 3.14 / Newton (GB)",
         " / ".join(f(mem.get(k), ".1f") for k in (("mjw", "3.6", "mesh"), ("mjw", "3.14", "mesh"), ("nt_pt", "3.14", "mesh"))),
         " / ".join(f(mem.get(k), ".1f") for k in (("mjw", "3.6", "pads1"), ("mjw", "3.14", "pads1"), ("nt_pt", "3.14", "pads1"))),
         "&#8211;", f(mem.get(("nt_hydro", "3.14", "mesh")))),
        ("PPO iteration, servo-limited plant (s)", f(sv("tpu27mesh", 0), ".2f"), f(sv("tpu27pads1", 0), ".2f"), "&#8211;", "&#8211;"),
        ("PPO iteration, working plant, idle GPU / during the runs (s)", two(pr.get("tpu27mesh"), wk("tpu27mesh", 0), ".2f"),
         two(pr.get("tpu27pads1"), wk("tpu27pads1", 0), ".2f"), "&#8211;", "&#8211;"),
        ("GPU memory in training, working plant process / servo plant whole GPU (GB)",
         two(wk("tpu27mesh", 1), sv("tpu27mesh", 1), ".1f"), two(wk("tpu27pads1", 1), sv("tpu27pads1", 1), ".1f"),
         "&#8211;", "&#8211;"),
    ]
    head = ["Measurement (&#181;s per step unless named)", "Point contact", "1&#8202;mm pads", "Drake hydro&#173;elastic",
            "Newton hydro&#173;elastic"]
    out = ["<table class='cap cost'><thead><tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr></thead><tbody>"]
    for r in rows:
        out.append(f"<tr><td>{r[0]}</td>" + "".join(f"<td class='num'>{c}</td>" for c in r[1:]) + "</tr>")
    out.append("</tbody></table>")
    return "".join(out)


# ------------------------------------------------------------------------------------------ derivation numbers
# (owner, 2026-10-14: the overview skipped mathematical steps that deck 23 of the contact tutorial fills; the numbers of
# those steps are computed here from the constants, the bed's rows and the calibration rows)

E_FIELD = 1e7                      # Drake's default modulus, the pads' E (Pa)
NU_HERTZ = 0.45                    # Poisson's ratio taken for TPU
BED_CAP_DEG = 35.0                 # the bed's pads (scripts/hom_contact_rig.py, cap_deg default)
G_ACC = 9.81


def cap_pad(spacing, cap_deg):
    """(sphere count, per-sphere stiffness in N/m) of a pad of `spacing` on a cap of half-angle `cap_deg` of the 10.55 mm
    fingertip, (8)."""
    A = 2 * math.pi * G.R ** 2 * (1 - math.cos(math.radians(cap_deg)))
    n = max(1, int(round(A / spacing ** 2)))
    return n, E_FIELD / G.R * (A / n) * ((G.R - G.RS) / G.R) ** 2


def flat_pad(spacing, cap_deg, F, tilt_deg=0.0):
    """A pad pressed with F (N) on a flat plate, every touching sphere an ideal spring of (8) (deck 23, pad-sampler): the
    lattice is tilted by `tilt_deg` about x before pressing, which moves the contact over the cap. Returns (spheres
    touching, arm in m about the pressing axis)."""
    n, K = cap_pad(spacing, cap_deg)
    dirs = G.fib_cap(n, math.radians(cap_deg))
    if tilt_deg:
        c, s = math.cos(math.radians(tilt_deg)), math.sin(math.radians(tilt_deg))
        dirs = dirs @ np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]).T
    z = (G.R - G.RS) * dirs[:, 2]
    rho = (G.R - G.RS) * np.hypot(dirs[:, 0], dirs[:, 1])
    force = lambda d0: K * np.clip(G.RS - ((G.R - d0) - z), 0.0, None)  # noqa: E731
    lo, hi = 0.0, 2e-3
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if force(mid).sum() < F else (lo, mid)
    fo = force(0.5 * (lo + hi))
    return int((fo > 0).sum()), float((fo * rho).sum() / fo.sum())


def _pct_span(v):
    """'from 46 % below to 4 % above' / 'from 2.7 to 1.3 % below' for signed % deviations."""
    q = lambda x: f"{abs(x):.0f}" if abs(x) >= 10 else f"{abs(x):.1f}"  # noqa: E731
    lo, hi = min(v), max(v)
    if hi < 0:
        return f"from {q(lo)} to {q(hi)}&#8202;% below"
    if lo > 0:
        return f"from {q(lo)} to {q(hi)}&#8202;% above"
    return f"from {q(lo)}&#8202;% below to {q(hi)}&#8202;% above"


def deck_values(T, creep, cal, hand=None, lo=None):
    """Placeholders of derivation steps 1-9 (deck 23 of the contact tutorial) computed from the constants and rows.
    lo: hom_chain_figures.compute(0.5), the chain's thumb pads of Figure 4 at 0.5 N."""
    v = {}
    kw = E_FIELD / G.R
    # step 1: the hinge balance on the bed's brake (task 5)
    brk = pick([r for r in T["brake"] if r.get("role", "bed") == "bed"], "drake_hydro", dt_ms=1.0) or {}
    d, N1 = brk.get("d_mm", 15.0) * 1e-3, brk.get("N1", 0.2)
    mgd = M_TOOL * G_ACC * d
    capf = lambda F: 2 * MU * F * C_LAW * F ** EXP_LAW  # noqa: E731
    phi = lambda F: math.degrees(math.acos(min(1.0, capf(F) / mgd)))  # noqa: E731
    v.update(HINGE_D=f"{d * 1e3:g}", HINGE_MGD=f"{mgd * 1e3:.2f}", HINGE_ARM1=f"{C_LAW * 1e3:.3f}",
             HINGE_CAP1=f"{capf(1.0) * 1e3:.2f}", HINGE_PHI1=f"{phi(1.0):.0f}",
             HINGE_FH=f"{(mgd / (2 * MU * C_LAW)) ** (1 / (1 + EXP_LAW)):.2f}", HINGE_N1=f"{N1:g}",
             HINGE_CAPN1=f"{capf(N1) * 1e3:.2f}", HINGE_PHIN1=f"{phi(N1):.1f}")
    # step 2: a sphere on the bed at 1 N
    d0 = math.sqrt(1.0 / (math.pi * kw * G.R))
    a_of = lambda F: math.sqrt(2 * G.R * math.sqrt(F / (math.pi * kw * G.R)))  # noqa: E731
    c_flat = 8 / 15 * (4 * G.R / (math.pi * kw)) ** 0.25
    v.update(KW=f"{kw * 1e-9:.3f}", BED_D0_UM=f"{d0 * 1e6:.0f}", BED_A_MM=f"{a_of(1.0) * 1e3:.2f}",
             BED_P0=f"{kw * d0 * 1e-6:.3f}", BED_KC=f"{2 * math.pi * kw * G.R * d0 * 1e-3:.1f}", C_FLAT=f"{c_flat * 1e3:.3f}",
             PATCH_A=f"{a_of(0.5) * 1e3:.1f}&#8211;{a_of(3.0) * 1e3:.1f}")
    # step 3: Hertz with Drake's modulus taken as a Young's modulus; torque ratios of the two-load test
    Es = E_FIELD / (1 - NU_HERTZ ** 2)
    aH = (3 * 1.0 * G.R / (4 * Es)) ** (1 / 3)
    v.update(HZ_ESTAR=f"{Es * 1e-6:.1f}", HZ_A=f"{aH * 1e3:.2f}", HZ_RBAR=f"{3 * math.pi / 16 * aH * 1e3:.2f}",
             T_RATIO_BED=f"{6 ** 1.25:.2f}", T_RATIO_HZ=f"{6 ** (4 / 3):.2f}", RATIO_015=f"{6 ** 0.15:.2f}")
    # finite elements of the planned prints under the screwdriver: arm at 3 N over the arm at 0.5 N, every recipe
    fe = os.path.join(EXP, "20261009-tpu_tip_fem/summary.json")
    rec = json.load(open(fe)).get("recipes", {}) if os.path.exists(fe) else {}
    fr = [r["arm"]["3.0"] / r["arm"]["0.5"] for r in rec.values() if (r.get("arm") or {}).get("0.5") and r["arm"].get("3.0")]
    v["FE_ARM_RATIO"] = _rng(fr, 2) if fr else "&#8211;"
    # step 5: the bed's pad, the chain's pad, a flat at 1 N, the placement range at 0.5 N, the 2 mm pad on the bed
    nb, kb = cap_pad(0.001, BED_CAP_DEG)
    nc, kc = cap_pad(0.001, math.degrees(G.CAP))
    n_f, arm_f = flat_pad(0.001, BED_CAP_DEG, 1.0)
    shrink = (G.R - G.RS) / G.R
    v.update(PAD_DENS=f"{1 / shrink ** 2:.3f}", PAD_ARMFAC=f"{shrink ** 0.25:.3f}", BED_N=str(nb), BED_CAP=f"{BED_CAP_DEG:g}",
             BED_KS=f"{kb:.2f}", CH_N=str(nc), CH_CAP=f"{math.degrees(G.CAP):g}", CH_KS=f"{kc:.0f}", FLAT_N=str(n_f),
             FLAT_ARM=f"{arm_f * 1e3:.3f}", FLAT_BED=f"{c_flat * 1e3:.3f}", FLAT_PRED=f"{c_flat * shrink ** 0.25 * 1e3:.3f}")
    ref = c_flat * 0.5 ** EXP_LAW
    for key, s in (("PL2", 0.002), ("PL1", 0.001), ("PL05", 0.0005)):
        tilt = [flat_pad(s, BED_CAP_DEG, 0.5, t) for t in np.arange(0.0, 8.001, 0.5)]
        v[key] = _pct_span([(a / ref - 1) * 100 for _, a in tilt])
        v[key + "_N"] = f"{min(n for n, _ in tilt)} or {max(n for n, _ in tilt)}" if s == 0.002 else ""
    p2 = pick(T["twist"], "mj_pads2", N=0.5, dt_ms=1.0) or {}
    law05 = C_LAW * 1e3 * 0.5 ** EXP_LAW
    v.update(P2_ONSET=f"{p2.get('rbar_onset_mm', float('nan')):.3f}", P2_KIN=f"{p2.get('rbar_kin_mm', float('nan')):.3f}",
             P2_ONSET_DEV=f"{(1 - p2.get('rbar_onset_mm', 0) / law05) * 100:.0f}",
             P2_KIN_DEV=f"{(1 - p2.get('rbar_kin_mm', 0) / law05) * 100:.0f}")
    # the chain's 2 mm thumb pad of Figure 4 at rest on the tool at 0.5 N, against the field's arm
    lo = lo or G.compute(0.5)
    dv = (lo[0.002]["arm"] / lo["field"]["arm"] - 1) * 100
    v["P2_FIG4"] = f"{abs(dv):.0f}&#8202;% {'above' if dv > 0 else 'below'}"
    # step 6: the released sphere and the arm against t_c (task 6 part T2 at 1 N, impratio 100)
    x = 4.0
    for _ in range(40):
        x -= ((1 + x) * math.exp(-x) - 0.05) / (-x * math.exp(-x))
    t1 = [r for r in creep if r.get("part") == "T1" and r.get("model") == "mj_pads1" and _eq(r.get("dt_ms"), 1.0)
          and _eq(r.get("impratio"), 100.0) and r.get("noslip_iterations", 0) == 0 and r.get("status") == "complete"]
    r0 = t1[0] if t1 else {"pred_d0": 0.865766, "pred_Lambda": 90.74, "pred_tc_s": 0.01}
    tc = r0["pred_tc_s"]
    v.update(REL_X=f"{x:.2f}", REL_MS=f"{x * tc * 1e3:.0f}", TC_MS=f"{tc * 1e3:g}", TC2_MS=f"{2 * tc * 1e3:g}")
    t2 = sorted(((_tc_of(r.get("rig_spec")), r) for r in creep if r.get("part") == "T2" and _eq(r.get("dt_ms"), 1.0)
                 and _eq(r.get("N"), 1.0) and _eq(r.get("impratio"), 100.0) and r.get("noslip_iterations", 0) == 0
                 and r.get("status") == "complete" and r.get("rbar_onset_mm")), key=lambda x: x[0])
    # slip at the three t_c of part T1, impratio 100 without noslip iterations (the noslip rows read 1.004)
    mus = sorted({round(r["mu_eff"], 3) for r in creep if r.get("part") == "T1" and _eq(r.get("N"), 1.0)
                  and _eq(r.get("dt_ms"), 1.0) and _eq(r.get("impratio"), 100.0) and r.get("noslip_iterations", 0) == 0
                  and r.get("status") == "complete" and r.get("mu_eff")})
    v.update(ONSET_TC=", ".join(f"{r['rbar_onset_mm']:.3f}" for _, r in t2[:-1]) + f" and {t2[-1][1]['rbar_onset_mm']:.3f}" if t2 else "",
             ONSET_TCS=_lst([t * 1e3 for t, _ in t2], "&#8202;ms") if t2 else "",
             KIN_TC=_rng([r["rbar_kin_mm"] for _, r in t2]) if t2 else "",
             MU_TC=(f"{mus[0]:.3f} at each" if len(mus) == 1 else _rng(mus)) if mus else "")
    # step 7: the bed's railed pads and the chain's thumb
    lam_b, d0_b = r0["pred_Lambda"], r0["pred_d0"]
    ks_b = 1.0 / ((1 - d0_b) * lam_b * tc ** 2)
    w_tool = 1.0 / M_TOOL
    wp_b = lam_b - w_tool
    _, kb05 = cap_pad(0.0005, BED_CAP_DEG)
    _, kc05 = cap_pad(0.0005, math.degrees(G.CAP))
    th = {r["mscale"]: r for r in cal if r.get("exp") == "pinch" and r.get("calib") == "loadtime" and _eq(r.get("dt_ms"), 1.0)
          and _eq(r.get("N_cmd"), 0.5)}
    lam_t = th[1.0]["Lambda_touch"] if 1.0 in th else 47.2
    ks_t = th[1.0]["K_target"] if 1.0 in th else kc
    fb = (wp_b + w_tool) / (wp_b + w_tool / 4)
    ft = lam_t / th[4.0]["Lambda_touch"] if 4.0 in th else float("nan")
    tmin = lambda k, lam: 1e3 / math.sqrt(k * lam)  # noqa: E731
    v.update(BED_WP=f"{wp_b:.1f}", W_TOOL=f"{w_tool:.2f}", BED_LAM=f"{lam_b:.2f}", BED_D0=f"{d0_b:.3f}", BED_PADG=f"{1e3 / wp_b:.0f}",
             BED_TCMIN=f"{tmin(ks_b, lam_b):.2f}", BED_KS05=f"{kb05:.0f}", BED_TCMIN05=f"{tmin(kb05, lam_b):.2f}",
             TH_WP=f"{lam_t - w_tool:.2f}", TH_LAM=f"{lam_t:.1f}", TH_KS=f"{ks_t:.0f}", TH_TCMIN=f"{tmin(ks_t, lam_t):.1f}",
             TH_TCMIN05=f"{tmin(kc05, lam_t):.1f}", TH_D0=f"{th[1.0]['d0']:.3f}" if 1.0 in th else "0.741",
             MASSF_BED=f"{fb:.2f}", MASSF_TH=f"{ft:.2f}", ARMF_BED=f"{(1 - fb ** -0.25) * 100:.0f}",
             ARMF_TH=f"{(1 - ft ** -0.25) * 100:.0f}")
    # step 9: the plate pad of step 5 at the bed's impedance, and the pads of these pages
    q = d0_b + (1 - d0_b) / n_f
    longer = [qq / (2 * (math.sqrt(1 + qq) - 1)) - 1 for qq in (0.2, 1.0)]
    d0s = sorted({round(d0_b, 3), round(th[1.0]["d0"], 3) if 1.0 in th else 0.741} | set((hand or {}).get("d0") or []))
    v.update(SB_D0=f"{d0_b:.4f}", SB_N=str(n_f), SB_Q=f"{q:.4f}", SB_BOUND=f"{step_bound(tc * 1e3, d0_b, n_f):.3f}",
             SB_DAMP=f"{q * tc * 1e3:.3f}", SB_LONGER=f"{longer[0] * 100:.0f}&#8211;{longer[1] * 100:.0f}",
             SB_ONE=f"{step_bound(tc * 1e3, 0.5, 1):.1f}", SB_D0S=f"{min(d0s):.2f}&#8211;{max(d0s):.2f}",
             SB_LIM=f"{step_bound(tc * 1e3, min(d0s), 1e9):.1f}&#8211;{step_bound(tc * 1e3, max(d0s), 1e9):.1f}")
    return v


# ------------------------------------------------------------------------------------------ presliding: the sprung skin
# (2026-10-12, owner: "can we add that documentation to the main ... overview"; rows of the native-compliance study,
# docs/experiments/20261007-native_compliance/, read through scripts/native_compliance_page.py)

NC_D = os.path.join(EXP, "20261007-native_compliance")
SKIN_MODELS = ("mj_pads1", "mj_pads1_skin", "mj_pads1_bristle20a", "mj_pads1_soft", "mj_pads1_lattice1", "mj_pads1_flex20")
SKIN_LBL = {"mj_pads1": "1&#8202;mm sphere pads", "mj_pads1_skin": "Pads on a sprung skin",
            "mj_pads1_bristle20a": "Pads on rolling bristles", "mj_pads1_soft": "Softening impedance",
            "mj_pads1_lattice1": "Lattice, 1&#8202;mm coupling", "mj_pads1_flex20": "MuJoCo flex (CPU)"}
SKIN_COL = {"mj_pads1": "var(--c-sphere)", "mj_pads1_skin": "var(--c-c4)", "mj_pads1_bristle20a": "var(--c-newton)",
            "mindlin": "var(--c-ref)"}


def svg_skin():
    """Figure 13: the sprung skin. (a) Section through one pad: rigid fingertip, the skin plate on its tangential spring and
    damper, the pad spheres, the tool. (b) The pad seen along its normal: two slides and the hinge."""
    import math as _m
    W, H = 1040, 320
    out = _svg_open(W, H, "The sprung skin: the pad spheres sit on a 2 g plate joined to the rigid fingertip by two tangential "
                          "slides and a hinge about the pad normal, each with a spring and a damper")
    ink, ink2, ink3, grn, amb = "var(--ink)", "var(--ink2)", "var(--ink3)", "var(--c-c4)", "var(--c-sphere)"
    out.append(f'<defs><marker id="skin-ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
               f'<path d="M0,0 L10,5 L0,10 z" style="fill:{ink}"/></marker></defs>')
    out.append(f'<text x="14" y="22" style="fill:{ink2}">(a) Section through one pad</text>'
               f'<text x="700" y="22" style="fill:{ink2}">(b) The pad seen along its normal</text>')
    cx, R0, a = 335, 700, 0.30
    cyc = 140 - R0                                   # common centre of the arcs, far above the drawing

    def arc(r, a0, a1, n=48):
        return [(cx + r * _m.sin(t), cyc + r * _m.cos(t)) for t in [a0 + (a1 - a0) * i / n for i in range(n + 1)]]

    def y_at(r, x):
        return cyc + _m.sqrt(r * r - (x - cx) ** 2)
    core = arc(R0, -a, a)
    out.append("<path d=\"M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in core) +
               f' L{core[-1][0]:.1f},44 L{core[0][0]:.1f},44 Z" style="fill:var(--sunk);stroke:{ink3};stroke-width:1.4"/>')
    out.append(f'<text x="{cx}" y="92" text-anchor="middle" style="fill:{ink2}">Fingertip, rigid</text>')
    Rp = R0 + 48                                      # skin plate, 10 px thick
    band = arc(Rp, -a + 0.02, a - 0.02) + arc(Rp + 10, a - 0.02, -a + 0.02)
    out.append("<path d=\"M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in band) +
               f' Z" style="fill:{grn};fill-opacity:0.22;stroke:{grn};stroke-width:1.6"/>')
    Rs, n = Rp + 17, 25                              # pad spheres on the plate's outer face
    for i in range(n):
        t = (-a + 0.04) + (2 * a - 0.08) * i / (n - 1)
        out.append(f'<circle cx="{cx + Rs * _m.sin(t):.1f}" cy="{cyc + Rs * _m.cos(t):.1f}" r="7" '
                   f'style="fill:{amb};fill-opacity:0.6;stroke:{amb};stroke-width:1"/>')
    ty = cyc + Rs + 7                                 # the tool touches the central spheres
    out.append(f'<rect x="120" y="{ty:.1f}" width="430" height="{H - 14 - ty:.1f}" style="fill:var(--rule2);stroke:{ink3};stroke-width:1.2"/>')
    out.append(f'<text x="{cx}" y="{H - 24}" text-anchor="middle" style="fill:{ink2}">Tool</text>')
    ay = ty + 30
    out.append(f'<line x1="{cx - 60}" y1="{ay:.1f}" x2="{cx + 40}" y2="{ay:.1f}" style="stroke:{ink};stroke-width:2" marker-end="url(#skin-ah)"/>'
               f'<text x="{cx + 52}" y="{ay + 5:.1f}" style="fill:{ink}">Tangential load</text>')
    # spring and damper between a tab on the fingertip (left) and a tab on the skin (right)
    xl, xr = cx - 72, cx + 72
    yc_l, yp_r = y_at(R0, xl), y_at(Rp, xr)
    ys, yd = yc_l + 14, yc_l + 32
    out.append(f'<line x1="{xl}" y1="{yc_l:.1f}" x2="{xl}" y2="{yd + 8:.1f}" style="stroke:{ink3};stroke-width:3"/>'
               f'<line x1="{xr}" y1="{yp_r:.1f}" x2="{xr}" y2="{ys - 8:.1f}" style="stroke:{grn};stroke-width:3"/>')
    out.append(_zigzag(xl, ys, xr, ys, ink, amp=4.5, n=8))
    xm = (xl + xr) / 2
    out.append(f'<line x1="{xl}" y1="{yd:.1f}" x2="{xm - 4}" y2="{yd:.1f}" style="stroke:{ink};stroke-width:1.6"/>'
               f'<path d="M{xm - 12},{yd - 6:.1f} L{xm + 26},{yd - 6:.1f} M{xm - 12},{yd + 6:.1f} L{xm + 26},{yd + 6:.1f} '
               f'M{xm - 12},{yd - 6:.1f} L{xm - 12},{yd + 6:.1f}" style="fill:none;stroke:{ink};stroke-width:1.6"/>'
               f'<line x1="{xm + 6}" y1="{yd - 4:.1f}" x2="{xm + 6}" y2="{yd + 4:.1f}" style="stroke:{ink};stroke-width:2.4"/>'
               f'<line x1="{xm + 6}" y1="{yd:.1f}" x2="{xr}" y2="{yd:.1f}" style="stroke:{ink};stroke-width:1.6"/>')
    # labels with leader lines
    lead = lambda x1, y1, x2, y2: f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" style="stroke:{ink3};stroke-width:1"/>'  # noqa: E731
    xs = cx - 175
    out.append(lead(xs, y_at(Rp + 5, xs), 112, 150) + f'<text x="106" y="155" text-anchor="end" style="fill:{grn}">Skin, 2&#8202;g</text>')
    xs2 = cx - 190
    out.append(lead(xs2, y_at(Rs, xs2) + 7, 112, 222) + f'<text x="106" y="227" text-anchor="end" style="fill:{amb}">Pad spheres</text>')
    out.append(lead(xr + 6, ys, 556, 150) + f'<text x="562" y="146" style="fill:{ink}">Spring and damper</text>'
               f'<text x="562" y="164" style="fill:{ink}">on each slide</text>')
    # (b) plan view: plate disc, sphere array, slide arrows, hinge
    px, py, pr = 875, 170, 100
    out.append(f'<circle cx="{px}" cy="{py}" r="{pr}" style="fill:{grn};fill-opacity:0.16;stroke:{grn};stroke-width:1.6"/>')
    sp = 15.0
    for i in range(-7, 8):
        for j in range(-7, 8):
            x, y = px + sp * (i + 0.5 * (j % 2)), py + sp * 0.866 * j
            if (x - px) ** 2 + (y - py) ** 2 < (pr - 9) ** 2:
                out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6.4" style="fill:{amb};fill-opacity:0.5;stroke:{amb};stroke-width:0.9"/>')
    out.append(f'<line x1="{px}" y1="{py}" x2="{px + pr + 36}" y2="{py}" style="stroke:{ink};stroke-width:2.2" marker-end="url(#skin-ah)"/>'
               f'<line x1="{px}" y1="{py}" x2="{px}" y2="{py - pr - 30}" style="stroke:{ink};stroke-width:2.2" marker-end="url(#skin-ah)"/>')
    out.append(f'<text x="{px + pr - 6}" y="{py + 24}" style="fill:{ink}">Slide 1</text>'
               f'<text x="{px + 10}" y="{py - pr - 18}" style="fill:{ink}">Slide 2</text>')
    hr = 30
    out.append(f'<path d="M{px + hr},{py} A{hr},{hr} 0 1 1 {px:.1f},{py - hr:.1f}" style="fill:none;stroke:{ink};stroke-width:2.2" '
               f'marker-end="url(#skin-ah)"/><circle cx="{px}" cy="{py}" r="4" style="fill:{ink}"/>')
    out.append(f'<text x="{px}" y="{py + pr + 30}" text-anchor="middle" style="fill:{ink}">Hinge about the pad normal</text>')
    out.append("</svg>")
    return "".join(out)


def svg_t8_loops(X):
    """Figure 14: task 8 at 1 N. (a) The tool's displacement against the applied force over the whole cycle at the bed's
    settings, with Mindlin's closed form (the native-compliance page's Figure 1 with this page's labels). (b) The
    displacement against time at the policy trainer's impratio of 10 and 2 ms step (2026-10-12), with Mindlin's."""
    import contact_reference_laws as L
    import contact_overview_text as TXT
    import numpy as np
    W, H = 860, 720
    out = _svg_open(W, H, "Task 8 at 1 N: tool displacement against axial force at the bed's settings, and against time at "
                          "the policy trainer's impratio of 10, for the pads, the skin, the bristles and Mindlin's closed form")
    out.append('<text x="20" y="22" style="fill:var(--ink)">(a) Bed settings: pads at impratio 100, skin and bristles at '
               '1000, 1 ms step</text>')
    fx, fy = _panel(out, 90, 66, 690, 230, (-1.1, 1.1), (-50, 50), [-1, -0.5, 0, 0.5, 1], [-40, -20, 0, 20, 40],
                    "Axial force on the tool (N)", "Tool displacement (&#181;m)")
    mc = L.mindlin_cycle(1.0)
    _path(out, fx, fy, list(zip(mc["F"], mc["u"] * 1e6)), SKIN_COL["mindlin"], dashed=True, width=2.4)
    for m in ("mj_pads1", "mj_pads1_bristle20a", "mj_pads1_skin"):
        r = X.t8r(m)
        if r is not None and r.get("trace"):
            tr = np.array(r["trace"])
            _path(out, fx, fy, [(F, u) for _, F, u in tr if abs(u) < 60], SKIN_COL[m], width=2.0)
    out.append('<text x="20" y="392" style="fill:var(--ink)">(b) The policy trainer&#8217;s impratio of 10, 2 ms step: '
               'displacement against time</text>')
    gx, gy = _panel(out, 90, 436, 690, 230, (0, 3.2), (-60, 160), [0, 0.5, 1, 1.5, 2, 2.5, 3], [-40, 0, 40, 80, 120, 160],
                    "Time in the cycle (s)", "Tool displacement (&#181;m)")
    t, _, u = TXT.mindlin_t8_trace(1.0)
    _path(out, gx, gy, list(zip(t, u * 1e6)), SKIN_COL["mindlin"], dashed=True, width=2.4)
    for m, arm in (("mj_pads1_ir10", None), ("mj_pads1_skin_ir10", "hand 0.26 kg")):
        r = TXT.t8_trainer(m, arm)
        if r is not None and r.get("trace"):
            col = SKIN_COL["mj_pads1_skin" if "skin" in m else "mj_pads1"]
            _path(out, gx, gy, [(a, c) for a, _, c in r["trace"]], col, width=2.0)
    out.append("</svg>")
    return "".join(out) + _legend_html([("Mindlin, closed form", SKIN_COL["mindlin"], True, None),
                                        ("1 mm sphere pads", SKIN_COL["mj_pads1"], False, None),
                                        ("Pads on a sprung skin", SKIN_COL["mj_pads1_skin"], False, None),
                                        ("Pads on rolling bristles", SKIN_COL["mj_pads1_bristle20a"], False, None)])


def skin_data():
    import native_compliance_page as NC
    return NC.Data()


def skin_table(X):
    """Table 3: what each presliding candidate adds (task 8 at 1 N) and keeps (tasks 1, 2, 5), and its cost."""
    import native_compliance_page as NC
    gp, cp = X.gpu("mj_pads1_ir1000"), X.cpu("mj_pads1_ir1000")
    head = ["Model", "Presliding (&#181;m)", "Recovered (&#181;m)", "Loop (&#181;J)", "Effective &#956;", "Twist arm / law",
            "80&#176; force (N)", f"CPU ({US})", "GPU, &#215; pads"]
    import contact_overview_text as TXT
    lbl = dict(SKIN_LBL, mj_pads1_ir1000="1&#8202;mm sphere pads, impratio 1000",
               mj_pads1_skin="Pads on a sprung skin, impratio 1000", mj_pads1_bristle20a="Pads on rolling bristles, impratio 1000")
    models = list(SKIN_MODELS)
    models.insert(1, "mj_pads1_ir1000")
    body = []
    for m in models:
        r8, t1, t5 = X.t8r(m), X.t1(m), X.t5(m)
        t2 = NC._t2_list(X, m, (1.0,))
        c = cp if m in ("mj_pads1", "mj_pads1_ir1000") else X.cpu(m)
        g = gp if m in ("mj_pads1", "mj_pads1_ir1000") else X.gpu(m)
        ok = lambda r: r is not None and r.get("status", "complete") == "complete"  # noqa: E731
        body.append([lbl[m], fmt(r8["u_presliding_um"], 1) if ok(r8) else "&#8211;",
                     fmt(r8["u_recovered_um"], 1) if ok(r8) else "&#8211;", fmt(r8["loop_area_uJ"], 2) if ok(r8) else "&#8211;",
                     fmt(t1.get("mu_eff"), 3) if t1 else "&#8211;", fmt(t2[0][0], 2) if t2 else "&#8211;",
                     fmt(t5["N_at_80_N"], 2) if ok(t5) else "&#8211;", num(c, ",.0f") if c else "&#8211;",
                     ("1.0" if m in ("mj_pads1", "mj_pads1_ir1000", "mj_pads1_soft") else (f"{g / gp:.1f}" if g else "CPU only"))])
        sc = TXT.t8_conv(m) if m == "mj_pads1_skin" else None
        if sc:
            body.append(["Same skin at a 0.025&#8202;ms step", fmt(sc["u_presliding_um"], 1), fmt(sc["u_recovered_um"], 1),
                         fmt(sc["loop_area_uJ"], 2)] + ["&#8211;"] * 5)
            for tm, arm, lab in (("mj_pads1_ir10", None, "1&#8202;mm sphere pads, impratio 10, 2&#8202;ms step"),
                                 ("mj_pads1_skin_ir10", "hand 0.26 kg", "Pads on a sprung skin, impratio 10, 2&#8202;ms step")):
                tr = TXT.t8_trainer(tm, arm)
                if tr:
                    body.append([lab, fmt(tr["u_presliding_um"], 1), fmt(tr["u_recovered_um"], 1),
                                 fmt(tr["loop_area_uJ"], 1)] + ["&#8211;"] * 5)
    d5 = X.t5("drake_hydro")
    body.append(["Drake hydroelastic", "0.05", "&#8722;0.05", "0.12", fmt((pick(X.bpu, "drake_hydro", N=1.0, dt_ms=1.0) or {}).get("mu_eff"), 3),
                 fmt((NC._t2_list(X, "drake_hydro", (1.0,)) or [[None]])[0][0], 2), fmt(d5["N_at_80_N"], 2) if d5 else "&#8211;",
                 num(X.cpu("drake_hydro"), ",.0f"), "&#8211;"])
    body.append(["Mindlin and Hertz, 10&#8202;MPa", "40.3", "38.0", "6.13", "1", "&#8211;", "&#8211;", "&#8211;", "&#8211;"])
    out = ["<table class='cap'><thead><tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr></thead><tbody>"]
    for row in body:
        cls = " class='ref'" if row[0].startswith("Mindlin") else ""
        out.append(f"<tr{cls}><td>{row[0]}</td>" + "".join(f"<td class='num'>{c}</td>" for c in row[1:]) + "</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def film_tag(path, poster, caption):
    return (f'<figure><video src="{R.data_uri(path, "video/mp4")}" poster="{R.data_uri(poster, "image/jpeg")}" controls muted '
            f'loop playsinline preload="metadata"></video><figcaption>{caption}</figcaption></figure>')


def _url(path):
    return open(path).read().strip() if os.path.exists(path) else None


RELATED = [  # (group, date, title, path relative to the repository, contents)
    ("Results on other pages", "2026-10-18",
     "Calibration of Drake's hydroelastic field and the 1 mm pads to finite elements of the printed 85A fingertips",
     "docs/experiments/20261018-fe_contact_calibration/20261018-fe_contact_calibration.html",
     "One modulus and a stiff modulus behind a series spring fitted to the finite-element approach of four printed tips; "
     "twist arm; pads against Drake on the same servo commands; the capsule's anisotropy"),
    ("Results on other pages", "2026-10-16", "Deployment readiness of the CAD hand with printed 85A magnet fingertips",
     "docs/experiments/20261016-sim2real_fingertip_readiness/20261016-cad_hand_fingertip_deployment_readiness.html",
     "Pads v1, v2 and Drake on printed 85A blocks and curved tips at their finite-element modulus; the CAD hand&#8217;s "
     "10&#8202;mm cylinder rerun with the printed fingers, where both builders hold the same runs; D1&#8211;D8 replays and the "
     "D6 policies on the new tips"),
    ("Results on other pages", "2026-10-15", "Capsule and ellipsoid fingertips against the printed block on the SR2 hand",
     "docs/experiments/20261015-curved_tips/20261015-capsule_and_ellipsoid_fingertips.html",
     "Nine curved tips through the bed, cost, trajectory and task protocol; probe maps of the printed tips against finite "
     "elements, Drake's field and the pad builders; the 10&#8202;mm cylinder under Drake and both builders"),
    ("Results on other pages", "2026-10-14", "Sphere-pad accuracy across fingertip geometry and fingertip co-design on the SR2 hand",
     "docs/experiments/20261014-tip_codesign/20261014-sphere_pad_accuracy_and_fingertip_codesign.html",
     "The pads against the law and Drake over fillet radius and spin rate; 24 printable tip shapes screened on the hand's "
     "tasks, print recipes and files"),
    ("Results on other pages", "2026-10-14", "Cost and control outcomes of the sphere-pad fingertip on the real_v1 hand",
     USE_PAGE, "CPU, MuJoCo-Warp and PPO cost of the pads and cheaper fingertip models; agreement with Drake on the "
     "servo-limited plant; usable physics step; planner rollout model; RL training and scoring scene (U1-U5)"),
    ("Results on other pages", "2026-10-14", "Scale-study turn on the CAD hand",
     "docs/experiments/20261014-scale_turn_cad_v1/20261014-scale_turn_cad_v1.html",
     "HOM control (arXiv 2609.25619) and sampling model-predictive control on the CAD-model hand; pad cost in the CPU and "
     "GPU planners"),
    ("Results on other pages", "2026-10-13", "In-hand cylinder turn on the scale-study layouts",
     "docs/experiments/20261013-scale_turn_controllers/20261013-scale_turn_hom_mpc.html",
     "HOM control and sampling model-predictive control on the real_v1 hand with the 1&#8202;mm pads as plant and rollout "
     "model; planner cost"),
    ("Results on other pages", "2026-10-13", "MuJoCo model of the SR2 finger assembled from its CAD parts",
     "docs/experiments/20261013-cad_v1_finger/20261013-cad_v1_finger_model.html",
     "1&#8202;mm pads on the CAD fingertip, MorphIt link spheres, step cost"),
    ("Results on other pages", "2026-10-13", "Sensitivity of the reorientation policies to fingertip contact parameters",
     "docs/experiments/20261013-contact_feedback/20261013-contact_parameter_sensitivity.html",
     "Closed loop against open-loop replay, analytical checks of the sphere pads"),
    ("Results on other pages", "2026-10-11", "Fingertip contact models on the servo-limited plant",
     "docs/experiments/20261011-servo_plant_policies/20261011-servo_plant_contact_model_policies.html",
     "Point-contact, pad and sprung-skin policies at the servos&#8217; strength: transfer, replays, training cost"),
    ("Results on other pages", "2026-10-06", "Printed fingertip across simulators", TIP_PATH,
     "The printed TPU block on the bed, creep against impratio, the chain at four impratios, the deployed turn"),
    ("Results on other pages", "2026-10-11", "Audit of the DiffMJX contact-gradient probes",
     "docs/experiments/20261011-diffmjx_audit/20261011-diffmjx_gradient_probe_audit.html",
     "The DiffMJX and contact-release studies script by script"),
    ("Results on other pages", "2026-10-12", "Presliding compliance from a sprung skin under the sphere pads",
     "docs/experiments/20261012-sprung_skin/20261012-sprung_skin_presliding.html",
     "Mindlin presliding, the skin&#8217;s construction and sizing, task 8 at the bed&#8217;s and the trainer&#8217;s settings, policies "
     "trained on the skin"),
    ("Results on other pages", "2026-10-05", "Contact comparison bed", "BED",
     "Five tasks on the two-pad rig for every model, with films and the numbers of Table 2"),
    ("Results on other pages", "2026-10-06", "Servo plant refit and the three-finger turn", "HT3",
     "Hand-object control of the turn, agreement across simulators, effective mass, pad cost in RL"),
    ("Results on other pages", "2026-10-07", "Newton hydroelastic tests",
     "docs/experiments/20261007-newton_hydro_tests/20261007-newton_hydroelastic_friction_reduction_edge.html",
     "Friction rows, contact reduction, voxel size and edge contact"),
    ("Results on other pages", "2026-10-07", "Presliding compliance with native MuJoCo elements",
     "docs/experiments/20261007-native_compliance/20261007-native_presliding_compliance.html",
     "Skin, rolling-bristle, softening and lattice candidates against Hertz and Mindlin"),
    ("Results on other pages", "2026-10-08", "Policies trained on four fingertip contact models",
     "docs/experiments/20261008-contact_model_policies/20261008-fingertip_contact_model_policies.html",
     "D6 reorientation: training, checkpoint watch, transfer and replays"),
    ("Results on other pages", "2026-10-08", "Finger-base spacing against object size",
     "docs/experiments/20261008-hand_object_scale/20261008-finger_spacing_object_size.html",
     "Simulated landscape on the 1 mm pads, bench protocol and paper plan"),
    ("Results on other pages", "2026-10-02", "Chain control on the SR2 hand", "CHAIN",
     "Derivation, eight contact models, the tasks of arXiv 2609.25619, cost"),
    ("Results on other pages", "2026-10-01", "Pinch friction torque in Drake and MuJoCo",
     "docs/experiments/20261001-hom_contact_patch/20261001-hom_pinch_contact_models.html", "First two-pad rig"),
    ("Results on other pages", "2026-10-01", "Pinch-brake loading on the SR2 hand",
     "docs/experiments/20261001-hom_hand_brake/20261001-hom_hand_brake.html", "Brake in Drake and MuJoCo on the hand"),
    ("Printed fingertips (topic overview: printed_fingertips.html)", "2026-10-09",
     "Stiffness, sphere-pad calibration and embedded magnet of the printed TPU fingertip",
     "docs/experiments/20261009-tpu_tip_fem/20261009-printed_tpu_fingertip_finite_elements.html",
     "Finite elements of the gyroid tip, the pads&#8217; modulus, magnet depth, print plan and calculator"),
    ("Printed fingertips (topic overview: printed_fingertips.html)", "2026-10-13",
     "Printability, stiffness and a one-command build of eFlesh lattice fingertips",
     "docs/experiments/20261013-eflesh_printability/20261013-eflesh_lattice_printability.html",
     "eFlesh audit, as-printed stiffness from the G-code"),
    ("Printed fingertips (topic overview: printed_fingertips.html)", "2026-10-13",
     "MLX90393 readout of printed fingertips on the five-chip board",
     "docs/experiments/20261013-magnet_layouts/20261013-mlx90393_fingertip_readout.html",
     "Force noise, disturbances, one or four magnets"),
    ("Codex audit, 2026-10-04", "2026-10-04", "Distributed fingertip contact audit",
     "docs/experiments/20261004-codex/20261004-distributed_contact.html", "Static stiffness, regression and the newer-MuJoCo check"),
    ("Codex audit, 2026-10-04", "2026-10-04", "Articulated holding, torsion, GPU and Newton",
     "docs/experiments/20261004-codex/20261004-articulated_contact_transfer.html", "Follow-up on the hand"),
    ("Codex audit, 2026-10-04", "2026-10-04", "Controller and physics-step sweep",
     "docs/experiments/20261004-codex/20261004-controller_timestep.html", "50 Hz control at steps from 50 &#181;s to 10 ms"),
    ("Codex audit, 2026-10-04", "2026-10-04", "Coupled-foundation models for the printed tip", "ASIDES",
     "How the coupling length would be measured"),
    ("Codex audit, 2026-10-04", "2026-10-04", "Video gallery", "docs/experiments/20261004-codex/20261004-contact_videos.html",
     "Films of the audit"),
    ("Earlier revisions of this page", "2026-10-09", "Revision of 2026-10-09",
     "docs/experiments/20261009-contact_overview/20261009-sphere_pad_contact_model.html", ""),
    ("Earlier revisions of this page", "2026-10-08", "Revision of 2026-10-08",
     "docs/experiments/20261008-contact_overview/20261008-sphere_pad_contact_model.html", ""),
    ("Earlier revisions of this page", "2026-10-07", "Revision of 2026-10-07",
     "docs/experiments/20261007-contact_overview/20261007-sphere_pad_contact_model.html", ""),
    ("Earlier revisions of this page", "2026-10-05", "Revision of 2026-10-05",
     "docs/experiments/20261005-contact_overview/20261005-sphere_pad_contact_model.html", ""),
]


def related():
    """Index of linked pages: a table grouped by kind, each title linked to the local file and, where one exists, the
    artifact (owner 2026-10-09: a more usable index at the bottom)."""
    alias = {"BED": BED_PATH, "HT3": HT3_PATH, "CHAIN": CHAIN_PATH, "ASIDES": ASIDES_PATH}
    here = os.path.dirname(os.path.relpath(OUT, ROOT))
    out, group = ["<table class='index'><thead><tr><th>Date</th><th>Page</th><th>Contents</th><th>Artifact</th></tr></thead><tbody>"], None
    groups = list(dict.fromkeys(g for g, *_ in RELATED))         # group order as listed, newest first within a group
    rows = [r for g in groups for r in sorted((r for r in RELATED if r[0] == g), key=lambda r: r[1], reverse=True)]
    for g, date, title, path, what in rows:
        path = alias.get(path, path)
        if g != group:
            out.append(f"<tr class='grp'><td colspan='4'>{g}</td></tr>")
            group = g
        u = _url(os.path.join(ROOT, os.path.dirname(path), "artifact_url.txt"))
        if path == CHAIN_PATH:
            u = u or CHAIN_URL
        art = f"<a href='{u}'>link</a>" if u else ""
        out.append(f"<tr><td class='num'>{date}</td><td><a href='{os.path.relpath(os.path.join(ROOT, path), os.path.join(ROOT, here))}'>"
                   f"{title}</a></td><td>{what}</td><td>{art}</td></tr>")
    out.append("</tbody></table>")
    return "".join(out)


def bed_url():
    if os.path.exists(BED_URL_FILE):
        u = open(BED_URL_FILE).read().strip()
        return u or None
    return None


# ------------------------------------------------------------------------------------------ main

def style_block():
    t = open(CHAIN_TPL).read()
    m = re.search(r"<style>(.*?)</style>", t, re.S)
    return "<style>" + m.group(1) + EXTRA_CSS + "</style>"


TEX_RE = CP.TEX_RE


def render_tex(t):
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None)
             for m in TEX_RE.finditer(t)]
    svgs = iter(texsvg.render(items, cache_path=TEX_CACHE, scale=TEX_SCALE))
    return TEX_RE.sub(lambda m: next(svgs), t), len(items)


def usability_section():
    """The paragraph and film of the 2026-10-14 usability study (U1-U5) on the real_v1 hand, from its summaries."""
    def js(rel):
        p = os.path.join(USE_DIR, rel)
        return json.load(open(p)) if os.path.exists(p) else None
    f1, f2, s3, s4, s5 = (js("cost/findings_U1.json"), js("agree/findings_U2.json"), js("step/summary_U3.json"),
                          js("planner/summary_U4.json"), js("rl/summary_U5.json"))
    if not (f1 and f2 and s3 and s5):
        return ""
    link = os.path.relpath(os.path.join(ROOT, USE_PAGE), os.path.dirname(OUT))
    url = _url(os.path.join(USE_DIR, "artifact_url.txt"))
    cpu = next(t for t in f1["tables"] if t["title"].startswith("CPU cost"))["rows"]
    grip = {r[1]: float(r[3].split()[0]) for r in cpu if r[0] == "16 mm cylinder"}
    gpu = next(t for t in f1["tables"] if t["title"].startswith("MuJoCo-Warp physics"))["rows"]
    g = {r[0]: float(r[2].split(" / ")[0]) for r in gpu if r[1] == "2048" and not r[2].startswith("skipped")}
    agree = {(r[0], r[1], r[2]): r[3] for r in f2["tables"][0]["rows"]}
    a = lambda m: agree[(m, "Servo", "1 ms, impratio 100")]  # noqa: E731
    sw = {(x["model"], x["plant"][:4], x["dt_ms"]): x for x in s3["sweep"]}
    C = {(c["final"], c["scene"], c["mode"], c["condition"]): c["n_held"] for c in s5["cells"]}
    kp = lambda sc, fs: " and ".join(str(C[(f, sc, "closed", "kp=0.36")]) for f in fs)  # noqa: E731
    s = ("<h3>Cost and control outcomes on the real_v1 hand</h3>"
         f"<p>On the real_v1 hand the 1&#8202;mm pads cost {grip['1 mm pads']:.0f}&#8202;&#181;s per grip step on one CPU core, "
         f"{grip['1 mm pads'] / grip['Block mesh, point contact']:.1f} times point contact on the TPU block. In MuJoCo-Warp they "
         f"cost {g['1 mm pads']:.2f}&#8202;&#181;s per world-step, {g['1 mm pads'] / g['Block mesh']:.1f} times the block mesh, and "
         "a PPO iteration costs 1.47 times the mesh&#8217;s (U1 of the "
         f"<a href=\"{link}\">usability page</a>" + (f", <a href=\"{url}\">artifact</a>" if url else "") + "). On the servo "
         f"plant the pads end the eight deployed plans within 3&#176; of Drake on {a('1 mm pads')} of 24 placements. 2&#8202;mm pads "
         f"do so on {a('2 mm pads')}, and point contact on {a('Point contact')} (U2). At a 2&#8202;ms step the pads keep "
         f"{sw[('pads', 'kp0.', 2.0)]['within3_1ms']} of 40 turns within 3&#176; of their own turn at 1&#8202;ms. With point contact "
         f"the hand keeps the tool on only {sw[('pt', 'kp0.', 1.0)]['held']} of 40 placements even at 1&#8202;ms (U3).")
    if s4:
        R4 = {r["config"]: r for r in s4["configs"]}
        p4, c4, pt = R4["icem_pin"], R4["icem_pin_pt4"], R4["icem_pin_pt"]
        s += (f" Pinned iCEM that rolls out condim&#160;4 or point contact in place of the pads replans "
              f"{p4['replan_s_3threads'] / c4['replan_s_3threads']:.0f} and {p4['replan_s_3threads'] / pt['replan_s_3threads']:.0f} "
              f"times faster. It loses a median {-c4['paired_change_median']:.0f}&#176; and {-pt['paired_change_median']:.0f}&#176; "
              "of held turn on the pad plant (U4).")
        fr = {v["variant"]: v for v in s4.get("friction", [])}
        if "mu0.5" in fr:
            v = fr["mu0.5"]
            s += (f" The pads planner itself, planning with friction 1 on a plant at 0.5, ends {v['three_finger']} of {v['n']} "
                  "runs with three fingertips loaded.")
    s += (" At half servo gain, RL policies trained on the TPU mesh keep the tool on the 1&#8202;mm pad plant as often as "
          f"pad-trained ones do. They hold {kp('tpu27pads1', ('Mesh s0', 'Mesh s1'))} of 128 rollouts, against "
          f"{kp('tpu27pads1', ('Pads s0', 'Pads s1'))}. The pad-trained policies fall to {kp('tpu27mesh', ('Pads s0', 'Pads s1'))} "
          "on the mesh plant (U5). Held counts do not show that two contact models give the same dynamics. The same 1&#8202;mm "
          "pads, built by the same builder, are therefore used for training, planning and evaluation, with the same contact "
          "numerics wherever the trainer still holds the grasp under them (owner, 2026-10-15).</p>")
    film = os.path.join(USE_DIR, "rl/media/20261014-rl_cross_kp036.mp4")
    poster = os.path.join(USE_DIR, "rl/media/20261014-rl_cross_kp036.jpg")
    if os.path.exists(film) and os.path.exists(poster):
        s += film_tag(film, poster, "Figure&#160;13. Half servo gain (0.36&#8202;N&#183;m/rad), closed loop. Top: the TPU-mesh "
                      "policy seed 0 on its own plant and on the 1&#8202;mm pad plant. Bottom: the 1&#8202;mm pad policy seed 0 on its "
                      "own plant and on the TPU-mesh plant. Each panel is the median world of 128; the pad policy drops the "
                      "tool on the mesh plant (U5 of the usability page).")
    cd = "docs/experiments/20261014-tip_codesign/20261014-sphere_pad_accuracy_and_fingertip_codesign.html"
    if os.path.exists(os.path.join(ROOT, cd)):
        link2 = os.path.relpath(os.path.join(ROOT, cd), os.path.dirname(OUT))
        url2 = _url(os.path.join(ROOT, "docs/experiments/20261014-tip_codesign/artifact_url.txt"))
        s += ("<p>Across fingertip shapes the pads keep the twist-onset torque within 2&#8202;% of the quasi-static law on "
              "fillets of 1.5&#8202;mm or more. Their sliding torque, however, grows with spin rate: it is 1.25 times the law at "
              "1&#8202;rad/s on the printed 2.7&#8202;mm fillet. MuJoCo&#8217;s elliptic cone adds a slip term to a sliding pad&#8217;s "
              "normal force, and the change to the friction rows that removes it breaks the hand&#8217;s closed-loop turns "
              "(<a href=\"" + link2 + "\">fingertip co-design page</a>" + (f", <a href=\"{url2}\">artifact</a>" if url2 else "")
              + ")." + _builder_sentence() + _curved_sentence() + _calibration_sentence() + "</p>")
    return s


def _calibration_sentence():
    """The calibration of Drake's field and the pads to the finite elements of the four printed 85A tips (2026-10-18)."""
    cp = "docs/experiments/20261018-fe_contact_calibration/20261018-fe_contact_calibration.html"
    if not os.path.exists(os.path.join(ROOT, cp)):
        return ""
    link = os.path.relpath(os.path.join(ROOT, cp), os.path.dirname(OUT))
    url = _url(os.path.join(ROOT, "docs/experiments/20261018-fe_contact_calibration/artifact_url.txt"))
    return (" The finite elements of the printed tips were also used for calibration. There a stiff modulus behind a "
            "9.7&#8202;N/mm series spring reproduces the approach of the 11/22 half-ellipsoid tip, within 5.2&#8202;% RMS in Drake "
            "and 5.6&#8202;% in 0.5&#8202;mm pads. On the same servo commands, however, the two end the tool&#8217;s turn 27&#8211;57&#176; apart on the 25&#8202;mm "
            "cell, and one spring cannot fit the capsule tip (<a href=\"" + link + "\">calibration page</a>"
            + (f", <a href=\"{url}\">artifact</a>" if url else "") + ").")


def _curved_sentence():
    """The curved-tip study (2026-10-15): spin arm on curved tips and the printed tips' stiffness against the models."""
    cp = "docs/experiments/20261015-curved_tips/20261015-capsule_and_ellipsoid_fingertips.html"
    if not os.path.exists(os.path.join(ROOT, cp)):
        return ""
    link = os.path.relpath(os.path.join(ROOT, cp), os.path.dirname(OUT))
    url = _url(os.path.join(ROOT, "docs/experiments/20261015-curved_tips/artifact_url.txt"))
    return (" Capsule and half-ellipsoid tips in the same envelope turn the tool on a third of the block&#8217;s friction arm. "
            "Their pads stay within 6&#8202;% of the law&#8217;s spin arm at 1&#8202;rad/s. Finite elements of the printed tips show "
            "a nearly uniform surface stiffness, which neither Drake&#8217;s field nor any pad builder reproduces (<a href=\""
            + link + "\">curved fingertips page</a>" + (f", <a href=\"{url}\">artifact</a>" if url else "") + ").")


def _builder_sentence():
    """Builder v2 (per-pad depth) as the default, with the trajectory-level check against Drake (tip study,
    builder_default/summary.json)."""
    p = os.path.join(ROOT, "docs/experiments/20261014-tip_codesign/builder_default/summary.json")
    if not os.path.exists(p):
        return ""
    r = json.load(open(p)).get("replay", {}).get("tpu2.7")
    if not r:
        return ""
    m = lambda pair: r["pairs"][pair]["ang_rms_deg"]["median"]  # noqa: E731
    return (" Since 15 October each pad takes its own depth below the tip&#8217;s volume centroid (builder v2). Over the servo "
            "plant&#8217;s plan replays, the tool&#8217;s orientation then differs from Drake&#8217;s by a median RMS of "
            f"{m('v2-drake'):.1f}&#176;. With one stiffness per finger it differs by {m('v1-drake'):.1f}&#176;, and between two "
            f"tool placements in Drake by {r['seed_spread']['drake']['ang_rms_deg']['median']:.1f}&#176;.")


def main():
    import contact_overview_text as OT  # prose blocks that depend on the literature notes and the bed findings
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    T = {name: bed(fn) for name, fn in (("pull", "pull_slip"), ("twist", "twist_slip"), ("roll", "roll"),
                                         ("shake", "shake"), ("brake", "brake"), ("creep", "creep"))}
    tor = load(os.path.join(EXP, "20261001-hom_contact_patch/torsion.jsonl"))
    cal = load(CAL)
    step_rows = json.load(open(os.path.join(CODEX, "data/task_timestep_results.json"))) \
        if os.path.exists(os.path.join(CODEX, "data/task_timestep_results.json")) else []
    gpu_pads = load(os.path.join(GPU, "gpu_scaling.jsonl"))
    gpu_newton = load(os.path.join(GPU, "newton_scaling.jsonl"))
    stab = stab_rows()
    cost = step_cost(T["pull"])
    M = metrics(T)
    curves = gpu_curves(gpu_pads, gpu_newton)
    batched = gpu_batched(curves)
    cost.update(gpu_one_world(curves))

    t = open(TPL).read()
    v = {"STYLE": style_block(), "BUILT": time.strftime("%Y-%m-%d %H:%M"), "CHAIN_PATH": CHAIN_PATH, "CHAIN_URL": CHAIN_URL,
         "BED_PATH": BED_PATH, "TEX_CACHE_REL": os.path.relpath(TEX_CACHE, ROOT)}
    u = bed_url()
    v["BED_LINK"] = f", <a href=\"{u}\">artifact</a>" if u else ""
    hi, lo = G.compute(3.0), G.compute(0.5)
    v["FIG_SECTION"] = G.svg_section(hi)
    v["FIG_PATCH"] = G.svg_patch(hi, lo)
    v["FIG_MODELS"] = svg_models()
    SX = skin_data()
    gp = SX.gpu("mj_pads1_ir1000")
    B = tip_rows()
    hc, rc, sc = hand_cost(), rl_cost(), state_cost()
    cx = lambda rows, a, b: ((pick(rows, a, N=1.0, dt_ms=1.0) or {}).get("creep_mm_s", 0)  # noqa: E731
                             / max((pick(rows, b, N=1.0, dt_ms=1.0) or {}).get("creep_mm_s", 0), 1e-15))
    creep_x = (cx(T["pull"], "mj_pads1", "drake_hydro"), cx(B["T1"], *BLOCK))
    v["CAP_TABLE"] = cap_table(cost, batched, {m: (SX.cpu(m), SX.gpu(m) / gp) for m in ("mj_pads1_skin", "mj_pads1_bristle20a")},
                               hand={k: hc.get(k) for k in ("mj_point3", "mj_pads1", "drake_hydro")}, creep_x=creep_x)
    v["BLOCK_TABLE"] = block_table(B)
    v["COST_TABLE"] = cost_table(cost, batched, hc, rc, sc, B)
    v["HAND_N"] = str(hc.get("n_paired", ""))
    v.update(deck_values(T, T["creep"], cal, hc, lo))
    v["BRISTLE_X"] = f"{SX.gpu('mj_pads1_bristle20a') / gp:.0f}"
    v["FIG_T8"] = svg_t8_loops(SX)
    fig, have_onset = svg_scaling(T["twist"], tor)
    v["FIG_SCALING"] = fig
    v["FIG_MASS"] = svg_mass(cal)
    v["MASS_TEXT"] = mass_text(cal)
    v["CREEP_TEXT"] = creep_text(T["pull"], T["creep"], B["T1"])
    v["FIG_STAB"] = svg_stab(stab)
    v["STAB_TEXT"] = stab_text(stab, tail=True)
    v["FIG_STEP"] = ifig_step(step_rows).inner() if step_rows else svg_step(step_rows)
    v["STEP_TEXT"] = step_text(step_rows)
    v["AGREE_TABLE"] = agree_table(M)
    v["FIG_COST"] = svg_cost(M, cost, batched)   # static: direct labels and the spread of the three pad models
    v["FIG_GPU"] = ifig_gpu(curves).inner() if curves else svg_gpu(curves)
    import simulator_agreement_figure as SAF
    v["FIG_AGREE_SIM"] = SAF.svg_agreement() + SAF.legend()
    v["GPU_NEWTON_SCRIPT"] = ", <code>scripts/newton_scaling.py</code>" if gpu_newton else ""
    fl = CP.FL
    v["PHI_DHY"], v["PHI_S1"] = f"{fl['dhy_closed']['phi_end']:.1f}", f"{fl['s1_closed']['phi_end']:.1f}"
    v["CLOSEUP_FILM"] = R.data_uri(os.path.join(CHAIN_D, "media/20261002-contact_closeups.mp4"), "video/mp4")
    v["CLOSEUP_POSTER"] = R.data_uri(os.path.join(CHAIN_D, "media/20261002-contact_closeups_poster.png"), "image/png")
    v["CHAIN_FILM"] = R.data_uri(os.path.join(CHAIN_D, "media/20261002-chain_six_models.mp4"), "video/mp4")
    v["CHAIN_POSTER"] = R.data_uri(os.path.join(CHAIN_D, "media/20261002-chain_six_models_poster.png"), "image/png")
    v["RELATED"] = related()
    v["USABILITY"] = usability_section()
    v["GLOSSARY"] = glossary_html(extra=GLOSSARY_RL)
    ctx = dict(T=T, M=M, cost=cost, cal=cal, stab=stab, step_rows=step_rows, gpu_pads=gpu_pads, gpu_newton=gpu_newton,
               tor=tor, have_onset=have_onset, curves=curves, batched=batched, deviations=deviations(M), bed_url=u, skin_X=SX,
               B=B, hand=hc, rl=rc, state=sc, creep_x=creep_x)
    v.update(OT.blocks(ctx))
    for k, val in v.items():
        t = t.replace("{{" + k + "}}", val)
    left = sorted(set(x.split("}}")[0] for x in t.split("{{")[1:]))
    if left:
        raise SystemExit(f"unfilled placeholders: {left}")
    t, n_tex = render_tex(t)
    import interactive_figure as IF
    t = IF.add_assets(retro_style.apply(t))  # plain page style (owner, 2026-10-09); interactive figures (2026-10-11)
    open(OUT, "w").write(t)
    print(f"formulas {n_tex}; wrote {OUT} ({os.path.getsize(OUT) / 1e6:.2f} MB)")
    deck_reminder()


# Deck 23 of the owner's contact tutorial explains this overview step by step (owner, 2026-10-14: "i also don't want the
# slides to get stale, so put a reminder somewhere here, whenever we edit the contact model docs in a significant
# fashion"). The deck's data files record the hand commit their numbers come from; DECK_CHECKED is the hand commit up to
# which someone last compared the deck with this overview and found nothing to change. Update it after such a check.
DECK = "/home/humanoid/wxie_workspace/contact/tutorial/slides/23_soft_fingertip_contact.html"
DECK_ROWS = ("/home/humanoid/wxie_workspace/contact/tutorial/data/hand_contact_bed_rows.json",
             "/home/humanoid/wxie_workspace/contact/tutorial/data/hand_skin_rows.json")
DECK_CHECKED = "0e6c03add46db53941f1e9514f764f63879ca128"
DECK_SOURCES = ("scripts/contact_overview_text.py", "scripts/contact_overview_page.py",
                "scripts/contact_overview_page.template.html", "docs/experiments/20261005-contact_bed",
                "docs/experiments/20261007-native_compliance", "docs/experiments/20261012-sprung_skin",
                "docs/experiments/20261008-contact_model_policies", "docs/experiments/20261011-servo_plant_policies")


def deck_reminder():
    """Print the hand commits since deck 23 last took its numbers that changed this overview or the rows it quotes."""
    import subprocess
    git = lambda *a: subprocess.run(["git", "-C", ROOT, *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    base = [DECK_CHECKED]
    for f in DECK_ROWS:
        if os.path.exists(f):
            d = json.load(open(f))
            base += [d.get("commit")] + [c[0] for c in (d.get("commits") or {}).values()]
    known = [c for c in base if c and git("cat-file", "-t", c) == "commit"]
    if not known:
        print(f"deck 23: no source commit of {DECK} found in this repository; compare the deck with the overview by hand")
        return
    newest = max(known, key=lambda c: int(git("log", "-1", "--format=%ct", c) or 0))
    log = git("log", "--oneline", f"{newest}..HEAD", "--", *DECK_SOURCES)
    if log:
        print(f"deck 23 may be stale: {len(log.splitlines())} commits since {newest[:8]} changed the overview or its rows."
              f"\n  Compare {DECK} with the changed sections; update the deck in its own repository (AGENTS.md there), or"
              f" set DECK_CHECKED in scripts/contact_overview_page.py to HEAD if nothing in the deck changes.")
        for line in log.splitlines()[:12]:
            print("   ", line)


if __name__ == "__main__":
    if sys.argv[1:] == ["--deck-check"]:
        deck_reminder()
    else:
        main()
