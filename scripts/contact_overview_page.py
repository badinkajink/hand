#!/usr/bin/env python3
r"""Build docs/experiments/20261007-contact_overview/20261007-sphere_pad_contact_model.html.

    python3 scripts/contact_overview_page.py

Each revision of the overview is a new dated file published to the overview's one artifact (artifact_url.txt beside
the page); the 2026-10-05 revision stays at docs/experiments/20261005-contact_overview/20261005-sphere_pad_contact_model.html.
The 2026-10-07 revision adds the agreement of the pads with Drake across MuJoCo, MuJoCo-Warp and Newton
(scripts/simulator_agreement_figure.py), the effective-mass argument that the pads build in and Newton's
hydroelastic contact lacked, the cost on one RL state and the replay of learned policies across implementations
(docs/experiments/20261006-hom_turn3/, 20261006-rl_contact/, 20261006-simulator_agreement/).

The overview page of the fingertip contact work of 1-7 October 2026: the components of a fingertip contact model,
the contact mechanics (patch torque, Winkler/hydroelastic and Hertz arms, lateral coupling), the sphere-packed pad in
MuJoCo (sampling, the soft-contact spring law, inverse-inertia calibration, friction creep, the collective step bound,
condim 4), the reference models, and the evidence: the comparison bed (docs/experiments/20261005-contact_bed/), the
inverse-inertia study (20261005-pad_calibration/), the GPU batches (20261005-gpu_scaling/), Codex's step sweep
(20261004-codex/data/task_timestep_results.json) and the chain page (20261002-hom_chain/), whose style block, Figures 1-2
and films are reused. Every number in the prose is computed here from those rows; figures whose rows are not yet written
render a short note instead. Math in the template, \( \) inline and \[ \] display, is LaTeX rendered to inline SVG by
scripts/texsvg.py, cached beside the page.
"""
from __future__ import annotations

import json
import math
import os
import re
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import hom_chain_figures as G  # noqa: E402
import hom_chain_page as CP  # noqa: E402
import hom_contact_patch_page as R  # noqa: E402
import texsvg  # noqa: E402

EXP = os.path.join(ROOT, "docs/experiments")
D = os.path.join(EXP, "20261007-contact_overview")
OUT = os.path.join(D, "20261007-sphere_pad_contact_model.html")
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
TEX_CACHE = os.path.join(D, "texsvg_cache.json")
TEX_SCALE = 1.1

CHAIN_PATH = "docs/experiments/20261002-hom_chain/20261002-hom_screwdriver_chain.html"
CHAIN_URL = "https://claude.ai/artifact/Lya2Bf6HnW81bEPRPjsjYf"
BED_PATH = "docs/experiments/20261005-contact_bed/20261005-contact_model_bed.html"
BED_URL_FILE = os.path.join(BED, "artifact_url.txt")          # written after the bed page is published
ASIDES_PATH = "docs/experiments/20261004-codex/20261004-contact_research_asides.html"
LIT_NOTES = os.path.join(ROOT, "docs/notes/20261005-contact_literature_notes.md")

C_LAW, EXP_LAW = 0.996e-3, 0.2498          # hydroelastic arm on the screwdriver, fitted to Drake (10-01 rig)
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
table.cap td,table.agree td{font-size:13.5px}
table.cap th,table.agree th{white-space:normal}
table.agree td:first-child{min-width:15em}
td.ref{background:var(--sunk)}
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
.legendrow{display:flex;flex-wrap:wrap;gap:6px 18px;margin:8px 0 0;font:400 12.5px/1.4 var(--f-mono);color:var(--ink2)}
"""


# ------------------------------------------------------------------------------------------ terms and metrics

GLOSSARY = [
    ("pad", "One fingertip&#8217;s contact with the tool. On the bed rig two pads, one per fingertip, pinch the tool; \\(N\\) is "
            "the normal force of each pad."),
    ("friction arm \\(\\bar r\\)", "Pressure-weighted mean distance of the contact patch from its centre, (1), in mm. A pad transmits "
            "at most \\(\\mu N\\bar r\\) about its normal before it spins. The hydroelastic law gives 0.84, 1.00 and 1.31&#8202;mm at 0.5, 1 "
            "and 3&#8202;N."),
    ("slip onset, effective \\(\\mu\\)", "Task&#160;1 raises an axial force on the tool at 2&#8202;N/s. Before onset the tool slides slowly "
            "at a speed proportional to the force; onset is the force at which the speed leaves that trend. Effective \\(\\mu\\) is the "
            "onset force divided by \\(2N\\) (two pads). Rigid Coulomb friction gives 1.000."),
    ("creep", "Steady sliding of the tool while the load stays below the slip force. Task&#160;1 holds the axial force at half the onset "
            "force for 1&#8202;s and reports the sliding speed in &#181;m/s; task&#160;2 holds half the onset torque and reports the turning "
            "speed in &#176;/s; in task&#160;4 it is the drift per cycle below the Coulomb threshold. Rigid Coulomb friction gives zero."),
    ("slip speed", "Mean sliding speed of the tool over the 50&#8202;ms after onset in task&#160;1, in mm/s. Rigid Coulomb friction with "
            "equal static and kinetic \\(\\mu\\) gives 34&#8202;mm/s."),
    ("arm at spin onset", "Task&#160;2 raises a torque about the pinch axis; the onset torque divided by \\(2\\mu N\\), in mm per pad. "
            "<i>Sliding arm</i>: the steady torque while the tool is driven to spin at 1&#8202;rad/s, divided by \\(2\\mu N\\). "
            "<i>Onset / law</i>: onset torque over the hydroelastic law&#8217;s \\(2\\mu N\\,cN^{1/4}\\). <i>Onset torque</i> on its own "
            "is the torque at spin onset, in mN&#8202;m; Drake gives 0.84, 1.97 and 7.65&#8202;mN&#8202;m at 0.5, 1 and 3&#8202;N."),
    ("drift per cycle", "Net displacement of the tool along its axis per 5&#8202;Hz shake cycle in task&#160;4. <i>Coulomb load ratio</i>: "
            "\\(m(g+a_\\text{pk})/(2\\mu N)\\); rigid Coulomb friction slips above 1."),
    ("swing end, largest swing, peak rate", "Task&#160;5 lowers the pinch force from 6 to 0.2&#8202;N over 4&#8202;s, with the tool&#8217;s "
            "centre of mass 15&#8202;mm off the pinch line, until the tool swings about the pinch axis. Swing end is the final angle "
            "(90&#176; = hanging), largest swing the largest angle over the run (Drake: 87.3&#176;), peak rate the largest angular "
            "speed of the swing, in &#176;/s."),
    ("rolling ratio", "Task&#160;3 moves one pad 10&#8202;mm along the tool&#8217;s surface at 10 or 50&#8202;mm/s while both pads press at "
            "\\(N\\), so the tool rolls between them. The rolling ratio is the tool&#8217;s rotation times its radius over half the pad "
            "travel. Rolling without slip gives 1 between flat pads and 1.008 between the two spheres, whose line of centres tilts as "
            "the pad moves; a contact point inside the tool surface rolls on a smaller radius and raises it. <i>Slip</i>: the path "
            "length of the relative motion of tool and pad at the contact point over the travel, in &#181;m."),
    ("tool turn, within 3&#176;, grip force", "Plan replay of the three-finger turn (Figure&#160;7): tool turn is the rotation "
            "of the tool axis toward vertical from its pose at the end of the grip, in degrees; a replay is within 3&#176; when its "
            "turn is within 3&#176; of Drake&#8217;s on the same hand and placement with the tool held in both; grip force is the sum of "
            "the three fingertips&#8217; normal forces on the tool at the end of the hold, in N."),
    ("physics step \\(\\Delta t\\)", "The simulator&#8217;s integration step; 1&#8202;ms unless stated. <i>&#181;s per step</i>: wall time of "
            "one physics step on one CPU core, controller and rendering excluded. <i>World-steps per second</i>: on the GPU, the number "
            "of parallel simulations times the steps each completes per second; <i>&#181;s per world-step</i> is its inverse."),
    ("\\(t_c\\), \\(d_0\\), \\(\\hat\\Lambda\\)", "MuJoCo contact parameters. \\(t_c\\), the <code>solref</code> time constant, sets how fast a "
            "penetration is corrected; \\(d_0\\), the <code>solimp</code> impedance, sets what fraction of that correction the solver "
            "enforces; \\(\\hat\\Lambda\\) is MuJoCo&#8217;s estimate of the contact&#8217;s inverse inertia (47.2&#8202;kg\\(^{-1}\\) for a pad on "
            "the real tool). Together they set the sphere stiffness, (10)."),
    ("inverse weight \\(w\\), effective mass \\(m_\\text{eff}\\)", "\\(w_b\\) is MuJoCo&#8217;s <code>body_invweight0</code> of body \\(b\\), its "
            "translational inverse inertia at the reference pose in 1/kg; for a contact between bodies 1 and 2, "
            "\\(\\hat\\Lambda = w_1 + w_2\\) and \\(m_\\text{eff} = 1/\\hat\\Lambda\\)."),
    ("\\(k_h\\), \\(k_f\\)", "Newton&#8217;s per-shape hydroelastic stiffness (pressure per unit depth, \\(E/h\\) for a layer of modulus "
            "\\(E\\) and thickness \\(h\\), in N/m&#179;) and its friction gain, which SolverMuJoCo turns into the friction-row time "
            "constant of each contact."),
    ("impratio, condim", "<code>impratio</code>: ratio of friction-row to normal-row stiffness in MuJoCo; larger values make friction "
            "stick harder. <code>condim</code>&#160;3: normal force and two friction directions; condim&#160;4 adds torsional friction about "
            "the normal."),
    ("pad spacing", "Distance between neighbouring sphere centres on a sphere pad. The 2, 1 and 0.5&#8202;mm pads have 51, 205 and 819 "
            "spheres on a 45&#176; cap."),
    ("relaxation time \\(t_r\\)", "Drake&#8217;s dissipation time for hydroelastic contact; the pads use \\(t_c=t_r/2\\) so both damp at "
            "the same rate."),
]


def glossary_html(keys=None):
    items = [g for g in GLOSSARY if keys is None or g[0] in keys]
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

def _panel(out, x0, y0, w, h, xs, ys, xt, yt, xlab, ylab, logx=False, logy=False, xfmt="{:g}", yfmt="{:g}"):
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


def _svg_open(w, h, label):
    return [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{label}" font-family="var(--f-mono)" font-size="12">']


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
    W, H = 990, 250
    out = _svg_open(W, H, "Five fingertip contact representations: point contact, condim 4 with a torsional coefficient, "
                          "a pad of small spheres, a hydroelastic pressure field over the overlap, and a lattice with "
                          "lateral springs.")
    out.append('<defs><marker id="ovar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
               'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--ink2)"/></marker></defs>')
    titles = ["(a) point contact", "(b) condim 4, μt(N)", "(c) sphere pad", "(d) hydroelastic field",
              "(e) sphere lattice (CSLC)"]
    notes = [("no torque about n", "one Coulomb cone"),
             ("τ ≤ μt N", "μt = μ c N¹ᐟ⁴, each step"),
             ("a soft contact per sphere", "T = Σ ρᵢ fₜᵢ"),
             ("p = E·δ/R", "integrated over the surface"),
             ("anchor + lateral springs", "spreads past the overlap")]
    Rr, yl = 62.0, 150.0                     # fingertip radius (px) and tool surface
    for i in range(5):
        x0 = 8 + i * 197
        cx = x0 + 96
        soft = i >= 2
        cy = yl - Rr + (7.0 if soft else 0.0)
        out.append(f'<text x="{cx}" y="22" text-anchor="middle" style="fill:var(--ink);font:600 13px var(--f-display)">{titles[i]}</text>')
        # tool
        out.append(f'<rect x="{x0 + 6}" y="{yl}" width="180" height="34" rx="3" style="fill:var(--sunk);stroke:var(--ink3);stroke-width:1"/>')
        out.append(f'<text x="{x0 + 176}" y="{yl + 22}" text-anchor="end" style="fill:var(--ink3);font-size:11px">tool</text>')
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
        out.append(f'<text x="{cx}" y="{yl + 56}" text-anchor="middle" style="fill:var(--ink2);font-size:11.5px">{notes[i][0]}</text>'
                   f'<text x="{cx}" y="{yl + 73}" text-anchor="middle" style="fill:var(--ink3);font-size:11.5px">{notes[i][1]}</text>')
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
    W, H = 980, 420
    out = _svg_open(W, H, "Friction arm against pad force on log axes: the foundation law, a Hertz law of the same value at "
                          "1 N, and the arm each simulator delivers.")
    fx, fy = _panel(out, 80, 46, 560, 300, (0.2, 5.0), (0.5, 4.2), (0.25, 0.5, 1, 2, 4), (0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0),
                    "pad force N per pad (N), log scale", "friction arm r̄ per pad (mm), log scale", True, True)
    Ns = [0.2 * (25 ** (k / 60)) for k in range(61)]
    _path(out, fx, fy, [(N, C_LAW * 1e3 * N ** EXP_LAW) for N in Ns], "var(--c-ref)")
    _path(out, fx, fy, [(N, C_LAW * 1e3 * N ** (1 / 3)) for N in Ns], "var(--c-ref)", dashed=True)
    out.append(f'<text x="{fx(0.36):.1f}" y="{fy(C_LAW * 1e3 * 0.36 ** EXP_LAW) - 12:.1f}" text-anchor="middle" style="fill:var(--ink3)">'
               f'foundation, c N¹ᐟ⁴</text>'
               f'<text x="{fx(4.4):.1f}" y="{fy(C_LAW * 1e3 * 4.4 ** (1 / 3)) - 9:.1f}" text-anchor="end" style="fill:var(--ink3)">'
               f'Hertz, N¹ᐟ³</text>')
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
    x0 = 720
    out.append(f'<text x="{x0}" y="58" style="fill:var(--ink2)">arm ratio, 3 N over 0.5 N</text>')
    rows = [("foundation 6¹ᐟ⁴", 6 ** 0.25, "var(--c-ref)"), ("Hertz 6¹ᐟ³", 6 ** (1 / 3), "var(--c-ref)")]
    for k in ORDER:
        lo = pick(tw, k, N=0.5, dt_ms=1.0)
        hi = pick(tw, k, N=3.0, dt_ms=1.0)
        if lo and hi and twist_arm(lo) and twist_arm(hi) and 1.3 <= twist_arm(hi) / twist_arm(lo) <= 2.0:
            rows.append((MODELS[k][0].replace("MuJoCo ", ""), twist_arm(hi) / twist_arm(lo), MODELS[k][1]))
    bx = lambda v: x0 + (v - 1.3) / (2.0 - 1.3) * 220  # noqa: E731
    for j, (lab, val, col) in enumerate(rows):
        y = 84 + j * 34
        out.append(f'<line x1="{x0}" x2="{x0 + 220}" y1="{y + 8}" y2="{y + 8}" style="stroke:var(--rule2)"/>'
                   f'<text x="{x0}" y="{y}" style="fill:var(--ink3);font-size:11.5px">{lab}</text>'
                   f'<circle cx="{bx(val):.1f}" cy="{y + 8}" r="4.5" style="fill:{col};stroke:var(--card);stroke-width:2"/>'
                   f'<text x="{bx(val) + 9:.1f}" y="{y + 12}" style="fill:var(--ink2);font-size:11.5px">{val:.3f}</text>')
    for v in (1.4, 1.6, 1.8, 2.0):
        out.append(f'<text x="{bx(v):.1f}" y="{84 + len(rows) * 34 + 8}" text-anchor="middle" style="fill:var(--ink3);font-size:11px">{v:g}</text>')
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
                          "hydroelastic law.")
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
    return (f"Calibrated with the nominal tool and reused for a tool four times heavier, each sphere comes out "
            f"{k4:.2f}&#215; too stiff, and for a tool four times lighter {k025:.2f}&#215;. Computing \\(\\hat\\Lambda\\) from the compiled "
            f"model for each pad&#8211;object pair at load, and \\(d_0\\) from it, removes the dependence at no run-time cost. "
            f"In MuJoCo&#160;3.6 \\(\\hat\\Lambda\\) is the sum of the bodies&#8217; <code>body_invweight0</code>, which is independent of "
            f"pose, so the load-time value is exact. With it the arm is {_rng(lo)}&#8202;mm at 0.5&#8202;N and "
            f"{_rng(hi)}&#8202;mm at 3&#8202;N for all three masses, {abs(statistics.mean(lo) / lawlo - 1) * 100:.0f} and "
            f"{abs(statistics.mean(hi) / lawhi - 1) * 100:.0f}&#8202;% under the hydroelastic law; with the fixed value it moves by "
            f"{num(d025, '+.0f')} and {num(d4, '+.0f')}&#8202;% (Figure&#160;5). The chain&#8217;s plant already calibrates each trial this way, "
            f"so the per-step remapping of every contact that Codex built is unnecessary.")


# ------------------------------------------------------------------------------------------ creep and stability text

def creep_text(pull, creep):
    """Prediction (12) against the measured T1 creep of the 1 mm pad and point contact, plus T6 if present."""
    out = []
    pads = [r for r in pull if r.get("model") == "mj_pads1" and _eq(r.get("dt_ms"), 1.0)]
    if pads:
        d0, tc, lam, kap = 0.741, 0.01, 47.2, 100.0
        cells = []
        for r in sorted(pads, key=lambda r: r["N"]):
            n = (r.get("settle") or {}).get("n_L", 0) + (r.get("settle") or {}).get("n_R", 0)
            if not n:
                continue
            pred = (1 - d0) * lam * tc / (2 * kap * n) * r["F_hold"] * 1e3
            cells.append(f"{r['N']:g}&#8202;N: {pred:.3f} predicted, {r['creep_mm_s']:.3f}&#8202;mm/s measured ({n} spheres)")
        if cells:
            out.append("For the 1&#8202;mm pad (\\(d_0\\)&#8202;=&#8202;0.741, \\(t_c\\)&#8202;=&#8202;10&#8202;ms, \\(\\kappa\\)&#8202;=&#8202;100) held at half its slip "
                       "force in bed task&#160;1, (12) gives " + "; ".join(cells) + ".")
    dk = [r for r in pull if r.get("model") == "drake_hydro" and _eq(r.get("dt_ms"), 1.0)]
    mj = [r for r in pull if r.get("model") in ("mj_pads1", "mj_point3") and _eq(r.get("dt_ms"), 1.0)]
    if dk and mj:
        ratio = statistics.median(r["creep_mm_s"] for r in mj) / max(statistics.median(r["creep_mm_s"] for r in dk), 1e-12)
        out.append(f"Drake&#8217;s SAP solver holds the same load with about {ratio:.0f}&#215; less creep "
                   f"({min(r['creep_mm_s'] for r in dk) * 1e3:.1f}&#8211;{max(r['creep_mm_s'] for r in dk) * 1e3:.1f}&#8202;&#181;m/s), "
                   "so creep is the largest disagreement between MuJoCo and the references in the bed.")
    if creep:
        out.append(creep_t6_text(creep))
    else:
        out.append("By (12), creep falls with a larger <code>impratio</code> and with more rows in contact; MuJoCo&#8217;s "
                   "<code>noslip_iterations</code> removes residual slip after the main solve. Bed task&#160;6, which tests both, has not run.")
    return " ".join(out)


def creep_t6_text(rows):
    """Summary of T6 (filled once the rows exist); written generically over the fields the bed script records."""
    try:
        base = [r for r in rows if r.get("noslip_iterations", 0) == 0 and _eq(r.get("impratio", 100), 100)]
        ns = [r for r in rows if r.get("noslip_iterations", 0) > 0]
        hi = [r for r in rows if r.get("noslip_iterations", 0) == 0 and r.get("impratio", 0) >= 1000]
        def med(rs):
            v = [first(r, "creep_mm_s", "creep_rate_mm_s") for r in rs]
            v = [x for x in v if x is not None]
            return statistics.median(v) if v else None
        b, n, h = med(base), med(ns), med(hi)
        parts = []
        if b and h:
            parts.append(f"Raising <code>impratio</code> from 100 to 1000 cuts the median creep of bed task&#160;6 from {b * 1e3:.1f} "
                         f"to {h * 1e3:.1f}&#8202;&#181;m/s")
        if b and n is not None:
            parts.append(f"ten <code>noslip_iterations</code> take it to {n * 1e3:.2f}&#8202;&#181;m/s")
        return ("; ".join(parts) + ".") if parts else ""
    except Exception:
        return ""


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
    """Bound (14) for the summary row: n = spheres in contact on one pad at the end of the settle."""
    n = r.get("n_L_ref") or 0
    return r["tc_ms"] * (r["d0"] + (1 - r["d0"]) / n) if n else None


def svg_stab(rows):
    S = [r for r in stab_summaries(rows, model="mj_pads1") if stab_bound(r)]
    if not S:
        return pending("Bed task&#160;7 rows (<code>stability.jsonl</code>) are not written yet.")
    W, H = 980, 430
    out = _svg_open(W, H, "Largest stable step against the collective-damping bound (14) for the 1 mm pad at three time constants "
                          "and seven impedances; each bar spans the largest step that held and the first that failed, and the "
                          "diagonal passes through 19 of the 21 bars.")
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
    """Counts and values for the stability text: pads with the clamp off against (14), the exceptions, point contact,
    and the pads with MuJoCo's default clamp."""
    pads = [r for r in stab_summaries(rows, model="mj_pads1") if stab_bound(r)]
    if not pads:
        return None
    inside, outside = [], []
    for r in pads:
        b, held, ff = stab_bound(r), r.get("dt_max_held_ms") or 0.0, r.get("dt_first_fail_ms")
        (inside if held <= b * 1.0001 and (ff is None or ff > b) else outside).append(r)
    pt = sorted(stab_summaries(rows, model="mj_point3"), key=lambda r: r["d0"])
    on = stab_summaries(rows, refsafe=True, model="mj_pads1")
    stiff = [r for r in on if r["d0"] >= 0.5]
    soft = [r for r in on if r["d0"] < 0.5]
    return dict(n=len(pads), inside=inside, outside=outside, point=pt, stiff=stiff, soft=soft,
                d0=sorted({r["d0"] for r in pads}), tc=sorted({r["tc_ms"] for r in pads}))


def _lst(v, unit=""):
    v = [f"{x:g}" for x in v]
    return (", ".join(v[:-1]) + " and " + v[-1] if len(v) > 1 else v[0]) + unit


def stab_text(rows):
    st = stab_stats(rows)
    if not st:
        return ""
    out = [f"Bed task&#160;7 holds the tool against gravity at 1&#8202;N per pad for 1&#8202;s with \\(d_0\\) from {st['d0'][0]:g} to "
           f"{st['d0'][-1]:g}, \\(t_c\\) of {_lst(st['tc'], '&#8202;ms')} and steps from 0.25 to 15&#8202;ms, with MuJoCo&#8217;s clamp "
           f"switched off so that \\(t_c\\) stays as set. For {len(st['inside'])} of the {st['n']} pairs the largest step that held and "
           f"the first that failed bracket (14) (Figure&#160;6)."]
    if st["outside"]:
        ex = sorted(st["outside"], key=lambda r: (r["tc_ms"], r["d0"]))
        kinds = {stab_fail_kind(r) for r in ex}
        how = "the tool is ejected" if kinds == {"ejected"} else "the run fails"
        out.append(f"In the other {len(ex)}, \\(d_0\\)&#8202;=&#8202;{_lst([r['d0'] for r in ex])} at \\(t_c\\)&#8202;=&#8202;"
                   f"{_lst(sorted({r['tc_ms'] for r in ex}), '&#8202;ms')}, {how} at {_lst(sorted({r['dt_first_fail_ms'] for r in ex}), '&#8202;ms')}, "
                   f"below bounds of {min(stab_bound(r) for r in ex):.0f}&#8211;{max(stab_bound(r) for r in ex):.0f}&#8202;ms.")
    if st["point"]:
        pt = st["point"]
        out.append(f"Point contact, one row per pad, fails at {_lst([r['dt_first_fail_ms'] for r in pt], '&#8202;ms')} for \\(d_0\\) "
                   f"{_lst([r['d0'] for r in pt])} at \\(t_c\\)&#8202;=&#8202;{pt[0]['tc_ms']:g}&#8202;ms: at the first tested step above "
                   f"\\(d_0t_c\\), well below the \\(t_c\\) that (14) gives for \\(n\\)&#8202;=&#8202;1.")
    if st["stiff"] and st["soft"]:
        out.append(f"With the clamp on, as MuJoCo runs by default, the pads with \\(d_0\\)&#8202;&#8805;&#8202;0.5 hold at every step up to "
                   f"{min(r['dt_max_held_ms'] for r in st['stiff']):g}&#8202;ms at all three \\(t_c\\), and those with \\(d_0\\) "
                   f"{_lst(sorted({r['d0'] for r in st['soft']}))} fail from {min(r['dt_first_fail_ms'] for r in st['soft']):g} to "
                   f"{max(r['dt_first_fail_ms'] for r in st['soft']):g}&#8202;ms, where the clamped bound "
                   f"\\(2\\Delta t\\,(d_0+(1-d_0)/n)\\) falls below \\(\\Delta t\\).")
    return " ".join(out)


# ------------------------------------------------------------------------------------------ Figure 7: step sweep

def svg_step(rows):
    if not rows:
        return pending("Codex&#8217;s task_timestep_results.json is missing.")
    W, H = 980, 400
    out = _svg_open(W, H, "Left: the tool angle at the end of the twist sequence stays within a few degrees for the sphere pad "
                          "and Drake from 50 microseconds to 10 milliseconds; the per-contact mapping diverges at every step "
                          "from 0.5 ms. Right: wall time per simulated second.")
    series = {"legacy": ("1 mm sphere pad", "var(--c-sphere)", False, "circle"),
              "drake": ("Drake hydroelastic", "var(--c-drake)", False, "circle"),
              "runtime_exact": ("per-contact mapping, runtime, exact Λ", "var(--c-c4)", True, "square"),
              "physical": ("per-contact mapping, fixed", "var(--ink3)", True, "square")}
    fx, fy = _panel(out, 70, 46, 380, 280, (0.04, 13.0), (-72.0, -40.0), (0.05, 0.1, 0.5, 1, 2, 5, 10), (-70, -60, -50, -40),
                    "physics step (ms), log scale", "(a) tool angle about the pinch axis at the end (°)", True, False)
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
                    "physics step (ms), log scale", "(b) physics wall s per simulated s, one core", True, True)
    for key, (lab, col, dashed, shape) in series.items():
        rs = sorted([r for r in rows if r["case"]["label"] == key and r["status"] == "complete"], key=lambda r: r["case"]["physics_dt"])
        pts = [(r["case"]["physics_dt"] * 1e3, (r.get("timing") or {}).get("physics_seconds_per_sim_second")) for r in rs]
        _path(out, fx, fy, pts, col, dashed=dashed, width=1.6)
        for x, y in pts:
            if y:
                _marker(out, fx(x), fy(y), col, shape, title=f"{lab}, {x:g} ms: {y:.3f} s per simulated s")
    out.append("</svg>")
    return "".join(out) + _legend_html([(v[0], v[1], v[2], v[3]) for v in series.values()])


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
    return (f"Codex ran the paper&#8217;s Exp&#160;2 twist sequence with a 20&#8202;ms controller and physics steps from 50&#8202;&#181;s to 10&#8202;ms. "
            f"The 1&#8202;mm pad ends the sequence at {num(min(a), '.1f')} to {num(max(a), '.1f')}&#176; about the pinch axis over all six steps, "
            f"Drake hydroelastic at {num(min(b), '.1f')} to {num(max(b), '.1f')}&#176;. Both include the same ratchet toward hanging that "
            f"the chain page traced to free pad spin. The four per-contact mappings diverged in {len(mfail)} of {len(mapped)} runs, "
            f"every step from 0.5&#8202;ms up, and the {len(mok)} that finished at 50&#8202;&#181;s ended at "
            f"{num(min(r['pinch_angle_end_deg'] for r in mok), '.0f')} to {num(max(r['pinch_angle_end_deg'] for r in mok), '.0f')}&#176;. "
            f"At equal step the pad costs {cost(dk, 0.001) / cost(leg, 0.001):.0f}&#215; less than Drake at 1&#8202;ms and "
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
                          "world-step (hollow).")
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
    ys = spread([fy(d) for _, d, _ in pts], 14.0)
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
        out.append(f'<text x="{xr:.1f}" y="{y + 4:.1f}" style="fill:{col}">{lab}, {d:.1f} %</text>')
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


def svg_gpu(curves):
    if not curves:
        return pending("GPU rows missing.")
    W, H = 980, 420
    out = _svg_open(W, H, "World-steps per second against the number of parallel worlds on one GPU, for the 1 mm sphere pads "
                          "in MuJoCo-Warp and Newton and for Newton's hydroelastic tip.")
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

def cap_table(cost, batched=None):
    batched = batched or {}

    def us(k, gpu=None):
        """Cost cell: one CPU core, or one GPU world and the batched cost per world-step (gpu = the GPU model key)."""
        parts = []
        if cost.get(k):
            parts.append(f"{num(cost[k], ',.0f')} {'one GPU world' if k in GPU_MODELS else 'CPU core'}")
        g = gpu or (k if k in GPU_MODELS else None)
        if g and cost.get(g) and g != k:
            parts.append(f"{num(cost[g], ',.0f')} one GPU world")
        if g and g in batched:
            b, n = batched[g]
            parts.append(f"{b:.2f} batched ({num(n, ',d')} worlds)")
        return "<br>".join(parts) or "&#8211;"
    Y, N_, P = "cap-y", "cap-n", "cap-p"
    rows = [
        ("point contact, condim&#160;3", [(Y, "one spring"), (P, "creeps"), (N_, "none"), (N_, "one point"), (N_, "no"), (N_, "no"), us("mj_point3"), (Y, "MuJoCo-Warp")]),
        ("condim&#160;4, \\(\\mu_t\\) rescheduled", [(Y, "one spring"), (P, "creeps"), (P, "fitted law, rescheduled"), (N_, "one point"), (N_, "no"), (N_, "no"), us("mj_point4s"), (P, "needs a per-step write")]),
        ("1&#8202;mm sphere pad", [(Y, "sampled foundation"), (P, "creeps"), (Y, "from the sphere spread"), (Y, "sampled"), (N_, "no"), (N_, "no"), us("mj_pads1", "mjw_pads1"), (Y, "MuJoCo-Warp, Newton")]),
        ("Drake hydroelastic", [(Y, "pressure field"), (Y, "creeps 100&#215; less"), (Y, "integrated"), (Y, "surface mesh"), (N_, "no"), (N_, "no"), us("drake_hydro"), (N_, "CPU only")]),
        ("Newton hydroelastic, \\(k_h/m_\\text{eff}\\)", [(Y, "pressure field"), (P, "MuJoCo-Warp rows"), (P, "0.76&#8211;0.88 of Drake"), (Y, "voxel surface"), (N_, "no"), (N_, "no"),
                                 us("newton_hydro_mc"), (Y, "Warp")]),
        ("Sphere lattice, CSLC", [(Y, "anchor springs"), (P, "presliding; no sliding dynamics"), (Y, "from the lattice"), (Y, "lattice"), (P, "presliding"), (Y, "lateral springs"), "&#8211;", (P, "&#8211;")]),
    ]
    head = ["contact model", "compliance", "stick, slip", "friction torque", "area, CoP",
            "pre-slip shear", "lateral spread", f"cost ({US} per step)", "GPU batch"]
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


def _url(path):
    return open(path).read().strip() if os.path.exists(path) else None


def related():
    items = [
        ("Servo plant refit, hand-object control of the three-finger turn, contact-model agreement, effective-mass scaling "
         "and the cost of sphere-pad fingertips in RL training", HT3_PATH, _url(os.path.join(HT3, "artifact_url.txt"))),
        ("Previous revision of this page (2026-10-05)", "docs/experiments/20261005-contact_overview/20261005-sphere_pad_contact_model.html",
         None),
        ("Chain control on the SR2 hand (derivation, eight contact models, the paper&#8217;s tasks, cost)", CHAIN_PATH, CHAIN_URL),
        ("Pinch friction torque in Drake and MuJoCo contact models (two-pad rig)",
         "docs/experiments/20261001-hom_contact_patch/20261001-hom_pinch_contact_models.html", "https://claude.ai/artifact/Rvfw1yQFgfWV8PdTvv7jsc"),
        ("Pinch-brake loading on the SR2 hand", "docs/experiments/20261001-hom_hand_brake/20261001-hom_hand_brake.html",
         "https://claude.ai/artifact/DJLxG63vLZXs2Ey6hiCBuc"),
        ("Distributed fingertip contact audit (Codex)", "docs/experiments/20261004-codex/20261004-distributed_contact.html", None),
        ("Articulated holding, torsion, GPU and Newton follow-up (Codex)", "docs/experiments/20261004-codex/20261004-articulated_contact_transfer.html", None),
        ("50&#8202;Hz controller and physics-step sweep (Codex)", "docs/experiments/20261004-codex/20261004-controller_timestep.html", None),
        ("Research asides: coupled-foundation models for the printed tip", ASIDES_PATH, None),
        ("Video gallery (Codex)", "docs/experiments/20261004-codex/20261004-contact_videos.html", None),
    ]
    url = bed_url()
    items.insert(0, ("Contact comparison bed: tasks, films and numbers", BED_PATH, url))
    out = []
    for name, path, u in items:
        link = f", <a href=\"{u}\">artifact</a>" if u else " (local)"
        out.append(f"{name}: <code>{path}</code>{link}")
    return "<br>".join(out)


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


def main():
    import contact_overview_text as OT  # prose blocks that depend on the literature notes and the bed findings
    os.makedirs(D, exist_ok=True)
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

    t = open(TPL).read()
    v = {"STYLE": style_block(), "BUILT": time.strftime("%Y-%m-%d %H:%M"), "CHAIN_PATH": CHAIN_PATH, "CHAIN_URL": CHAIN_URL,
         "BED_PATH": BED_PATH, "TEX_CACHE_REL": os.path.relpath(TEX_CACHE, ROOT)}
    u = bed_url()
    v["BED_LINK"] = f", <a href=\"{u}\">artifact</a>" if u else ""
    hi, lo = G.compute(3.0), G.compute(0.5)
    dev = lambda res, s: (res[s]["arm"] / res["field"]["arm"] - 1) * 100  # noqa: E731
    v["LO_N2"] = str(lo[0.002]["n_in"])
    v["LO_DEV2"] = f"{dev(lo, 0.002):.0f}"
    v["LO_DEV_FINE"] = f"{max(abs(dev(lo, s)) for s in (0.001, 0.0005)):.0f}"
    v["FIG_SECTION"] = G.svg_section(hi)
    v["FIG_PATCH"] = G.svg_patch(hi, lo)
    v["FIG_MODELS"] = svg_models()
    v["CAP_TABLE"] = cap_table(cost, batched)
    fig, have_onset = svg_scaling(T["twist"], tor)
    v["FIG_SCALING"] = fig
    v["FIG_MASS"] = svg_mass(cal)
    v["MASS_TEXT"] = mass_text(cal)
    v["CREEP_TEXT"] = creep_text(T["pull"], T["creep"])
    v["FIG_STAB"] = svg_stab(stab)
    v["STAB_TEXT"] = stab_text(stab)
    v["FIG_STEP"] = svg_step(step_rows)
    v["STEP_TEXT"] = step_text(step_rows)
    v["AGREE_TABLE"] = agree_table(M)
    v["FIG_COST"] = svg_cost(M, cost, batched)
    v["FIG_GPU"] = svg_gpu(curves)
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
    v["GLOSSARY"] = glossary_html()
    ctx = dict(T=T, M=M, cost=cost, cal=cal, stab=stab, step_rows=step_rows, gpu_pads=gpu_pads, gpu_newton=gpu_newton,
               tor=tor, have_onset=have_onset, curves=curves, batched=batched, deviations=deviations(M), bed_url=u)
    v.update(OT.blocks(ctx))
    for k, val in v.items():
        t = t.replace("{{" + k + "}}", val)
    left = sorted(set(x.split("}}")[0] for x in t.split("{{")[1:]))
    if left:
        raise SystemExit(f"unfilled placeholders: {left}")
    t, n_tex = render_tex(t)
    open(OUT, "w").write(t)
    print(f"formulas {n_tex}; wrote {OUT} ({os.path.getsize(OUT) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
