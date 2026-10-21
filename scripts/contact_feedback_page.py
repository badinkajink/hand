#!/usr/bin/env python3
"""Result page: sensitivity of the servo-plant reorientation policies to fingertip contact parameters, with analytical
checks of the sphere pads (2026-10-13).

Rows: docs/experiments/20261013-contact_feedback/replay.jsonl and census.jsonl (scripts/contact_feedback_replay.py),
pad_laws.json (scripts/pad_contact_rig.py), media/20261013-closed_open.{mp4,jpg,json} (contact_feedback_replay.py --film).
Every number in the text is computed here from those files.

    .venv/bin/python scripts/contact_feedback_page.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import contact_overview_page as P  # noqa: E402
import retro_style  # noqa: E402
import texsvg  # noqa: E402

ROOT = P.ROOT
D = os.path.join(ROOT, "docs/experiments/20261013-contact_feedback")
OUT = os.path.join(D, "20261013-contact_parameter_sensitivity.html")
TPL = os.path.join(ROOT, "scripts/contact_feedback_page.template.html")
TEX_CACHE = os.path.join(D, "texsvg_cache.json")
SERVO_PAGE = "../20261011-servo_plant_policies/20261011-servo_plant_contact_model_policies.html"
SKIN_PAGE = "../20261012-sprung_skin/20261012-sprung_skin_presliding.html"
TIP_PAGE = "../20261009-tpu_tip_fem/20261009-printed_tpu_fingertip_finite_elements.html"

POL = [("tpu27mesh", 0), ("tpu27mesh", 1), ("tpu27pads1", 0), ("tpu27pads1", 1), ("tpu27skin", 0), ("tpu27skin", 1)]
TAG = {("tpu27mesh", s): f"20261011-d6_servo_tpu27mesh_40M_s{s}" for s in (0, 1)} | \
      {("tpu27pads1", s): f"20261011-d6_servo_tpu27pads1_40M_s{s}" for s in (0, 1)} | \
      {("tpu27skin", s): f"20261012-d6_servo_tpu27skin_40M_s{s}" for s in (0, 1)}
COL = {"box": "#48545D", "tpu27mesh": "#2a78d6", "tpu27pads1": "#eb6834", "tpu27skin": "#1baf7a"}   # servo-plant page
ARM = {"tpu27mesh": "TPU mesh", "tpu27pads1": "Pads", "tpu27skin": "Skin", "box": "Box tip"}
TIP = [("tpu27mesh", "TPU mesh"), ("tpu27pads1", "1 mm pads"), ("tpu27skin", "Sprung skin"), ("box", "Box tip")]
PHYS = [("friction=0.5", "Friction &#215;0.5"), ("friction=0.7", "Friction &#215;0.7"), ("friction=1.3", "Friction &#215;1.3"),
        ("mass=0.8", "Tool mass &#215;0.8"), ("mass=1.2", "Tool mass &#215;1.2"), ("mass=2", "Tool mass &#215;2"),
        ("kp=0.36", "Servo gain 0.36&#160;N&#160;m/rad"), ("fr=0.12", "Torque limit 0.12&#160;N&#160;m")]
N = 64


class Data:
    def __init__(self):
        self.rows = {}
        for line in open(os.path.join(D, "replay.jsonl")):
            r = json.loads(line)
            if r.get("status") == "ok":
                self.rows[(r["tag"], r["scene"], r["mode"], r.get("perturb", "nominal"))] = r
        self.laws = json.load(open(os.path.join(D, "pad_laws.json")))
        cp = os.path.join(D, "census.jsonl")
        self.census = {json.loads(x)["tag"]: json.loads(x) for x in open(cp)} if os.path.exists(cp) else {}
        fp = os.path.join(D, "media/20261013-closed_open.json")
        self.film = json.load(open(fp)) if os.path.exists(fp) else None

    def cell(self, arm, seed, group, cond, mode):
        tag = TAG[(arm, seed)]
        key = (tag, cond, mode, "nominal") if group == "tip" else (tag, arm, mode, cond)
        return self.rows.get(key)


def pname(a, sd):
    return f"{'TPU-mesh' if a == 'tpu27mesh' else ARM[a].lower()} seed {sd}"


def hc(r):
    return r["held_cos_mean"] if r and r["n_held"] else None


# ------------------------------------------------------------------------------------------ aggregates

def tip_stats(X: Data) -> dict:
    """Closed loop against open-loop replay under the four tip models (24 cells)."""
    c_h = o_h = n = 0
    d, own = [], []
    for arm, s in POL:
        for scene, _ in TIP:
            c, o = X.cell(arm, s, "tip", scene, "closed"), X.cell(arm, s, "tip", scene, "open")
            if not (c and o):
                continue
            c_h, o_h, n = c_h + c["n_held"], o_h + o["n_held"], n + N
            if hc(c) is not None and hc(o) is not None:
                d.append(abs(hc(o) - hc(c)))
                if scene == arm:
                    own.append((c["n_held"], o["n_held"], abs(hc(o) - hc(c))))
    return dict(c_h=c_h, o_h=o_h, n=n, d_max=max(d), d_mean=float(np.mean(d)), own=own,
                own_same_held=all(a == b for a, b, _ in own), own_d_max=max(x for _, _, x in own))


def phys_stats(X: Data) -> dict:
    out = {}
    for cond, _ in PHYS:
        cs = [X.cell(a, s, "phys", cond, "closed") for a, s in POL]
        os_ = [X.cell(a, s, "phys", cond, "open") for a, s in POL]
        if not all(cs) or not all(os_):
            continue
        dc = [(hc(o) - hc(c), a, s) for (a, s), c, o in zip(POL, cs, os_) if hc(c) is not None and hc(o) is not None]
        out[cond] = dict(c_h=sum(c["n_held"] for c in cs), o_h=sum(o["n_held"] for o in os_), n=N * len(cs),
                         worst=min(dc) if dc else None,
                         cells=[(a, s, c["n_held"], o["n_held"], hc(c), hc(o)) for (a, s), c, o in zip(POL, cs, os_)])
    return out


# ------------------------------------------------------------------------------------------ figure 1

def svg_feedback(X: Data) -> str:
    W_, FS = 828, 14
    LAB, PA0, PA1, PB0, PB1 = 8, 214, 462, 506, 812
    LANE, ROWGAP, TOP = 8, 9, 78
    CLO, CHI = 0.4, 1.0
    xa = lambda v: PA0 + (PA1 - PA0) * v  # noqa: E731
    xb = lambda v: PB0 + (PB1 - PB0) * (min(max(v, CLO), CHI) - CLO) / (CHI - CLO)  # noqa: E731
    out, marks, heads, y = [], [], [], TOP
    for gname, group, conds in (("Tip model, nominal plant", "tip", TIP),
                                ("Physical change, each policy on its own tip", "phys", PHYS)):
        y += 6
        heads.append(f'<rect x="{LAB}" y="{y}" width="{PB1 - LAB}" height="21" fill="#fff"/>'
                     f'<text x="{LAB}" y="{y + 15}" font-size="{FS}" font-weight="bold" fill="#000">{gname}</text>')
        y += 26
        for cond, label in conds:
            h = LANE * len(POL)
            out.append(f'<text x="{LAB + 10}" y="{y + h / 2 + 5:.1f}" font-size="{FS}" fill="#000">{label}</text>')
            for gx0, gx1 in ((PA0, PA1), (PB0, PB1)):
                out.append(f'<rect x="{gx0}" y="{y - 3}" width="{gx1 - gx0}" height="{h + 6}" fill="#f4f4f2"/>')
            for k, (arm, s) in enumerate(POL):
                ly = y + LANE * k + LANE / 2
                c, o = X.cell(arm, s, group, cond, "closed"), X.cell(arm, s, group, cond, "open")
                if not (c and o):
                    continue
                own = " (its own tip)" if group == "tip" and cond == arm else ""
                tip = lambda r, mode: (f"{ARM[arm]} seed {s}, {label.replace('&#215;', 'x')}{own}, {mode}: held "  # noqa: E731
                                       f"{r['n_held']}/{N}" + (f", held cosine {hc(r):.3f}" if hc(r) is not None else ", none held"))
                for fx, vc, vo in ((xa, c["n_held"] / N, o["n_held"] / N), (xb, hc(c), hc(o))):
                    if vc is not None and vo is not None and abs(fx(vc) - fx(vo)) > 0.5:
                        x0, x1 = sorted((fx(vc), fx(vo)))
                        out.append(f'<line x1="{x0:.1f}" x2="{x1:.1f}" y1="{ly:.1f}" y2="{ly:.1f}" stroke="{COL[arm]}" '
                                   f'stroke-width="2.5" stroke-opacity="0.55" stroke-linecap="round"/>')
                    for v, mode, r in ((vo, "open-loop replay", o), (vc, "closed loop", c)):
                        if v is not None:
                            marks.append((mode, fx(v), ly, COL[arm], tip(r, mode), v if fx is xb else None))
            y += h + ROWGAP
    for mode, x, ly, col, tip, v in marks:          # rings first, filled dots on top
        if v is not None and v < CLO:
            fill = col if mode == "closed loop" else "#fff"
            out.append(f'<path d="M{PB0 - 1:.1f},{ly:.1f} l7,-4.5 l0,9 z" fill="{fill}" stroke="{col}" stroke-width="1.5">'
                       f'<title>{tip}</title></path><text x="{PB0 + 10:.1f}" y="{ly + 4.5:.1f}" font-size="13" '
                       f'fill="#000">{v:.2f}</text>')
        elif mode == "closed loop":
            out.append(f'<circle cx="{x:.1f}" cy="{ly:.1f}" r="3.8" fill="{col}" stroke="#fff" stroke-width="1">'
                       f'<title>{tip}</title></circle>')
        else:
            out.append(f'<circle cx="{x:.1f}" cy="{ly:.1f}" r="3.3" fill="#fff" stroke="{col}" stroke-width="1.7">'
                       f'<title>{tip}</title></circle>')
    bottom = y + 2
    ax = []
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        ax.append(f'<line x1="{xa(v):.1f}" x2="{xa(v):.1f}" y1="{TOP - 4}" y2="{bottom}" stroke="#d9d9d9" stroke-width="1"/>'
                  f'<text x="{xa(v):.1f}" y="{bottom + 18}" font-size="{FS}" text-anchor="middle" fill="#000">{v:g}</text>')
    for v in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        ax.append(f'<line x1="{xb(v):.1f}" x2="{xb(v):.1f}" y1="{TOP - 4}" y2="{bottom}" stroke="#d9d9d9" stroke-width="1"/>'
                  f'<text x="{xb(v):.1f}" y="{bottom + 18}" font-size="{FS}" text-anchor="middle" fill="#000">{v:g}</text>')
    ax.append(f'<text x="{(PA0 + PA1) / 2}" y="{TOP - 12}" font-size="{FS}" font-weight="bold" text-anchor="middle" '
              f'fill="#000">Held share</text><text x="{(PB0 + PB1) / 2}" y="{TOP - 12}" font-size="{FS}" font-weight="bold" '
              f'text-anchor="middle" fill="#000">Held cosine</text>')
    ax.append(f'<text x="{(PA0 + PA1) / 2}" y="{bottom + 38}" font-size="{FS}" text-anchor="middle" fill="#444">Share of '
              f'rollouts holding the tool at 5 s</text><text x="{(PB0 + PB1) / 2}" y="{bottom + 38}" font-size="{FS}" '
              f'text-anchor="middle" fill="#444">Mean final cosine of the held rollouts</text>')
    lg, lx = [], LAB
    for a, w in (("tpu27mesh", 108), ("tpu27pads1", 74), ("tpu27skin", 74)):
        lg.append(f'<rect x="{lx}" y="9" width="14" height="14" fill="{COL[a]}"/><text x="{lx + 20}" y="21" font-size="{FS}" '
                  f'fill="#000">{ {"tpu27mesh": "TPU mesh", "tpu27pads1": "Pads", "tpu27skin": "Skin"}[a]}</text>')
        lx += w
    lx += 14
    lg.append(f'<circle cx="{lx + 5}" cy="16" r="4.2" fill="#666" stroke="#fff" stroke-width="1.2"/>'
              f'<text x="{lx + 14}" y="21" font-size="{FS}" fill="#000">Closed loop</text>'
              f'<circle cx="{lx + 112}" cy="16" r="3.8" fill="#fff" stroke="#666" stroke-width="1.8"/>'
              f'<text x="{lx + 121}" y="21" font-size="{FS}" fill="#000">Open-loop replay of its own actions</text>')
    H_ = bottom + 50
    return (f'<svg viewBox="0 0 {W_} {H_}" width="100%" role="img" aria-label="Held share and held cosine of six policies, '
            f'closed loop and open-loop replay, under four tip models and eight physical changes" style="max-width:{W_}px;'
            f'font-family:inherit">' + "".join(lg + ax + heads + out) + "</svg>")


# ------------------------------------------------------------------------------------------ tables

def _table(head, rows, cls="cap"):
    t = [f"<table class='{cls}'><thead><tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr></thead><tbody>"]
    t += ["<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows]
    return "".join(t) + "</tbody></table>"


def table_static(L):
    rows = [(f"{s['F']:g}", s["n_active"], f"{s['k_min']:.1f}", f"{s['approach_m'] * 1e6:.1f}",
             f"{s['approach_law_m'] * 1e6:.1f}", f"{(s['approach_m'] / s['approach_law_m'] - 1) * 100:+.1f}",
             f"{s['twist_Nm'] * 1e3:.2f}", f"{s['twist_law_Nm'] * 1e3:.2f}",
             f"{(s['twist_Nm'] / s['twist_law_Nm'] - 1) * 100:+.1f}") for s in L["static"]]
    return _table(["Load (N)", "Pads touching", "Force over penetration (N/m)", "Approach, pads (&#181;m)",
                   "Approach, law (&#181;m)", "Difference (%)", "Twist limit, pads (mN&#8202;m)", "Twist limit, law (mN&#8202;m)",
                   "Difference (%)"], rows)


def table_friction(L):
    rows = []
    for c in L["creep"]:
        rows.append((f"Tangential 0.5&#8202;&#956;N", f"{c['impratio']:g}", f"creeps {c['v_m_s'] * 1e3:.4f}&#8202;mm/s",
                     f"(2): {c['v_law_m_s'] * 1e3:.4f}&#8202;mm/s"))
    for c in L["creep"]:
        rows.append((f"Tangential 1.2&#8202;&#956;N", f"{c['impratio']:g}", f"slides {c['slide_m_in_0p1s'] * 1e3:.1f}&#8202;mm in 0.1&#8202;s",
                     "above the limit &#956;N: slides"))
    for t in L["twist"]:
        rows.append((f"Twist {t['frac']:g}&#8202;&#964;<sub>max</sub>", "10", f"turns {t['turned_deg_in_0p2s']:.2f}&#176; in 0.2&#8202;s",
                     "below the limit: creeps" if t["frac"] < 1 else "above the limit: spins"))
    return _table(["Load at 1&#8202;N normal", "impratio", "Rig", "Closed form"], rows)


def table_mass(L):
    rows = []
    for m in L["mass"]:
        rows.append((f"{m['tool_mass_kg'] * 1e3:.1f}", f"{m['d0']:.4f}", f"{m['k_rebuilt']:.1f}",
                     f"{m['creep_m_s'] * 1e3:.4f}",
                     f"{m['k_stale']:.0f} ({(m['k_stale'] / L['pad_K'] - 1) * 100:+.0f}&#8202;%)" if "k_stale" in m else "built here"))
    return _table(["Tool mass (g)", "Impedance \\(d\\)", "Force over penetration, pads built for this tool (N/m)",
                   "Creep at 0.5&#8202;&#956;N (mm/s)", "Force over penetration, pads built for 24.5&#8202;g (N/m)"], rows)


def table_modulus(L, k_servo):
    rows = []
    for r in L["modulus"]:
        rows.append((f"{r['E_Pa'] / 1e6:g}", f"{r['pad_K']:.0f}", f"{r['approach_m_1N'] * 1e6:.1f}",
                     " / ".join(f"{r[f'arm_m_{F:g}N'] * 1e3:.2f}" for F in (0.5, 1.0, 1.5)),
                     f"{r['tangent_N_m_1N'] / 1e3:.1f}", f"{r['tangent_N_m_1N'] / k_servo:.0f}"))
    return _table(["E (MPa)", "Pad spring K<sub>s</sub> (N/m)", "Approach at 1&#8202;N (&#181;m)",
                   "Twist arm at 0.5 / 1 / 1.5&#8202;N (mm)", "Contact tangent stiffness at 1&#8202;N (kN/m)",
                   "Over the servo&#8217;s k<sub>tip</sub>"], rows)


# ------------------------------------------------------------------------------------------ text

def glossary():
    G = [
        ("Servo-limited plant", "The finger joints as position servos with gain 0.72&#8202;N&#8202;m/rad and a 0.18&#8202;N&#8202;m torque limit, "
         "the SCS0009&#8217;s strength; the plant of the servo-plant study, whose six final policies are evaluated here."),
        ("Tip models", "Box tip: the legacy rigid box, point contact. TPU mesh: the printed TPU block as one convex mesh, point "
         "contact. Pads (1&#8202;mm pads): the block&#8217;s surface as spheres of 0.75&#8202;mm radius at 1&#8202;mm spacing. Sprung skin: the "
         "pads on a plate sprung to the fingertip. The policies were trained on the TPU mesh, the pads or the skin."),
        ("Held", "A rollout holds the tool at 5&#8202;s when at least two fingertips each press with at least 0.24&#8202;N (the tool&#8217;s "
         "weight) and the tool is above 60&#8202;mm; held share is the fraction of the 64 rollouts that hold."),
        ("Held cosine", "Cosine between the tool&#8217;s axis and the vertical at 5&#8202;s, averaged over the held rollouts; 1 is "
         "upright, and the turn starts at about 0."),
        ("Closed loop", "The policy computes each action at 50&#8202;Hz from its observations: finger joint angles and velocities, the "
         "tool&#8217;s position and orientation relative to the palm, the reference trajectory, its last action. It observes no "
         "contact force."),
        ("Open-loop replay", "The 64 action sequences the policy produced closed loop on its own tip, applied step for step: "
         "rollout k replays sequence k through the same scripted grasp, lift and mapping of actions to servo targets."),
        ("Physical changes", "Friction &#215;s scales every geom&#8217;s sliding coefficient (nominal 1 on the pads); tool mass &#215;s "
         "scales the tool&#8217;s mass and inertia (nominal 16&#8202;g); servo gain sets the finger servos&#8217; gain in N&#8202;m/rad "
         "(nominal 0.72); torque limit sets their torque limit in N&#8202;m (nominal 0.18)."),
        ("\\(K_s\\), \\(a_s\\), \\(h\\)", "A pad&#8217;s spring stiffness in N/m, its share of tip surface (about 1&#8202;mm&#178;) and the "
         "foundation depth (8.5&#8202;mm, half the block&#8217;s depth), (1); 1,200&#8202;N/m at \\(E\\)&#8202;=&#8202;10&#8202;MPa."),
        ("\\(\\hat{A}\\), \\(t_c\\), \\(d\\)", "MuJoCo&#8217;s approximate inverse inertia of a contact row in 1/kg, the sum of the two "
         "bodies&#8217; translational <code>body_invweight0</code> (thumb tip 29.9, 24.5&#8202;g tool 40.7); the solref time constant "
         "(0.01&#8202;s); the solimp impedance, set equal at both ends of its range on the pads."),
        ("Approach", "How far the tool has moved into the fingertip face since first touch, in &#181;m."),
        ("Twist limit", "\\(\\tau_\\text{max}\\), the torque about the contact normal at which every pad slides, (3); the twist "
         "arm is \\(\\tau_\\text{max}/\\mu N\\) in mm."),
        ("Creep", "Steady sliding speed of the tool under a tangential load below the slip force &#956;N, in mm/s, from MuJoCo&#8217;s "
         "regularised friction, (2); rigid Coulomb friction gives zero."),
        ("impratio", "MuJoCo&#8217;s ratio of a contact&#8217;s frictional to normal impedance; the creep speed falls in inverse "
         "proportion. The policies train at 10, the hand-object scale study runs at 100."),
        ("\\(k_\\text{tip}\\)", "A fingertip&#8217;s stiffness along its palmar normal through its three servos, (3), in N/m: "
         "\\(J\\) is the face centre&#8217;s translational Jacobian and \\(K_q\\) the diagonal of servo gains."),
        ("Contact tangent stiffness", "Slope of the normal force against the approach at 1&#8202;N, in kN/m, measured on the rig "
         "between 1 and 1.05&#8202;N."),
    ]
    return '<dl class="glossary">' + "".join(f"<dt>{a}</dt><dd>{b}</dd>" for a, b in G) + "</dl>"


def main():
    X = Data()
    L = X.laws
    T = tip_stats(X)
    PH = phys_stats(X)
    k_servo = float(np.median([f["stiffness_N_m"] for f in L["servo"]["fingers"].values()]))
    kt = {round(r["E_Pa"]): r["tangent_N_m_1N"] for r in L["modulus"]}
    arms = {round(r["E_Pa"]): r["arm_m_1N"] for r in L["modulus"]}
    arm_change = (arms[20000000] / arms[5000000] - 1) * 100
    st = {s["F"]: s for s in L["static"]}
    fr5, kp, fr = PH.get("friction=0.5"), PH.get("kp=0.36"), PH.get("fr=0.12")
    m2 = PH.get("mass=2")
    fr7 = PH.get("friction=0.7")
    v = {"STYLE": P.style_block(), "BUILT": time.strftime("%Y-%m-%d %H:%M"), "MUJOCO": L["mujoco"]}
    v["LEDE"] = (
        f"On the servo-limited plant the reorientation policies need their feedback to cope with friction and servo strength, "
        f"and hardly need it for the fingertip contact model. Replayed open loop on another tip model, a policy&#8217;s own actions "
        f"hold the tool in {T['o_h']:,} of {T['n']:,} rollouts, against {T['c_h']:,} closed loop, and end within {T['d_max']:.2f} of "
        f"the closed loop&#8217;s held cosine. With friction halved the closed loop holds {fr5['c_h']} of {fr5['n']} rollouts and the replay "
        f"{fr5['o_h']}; at half the servo gain {kp['c_h']} against {kp['o_h']}. The pads&#8217; contact stiffness is a physical "
        f"value, {L['pad_K']:,.0f}&#8202;N/m per pad at 10&#8202;MPa for tools of 6 to 98&#8202;g, and a fourfold change of the modulus "
        f"moves the twist arm by {abs(arm_change):.0f}&#8202;%.")
    v["GLOSSARY"] = glossary()
    v["FB_METHOD"] = (
        f"Each of the six final policies of the <a href=\"{SERVO_PAGE}\">servo-plant study</a> ran 64 deterministic rollouts in "
        "MuJoCo-Warp on its own tip, and its 64 action sequences were kept. On each of the four tip models, and on its own tip "
        "under eight physical changes, the policy then ran closed loop, and its kept actions were replayed open loop "
        "(<code>scripts/contact_feedback_replay.py</code>). The replay on a policy&#8217;s own tip is the control: it reproduces "
        f"the closed loop&#8217;s held count in {'all six' if T['own_same_held'] else 'not all six'} policies and its held cosine "
        f"within {T['own_d_max']:.3f}.")
    box_d = [abs(hc(X.cell(a, s, 'tip', 'box', 'open')) - hc(X.cell(a, s, 'tip', 'box', 'closed'))) for a, s in POL]
    v["FB_TIP"] = (
        f"The tip model leaves the closed loop and the replay alike (Figure&#160;1, top). Over the 24 cells of six policies and "
        f"four tip models the closed loop holds {T['c_h']:,} of {T['n']:,} rollouts and the replay {T['o_h']:,}; their held cosines "
        f"differ by {T['d_mean']:.3f} on average and by {T['d_max']:.3f} at most. The box tip, a rigid box with point contact "
        f"and the largest change of contact geometry, moves the replay&#8217;s held cosine by up to {max(box_d):.3f} from the "
        "closed loop&#8217;s, in either direction.")
    hurt = ("friction=0.5", "friction=0.7", "kp=0.36", "fr=0.12")
    lost = {}
    for a, sd in POL:
        lost[(a, sd)] = max(c - o for k in hurt for a_, s_, c, o, _, _ in PH[k]["cells"] if (a_, s_) == (a, sd))
    robust = [pname(a, sd) for (a, sd), x in lost.items()
              if all(o == N for k in PH for a_, s_, c, o, _, _ in PH[k]["cells"] if (a_, s_) == (a, sd))]
    needy = [(a, sd, x) for (a, sd), x in lost.items() if x >= 8]
    better = [(k, a_, s_, c, o) for k in PH for a_, s_, c, o, _, _ in PH[k]["cells"] if o > c + 4]
    unchanged = [lbl for k, lbl in PHYS if k in PH and PH[k]["c_h"] == PH[k]["o_h"] == PH[k]["n"]]
    small = [k for k, _ in PHYS if k in PH and k not in hurt]
    floor = min(min(PH[k]["c_h"], PH[k]["o_h"]) for k in small)
    v["FB_PHYS"] = (
        f"Physical changes separate the two (Figure&#160;1, bottom; Figure&#160;2). With friction halved the closed loop holds "
        f"{fr5['c_h']} of {fr5['n']} rollouts and the replay {fr5['o_h']}, and at friction &#215;0.7 {fr7['c_h']} against "
        f"{fr7['o_h']}; at half the servo gain {kp['c_h']} against {kp['o_h']}, and at a 0.12&#8202;N&#8202;m torque limit "
        f"{fr['c_h']} against {fr['o_h']}. Friction &#215;1.3 and the three mass changes leave both modes holding at least "
        f"{floor} of {fr5['n']}, though a tool twice as heavy ends the replay of {pname(m2['worst'][1], m2['worst'][2])} "
        f"{abs(m2['worst'][0]):.2f} below its closed loop&#8217;s held cosine. The reliance on feedback varies between seeds more "
        f"than between tip arms: the replays of {' and '.join(robust)} hold every rollout under all eight changes, while those "
        f"of {', '.join(pname(a, sd) for a, sd, _ in needy[:-1])} and {pname(needy[-1][0], needy[-1][1])} lose "
        f"{min(x for _, _, x in needy)} to {max(x for _, _, x in needy)} of 64 rollouts under at least one of the four "
        "reductions. "
        + (f"Feedback costs rollouts in one cell: with friction halved, {pname(better[0][1], better[0][2])} holds "
           f"{better[0][3]} of 64 closed loop and {better[0][4]} replayed. " if len(better) == 1 else "")
        + "The policies observe no contact force; they correct from the tool&#8217;s pose relative to the palm, which the "
        "bench&#8217;s AprilTag tracker measures to 0.017&#176; and 0.03&#8202;mm rms.")
    if X.film:
        cap_rows = "; ".join(f"{f['label'].replace(chr(215), '&#215;')}: world {f['world']}, closed loop final cosine "
                             f"{f['closed_cos']:+.2f}{'' if f['closed_held'] else ' (dropped)'}, replay {f['open_cos']:+.2f}"
                             f"{'' if f['open_held'] else ' (dropped)'}" for f in X.film["rows"])
        fr_film = X.film["rows"][1]
        fr_fig = X.cell("tpu27mesh", 1, "phys", "friction=0.5", "open")
        spread = (f" The film is a separate run of the same evaluation: there the replay at halved friction holds "
                  f"{fr_film['open_n_held']} of 64, against {fr_fig['n_held']} in Figure&#160;1&#8217;s run, the GPU solver&#8217;s "
                  "run-to-run spread." if fr_fig and fr_film["open_n_held"] != fr_fig["n_held"] else "")
        v["FILM"] = P.film_tag(os.path.join(D, "media/20261013-closed_open.mp4"), os.path.join(D, "media/20261013-closed_open.jpg"),
                               "Figure&#160;2. Closed loop (left) and the open-loop replay of the same policy&#8217;s own actions "
                               "(right), the same world in both, at real speed. Top: the TPU-mesh policy, seed 0, on the box tip. "
                               "Bottom: the TPU-mesh policy, seed 1, on its own tip with friction halved. The world is the one "
                               f"whose replay ends at the median final cosine of the 64. {cap_rows}. Each panel&#8217;s label "
                               f"gives the counts over all 64 rollouts.{spread}")
        v["FILM"] = v["FILM"].replace("<figure>", f'<figure style="max-width:{P.COLUMN_PX}px">', 1)
    else:
        v["FILM"] = ""
    v["FB_WHY"] = (
        f"The tip model hardly changes the motion because the finger servos are much softer than the contact. Through the three "
        f"servos at 0.72&#8202;N&#8202;m/rad each fingertip is a spring of {k_servo:.0f}&#8202;N/m along its palmar normal at the grasp, (3), "
        f"and the pad contact&#8217;s tangent stiffness at 1&#8202;N is {kt[10000000] / 1e3:.1f}&#8202;kN/m, {kt[10000000] / k_servo:.0f} times "
        "stiffer. The servo&#8217;s deflection therefore sets the grip force of a commanded squeeze whichever tip model touches "
        "the tool. "
        "Friction and servo strength change the balance of forces on the tool, and a replay cannot respond to the motion "
        "that follows.")
    v["LAW_MODEL"] = (
        "The pads represent the printed tip as a Winkler foundation, a pressure \\(p=(E/h)\\,\\delta\\) on the overlap \\(\\delta\\) "
        "with \\(h\\)&#8202;=&#8202;8.5&#8202;mm, discretised as one sphere of 0.75&#8202;mm radius per 1&#8202;mm&#178; of tip surface, each a spring "
        "\\(K_s\\). MuJoCo&#8217;s soft contact gives a contact row at rest the stiffness \\(K\\) of (1), and the builder "
        "(<code>reorient_backends.replace_tips</code>) solves \\(d\\) per finger so that \\(K\\) equals the median \\(K_s\\) for the "
        "inverse weight of the scene&#8217;s own tool. The time constant \\(t_c\\) sets the damping and the creep, and \\(d\\) the "
        "stiffness.")
    v["LAW_RIG"] = (
        "The rig (<code>scripts/pad_contact_rig.py</code>) is the tracked real_v1 screwdriver scene with the production pads: the "
        "hand is rigid, gravity is off, and the tool is pressed onto the thumb&#8217;s palmar face with its axis across the face. "
        f"Every touching pad&#8217;s force over its penetration is {st[1.0]['k_min']:,.1f}&#8202;N/m at every load, \\(K_s\\) within the "
        "six significant digits the scene file stores for \\(d\\), "
        "and the normal forces balance the applied load within 0.2&#8202;%. The approach and the twist limit follow the pressure law "
        f"integrated over the block&#8217;s surface at 0.1&#8202;mm within {max(abs(s['approach_m'] / s['approach_law_m'] - 1) for s in L['static']) * 100:.0f}&#8202;% "
        f"and {max(abs(s['twist_Nm'] / s['twist_law_Nm'] - 1) for s in L['static']) * 100:.0f}&#8202;% (Table&#160;1); the largest "
        "difference is at 0.5&#8202;N, where two rows of pads sample a patch 2&#8202;mm wide.")
    v["TABLE_STATIC"] = table_static(L)
    v["TABLE_STATIC_NOTE"] = (
        "Table&#160;1. The tool pressed onto the thumb&#8217;s pads at four loads, read after 0.6&#8202;s under load. Force over penetration: each "
        "touching pad&#8217;s normal force over its penetration, identical for all of them. Law: the pressure \\(p=(E/h)\\,\\delta\\) "
        "integrated over the rounded block&#8217;s surface sampled at 0.1&#8202;mm, the same \\(E\\) and \\(h\\). Twist limit: \\(\\mu\\sum f_i\\rho_i\\) "
        "about the normal through the patch centre, with \\(\\mu\\)&#8202;=&#8202;1.")
    c10 = next(c for c in L["creep"] if c["impratio"] == 10)
    c100 = next(c for c in L["creep"] if c["impratio"] == 100)
    tw = {t["frac"]: t for t in L["twist"]}
    v["LAW_FRICTION"] = (
        "Below the slip force each pad&#8217;s friction rows hold the tool through a damper, so a held tangential load \\(F_t\\) "
        f"makes the tool creep at the speed of (2), with \\(n\\) touching pads. At 1&#8202;N and half the slip force the tool creeps "
        f"{c10['v_m_s'] * 1e3:.4f}&#8202;mm/s at impratio 10, where (2) gives {c10['v_law_m_s'] * 1e3:.4f}, and "
        f"{c100['v_m_s'] * 1e3:.5f}&#8202;mm/s at impratio 100, where it gives {c100['v_law_m_s'] * 1e3:.5f}; 1.2 times the slip force "
        f"slides it {c10['slide_m_in_0p1s'] * 1e3:.0f}&#8202;mm in 0.1&#8202;s. MuJoCo takes the friction coefficient of the geom with the "
        "higher contact priority, so the pads&#8217; coefficient of 1 sets this limit and the tool&#8217;s 2.4 is unused. "
        f"A twist of half the twist limit turns the tool {tw[0.5]['turned_deg_in_0p2s']:.2f}&#176; in 0.2&#8202;s and 1.2 times "
        f"the limit {tw[1.2]['turned_deg_in_0p2s']:.1f}&#176; (Table&#160;2). In the last form of (2) the inverse weight has cancelled, "
        "so the creep speed does not depend on the tool&#8217;s mass either.")
    v["TABLE_FRICTION"] = table_friction(L)
    v["TABLE_FRICTION_NOTE"] = (
        "Table&#160;2. Tangential force along the tool&#8217;s axis, applied at the contact line, and torque about the contact normal, "
        "with the tool pressed on at 1&#8202;N. Creep measured after 0.4&#8202;s of load; a 98&#8202;g tool reaches its steady creep by "
        "0.3&#8202;s. "
        "\\(\\tau_\\text{max}\\) from the pads&#8217; static forces, (3).")
    ml = {round(m["scale"], 2): m for m in L["mass"]}
    v["LAW_MASS"] = (
        f"The stiffness is independent of the tool&#8217;s mass when the pads are built with the tool in the scene. Rebuilt for "
        f"tools of {ml[0.25]['tool_mass_kg'] * 1e3:.1f}, {ml[1.0]['tool_mass_kg'] * 1e3:.1f} and {ml[4.0]['tool_mass_kg'] * 1e3:.1f}&#8202;g, "
        f"the pads give {ml[1.0]['k_rebuilt']:,.1f}&#8202;N/m for each within "
        f"{max(abs(m['k_rebuilt'] / ml[1.0]['k_rebuilt'] - 1) for m in ml.values()) * 100:.3f}&#8202;%, and the tool creeps at "
        f"{ml[1.0]['creep_m_s'] * 1e3:.4f}&#8202;mm/s within {max(abs(m['creep_m_s'] / ml[1.0]['creep_m_s'] - 1) for m in ml.values()) * 100:.2f}&#8202;% "
        "(Table&#160;3). Pads built for the "
        f"24.5&#8202;g tool give {ml[0.25]['k_stale']:.0f}&#8202;N/m on the 6&#8202;g tool and {ml[4.0]['k_stale']:,.0f} on the 98&#8202;g tool, "
        "the ratio of the inverse weights. The RL scenes and the hand-object scale study build their pads with their own tool in "
        "the scene. No run changes the mass after the build: the trainer&#8217;s mass range (<code>cube_mass_scale_range</code>) is "
        "declared and applied by no event, and the evaluator&#8217;s mass perturbation writes the mass without recomputing "
        "<code>body_invweight0</code>, which keeps the built stiffness. mjlab&#8217;s own mass randomisation recomputes it and would "
        "move the stiffness by the inverse-weight ratio.")
    v["TABLE_MASS"] = table_mass(L)
    v["TABLE_MASS_NOTE"] = (
        "Table&#160;3. Force over penetration of the touching pads at 1&#8202;N (the median; all touching pads agree) and the creep "
        "speed at half the slip force, impratio 10, for three tool masses. Right: the 24.5&#8202;g build kept and the tool&#8217;s "
        "density changed after it.")
    if X.census:
        cs = list(X.census.values())
        sh = [c["hand_share"] for c in cs if c["hand_share"] is not None]
        bodies = {}
        for c in cs:
            for b, val in c["hand_force_by_body_Ns"].items():
                bodies[b] = bodies.get(b, 0.0) + val
        top = sorted(bodies.items(), key=lambda kv: -kv[1])[:3]
        v["LAW_HAND"] = (
            "Contacts between the tool and the hand outside the fingertips (the finger links and the palm) keep MuJoCo&#8217;s default "
            "solref and solimp, whose stiffness does scale with the inverse weight. In the six policies&#8217; own closed-loop rollouts "
            f"they transmit {min(sh) * 100:.2f} to {max(sh) * 100:.2f}&#8202;% of the tool&#8217;s normal contact force from the turn&#8217;s onset "
            "to 5&#8202;s" + (f", most of it on {', '.join(b.replace('_', ' ') for b, _ in top)}" if top and max(sh) > 0.005 else "")
            + " (<code>census.jsonl</code>).")
    else:
        v["LAW_HAND"] = ""
    v["PARAM_SERVO"] = (
        f"The outcome depends on \\(\\mu\\), the servo and the tip&#8217;s geometry more than on the modulus. Between 5 and 20&#8202;MPa, which "
        "spans the finite-element estimates of the printed tip (14.6&#8202;MPa homogenised, 19.1&#8202;MPa from its G-code; "
        f"<a href=\"{TIP_PAGE}\">printed TPU fingertip</a>), the twist arm at 1&#8202;N changes from {arms[5000000] * 1e3:.2f} to "
        f"{arms[20000000] * 1e3:.2f}&#8202;mm (Table&#160;4). The tool lies across the flat palmar face, so the patch spans the "
        "face&#8217;s 9.4&#8202;mm flat width at any load, and the modulus changes only the patch&#8217;s 2&#8202;mm extent along the "
        "finger. The contact stays "
        f"{kt[5000000] / k_servo:.0f} to {kt[20000000] / k_servo:.0f} times stiffer than the servo.")
    v["TABLE_MODULUS"] = table_modulus(L, k_servo)
    v["TABLE_MODULUS_NOTE"] = (
        f"Table&#160;4. The rig at five moduli; the pads rebuilt for each. \\(k_\\text{{tip}}\\)&#8202;=&#8202;{k_servo:.0f}&#8202;N/m, the "
        "median of the three fingers at the CEM grasp of the servo-plant pad scene (thumb, index and middle within 1&#8202;N/m).")
    v["PARAM_MU"] = (
        "Friction enters in proportion: the slip force is \\(\\mu N\\) and the twist limit \\(\\mu\\sum f_i\\rho_i\\). The scenes use "
        "\\(\\mu\\)&#8202;=&#8202;1 on the pads and \\(E\\)&#8202;=&#8202;10&#8202;MPa, neither measured on the printed tip. The closed-loop "
        f"policies hold at \\(\\mu\\)&#8202;=&#8202;0.5 ({fr5['c_h']} of {fr5['n']}) where their replayed actions hold {fr5['o_h']}, so "
        "\\(\\mu\\) is the value to measure first; the bench&#8217;s open-loop plans depend on it most. Training randomised "
        "friction over &#215;0.55 to &#215;1.15.")
    v["FUTURE"] = (
        "<ol class='open'>"
        "<li><b>Friction coefficient of the printed tip.</b> Pull test of the TPU tip on the screwdriver handle with "
        "<code>scripts/tpu_tip_calibrate.py</code> (which also returns \\(E_s\\) and \\(E^*\\)). A measured \\(\\mu\\) below 0.5 lies "
        "outside the range in which the closed-loop policies held here and below the trainer&#8217;s randomisation (&#215;0.55 to "
        "&#215;1.15), and the policies would then be retrained with the randomisation widened to span it.</li>"
        "<li><b>Feedback on the bench.</b> Run the best servo-plant policy closed loop with the tracker, log its servo targets, "
        "and replay the log open loop from the same grasp as a deploy plan&#8217;s <code>_traj.csv</code>. This page predicts equal "
        "outcomes on the nominal handle and a lead for the closed loop on a handle with lower friction.</li>"
        "<li><b>Touch input.</b> Before a policy observes the field of a magnet tip (the gyroid and eFlesh tips of "
        "<code>docs/experiments/20261013-eflesh_printability/</code>), the simulator has to produce the tip&#8217;s "
        "elastic deflection, which the pads do not store: below slip the tool creeps at (2) instead. Map the pads&#8217; net "
        "fingertip force through the finite-element compliance (17.6&#8202;&#181;m/N normal, 9 and 28&#8202;&#181;m/N tangential) for the "
        f"normal channel, and test the shear channel on the bed&#8217;s task 8 with the <a href=\"{SKIN_PAGE}\">sprung skin</a> at "
        "impratio 100 or more before training on it. HydroShear (Dang et al., arXiv 2603.00446) reports 93&#8202;% success "
        "against 58&#8211;61&#8202;% for GelSight policies when the simulator models stick-slip and path-dependent shear.</li>"
        "<li><b>Stiffness guard in the scene builders.</b> Assert <code>pad_contact_rig.realised_stiffness</code> equal to the "
        "build&#8217;s <code>pad_K</code> after <code>scripts/make_pad_morphology_run.py</code> and "
        "<code>scripts/hand_object_scale_sim.py</code> write a scene. If tool-mass randomisation is added to the trainer, apply "
        "it without mjlab&#8217;s <code>set_const</code> recompute, or re-solve \\(d\\) per world.</li>"
        "<li><b>Regression tests.</b> <code>PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest "
        "tests/test_pad_contact_laws.py</code> (15 tests, 15&#8202;s) after any change to <code>reorient_backends.replace_tips</code>, "
        "<code>fingertip_geometry.py</code> or the MuJoCo version.</li></ol>")
    v["FOOTER"] = (
        "<p>Rebuild: <code>.venv/bin/python scripts/pad_contact_rig.py</code> (Tables&#160;1&#8211;4, <code>pad_laws.json</code>), "
        "<code>MUJOCO_GL=egl WARP_CACHE_PATH=$(mktemp -d) uv run --extra rl --extra gpu python scripts/contact_feedback_replay.py "
        "--perturb friction=0.5 friction=0.7 friction=1.3 mass=0.8 mass=1.2 mass=2 kp=0.36 fr=0.12 --census --film</code> "
        "(Figures&#160;1&#8211;2, <code>replay.jsonl</code>, <code>census.jsonl</code>, <code>acts/</code>), then "
        "<code>.venv/bin/python scripts/contact_feedback_page.py</code>. Tests: <code>tests/test_pad_contact_laws.py</code>.</p>")
    v["FIG_FEEDBACK"] = svg_feedback(X)
    t = open(TPL).read()
    for k, val in v.items():
        t = t.replace("{{" + k + "}}", val)
    left = sorted(set(x.split("}}")[0] for x in t.split("{{")[1:]))
    if left:
        raise SystemExit(f"unfilled placeholders: {left}")
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None)
             for m in P.TEX_RE.finditer(t)]
    svgs = iter(texsvg.render(items, cache_path=TEX_CACHE, scale=P.TEX_SCALE))
    t = P.TEX_RE.sub(lambda m: next(svgs), t)
    open(OUT, "w").write(retro_style.apply(t))
    print(f"formulas {len(items)}; wrote {OUT} ({os.path.getsize(OUT) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
