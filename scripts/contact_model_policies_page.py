#!/usr/bin/env python3
r"""Build docs/experiments/20261008-contact_model_policies/20261008-fingertip_contact_model_policies.html.

    logs/20261001-hom_contact/venv/bin/python scripts/contact_model_policies_page.py

Steps 2-7 of docs/handoff/20261007-contact_model_policy_training.md: D6 reorientation trained from scratch on the working
plant under four fingertip contact models (the legacy box tip and the TPU block mesh with MuJoCo point contact, the TPU
block as 1 mm sphere pads, the pads on a spring-mounted skin), three seeds each, 40 M steps
(scripts/rl_contact_train_queue_20261008.sh); the checkpoint evaluations, the final policies' physics, the transfer matrix,
the open-loop replays in CPU MuJoCo, Drake and Newton, the perturbations and the films (scripts/rl_contact_eval.py,
scripts/rl_policy_replay.py, logs/20261008-contact_model_policies/post_train.sh). Every number in the prose is computed here
from the rows in docs/experiments/20261008-contact_model_policies/; a section whose rows are not written yet renders a note.
Style, SVG helpers and LaTeX rendering are those of scripts/contact_overview_page.py.

The arm palette is the dataviz reference instance's slots 1-3 (blue, orange, aqua) for the three TPU arms with the legacy
box tip as the dashed ink reference; checked in both modes with the reference validator's formulas (all pairs: CVD
Delta E >= 9.2, normal >= 20.9; aqua is 2.8:1 on white, so every series is also labelled in text and tables).
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import contact_overview_page as P  # noqa: E402
import texsvg  # noqa: E402

D = os.path.join(P.EXP, "20261008-contact_model_policies")
OUT = os.path.join(D, "20261008-fingertip_contact_model_policies.html")
TPL = os.path.join(ROOT, "scripts/contact_model_policies_page.template.html")
TEX_CACHE = os.path.join(D, "texsvg_cache.json")
NOTE = "docs/handoff/20261007-contact_model_policy_training.md"
COMPLIANCE_PAGE = "docs/experiments/20261007-native_compliance/20261007-native_presliding_compliance.html"
TURN_PAGE = "docs/experiments/20261006-hom_turn3/20261006-servo_refit_hom_turn_pad_cost.html"
OVERVIEW = "docs/experiments/20261008-contact_overview/20261008-sphere_pad_contact_model.html"
fmt, num = P.fmt, P.num
ARMS = ("box", "tpu27mesh", "tpu27pads1", "tpu27skin")
SCENES = ("box", "tpu27mesh", "tpu27meshc4", "tpu27pads1", "tpu27skin")
ENGINES = ("mujoco", "drake", "newton")
LBL = {"box": "box tip, point contact", "tpu27mesh": "TPU block, point contact", "tpu27meshc4": "TPU block, condim 4",
       "tpu27pads1": "TPU block, 1&#8202;mm pads", "tpu27skin": "TPU block, pads on a skin"}
SHORT = {"box": "box", "tpu27mesh": "TPU mesh", "tpu27meshc4": "condim 4", "tpu27pads1": "pads", "tpu27skin": "skin"}
COL = {"box": "var(--a-box)", "tpu27mesh": "var(--a-mesh)", "tpu27pads1": "var(--a-pads)", "tpu27skin": "var(--a-skin)"}
ENG_LBL = {"mujoco": "CPU MuJoCo", "drake": "Drake hydroelastic", "newton": "Newton hydroelastic"}
STEPS_PER_IT = 2048 * 24
N_ROLL = 64
FIG, TAB = [0], [0]
EXTRA_CSS = """
:root{--a-box:#48545D;--a-mesh:#2a78d6;--a-pads:#eb6834;--a-skin:#1baf7a;--hm0:#eef3f9;--hm1:#cde2fb;--hm2:#9ec5f4;
--hm3:#6da7ec;--hm4:#3987e5;--hm5:#256abf;--hm6:#184f95}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--a-box:#A8B4BD;--a-mesh:#3987e5;--a-pads:#d95926;
--a-skin:#199e70;--hm0:#1f262c;--hm1:#104281;--hm2:#184f95;--hm3:#1c5cab;--hm4:#2a78d6;--hm5:#5598e7;--hm6:#86b6ef}}
:root[data-theme="dark"]{--a-box:#A8B4BD;--a-mesh:#3987e5;--a-pads:#d95926;--a-skin:#199e70;--hm0:#1f262c;--hm1:#104281;
--hm2:#184f95;--hm3:#1c5cab;--hm4:#2a78d6;--hm5:#5598e7;--hm6:#86b6ef}
figure.films{display:grid;grid-template-columns:1fr 1fr;gap:14px}
figure.films video,figure.films img{width:100%;border-radius:8px}
figure.films figcaption{grid-column:1/-1}
@media (max-width:640px){figure.films{grid-template-columns:1fr}}
"""


# ------------------------------------------------------------------------------------------ rows

def rows(name):
    p = os.path.join(D, name)
    if not os.path.exists(p):
        return []
    out = []
    for line in open(p):
        line = line.strip()
        if line:
            r = json.loads(line)
            if r.get("status", "ok") in ("ok", "complete"):
                out.append(r)
    return out


def last_by(rs, keys):
    """The newest row per key (rows are appended in time order)."""
    d = {}
    for r in rs:
        d[tuple(r.get(k) for k in keys)] = r
    return list(d.values())


class Data:
    def __init__(self):
        self.ck = last_by(rows("ckpt_eval.jsonl"), ("tag", "iteration"))
        self.fin = last_by(rows("final_eval.jsonl"), ("tag",))
        self.tr = last_by(rows("transfer.jsonl"), ("tag", "scene"))
        self.rb = last_by(rows("robust.jsonl"), ("tag", "perturb"))
        self.tb = rows("tb_dynamics.jsonl")
        self.cost = {r["tag"]: r for r in rows("run_costs.jsonl")}
        self.rep = [r for r in rows("policy_replay.jsonl") if r.get("engine") and r.get("kind") != "record"]
        self.rec = [r for r in rows("policy_replay.jsonl") if r.get("kind") == "record"]
        self.wall = {}
        for r in self.tb:
            self.wall.setdefault(r["tag"], {})[r["iteration"]] = r.get("wall_time")
        p = os.path.join(D, "films.json")
        self.films = json.load(open(p)) if os.path.exists(p) else []

    def finals(self, arm):
        return sorted((r for r in self.fin if r["arm"] == arm), key=lambda r: r["seed"])

    def curve(self, arm):
        """{seed: [(iteration, env_steps, n_held, held_cos_mean, cos_mean, tag)]}"""
        out = {}
        for r in sorted((r for r in self.ck if r["arm"] == arm), key=lambda r: (r["seed"], r["iteration"])):
            out.setdefault(r["seed"], []).append((r["iteration"], r["env_steps"], r["n_held"], r["held_cos_mean"],
                                                  r["cos_mean"], r["tag"]))
        return out

    def hours(self, tag, it):
        w = self.wall.get(tag, {})
        if not w or it not in w or w.get(it) is None:
            return None
        t0 = w[min(w)]
        # the first logged iteration ends one iteration time after the start
        c = self.cost.get(tag, {})
        return (w[it] - t0 + (c.get("s_per_it_median") or 0.0)) / 3600.0

    def replays(self, arm, engine):
        return [r for r in self.rep if r["engine"] == engine and f"_{arm}_40M_" in r["dir"]]


def tag_arm(tag):
    for a in ARMS:
        if f"_{a}_40M_" in tag:
            return a
    return None


# ------------------------------------------------------------------------------------------ html helpers

def figure(svg, caption, legend=None):
    FIG[0] += 1
    lg = P._legend_html(legend) if legend else ""
    return f'<figure class="diagram">{svg}{lg}<figcaption>Figure&#160;{FIG[0]}. {caption}</figcaption></figure>'


def table(head, body, caption=None, note=None, text_cols=(0,)):
    out = []
    if caption:
        TAB[0] += 1
        out.append(f"<p class='tcap'><b>Table&#160;{TAB[0]}.</b> {caption}</p>")
    cls = lambda i: "" if i in text_cols else " class=num"  # noqa: E731
    out.append("<div class='tw'><table><thead><tr>" + "".join(f"<th{cls(i)}>{h}</th>" for i, h in enumerate(head))
               + "</tr></thead><tbody>")
    for r in body:
        if isinstance(r, str):
            out.append(f"<tr class='grp'><td colspan='{len(head)}'>{r}</td></tr>")
        else:
            out.append("<tr>" + "".join(f"<td{cls(i)}>{c}</td>" for i, c in enumerate(r)) + "</tr>")
    out.append("</tbody></table></div>")
    if note:
        out.append(f"<p class='tnote'>{note}</p>")
    return "".join(out)


def swatch(arm):
    dash = "border-top:2px dashed" if arm == "box" else "border-top:3px solid"
    return f'<span style="display:inline-block;width:18px;vertical-align:middle;{dash} {COL[arm]};margin-right:6px"></span>'


def legend(arms=ARMS):
    return [(LBL[a], COL[a], a == "box", None) for a in arms]


def f1(x, nd=1):
    return fmt(x, nd)


def frac(n, d):
    return f"{n}/{d}" if d else "&#8211;"


# ------------------------------------------------------------------------------------------ figures

def _gap_path(out, fx, fy, xs, ys, colour, dashed=False, width=2.0):
    """A line through (xs, ys) that breaks where y is None."""
    seg = []
    for x, y in list(zip(xs, ys)) + [(None, None)]:
        if x is None or y is None:
            if len(seg) >= 2:
                P._path(out, fx, fy, seg, colour, dashed=dashed, width=width)
            seg = []
        else:
            seg.append((x, y))


def _band(out, fx, fy, xs, lo, hi, colour):
    pts = [(x, a, b) for x, a, b in zip(xs, lo, hi) if a is not None and b is not None]
    if len(pts) < 2:
        return
    d = "M" + " L".join(f"{fx(x):.1f},{fy(b):.1f}" for x, a, b in pts) + " L" + \
        " L".join(f"{fx(x):.1f},{fy(a):.1f}" for x, a, b in reversed(pts)) + " Z"
    out.append(f'<path d="{d}" style="fill:{colour};fill-opacity:.14;stroke:none"/>')


def _series(X: Data, arm, key, xkind):
    """Median and seed range of `key` (n_held | held_cos) at each evaluated checkpoint, x in M steps or hours."""
    cur = X.curve(arm)
    by_it = {}
    for seed, pts in cur.items():
        for it, steps, nh, hc, cm, tag in pts:
            x = steps / 1e6 if xkind == "steps" else X.hours(tag, it)
            v = nh if key == "n_held" else hc
            by_it.setdefault(it, []).append((x, v))
    xs, med, lo, hi, n = [], [], [], [], []
    for it in sorted(by_it):
        vals = [v for _, v in by_it[it] if v is not None]
        xx = [x for x, _ in by_it[it] if x is not None]
        if not xx:
            continue
        xs.append(float(np.median(xx)))
        med.append(float(np.median(vals)) if vals else None)
        lo.append(min(vals) if vals else None)
        hi.append(max(vals) if vals else None)
        n.append(len(by_it[it]))
    return xs, med, lo, hi, n


def svg_curves(X: Data):
    if not X.ck:
        return P.pending("Checkpoint evaluations not written yet (ckpt_eval.jsonl).")
    W, H = 990, 600
    out = P._svg_open(W, H, "Held rollouts and held cosine of the deterministic policy at its checkpoints, per contact "
                            "model, against environment steps and against training wall-clock time")
    hmax = max([X.hours(r["tag"], r["iteration"]) or 0 for r in X.ck] + [0.5])
    hx = [0, 1, 2, 3] if hmax <= 3.2 else [0, 1, 2, 3, 4]
    panels = [(70, 40, "steps", "n_held"), (570, 40, "hours", "n_held"), (70, 330, "steps", "held_cos"),
              (570, 330, "hours", "held_cos")]
    for x0, y0, xk, key in panels:
        xs_rng = (0, 40) if xk == "steps" else (0, max(hx))
        xt = (0, 10, 20, 30, 40) if xk == "steps" else hx
        ys, yt, ylab = ((0, 64), (0, 16, 32, 48, 64), "rollouts held at 5 s (of 64)") if key == "n_held" else \
            ((-0.2, 1.0), (0.0, 0.25, 0.5, 0.75, 1.0), "held cosine (mean over held rollouts)")
        xlab = "environment steps (millions)" if xk == "steps" else "training wall-clock time (h)"
        fx, fy = P._panel(out, x0, y0, 360, 200, xs_rng, ys, xt, yt, xlab, ylab,
                          yfmt=("{:g}" if key == "n_held" else "{:.2f}"))
        if key == "held_cos":
            out.append(f'<line x1="{x0}" x2="{x0 + 360}" y1="{fy(0.9):.1f}" y2="{fy(0.9):.1f}" '
                       f'style="stroke:var(--ink3);stroke-dasharray:2 4"/><text x="{x0 + 362}" y="{fy(0.9) + 4:.1f}" '
                       f'style="fill:var(--ink3)">0.9</text>')
        for arm in ARMS:
            xs, med, lo, hi, n = _series(X, arm, key, xk)
            if not xs:
                continue
            _band(out, fx, fy, xs, lo, hi, COL[arm])
            _gap_path(out, fx, fy, xs, med, COL[arm], dashed=(arm == "box"))
            for x, m, k in zip(xs, med, n):
                if m is not None:
                    P._marker(out, fx(x), fy(m), COL[arm], r=3.2,
                              title=f"{SHORT[arm]}: {m:.3g} (median of {k} seed{'s' if k > 1 else ''})")
    out.append("</svg>")
    return "".join(out)


def svg_final(X: Data):
    """Final cosine of every rollout of every final policy, per arm and seed; held rollouts filled."""
    if not X.fin:
        return P.pending("Final-policy evaluations not written yet (final_eval.jsonl).")
    W, H = 990, 330
    out = P._svg_open(W, H, "Final cosine of the tool axis with vertical in each of 64 rollouts per seed, per contact "
                            "model; filled dots held the tool at 5 s, hollow dots had dropped it")
    x0, y0, w, h = 70, 30, 880, 230
    fy = lambda v: y0 + h - (v + 1.0) / 2.0 * h  # noqa: E731
    for v in (-1.0, -0.5, 0.0, 0.5, 0.9, 1.0):
        out.append(f'<line x1="{x0}" x2="{x0 + w}" y1="{fy(v):.1f}" y2="{fy(v):.1f}" style="stroke:var(--rule2)'
                   f'{";stroke-dasharray:2 4" if v == 0.9 else ""}"/><text x="{x0 - 8}" y="{fy(v) + 4:.1f}" '
                   f'text-anchor="end" style="fill:var(--ink3)">{v:+.1f}</text>')
    out.append(f'<text x="{x0}" y="{y0 - 12}" style="fill:var(--ink2)">final cosine of the tool axis with vertical (+1 tip down)</text>')
    gw = w / len(ARMS)
    rng = np.random.default_rng(7)
    for i, arm in enumerate(ARMS):
        gx = x0 + i * gw
        out.append(f'<text x="{gx + gw / 2:.1f}" y="{y0 + h + 22}" text-anchor="middle" style="fill:var(--ink2)">'
                   f'{SHORT[arm]}</text>')
        fins = X.finals(arm)
        for j, seed in enumerate((0, 1, 2)):
            r = next((r for r in fins if r["seed"] == seed), None)
            cx = gx + gw * (j + 1) / 4
            out.append(f'<text x="{cx:.1f}" y="{y0 + h + 38}" text-anchor="middle" style="fill:var(--ink3);font-size:11px">'
                       f's{seed}</text>')
            if r is None:
                continue
            cs, hs = r["final_cos"], r["held_final"]
            jit = rng.uniform(-gw / 10, gw / 10, len(cs))
            for c, hd, dx in zip(cs, hs, jit):
                if hd:
                    out.append(f'<circle cx="{cx + dx:.1f}" cy="{fy(c):.1f}" r="2.6" style="fill:{COL[arm]};'
                               f'fill-opacity:.75;stroke:var(--card);stroke-width:.6"/>')
                else:
                    out.append(f'<circle cx="{cx + dx:.1f}" cy="{fy(c):.1f}" r="2.4" style="fill:none;'
                               f'stroke:{COL[arm]};stroke-width:1;stroke-opacity:.7"/>')
            if r.get("held_cos_mean") is not None:
                yv = fy(r["held_cos_mean"])
                out.append(f'<line x1="{cx - gw / 7:.1f}" x2="{cx + gw / 7:.1f}" y1="{yv:.1f}" y2="{yv:.1f}" '
                           f'style="stroke:var(--ink);stroke-width:2"><title>{SHORT[arm]} s{seed}: held '
                           f'{r["n_held"]}/64, held cos {r["held_cos_mean"]:.3f}</title></line>')
    out.append("</svg>")
    return "".join(out)


def _hm_class(v):
    if v is None:
        return "var(--hm0)", "var(--ink3)"
    k = min(6, int(v * 6.999))
    return f"var(--hm{k})", ("var(--card)" if k >= 4 else "var(--ink)")


def svg_transfer(X: Data):
    if not X.tr and not X.rep:
        return P.pending("Transfer matrix and replays not written yet (transfer.jsonl, policy_replay.jsonl).")
    cols = [("mjw", s) for s in SCENES] + [("rep", e) for e in ENGINES]
    W = 990
    cw, ch, x0, y0 = 96, 58, 190, 70
    H = y0 + ch * len(ARMS) + 70
    out = P._svg_open(W, H, "Held fraction of every training arm's final policies evaluated under each contact model in "
                            "MuJoCo-Warp and replayed open loop in CPU MuJoCo, Drake and Newton")
    for j, (kind, s) in enumerate(cols):
        cx = x0 + j * cw + (24 if kind == "rep" else 0)
        lab = SHORT[s] if kind == "mjw" else ENG_LBL[s].split()[0] + (" MuJoCo" if s == "mujoco" else "")
        out.append(f'<text x="{cx + cw / 2:.1f}" y="{y0 - 10}" text-anchor="middle" style="fill:var(--ink2)">{lab}</text>')
    out.append(f'<text x="{x0 + 2.5 * cw:.1f}" y="{y0 - 34}" text-anchor="middle" style="fill:var(--ink)">'
               f'MuJoCo-Warp, closed loop, 64 rollouts per seed</text>')
    out.append(f'<text x="{x0 + 5 * cw + 24 + 1.5 * cw:.1f}" y="{y0 - 34}" text-anchor="middle" style="fill:var(--ink)">'
               f'open-loop replay, 6 worlds per seed</text>')
    for i, arm in enumerate(ARMS):
        cy = y0 + i * ch
        out.append(f'<text x="{x0 - 12}" y="{cy + ch / 2 + 4:.1f}" text-anchor="end" style="fill:var(--ink2)">'
                   f'trained: {SHORT[arm]}</text>')
        for j, (kind, s) in enumerate(cols):
            cx = x0 + j * cw + (24 if kind == "rep" else 0)
            if kind == "mjw":
                rs = [r for r in X.tr if r["arm"] == arm and r["scene"] == s]
                nh, nn = sum(r["n_held"] for r in rs), sum(r["n"] for r in rs)
                hc = [r["held_cos_mean"] for r in rs if r.get("held_cos_mean") is not None]
            else:
                rs = X.replays(arm, s)
                nh, nn = sum(1 for r in rs if r.get("held_end")), len(rs)
                hc = [r["cos_end"] for r in rs if r.get("held_end")]
            v = nh / nn if nn else None
            fill, ink = _hm_class(v)
            own = kind == "mjw" and s == arm
            out.append(f'<rect x="{cx + 2}" y="{cy + 2}" width="{cw - 4}" height="{ch - 4}" rx="4" style="fill:{fill};'
                       f'stroke:{"var(--ink)" if own else "none"};stroke-width:2"><title>{SHORT[arm]} on '
                       f'{SHORT.get(s, s)}: held {nh}/{nn}{", held cos %.3f" % np.mean(hc) if hc else ""}</title></rect>')
            if nn:
                out.append(f'<text x="{cx + cw / 2:.1f}" y="{cy + ch / 2 - 2:.1f}" text-anchor="middle" style="fill:{ink};'
                           f'font-size:13px;font-weight:500">{100 * v:.0f}%</text>')
                out.append(f'<text x="{cx + cw / 2:.1f}" y="{cy + ch / 2 + 14:.1f}" text-anchor="middle" style="fill:{ink};'
                           f'font-size:11px">{("cos %.2f" % np.mean(hc)) if hc else "&#8211;"}</text>')
    out.append(f'<text x="{x0}" y="{H - 24}" style="fill:var(--ink3);font-size:11.5px">Cell: rollouts held at the end '
               f'(load test), pooled over seeds, and the mean final cosine of the held ones. Outlined: the contact model '
               f'the policy trained on.</text>')
    out.append("</svg>")
    return "".join(out)


PERT = [("friction", "sliding friction, x nominal", (0.7, 1.0, 1.3)), ("mass", "tool mass, x nominal", (0.8, 1.0, 1.2)),
        ("kp", "finger servo kp (N m/rad)", (2.0, 4.0, 6.0, 10.0)), ("dt", "physics step (ms)", (1.0, 2.0)),
        ("noise", "grasp-pose noise (mm, deg)", (0.0, 1.0))]


def _rb_value(X, arm, kind, v):
    """Pooled held fraction of an arm's final policies under one perturbation value (nominal at the reference)."""
    if (kind == "friction" and v == 1.0) or (kind == "mass" and v == 1.0) or (kind == "kp" and v == 4.0) or \
            (kind == "dt" and v == 2.0) or (kind == "noise" and v == 0.0):
        rs = [r for r in X.rb if r["arm"] == arm and r["perturb"] == "nominal"]
    elif kind == "dt":
        rs = [r for r in X.rb if r["arm"] == arm and r["perturb"] == "dt=0.001"]
    elif kind == "noise":
        rs = [r for r in X.rb if r["arm"] == arm and r["kind"] == "noise"]
    else:
        rs = [r for r in X.rb if r["arm"] == arm and r["kind"] == kind and abs(float(r["value"]) - v) < 1e-9]
    nn = sum(r["n"] for r in rs)
    return (sum(r["n_held"] for r in rs) / nn) if nn else None, rs


def svg_robust(X: Data):
    if not X.rb:
        return P.pending("Perturbation rows not written yet (robust.jsonl).")
    W, H = 990, 300
    out = P._svg_open(W, H, "Held fraction of each arm's final policies under one perturbation at a time")
    pw, gap, x0 = 150, 46, 60
    for k, (kind, lab, vals) in enumerate(PERT):
        px = x0 + k * (pw + gap)
        pos = {v: i for i, v in enumerate(vals)}
        fx0, fy = P._panel(out, px, 40, pw, 170, (-0.4, len(vals) - 0.6), (0, 1), [], (0, 0.5, 1),
                           lab, "held fraction" if k == 0 else "", yfmt="{:g}")
        for v in vals:
            tx = fx0(pos[v])
            vt = ("1" if kind == "dt" and v == 1.0 else "2") if kind == "dt" else \
                (("0" if v == 0 else "2, 5") if kind == "noise" else f"{v:g}")
            out.append(f'<text x="{tx:.1f}" y="{40 + 170 + 17}" text-anchor="middle" style="fill:var(--ink3)">{vt}</text>')
        for arm in ARMS:
            pts = []
            for v in vals:
                f, rs = _rb_value(X, arm, kind, v)
                if f is not None:
                    pts.append((pos[v], f, len(rs)))
            P._path(out, fx0, fy, [(a, b) for a, b, _ in pts], COL[arm], dashed=(arm == "box"), width=1.8)
            for a, b, n in pts:
                P._marker(out, fx0(a), fy(b), COL[arm], r=3.4, title=f"{SHORT[arm]}: {b:.2f} ({n} policies)")
    out.append("</svg>")
    return "".join(out)


# ------------------------------------------------------------------------------------------ films

def video(rel, cls=""):
    p = os.path.join(D, rel)
    if not os.path.exists(p):
        return ""
    jpg = p[:-4] + ".jpg"
    po = f' poster="{P.R.data_uri(jpg, "image/jpeg")}"' if os.path.exists(jpg) else ""
    return f'<video src="{P.R.data_uri(p, "video/mp4")}"{po} controls muted loop playsinline preload="metadata"></video>'


def img(rel):
    p = os.path.join(D, rel)
    return f'<img src="{P.R.data_uri(p, "image/jpeg")}" alt="{os.path.basename(rel)}">' if os.path.exists(p) else ""


def films(X: Data):
    if not X.films:
        return P.pending("Films not rendered yet (post_train.sh: close-up renders of the best and median seed per arm "
                         "and three-engine replay films).")
    out = []
    for arm in ARMS:
        fs = [f for f in X.films if f["arm"] == arm]
        clips = [(f, f"media/{f['tag']}_{f['role']}.mp4") for f in fs]
        clips = [(f, c) for f, c in clips if os.path.exists(os.path.join(D, c))]
        if not clips:
            continue
        FIG[0] += 1
        body = "".join(video(c) for _, c in clips) + "".join(img(c[:-4] + "_strip.jpg") for _, c in clips)
        names = "; ".join(f"{f['role']} seed s{f['tag'][-1]} (<code>docs/experiments/20261008-contact_model_policies/{c}</code>)"
                          for f, c in clips)
        out.append(f'<figure class="films">{body}<figcaption>Figure&#160;{FIG[0]}. {LBL[arm]}: {names}. '
                   f'{film_caption(X, arm, fs)}</figcaption></figure>')
    for arm in ARMS:
        f = next((f for f in X.films if f["arm"] == arm and f["role"] == "median"), None)
        if f is None:
            continue
        rel = f"media/{f['tag']}_replay_three.mp4"
        if not os.path.exists(os.path.join(D, rel)):
            continue
        FIG[0] += 1
        out.append(f'<figure>{video(rel)}<figcaption>Figure&#160;{FIG[0]}. {LBL[arm]}, median seed, world 0: the '
                   f'MuJoCo-Warp rollout it trained on (left), and its finger targets from the onset replayed open loop in '
                   f'Drake (centre) and Newton (right) with the TPU block as a hydroelastic tip. '
                   f'<code>docs/experiments/20261008-contact_model_policies/{rel}</code></figcaption></figure>')
    return "".join(out) or P.pending("Films listed in films.json but not rendered yet.")


FILM_NOTES = {}      # arm -> sentence written after watching the films (filled in the next revision)


def film_caption(X, arm, fs):
    return FILM_NOTES.get(arm, "")


# ------------------------------------------------------------------------------------------ sections

GLOSSARY = [
    ("arm", "One fingertip contact model the policies train on. All arms share the hand (real_v1 D6), the working plant "
     "(finger servo kp 4&#8202;N&#8202;m/rad, damping 0.08&#8202;N&#8202;m&#8202;s/rad, torque limit 1&#8202;N&#8202;m), "
     "the tool (the 24.5&#8202;g screwdriver, \\(\\mu\\) = 1), the solver (MuJoCo-Warp, 2&#8202;ms, 10 Newton iterations, "
     "elliptic cones, impratio 10), the recipe and its flags, and differ only in the fingertip."),
    ("env step, iteration", "One env step is one 20&#8202;ms policy step of one of the 2,048 parallel envs; one PPO "
     "iteration collects 24 steps per env, 49,152 env steps. 40&#8202;M env steps are 813 iterations."),
    ("held (load test)", "At a policy step, at least two fingertips each press on the tool with at least 0.24&#8202;N (its "
     "weight) and the tool is above 60&#8202;mm. A rollout is held when this is true at its last step (5&#8202;s)."),
    ("cos", "The signed cosine of the tool&#8217;s own axis with world vertical, \\(\\cos\\theta = R_{zz}\\); +1 is tip "
     "down, the goal of the turn; the tool starts horizontal (cos&#8202;0)."),
    ("held cosine", "The mean of the final cos over the held rollouts of one evaluation. The turn the policy achieved, "
     "counted only where the hand still carries the tool."),
    ("evaluation", "64 rollouts of the deterministic policy (its mean action) in parallel envs for 250 policy steps, with "
     "the training&#8217;s timing (residual and reorientation reward from step 58, after the scripted grasp and lift), no "
     "early termination and no randomisation. The 64 differ only through the GPU contact solve."),
    ("seed spread", "Range of the three seeds&#8217; held cosines (or held fractions) of one arm."),
    ("rollout spread", "Standard deviation of the final cos over the held rollouts of one evaluation, averaged over seeds."),
    ("grip", "Sum of the three fingertips&#8217; net contact force on the tool, mean over the held steps from step 58, N."),
    ("|&#916;a|", "Mean absolute change of the 9-dimensional action between consecutive policy steps, in action units "
     "(one unit is 0.5&#8202;rad of servo target); 0 for a policy that holds still."),
    ("contacts per step", "Fingertip-tool contacts summed over the three tips (the fingertip sensor&#8217;s match count), mean "
     "over the held steps."),
    ("penetration", "Deepest fingertip-tool penetration over a rollout&#8217;s held steps, mm; mean over held rollouts."),
    ("creep", "Displacement of the tool against the palm over the last second of the rollout divided by 0.98&#8202;s "
     "(mm/s), and the rotation of its axis over the same second (&#176;/s); median over held rollouts."),
    ("peak force", "Largest single-fingertip net force over a rollout&#8217;s held steps, N; mean over held rollouts."),
    ("transfer gap", "Held fraction (or held cosine) under another contact model or simulator minus that under the "
     "policy&#8217;s own contact model in MuJoCo-Warp."),
    ("open-loop replay", "From the state at policy step 58 of a MuJoCo-Warp rollout, the policy&#8217;s recorded finger "
     "targets are played into another simulator at 50&#8202;Hz with the palm welded at its lifted pose "
     "(<code>scripts/rl_policy_replay.py</code>); the policy does not see that simulator&#8217;s state."),
]


def glossary():
    return '<dl class="glossary">' + "".join(f"<dt>{a}</dt><dd>{b}</dd>" for a, b in GLOSSARY) + "</dl>"


def arms_section(X: Data):
    sk = json.load(open(os.path.join(P.ROOT, "results/phase1/real_v1/20261008-sv1_u0308_b050_work_tip_tpu2.7skin/summary.json")))
    s = sk.get("fingertip", {}).get("skin", {})
    head = ["contact model", "contact", "s per iteration", "hours per 40&#8202;M", "GPU (GB)", "host RSS (GB)",
            "failed runs"]
    body = []
    desc = {"box": ("one 10.55&#215;21.1&#215;15&#8202;mm box per tip", "point, condim 3"),
            "tpu27mesh": ("the printed TPU block, 2.7&#8202;mm fillets, one convex mesh", "point, condim 3"),
            "tpu27meshc4": ("the TPU block mesh", "point, condim 4"),
            "tpu27pads1": ("1,060 spheres of 0.75&#8202;mm on the block per tip", "pads (sphere-packed patch)"),
            "tpu27skin": ("the pads on a sprung skin body per tip", "pads + presliding spring")}
    for a in SCENES:
        cs = [c for t, c in X.cost.items() if tag_arm(t) == a and c.get("finished")]
        spi = [c["s_per_it_median"] for c in cs if c.get("s_per_it_median")]
        hrs = [c["wall_h"] for c in cs if c.get("wall_h")]
        gpu = [c["gpu_mem_peak_mb"] for c in cs if c.get("gpu_mem_peak_mb")]
        rss = [c["host_rss_peak_gb"] for c in cs if c.get("host_rss_peak_gb")]
        fl = sum(c.get("failed_attempts", 0) for c in X.cost.values() if tag_arm(c["tag"]) == a)
        trained = a in ARMS
        body.append([f"{swatch(a) if trained else ''}{LBL[a]}", desc[a][1],
                     f1(np.median(spi), 2) if spi else ("probe 5.31" if a == "tpu27meshc4" else "&#8211;"),
                     f1(np.median(hrs), 2) if hrs else ("not trained" if not trained else "&#8211;"),
                     f1(max(gpu) / 1e3, 1) if gpu else "&#8211;", f1(max(rss), 1) if rss else "&#8211;",
                     str(fl) if trained else "&#8211;"])
    txt = ("<p>The legacy box tip is the fingertip every policy of this program trained on through 2026-10-05: one box "
           "geom of 10.55&#215;21.1&#215;15&#8202;mm per tip, friction 1 (torsional and rolling coefficients present but "
           "inactive at condim 3), MuJoCo point contact. The TPU arms replace it by the printed tip, a 17&#215;14.8&#215;22&#8202;mm "
           "block with 2.7&#8202;mm fillets. Condim 4 bounds the spin torque of a point contact by "
           "\\(\\mu_\\text{spin} N\\) with \\(\\mu_\\text{spin} = \\mu\\,\\bar r_\\text{on}\\), the pads&#8217; twist-onset arm on "
           "the TPU tip at the policies&#8217; grip: 2.52, 2.74, 2.96 and 3.12&#8202;mm at 1, 3, 10 and 20&#8202;N "
           "(<code>twist_onset/tip_T2.jsonl</code>), 3.0&#8202;mm at the typical 12&#8202;N. "
           f"The skin is the presliding candidate kept on the native-compliance page (<code>{COMPLIANCE_PAGE}</code>): "
           "each tip&#8217;s pads sit on a child body joined by two slides tangent to the palmar face and a hinge about its "
           "normal, with Mindlin&#8217;s and Lubkin&#8217;s initial stiffness "
           "\\(k_t = 8Ga/(2-\\nu)\\), \\(k_\\theta = 16Ga^3/3\\) "
           f"on the pads&#8217; own patch at 12&#8202;N (\\(a\\) = {f1(1e3 * s.get('a', float('nan')), 2)}&#8202;mm, "
           f"{f1(s.get('kt', float('nan')) / 1e3, 0)}&#8202;kN/m and {f1(s.get('kth', float('nan')), 2)}&#8202;N&#8202;m/rad) "
           "and critical damping. Three changes from the bed version were needed on the hand: impratio 1000 diverges under "
           "the trainer&#8217;s ten solver iterations for the pads and the skin alike, so the skin keeps the env&#8217;s 10; "
           "MuJoCo-Warp threw the tool within 0.1&#8202;s at the bed&#8217;s armature (spring period \\(\\omega\\,\\Delta t\\) "
           "3.6), so the armature is set to \\(\\omega\\,\\Delta t\\) = 1 (0.26&#8202;kg on the slides, acting on the "
           "skin&#8217;s own micrometre motion only); and its broadphase needs 768 contact slots per world.</p>")
    cap = ("Contact models, training cost per 40&#8202;M-step run on one RTX 4070 Ti SUPER, and failed runs. Condim 4 was "
           "timed (five iterations) and is evaluated in the transfer matrix; the 24 GPU-hour cap fitted four trained arms "
           "of three seeds. GPU memory is the whole card, desktop included.")
    return txt + table(head, body, cap, text_cols=(0, 1))


def curves_section(X: Data):
    svg = svg_curves(X)
    cap = ("Deterministic evaluation of every second saved checkpoint (82 iterations, 4.0&#8202;M env steps) and the final "
           "one: rollouts held at 5&#8202;s (top) and their mean final cosine (bottom), against env steps (left) and "
           "training wall-clock time (right). Line: median of the seeds; band: their range. A gap in the bottom row is an "
           "evaluation in which no rollout held.")
    return figure(svg, cap, legend()) + curves_text(X)


def curves_text(X: Data):
    return ""


def final_section(X: Data):
    svg = svg_final(X)
    cap = ("Final cosine of each rollout of the final policies (64 per seed), held (filled) and dropped (hollow); bar: "
           "the seed&#8217;s held cosine.")
    out = figure(svg, cap, legend())
    h1 = ["contact model", "held", "held cos, median seed", "seed spread", "rollout spread", "reach 0.9 held",
          "step at 0.9"]
    h2 = ["contact model", "grip (N)", "peak force (N)", "penetration (mm)", "creep (mm/s)", "creep (&#176;/s)", "|&#916;a|",
          "contacts"]
    b1, b2 = [], []
    for a in ARMS:
        fs = X.finals(a)
        if not fs:
            continue
        hcs = [r["held_cos_mean"] for r in fs if r.get("held_cos_mean") is not None]
        md = lambda k: float(np.median([r[k] for r in fs if r.get(k) is not None])) if any(r.get(k) is not None for r in fs) else None  # noqa: E731
        b1.append([f"{swatch(a)}{LBL[a]}", frac(sum(r["n_held"] for r in fs), N_ROLL * len(fs)),
                   f1(float(np.median(hcs)), 3) if hcs else "&#8211;",
                   f1(max(hcs) - min(hcs), 3) if len(hcs) > 1 else "&#8211;", f1(md("held_cos_sd"), 3),
                   frac(sum(r["n_reach09_held"] for r in fs), N_ROLL * len(fs)), f1(md("t09_median"), 0)])
        b2.append([f"{swatch(a)}{LBL[a]}", f1(md("grip_N"), 1), f1(md("peak_force_N_mean"), 0), f1(md("pen_max_mm_mean"), 2),
                   f1(md("creep_mm_s_median"), 2), f1(md("creep_deg_s_median"), 2), f1(md("dact_held"), 2),
                   f1(md("contacts_per_step"), 1)])
    out += table(h1, b1, "Final policies, 64 rollouts per seed: held rollouts and the turn. Held and reach-0.9 counts "
                         "pool the seeds; the other columns are medians over seeds.")
    out += table(h2, b2, "Final policies over their held steps: forces, penetration, creep in the last second, action "
                         "change and contacts (definitions in Terms and metrics); medians over seeds.")
    return out


def transfer_section(X: Data):
    cap = ("Transfer of the final policies. Left: closed-loop evaluation in MuJoCo-Warp under every contact model "
           "(the condim-4 TPU mesh was not trained). Right: open-loop replay of the recorded finger targets from the onset "
           "of the turn in CPU MuJoCo (each arm&#8217;s own tips), Drake and Newton (the TPU block as a hydroelastic tip; "
           "Newton with \\(k_h\\) divided by the tip-tool effective mass and its friction rows at the pads&#8217; "
           "10&#8202;ms). Drake is one more discretization of the pressure law, not a ground truth.")
    return figure(svg_transfer(X), cap)


def robust_section(X: Data):
    cap = ("Held fraction of each arm&#8217;s final policies (pooled over seeds, 64 rollouts each) under one perturbation "
           "at a time: every geom&#8217;s sliding friction scaled, the tool&#8217;s mass and inertia scaled, the finger "
           "servos&#8217; position gain over the 2&#8211;10&#8202;N&#8202;m/rad range of the 2026-10-06 bench readbacks, "
           "a 1&#8202;ms physics step (the training step is 2&#8202;ms; the policy stays at 50&#8202;Hz), and the tool "
           "spawned with uniform noise of &#177;2&#8202;mm and &#177;5&#176;.")
    return figure(svg_robust(X), cap, legend())


def lede(X: Data):
    n = len(X.fin)
    if n < 12:
        done = ", ".join(f"{SHORT[a]} {len(X.finals(a))}/3" for a in ARMS)
        return (f"Training in progress: final policies evaluated so far {done}. The findings are written when all twelve "
                f"runs, the transfer matrix, the perturbations and the films are in.")
    return ""


def open_items(X: Data):
    items = []
    return "<ul class='open'>" + "".join(f"<li><b>{a}</b> {b}</li>" for a, b in items) + "</ul>" if items else \
        P.pending("Written with the findings.")


def byline():
    return ("D6 (<code>results/phase1/real_v1/20261006-sv1_u0308_b050_work_tip*</code>, "
            "<code>20261008-*_tpu2.7mesh_c4</code>, <code>20261008-*_tpu2.7skin</code>, scenes by "
            "<code>scripts/make_work_plant_runs.py</code>) &#183; PPO, 2,048 envs, 40&#8202;M steps, seeds 0&#8211;2 "
            "(<code>scripts/rl_contact_train_queue_20261008.sh</code>, runs <code>results/rl/20261008-d6_work_*_40M_s*</code>) "
            "&#183; evaluation <code>scripts/rl_contact_eval.py</code>, replays and films <code>scripts/rl_policy_replay.py</code> "
            "&#183; rows <code>docs/experiments/20261008-contact_model_policies/</code> &#183; builder "
            "<code>scripts/contact_model_policies_page.py</code> &#183; hand-off <code>" + NOTE + "</code> &#183; overview "
            f"<code>{OVERVIEW}</code>")


def render_tex(t):
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None)
             for m in P.TEX_RE.finditer(t)]
    svgs = iter(texsvg.render(items, cache_path=TEX_CACHE, scale=P.TEX_SCALE))
    return P.TEX_RE.sub(lambda m: next(svgs), t), len(items)


def main():
    X = Data()
    v = {"STYLE": P.style_block().replace("</style>", EXTRA_CSS + "</style>"), "BUILT": time.strftime("%Y-%m-%d %H:%M"),
         "BYLINE": byline()}
    v["LEDE"] = lede(X)
    v["GLOSSARY"] = glossary()
    v["ARMS"] = arms_section(X)
    v["FILMS"] = films(X)
    v["CURVES"] = curves_section(X)
    v["FINAL"] = final_section(X)
    v["TRANSFER"] = transfer_section(X)
    v["ROBUST"] = robust_section(X)
    v["OPEN"] = open_items(X)
    v["FOOTER"] = ("<p>Rebuild: <code>logs/20261001-hom_contact/venv/bin/python scripts/contact_model_policies_page.py</code>. "
                   "Rows: <code>docs/experiments/20261008-contact_model_policies/*.jsonl</code>; films "
                   "<code>docs/experiments/20261008-contact_model_policies/media/</code>.</p>")
    t = open(TPL).read()
    for k, val in v.items():
        t = t.replace("{{" + k + "}}", val)
    left = sorted(set(x.split("}}")[0] for x in t.split("{{")[1:]))
    if left:
        raise SystemExit(f"unfilled placeholders: {left}")
    t, n = render_tex(t)
    open(OUT, "w").write(t)
    print(f"formulas {n}; wrote {OUT} ({os.path.getsize(OUT) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
