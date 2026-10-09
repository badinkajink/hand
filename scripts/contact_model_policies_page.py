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
figure.films .cell p.filmlab{font-size:13px;margin:4px 0 0 0;line-height:1.35}
figure.films video,figure.films img{width:100%;border-radius:8px}
figure.films figcaption{grid-column:1/-1}
figure.sheet img{width:100%;border-radius:6px}
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
        self.watch = last_by(rows("watch.jsonl"), ("tag", "iteration"))
        self.verdict = {(r["tag"], r["iteration"]): r for r in last_by(rows("watch_verdicts.jsonl"), ("tag", "iteration"))}
        self.stops = {r["tag"]: r for r in rows("stops.jsonl")}
        self.pauses = []        # (t0, t1) epoch s: the training run was paused (checkpoint watch, short GPU jobs)
        pp = os.path.join(ROOT, "logs/20261008-contact_model_policies/watch_pauses.tsv")
        if os.path.exists(pp):
            for line in open(pp):
                f = line.rstrip("\n").split("\t")
                if len(f) == 5:
                    self.pauses.append((float(f[2]), float(f[3])))

    def paused(self, a, b):
        """Seconds of pause between epoch times a and b."""
        return sum(max(0.0, min(b, t1) - max(a, t0)) for t0, t1 in self.pauses)

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
        # the first logged iteration ends one iteration time after the start; pauses for the watch are not training
        c = self.cost.get(tag, {})
        return (w[it] - t0 - self.paused(t0, w[it]) + (c.get("s_per_it_median") or 0.0)) / 3600.0

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


SEED_SHAPE = {0: "circle", 1: "square", 2: "diamond"}


def _seed_lines(X: Data, arm, key, xkind):
    """{tag: [(x, value)]} of `key` (n_held | held_cos) at each evaluated checkpoint of each seed, x in M steps or
    training hours."""
    out = {}
    for seed, pts in X.curve(arm).items():
        for it, steps, nh, hc, cm, tag in pts:
            x = steps / 1e6 if xkind == "steps" else X.hours(tag, it)
            if x is None:
                continue
            out.setdefault(tag, []).append((x, nh if key == "n_held" else hc))
    return out


def _draw_seeds(out, X: Data, fx, fy, lines, arm, clamp=None):
    """One thin line per seed with the seed's marker shape; a stopped run ends in a cross."""
    for tag, pts in sorted(lines.items()):
        seed = int(tag[-1])
        ys = [None if v is None else (clamp(v) if clamp else v) for _, v in pts]
        _gap_path(out, fx, fy, [x for x, _ in pts], ys, COL[arm], dashed=(arm == "box"), width=1.6)
        for (x, _), y in zip(pts, ys):
            if y is not None:
                P._marker(out, fx(x), fy(y), COL[arm], shape=SEED_SHAPE.get(seed, "circle"), r=2.8,
                          title=f"{SHORT[arm]} s{seed}: {y:.3g}")
        if tag in X.stops and pts:
            x, y = pts[-1][0], next((v for v in reversed(ys) if v is not None), None)
            if y is not None:
                P._marker(out, fx(x) + 9, fy(y), COL[arm], shape="cross", r=4.5, title=f"{SHORT[arm]} s{seed} stopped")


def svg_curves(X: Data):
    if not X.ck:
        return P.pending("Checkpoint evaluations not written yet (ckpt_eval.jsonl).")
    W, H = 990, 600
    out = P._svg_open(W, H, "Held rollouts and held cosine of the deterministic policy at its checkpoints, one line per "
                            "seed and contact model, against environment steps and against training wall-clock time")
    hmax = max([X.hours(r["tag"], r["iteration"]) or 0 for r in X.ck] + [0.5])
    hx = [0, 1, 2, 3] if hmax <= 3.2 else [0, 1, 2, 3, 4]
    panels = [(70, 40, "steps", "n_held"), (570, 40, "hours", "n_held"), (70, 330, "steps", "held_cos"),
              (570, 330, "hours", "held_cos")]
    for x0, y0, xk, key in panels:
        xs_rng = (0, 41) if xk == "steps" else (0, max(hx))
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
            _draw_seeds(out, X, fx, fy, _seed_lines(X, arm, key, xk), arm)
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
        lab = (f'<text x="{x0 + w + 4}" y="{fy(v) + 4:.1f}" style="fill:var(--ink3)">0.9</text>' if v == 0.9 else
               f'<text x="{x0 - 8}" y="{fy(v) + 4:.1f}" text-anchor="end" style="fill:var(--ink3)">{v:+.1f}</text>')
        out.append(f'<line x1="{x0}" x2="{x0 + w}" y1="{fy(v):.1f}" y2="{fy(v):.1f}" style="stroke:var(--rule2)'
                   f'{";stroke-dasharray:2 4" if v == 0.9 else ""}"/>{lab}')
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
    H = y0 + ch * len(ARMS) + 20
    out = P._svg_open(W, H, "Held fraction of every training arm's final policies evaluated under each contact model in "
                            "MuJoCo-Warp and replayed open loop in CPU MuJoCo, Drake and Newton")
    for j, (kind, s) in enumerate(cols):
        cx = x0 + j * cw + (24 if kind == "rep" else 0)
        lab = SHORT[s] if kind == "mjw" else ENG_LBL[s].split()[0] + (" MuJoCo" if s == "mujoco" else "")
        out.append(f'<text x="{cx + cw / 2:.1f}" y="{y0 - 10}" text-anchor="middle" style="fill:var(--ink2)">{lab}</text>')
    out.append(f'<text x="{x0 + 2.5 * cw:.1f}" y="{y0 - 34}" text-anchor="middle" style="fill:var(--ink)">'
               f'MuJoCo-Warp, closed loop, 64 rollouts per seed</text>')
    nw = sorted({len([r for r in X.rep if r["dir"] == d and r["engine"] == e]) for d in {r["dir"] for r in X.rep}
                 for e in ENGINES} - {0})
    wl = f"{nw[0]}" if len(nw) == 1 else (f"{nw[0]}&#8211;{nw[-1]}" if nw else "&#8211;")
    out.append(f'<text x="{x0 + 5 * cw + 24 + 1.5 * cw:.1f}" y="{y0 - 34}" text-anchor="middle" style="fill:var(--ink)">'
               f'open-loop replay, {wl} worlds per seed</text>')
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
    out.append("</svg>")
    return "".join(out)


WATCH_SIG = [("ang_jerk_hold_median", "tool shaking in the hold (rad/s&#178;)", (1, 1000), (1, 10, 100, 1000), True, 1.0),
             ("dact_held", "|&#916;a| per policy step (action units)", (0, 0.8), (0, 0.2, 0.4, 0.6, 0.8), False, 1.0),
             ("ctrl_sat_frac", "servo targets at a range limit (%)", (0, 30), (0, 10, 20, 30), False, 100.0),
             ("share_min", "smallest finger share of the grip (%)", (0, 34), (0, 10, 20, 30), False, 100.0)]


def _watch_lines(X: Data, arm, key, scale):
    out = {}
    for r in sorted((r for r in X.watch if r["arm"] == arm), key=lambda r: r["iteration"]):
        v = r.get(key)
        out.setdefault(r["tag"], []).append(((r["iteration"] + 1) * STEPS_PER_IT / 1e6, None if v is None else v * scale))
    return out


def svg_watch(X: Data):
    if not X.watch:
        return P.pending("Checkpoint watch rows not written yet (watch.jsonl).")
    W, H = 990, 600
    out = P._svg_open(W, H, "Checkpoint watch signals per contact model against environment steps: the tool's angular "
                            "shaking in the hold, the action change per policy step, the servo targets at a range limit and "
                            "the smallest finger's share of the grip force")
    for k, (key, lab, ys, yt, logy, scale) in enumerate(WATCH_SIG):
        x0, y0 = (70, 570)[k % 2], (40, 330)[k // 2]
        if logy:
            fx, fy = P._panel(out, x0, y0, 360, 200, (0, 41), ys, (0, 10, 20, 30, 40), yt,
                              "environment steps (millions)", lab, logy=True, yfmt="{:g}")
            out.append(f'<line x1="{x0}" x2="{x0 + 360}" y1="{fy(40):.1f}" y2="{fy(40):.1f}" style="stroke:var(--ink3);'
                       f'stroke-dasharray:2 4"/><text x="{x0 + 362}" y="{fy(40) + 4:.1f}" style="fill:var(--ink3)">40</text>')
        else:
            fx, fy = P._panel(out, x0, y0, 360, 200, (0, 41), ys, (0, 10, 20, 30, 40), yt,
                              "environment steps (millions)", lab, yfmt="{:g}")
        clamp = lambda v: min(max(v, ys[0]), ys[1])  # noqa: E731
        for arm in ARMS:
            _draw_seeds(out, X, fx, fy, _watch_lines(X, arm, key, scale), arm, clamp=clamp)
    out.append("</svg>")
    return "".join(out)


def stop_outcome(X: Data, tag):
    """(iteration, reason) of the first watched checkpoint at which the stop rule fired, or None."""
    rs = sorted((r for r in X.watch if r["tag"] == tag), key=lambda r: r["iteration"])
    for r in rs:
        if r.get("stop_rule"):
            return r["iteration"], r.get("stop_reason", "")
    return None


def stops_table(X: Data):
    if not X.watch:
        return ""
    head = ["run", "evaluated to", "held at the last watch", "stop rule fired at", "reason", "outcome"]
    body = []
    tags = sorted({r["tag"] for r in X.watch}, key=lambda t: (t[-1], ARMS.index(tag_arm(t)) if tag_arm(t) else 9))
    for tag in tags:
        rs = sorted((r for r in X.watch if r["tag"] == tag), key=lambda r: r["iteration"])
        last = rs[-1]
        so = stop_outcome(X, tag)
        st = X.stops.get(tag)
        if st:
            outcome = f"stopped at {(st['iteration'] + 1) * STEPS_PER_IT / 1e6:.1f}&#8202;M"
        elif last["iteration"] >= 812:
            outcome = "trained to 40&#8202;M" + (" (watched after the run)" if tag.endswith("_s0") else "")
        else:
            outcome = "training"
        body.append([f"{swatch(tag_arm(tag))}{SHORT[tag_arm(tag)]} s{tag[-1]}",
                     f"{(last['iteration'] + 1) * STEPS_PER_IT / 1e6:.1f}&#8202;M",
                     f"{last['n_held']}/64" + (f", cos {last['held_cos_mean']:.2f}" if last.get("held_cos_mean") is not None else ""),
                     f"{(so[0] + 1) * STEPS_PER_IT / 1e6:.1f}&#8202;M" if so else "&#8211;",
                     so[1].replace("tool jerk", "shaking").replace("1/s^2", "rad/s&#178;").replace("->", "&#8594;") if so else "&#8211;", outcome])
    cap = ("The owner&#8217;s stopping rule applied at every watched checkpoint (4.0&#8202;M steps apart): stop when every "
           "checkpoint of the last 10&#8202;M steps is degenerate, or when the held cosine gained under 0.02 (and the held "
           "count under 8 of 64) over 10&#8202;M steps; the films decide. The seed-0 runs finished before the watch existed "
           "and were watched afterwards.")
    return table(head, body, cap, text_cols=(0, 4, 5))


TB_SIG = [("Train/mean_reward", "mean episode return", (0, 500), (0, 100, 200, 300, 400, 500)),
          ("Train/mean_episode_length", "mean episode length (policy steps, of 250)", (0, 260), (0, 50, 100, 150, 200, 250)),
          ("Episode_Termination/tip_lost", "tip-lost terminations per iteration (of 2,048 envs)", (0, 50), (0, 10, 20, 30, 40, 50)),
          ("Episode_Reward/target_axis_alignment", "alignment reward per episode", (0, 70), (0, 10, 20, 30, 40, 50, 60, 70))]


def _tb_lines(X: Data, arm, key, every=8, win=9):
    """{tag: [(M steps, value)]}: `key` per iteration smoothed over `win` iterations, every `every`-th iteration."""
    by_tag = {}
    for r in X.tb:
        if tag_arm(r["tag"]) == arm and r.get(key) is not None:
            by_tag.setdefault(r["tag"], {})[r["iteration"]] = r[key]
    out = {}
    for tag, d in by_tag.items():
        its = sorted(d)
        v = np.array([d[i] for i in its], float)
        vs = np.convolve(np.pad(v, (win // 2, win // 2), mode="edge"), np.ones(win) / win, mode="valid")
        out[tag] = [((i + 1) * STEPS_PER_IT / 1e6, float(x)) for i, x in zip(its, vs) if i % every == 0 or i == its[-1]]
    return out


def svg_tb(X: Data):
    if not X.tb:
        return P.pending("Training curves not extracted yet (tb_dynamics.jsonl).")
    W, H = 990, 600
    out = P._svg_open(W, H, "Training curves per contact model against environment steps: episode return, episode "
                            "length, tip-lost terminations and the alignment reward")
    for k, (key, lab, ys, yt) in enumerate(TB_SIG):
        x0, y0 = (70, 570)[k % 2], (40, 330)[k // 2]
        fx, fy = P._panel(out, x0, y0, 360, 200, (0, 41), ys, (0, 10, 20, 30, 40), yt, "environment steps (millions)",
                          lab, yfmt="{:g}")
        clamp = lambda v: min(max(v, ys[0]), ys[1])  # noqa: E731
        for arm in ARMS:
            for tag, pts in sorted(_tb_lines(X, arm, key).items()):
                _gap_path(out, fx, fy, [x for x, _ in pts], [clamp(v) for _, v in pts], COL[arm], dashed=(arm == "box"),
                          width=1.5)
                if tag in X.stops and pts:
                    P._marker(out, fx(pts[-1][0]) + 8, fy(clamp(pts[-1][1])), COL[arm], shape="cross", r=4.5,
                              title=f"{SHORT[arm]} s{tag[-1]} stopped")
    out.append("</svg>")
    return "".join(out)


SHEET_ITS = (82, 164, 328, 574, 812)


def watch_sheet(tag, its=SHEET_ITS):
    """The median rollout's strip at the checkpoints `its`, halved and stacked, as a JPEG data URI."""
    from io import BytesIO
    import base64
    from PIL import Image
    ims = []
    for it in its:
        p = os.path.join(D, "watch", tag, f"it{it:04d}_strip.jpg")
        if not os.path.exists(p):
            continue
        im = Image.open(p)
        w, h = im.size
        ims.append(im.crop((0, 0, w, h // 2)).resize((w // 2, h // 4)))
    if not ims:
        return ""
    sheet = Image.new("RGB", (ims[0].size[0], sum(i.size[1] for i in ims)), "white")
    y = 0
    for im in ims:
        sheet.paste(im, (0, y))
        y += im.size[1]
    buf = BytesIO()
    sheet.save(buf, "JPEG", quality=84)
    return (f'<img src="data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}" '
            f'alt="checkpoint strips of {tag}">')


def watch_sheets(X: Data):
    out = []
    for arm in ARMS:
        tags = sorted({r["tag"] for r in X.watch if r["arm"] == arm})
        if not tags:
            continue
        tag = tags[0]
        im = watch_sheet(tag)
        if not im:
            continue
        FIG[0] += 1
        notes = [X.verdict.get((tag, it)) for it in SHEET_ITS]
        seen = " ".join(f"{(it + 1) * STEPS_PER_IT / 1e6:.0f}&#8202;M: {v['seen']}." for it, v in zip(SHEET_ITS, notes) if v)
        out.append(f'<figure class="sheet">{im}<figcaption>Figure&#160;{FIG[0]}. {LBL[arm]}, seed {tag[-1]}: the median '
                   f'rollout of the 64 at the checkpoints of 4, 8, 16, 28 and 40&#8202;M steps (rows), at policy steps 21, '
                   f'58, 91, 131, 181 and 250 (columns). {seen} Every watched checkpoint&#8217;s strip and film: '
                   f'<code>docs/experiments/20261008-contact_model_policies/watch/{tag}/</code>.</figcaption></figure>')
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


def median_seed(X: Data, arm):
    """The final-evaluation row of an arm's median finished seed (held fraction, then held cosine), or None."""
    rs = sorted(X.finals(arm), key=lambda r: (r["held_frac"], r["held_cos_mean"] if r["held_cos_mean"] is not None else -2))
    return rs[(len(rs) - 1) // 2] if rs else None


def final_film(X: Data, arm):
    """(film, strip) paths relative to D for an arm's median seed: post_train.sh's close-up render when it exists, else
    the checkpoint watch's film of the final checkpoint."""
    r = median_seed(X, arm)
    if r is None:
        return None, None, None
    for role in ("median", "best"):
        f = next((f for f in X.films if f["arm"] == arm and f["role"] == role and f["tag"] == r["tag"]), None)
        if f and os.path.exists(os.path.join(D, f"media/{f['tag']}_{role}.mp4")):
            return r, f"media/{f['tag']}_{role}.mp4", f"media/{f['tag']}_{role}_strip.jpg"
    rel = f"watch/{r['tag']}/it{r['iteration']:04d}_median.mp4"
    if os.path.exists(os.path.join(D, rel)):
        return r, rel, f"watch/{r['tag']}/it{r['iteration']:04d}_strip.jpg"
    return r, None, None


def films(X: Data):
    cells = []
    for arm in ARMS:
        r, film, strip = final_film(X, arm)
        if film is None:
            continue
        hc = f", held cos {r['held_cos_mean']:.2f}" if r.get("held_cos_mean") is not None else ""
        cells.append(f'<div class="cell">{video(film)}<p class="filmlab">{swatch(arm)}<b>{LBL[arm]}</b>, seed {r["seed"]}: '
                     f'held {r["n_held"]}/64{hc}. {FILM_NOTES.get(arm, "")} <code>{film}</code></p></div>')
    if not cells:
        return P.pending("Films not rendered yet (post_train.sh: close-up renders of the best and median seed per arm "
                         "and three-engine replay films).")
    FIG[0] += 1
    out = [f'<figure class="films">{"".join(cells)}<figcaption>Figure&#160;{FIG[0]}. The final policy of each contact '
           f'model&#8217;s median seed, rollout with the median final cosine of 64, from the scripted grasp and lift '
           f'(0&#8211;1.16&#8202;s) through the policy&#8217;s turn and hold to 5&#8202;s; camera 0.24&#8202;m from the '
           f'hand, the turn in the image plane (thumb in front, index left, middle right). Paths relative to '
           f'<code>docs/experiments/20261008-contact_model_policies/</code>.</figcaption></figure>']
    for arm in ARMS:
        r = median_seed(X, arm)
        if r is None:
            continue
        rel = f"media/{r['tag']}_replay_three.mp4"
        if not os.path.exists(os.path.join(D, rel)):
            continue
        FIG[0] += 1
        out.append(f'<figure>{video(rel)}<figcaption>Figure&#160;{FIG[0]}. {LBL[arm]}, median seed, world 0: the '
                   f'MuJoCo-Warp rollout it trained on (left), and its finger targets from the onset replayed open loop in '
                   f'Drake (centre) and Newton (right) with the TPU block as a hydroelastic tip. '
                   f'<code>docs/experiments/20261008-contact_model_policies/{rel}</code></figcaption></figure>')
    return "".join(out)


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
    ("shaking", "Mean absolute change of the tool&#8217;s angular speed between consecutive policy steps, divided by the "
     "20&#8202;ms step, over the held steps of the hold after the turn (policy steps 150&#8211;250), rad/s&#178;; median "
     "over held rollouts. A tool resting in a still grip reads near 0; <code>trajectory_health</code> fails a policy for "
     "jitter above 40&#8202;rad/s&#178;."),
    ("checkpoint watch", "At every 82nd iteration (4.0&#8202;M env steps) the training run is paused, the checkpoint "
     "evaluated (64 rollouts) and its median and worst rollouts rendered as frame strips and a film, which are looked at "
     "before the run continues; the owner&#8217;s rule stops a run that is degenerate for 10&#8202;M steps (drops every "
     "rollout, an idle finger, jitter, saturated actions, or the tool on the palm) or whose held cosine gains under 0.02 "
     "over 10&#8202;M steps. Jitter counts when the shaking exceeds 40&#8202;rad/s&#178; and the consecutive-step frames "
     "show the tool moving in the grip."),
    ("servo targets at a limit", "Share of the nine finger servo targets (anchor plus residual) at their actuator&#8217;s "
     "range limit over the active steps; the residual itself is not clipped in these runs."),
    ("finger share", "One fingertip&#8217;s share of the summed fingertip-tool force over the held steps."),
    ("grip change per step", "Median absolute change of the summed fingertip-tool force between consecutive policy "
     "steps over the held steps of the hold (steps 150&#8211;250), N; a steady grip reads near 0."),
    ("open-loop replay", "From the state at policy step 58 of a MuJoCo-Warp rollout, the policy&#8217;s recorded finger "
     "targets are played into another simulator at 50&#8202;Hz with the palm welded at its lifted pose "
     "(<code>scripts/rl_policy_replay.py</code>); the policy does not see that simulator&#8217;s state."),
]


def glossary():
    return '<dl class="glossary">' + "".join(f"<dt>{a}</dt><dd>{b}</dd>" for a, b in GLOSSARY) + "</dl>"


PROBE_TAG = {"box": "a_box", "tpu27mesh": "b_mesh", "tpu27pads1": "c_pads1", "tpu27meshc4": "d_meshc4", "tpu27skin": "e_skin"}


def probe_spit(arm):
    """Median s/it of the five-iteration probe run of 2026-10-08 (idle GPU, before the queue)."""
    import re
    p = os.path.join(ROOT, f"logs/20261008-contact_model_policies/train_20261008-d6_work_probe5_{PROBE_TAG[arm]}.log")
    if not os.path.exists(p):
        return None
    its = [float(x) for x in re.findall(r"Iteration time: ([0-9.]+)s", open(p, errors="replace").read())]
    return float(np.median(its)) if its else None


def run_hours(X: Data, tag):
    """Training wall-clock hours of a run with the watch pauses removed (queue log start to end)."""
    import re
    q = open(os.path.join(ROOT, "logs/20261008-contact_model_policies/train_queue.log")).read()
    st = re.findall(rf"(\S+ \S+) start {re.escape(tag)} ", q)
    en = re.findall(rf"(\S+ \S+) (?:end|FAILED) {re.escape(tag)} ", q)
    if not st or not en:
        return None
    t0, t1 = (time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S")) for x in (st[-1], en[-1]))
    return (t1 - t0 - X.paused(t0, t1)) / 3600.0


def arms_section(X: Data):
    sk = json.load(open(os.path.join(P.ROOT, "results/phase1/real_v1/20261008-sv1_u0308_b050_work_tip_tpu2.7skin/summary.json")))
    s = sk.get("fingertip", {}).get("skin", {})
    head = ["contact model", "contact", "s per iteration, idle GPU", "s per iteration, runs", "GPU-h per 40&#8202;M",
            "GPU-h spent", "GPU memory (GB)", "failed runs"]
    body = []
    contact = {"box": "point, condim 3", "tpu27mesh": "point, condim 3", "tpu27meshc4": "point, condim 4",
               "tpu27pads1": "1,060 spheres per tip", "tpu27skin": "the pads on a sprung skin"}
    for a in SCENES:
        cs = [c for t, c in X.cost.items() if tag_arm(t) == a]
        spi = [c["s_per_it_median"] for c in cs if c.get("s_per_it_median")]
        h40 = [run_hours(X, c["tag"]) for c in cs if c.get("finished")]
        spent = [run_hours(X, c["tag"]) for c in cs]
        gpu = [c.get("gpu_mem_proc_mb") or c.get("gpu_mem_peak_mb") for c in cs if c.get("gpu_mem_proc_mb") or c.get("gpu_mem_peak_mb")]
        fl = sum(c.get("failed_attempts", 0) for c in cs if not c.get("stopped_at"))
        trained = a in ARMS
        body.append([f"{swatch(a) if trained else ''}{LBL[a]}", contact[a], f1(probe_spit(a), 2),
                     f1(float(np.median(spi)), 2) if spi else "&#8211;",
                     f1(float(np.median([h for h in h40 if h])), 2) if any(h40) else ("not trained" if not trained else "&#8211;"),
                     f1(sum(h for h in spent if h), 2) if any(spent) else "&#8211;",
                     f1(max(gpu) / 1e3, 1) if gpu else "&#8211;", str(fl) if trained else "&#8211;"])
    tot = sum(h for c in X.cost.values() if (h := run_hours(X, c["tag"])))
    txt = ("<p>All arms train the same hand, plant, tool, solver, recipe and flags and differ only in the fingertip. The "
           "box tip is the fingertip every policy of this program trained on until 2026-10-05. The TPU arms replace it by "
           "the printed tip, a 17&#215;14.8&#215;22&#8202;mm block with 2.7&#8202;mm fillets, as one convex mesh with "
           "point contact, or as the 1&#8202;mm sphere pads of the overview page "
           f"(<code>{OVERVIEW}</code>). The skin mounts each tip&#8217;s pads on a child body with two tangent slides and "
           "a hinge about the palmar normal, sprung by Mindlin&#8217;s and Lubkin&#8217;s initial stiffness "
           "\\(k_t = 8Ga/(2-\\nu)\\) and \\(k_\\theta = 16Ga^3/3\\) "
           f"on the pads&#8217; patch at 12&#8202;N (\\(a\\) = {f1(1e3 * s.get('a', float('nan')), 2)}&#8202;mm, "
           f"{f1(s.get('kt', float('nan')) / 1e3, 0)}&#8202;kN/m, {f1(s.get('kth', float('nan')), 2)}&#8202;N&#8202;m/rad) "
           f"and critically damped; it is the presliding candidate kept on the native-compliance page "
           f"(<code>{COMPLIANCE_PAGE}</code>). On the hand it runs at the env&#8217;s impratio 10 (1000 diverges under ten "
           "solver iterations) with the slide armature set to \\(\\omega\\,\\Delta t = 1\\). Condim 4, a point "
           "contact whose spin torque is bounded by \\(\\mu_\\text{spin} N\\) with \\(\\mu_\\text{spin}\\) = 3.0&#8202;mm "
           "(the pads&#8217; twist-onset arm at 10&#8211;20&#8202;N, <code>twist_onset/tip_T2.jsonl</code>), is evaluated "
           f"in the transfer matrix but not trained. Training used {tot:.1f} of the 24 GPU-hours budgeted.</p>")
    cap = ("Contact models and training cost on one RTX 4070 Ti SUPER. Idle GPU: median of a five-iteration probe before the "
           "queue. Runs: median over each arm&#8217;s runs, which shared the workstation with other jobs (an Isaac Lab "
           "evaluation on about ten CPU cores for most of the night). GPU-hours exclude the watch&#8217;s pauses; "
           "&#8216;spent&#8217; includes the runs stopped early. GPU memory: the training process.")
    return txt + table(head, body, cap, text_cols=(0, 1))


def curves_section(X: Data):
    svg = svg_curves(X)
    cap = ("Deterministic evaluation of every second saved checkpoint (82 iterations, 4.0&#8202;M env steps) and the final "
           "one: rollouts held at 5&#8202;s (top) and their mean final cosine (bottom), against env steps (left) and "
           "training wall-clock time (right). One line per seed (marker: circle s0, square s1, diamond s2); a cross "
           "ends a run stopped by the checkpoint watch. A gap in the bottom row is an evaluation in which no rollout held.")
    out = figure(svg, cap, legend()) + curves_text(X)
    wcap = ("Signals of the checkpoint watch at the same checkpoints: the tool&#8217;s shaking over the hold after the "
            "turn (policy steps 150&#8211;250; dashed line: trajectory_health&#8217;s jitter limit), the mean action change "
            "per policy step over the held steps, the share of finger servo targets at their actuator&#8217;s range limit, "
            "and the smallest of the three fingers&#8217; shares of the grip force. Medians over held rollouts; one line "
            "per seed as above.")
    tcap = ("Training curves from the event files, smoothed over nine iterations: the stochastic policy&#8217;s mean "
            "episode return and length, tip-lost terminations (a fingertip off the tool for 15 steps ends the episode) and "
            "the alignment term of the return. One line per seed; a cross ends a stopped run.")
    out += figure(svg_tb(X), tcap, legend())
    out += figure(svg_watch(X), wcap, legend()) + stops_table(X) + watch_sheets(X)
    return out


def first_hold(X: Data, tag, n=60):
    """Env steps (M) of the first evaluated checkpoint after the initial one at which at least `n` of 64 rollouts held,
    or None."""
    rs = sorted((r for r in X.ck if r["tag"] == tag and r["iteration"] > 0), key=lambda r: r["iteration"])
    return next(((r["env_steps"] / 1e6) for r in rs if r["n_held"] >= n), None)


def curves_text(X: Data):
    parts = []
    for arm in ARMS:
        tags = sorted({r["tag"] for r in X.ck if r["arm"] == arm})
        if not tags:
            continue
        fh = []
        for t in tags:
            v = first_hold(X, t)
            fh.append(f"s{t[-1]} {v:.0f}&#8202;M" if v is not None else
                      f"s{t[-1]} not by {max(r['env_steps'] for r in X.ck if r['tag'] == t) / 1e6:.0f}&#8202;M")
        parts.append(f"{SHORT[arm]} {', '.join(fh)}")
    if not parts:
        return ""
    return ("<p>The checkpoint at which a run first held at least 60 of 64 rollouts: " + "; ".join(parts) + ". An "
            "episode ends 15 steps after a fingertip leaves the tool, so a policy that drops the tool collects the return "
            "of about 75 steps (Figure " + str(FIG[0] + 1) + "); the box tip&#8217;s seed 0 stayed there for 28&#8202;M "
            "steps.</p>")


def svg_traces(X: Data):
    """cos(t) of the 64 rollouts of each arm's median seed (final policy), held at 5 s in the arm's colour, dropped in
    grey; grip force (sum of the three tips) below."""
    W, H = 990, 470
    out = P._svg_open(W, H, "Tool cosine against time in each of the 64 rollouts of every contact model's median-seed final "
                            "policy, and the summed fingertip force")
    pw, gap, x0 = 210, 32, 60
    drawn = 0
    for k, arm in enumerate(ARMS):
        r = median_seed(X, arm)
        p = os.path.join(D, "final_traces", f"{r['tag']}.npz") if r else None
        if not p or not os.path.exists(p):
            continue
        tr = np.load(p)
        cos, z = tr["cos"].astype(float), tr["z"].astype(float)
        f = tr["force"].astype(float)
        T = cos.shape[0]
        held = np.array(r["held_final"])
        t = (np.arange(T) + 1) * 0.02
        px = x0 + k * (pw + gap)
        fx, fy = P._panel(out, px, 40, pw, 200, (0, 5), (-1, 1), (0, 1, 2, 3, 4, 5), (-1, -0.5, 0, 0.5, 1),
                          "time (s)", f"{SHORT[arm]} s{r['seed']}: cos" if k == 0 else f"{SHORT[arm]} s{r['seed']}",
                          yfmt="{:g}")
        out.append(f'<line x1="{fx(1.16):.1f}" x2="{fx(1.16):.1f}" y1="40" y2="240" style="stroke:var(--ink3);'
                   f'stroke-dasharray:2 4"/>')
        for e in np.argsort(held):                       # dropped first, held on top
            pts = list(zip(t[::3], cos[::3, e]))
            P._path(out, fx, fy, pts, COL[arm] if held[e] else "var(--bad)", width=0.9)
        fx2, fy2 = P._panel(out, px, 300, pw, 110, (0, 5), (0, 150), (0, 1, 2, 3, 4, 5), (0, 50, 100, 150), "time (s)",
                            "grip (N)" if k == 0 else "", yfmt="{:g}")
        g = f.sum(-1)
        for e in np.argsort(held):
            P._path(out, fx2, fy2, list(zip(t[::3], np.minimum(g[::3, e], 150))), COL[arm] if held[e] else "var(--bad)",
                    width=0.7)
        drawn += 1
    out.append("</svg>")
    return "".join(out) if drawn else P.pending("Final traces not written yet (final_traces/).")


JN9 = [f"{f} {j}" for f in ("thumb", "index", "middle") for j in ("yaw", "mcp", "pip")]


def pinned_desc(tag, it):
    """Servo targets at a range limit in 90 % of the active steps of most rollouts, with the limit's side (the lower
    limit of an mcp or pip joint is its extension limit), from the watch traces of checkpoint `it`."""
    p = os.path.join(ROOT, "logs/20261008-contact_model_policies/watch_traces", f"{tag}_it{it:04d}.npz")
    if not os.path.exists(p):
        return None
    tr = np.load(p)
    c = tr["ctrl"].astype(float)[int(tr["residual_from"]):]
    lo, hi = tr["ctrl_lo"], tr["ctrl_hi"]
    out = []
    for j in range(9):
        at_lo = ((c[:, :, j] <= lo[j] + 1e-3).mean(0) > 0.9).mean()
        at_hi = ((c[:, :, j] >= hi[j] - 1e-3).mean(0) > 0.9).mean()
        if max(at_lo, at_hi) > 0.5:
            side = ("extension" if at_lo > at_hi else "flexion") if "yaw" not in JN9[j] else ("lower" if at_lo > at_hi else "upper")
            out.append(f"{JN9[j]} at its {side} limit")
    return out


def grip_chatter(tag):
    """Median |change of the summed fingertip force| between consecutive policy steps over the hold (steps 150-249),
    over the held steps of all rollouts of the final evaluation, N."""
    p = os.path.join(D, "final_traces", f"{tag}.npz")
    if not os.path.exists(p):
        return None
    tr = np.load(p)
    fo, z = tr["force"].astype(float), tr["z"].astype(float)
    held = (z > 0.06) & ((fo >= 0.24).sum(-1) >= 2)
    g = fo.sum(-1)[150:]
    h = held[150:]
    m = h[1:] & h[:-1]
    dg = np.abs(np.diff(g, axis=0))[m]
    return float(np.median(dg)) if dg.size else None


def final_section(X: Data):
    svg = svg_final(X)
    cap = ("Final cosine of each rollout of the final policies (64 per seed), held (filled) and dropped (hollow); bar: "
           "the seed&#8217;s held cosine.")
    out = figure(svg, cap, legend())
    out += figure(svg_traces(X), "The 64 rollouts of each contact model&#8217;s median-seed final policy: the tool&#8217;s "
                  "cosine with vertical (top) and the summed fingertip force on it (bottom, clipped at 150&#8202;N) against "
                  "time; rollouts that held the tool at 5&#8202;s in the arm&#8217;s colour, the others in red. Dashed: the residual "
                  "policy&#8217;s onset after the scripted grasp and lift (step 58).")
    h1 = ["contact model", "held", "held cos, median seed", "seed spread", "rollout spread", "reach 0.9 held",
          "step at 0.9"]
    h2 = ["contact model", "grip (N)", "grip change per step (N)", "peak force (N)", "penetration (mm)", "creep (mm/s)",
          "creep (&#176;/s)", "shaking (rad/s&#178;)", "|&#916;a|", "pinned targets"]
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
        wf = [w for w in X.watch if w["tag"] in {r["tag"] for r in fs} and w["iteration"] == fs[0]["iteration"]]
        shk = [w["ang_jerk_hold_median"] for w in wf if w.get("ang_jerk_hold_median") is not None]
        pin = sorted({d for r in fs for d in (pinned_desc(r["tag"], r["iteration"]) or [])})
        gc = [v for v in (grip_chatter(r["tag"]) for r in fs) if v is not None]
        b2.append([f"{swatch(a)}{LBL[a]}", f1(md("grip_N"), 1), f1(float(np.median(gc)), 1) if gc else "&#8211;",
                   f1(md("peak_force_N_mean"), 0), f1(md("pen_max_mm_mean"), 2),
                   f1(md("creep_mm_s_median"), 2), f1(md("creep_deg_s_median"), 2),
                   f1(float(np.median(shk)), 0) if shk else "&#8211;", f1(md("dact_held"), 2),
                   ", ".join(pin) if pin else "none"])
    out += table(h1, b1, "Final policies, 64 rollouts per seed: held rollouts and the turn. Held and reach-0.9 counts "
                         "pool the seeds; the other columns are medians over seeds.")
    out += table(h2, b2, "Final policies over their held steps: forces, penetration, creep in the last second, shaking "
                         "in the hold, action change, and the servo targets held at their range limit in 90&#8202;% of "
                         "the steps (definitions in Terms and metrics); medians over seeds.", text_cols=(0, 9))
    return out


def transfer_section(X: Data):
    cap = ("Transfer of the final policies. Cell: rollouts held at the end (load test), pooled over seeds, and the mean "
           "final cosine of the held ones; outlined, the contact model the policy trained on. "
           "Left: closed-loop evaluation in MuJoCo-Warp under every contact model "
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


def summary(X: Data):
    """Per arm: finished and stopped seeds, pooled held counts and held cosines of the final policies, open-loop replay
    counts per engine, the tool's angular jerk in the hold at the last watched checkpoint, and the median s/it."""
    out = {}
    for arm in ARMS:
        fins = X.finals(arm)
        rep = {e: X.replays(arm, e) for e in ENGINES}
        last = {}
        for r in X.watch:
            if r["arm"] == arm and (r["tag"] not in last or r["iteration"] > last[r["tag"]]["iteration"]):
                last[r["tag"]] = r
        fin_tags = {r["tag"] for r in fins}
        jerk_fin = [r.get("ang_jerk_hold_median") for t, r in last.items() if t in fin_tags and r.get("ang_jerk_hold_median")]
        cs = [c for t, c in X.cost.items() if tag_arm(t) == arm and c.get("s_per_it_median")]
        out[arm] = dict(
            n_fin=len(fins), stopped=sorted(t for t in X.stops if tag_arm(t) == arm),
            held=sum(r["n_held"] for r in fins), n=sum(r["n"] for r in fins),
            hcos=[r["held_cos_mean"] for r in fins if r.get("held_cos_mean") is not None],
            rep={e: (sum(1 for r in v if r.get("held_end")), len(v)) for e, v in rep.items()},
            jerk=jerk_fin, spit=float(np.median([c["s_per_it_median"] for c in cs])) if cs else None,
            grip=[r["grip_N"] for r in fins if r.get("grip_N") is not None],
            creep=[r["creep_mm_s_median"] for r in fins if r.get("creep_mm_s_median") is not None])
    return out


def _rng(v, nd=2):
    if not v:
        return "&#8211;"
    lo, hi = min(v), max(v)
    return f"{lo:.{nd}f}" if f"{lo:.{nd}f}" == f"{hi:.{nd}f}" else f"{lo:.{nd}f}&#8211;{hi:.{nd}f}"


def lede(X: Data):
    S = summary(X)
    if not all(S[a]["n_fin"] for a in ARMS):
        done = ", ".join(f"{SHORT[a]} {S[a]['n_fin']}" for a in ARMS)
        return f"Training in progress: finished seeds per contact model {done}."
    comp = [a for a in ("tpu27pads1", "tpu27skin")]
    point = [a for a in ("box", "tpu27mesh")]
    rep = lambda arms: [sum(S[a]["rep"][e][k] for a in arms for e in ENGINES) for k in (0, 1)]  # noqa: E731
    rc, rp = rep(comp), rep(point)
    parts = [f"Replayed open loop in CPU MuJoCo, Drake and Newton, the finger targets of the policies trained on the "
             f"1&#8202;mm sphere pads and on the pads mounted on a sprung skin kept the screwdriver in "
             f"{rc[0]} of {rc[1]} replays, and those trained with MuJoCo point contact on the box tip or the TPU block "
             f"mesh in {rp[0]} of {rp[1]}."]
    jp = [j for a in point for j in S[a]["jerk"]]
    jc = [j for a in comp for j in S[a]["jerk"]]
    stopped = [t for a in point for t in S[a]["stopped"]]
    if jp and jc:
        parts.append(f"In their own simulator the point-contact policies shake the tool: the mean change of its angular "
                     f"speed over the hold is {_rng(jp, 0)}&#8202;rad/s&#178; at 40&#8202;M steps against "
                     f"{_rng(jc, 0)}&#8202;rad/s&#178; for "
                     f"the pads and the skin" + (f", and the checkpoint watch stopped {len(stopped)} of their later "
                     f"seeds at 16&#8202;M steps for jitter." if stopped else "."))
    hb, hm, hp, hs = (S[a]["hcos"] for a in ARMS)
    parts.append(f"They turn further where they hold: final held cosine {_rng(hb)} for the box tip "
                 f"({S['box']['held']}/{S['box']['n']} rollouts held) and {_rng(hm)} for the TPU mesh, against "
                 f"{_rng(hp)} for the pads and {_rng(hs)} for the skin, all of whose rollouts held "
                 f"({S['tpu27pads1']['held'] + S['tpu27skin']['held']}/{S['tpu27pads1']['n'] + S['tpu27skin']['n']}).")
    if S["tpu27skin"]["spit"] and S["tpu27mesh"]["spit"]:
        parts.append(f"A training iteration costs {S['tpu27pads1']['spit'] / S['tpu27mesh']['spit']:.1f}&#215; the "
                     f"TPU mesh&#8217;s GPU time with the pads and {S['tpu27skin']['spit'] / S['tpu27mesh']['spit']:.1f}"
                     f"&#215; with the skin.")
    return " ".join(parts)


def open_items(X: Data):
    items = [
        ("Bench replay of the finger targets.", "The replays test the policies in three simulators and none of them is the "
         "printed fingertip. The next measurement plays the recorded finger targets of the median rollout of the pads, "
         "skin and TPU-mesh final policies on the hand at 50&#8202;Hz from the lifted grasp (a plan exported from "
         "<code>logs/20261008-contact_model_policies/replay/&lt;tag&gt;/rec.npz</code> through the real_v1 deploy "
         "format, 10 trials each, tracked). The result that contradicts this page: the TPU-mesh targets hold the tool as "
         "often as the pad or skin targets."),
        ("Grip force.", "The recipe copied from the 2026-09-17 60&#8202;M run gives the grip term a weight of +0.25, a small "
         "bonus above 4&#8202;N per pad, and every final policy grips with 40&#8211;100&#8202;N on a 24.5&#8202;g tool. "
         "Fine-tune the pads and skin finals with weight &#8722;5 (the 2026-09-17 finetune setting) and repeat the "
         "replays; the comparison holds if the replays still keep the tool at a grip near 10&#8202;N."),
        ("Unclipped residual.", "The runs train without an action clip: actions reach 3 units (1.5&#8202;rad past the anchor) "
         "and one pip servo target per compliant policy sits at its extension limit in 95&#8202;% of the steps (Table 4). A "
         "deployable residual needs <code>clip_actions</code> 1.0 (<code>src/morphohand/rl/ppo_config.py</code>); "
         "retrain one seed per compliant arm with it and compare the held cosine."),
        ("Seeds of the point-contact arms.", "The watch stopped point-contact runs at 16&#8202;M steps for jitter, so those "
         "arms have fewer seeds at 40&#8202;M than the compliant ones; their seed spread rests on the runs that "
         "finished."),
    ]
    return "<ul class='open'>" + "".join(f"<li><b>{a}</b> {b}</li>" for a, b in items) + "</ul>"


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
