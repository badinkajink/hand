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
box tip as the dashed ink reference; checked with the reference validator's formulas (all pairs: CVD Delta E >= 9.2,
normal >= 20.9; aqua is 2.8:1 on white, so every series is also labelled in text and tables). The page is written in the
plain light style of scripts/retro_style.py (owner, 2026-10-09), and its charts set text at 16 units on a 990-unit
viewBox, 13.4 px in the 828 px column.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import contact_overview_page as P  # noqa: E402
import retro_style  # noqa: E402
import texsvg  # noqa: E402

D = os.path.join(P.EXP, "20261008-contact_model_policies")
OUT = os.path.join(D, "20261008-fingertip_contact_model_policies.html")
TPL = os.path.join(ROOT, "scripts/contact_model_policies_page.template.html")
TEX_CACHE = os.path.join(D, "texsvg_cache.json")
NOTE = "docs/handoff/20261007-contact_model_policy_training.md"
COMPLIANCE_PAGE = "docs/experiments/20261007-native_compliance/20261007-native_presliding_compliance.html"
TURN_PAGE = "docs/experiments/20261006-hom_turn3/20261006-servo_refit_hom_turn_pad_cost.html"
OVERVIEW = "docs/experiments/20261009-contact_overview/20261009-sphere_pad_contact_model.html"
fmt, num = P.fmt, P.num
ARMS = ("box", "tpu27mesh", "tpu27pads1", "tpu27skin")
SCENES = ("box", "tpu27mesh", "tpu27meshc4", "tpu27pads1", "tpu27skin")
ENGINES = ("mujoco", "drake", "newton")
LBL = {"box": "Box tip, point contact", "tpu27mesh": "TPU block, point contact", "tpu27meshc4": "TPU block, condim 4",
       "tpu27pads1": "TPU block, 1&#8202;mm pads", "tpu27skin": "TPU block, pads on a skin"}
SHORT = {"box": "box", "tpu27mesh": "TPU mesh", "tpu27meshc4": "condim 4", "tpu27pads1": "pads", "tpu27skin": "skin"}
CAP = {"box": "Box tip", "tpu27mesh": "TPU mesh", "tpu27meshc4": "Condim 4", "tpu27pads1": "Pads", "tpu27skin": "Skin"}
PANEL = {"box": "Box tip", "tpu27mesh": "TPU mesh", "tpu27pads1": "1&#8202;mm pads", "tpu27skin": "Pads on a skin"}
COL = {"box": "var(--a-box)", "tpu27mesh": "var(--a-mesh)", "tpu27pads1": "var(--a-pads)", "tpu27skin": "var(--a-skin)"}
ENG_LBL = {"mujoco": "CPU MuJoCo", "drake": "Drake hydroelastic", "newton": "Newton hydroelastic"}
STEPS_PER_IT = 2048 * 24
N_ROLL = 64
FIG, TAB = [0], [0]
EXTRA_CSS = """
:root{--a-box:#48545D;--a-mesh:#2a78d6;--a-pads:#eb6834;--a-skin:#1baf7a;--hm0:#eef3f9;--hm1:#cde2fb;--hm2:#9ec5f4;
--hm3:#6da7ec;--hm4:#3987e5;--hm5:#256abf;--hm6:#184f95}
.col > figure,.col > .tw{width:auto !important;max-width:100% !important}
figure.films{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}
figure.films.three{grid-template-columns:repeat(3,minmax(0,1fr))}
figure.films .cell,figure.replays .cell{min-width:0}
figure.replays .cell{margin-bottom:14px}
figure.films video,figure.films img,figure.replays video{width:100%;height:auto}
figure .cell p.filmlab{font-size:16px !important;line-height:1.35 !important;text-align:left !important;margin:4px 0 0 0}
figure .cell p.filmlab code{overflow-wrap:anywhere}
figure.films figcaption{grid-column:1/-1}
figure.diagram .legendrow{justify-content:center}
@media (max-width:640px){figure.films,figure.films.three{grid-template-columns:minmax(0,1fr)}}
details.sheets{margin:24px 0;border:1px solid #000;padding:8px 12px}
details.sheets summary{cursor:pointer;font-size:17px}
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
        # one row per run, engine and world (the newest; a run's directory was written relative once)
        self.rep = last_by([dict(r, dir=os.path.basename(r["dir"])) for r in rows("policy_replay.jsonl")
                            if r.get("engine") and r.get("kind") != "record"], ("dir", "engine", "world", "hold_test_s"))
        self.rec = [r for r in rows("policy_replay.jsonl") if r.get("kind") == "record"]
        self.wall = {}
        for r in self.tb:
            self.wall.setdefault(r["tag"], {})[r["iteration"]] = r.get("wall_time")
        p = os.path.join(D, "films.json")
        self.films = json.load(open(p)) if os.path.exists(p) else []
        self.watch = last_by(rows("watch.jsonl"), ("tag", "iteration"))
        self.verdict = {(r["tag"], r["iteration"]): r for r in last_by(rows("watch_verdicts.jsonl"), ("tag", "iteration"))}
        self.stops = {r["tag"]: r for r in rows("stops.jsonl")}
        # stopped as degenerate (no final policy); a plateau stop's last checkpoint is the run's final policy
        self.degen = {t: r for t, r in self.stops.items() if not r.get("reason", "").startswith("plateau")}
        self.pauses = []        # (t0, t1) epoch s: the training run was paused (checkpoint watch, short GPU jobs)
        self.live = set()       # runs watched while they trained (the others were watched after their run)
        pp = os.path.join(ROOT, "logs/20261008-contact_model_policies/watch_pauses.tsv")
        if os.path.exists(pp):
            for line in open(pp):
                f = line.rstrip("\n").split("\t")
                if len(f) == 5:
                    self.pauses.append((float(f[2]), float(f[3])))
                    if ":" not in f[0]:
                        self.live.add(f[0])

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

    def replays(self, arm, engine, stopped=False):
        """Replay rows of an arm's finished runs (or, with `stopped`, of its runs stopped early)."""
        return [r for r in self.rep if r["engine"] == engine and f"_{arm}_40M_" in r["dir"]
                and ((r["dir"] in self.degen) == stopped) and r.get("hold_test_s") is None]


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


def _arm_median(X: Data, arm, key, xkind):
    """Per evaluated iteration of an arm: (x, median, min, max, seeds) of `key` (n_held | held_cos) over the seeds
    evaluated there; x in M steps or training hours (the seeds' median)."""
    by_it = {}
    for seed, pts in X.curve(arm).items():
        for it, steps, nh, hc, cm, tag in pts:
            x = steps / 1e6 if xkind == "steps" else X.hours(tag, it)
            by_it.setdefault(it, []).append((x, nh if key == "n_held" else hc))
    out = []
    for it in sorted(by_it):
        xs = [x for x, _ in by_it[it] if x is not None]
        vs = [v for _, v in by_it[it] if v is not None]
        if xs:
            out.append((float(np.median(xs)), float(np.median(vs)) if vs else None, min(vs) if vs else None,
                        max(vs) if vs else None, len(by_it[it])))
    return out


def svg_curves(X: Data):
    """Top: one panel per contact model, the held cosine of each seed's checkpoints against env steps (marker filled when
    at least 60 of 64 rollouts held, hollow when fewer, a tick on the floor when none). Bottom: the median over seeds and
    the seeds' range of every contact model against training wall-clock time."""
    if not X.ck:
        return P.pending("Checkpoint evaluations not written yet (ckpt_eval.jsonl).")
    W, H = 990, 640
    out = P._svg_open(W, H, fs=16, label="Held cosine of each seed's checkpoints per contact model against environment steps, and the "
                            "median over seeds of the held cosine and the held rollouts against training wall-clock time")
    pw = 185
    for k, arm in enumerate(ARMS):
        x0 = 62 + k * 236
        fx, fy = P._panel(out, x0, 40, pw, 200, (0, 41), (-0.2, 1.0), (0, 10, 20, 30, 40), (0.0, 0.25, 0.5, 0.75, 1.0),
                          "Env steps (M)", PANEL[arm], yfmt="{:.2f}" if k == 0 else (lambda v: ""))
        out.append(f'<line x1="{x0}" x2="{x0 + pw}" y1="{fy(0.9):.1f}" y2="{fy(0.9):.1f}" '
                   f'style="stroke:var(--ink3);stroke-dasharray:2 4"/>')
        off = {0: -0.7, 1: 0.0, 2: 0.7}
        for r in X.ck:              # 10th-90th percentile of the held rollouts' final cosine, behind the markers
            if r["arm"] != arm or not r["n_held"] or r["iteration"] == 0:
                continue
            held = [c for c, h in zip(r.get("final_cos") or [], r.get("held_final") or []) if h]
            if len(held) >= 2:
                q10, q90 = np.percentile(held, [10, 90])
                xb = fx(r["env_steps"] / 1e6 + off.get(r["seed"], 0.0))
                out.append(f'<line x1="{xb:.1f}" x2="{xb:.1f}" y1="{fy(q10):.1f}" y2="{fy(q90):.1f}" style="stroke:'
                           f'{COL[arm]};stroke-width:3;stroke-opacity:.35"/>')
        for seed, pts in sorted(X.curve(arm).items()):
            tag = pts[0][5]
            xs = [s / 1e6 + (off.get(seed, 0.0) if it else 0.0) for it, s, _, _, _, _ in pts]
            ys = [hc if nh > 0 else None for _, _, nh, hc, _, _ in pts]
            _gap_path(out, fx, fy, xs, ys, COL[arm], dashed=(arm == "box"), width=1.5)
            for (it, s, nh, hc, cm, _), x in zip(pts, xs):
                tip = f"{SHORT[arm]} s{seed} at {x:.1f} M: {nh}/64 held" + (f", held cos {hc:.3f}" if nh else "")
                if nh == 0:
                    out.append(f'<line x1="{fx(x):.1f}" x2="{fx(x):.1f}" y1="{fy(-0.2) - 9:.1f}" y2="{fy(-0.2) - 1:.1f}" '
                               f'style="stroke:{COL[arm]};stroke-width:2"><title>{tip}</title></line>')
                else:
                    P._marker(out, fx(x), fy(hc), COL[arm], shape=SEED_SHAPE.get(seed, "circle"), hollow=nh < 60,
                              r=3.0, title=tip)
            if tag in X.stops:
                yl = next((v for v in reversed(ys) if v is not None), None)
                P._marker(out, fx(xs[-1]) + 8, fy(yl if yl is not None else -0.2) - (5 if yl is None else 0), COL[arm],
                          shape="cross", r=4.2, title=f"{SHORT[arm]} s{seed} stopped at {xs[-1]:.1f} M")
    hmax = max([X.hours(r["tag"], r["iteration"]) or 0 for r in X.ck] + [0.5])
    hx = [0, 1, 2, 3] if hmax <= 3.2 else [0, 1, 2, 3, 4]
    for j, xk in enumerate(("steps", "hours")):
        x0 = 62 + j * 492
        xr, xt, xlab = ((0, 41), (0, 10, 20, 30, 40), "Environment steps (millions)") if xk == "steps" else \
            ((0, max(hx)), hx, "Training wall-clock time (h)")
        fx, fy = P._panel(out, x0, 360, 410, 200, xr, (-0.2, 1.0), xt, (0.0, 0.25, 0.5, 0.75, 1.0), xlab,
                          "Held cosine, median of seeds and range", yfmt="{:.2f}")
        out.append(f'<line x1="{x0}" x2="{x0 + 410}" y1="{fy(0.9):.1f}" y2="{fy(0.9):.1f}" '
                   f'style="stroke:var(--ink3);stroke-dasharray:2 4"/><text x="{x0 + 6}" y="{fy(0.9) - 5:.1f}" '
                   f'style="fill:var(--ink3)">0.9</text>')
        for arm in ARMS:
            ser = _arm_median(X, arm, "held_cos", xk)
            xs = [s[0] for s in ser]
            _band(out, fx, fy, xs, [s[2] for s in ser], [s[3] for s in ser], COL[arm])
            _gap_path(out, fx, fy, xs, [s[1] for s in ser], COL[arm], dashed=(arm == "box"), width=2.2)
            for x, m, lo, hi, n in ser:
                if m is not None:
                    P._marker(out, fx(x), fy(m), COL[arm], r=2.6,
                              title=f"{SHORT[arm]} at {x:.2f} {'M steps' if xk == 'steps' else 'h'}: {m:.3g} "
                                    f"(median of the {n} seed{'s' if n > 1 else ''} evaluated, range {lo:.3g}-{hi:.3g})")
    out.append("</svg>")
    return "".join(out)


def svg_final(X: Data):
    """Final cosine of the held rollouts of every final policy, per arm and seed, on the held range of the axis; the
    rollouts that dropped the tool are counted under each seed."""
    if not X.fin:
        return P.pending("Final-policy evaluations not written yet (final_eval.jsonl).")
    W, H = 990, 350
    out = P._svg_open(W, H, fs=16, label="Final cosine of the tool axis with vertical in each held rollout of 64 per seed, per contact "
                            "model, and the number of rollouts that dropped the tool")
    x0, y0, w, h = 96, 34, 870, 220
    lo, hi = 0.5, 1.0
    fy = lambda v: y0 + h - (v - lo) / (hi - lo) * h  # noqa: E731
    for v in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        out.append(f'<line x1="{x0}" x2="{x0 + w}" y1="{fy(v):.1f}" y2="{fy(v):.1f}" style="stroke:var(--rule2)'
                   f'{";stroke-dasharray:2 4" if v == 0.9 else ""}"/><text x="{x0 - 8}" y="{fy(v) + 4:.1f}" '
                   f'text-anchor="end" style="fill:var(--ink3)">{v:.1f}</text>')
    out.append(f'<text x="{x0}" y="{y0 - 14}" style="fill:var(--ink2)">Final cosine of the tool axis with vertical in '
               f'the held rollouts (+1 tip down)</text>')
    out.append(f'<text x="{x0 - 8}" y="{y0 + h + 44}" text-anchor="end" style="fill:var(--ink3)">Dropped</text>')
    gw = w / len(ARMS)
    rng = np.random.default_rng(7)
    for i, arm in enumerate(ARMS):
        gx = x0 + i * gw
        out.append(f'<text x="{gx + gw / 2:.1f}" y="{y0 + h + 72}" text-anchor="middle" style="fill:var(--ink2)">'
                   f'{PANEL[arm]}</text>')
        fins = X.finals(arm)
        for j, seed in enumerate((0, 1, 2)):
            r = next((r for r in fins if r["seed"] == seed), None)
            cx = gx + gw * (j + 1) / 4
            out.append(f'<text x="{cx:.1f}" y="{y0 + h + 20}" text-anchor="middle" style="fill:var(--ink3)">'
                       f's{seed}</text>')
            if r is None:
                out.append(f'<text x="{cx:.1f}" y="{y0 + h + 44}" text-anchor="middle" style="fill:var(--ink3)">'
                           f'&#8211;</text>')
                continue
            cs, hs = r["final_cos"], r["held_final"]
            held = [c for c, hd in zip(cs, hs) if hd]
            jit = rng.uniform(-gw / 11, gw / 11, len(held))
            for c, dx in zip(held, jit):
                out.append(f'<circle cx="{cx + dx:.1f}" cy="{fy(min(max(c, lo), hi)):.1f}" r="2.5" style="fill:{COL[arm]};'
                           f'fill-opacity:.7;stroke:var(--card);stroke-width:.5"/>')
            out.append(f'<text x="{cx:.1f}" y="{y0 + h + 44}" text-anchor="middle" style="fill:'
                       f'{"var(--bad)" if len(held) < len(cs) else "var(--ink3)"}">{len(cs) - len(held)}</text>')
            if r.get("held_cos_mean") is not None:
                yv = fy(r["held_cos_mean"])
                out.append(f'<line x1="{cx - gw / 8:.1f}" x2="{cx + gw / 8:.1f}" y1="{yv:.1f}" y2="{yv:.1f}" '
                           f'style="stroke:var(--ink);stroke-width:2"><title>{SHORT[arm]} s{seed}: held '
                           f'{r["n_held"]}/64, held cos {r["held_cos_mean"]:.3f}, sd {r["held_cos_sd"]:.3f}</title></line>')
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
    pol = sorted({(r["arm"], r["tag"]) for r in X.fin}, key=lambda p: (ARMS.index(p[0]), p[1]))
    W = 990
    cw, ch, x0, y0 = 96, 54, 190, 84
    H = y0 + ch * len(pol) + 6 * (len({a for a, _ in pol}) - 1) + 20
    out = P._svg_open(W, H, fs=16, label="Held fraction of every final policy evaluated under each contact model in "
                            "MuJoCo-Warp and replayed open loop in CPU MuJoCo, Drake and Newton")
    for j, (kind, s) in enumerate(cols):
        cx = x0 + j * cw + (24 if kind == "rep" else 0)
        lab = CAP[s] if kind == "mjw" else ENG_LBL[s].split()[0] + (" MuJoCo" if s == "mujoco" else "")
        out.append(f'<text x="{cx + cw / 2:.1f}" y="{y0 - 12}" text-anchor="middle" style="fill:var(--ink2)">{lab}</text>')
    nr = sorted({r["n"] for r in X.tr})
    rl = f"{nr[0]}" if len(nr) == 1 else (f"{nr[0]}&#8211;{nr[-1]}" if nr else "&#8211;")
    out.append(f'<text x="{x0 + 2.5 * cw:.1f}" y="{y0 - 42}" text-anchor="middle" style="fill:var(--ink)">'
               f'MuJoCo-Warp, closed loop, {rl} rollouts</text>')
    nw = sorted({len([r for r in X.rep if r["dir"] == t and r["engine"] == e and r.get("hold_test_s") is None])
                 for _, t in pol for e in ENGINES} - {0})
    wl = f"{nw[0]}" if len(nw) == 1 else (f"{nw[0]}&#8211;{nw[-1]}" if nw else "&#8211;")
    out.append(f'<text x="{x0 + 5 * cw + 24 + 1.5 * cw:.1f}" y="{y0 - 42}" text-anchor="middle" style="fill:var(--ink)">'
               f'Open-loop replay, {wl} worlds</text>')
    cy, prev = y0 - ch, None
    for arm, tag in pol:
        cy += ch + (6 if prev and arm != prev else 0)
        prev = arm
        out.append(f'<text x="{x0 - 12}" y="{cy + ch / 2 + 5:.1f}" text-anchor="end" style="fill:var(--ink2)">'
                   f'{CAP[arm]} s{tag[-1]}</text>')
        out.append(f'<rect x="{x0 - 8}" y="{cy + 6}" width="4" height="{ch - 12}" style="fill:{COL[arm]}"/>')
        for j, (kind, s) in enumerate(cols):
            cx = x0 + j * cw + (24 if kind == "rep" else 0)
            if kind == "mjw":
                rs = [r for r in X.tr if r["tag"] == tag and r["scene"] == s]
                nh, nn = sum(r["n_held"] for r in rs), sum(r["n"] for r in rs)
                hc = [r["held_cos_mean"] for r in rs if r.get("held_cos_mean") is not None]
            else:
                rs = [r for r in X.replays(arm, s) if r["dir"] == tag]
                nh, nn = sum(1 for r in rs if r.get("held_end")), len(rs)
                hc = [r["cos_end"] for r in rs if r.get("held_end")]
            v = nh / nn if nn else None
            fill, ink = _hm_class(v)
            own = kind == "mjw" and s == arm
            out.append(f'<rect x="{cx + 2}" y="{cy + 2}" width="{cw - 4}" height="{ch - 4}" rx="4" style="fill:{fill};'
                       f'stroke:{"var(--ink)" if own else "none"};stroke-width:2"><title>{SHORT[arm]} s{tag[-1]} on '
                       f'{SHORT.get(s, s)}: held {nh}/{nn}{", held cos %.3f" % np.mean(hc) if hc else ""}</title></rect>')
            if nn:
                lab = f"{100 * v:.0f}%" if kind == "mjw" else f"{nh}/{nn}"
                out.append(f'<text x="{cx + cw / 2:.1f}" y="{cy + ch / 2 - 3:.1f}" text-anchor="middle" style="fill:{ink};'
                           f'font-size:16px;font-weight:600">{lab}</text>')
                out.append(f'<text x="{cx + cw / 2:.1f}" y="{cy + ch / 2 + 15:.1f}" text-anchor="middle" style="fill:{ink};'
                           f'font-size:15.5px">{("cos %.2f" % np.mean(hc)) if hc else "&#8211;"}</text>')
    out.append("</svg>")
    return "".join(out)


WATCH_SIG = [("ang_jerk_hold_median", "Tool shaking in the hold (rad/s&#178;)", (1, 1000), (1, 10, 100, 1000), True, 1.0),
             ("cos_step_hold_median", "Tool rocking, cosine change per step", (0.0001, 0.1), (0.0001, 0.001, 0.01, 0.1), True, 1.0),
             ("dact_held", "|&#916;a| per policy step (action units)", (0, 1.0), (0, 0.25, 0.5, 0.75, 1.0), False, 1.0),
             ("resid_over_frac", "Actions at or past the budget (%)", (0, 60), (0, 20, 40, 60), False, 100.0),
             ("ctrl_sat_frac", "Servo targets at a range limit (%)", (0, 30), (0, 10, 20, 30), False, 100.0),
             ("share_min", "Smallest finger share of the grip (%)", (0, 34), (0, 10, 20, 30), False, 100.0)]


def watch_value(r, key):
    """A watch row's signal; the rocking of rows written before it existed comes from the row's saved traces."""
    if r.get(key) is not None or key != "cos_step_hold_median":
        return r.get(key)
    p = os.path.join(ROOT, "logs/20261008-contact_model_policies/watch_traces", f"{r['tag']}_it{r['iteration']:04d}.npz")
    if not os.path.exists(p):
        return None
    tr = np.load(p)
    cos, z, fo = tr["cos"].astype(float), tr["z"].astype(float), tr["force"].astype(float)
    held = (z > 0.06) & ((fo >= 0.24).sum(-1) >= 2)
    hh = held[150:]
    d = np.abs(np.diff(cos[150:], axis=0))[hh[1:] & hh[:-1]]
    return float(np.median(d)) if d.size else None


def _watch_lines(X: Data, arm, key, scale):
    out = {}
    for r in sorted((r for r in X.watch if r["arm"] == arm), key=lambda r: r["iteration"]):
        v = watch_value(r, key)
        out.setdefault(r["tag"], []).append(((r["iteration"] + 1) * STEPS_PER_IT / 1e6, None if v is None else v * scale))
    return out


def svg_watch(X: Data):
    if not X.watch:
        return P.pending("Checkpoint watch rows not written yet (watch.jsonl).")
    W, H = 990, 890
    out = P._svg_open(W, H, fs=16, label="Checkpoint watch signals per contact model against environment steps: the tool's shaking and "
                            "rocking in the hold, the action change per policy step, the actions at or past the residual "
                            "budget, the servo targets at a range limit and the smallest finger's share of the grip force")
    for k, (key, lab, ys, yt, logy, scale) in enumerate(WATCH_SIG):
        x0, y0 = (70, 570)[k % 2], (40, 330, 620)[k // 2]
        if logy:
            fx, fy = P._panel(out, x0, y0, 360, 200, (0, 41), ys, (0, 10, 20, 30, 40), yt,
                              "Environment steps (millions)", lab, logy=True, yfmt="{:g}")
            lim = 40 if key == "ang_jerk_hold_median" else 0.005
            out.append(f'<line x1="{x0}" x2="{x0 + 360}" y1="{fy(lim):.1f}" y2="{fy(lim):.1f}" style="stroke:var(--ink3);'
                       f'stroke-dasharray:2 4"/><text x="{x0 + 362}" y="{fy(lim) + 4:.1f}" style="fill:var(--ink3)">{lim:g}</text>')
        else:
            fx, fy = P._panel(out, x0, y0, 360, 200, (0, 41), ys, (0, 10, 20, 30, 40), yt,
                              "Environment steps (millions)", lab, yfmt="{:g}")
        clamp = lambda v: min(max(v, ys[0]), ys[1])  # noqa: E731
        for arm in ARMS:
            _draw_seeds(out, X, fx, fy, _watch_lines(X, arm, key, scale), arm, clamp=clamp)
    out.append("</svg>")
    return "".join(out)


def stop_outcome(X: Data, tag):
    """(iteration, reason) of the first watched checkpoint at which the stop rule (rl_contact_eval.stop_rule, with the
    rocking of rows written before it existed taken from their traces) fires, or None."""
    import rl_contact_eval as E
    rs = sorted((dict(r, cos_step_hold_median=watch_value(r, "cos_step_hold_median")) for r in X.watch if r["tag"] == tag),
                key=lambda r: r["iteration"])
    for k in range(len(rs)):
        stop, why = E.stop_rule(rs[:k + 1])
        if stop:
            return rs[k]["iteration"], why
    return None


STOP_KIND = {"drops every rollout": "every rollout dropped", "jitter": "the tool shaken",
             "rocking": "the tool rocking every policy step", "idle finger": "an idle finger",
             "saturated": "servo targets pinned", "palm": "the tool on the palm"}


def degenerate_stops(X: Data, arm):
    """[(seed, env steps in millions, kinds, live)] of an arm's runs that meet the stop rule for a degenerate reason:
    those the watch stopped, and the finished ones at which the recomputed rule fires before 40 M. `kinds` are the
    degenerate flags raised over the rule's 10 M window."""
    import rl_contact_eval as E
    out = []
    for tag in sorted({r["tag"] for r in X.watch if r["arm"] == arm}):
        so = stop_outcome(X, tag)
        st = X.degen.get(tag)
        if not st and not (so and so[0] < 812 and so[1].startswith("degenerate")):
            continue
        it = st["iteration"] if st else so[0]
        m = (it + 1) * STEPS_PER_IT / 1e6
        rs = sorted((dict(r, cos_step_hold_median=watch_value(r, "cos_step_hold_median")) for r in X.watch
                     if r["tag"] == tag and r["iteration"] <= it), key=lambda r: r["iteration"])
        back = [r for r in rs if (r["iteration"] + 1) * STEPS_PER_IT <= (it + 1) * STEPS_PER_IT - 10_000_000]
        rs = [r for r in rs if not back or r["iteration"] >= back[-1]["iteration"]]     # the rule's window
        kinds = []
        for r in rs:
            for f in E.degenerate_flags(r):
                k = next((v for key, v in STOP_KIND.items() if f.startswith(key)), None)
                if f.startswith("jitter") and (r.get("tilt_reversal_frac") or 0) > 0.5:
                    k = STOP_KIND["rocking"]          # the jerk test fired, and the tilt reverses at most steps
                if k and k not in kinds:
                    kinds.append(k)
        out.append((int(tag[-1]), m, kinds, bool(st)))
    return out


def stops_table(X: Data):
    if not X.watch:
        return ""
    head = ["Run", "Last watch", "Rule fired", "Reason", "Outcome"]
    body = []
    tags = sorted({r["tag"] for r in X.watch}, key=lambda t: (t[-1], ARMS.index(tag_arm(t)) if tag_arm(t) else 9))
    kinds = {}
    for arm in ARMS:
        for d in degenerate_stops(X, arm):
            kinds[(arm, d[0])] = d[2]
    mst = lambda it: f"{(int(it) + 1) * STEPS_PER_IT / 1e6:.1f}"  # noqa: E731

    def reason(tag, so):
        m = re.match(r"degenerate at every checkpoint from it (\d+) to (\d+)", so[1])
        if m:
            k = kinds.get((tag_arm(tag), int(tag[-1])), [])
            return f"Degenerate from {mst(m.group(1))} to {mst(m.group(2))}&#8202;M: {', '.join(k)}"
        m = re.match(r"held cos ([\d.]+) at it (\d+) -> ([\d.]+) at it (\d+)", so[1])
        if m:
            return (f"Held cosine {m.group(1)} at {mst(m.group(2))}&#8202;M, {m.group(3)} at {mst(m.group(4))}&#8202;M: "
                    "a gain under 0.02")
        return so[1]
    for tag in tags:
        rs = sorted((r for r in X.watch if r["tag"] == tag), key=lambda r: r["iteration"])
        last = rs[-1]
        so = stop_outcome(X, tag)
        st = X.stops.get(tag)
        if st:
            outcome = f"Stopped at {mst(st['iteration'])}&#8202;M" + \
                ("; its last checkpoint is the final policy" if tag not in X.degen else "")
        elif last["iteration"] >= 812:
            outcome = "Trained to 40&#8202;M" + ("; watched after the run" if tag not in X.live else "")
        else:
            outcome = "Training"
        body.append([f"<span style='white-space:nowrap'>{swatch(tag_arm(tag))}{CAP[tag_arm(tag)]} s{tag[-1]}</span>",
                     f"<span style='white-space:nowrap'>{last['n_held']}/64</span>" +
                     (f"<br>cos {last['held_cos_mean']:.2f}" if last.get("held_cos_mean") is not None else ""),
                     f"{mst(so[0])}&#8202;M" if so else "&#8211;",
                     reason(tag, so) if so else "&#8211;", outcome])
    cap = ("The owner&#8217;s stopping rule at every watched checkpoint (4.0&#8202;M steps apart): stop when every "
           "checkpoint of the last 10&#8202;M steps is degenerate, or when the held cosine gained under 0.02 (and the held "
           "count under 8 of 64) over 10&#8202;M steps. Last watch: rollouts held and held cosine at the last watched "
           "checkpoint. The seed-0 runs finished before the watch existed, and the skin trainer leaves too little GPU "
           "memory for a second process: those runs were watched after they ended.")
    return table(head, body, cap, text_cols=(0, 3, 4))


TB_SIG = [("Train/mean_reward", "Mean episode return", (0, 500), (0, 100, 200, 300, 400, 500)),
          ("Train/mean_episode_length", "Mean episode length (steps, of 250)", (0, 260), (0, 50, 100, 150, 200, 250)),
          ("Episode_Termination/tip_lost", "Tip-lost terminations per iteration", (0, 50), (0, 10, 20, 30, 40, 50)),
          ("Episode_Reward/target_axis_alignment", "Alignment reward per episode", (0, 70), (0, 10, 20, 30, 40, 50, 60, 70)),
          ("Policy/mean_std", "Action noise of the policy (std)", (0, 0.45), (0, 0.1, 0.2, 0.3, 0.4)),
          ("Episode_Reward/grip_force_excess", "Grip-force term per episode", (0, 8), (0, 2, 4, 6, 8))]


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
    W, H = 990, 890
    out = P._svg_open(W, H, fs=16, label="Training curves per contact model against environment steps: episode return, episode "
                            "length, tip-lost terminations, the alignment reward, the policy's action noise and the "
                            "grip-force term")
    for k, (key, lab, ys, yt) in enumerate(TB_SIG):
        x0, y0 = (70, 570)[k % 2], (40, 330, 620)[k // 2]
        fx, fy = P._panel(out, x0, y0, 360, 200, (0, 41), ys, (0, 10, 20, 30, 40), yt, "Environment steps (millions)",
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
    """The seed-0 runs' checkpoint sheets, collapsed under one summary line."""
    f0 = FIG[0] + 1
    body = _watch_sheets(X)
    if not body:
        return ""
    return (f'<details class="sheets"><summary>Checkpoint sheets of the four seed-0 runs: the median rollout at 4, 8, 16, 28 '
            f'and 40&#8202;M steps, with what each checkpoint showed (Figures&#160;{f0}&#8211;{FIG[0]})</summary>{body}'
            '</details>')


def _watch_sheets(X: Data):
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


PERT = [("friction", "Friction (&#215;)", (0.7, 1.0, 1.3)), ("mass", "Tool mass (&#215;)", (0.8, 1.0, 1.2)),
        ("kp", "Servo gain (N m/rad)", (2.0, 4.0, 6.0, 10.0)), ("dt", "Physics step (ms)", (1.0, 2.0)),
        ("noise", "Pose noise (mm, &#176;)", (0.0, 1.0))]


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
    out = P._svg_open(W, H, fs=16, label="Held fraction of each arm's final policies under one perturbation at a time")
    pw, gap, x0 = 140, 46, 66
    for k, (kind, lab, vals) in enumerate(PERT):
        px = x0 + k * (pw + gap)
        pos = {v: i for i, v in enumerate(vals)}
        fx0, fy = P._panel(out, px, 40, pw, 170, (-0.4, len(vals) - 0.6), (0, 1), [], (0, 0.5, 1),
                           lab, "Held fraction" if k == 0 else "", yfmt="{:g}")
        for v in vals:
            tx = fx0(pos[v])
            vt = ("1" if kind == "dt" and v == 1.0 else "2") if kind == "dt" else \
                (("0" if v == 0 else "2, 5") if kind == "noise" else f"{v:g}")
            out.append(f'<text x="{tx:.1f}" y="{40 + 170 + 19}" text-anchor="middle" style="fill:var(--ink3)">{vt}</text>')
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


def best_seed(X: Data, arm):
    """The final-evaluation row of an arm's best finished seed (held fraction, then held cosine), or None."""
    rs = sorted(X.finals(arm), key=lambda r: (r["held_frac"], r["held_cos_mean"] if r["held_cos_mean"] is not None else -2))
    return rs[-1] if rs else None


def seed_film(X: Data, r):
    """(film, strip) paths relative to D for a final policy: post_train.sh's close-up render when it exists, else the
    checkpoint watch's film of its final checkpoint (the rollout with the median final cosine of 64)."""
    for role in ("median", "best"):
        f = next((f for f in X.films if f["role"] == role and f["tag"] == r["tag"]), None)
        if f and os.path.exists(os.path.join(D, f"media/{f['tag']}_{role}.mp4")):
            return f"media/{f['tag']}_{role}.mp4", f"media/{f['tag']}_{role}_strip.jpg"
    rel = f"watch/{r['tag']}/it{r['iteration']:04d}_median.mp4"
    if os.path.exists(os.path.join(D, rel)):
        return rel, f"watch/{r['tag']}/it{r['iteration']:04d}_strip.jpg"
    return None, None


def _film_cell(X: Data, arm, r, role):
    film, _ = seed_film(X, r)
    if film is None:
        return ""
    hc = f", held cos {r['held_cos_mean']:.2f}" if r.get("held_cos_mean") is not None else ""
    return (f'<div class="cell">{video(film)}<p class="filmlab">{swatch(arm)}<b>{LBL[arm]}</b>, seed {r["seed"]} '
            f'({role}): held {r["n_held"]}/64{hc}. {FILM_NOTES.get(r["tag"], "")} <code>{film}</code></p></div>')


def films(X: Data):
    cells = []
    for arm in ARMS:
        r = median_seed(X, arm)
        if r is not None:
            only = len(X.finals(arm)) == 1
            cells.append(_film_cell(X, arm, r, "the only finished seed" if only else "median seed"))
    cells = [c for c in cells if c]
    if not cells:
        return P.pending("Films not rendered yet (post_train.sh: close-up renders of the best and median seed per arm "
                         "and three-engine replay films).")
    src = [seed_film(X, r)[0] for r in (median_seed(X, a) for a in ARMS) if r is not None]
    how = ("one deterministic rollout rendered close up by <code>scripts/rl_render_reorient.py</code> at "
           "960&#215;720" if all(f and f.startswith("media/") for f in src) else
           "the rollout with the median final cosine of 64")
    FIG[0] += 1
    out = [f'<figure class="films">{"".join(cells)}<figcaption>Figure&#160;{FIG[0]}. The final policy of each contact '
           f'model&#8217;s median seed (seeds ranked by held rollouts, then held cosine), {how}: the scripted grasp and '
           f'lift (0&#8211;1.16&#8202;s), then the policy&#8217;s turn and hold to 5&#8202;s; camera 0.24&#8202;m from '
           f'the hand, the turn in the image plane (thumb in front, index left, middle right). Force ranges are the '
           f'10th&#8211;90th percentiles of the summed fingertip force over the held steps of the hold (steps '
           f'150&#8211;250) of the 64 rollouts of the final evaluation. Paths relative to '
           f'<code>docs/experiments/20261008-contact_model_policies/</code>.</figcaption></figure>']
    best = []
    for arm in ARMS:
        b, m = best_seed(X, arm), median_seed(X, arm)
        if b is not None and m is not None and b["tag"] != m["tag"]:
            best.append(_film_cell(X, arm, b, "best seed"))
    best = [c for c in best if c]
    if best:
        FIG[0] += 1
        out.append(f'<figure class="films three">{"".join(best)}<figcaption>Figure&#160;{FIG[0]}. The best seed of the '
                   f'contact models with more than one finished seed, rendered as in Figure&#160;{FIG[0] - 1}.'
                   f'</figcaption></figure>')
    rep = []
    for arm in ARMS:
        r = median_seed(X, arm)
        rel = f"media/{r['tag']}_replay_three.mp4" if r else None
        if rel and os.path.exists(os.path.join(D, rel)):
            rep.append(f'<div class="cell">{video(rel)}<p class="filmlab">{swatch(arm)}<b>{LBL[arm]}</b>, seed '
                       f'{r["seed"]}, world 0. {FILM_NOTES.get(r["tag"] + ":replay", "")} <code>{rel}</code></p></div>')
    if rep:
        FIG[0] += 1
        out.append(f'<figure class="replays">{"".join(rep)}<figcaption>Figure&#160;{FIG[0]}. The median seed of each '
                   f'contact model, world 0: the MuJoCo-Warp rollout it trained on (left), and its finger targets from the '
                   f'onset of the turn replayed open loop in Drake (centre) and Newton (right) with the TPU block as a '
                   f'hydroelastic tip and the palm welded at its lifted pose. Paths relative to '
                   f'<code>docs/experiments/20261008-contact_model_policies/</code>.</figcaption></figure>')
    return "".join(out)


FILM_NOTES = {       # run tag -> what its final film shows, written after watching it (numbers from final_traces);
    # "<tag>:replay" -> what its three-engine replay film shows (overlay cosines read off the frames)
    "20261008-d6_work_box_40M_s0:replay": "In MuJoCo-Warp the box tip stands the tool up (cos 1.00 at 0.42&#8202;s) and "
    "holds it at 0.94&#8211;0.96. Replayed open loop, the same targets throw the tool sideways: it is on the floor "
    "0.56&#8202;s after the onset in Drake and 0.22&#8202;s after it in Newton.",
    "20261008-d6_work_tpu27mesh_40M_s1:replay": "MuJoCo-Warp turns the tool to cos 0.81 by 0.42&#8202;s and holds it at "
    "0.84. Replayed in Drake, the targets turn it to 0.80 and hold it at 0.74&#8211;0.78 until it drops at "
    "3.4&#8202;s; in Newton the tool slips from 0.64 and is on the floor within a second.",
    "20261008-d6_work_tpu27pads1_40M_s2:replay": "MuJoCo-Warp, Drake and Newton hold the tool at the same angle: cos "
    "0.61, 0.61 and 0.62 at 0.42&#8202;s, and 0.61, 0.58 and 0.60 at 3.8&#8202;s.",
    "20261008-d6_work_tpu27skin_40M_s1:replay": "All three simulators hold the tool: MuJoCo-Warp turns it to cos 0.59 by "
    "0.42&#8202;s and 0.63 by 3.8&#8202;s, Drake and Newton to 0.53 by 0.42&#8202;s and keep it at "
    "0.55&#8211;0.56.",
    "20261008-d6_work_box_40M_s0": "The thumb and the index finger stand the tool up within a quarter second of the onset, "
    "the middle finger mostly off it, and the held tool tilts by a few degrees from one frame to the next; in the hold "
    "the summed tip force ranges over 41&#8211;148&#8202;N.",
    "20261008-d6_work_tpu27mesh_40M_s0": "The three fingertips turn the tool to cos 0.75 within a quarter second, and it "
    "creeps on to 0.86 by 5&#8202;s; the summed tip force ranges over 38&#8211;62&#8202;N.",
    "20261008-d6_work_tpu27mesh_40M_s1": "The three fingertips turn the tool to cos 0.73 within a quarter second and to "
    "0.80 by 0.7&#8202;s, and hold it there; the summed tip force ranges over 44&#8211;67&#8202;N.",
    "20261008-d6_work_tpu27pads1_40M_s0": "The three pads turn the tool to cos 0.63 within a quarter second, and it "
    "creeps on to 0.70 by 5&#8202;s; the summed tip force stays at 42&#8211;44&#8202;N.",
    "20261008-d6_work_tpu27pads1_40M_s2": "The three pads turn the tool to cos 0.56 within a tenth of a second, sliding "
    "it 19&#8202;mm up in the grip, and hold it still at 0.61 from 0.3&#8202;s with a summed tip force of "
    "52&#8211;57&#8202;N. The watch stopped the run at 36&#8202;M steps on the plateau test.",
    "20261008-d6_work_tpu27skin_40M_s0": "The three pads turn the tool to cos 0.67 within a quarter second and to 0.79 by "
    "0.7&#8202;s, then hold it still at 46&#8211;52&#8202;N.",
    "20261008-d6_work_tpu27skin_40M_s1": "The three pads turn the tool to cos 0.55 within 0.14&#8202;s, and it creeps on "
    "to 0.63 by 5&#8202;s; the summed tip force stays at 46&#8211;49&#8202;N.",
}


# ------------------------------------------------------------------------------------------ sections

GLOSSARY = [
    ("arm", "One fingertip contact model. The arms share the hand (real_v1 D6), the working plant (servo gain \\(k_p\\) = "
     "4&#8202;N&#8202;m/rad, damping 0.08&#8202;N&#8202;m&#8202;s/rad), the 24.5&#8202;g screwdriver (\\(\\mu\\) = 1), "
     "MuJoCo-Warp at 2&#8202;ms with impratio 10, and the recipe."),
    ("env step, iteration", "One 20&#8202;ms policy step of one of 2,048 parallel envs; a PPO iteration is 24 steps per "
     "env, 49,152 env steps, and 40&#8202;M env steps are 813 iterations."),
    ("held (load test)", "At least two fingertips press on the tool with 0.24&#8202;N (its weight) or more and the tool is "
     "above 60&#8202;mm; a rollout is held when this is true at 5&#8202;s."),
    ("cos", "Cosine of the tool axis with world vertical, \\(R_{zz}\\): +1 is tip down, the goal; the tool starts "
     "horizontal, at 0."),
    ("held cosine", "Mean final cos over the held rollouts of one evaluation."),
    ("evaluation, closed loop", "64 rollouts of the deterministic policy in parallel MuJoCo-Warp envs over 250 policy steps, "
     "with the training&#8217;s timing (policy from step 58, after the scripted grasp and lift), no early termination "
     "and no randomisation; 256 per transfer cell, 128 per perturbation."),
    ("reach 0.9, step at 0.9", "Rollouts whose tool passes cos 0.9 while held, and the median policy step at which it "
     "first does."),
    ("seed spread", "Range of the held cosines of an arm&#8217;s seeds with a final policy."),
    ("rollout spread", "Standard deviation of the final cos over one evaluation&#8217;s held rollouts."),
    ("grip", "Summed net force of the three fingertips on the tool, mean over the held steps from step 58, in N."),
    ("grip change per step", "Median absolute change of the grip between policy steps in the hold (steps "
     "150&#8211;250), in N."),
    ("peak force", "Largest single-fingertip force over a rollout&#8217;s held steps, in N."),
    ("penetration", "Deepest fingertip&#8211;tool penetration over a rollout&#8217;s held steps, in mm."),
    ("creep", "Tool displacement against the palm over the last second, in mm/s, and rotation of its axis, in "
     "&#176;/s."),
    ("|&#916;a|", "Mean absolute change of the 9-dimensional action between policy steps; one action unit is "
     "0.5&#8202;rad of servo target."),
    ("shaking", "Mean absolute change of the tool&#8217;s angular speed per policy step in the hold, divided by 20&#8202;ms, "
     "in rad/s&#178;; <code>trajectory_health</code> fails a policy above 40."),
    ("rocking", "Median absolute change of cos between policy steps in the hold; the tool rocks at the policy rate above "
     "0.005 when its tilt also reverses at most steps."),
    ("servo targets at a limit", "Share of the nine finger servo targets (anchor plus residual) at their actuator&#8217;s "
     "range limit; the runs do not clip the residual."),
    ("finger share", "One fingertip&#8217;s share of the grip over the held steps."),
    ("checkpoint watch", "Every 82 iterations (4.0&#8202;M env steps) the run pauses for an evaluation and films of its "
     "median and worst rollouts, looked at before it goes on. The owner&#8217;s rule stops a run that is degenerate "
     "(every rollout dropped, an idle finger, shaking above 40 with rocking above 0.005, rocking at the policy rate, "
     "pinned servo targets or the tool on the palm) at every checkpoint for 10&#8202;M steps, or whose held cosine gains "
     "under 0.02 over 10&#8202;M steps."),
    ("condim", "MuJoCo&#8217;s contact dimension: 3 gives a point contact with sliding friction, 4 adds a torsional "
     "friction bounded by \\(\\mu_\\text{spin} N\\)."),
    ("open-loop replay", "The recorded finger targets of a MuJoCo-Warp rollout from step 58, played at 50&#8202;Hz into "
     "another simulator with the palm welded at its lifted pose (<code>scripts/rl_policy_replay.py</code>)."),
    ("transfer gap", "Held fraction or held cosine under another contact model or simulator minus that in the training "
     "model in MuJoCo-Warp."),
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
    head = ["Contact model", "Iteration (s)<br>idle GPU", "Iteration (s)<br>runs", "GPU-h per<br>40&#8202;M steps",
            "GPU-h<br>spent", "GPU memory<br>(GB)", "Failed<br>starts"]
    body = []
    for a in SCENES:
        cs = [c for t, c in X.cost.items() if tag_arm(t) == a]
        spi = [c["s_per_it_median"] for c in cs if c.get("s_per_it_median")]
        h40 = [run_hours(X, c["tag"]) for c in cs if c.get("finished")]
        spent = [run_hours(X, c["tag"]) for c in cs]
        # the training process's own GPU memory (sampled per process from 2026-10-08 15:19); the whole GPU's peak only
        # for an arm none of whose runs was sampled
        gpu = [c["gpu_mem_proc_mb"] for c in cs if c.get("gpu_mem_proc_mb")] or \
            [c["gpu_mem_peak_mb"] for c in cs if c.get("gpu_mem_peak_mb")]
        fl = sum(c.get("failed_attempts", 0) for c in cs if not c.get("stopped_at"))
        trained = a in ARMS
        body.append([f"{swatch(a) if trained else ''}{LBL[a]}", f1(probe_spit(a), 2),
                     f1(float(np.median(spi)), 2) if spi else "&#8211;",
                     f1(float(np.median([h for h in h40 if h])), 2) if any(h40) else ("Not trained" if not trained else "&#8211;"),
                     f1(sum(h for h in spent if h), 2) if any(spent) else "&#8211;",
                     f1(max(gpu) / 1e3, 1) if gpu else "&#8211;", str(fl) if trained else "&#8211;"])
    tot = sum(h for c in X.cost.values() if (h := run_hours(X, c["tag"])))
    txt = ("<p>All arms train the same hand, plant, tool, solver, recipe and flags and differ only in the fingertip. The "
           "box tip is the fingertip every policy of this program trained on until 2026-10-05. The TPU arms replace it by "
           "the printed tip, a 17&#215;14.8&#215;22&#8202;mm block with 2.7&#8202;mm fillets, as one convex mesh with "
           "point contact (condim 3, like the box tip), or as the 1&#8202;mm sphere pads of the overview page, 1,060 "
           f"spheres per tip (<code>{OVERVIEW}</code>). The skin mounts each tip&#8217;s pads on a child body with two tangent slides and "
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
    cap = ("Contact models and training cost on one RTX 4070 Ti SUPER. Iteration: wall time of one PPO iteration, on an idle "
           "GPU the median of a five-iteration probe before the queue, in the runs the median over each arm&#8217;s runs, "
           "which shared the workstation with other jobs (an Isaac Lab evaluation on about ten CPU cores for most of the "
           "night). GPU-hours exclude the watch&#8217;s pauses; &#8216;spent&#8217; includes the runs stopped early. GPU "
           "memory: the training process.")
    return txt + table(head, body, cap, text_cols=(0,))


def curves_section(X: Data):
    svg = svg_curves(X)
    cap = ("Deterministic evaluation of every second saved checkpoint (82 iterations, 4.0&#8202;M env steps) and the final "
           "one. Top: the held cosine of each seed against env steps, one panel per contact model (marker: circle s0, "
           "square s1, diamond s2; filled when at least 60 of the 64 rollouts held, hollow when fewer, a tick on the "
           "floor when none held; seeds set 0.7&#8202;M apart), with a bar from the 10th to the 90th percentile of the "
           "held rollouts&#8217; final cosine; a cross ends a run stopped by the checkpoint watch. Bottom: the median "
           "over seeds "
           "(line) and the seeds&#8217; range (band) of the held cosine against env steps and against training "
           "wall-clock time with the watch&#8217;s pauses removed. At iteration 0 the residual is near zero and the "
           "scripted grasp holds the tool about level (cos &#8722;0.10 to &#8722;0.02) in every rollout.")
    out = figure(svg, cap, legend()) + curves_text(X)
    wcap = ("Signals of the checkpoint watch at the same checkpoints, over the hold after the turn (policy steps "
            "150&#8211;250): the tool&#8217;s shaking (dashed: trajectory_health&#8217;s jitter limit) and rocking, the "
            "median change of its cosine between policy steps (dashed: 0.005); the mean action change per policy step; the "
            "share of action components at or past one unit (0.5&#8202;rad, the residual&#8217;s nominal budget; the runs "
            "do not clip actions); the share of finger servo targets at their actuator&#8217;s range limit; and the "
            "smallest of the three fingers&#8217; shares of the grip force. Medians over held rollouts; one line per seed "
            "as above.")
    tcap = ("Training curves from the event files, smoothed over nine iterations: the stochastic policy&#8217;s mean "
            "episode return and length, tip-lost terminations per iteration of 2,048 envs (a fingertip off the tool for 15 steps "
            "ends the episode), "
            "the alignment term of the return, the standard deviation of the policy&#8217;s Gaussian action noise "
            "(0.30 at the start) and the grip-force term (weight +0.25 on the tip force above 4&#8202;N per pad). One "
            "line per seed; a cross ends a stopped run. Against wall-clock time: Figure " + str(FIG[0]) + " and "
            "Table 1.")
    out += figure(svg_tb(X), tcap, legend()) + tb_text(X)
    out += figure(svg_watch(X), wcap, legend()) + stops_table(X) + watch_sheets(X)
    return out


def tb_text(X: Data):
    """The action noise each arm's finished runs end with, from the event files."""
    end = {}
    for r in X.tb:
        if r.get("Policy/mean_std") is not None and (r["tag"] not in end or r["iteration"] > end[r["tag"]]["iteration"]):
            end[r["tag"]] = r
    fin = {r["tag"] for r in X.fin}
    v = {a: [end[t]["Policy/mean_std"] for t in fin if tag_arm(t) == a and t in end] for a in ARMS}
    if not all(v.values()):
        return ""
    return (f"<p>The policies trained on the pads and the skin end training with an action noise of "
            f"{_rng(v['tpu27pads1'] + v['tpu27skin'])}, the TPU-mesh policies with {_rng(v['tpu27mesh'])} and the "
            f"box-tip policy with {_rng(v['box'])}; every run starts at 0.30.</p>")


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
    W, H = 990, 520
    out = P._svg_open(W, H, fs=16, label="Tool cosine against time in each of the 64 rollouts of every contact model's median-seed final "
                            "policy, and the summed fingertip force")
    pw, gap, x0 = 190, 40, 66
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
                          "Time (s)", f"{CAP[arm]} s{r['seed']}, cos" if k == 0 else f"{CAP[arm]} s{r['seed']}",
                          xfmt="{:g}", yfmt="{:g}" if k == 0 else (lambda v: ""))
        out.append(f'<line x1="{fx(1.16):.1f}" x2="{fx(1.16):.1f}" y1="40" y2="240" style="stroke:var(--ink3);'
                   f'stroke-dasharray:2 4"/>')
        for e in np.argsort(held):                       # dropped first, held on top
            pts = list(zip(t[::3], cos[::3, e]))
            P._path(out, fx, fy, pts, COL[arm] if held[e] else "var(--bad)", width=0.9)
        fx2, fy2 = P._panel(out, px, 336, pw, 110, (0, 5), (0, 150), (0, 1, 2, 3, 4, 5), (0, 50, 100, 150), "Time (s)",
                            "Grip (N)" if k == 0 else "", yfmt="{:g}" if k == 0 else (lambda v: ""))
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


def grip_range(tag):
    """10th and 90th percentile of the summed fingertip force over the held steps of the hold (steps 150-249), N."""
    p = os.path.join(D, "final_traces", f"{tag}.npz")
    if not os.path.exists(p):
        return None
    tr = np.load(p)
    fo, z = tr["force"].astype(float), tr["z"].astype(float)
    held = (z > 0.06) & ((fo >= 0.24).sum(-1) >= 2)
    g = fo.sum(-1)[150:][held[150:]]
    return (float(np.percentile(g, 10)), float(np.percentile(g, 90))) if g.size else None


def turn_time(tag, onset=58):
    """Seconds after the onset at which the median cos of the rollouts held at 5 s first reaches 90 % of its final value."""
    p = os.path.join(D, "final_traces", f"{tag}.npz")
    if not os.path.exists(p):
        return None
    tr = np.load(p)
    cos, z, fo = tr["cos"].astype(float), tr["z"].astype(float), tr["force"].astype(float)
    held = ((z > 0.06) & ((fo >= 0.24).sum(-1) >= 2))[-1]
    if not held.any():
        return None
    m = np.median(cos[:, held], axis=1)
    k = int(np.argmax(m[onset:] >= 0.9 * m[-1]))
    return (k + 1) * 0.02


def final_text(X: Data):
    tt = [t for t in (turn_time(r["tag"]) for a in ARMS for r in X.finals(a)) if t is not None]
    parts = []
    for arm in ARMS:
        rr = [grip_range(r["tag"]) for r in X.finals(arm)]
        rr = [x for x in rr if x]
        if rr:
            parts.append(f"{SHORT[arm]} " + ", ".join(f"{lo:.0f}&#8211;{hi:.0f}" for lo, hi in rr))
    if not parts:
        return ""
    return (f"<p>In the rollouts that hold the tool at 5&#8202;s, every final policy brings it to 90&#8202;% of its final "
            f"cosine within {max(tt):.2f}&#8202;s of the onset (Figure " + str(FIG[0]) + "). The grip in the hold is "
            "steady on the pads and the skin and fluctuates with point contact: the summed fingertip force spans "
            "(10th&#8211;90th percentile over the held steps, per seed) " + "; ".join(parts) + "&#8202;N.</p>")


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
    stopped = ", ".join(f"{SHORT[tag_arm(t)]} s{t[-1]} at {(r['iteration'] + 1) * STEPS_PER_IT / 1e6:.0f}&#8202;M"
                        for t, r in sorted(X.degen.items(), key=lambda kv: (ARMS.index(tag_arm(kv[0])), kv[0])))
    cap = ("Final cosine of the held rollouts of the final policies (64 rollouts per seed; the axis starts at 0.5, below "
           "every held rollout) and, under each seed, the number of rollouts that dropped the tool; bar: the seed&#8217;s "
           "held cosine. A dash marks a run the watch stopped as degenerate, which has no final policy: " + stopped +
           " steps.")
    out = figure(svg, cap)
    out += figure(svg_traces(X), "The 64 rollouts of each contact model&#8217;s median-seed final policy: the tool&#8217;s "
                  "cosine with vertical (top) and the summed fingertip force on it (bottom, clipped at 150&#8202;N) against "
                  "time; rollouts that held the tool at 5&#8202;s in the arm&#8217;s colour, the others in red. Dashed: the residual "
                  "policy&#8217;s onset after the scripted grasp and lift (step 58).")
    out += final_text(X)
    h1 = ["Contact model", "Held", "Held cos,<br>median of seeds", "Seed<br>spread", "Rollout<br>spread",
          "Reach 0.9<br>held", "Step<br>at 0.9"]
    h2 = ["Contact model", "Grip<br>(N)", "Grip change<br>per step (N)", "Peak<br>force (N)", "Penetration<br>(mm)",
          "Creep<br>(mm/s)", "Creep<br>(&#176;/s)", "Shaking<br>(rad/s&#178;)", "|&#916;a|"]
    b1, b2, pins = [], [], []
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
        wf = [w for r in fs for w in X.watch if w["tag"] == r["tag"] and w["iteration"] == r["iteration"]]
        shk = [w["ang_jerk_hold_median"] for w in wf if w.get("ang_jerk_hold_median") is not None]
        pin = sorted({d for r in fs for d in (pinned_desc(r["tag"], r["iteration"]) or [])})
        gc = [v for v in (grip_chatter(r["tag"]) for r in fs) if v is not None]
        b2.append([f"{swatch(a)}{LBL[a]}", f1(md("grip_N"), 1), f1(float(np.median(gc)), 1) if gc else "&#8211;",
                   f1(md("peak_force_N_mean"), 0), f1(md("pen_max_mm_mean"), 2),
                   f1(md("creep_mm_s_median"), 2), f1(md("creep_deg_s_median"), 2),
                   f1(float(np.median(shk)), 0) if shk else "&#8211;", f1(md("dact_held"), 2)])
        pins.append(f"{SHORT[a]} {', '.join(pin) if pin else 'none'}")
    out += table(h1, b1, "Final policies, 64 rollouts per seed: held rollouts and the turn. Held and reach-0.9 counts "
                         "pool the seeds; the other columns are medians over seeds.")
    out += table(h2, b2, "Final policies over their held steps: forces, penetration, creep in the last second, shaking "
                         "in the hold and action change (definitions in Terms and metrics); medians over seeds.",
                 note="Servo targets at their range limit in 90&#8202;% of the active steps: " + "; ".join(pins) + ".")
    return out


def transfer_section(X: Data):
    cap = ("Transfer of the final policies, one row per contact model and seed. Cell: the share of rollouts held at the end "
           "(load test; for the replays, worlds held of worlds replayed) and the mean final cosine of the held ones; "
           "outlined, the contact model the policy trained on. "
           "Left: closed-loop evaluation in MuJoCo-Warp under every contact model "
           "(the condim-4 TPU mesh was not trained). Right: open-loop replay of the recorded finger targets from the onset "
           "of the turn in CPU MuJoCo (each arm&#8217;s own tips), Drake and Newton (the TPU block as a hydroelastic tip; "
           "Newton with \\(k_h\\) divided by the tip-tool effective mass and its friction rows at the pads&#8217; "
           "10&#8202;ms). Drake discretizes the same pressure law as the pads.")
    return figure(svg_transfer(X), cap) + transfer_text(X)


def replay_stats(X: Data, arms, engine):
    """(fall times in s of the replays that dropped the tool, cos_end minus the MuJoCo-Warp rollout's final cosine of
    the replays that held it) for the finished runs of `arms` replayed in `engine`."""
    rec = {r["tag"]: r for r in X.rec}
    fall, dcos = [], []
    for arm in arms:
        for r in X.replays(arm, engine):
            if not r.get("held_end"):
                t = next((x[0] for x in r["trace"] if x[2] < 0.06), None)
                if t is not None:
                    fall.append(t)
            elif r["dir"] in rec:
                dcos.append(r["cos_end"] - rec[r["dir"]]["mjw_cos_end"][r["world"]])
    return fall, dcos


TPU_SCENES = ("tpu27mesh", "tpu27meshc4", "tpu27pads1", "tpu27skin")


def tpu_transfer(X: Data, arm):
    """(held, rollouts, {seed: (held, rollouts)}) of an arm's final policies evaluated closed loop in MuJoCo-Warp under
    the contact models of the TPU block other than their own, or None before the matrix is complete."""
    rs = [r for r in X.tr if r["arm"] == arm and r["scene"] in TPU_SCENES and r["scene"] != arm]
    if len(rs) < len(X.finals(arm)) * (len(TPU_SCENES) - (arm in TPU_SCENES)):
        return None
    per = {}
    for r in rs:
        h, n = per.get(r["seed"], (0, 0))
        per[r["seed"]] = (h + r["n_held"], n + r["n"])
    return sum(r["n_held"] for r in rs), sum(r["n"] for r in rs), per


def transfer_text(X: Data):
    """Sentences on the transfer matrix (MuJoCo-Warp, other contact models) and the replays, from the rows."""
    out = []
    T = {a: tpu_transfer(X, a) for a in ARMS}
    if all(T.values()):
        pc = lambda t, w="": f"{t[0]:,} of {t[1]:,}{w} ({100 * t[0] / t[1]:.0f}&#8202;%"  # noqa: E731
        m = sorted(T["tpu27mesh"][2].items(), key=lambda kv: -kv[1][0])
        ms = ", ".join(f"seed {s} {'held ' if k == 0 else ''}{h:,}" for k, (s, (h, n)) in enumerate(m))
        box = [r for r in X.tr if r["scene"] == "box" and r["arm"] != "box"]
        bh = {a: (sum(r["n_held"] for r in box if r["arm"] == a), sum(r["n"] for r in box if r["arm"] == a))
              for a in ARMS if a != "box"}
        held_box = [a for a, (h, n) in bh.items() if h > 0.25 * n]
        out.append(f"Closed loop in MuJoCo-Warp under the other contact models of the TPU block, the skin policies held "
                   f"{pc(T['tpu27skin'], ' rollouts')}), the pad policies {pc(T['tpu27pads1'])}) and the TPU-mesh "
                   f"policies {pc(T['tpu27mesh'])}: {ms}). The box-tip policy held {pc(T['box'])}) on the TPU block, and on the box "
                   f"tip, a different shape, " + (f"only the {' and '.join(SHORT[a] for a in held_box)} policies held more "
                   f"than a quarter of the rollouts ("
                   + (f"{bh[held_box[0]][0]} of {bh[held_box[0]][1]}" if len(held_box) == 1 else
                      ', '.join(f'{SHORT[a]} {bh[a][0]} of {bh[a][1]}' for a in held_box)) + ")."
                   if held_box else "no other policy held more than a quarter of the rollouts."))
    rep_txt = []
    if X.rep:
        pt, comp = ("box", "tpu27mesh"), ("tpu27pads1", "tpu27skin")
        fm, _ = replay_stats(X, pt, "mujoco")
        fd, _ = replay_stats(X, pt, "drake")
        fn, _ = replay_stats(X, pt, "newton")
        _, cm = replay_stats(X, comp, "mujoco")
        _, cd = replay_stats(X, comp, "drake")
        _, cn = replay_stats(X, comp, "newton")
        if fm and fd and cm and cd:
            md = lambda v: float(np.median(v))  # noqa: E731
            rep_txt.append(f"In the open-loop replays (Figure&#160;{FIG[0]}, right), CPU MuJoCo runs each policy&#8217;s "
                           f"own fingertip model, so for the point-contact policies only the implementation changes; their "
                           f"tool falls a median {md(fm):.2f}&#8202;s after the onset there ({md(fd):.2f}&#8202;s in Drake"
                           + (f", {md(fn):.2f}&#8202;s in Newton" if fn else "") + ").")
            sg = lambda v, nd: f"{v:+.{nd}f}".replace("-", "&#8722;")  # noqa: E731
            rel = lambda v: f"{abs(v):.2f} {'below' if v < 0 else 'above'} it"  # noqa: E731
            gap = lambda v: f"{v:+.3f}".replace("-", "&#8722;") if abs(v) < 0.01 else f"{v:+.2f}".replace("-", "&#8722;")  # noqa: E731
            rep_txt.append(f"The transfer gap in cosine of the pad and skin replays, their final cosine minus that of the "
                           f"MuJoCo-Warp rollout they replay, is a median {gap(md(cm))} in CPU MuJoCo ({sg(min(cm), 3)} to "
                           f"{sg(max(cm), 3)}), {gap(md(cd))} in Drake" + (f" and {gap(md(cn))} in Newton" if cn else "")
                           + ".")
        WHY = {"the tool rocking every policy step": "rocking the tool", "the tool shaken": "shaking the tool",
               "every rollout dropped": "dropping every rollout"}
        st = []
        for tag in sorted(X.degen, key=lambda t: (ARMS.index(tag_arm(t)), t)):
            arm = tag_arm(tag)
            cnt = {g: (sum(1 for r in X.replays(arm, g, True) if r.get("held_end") and r["dir"].endswith(tag)),
                       sum(1 for r in X.replays(arm, g, True) if r["dir"].endswith(tag))) for g in ENGINES}
            if not any(n for _, n in cnt.values()):
                continue
            kinds = next((d[2] for d in degenerate_stops(X, arm) if d[0] == int(tag[-1])), [])
            why = next((WHY[k] for k in ("the tool rocking every policy step", "the tool shaken", "every rollout dropped")
                        if k in kinds), "")
            name = {"box": "box-tip", "tpu27mesh": "TPU-mesh", "tpu27pads1": "pad", "tpu27skin": "skin"}[arm]
            m = (X.degen[tag]["iteration"] + 1) * STEPS_PER_IT / 1e6
            st.append((f"{name} seed {tag[-1]} ({m:.0f}&#8202;M steps" + (f", {why}" if why else "") + ")", cnt))
        if st:
            def cn_(c):
                return (f"{c['drake'][0]} of {c['drake'][1]} Drake, {c['mujoco'][0]} of {c['mujoco'][1]} CPU MuJoCo and "
                        f"{c['newton'][0]} of {c['newton'][1]} Newton replays")
            rep_txt.append("The last checkpoints of runs the watch stopped kept the tool in " +
                           "; ".join(f"{cn_(c)} for {lab}" for lab, c in st) + ".")
    out = [" ".join(out)] if out else []
    if rep_txt:
        out.append(" ".join(rep_txt))
    return "".join(f"<p>{x}</p>" for x in out)


def robust_section(X: Data):
    nr = sorted({r["n"] for r in X.rb})
    cap = (f"Held fraction of each arm&#8217;s final policies (pooled over seeds, {'&#8211;'.join(map(str, nr[::max(1, len(nr) - 1)])) or 64} "
           "rollouts each) under one perturbation "
           "at a time: every geom&#8217;s sliding friction scaled, the tool&#8217;s mass and inertia scaled, the finger "
           "servos&#8217; position gain over the 2&#8211;10&#8202;N&#8202;m/rad range of the 2026-10-06 bench readbacks, "
           "a 1&#8202;ms physics step (the training step is 2&#8202;ms; the policy stays at 50&#8202;Hz), and the tool "
           "spawned with uniform noise of &#177;2&#8202;mm and &#177;5&#176;.")
    return figure(svg_robust(X), cap, legend()) + robust_text(X)


def robust_text(X: Data):
    """Sentences on the perturbation rows: the servo gain, the other perturbations, from robust.jsonl."""
    if len(X.rb) < len(X.fin) * 10:
        return ""
    rb = {(r["tag"], r["perturb"]): r for r in X.rb}
    tags = sorted({r["tag"] for r in X.fin}, key=lambda t: (ARMS.index(tag_arm(t)), t))
    pt = [t for t in tags if tag_arm(t) in ("box", "tpu27mesh")]
    cp = [t for t in tags if tag_arm(t) in ("tpu27pads1", "tpu27skin")]
    nm = lambda t: f"{SHORT[tag_arm(t)]} seed {t[-1]}"  # noqa: E731
    held = lambda t, p: f"{rb[(t, p)]['n_held']} of {rb[(t, p)]['n']}"  # noqa: E731

    def pool(ts, p):
        return sum(rb[(t, p)]["n_held"] for t in ts), sum(rb[(t, p)]["n"] for t in ts)
    k10 = [t for t in tags if rb[(t, "kp=10")]["n_held"] > 0.25 * rb[(t, "kp=10")]["n"]]
    lost = [rb[(t, "kp=10")]["lost_step_median"] for t in tags if rb[(t, "kp=10")].get("lost_step_median") is not None]
    p6, c6 = pool(pt, "kp=6"), pool(cp, "kp=6")
    exc = (" in all but " + ", ".join(f"{nm(t)} ({held(t, 'kp=10')} held)" for t in k10)) if k10 else ""
    out = [f"At a finger servo gain \\(k_p\\) of 10&#8202;N&#8202;m/rad, "
           f"2.5&#215; the training plant&#8217;s, the policies lose the tool after the onset (median loss at policy step "
           f"{min(lost):.0f}&#8211;{max(lost):.0f}){exc}; at 6&#8202;N&#8202;m/rad the point-contact policies held "
           f"{p6[0]} of {p6[1]} rollouts and the pad and skin policies {c6[0]} of {c6[1]}."]
    other = ("friction=0.7", "friction=1.3", "mass=0.8", "mass=1.2", "dt=0.001", "noise=2,5")
    low = [(t, p) for t in tags if tag_arm(t) != "box" for p in other if rb[(t, p)]["n_held"] < 0.95 * rb[(t, p)]["n"]]
    lab = lambda p: {"dt=0.001": "the 1&#8202;ms step", "noise=2,5": "the pose noise"}.get(p, p.replace("=", " &#215;"))  # noqa: E731
    bx = [rb[(t, p)]["n_held"] for t in pt if tag_arm(t) == "box" for p in ("nominal",) + other]
    out.append("Friction &#177;30&#8202;%, tool mass &#177;20&#8202;%, the 1&#8202;ms step and the grasp-pose noise "
               "keep every TPU-block policy at 95&#8202;% held or more"
               + (" except " + " and ".join(f"{nm(t)} at {lab(p)} ({held(t, p)})" for t, p in low) if low else "")
               + (f"; the box-tip policy holds {min(bx)}&#8211;{max(bx)} of 128 under these and at the nominal." if bx
                  else "."))
    turn = [t for t in tags if rb[(t, "kp=2")].get("held_cos_mean") is not None
            and rb[(t, "kp=2")]["held_cos_mean"] < rb[(t, "nominal")]["held_cos_mean"] - 0.1]
    if turn:
        out.append("At 2&#8202;N&#8202;m/rad "
                   + " and ".join(f"{nm(t)} turns the tool to {rb[(t, 'kp=2')]['held_cos_mean']:.2f} instead of "
                                  f"{rb[(t, 'nominal')]['held_cos_mean']:.2f}" for t in turn) + ".")
    return "<p>" + " ".join(out) + "</p>"


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
            n_fin=len(fins), stopped=sorted(t for t in X.degen if tag_arm(t) == arm),
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
    """About 120 words (owner, 2026-10-09: a lede of about 100 words): the transfer, then training and the trade."""
    S = summary(X)
    if not all(S[a]["n_fin"] for a in ARMS):
        done = ", ".join(f"{SHORT[a]} {S[a]['n_fin']}" for a in ARMS)
        return f"Training in progress: finished seeds per contact model {done}."
    comp, point = ("tpu27pads1", "tpu27skin"), ("box", "tpu27mesh")
    W = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 12: "Twelve"}
    rep = lambda arms: [sum(S[a]["rep"][e][k] for a in arms for e in ENGINES) for k in (0, 1)]  # noqa: E731
    rc, rp = rep(comp), rep(point)
    n_runs = len({r["tag"] for r in X.watch})
    p1 = [f"{W.get(n_runs, n_runs) if n_runs > 6 else n_runs} PPO runs of up to 40&#8202;M steps in MuJoCo-Warp trained "
          f"the D6 screwdriver turn. Replayed open loop in CPU MuJoCo, Drake and "
          f"Newton, the finger targets of the pad and skin policies (1&#8202;mm sphere pads, bare or on a sprung skin) kept "
          f"the tool in {rc[0]} of {rc[1]} replays, those of the point-contact policies in {rp[0]} of {rp[1]}."]
    T = {a: tpu_transfer(X, a) for a in ("tpu27mesh", "tpu27pads1", "tpu27skin")}
    if all(T.values()):
        pct = lambda t: f"{100 * t[0] / t[1]:.0f}&#8202;%"  # noqa: E731
        p1.append(f"Closed loop under the other TPU-block contact models, the skin policies held {pct(T['tpu27skin'])} of "
                  f"rollouts, the pad policies {pct(T['tpu27pads1'])} and the TPU-mesh ones {pct(T['tpu27mesh'])}.")
    deg = {fam: (sum(len(degenerate_stops(X, a)) for a in arms), sum(len({r["tag"] for r in X.watch if r["arm"] == a})
                                                                      for a in arms))
           for fam, arms in (("point", point), ("comp", comp))}
    wd = lambda n: W.get(n, n)  # noqa: E731
    p2 = [f"The checkpoint watch found {wd(deg['point'][0])} of {wd(deg['point'][1])} point-contact runs and "
          f"{wd(deg['comp'][0])} of {wd(deg['comp'][1])} pad and skin runs degenerate for 10&#8202;M steps: every "
          f"rollout dropped, or the tool shaken or rocked."]
    hp = [h for a in point for h in S[a]["hcos"]]
    hc = [h for a in comp for h in S[a]["hcos"]]
    pm, pp, ps = probe_spit("tpu27mesh"), probe_spit("tpu27pads1"), probe_spit("tpu27skin")
    cost = (f"; pad and skin training cost {pp / pm:.1f}&#215; and {ps / pm:.1f}&#215; per iteration" if pm and pp and ps
            else "")
    p2.append(f"Point-contact policies turned the tool further where they held it (cosine with vertical {_rng(hp)} against "
              f"{_rng(hc)}){cost}.")
    return " ".join(p1) + '</p><p class="lede">' + " ".join(p2)


def open_items(X: Data):
    W = {1: "one", 2: "two", 3: "three"}
    NM = {"box": "box-tip", "tpu27mesh": "TPU-mesh", "tpu27pads1": "pad", "tpu27skin": "skin"}
    parts = [f"{W.get(len(X.finals(a)), len(X.finals(a)))} {NM[a]} seed{'s' if len(X.finals(a)) != 1 else ''}"
             for a in ARMS]
    fin_txt = ", ".join(parts[:-1]) + " and " + parts[-1] + " with a final policy"
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
         "and one pip servo target per pad or skin policy sits at its extension limit in 95&#8202;% of the steps (Table 4). A "
         "deployable residual needs <code>clip_actions</code> 1.0 (<code>src/morphohand/rl/ppo_config.py</code>); "
         "retrain one pad and one skin seed with it and compare the held cosine."),
        ("Servo gain.", "Every final policy but skin seed 2 loses the tool at \\(k_p\\) = 10&#8202;N&#8202;m/rad, the top of "
         "the 2026-10-06 readback range, and the runs train at 4. Fine-tune the pad and skin finals with \\(k_p\\) drawn per "
         "episode from 2&#8211;10 (a reset event in <code>_build_events</code> of "
         "<code>src/morphohand/rl/env_build.py</code>, which has none yet) and repeat <code>scripts/rl_contact_eval.py "
         "robust --perturb kp=2 kp=6 kp=10</code>; a fine-tuned policy that still loses the tool at 10 would mean the "
         "turn depends on the soft servo."),
        ("Seeds of the point-contact arms.", f"The stop rule leaves {fin_txt}; the point-contact seed spread rests on "
         "those runs, and the box tip&#8217;s only finished seed meets the rule in hindsight. Two more box-tip and "
         "TPU-mesh seeds trained to 40&#8202;M without the rule (about 1.3&#8202;GPU-h each) show whether a "
         "point-contact policy that keeps the tool still exists; one that holds 64/64 with a shaking under "
         "40&#8202;rad/s&#178; and survives the CPU MuJoCo replay would contradict the finding that point-contact "
         "training shakes the tool loose."),
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
    v["TRANSFER"] = transfer_section(X)
    v["CURVES"] = curves_section(X)
    v["FINAL"] = final_section(X)
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
    t = retro_style.apply(t)  # plain page style (owner, 2026-10-09)
    open(OUT, "w").write(t)
    print(f"formulas {n}; wrote {OUT} ({os.path.getsize(OUT) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
