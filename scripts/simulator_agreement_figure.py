#!/usr/bin/env python3
"""Agreement of fingertip contact models with Drake hydroelastic on three tasks (2026-10-06).

svg_agreement(): one row per contact model and one column per task metric, each a difference from or a ratio to
Drake hydroelastic on the same task. svg_turn_pairs(): the plan-replay tool turn of every rollout against Drake's,
one panel per model. Both return inline SVG in the house style (colour tokens of contact_overview_page.style_block)
for the hom_turn3 page, in place of its per-hand turn strip, and for the contact-model overview.

Tasks: twist and brake on the 2026-10-05 contact bed (two pads pinch the 24.5 g tool; onset torque at 0.5, 1 and
3 N; largest swing of the tool under gravity), and the open-loop plan replay of the three-finger turn on D1-D8
(working plant kp 4 N m/rad, TPU tip with 2.7 mm fillets, mu 1, placement jitter j2; seeds 0-2 pair with Drake by
hand and seed). Contact: 1 mm sphere pads in MuJoCo CPU, MuJoCo-Warp and Newton; Newton hydroelastic with kh times
the solver's inverse-weight sum; MuJoCo point contact (condim 3). A model without rows draws as "not run".

Rows: the bed (20261005-contact_bed, 20261006-newton_mass_scaling), the plan replays (20261006-hom_turn3) and
20261006-simulator_agreement: point contact (plans_mujoco_pt.jsonl), MuJoCo-Warp and Newton pads
(plans_mjwarp_pads.jsonl, plans_newton_pads.jsonl), the mass-corrected Newton turn rerun with pad forces
(plans_newton_hydro_mc.jsonl), and the bed rows of contact_bed_newton.py --outdir there (twist_slip_newton.jsonl,
brake_newton.jsonl: models mjw_pads1, newton_pads1, newton_hydro_mc).

  python3 scripts/simulator_agreement_figure.py   # writes a preview PNG next to the rows
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import contact_overview_page as P  # noqa: E402

EXP = ROOT / "docs/experiments"
BED = EXP / "20261005-contact_bed"
MSC = EXP / "20261006-newton_mass_scaling"
HT = EXP / "20261006-hom_turn3"
SA = EXP / "20261006-simulator_agreement"
NS = (0.5, 1.0, 3.0)
TOL_DEG = 3.0

# key, label, colour, shape, plan-replay rows, model name in the bed rows
MODELS = [
    ("mj_pads", "MuJoCo CPU, 1 mm pads", "var(--c-sphere)", "circle", HT / "plans_mujoco.jsonl", "mj_pads1"),
    ("mjw_pads", "MuJoCo-Warp, 1 mm pads", "var(--c-c4)", "circle", SA / "plans_mjwarp_pads.jsonl", "mjw_pads1"),
    ("nt_pads", "Newton, 1 mm pads", "var(--c-newton)", "circle", SA / "plans_newton_pads.jsonl", "newton_pads1"),
    ("nt_hydro", "Newton hydro, mass-corrected", "var(--c-newton)", "diamond", SA / "plans_newton_hydro_mc.jsonl",
     "newton_hydro_mc"),
    ("mj_pt", "MuJoCo CPU, point contact", "var(--ink3)", "square", SA / "plans_mujoco_pt.jsonl", "mj_point3"),
]
TITLES = {"mj_pads": ("MuJoCo CPU", "1 mm pads"), "mjw_pads": ("MuJoCo-Warp", "1 mm pads"),
          "nt_pads": ("Newton", "1 mm pads"), "nt_hydro": ("Newton hydro", "mass-corrected"),
          "mj_pt": ("MuJoCo CPU", "point contact")}


def jl(path):
    path = Path(path)
    return [json.loads(line) for line in path.open() if line.strip()] if path.exists() else []


def _ok(r):
    return r.get("status", "complete") in ("complete", "ok")


def _bed(files, field):
    out = {}
    for f in files:
        for r in jl(f):
            if r.get("dt_ms", 1.0) == 1.0 and _ok(r) and r.get(field) is not None:
                out[(r["model"], r.get("N"))] = r
    return out


def data():
    """Per model: twist ratios (0.5/1/3 N), brake (swing difference, held), turn pairs and grip-force ratios."""
    tw = _bed([BED / "twist_slip.jsonl", BED / "twist_slip_newton.jsonl", MSC / "twist_slip_newton.jsonl",
               SA / "twist_slip_newton.jsonl"], "tau_onset_Nm")
    br = _bed([BED / "brake.jsonl", SA / "brake_newton.jsonl"], "phi_max_deg")
    dk = {(r["hand"], int(r["seed"])): r for r in jl(HT / "plans_drake.jsonl") if _ok(r)}
    out = {}
    for key, _lab, _col, _shape, turn_rows, bed_model in MODELS:
        d = {}
        if all((bed_model, N) in tw for N in NS):
            d["twist"] = [tw[(bed_model, N)]["tau_onset_Nm"] / tw[("drake_hydro", N)]["tau_onset_Nm"] for N in NS]
        b = br.get((bed_model, None))
        if b:
            d["brake"] = [(b["phi_max_deg"] - br[("drake_hydro", None)]["phi_max_deg"], bool(b.get("held_station")))]
        rows = {(r["hand"], int(r["seed"])): r for r in jl(turn_rows) if _ok(r)}
        pairs = [(dk[k], rows[k]) for k in sorted(dk) if k in rows]
        if pairs:
            d["pairs"] = [(a["turn_end_deg"], m["turn_end_deg"], bool(m["held_end"])) for a, m in pairs]
            d["turn"] = [(m["turn_end_deg"] - a["turn_end_deg"], bool(m["held_end"])) for a, m in pairs]
            grip = [m["F_pads_hold_N"] / a["F_pads_hold_N"] for a, m in pairs
                    if a["held_end"] and m["held_end"] and m.get("F_pads_hold_N") and a.get("F_pads_hold_N")]
            if grip:
                d["grip"] = [(g, True) for g in grip]
            d["n"] = len(pairs)
            d["held"] = sum(m["held_end"] for _a, m in pairs)
            d["within"] = sum(1 for dv, h in d["turn"] if h and abs(dv) <= TOL_DEG)
        out[key] = d
    return out


def _text(x, y, s, anchor="start", fill="var(--ink)", size=None, extra=""):
    fs = f' font-size="{size}"' if size else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}"{fs} style="fill:{fill}"{extra}>{s}</text>'


def _scale(kind, lim, x0, w):
    if kind == "log":
        a, b = math.log(lim[0]), math.log(lim[1])
        return lambda v: x0 + (math.log(min(max(v, lim[0]), lim[1])) - a) / (b - a) * w
    return lambda v: x0 + (min(max(v, lim[0]), lim[1]) - lim[0]) / (lim[1] - lim[0]) * w


def _swarm(xs, yc, r, step, slots=3):
    """Greedy beeswarm: place each x at the first vertical slot (0, +1, -1, ...) clear of earlier markers."""
    placed, ys = [], []
    order = [0] + [s for k in range(1, slots + 1) for s in (k, -k)]
    for x in xs:
        for s in order:
            y = yc + s * step
            if all(abs(x - px) >= 2 * r or abs(y - py) >= 2 * r for px, py in placed):
                break
        placed.append((x, y))
        ys.append(y)
    return ys


COLS = [  # key, task, metric, relation to Drake, axis, limits, ticks, Drake band
    ("twist", "Twist, bed", "onset torque", "&#247; Drake", "log", (0.03, 3.0), (0.1, 0.3, 1, 3), (0.9, 1.1)),
    ("brake", "Brake, bed", "largest swing", "&#8722; Drake (&#176;)", "lin", (-10.0, 100.0), (0, 50, 100), (-3, 3)),
    ("turn", "Turn, hand", "tool turn", "&#8722; Drake (&#176;)", "lin", (-50.0, 40.0), (-40, 0, 40), (-3, 3)),
    ("grip", "Turn, hand", "grip force", "&#247; Drake", "log", (0.5, 2.0), (0.5, 1, 2), (0.9, 1.1)),
]


def svg_agreement(D=None):
    D = D or data()
    LW, CW, GAP = 252, 132, 28
    W = LW + len(COLS) * (CW + GAP)
    y0, RH = 68, 50
    sm = math.ceil(13.0 * W / P.COLUMN_PX * 10) / 10     # small text: 13 px as displayed in the 828 px column
    yb = y0 + RH * len(MODELS)
    out = P._svg_open(W, yb + 36, "Agreement of five contact models with Drake hydroelastic on three tasks")
    for i in range(len(MODELS) + 1):
        out.append(f'<line x1="8" x2="{W - 4}" y1="{y0 + i * RH}" y2="{y0 + i * RH}" style="stroke:var(--rule2)"/>')
    for j, (ck, t1, t2, t3, kind, lim, ticks, band) in enumerate(COLS):
        x0 = LW + j * (CW + GAP)
        fx = _scale(kind, lim, x0, CW)
        out.append(_text(x0 + CW / 2, 17, t1, "middle"))
        out.append(_text(x0 + CW / 2, 35, t2, "middle", "var(--ink2)", sm))
        out.append(_text(x0 + CW / 2, 53, t3, "middle", "var(--ink2)", sm))
        out.append(f'<rect x="{fx(band[0]):.1f}" y="{y0}" width="{fx(band[1]) - fx(band[0]):.1f}" height="{yb - y0}" '
                   'style="fill:color-mix(in srgb,var(--c-drake) 16%,transparent)"/>')
        ref = 1.0 if kind == "log" else 0.0
        out.append(f'<line x1="{fx(ref):.1f}" x2="{fx(ref):.1f}" y1="{y0}" y2="{yb}" '
                   'style="stroke:var(--c-drake);stroke-width:1.5"/>')
        out.append(f'<line x1="{x0}" x2="{x0 + CW}" y1="{yb}" y2="{yb}" style="stroke:var(--ink3)"/>')
        for t in ticks:
            out.append(f'<line x1="{fx(t):.1f}" x2="{fx(t):.1f}" y1="{yb}" y2="{yb + 4}" style="stroke:var(--ink3)"/>')
            out.append(_text(fx(t), yb + 18, f"{t:g}", "middle", "var(--ink3)", sm))
        for i, (mk, _lab, col, shape, _f, _b) in enumerate(MODELS):
            yc = y0 + (i + 0.5) * RH
            vals = D[mk].get(ck)
            if not vals:
                out.append(_text(x0 + CW / 2, yc + 4, "not run" if ck != "grip" or "turn" not in D[mk]
                                 else "not recorded", "middle", "var(--ink3)", sm, ' font-style="italic"'))
                continue
            vals = [(v, True) if not isinstance(v, tuple) else v for v in vals]
            xs = [fx(v) for v, _h in vals]
            ys = _swarm(xs, yc, 3.4, 6.2)
            for (v, held), x, y in zip(vals, xs, ys):
                P._marker(out, x, y, col, shape=shape, hollow=not held, r=3.4)
    for i, (mk, lab, col, _shape, _f, _b) in enumerate(MODELS):
        yc = y0 + (i + 0.5) * RH
        d = D[mk]
        out.append(_text(LW - 12, yc - 3, lab, "end", col if col != "var(--ink3)" else "var(--ink)"))
        sub = (f"turn &#177;{TOL_DEG:g}&#176;: {d['within']}/{d['n']}, held {d['held']}/{d['n']}"
               if "turn" in d else "turn not run")
        out.append(_text(LW - 12, yc + 15, sub, "end", "var(--ink3)", sm))
    out.append("</svg>")
    return "\n".join(out)


def svg_turn_pairs(D=None):
    D = D or data()
    PW, ML, MT, MB, GAP = 124, 40, 52, 44, 18
    W = ML + len(MODELS) * (PW + GAP)
    sm = math.ceil(13.0 * W / P.COLUMN_PX * 10) / 10     # small text: 13 px as displayed in the 828 px column
    lim = (-20.0, 80.0)
    out = P._svg_open(W, MT + PW + MB, "Tool turn of each plan replay against Drake, one panel per contact model")
    span = lim[1] - lim[0]
    for i, (mk, _lab, col, shape, _f, _b) in enumerate(MODELS):
        x0 = ML + i * (PW + GAP)
        fx = lambda v, x0=x0: x0 + (min(max(v, lim[0]), lim[1]) - lim[0]) / span * PW  # noqa: E731
        fy = lambda v: MT + PW - (min(max(v, lim[0]), lim[1]) - lim[0]) / span * PW  # noqa: E731
        t1, t2 = TITLES[mk]
        out.append(_text(x0 + PW / 2, 14, t1, "middle", col if col != "var(--ink3)" else "var(--ink)"))
        out.append(_text(x0 + PW / 2, 28, t2, "middle", "var(--ink2)", sm))
        d = D[mk]
        out.append(f'<clipPath id="tp{i}"><rect x="{x0}" y="{MT}" width="{PW}" height="{PW}"/></clipPath>')
        out.append(f'<rect x="{x0}" y="{MT}" width="{PW}" height="{PW}" style="fill:none;stroke:var(--rule)"/>')
        for t in (0, 30, 60):
            out.append(f'<line x1="{fx(t):.1f}" x2="{fx(t):.1f}" y1="{MT}" y2="{MT + PW}" style="stroke:var(--rule2)"/>')
            out.append(f'<line x1="{x0}" x2="{x0 + PW}" y1="{fy(t):.1f}" y2="{fy(t):.1f}" style="stroke:var(--rule2)"/>')
            out.append(_text(fx(t), MT + PW + 14, f"{t}", "middle", "var(--ink3)", sm))
            if i == 0:
                out.append(_text(x0 - 6, fy(t) + 4, f"{t}", "end", "var(--ink3)", sm))
        a, b = lim
        band = (f"{fx(a):.1f},{fy(a + TOL_DEG):.1f} {fx(b):.1f},{fy(b + TOL_DEG):.1f} "
                f"{fx(b):.1f},{fy(b - TOL_DEG):.1f} {fx(a):.1f},{fy(a - TOL_DEG):.1f}")
        out.append(f'<g clip-path="url(#tp{i})"><polygon points="{band}" '
                   'style="fill:color-mix(in srgb,var(--c-drake) 16%,transparent)"/>'
                   f'<line x1="{fx(a):.1f}" y1="{fy(a):.1f}" x2="{fx(b):.1f}" y2="{fy(b):.1f}" '
                   'style="stroke:var(--c-drake);stroke-width:1.2"/></g>')
        if "pairs" not in d:
            out.append(_text(x0 + PW / 2, MT + PW / 2 + 4, "not run", "middle", "var(--ink3)", sm,
                             ' font-style="italic"'))
            continue
        out.append(_text(x0 + PW / 2, 42, f"{d['within']}/{d['n']} within {TOL_DEG:g}&#176;", "middle",
                         "var(--ink2)", sm))
        for xd, ym, held in d["pairs"]:
            P._marker(out, fx(xd), fy(ym), col, shape=shape, hollow=not held, r=3.0)
    out.append(_text(ML + (W - ML) / 2, MT + PW + 34, "Drake hydroelastic: tool turn (&#176;)", "middle",
                     "var(--ink2)"))
    out.append(_text(12, MT + PW / 2, "model: tool turn (&#176;)", "middle", "var(--ink2)", None,
                     f' transform="rotate(-90 12 {MT + PW / 2})"'))
    out.append("</svg>")
    return "\n".join(out)


def legend():
    def key(draw):
        out = ['<svg viewBox="0 0 26 14" width="26" height="14" style="vertical-align:-2px">']
        draw(out)
        out.append("</svg>")
        return "".join(out)
    held = key(lambda o: P._marker(o, 13, 7, "var(--ink2)", r=4.2))
    drop = key(lambda o: P._marker(o, 13, 7, "var(--ink2)", hollow=True, r=4.2))
    drake = key(lambda o: o.append('<rect x="1" y="2" width="24" height="10" style="fill:color-mix(in srgb,'
                                   'var(--c-drake) 16%,transparent)"/><line x1="1" x2="25" y1="7" y2="7" '
                                   'style="stroke:var(--c-drake);stroke-width:1.5"/>'))
    items = [(held, "tool held to the end"), (drop, "tool dropped or released"),
             (drake, "Drake hydroelastic, &#177;3&#176; or &#177;10&#8202;% band")]
    return ('<div class="legendrow">' + "".join(f"<span>{k} {t}</span>" for k, t in items) + "</div>")


def main():
    D = data()
    for mk, *_ in MODELS:
        d = D[mk]
        tw = ", ".join(f"{v:.2f}" for v in d.get("twist", [])) or "-"
        print(f"{mk:9s} twist {tw:18s} brake {d.get('brake', '-')}  turn within {d.get('within', '-')}/{d.get('n', '-')} "
              f"held {d.get('held', '-')}  grip {'%.2f-%.2f' % (min(g for g, _ in d['grip']), max(g for g, _ in d['grip'])) if d.get('grip') else '-'}")
    SA.mkdir(parents=True, exist_ok=True)
    html = ROOT / "logs/20261006-simulator_agreement/preview.html"
    html.parent.mkdir(parents=True, exist_ok=True)
    html.write_text("<!doctype html><meta charset='utf-8'>" + P.style_block()
                    + "<body style='background:var(--card);padding:16px;width:820px'>"
                    + svg_agreement(D) + "<div style='height:24px'></div>" + svg_turn_pairs(D) + legend() + "</body>")
    png = SA / "20261006-simulator_agreement_preview.png"
    subprocess.run(["google-chrome", "--headless=new", "--disable-gpu", "--hide-scrollbars", "--window-size=860,720",
                    "--force-device-scale-factor=1.5", f"--screenshot={png}", html.as_uri()],
                   check=True, capture_output=True)
    print("preview", png)


if __name__ == "__main__":
    main()
