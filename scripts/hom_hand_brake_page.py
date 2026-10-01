#!/usr/bin/env python3
"""Build docs/experiments/20261001-hom_hand_brake/20261001-hom_hand_brake.html from hand_brake.jsonl.

    python3 scripts/hom_hand_brake_page.py

Reads the rows written by `scripts/hom_hand_brake.py study` and the two-pad rig's calibration.json
(the quasi-static law); charts and tables reuse scripts/hom_contact_patch_page.py's helpers.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import hom_contact_patch_page as R  # noqa: E402  (shared table/chart helpers, quasi-static law)

D = os.path.join(ROOT, "docs/experiments/20261001-hom_hand_brake")
OUT = os.path.join(D, "20261001-hom_hand_brake.html")
TPL = os.path.join(ROOT, "scripts/hom_hand_brake_page.template.html")
RIG_URL = "https://claude.ai/artifact/Rvfw1yQFgfWV8PdTvv7jsc"
RIG_PATH = "docs/experiments/20261001-hom_contact_patch/20261001-hom_pinch_contact_models.html"
SPEC = {"dpt": "drake:point", "d001": "drake:hydro:rt0.01", "d002": "drake:hydro:rt0.02", "ddef": "drake:hydro",
        "mp3": "mj:point3", "p4s": "mj:point4s", "s1": "mj:spheres:s1:rs0.75:tr0.02",
        "s1tr01": "mj:spheres:s1:rs0.75:tr0.1"}
LABEL = {"dpt": "Drake point contact", "d001": "Drake hydroelastic, relaxation 0.01&#8202;s",
         "d002": "Drake hydroelastic, relaxation 0.02&#8202;s", "ddef": "Drake hydroelastic, relaxation 0.1&#8202;s (default)",
         "mp3": "MuJoCo point contact, condim&#160;3", "p4s": "MuJoCo condim&#160;4, &#956;<sub>t</sub> rescheduled",
         "s1": "MuJoCo 1&#8202;mm sphere fingertips, relaxation 0.02&#8202;s",
         "s1tr01": "MuJoCo 1&#8202;mm sphere fingertips, relaxation 0.1&#8202;s"}
ROWS = [json.loads(l) for l in open(os.path.join(D, "hand_brake.jsonl")) if l.strip()]


def row(key, d=15.0):
    return next((r for r in ROWS if r["spec"] == SPEC[key] and abs(r["d_mm"] - d) < 1e-6), None)


def series(key, d=15.0):
    tr = np.array(row(key, d)["trace"])
    tr = tr[tr[:, 0] >= 0.6][::3]
    pad = 0.5 * (tr[:, 4] + tr[:, 5])
    m = pad > 0.05
    return [(float(n), float(p)) for n, p in zip(pad[m], tr[m, 2])]


def chart(which):
    qs = [(N, R.phi_qs(N, 0.015)) for N in np.geomspace(4.0, 0.25, 140)]
    ref = {"name": "quasi-static law", "color": R.COL["ref"], "dash": True, "pts": qs}
    if which == "A":
        ss = [ref, {"name": "Drake hydroelastic 0.01 s", "color": R.COL["drake"], "pts": series("d001")},
              {"name": "MuJoCo sphere tips 0.02 s", "color": R.COL["sphere"], "pts": series("s1")},
              {"name": "MuJoCo condim 4 rescheduled", "color": R.COL["c4"], "dash": True, "pts": series("p4s")}]
        leg = [{"name": "Drake hydroelastic, 0.01&#8202;s", "color": R.COL["drake"]},
               {"name": "MuJoCo sphere fingertips, 0.02&#8202;s", "color": R.COL["sphere"]},
               {"name": "MuJoCo condim&#160;4, rescheduled", "color": R.COL["c4"], "dash": True},
               {"name": "quasi-static law", "color": R.COL["ref"], "dash": True}]
        title = "Short relaxation"
    else:
        ss = [ref, {"name": "Drake hydroelastic 0.1 s", "color": R.COL["drake"], "pts": series("ddef")},
              {"name": "MuJoCo sphere tips 0.1 s", "color": R.COL["sphere"], "pts": series("s1tr01")}]
        leg = [{"name": "Drake hydroelastic, 0.1&#8202;s (default)", "color": R.COL["drake"]},
               {"name": "MuJoCo sphere fingertips, 0.1&#8202;s", "color": R.COL["sphere"]},
               {"name": "quasi-static law", "color": R.COL["ref"], "dash": True}]
        title = "0.1&#8202;s relaxation"
    return R.svg_chart(title, "&#966; in degrees against pad force, d&#8202;=&#8202;15&#8202;mm", ss, leg, xlog=True, xmin=0.25,
                       xmax=4.4, xticks=(4, 2, 1, 0.5, 0.25), ymin=-5, ymax=130, yticks=(0, 30, 60, 90, 120), rev=True,
                       xname="N", xd=2, yd=1, xlabel="pad force, N (falls left to right)", w=560, h=380)


def table():
    head = ["model", "N at 10&#176;", "N at 45&#176;", "N at 80&#176;", "final &#966;", "slip mm", "max tilt", "CPU s"]
    body = []
    for d in (15.0, 30.0):
        body.append(f"d&#8202;=&#8202;{d:g}&#8202;mm")
        body.append(["quasi-static law, rig&#8217;s Drake r&#772;(N)"] + [R.f2(R.n_qs(a, d * 1e-3)) for a in (10, 45, 80)]
                    + ["90&#176;", "", "", ""])
        for k in SPEC:
            r = row(k, d)
            if r is None:
                continue
            tr = r["trace"]
            pad0 = 0.5 * (tr[0][4] + tr[0][5])
            point = r["N_at_10"] is not None and r["N_at_10"] > 0.8 * pad0
            last = tr[-1]
            dropped = last[7] == 0 or last[8] == 0 or abs(r["slip_end_mm"]) > 20
            cells = ["at release" if point else R.f2(r["N_at_10"])]
            for a in (45, 80):
                v = r[f"N_at_{a}"]
                cells.append(("at release", "cell c0") if point else R.closeness(v, R.n_qs(a, d * 1e-3), nd=2))
            cells.append("dropped" if dropped else ("swinging" if point else f"{r['phi_end']:.1f}&#176;"))
            cells.append("&#8211;" if dropped else f"{abs(r['slip_end_mm']):.2f}")
            cells.append(f"{r['tilt_max']:.1f}&#176;")
            cells.append(f"{r['wall_s']:.1f}")
            body.append([LABEL[k]] + cells)
    return R.table(head, body, cls_num={1, 2, 3, 4, 5, 6, 7}) + (
        '<p class="note" style="font-size:13.5px;color:var(--ink3)">N at an angle = mean pad force when the tool first passes it, '
        'shaded by distance from the quasi-static law (within 5, 10, 25&#8202;%, beyond). Max tilt = largest angle of the tool axis '
        'out of the plane normal to the pinch axis. CPU s = wall time of the 5.6&#8202;s run with its film on one core.</p>')


def main():
    t = open(TPL).read()
    v = {"BUILT": time.strftime("%Y-%m-%d %H:%M"), "DRAKE_V": R.DRAKE_V, "MUJOCO_V": R.MUJOCO_V,
         "RIG_URL": RIG_URL, "RIG_PATH": RIG_PATH, "C_FIT": f"{R.CF * 1e3:.2f}", "N_ROWS": str(len(ROWS))}
    for key, tag in (("d001", "D"), ("p4s", "C4"), ("s1", "SP")):
        r = row(key)
        v[f"H45_{tag}"], v[f"H80_{tag}"] = f"{r['N_at_45']:.2f}", f"{r['N_at_80']:.2f}"
    v["Q45"], v["Q80"] = f"{R.n_qs(45, 0.015):.2f}", f"{R.n_qs(80, 0.015):.2f}"
    dev = lambda k: [abs(row(k, d)[f"N_at_{a}"] / row("d001", d)[f"N_at_{a}"] - 1) * 100  # noqa: E731
                     for d in (15.0, 30.0) for a in (45, 80)]
    v["AGREE_C4"] = f"{max(dev('p4s')):.0f}"
    v["AGREE_SP"] = f"{min(dev('s1')):.0f}&#8211;{max(dev('s1')):.0f}"
    v["DROP_GAP"] = f"{max(row(k)['N_at_45'] - row(k)['N_at_80'] for k in ('ddef', 's1tr01')):.1f}"
    v["CHART_A"], v["CHART_B"] = chart("A"), chart("B")
    v["TABLE"] = table()
    v["FILM"] = R.data_uri(os.path.join(D, "media/20261001-hand_brake_eight_models.mp4"), "video/mp4")
    v["POSTER"] = R.data_uri(os.path.join(D, "media/20261001-hand_brake_eight_models_poster.png"), "image/png")
    v["FILM_PATH"] = "docs/experiments/20261001-hom_hand_brake/media/20261001-hand_brake_eight_models.mp4"
    for k, val in v.items():
        t = t.replace("{{" + k + "}}", val)
    left = sorted(set(x.split("}}")[0] for x in t.split("{{")[1:]))
    if left:
        raise SystemExit(f"unfilled placeholders: {left}")
    open(OUT, "w").write(t)
    print(f"wrote {OUT} ({os.path.getsize(OUT) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
