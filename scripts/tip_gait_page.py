#!/usr/bin/env python3
"""Build the bench partial-reorientation page from tip_gait_data.json.

    python3 scripts/tip_gait_data.py
    python3 scripts/tip_gait_page.py
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os

import numpy as np

ROOT = "docs/experiments/20260916-tip_gait"
DATA = f"{ROOT}/tip_gait_data.json"
TPL = "scripts/tip_gait_page.template.html"
OUT = f"{ROOT}/20260916-bench_partial_reorientation.html"
HANDS = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8"]


def uri(path):
    p = os.path.join(ROOT, path)
    return f"data:{mimetypes.guess_type(p)[0]};base64," + base64.b64encode(open(p, "rb").read()).decode()


def table(head, rows):
    out = ['<div class="tw"><table><thead><tr>']
    for h in head:
        out.append(f'<th{" class=num" if h.startswith("#") else ""}>{h.lstrip("#")}</th>')
    out.append("</tr></thead><tbody>")
    for r in rows:
        klass = ""
        if isinstance(r, tuple) and len(r) == 2 and isinstance(r[1], str) and isinstance(r[0], list):
            r, klass = r
        out.append(f'<tr{" class=" + klass if klass else ""}>')
        for h, c in zip(head, r):
            if isinstance(c, tuple):
                out.append(f'<td class="{c[1]}">{c[0]}</td>')
            else:
                out.append(f'<td{" class=num" if h.startswith("#") else ""}>{c}</td>')
        out.append("</tr>")
    out.append("</tbody></table></div>")
    return "\n".join(out)


def cellc(h, n, extra=""):
    frac = h / n if n else 0
    cls = "c4" if frac >= 0.99 else "c3" if frac >= 0.6 else "c2" if frac >= 0.4 else "c1" if frac > 0 else "c0"
    return (f"{h}/{n}{extra}", f"cell {cls}")


def table_trace(d):
    rows = []
    for s in d["trace_d7"]:
        c = s["contacts"]

        def cell(k):
            v = c.get(k)
            return "&#8211;" if v is None else f"{v['axial_mm']:+.0f} mm &#183; {v['N']:.2f} N"
        rows.append([s["phase"], f"{s['t']:.2f}", f"{s['from_vertical_deg']:.0f}", f"{s['z_mm']:.0f}",
                     cell("thumb"), cell("index"), cell("middle"), cell("plate"), cell("post")])
    return table(["phase", "#t (s)", "#from vertical (deg)", "#centre z (mm)", "thumb", "index", "middle",
                  "palm plate", "post"], rows)


def table_plate(d):
    rows = []
    for h in HANDS:
        hold, turn = d["hw"][h]
        r = [h, hold, f"{90 - turn:.0f}"]
        for plant in ("fast", "cal"):
            for plate in (0.0, 25.0):
                cells = [x for x in d["plate"] if x["hand"] == h and x["plant"] == plant and x["plate_mm"] == plate]
                held = [x for x in cells if x["held"]]
                lost = sum(1 for x in cells if x["lost_at_grip"])
                degs = ", ".join(f"{x['from_vertical_deg']:.0f}" for x in held) or "&#8211;"
                r.append(degs)
                r.append(cellc(len(held), len(cells), f" ({lost} lost at grip)" if lost else ""))
        rows.append(r)
    return table(["hand", "bench hold", "#bench, from vertical (deg)",
                  "fast, plate 0: deg", "held", "fast, plate +25: deg", "held",
                  "cal, plate 0: deg", "held", "cal, plate +25: deg", "held"], rows)


def table_angle(d):
    rows = []
    order = ["rv05_manual_b85", "rv05_manual_a60", "rv05_manual_a75", "rv05_manual_a90", "rv05_manual_a110"]
    for tag in order:
        rs = [x for x in d["angle"] if x["plan"] == tag]
        if not rs:
            continue
        r = ["deployed plan (b85)" if tag.endswith("b85") else f"re-export, {abs(rs[0]['angle_cmd']):.0f}&#176;",
             f"{abs(rs[0]['angle_cmd']):.0f}"]
        for plant in ("fast", "cal"):
            xs = [x for x in rs if x["plant"] == plant]
            r.append(", ".join(f"{x['from_vertical_deg']:.0f}" for x in xs))
            r.append(", ".join(f"{x['f_pad_end']:.2f}" for x in xs))
            r.append(f"{sum(1 for x in xs if not x['dropped'])}/{len(xs)}")
        rows.append(r)
    return table(["plan", "#commanded turn (deg)", "fast: from vertical (deg)", "pad N", "held",
                  "cal: from vertical (deg)", "pad N", "held"], rows)


def table_gait(d):
    rows = []
    for r in d["gait"]:
        for t in r["tips"]:
            rows.append([r["plant"], f"tip {t['tip']}", f"{np.degrees(np.arccos(np.clip(t['cos_rest'], -1, 1))):.0f}",
                         f"{t['pads_rest']}", f"{t['f_pad_rest']['thumb']:.2f} / {t['f_pad_rest']['index']:.2f} / {t['f_pad_rest']['middle']:.2f}",
                         f"{np.degrees(np.arccos(np.clip(t['cos_regrip'], -1, 1))):.0f}", f"{t['pads_regrip']}",
                         f"{t['closure_deg']['thumb']:+.0f} / {t['closure_deg']['index']:+.0f} / {t['closure_deg']['middle']:+.0f}",
                         ("carried", "ok") if t["carried"] else ("lost", "no")])
    return table(["plant", "step", "#rest: from vertical (deg)", "#pads &#8805; 0.24 N", "pad N th / ix / md",
                  "#regrip: from vertical (deg)", "#pads", "regrip closure th / ix / md (deg)", "tool"], rows)


def main():
    d = json.load(open(DATA))
    tpl = open(TPL).read()
    d7_cal_pl0 = [x for x in d["plate"] if x["hand"] == "D7" and x["plant"] == "cal" and x["plate_mm"] == 0 and x["held"]]
    d7_fast_pl25 = [x for x in d["plate"] if x["hand"] == "D7" and x["plant"] == "fast" and x["plate_mm"] == 25 and x["held"]]
    sub = {
        "N_PLATE": str(d["n"]["plate"]), "N_ANGLE": str(d["n"]["angle"]), "N_GAIT": str(d["n"]["gait"]),
        "D7_CAL_PL0": ", ".join(f"{x['from_vertical_deg']:.0f}" for x in d7_cal_pl0),
        "D7_FAST_PL25": ", ".join(f"{x['from_vertical_deg']:.0f}" for x in d7_fast_pl25),
        "TABLE_TRACE": table_trace(d), "TABLE_PLATE": table_plate(d), "TABLE_ANGLE": table_angle(d),
        "TABLE_GAIT": table_gait(d),
        "I_D7_STRIP": uri("D7_rv05_manual_b85__cal__s0_strip.png"),
        "I_PLATE_STRIP": uri("D7_plate_strip.png"),
        "V_D7_PL0": uri("videos/rv05_manual_b85__fast_pl0__s0.mp4"),
        "V_D7_PL25": uri("videos/rv05_manual_b85__fast_pl25__s0.mp4"),
        "V_D7_GAIT": uri("videos/D7_rv05_manual_b85__cal__s0.mp4"),
    }
    for k, v in sub.items():
        tpl = tpl.replace("{{" + k + "}}", v)
    left = [ln for ln in tpl.splitlines() if "{{" in ln]
    assert not left, left[:3]
    open(OUT, "w").write(tpl)
    print(f"wrote {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
