#!/usr/bin/env python3
"""Build the control-diagnosis page from the ablation and sweep JSON.

    uv run python scripts/real_v1_control_diagnosis_page.py
"""
from __future__ import annotations

import base64, json, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "docs/experiments"
OUT = E / "20260906-control_diagnosis/20260906-control_diagnosis.html"
TPL = Path(__file__).with_name("real_v1_control_diagnosis_page.template.html")
DID = {"sv1_w6689_b060": "D1", "sv1_w2360_b075": "D2", "sv1_u1364_b080": "D3",
       "g12_b095": "D4", "sv1_u0060_b75": "D5", "sv1_u0308_b050": "D6",
       "rv05_manual_b85": "D7", "sv1_w0099_b100": "D8"}
ORDER = list(DID)
CHAIN_MM = 68.11
# scripts/.../ext.py, closed grip, mount-to-pad distance per finger.
EXT = {"CEM (the reference)": (62.6, 63.7, 65.4),
       "fitter, squeeze 2 mm": (67.1, 67.7, 66.3),
       "fitter, squeeze 4 mm": (66.9, 67.6, 66.1),
       "fitter, squeeze 6 mm": (66.8, 67.4, 65.9),
       "fitter, squeeze 8 mm": (66.6, 67.3, 65.6),
       "fitter, squeeze 10 mm": (66.3, 67.1, 65.4),
       "fitter, squeeze 12 mm": (66.0, 66.8, 65.1)}


def table(head, rows, nums, hi=frozenset()):
    th = "".join(f'<th{" class=num" if i in nums else ""}>{h}</th>' for i, h in enumerate(head))
    body = []
    for k, r in enumerate(rows):
        td = "".join(f'<td{" class=num" if i in nums else ""}>{c}</td>' for i, c in enumerate(r))
        body.append(f'<tr{" class=hi" if k in hi else ""}>{td}</tr>')
    return ('<div class="tw"><table><thead><tr>' + th + "</tr></thead><tbody>"
            + "".join(body) + "</tbody></table></div>")


def data_uri(p: Path, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()


def rows_of(p: Path):
    return json.loads(p.read_text())["rows"]


def seam(g, phase, field):
    v = []
    for r in g:
        s = {x["phase"]: x for x in r.get("seams", [])}.get(phase)
        if s and s.get(field) is not None:
            v.append(s[field])
    return st.mean(v) if v else None


def fig_ext() -> str:
    """Finger extension at the closed grip against the 68.11 mm chain."""
    W, H, L, R = 1020, 268, 210, 120
    x0, x1 = L, W - R
    lo, hi = 60.0, 68.11
    sx = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)
    p = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" '
         f'aria-label="finger extension at the closed grip">']
    for v in (60, 62, 64, 66, 68):
        p.append(f'<line class="grid" x1="{sx(v):.1f}" y1="34" x2="{sx(v):.1f}" y2="{H-40}"/>')
        p.append(f'<text class="tick" x="{sx(v):.1f}" y="{H-24}" text-anchor="middle">'
                 f'{v}</text>')
    p.append(f'<line x1="{sx(CHAIN_MM):.1f}" y1="28" x2="{sx(CHAIN_MM):.1f}" y2="{H-40}" '
             f'stroke="var(--bad)" stroke-width="1.5"/>')
    p.append(f'<text class="mark" x="{sx(CHAIN_MM):.1f}" y="20" text-anchor="middle" '
             f'fill="var(--bad)">REACH SHELL 68.11</text>')
    y = 46
    for name, ext in EXT.items():
        ref = name.startswith("CEM")
        col = "var(--s2)" if ref else "var(--s1)"
        p.append(f'<text class="ser" x="{L-12}" y="{y+4}" text-anchor="end" '
                 f'fill="var(--ink2)">{name}</text>')
        p.append(f'<line x1="{sx(min(ext)):.1f}" y1="{y}" x2="{sx(max(ext)):.1f}" y2="{y}" '
                 f'stroke="{col}" stroke-width="2" stroke-linecap="round"/>')
        for v in ext:
            p.append(f'<circle cx="{sx(v):.1f}" cy="{y}" r="4.5" fill="{col}"/>')
        p.append(f'<text class="val" x="{x1+10}" y="{y+4}" fill="{col}">'
                 f'{CHAIN_MM - max(ext):.2f} mm left</text>')
        y += 28
    p.append(f'<text class="axlab" x="{(x0+x1)/2:.0f}" y="{H-6}" text-anchor="middle">'
             f'MOUNT-TO-PAD DISTANCE AT THE CLOSED GRIP / mm</text>')
    p.append("</svg>")
    return "".join(p)


def main() -> int:
    V = {}
    ab = rows_of(E / "20260906-ablate/ablate.json")
    arms = ["baseline", "axis_k=0.05", "clip=0.85", "angle=-60", "tips=box", "grasp=fit",
            "all=deployed"]
    LABEL = {"baseline": "the 2026-09-03 reference, reproduced",
             "axis_k=0.05": "pivot 0.25 &#8594; 0.05 (the plans')",
             "clip=0.85": "clip 0.50 &#8594; 0.85 (the plan's)",
             "angle=-60": "turn &minus;90&#176; &#8594; &minus;60&#176; (the plan's)",
             "tips=box": "sphere pads &#8594; the chain's flat pads",
             "grasp=fit": "CEM grasp &#8594; the bench fitter",
             "all=deployed": "all four together"}

    def cell(g, ph, f, fmt="{:+.3f}"):
        v = seam(g, ph, f)
        return fmt.format(v) if v is not None else "&#8212;"

    V["ABLATE_TBL"] = table(
        ["one factor changed", "lift: pads / N", "turned: cos", "reoriented: cos",
         "reoriented: pads / N", "chain"],
        [[LABEL[a],
          f'{cell(g, "lifted", "pad_contacts", "{:.1f}")} / '
          f'{cell(g, "lifted", "pad_force_N", "{:.2f}")}',
          f'<span class="{"no" if (seam(g, "turned", "cos") or 0) < -0.5 else ""}">'
          f'{cell(g, "turned", "cos")}</span>',
          cell(g, "reoriented", "cos"),
          f'{cell(g, "reoriented", "pad_contacts", "{:.1f}")} / '
          f'{cell(g, "reoriented", "pad_force_N", "{:.2f}")}',
          f'{sum(1 for r in g if r.get("ok"))}/{len(g)}']
         for a in arms for g in [[r for r in ab if r.get("arm") == a]] if g],
        nums={1, 2, 3, 4, 5}, hi={1})

    V["EXT_FIG"] = fig_ext()
    V["EXT_TBL"] = table(
        ["grasp", "thumb", "index", "middle", "% of the chain", "headroom"],
        [[k] + [f"{v:.1f} mm" for v in ext]
         + [f"{max(ext)/CHAIN_MM*100:.1f}%", f"{CHAIN_MM - max(ext):.2f} mm"]
         for k, ext in EXT.items()],
        nums={1, 2, 3, 4, 5}, hi={0})

    sq = rows_of(E / "20260906-fitter_force/ablate.json")
    V["SQ_TBL"] = table(
        ["fitted grasp", "lift: cos / z", "lift: pads / N", "reoriented: cos", "chain"],
        [[a.replace("fit_sq", "squeeze ").replace("mm", " mm").replace("baseline",
                                                                      "the reference"),
          f'{cell(g, "lifted", "cos", "{:+.2f}")} / {cell(g, "lifted", "z", "{:.3f}")}',
          f'{cell(g, "lifted", "pad_contacts", "{:.1f}")} / '
          f'{cell(g, "lifted", "pad_force_N", "{:.2f}")}',
          cell(g, "reoriented", "cos"),
          f'{sum(1 for r in g if r.get("ok"))}/{len(g)}']
         for a in ["baseline"] + [f"fit_sq{v}mm" for v in (2, 4, 6, 8, 10, 12)]
         for g in [[r for r in sq if r.get("arm") == a]] if g],
        nums={1, 2, 3, 4}, hi={0})

    op = json.loads((E / "OPERATING_POINTS.json").read_text())
    V["OP_TBL"] = table(
        ["hand", "id", "plan clip", "bench band", "plan inside it?", "carry", "chain"],
        [[f'<code>{t}</code>', op["hands"][t]["id"],
          f'{op["hands"][t]["plan"]["budget_rad"]:.2f}',
          str(op["hands"][t]["bench"]["clip_band_rad"] or "no data"),
          ('<span class="ok">yes</span>' if op["hands"][t]["bench"]["clip_at_plan_holds"]
           else '<span class="no">no</span>'),
          "unmeasured", "unmeasured"] for t in ORDER],
        nums={2, 3, 4, 5, 6})

    band = rows_of(E / "20260906-chain_band/chain_hands.json")
    held = lambda s: s and (s.get("pad_contacts") or 0) >= 2 and (s.get("pad_force_N") or 0) >= 0.24
    V["BAND_N"] = str(len(band))
    V["BAND_HELD"] = str(sum(1 for r in band
                             if held({x["phase"]: x for x in r.get("seams", [])}
                                     .get("reoriented"))))
    V["BAND_OK"] = str(sum(1 for r in band if r.get("ok")))
    best = []
    for t in ORDER:
        for a in sorted({r["arm"] for r in band if r["tag"] == t}):
            g = [r for r in band if r["tag"] == t and r["arm"] == a]
            c = seam(g, "reoriented", "cos")
            if c is not None:
                best.append((c, DID[t], a, seam(g, "reoriented", "pad_contacts"),
                             seam(g, "reoriented", "pad_force_N")))
    best.sort(key=lambda x: -x[0])
    V["BAND_TBL"] = table(
        ["id", "pivot / clip", "reoriented: cos", "pads", "force"],
        [[i, a.replace("load0_t550_b", "clip ").replace("_k", " / pivot ").replace("_a-90", ""),
          f"{c:+.3f}", f"{p:.1f}", f'<span class="{"ok" if F >= 0.24 else "no"}">{F:.2f} N</span>']
         for c, i, a, p, F in best[:8]],
        nums={2, 3, 4})

    # ---- the turn on the deployed hands: pivot at the clip the band sweep stepped over ----
    piv = rows_of(E / "20260906-pivot/chain_hands.json")
    V["PIVOT_N"] = str(len(piv))
    rows = []
    for t in ORDER:
        g = [r for r in piv if r["tag"] == t]
        best, key = None, None
        for r in g:
            s_ = {x["phase"]: x for x in r.get("seams", [])}.get("reoriented", {})
            k = ((s_.get("pad_contacts") or 0) >= 2 and (s_.get("pad_force_N") or 0) >= 0.24
                 and (s_.get("z") or 0) > 0.08, s_.get("cos") or -1.0)
            if key is None or k > key:
                best, key = r, k
        s_ = {x["phase"]: x for x in best.get("seams", [])}.get("reoriented", {})
        rows.append((DID[t], t, best.get("axis_k"), s_.get("cos") or 0.0,
                     s_.get("pad_contacts") or 0, s_.get("pad_force_N") or 0.0,
                     s_.get("z") or 0.0, best.get("drop_stage"), key[0]))
    rows.sort(key=lambda r: (-r[8], -r[3]))
    V["PIVOT_HELD"] = str(sum(1 for r in rows if r[8]))
    V["PIVOT_TBL"] = table(
        ["id", "hand", "pivot", "reoriented: cos", "pads", "force", "z", "lost at"],
        [[i, f"<code>{t}</code>", f"{ak:.2f}", f"{c:+.3f}", f"{p_:d}",
          f'<span class="{"ok" if F >= 0.24 and p_ >= 2 else "no"}">{F:.2f} N</span>',
          f"{z*1000:.0f} mm", f"<code>{d}</code>" if d else "&#8212;"]
         for i, t, ak, c, p_, F, z, d, _ in rows],
        nums={2, 3, 4, 5, 6}, hi={0, 1, 2})

    F = E / "20260906-held_turn_films"
    V["FILM_GRID"] = data_uri(F / "20260906-chain_grid.mp4", "video/mp4")
    V["FILM_D5"] = data_uri(
        F / "20260906-sv1_u0060_b75_angle_deg-90_axis_k0.15_budget0.5_turn_steps550_s2_seams.png",
        "image/png")
    V["FILM_D3"] = data_uri(
        F / "20260906-sv1_u1364_b080_angle_deg-90_axis_k0.35_budget0.5_turn_steps550_s0_seams.png",
        "image/png")

    html = TPL.read_text()
    for k, v in V.items():
        html = html.replace("{{" + k + "}}", v)
    assert "{{" not in html, [s[:36] for s in html.split("{{")[1:]]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html)
    print(f"-> {OUT}  ({OUT.stat().st_size/1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
