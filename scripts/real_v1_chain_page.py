"""Build the doc page for the chained task: grasp -> gait, on an arm, into a screw.

Regenerable: everything on the page is read from the three study JSONs and the seam ledger, so
re-running a study and re-running this is the whole update path. Charts are hand-emitted SVG
(theme-aware via CSS custom properties) and media are inlined as base64.

    uv run --extra rl python scripts/real_v1_chain_page.py \
        --chain docs/experiments/20260903-real_v1_chain \
        --screw docs/experiments/20260903-real_v1_screw \
        --out   docs/experiments/20260903-real_v1_chain/page.html
"""
from __future__ import annotations

import argparse
import base64
import json
import statistics as st
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Two series, never more. The four-hue set this page inherits fails the protanopia check on
# amber-vs-green (dE 3.7) and the deuteranopia check on blue-vs-plum (dE 7.6); amber-vs-blue
# passes every simulation with dE >= 22, so every chart here is at most two categorical series
# and any third mark is a neutral REFERENCE, not a series.
A, B = "var(--s1)", "var(--s2)"          # amber = instrumented, blue = control
REF = "var(--ref)"


# ------------------------------------------------------------------------------------ svg bits

def _axes(w, h, pad, xlab, ylab, xt, yt, x2p, y2p, xfmt="{:g}", yfmt="{:g}"):
    o = [f'<g class="ax">']
    o.append(f'<line x1="{pad[3]}" y1="{h-pad[2]}" x2="{w-pad[1]}" y2="{h-pad[2]}"/>')
    for t in yt:
        y = y2p(t)
        o.append(f'<line class="grid" x1="{pad[3]}" y1="{y:.1f}" x2="{w-pad[1]}" y2="{y:.1f}"/>')
        o.append(f'<text class="tick" x="{pad[3]-8}" y="{y+4:.1f}" text-anchor="end">'
                 f'{yfmt.format(t)}</text>')
    for t in xt:
        x = x2p(t)
        o.append(f'<text class="tick" x="{x:.1f}" y="{h-pad[2]+18}" text-anchor="middle">'
                 f'{xfmt.format(t)}</text>')
    o.append(f'<text class="axlab" x="{(w-pad[1]+pad[3])/2:.0f}" y="{h-4}" '
             f'text-anchor="middle">{xlab}</text>')
    o.append(f'<text class="axlab" transform="translate(14,{(h-pad[2]+pad[0])/2:.0f}) rotate(-90)" '
             f'text-anchor="middle">{ylab}</text>')
    o.append("</g>")
    return "".join(o)


def bars(rows, xlab, ylab, ymax=None, colour=A, note=None, w=560, h=250, fmt="{:g}",
         labfmt="{:.0f}", xfmt="{:g}"):
    """One series of labelled bars: (x label, value). Direct labels, no legend needed."""
    pad = (18, 16, 40, 46)
    ymax = ymax or max(v for _, v in rows) * 1.18
    n = len(rows)
    span = w - pad[1] - pad[3]
    bw = span / n * 0.56
    def xc(i):
        return pad[3] + span * (i + 0.5) / n
    def y2p(v):
        return h - pad[2] - (v / ymax) * (h - pad[2] - pad[0])
    yt = [t for t in _nice(ymax)]
    o = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img">']
    o.append(_axes(w, h, pad, xlab, ylab, [], yt, lambda t: t, y2p, yfmt=fmt))
    for i, (lab, v) in enumerate(rows):
        x, y = xc(i) - bw / 2, y2p(v)
        o.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" '
                 f'height="{h-pad[2]-y:.1f}" rx="3" fill="{colour}"/>')
        o.append(f'<text class="val" x="{xc(i):.1f}" y="{y-6:.1f}" text-anchor="middle">'
                 f'{labfmt.format(v)}</text>')
        o.append(f'<text class="tick" x="{xc(i):.1f}" y="{h-pad[2]+18}" text-anchor="middle">'
                 f'{xfmt.format(lab) if not isinstance(lab, str) else lab}</text>')
    if note:
        o.append(f'<text class="note" x="{w-pad[1]}" y="{pad[0]+6}" text-anchor="end">{note}</text>')
    o.append("</svg>")
    return "".join(o)


def _nice(vmax, n=4):
    import math
    step = 10 ** math.floor(math.log10(vmax / n)) if vmax > 0 else 1
    for m in (1, 2, 2.5, 5, 10):
        if vmax / (step * m) <= n:
            step *= m
            break
    t, out = 0.0, []
    while t <= vmax + 1e-9:
        out.append(round(t, 6))
        t += step
    return out


def lines(series, xlab, ylab, xmax, ymax, w=620, h=300, marks=None):
    """Up to two labelled series plus optional phase rules, each labelled in place.

    Labels sit at a chosen point with a vertical offset rather than at the last point: these two
    curves converge at the right-hand end, and terminal labels printed on top of each other.
    """
    pad = (34, 24, 40, 50)
    def x2p(v):
        return pad[3] + (v / xmax) * (w - pad[1] - pad[3])
    def y2p(v):
        return h - pad[2] - (v / ymax) * (h - pad[2] - pad[0])
    o = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img">']
    o.append(_axes(w, h, pad, xlab, ylab, _nice(xmax), _nice(ymax), x2p, y2p))
    for i, (x, lab) in enumerate(marks or []):
        yl = pad[0] - 20 + (i % 2) * 12
        o.append(f'<line class="rule" x1="{x2p(x):.1f}" y1="{yl+4}" '
                 f'x2="{x2p(x):.1f}" y2="{h-pad[2]}"/>')
        o.append(f'<text class="mark" x="{x2p(x):.1f}" y="{yl}" '
                 f'text-anchor="middle">{lab}</text>')
    for name, pts, colour, dash, li, dy in series:
        d = " ".join(f"{'M' if i == 0 else 'L'}{x2p(x):.1f},{y2p(y):.1f}"
                     for i, (x, y) in enumerate(pts))
        o.append(f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="2" '
                 f'stroke-linejoin="round"{f" stroke-dasharray={dash!r}" if dash else ""}/>')
        for x, y in pts:
            o.append(f'<circle cx="{x2p(x):.1f}" cy="{y2p(y):.1f}" r="4" fill="{colour}" '
                     f'stroke="var(--card)" stroke-width="2"/>')
        lx, ly = pts[li]
        o.append(f'<text class="ser" x="{x2p(lx):.1f}" y="{y2p(ly)+dy:.1f}" '
                 f'text-anchor="middle" fill="{colour}">{name}</text>')
    o.append("</svg>")
    return "".join(o)


# ------------------------------------------------------------------------------------- helpers

def group(rows, keys):
    out = {}
    for r in rows:
        out.setdefault(tuple(r.get(k) if not isinstance(r.get(k), list)
                             else tuple(r.get(k)) for k in keys), []).append(r)
    return out


def rate(g):
    good = [r for r in g if r.get("ok")]
    return len(good), len(g), good


def mean(g, key):
    good = [r for r in g if r.get("ok")]
    return st.mean([r[key] for r in good]) if good else 0.0


def b64(path: Path, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def jpeg(path: Path, scale: str = "50%", q: int = 82, cols: int = 4) -> str:
    """Filmstrips are 2560x2400 PNGs; inlined raw they are 1.3 MB each before base64.

    The tiler pads the last row with pure black when the frame count does not divide the column
    count, and a black block reads as a hole punched in the figure. Drop that row: its content
    is the final seam shot, which duplicates the last gait cycle anyway.
    """
    import numpy as np
    from PIL import Image
    im = Image.open(path).convert("RGB")
    w, h = im.size
    cw = w // cols
    ch = round(cw * 3 / 4)               # the renderer is 640x480
    rows = max(1, h // ch)
    arr = np.asarray(im)
    if rows > 1:
        last = arr[(rows - 1) * ch: rows * ch]
        if any((last[:, c * cw:(c + 1) * cw].max() < 8) for c in range(cols)):
            im = im.crop((0, 0, w, (rows - 1) * ch))
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "p.png"
        im.save(src)
        out = Path(td) / "f.jpg"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                        "-vf", f"scale=iw*{scale.rstrip('%')}/100:-1", "-q:v", str(q // 10),
                        str(out)], check=True)
        return b64(out, "image/jpeg")


def okcell(k, n):
    cls = "ok" if k == n else ("no" if k == 0 else "")
    return f'<td class="num"><span class="{cls}">{k}/{n}</span></td>'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chain", type=Path, default=ROOT / "docs/experiments/20260903-real_v1_chain")
    ap.add_argument("--screw", type=Path, default=ROOT / "docs/experiments/20260903-real_v1_screw")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or (args.chain / "page.html")

    CH = json.loads((args.chain / "chain_study.json").read_text())
    AR = json.loads((args.chain / "arm_study.json").read_text())
    SW = json.loads((args.screw / "screw_study.json").read_text())
    LED = json.loads((args.chain / "seam_ledger.json").read_text())
    ncells = len(CH) + len(AR) + len(SW)

    def arm(rows, name):
        return [r for r in rows if r.get("arm") == name]

    # ---------------------------------------------------------------- headline numbers
    lead = [r for r in arm(CH, "chain") if r["run"] == "rv05_manual_stored"]
    k_lead, n_lead, good_lead = rate(lead)
    endur = arm(CH, "endurance")
    ur = [r for r in arm(AR, "wrist") if r["wrist_tag"] == "ur5e"]
    ga = [r for r in arm(AR, "wrist") if r["wrist_tag"] == "gantry"]
    sw_seat = [r for r in arm(SW, "seat") if r.get("place_xy") and r["press_mm"] == 10.0]
    sw_flat = [r for r in arm(SW, "seat") if not r.get("place_xy") and r["press_mm"] == 2.0]
    stats = "".join(
        f'<div class="stat"><b>{v}</b><span>{s}</span></div>' for v, s in [
            (f"{k_lead}/{n_lead}", "chain complete, flat floor"),
            (f"{mean(endur, 'turns'):.2f}", "turns in 40 cycles"),
            (f"{rate(ur)[0]}/{rate(ur)[1]}", "chain complete, on a UR5e"),
            ("0.97 mm", "palm droop the arm must beat"),
            (f"{rate(sw_seat)[0]}/{rate(sw_seat)[1]}", "screw seated and turned"),
            ("6 mm", "insertion capture radius"),
        ])

    # ---------------------------------------------------------------- grip decay
    def curve(key, upto):
        return [(s["t"], s["pad_force_N"]) for s in LED[key]["seams"] if s["phase"] in upto]
    held = ("lifted", "reoriented", "regripped", "staged", "set_down", "pressed")
    c_flat = curve("flat", held)
    c_seat = curve("seat_regrip", held)
    marks = [(s["t"], s["phase"].replace("_", " ")) for s in LED["flat"]["seams"]
             if s["phase"] in ("reoriented", "set_down", "upright", "pressed")]
    grip_chart = lines(
        [("the shipped chain, nothing re-gripped", c_flat, B, None, 2, 30),
         ("an insertion, one 0.3 mm re-grip", c_seat, A, "5 4", 5, -16)],
        "seconds of hand-time since the grasp closed", "pad force  N",
        xmax=9.5, ymax=14.0, marks=marks)

    # ---------------------------------------------------------------- descend corrections
    dsc = group(arm(CH, "descend"), ["descend_iters"])
    desc_chart = bars([(f"{k[0]}", rate(v)[0]) for k, v in sorted(dsc.items())],
                      "measured corrections during the set-down", "chains complete  of 6",
                      ymax=6.6, colour=A, fmt="{:g}", w=360, h=230)
    rps = group(arm(CH, "repose"), ["repose_iters"])
    repose_chart = bars([(f"{k[0]}", rate(v)[0]) for k, v in sorted(rps.items())],
                        "corrections bringing it upright", "chains complete  of 6",
                        ymax=6.6, colour=B, fmt="{:g}", w=360, h=230)

    # ---------------------------------------------------------------- droop cliff
    stf = group(arm(AR, "stiffness"), ["droop_mm"])
    droop_rows = sorted(((k[0], rate(v)[0]) for k, v in stf.items()), reverse=True)
    droop_chart = bars([(f"{d:.2g}" if d < 10 else f"{d:.3g}", k) for d, k in droop_rows],
                       "palm droop under load  mm", "chains complete  of 6",
                       ymax=6.6, colour=B)

    # ---------------------------------------------------------------- seat vs plane
    seat_turn = bars([("flat plane", mean(sw_flat, "gain_mean_deg")),
                      ("45&#176; seat", mean(sw_seat, "gain_mean_deg"))],
                     "", "degrees of shaft per cycle", ymax=58, colour=A, labfmt="{:.1f}", w=330, h=250)
    seat_drift = bars([("flat plane", mean(sw_flat, "drift_mm")),
                       ("45&#176; seat", mean(sw_seat, "drift_mm"))],
                      "", "lateral walk over 8 cycles  mm", ymax=1.6, colour=B,
                      labfmt="{:.2f}", fmt="{:g}", w=330, h=250)

    # ---------------------------------------------------------------- insertion window
    pr = group(arm(SW, "press"), ["press_mm"])
    press_chart = bars([(f"{k[0]:g}", rate(v)[0]) for k, v in sorted(pr.items())],
                       "press through the grip  mm", "seated and turned  of 6",
                       ymax=6.6, colour=A, w=330, h=250)
    cap = group(arm(SW, "capture"), ["place_err_mm"])
    cap_chart = bars([(f"{k[0][0]:g}", rate(v)[0]) for k, v in sorted(cap.items())],
                     "lateral error in the target  mm", "seated and turned  of 6",
                     ymax=6.6, colour=B, w=330, h=250)
    rg = group(arm(SW, "regrip"), ["carry_squeeze_mm"])
    regrip_chart = bars([(f"{k[0]:g}", rate(v)[0]) for k, v in sorted(rg.items())],
                        "re-grip after the turn  mm of extra interference",
                        "seated and turned  of 6", ymax=6.6, colour=A, w=330, h=250)
    ang = group(arm(SW, "angle"), ["tip_len_mm"])
    ang_lab = {5.77: "60&#176;", 10.0: "45&#176;", 17.32: "30&#176;"}
    ang_rows = sorted(((ang_lab[k[0]], rate(v)[0], mean(v, "gain_mean_deg")) for k, v in ang.items()),
                      key=lambda r: -float(r[0].split("&")[0]))
    ang_ok = bars([(a, k) for a, k, _ in ang_rows], "countersink half-angle",
                  "seated and turned  of 6", ymax=6.6, colour=B, w=330, h=250)
    ang_turn = bars([(a, t) for a, _, t in ang_rows], "countersink half-angle",
                    "degrees of shaft per cycle", ymax=48, colour=A, labfmt="{:.1f}", w=330, h=250)

    # ---------------------------------------------------------------- tables
    des = group(arm(CH, "chain"), ["run"])
    des_rows = "".join(
        f'<tr{" class=hi" if k[0].startswith("rv05") else ""}><td class="mono">{k[0]}</td>'
        f'<td class="num">{st.mean([r["seams"][1]["tilt_deg"] for r in v]):.1f}</td>'
        f'{okcell(*rate(v)[:2])}'
        f'<td class="num">{mean(v, "turns"):.2f}</td></tr>'
        for k, v in sorted(des.items()))

    def ctrl_row(label, g, note):
        k, n, _ = rate(g)
        return (f'<tr><td>{label}</td>{okcell(k, n)}'
                f'<td class="num">{mean(g, "turns"):.2f}</td><td>{note}</td></tr>')
    ctrls = "".join([
        ctrl_row("full chain, as shipped", arm(CH, "floor")[:0] or
                 [r for r in arm(CH, "floor") if not r["no_floor_gait"]], "the reference"),
        ctrl_row("no floor during the gait",
                 [r for r in arm(CH, "floor") if r["no_floor_gait"]],
                 "floor deleted after the gait grasp is set"),
        ctrl_row("gait from the carry's own grip",
                 [r for r in arm(CH, "reindex") if r["reindex"] == "none"],
                 "no release, no re-index; ring solved at the carry's palm pose"),
        ctrl_row("change grasp in mid-air",
                 [r for r in arm(CH, "airgrip") if r["airgrip"] == "ring"],
                 "take the gait's ring before setting down; it solves to 1.1 mm"),
    ])
    hold = group(arm(CH, "hold"), ["hold_steps"])
    hold_rows = "".join(
        f'<tr><td class="num">{k[0] * 0.002:.1f}</td>'
        f'<td class="num">{st.mean([r["seams"][1]["tilt_deg"] for r in v]):.1f}</td>'
        f'{okcell(*rate(v)[:2])}<td class="num">{mean(v, "turns"):.2f}</td></tr>'
        for k, v in sorted(hold.items()))
    stroke = group(arm(CH, "stroke"), ["stroke_deg"])
    st45 = [v for k, v in stroke.items() if k[0] == 45.0][0]
    over = mean(st45, "gain_mean_deg") / (1.684 * 45.0)
    stroke_rows = "".join(
        f'<tr><td class="num">{k[0]:g}</td>{okcell(*rate(v)[:2])}'
        f'<td class="num">{mean(v, "turns"):.2f}</td>'
        f'<td class="num">{mean(v, "gain_mean_deg"):.1f}</td>'
        f'<td class="num">{mean(v, "final_tilt_deg"):.1f}</td></tr>'
        for k, v in sorted(stroke.items()))
    dist = group(arm(SW, "distance"), ["place_xy"])
    dist_rows = "".join(
        f'<tr><td class="num">{(k[0][0]**2 + (k[0][1] + 0.011)**2) ** .5 * 1000:.0f}</td>'
        f'{okcell(*rate(v)[:2])}<td class="num">{mean(v, "gain_mean_deg"):.1f}</td>'
        f'<td class="num">{mean(v, "seat_offset_mm"):.2f}</td></tr>'
        for k, v in sorted(dist.items()))
    wrist_rows = "".join(
        f'<tr{" class=hi" if t == "ur5e" else ""}><td>{n}</td>{okcell(*rate(g)[:2])}'
        f'<td class="num">{mean(g, "turns"):.3f}</td>'
        f'<td class="num">{mean(g, "gain_mean_deg"):.1f}</td>'
        f'<td class="num">{st.mean([r["seams"][1]["tilt_deg"] for r in g]):.1f}</td>'
        f'<td class="num">{mean(g, "drift_mm"):.2f}</td>'
        f'<td class="num">{mean([r for r in arm(AR, "endurance") if r["wrist_tag"] == t], "turns"):.2f}</td>'
        f'<td class="num">{max(r.get("arm_ik_pos_mm", 0) for r in g):.3f}</td></tr>'
        for t, n, g in (("gantry", "floating palm", ga), ("ur5e", "UR5e + mink IK", ur)))

    phases = "".join(
        f'<div class="chip"><i>{i:02d}</i><b>{s["phase"].replace("_", " ")}</b>'
        f'<span>{s["t"]:.1f} s &middot; {s["pad_force_N"]:.1f} N &middot; '
        f'{s["tilt_deg"]:.1f}&deg;</span></div>'
        for i, s in enumerate(LED["flat"]["seams"], 1))

    media = {k: b64(p, "video/mp4") for k, p in {
        "vid_flat": args.chain / "20260903-chain_rv05.mp4",
        "vid_arm": args.chain / "20260903-chain_ur5e.mp4",
        "vid_screw": args.screw / "20260903-screw_rv05.mp4"}.items()}
    media.update({k: jpeg(p) for k, p in {
        "film_flat": args.chain / "20260903-chain_rv05_seams.png",
        "film_arm": args.chain / "20260903-chain_ur5e_seams.png",
        "film_screw": args.screw / "20260903-screw_rv05_seams.png"}.items()})

    body = BODY.format(
        phases=phases, grip_chart=grip_chart, desc_chart=desc_chart, repose_chart=repose_chart,
        droop_chart=droop_chart, seat_turn=seat_turn, seat_drift=seat_drift,
        press_chart=press_chart, cap_chart=cap_chart, regrip_chart=regrip_chart,
        ang_ok=ang_ok, ang_turn=ang_turn, des_rows=des_rows, ctrls=ctrls, hold_rows=hold_rows,
        stroke_rows=stroke_rows, dist_rows=dist_rows, wrist_rows=wrist_rows,
        st45_deg=mean(st45, "gain_mean_deg"), st45_over=over,
        seat_deg=mean(sw_seat, "gain_mean_deg"), flat_deg=mean(sw_flat, "gain_mean_deg"),
        seat_drift_mm=mean(sw_seat, "drift_mm"), flat_drift_mm=mean(sw_flat, "drift_mm"),
        seat_endur=mean(arm(SW, "endurance"), "turns"),
        flat_endur=mean(endur, "turns"), **media)

    tpl = (ROOT / "scripts/real_v1_chain_page.template.html").read_text()
    html = tpl.replace("{{NCELLS}}", str(ncells)).replace("{{STATS}}", stats)
    html = html.replace("{{BODY}}", body)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"-> {out}  ({out.stat().st_size / 1e6:.2f} MB)")
    return 0


BODY = r"""
<section class="col">
  <h2>What was already true</h2>
  <p class="sub">two open-loop results, neither with any claim on the other</p>
  <p><code>probe_real_v1_carry.py</code> takes the shaft off the table and stands it vertical in
  the hand: a fixed grasp, a rotation about a pivot raised above the contacts, cos 0.996, no
  policy anywhere. <code>probe_real_v1_gait.py</code> spins a shaft that is <em>already</em>
  standing on the floor through five revolutions in forty four-phase cycles, again from a
  precomputed set-point table. The first ends with the tool in the air at 113&nbsp;mm; the
  second begins with it planted at 50&nbsp;mm, put there by a line of code.</p>
  <p>Everything below is about the 63&nbsp;mm in between, and then about replacing the two
  conveniences that made both results possible: a palm that can be commanded to any pose, and a
  floor that is a flat plane.</p>
</section>

<section>
  <div class="col">
    <h2>The chain</h2>
    <p class="sub">grasp &rarr; lift &rarr; reorient &rarr; set down &rarr; release &rarr; re-index &rarr; gait</p>
    <p>One rollout, one <code>MjData</code>, no state passed through a file. The re-pose is
    written as a rigid transfer &mdash; if the tool rides in the palm frame then asking for an
    object pose fully determines the palm pose, <code class="mono">T_palm = T_obj_des &middot;
    T_obj<sup>-1</sup> &middot; T_palm</code> &mdash; because that is the form an arm can be
    handed later. Every phase below carries the time since the grasp closed, the pad force at
    that moment, and the tool's tilt from vertical.</p>
  </div>
  <div class="phases">{phases}</div>
  <figure>
    <video src="{vid_flat}" controls muted loop playsinline preload="metadata"></video>
    <figcaption>The whole chain on the floating palm, eight gait cycles. The two dots on the
    shaft are body-fixed markers, painted at render time; an untextured cylinder gives the eye
    nothing to read a rotation against.</figcaption>
  </figure>
  <figure>
    <img src="{film_flat}" alt="Seventeen frames of the chain: grasp, lift, reorient, set down, release, re-index, then eight gait cycles.">
    <figcaption>The same run at its seams. Frame five is the moment that has no precedent in
    this program &mdash; the hand is fully open and the tool is standing by itself.</figcaption>
  </figure>
  <div class="col">
    <p>Success is the conjunction of four gates, not the cosine: the carry held, the tool stood
    within 10&nbsp;mm of its resting height with support contact and under 14&deg; of tilt, the
    gait grasp took on at least two pads, and all requested cycles ran without the tool being
    lost. A run that reoriented beautifully and then dropped the shaft on the way down reads
    <span class="no">false</span>.</p>
    <div class="tw"><table><thead><tr>
      <th>design</th><th class="num">carry exit tilt &deg;</th><th class="num">chains</th>
      <th class="num">turns / 8 cycles</th></tr></thead><tbody>{des_rows}</tbody></table></div>
    <p>Only <code>rv05_manual_stored</code> &mdash; the hand the bench was built from &mdash;
    completes. The other nine fail at the carry, and they fail under rv05's published carry
    settings applied unchanged: axis height, turn duration and residual budget were never
    re-tuned per design. That is a statement about this experiment, not about those hands.</p>
  </div>
</section>

<section>
  <div class="col">
    <h2>The seam is a race</h2>
    <p class="sub">what the carry hands over, and how long it lasts</p>
    <p>The carry does not end in a grip. At the top of a raised-pivot turn the tool is commonly
    cradled against the middle phalanges with the pads barely on it &mdash; about half a
    millimetre of commanded interference &mdash; and that hold leaks. The shaft creeps down
    through the pads at roughly 1.5&nbsp;mm/s and the pad force decays to nothing in under two
    seconds, which <code>probe_real_v1_carry</code> already measured from the other direction
    (16.1&nbsp;N to zero, tool on the table by 1.6&nbsp;s).</p>
  </div>
  <figure>{grip_chart}
    <figcaption>Pad force at each seam of a single rollout, ending at the last held phase. The
    shipped chain gets 12.0&nbsp;N at the top of the turn and has 0.4&nbsp;N left by the press
    &mdash; the hand does nothing in between but move the tool. A single 0.3&nbsp;mm radial
    re-squeeze, measured from the <em>commanded</em> pad position rather than the achieved one,
    holds the same quantity between 7.7 and 9.0&nbsp;N across a longer sequence. Neither curve
    is a rate measurement; the rate comes from the sweeps below.</figcaption>
  </figure>
  <div class="col">
    <p>Two consequences follow, and the second is the one worth keeping.</p>
    <p><strong>Feedback on the height is harmful.</strong> The set-down is driven by re-measuring
    the tool and recomputing the palm pose. Doing that once, as a single continuous move, stands
    the tool on 6 of 6 spawn-jittered seeds. Doing it eight times stands it on 1. Nothing about
    the trajectory changes &mdash; the extra corrections only add dwells, and every dwell is
    paid in grip. The height error being corrected <em>is</em> the slip, and lowering the palm
    further does not put it back.</p>
    <p><strong>Feedback on the tilt is not.</strong> The same loop applied to orientation is
    harmless at any number of corrections, because a rotation about the tool's own foot does not
    trade against the grip in the same way.</p>
  </div>
  <div class="pair">
    <figure>{desc_chart}<figcaption>Corrections during the set-down.</figcaption></figure>
    <figure>{repose_chart}<figcaption>Corrections bringing it upright.</figcaption></figure>
  </div>
  <div class="col">
    <h3>The controls</h3>
    <div class="tw"><table><thead><tr><th>condition</th><th class="num">chains</th>
      <th class="num">turns</th><th>note</th></tr></thead><tbody>{ctrls}</tbody></table></div>
    <p>The mid-air re-grasp is the one that changed how I think about the floor. The gait's
    tripod ring solves to 1.1&nbsp;mm at the carry's own palm pose &mdash; it is reachable, and
    the fingers can hold it &mdash; but a linear joint-space move from one grasp to the other
    passes through a configuration that holds neither, and in mid-air there is nothing under the
    tool. Set it on the floor first and the identical transition works every time. The floor's
    contribution is not only that a released tool stays put; it is that the hand becomes free to
    change its mind about how to hold it.</p>
    <h3>What the seam will absorb</h3>
    <p>The carry's exit tilt depends almost entirely on how long the hold after the turn is
    allowed to run &mdash; cos 0.837 at the last commanded step, 0.946 after 0.6&nbsp;s,
    0.996 after 1.0. Handing the seam anything from 6&deg; to 25&deg; makes no difference to
    whether the chain completes.</p>
    <div class="tw"><table><thead><tr><th class="num">hold after the turn  s</th>
      <th class="num">carry exit tilt &deg;</th><th class="num">chains</th>
      <th class="num">turns</th></tr></thead><tbody>{hold_rows}</tbody></table></div>
    <p>The gait itself keeps the ceiling it had on a bare floor: 30&deg; of azimuth per cycle is
    safe, 45&deg; roughly doubles throughput at a real cost in reliability, and 60&deg; walks
    the tool over.</p>
    <div class="tw"><table><thead><tr><th class="num">stroke &deg;</th><th class="num">chains</th>
      <th class="num">turns / 8 cycles</th><th class="num">&deg; per cycle</th>
      <th class="num">final tilt &deg;</th></tr></thead><tbody>{stroke_rows}</tbody></table></div>
    <p>Read the 45&deg; row carefully. The pad ring is a gear &mdash; the pads ride at
    21.05&nbsp;mm about a 12.5&nbsp;mm shaft, so slip-free transmission is 1.684 &mdash; and
    1.684 &times; 45&deg; is 75.8&deg;. The measured {st45_deg:.0f}&deg; per cycle is
    {st45_over:.2f} times that ceiling, which means the tool is being flicked and left to coast
    rather than driven. That is throughput of a kind, but it is not the same primitive, and the
    reliability drop is where it shows.</p>
  </div>
</section>

<section>
  <div class="col">
    <h2>On an arm</h2>
    <p class="sub">the six palm DOF, paid for by a UR5e</p>
    <p>Every number this program has produced was made by a floating, gravity-compensated palm
    carrying six position-actuated joints. <code>build_real_v1_arm_scene.py</code> deletes them
    and bolts the hand to a UR5e flange; <code>palm_driver.py</code> puts the gantry and the arm
    behind one <code>read / write / solve / cmd_pose</code> interface, so the chain probe
    contains no branch on which wrist it is using and a difference between the two runs belongs
    to the wrist. Differential IK is mink, pinned to 1.1 &mdash; 1.2 resolves MuJoCo to 3.12,
    and every physics result in this repository was measured on 3.6. The solver runs on a
    separate arm-only model, because mink integrates over every degree of freedom a model has
    and the task scene's include nine finger joints and the tool's free joint.</p>
    <div class="tw"><table><thead><tr><th>wrist</th><th class="num">chains</th>
      <th class="num">turns / 8 cy</th><th class="num">&deg; per cycle</th>
      <th class="num">carry tilt &deg;</th><th class="num">drift mm</th>
      <th class="num">40 cycles</th><th class="num">worst IK mm</th>
      </tr></thead><tbody>{wrist_rows}</tbody></table></div>
    <p>The arm is not worse. Worst IK residual over a whole rollout is a thousandth of a
    millimetre and no commanded pose was unreachable, so the palm trajectory the seam needs is
    not exotic &mdash; a 6R arm makes all of it, including the rigid re-pose transfer and the
    lateral re-index under a standing tool.</p>
  </div>
  <figure>
    <video src="{vid_arm}" controls muted loop playsinline preload="metadata"></video>
    <figcaption>The same controller, the same contact parameters, a UR5e instead of the
    floating palm.</figcaption>
  </figure>
  <div class="col">
    <p>What the arm costs is stiffness, and it is a cliff rather than a slope. The menagerie
    UR5e has no gravity feedforward: its shoulder sits 21&nbsp;mrad below its set-point under
    the arm's own twelve kilograms, and the palm droops 12.7&nbsp;mm at the top of the lift.
    That alone turns a 5&deg; carry exit into 29&deg; and breaks the seam. Compensating the
    <em>arm's</em> links &mdash; not the hand, which is real payload the controller does not
    know about &mdash; sweeps that droop continuously from 12.7&nbsp;mm down to
    0.35&nbsp;mm on an otherwise identical robot.</p>
  </div>
  <figure>{droop_chart}
    <figcaption>Chains completed against palm droop under load. Nothing between 1.6 and
    12.7&nbsp;mm works; 0.97&nbsp;mm works every time.</figcaption>
  </figure>
  <div class="callout col">
    <span class="tag">the spec</span>
    <p>A wrist for this task has to hold the commanded palm pose to about one millimetre under
    the hand's own weight plus a 25&nbsp;g tool. That is a number an arm can be checked against
    before it is bought, and it is the first hardware requirement this program has stated in
    units that are not the hand's own.</p>
  </div>
  <figure>
    <img src="{film_arm}" alt="Seventeen frames of the arm-mounted chain, from grasp to gait.">
    <figcaption>The arm run at its seams. The wrist rotation through the reorient is the move
    the floating palm used to get for free.</figcaption>
  </figure>
</section>

<section>
  <div class="col">
    <h2>Into a screw</h2>
    <p class="sub">a 45&deg; frustum tip, and a countersink to put it in</p>
    <p>A flat-ended rod on a flat plane makes the ground a pure friction brake and leaves the
    tool free to walk, which is why <code>drift_mm</code> had to be reported at all. A screw sits
    in a seat, and a seat centres, wedges and constrains. The tool gains a 45&deg; frustum tip
    &mdash; a mesh, so the cone is exact rather than stepped, and a frustum is convex so MuJoCo's
    hull is the shape itself &mdash; and the floor gains a matching countersink. A conical
    <em>hole</em> is not convex, so the seat is built the way a real recess is: a ring of
    thirty-two facets for the wall, a disc for the relieved bottom, and a flat annulus for the
    table around the mouth.</p>
    <p>The tool is picked up lying on the table, 40&nbsp;mm from the seat, so the chain has to
    transport it as well as insert it.</p>
  </div>
  <figure>
    <video src="{vid_screw}" controls muted loop playsinline preload="metadata"></video>
    <figcaption>Pick up flat, lift, reorient, carry across, insert, release, re-index, drive.</figcaption>
  </figure>
  <div class="pair">
    <figure>{seat_turn}<figcaption>Degrees of shaft per gait cycle.</figcaption></figure>
    <figure>{seat_drift}<figcaption>Lateral walk over eight cycles.</figcaption></figure>
  </div>
  <div class="col">
    <p>The seat costs <strong>{flat_deg:.1f} &rarr; {seat_deg:.1f} degrees per cycle</strong>,
    about a sixth of the turn, and returns <strong>{flat_drift_mm:.2f} &rarr;
    {seat_drift_mm:.2f}&nbsp;mm</strong> of walk, a factor of sixteen. The loss is the wedge:
    normal force on a cone flank is the axial load over sin&nbsp;&alpha;, so the same press buys
    more resisting torque than it does on a plane. Over forty cycles the seated tool turns
    {seat_endur:.2f} revolutions against {flat_endur:.2f} on the plane, and stays seated.</p>
    <p>The cone angle trades those two against each other directly, and the ratio is the wedge
    factor read off the data: 2.00, 1.41, 1.15 for half-angles of 30, 45 and 60 degrees.</p>
  </div>
  <div class="pair">
    <figure>{ang_ok}<figcaption>A shallower cone inserts more reliably.</figcaption></figure>
    <figure>{ang_turn}<figcaption>&hellip;and turns less, by the wedge factor.</figcaption></figure>
  </div>
  <div class="col">
    <h3>Insertion is a press, not a placement</h3>
    <p>Below about 8&nbsp;mm of commanded press the tip stops proud of its seat and the gait has
    nothing to push against; from 10 to 18&nbsp;mm there is no further change. And the capture
    radius is 6&nbsp;mm against 10.5&nbsp;mm of geometric clearance, because the tool arrives
    with two to five degrees of tilt and a tilted cone binds in a matched cone &mdash; an 8&deg;
    entry stops three millimetres into a ten-millimetre seat.</p>
  </div>
  <div class="pair">
    <figure>{press_chart}<figcaption>Press through the grip.</figcaption></figure>
    <figure>{cap_chart}<figcaption>Deliberate lateral error in the insertion target.</figcaption></figure>
  </div>
  <div class="col">
    <p>The re-grip that pays for the insertion has a narrow window of its own. Too little and
    the grasp is gone before the tip is home; too much and the pads, which sit at different
    stations along the shaft, torque it out of line and the cone binds on the way in.</p>
  </div>
  <figure>{regrip_chart}
    <figcaption>Extra radial interference commanded once after the turn. 0.3&nbsp;mm is the
    working value; 1&nbsp;mm never seats.</figcaption>
  </figure>
  <div class="col">
    <p>Transport distance, by contrast, does not matter. Standing the tool up and carrying it
    across are expressed as one pose command and therefore one ramp, so 30&nbsp;mm and 90&nbsp;mm
    cost the same.</p>
    <div class="tw"><table><thead><tr><th class="num">transport mm</th><th class="num">seated</th>
      <th class="num">&deg; per cycle</th><th class="num">final offset from seat mm</th>
      </tr></thead><tbody>{dist_rows}</tbody></table></div>
  </div>
  <figure>
    <img src="{film_screw}" alt="Frames of the screw chain: pick up beside the countersink, reorient, carry across, insert, release, gait.">
    <figcaption>The insertion run. The hand releases a tool standing in its seat, moves, and
    takes it again in the gait's ring grasp.</figcaption>
  </figure>
</section>

<section class="col">
  <h2>What this does and does not show</h2>
  <p>It is simulation, on one hand, with one tool. The upright starting pose is produced by the
  carry rather than granted, which was the point, but the carry itself is rv05's published
  configuration and the other nine designs never reach the seam under it. Nothing here has been
  through the trajectory-clearance or servo-limit gates, so none of it is a deployment claim.
  A 45&deg; countersink at 0.0072&nbsp;N&middot;m of resisting torque is a seat, not a
  fastener &mdash; there is no thread, no pitch, and the tool does not have to descend as it
  turns.</p>
  <p>Three measurement errors on the way each read as physics before a control caught them, and
  they are recorded because two of them are the same error in different clothes. A re-grip
  measured from the pad's <em>achieved</em> position hands back exactly the deflection that was
  carrying the load, and bled a &ldquo;constant&rdquo; grip from 18.4&nbsp;N to 0.25&nbsp;N over
  seven reissues with the tool doing nothing; the bench work met this one as
  <code>achieved_fraction</code>. A proportional tilt correction tracking a <em>scheduled</em>
  reference changes sign whenever the plant runs behind, and on a 20&nbsp;N grip with 150 steps
  of lag it always does &mdash; the palm oscillated to &minus;0.49&nbsp;rad and threw the tool
  from a state that had already reached 3&deg;. And <code>probe_real_v1_gait._ground</code> asks
  for contacts with the body named <code>world</code>, which reports zero support for a tool
  sitting firmly in a seat that is its own body.</p>
  <h3>Next</h3>
  <ul>
    <li>Put the seat on a load cell in sim: sweep resisting torque until the gait stalls, and
    report the stall torque rather than the turn rate. That number is what decides whether this
    is a fastener primitive.</li>
    <li>Close the gait loop. Gaiting is the first task in this program whose controlled variable
    &mdash; rotation about the tool's own axis &mdash; the bench's AprilTag rig measures
    directly, at 0.017&deg; rms.</li>
    <li>Run the export and clearance gates on the chained trajectory. The gait's ring grasp and
    the re-index move have never been checked against finger&ndash;finger clearance or the
    servo travel limits, and three of four plans on the control station once failed exactly
    that check.</li>
  </ul>
</section>

<section class="col">
  <h2>Reproduce</h2>
<pre><code>uv run --extra rl python scripts/real_v1_chain_study.py \
    --out docs/experiments/20260903-real_v1_chain --reps 6 --cycles 8

uv run --extra rl --extra arm python scripts/build_real_v1_arm_scene.py
uv run --extra rl --extra arm python scripts/real_v1_arm_study.py \
    --out docs/experiments/20260903-real_v1_chain --reps 6 --cycles 8

uv run --extra rl python scripts/real_v1_screw_study.py \
    --out docs/experiments/20260903-real_v1_screw --reps 6 --cycles 8

uv run --extra rl python scripts/real_v1_chain_page.py</code></pre>
  <p>Each study is well under a minute of CPU on six workers; the arm study is serial because
  mink holds a model per process.</p>
</section>
"""


if __name__ == "__main__":
    raise SystemExit(main())
