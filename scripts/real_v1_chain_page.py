"""Build the doc page for the chained task: grasp -> gait, on an arm, into a screw.

Regenerable: everything on the page is read from the three study JSONs and the seam ledger, so
re-running a study and re-running this is the whole update path. Charts are hand-emitted SVG
(theme-aware via CSS custom properties) and media are inlined as base64.

    uv run --extra rl python scripts/real_v1_chain_page.py \
        --chain docs/experiments/20260903-real_v1_chain \
        --screw docs/experiments/20260903-real_v1_screw \
        --out   docs/experiments/20260903-real_v1_chain/20260903-real_v1_chain_and_countersink.html
"""
from __future__ import annotations

import argparse
import base64
import json
import statistics as st
import subprocess
import tempfile
from pathlib import Path
import retro_style

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
    cap = lambda t: t[0].upper() + t[1:] if t and t[0].islower() and t.split(" ")[0].isalpha() else t  # noqa: E731
    o.append(f'<text class="axlab" x="{(w-pad[1]+pad[3])/2:.0f}" y="{h-4}" '
             f'text-anchor="middle">{cap(xlab)}</text>')
    # the y title sits above the plot, horizontal: a rotated title longer than the plot height was clipped
    o.append(f'<text class="axlab" x="{pad[3]-8}" y="15" text-anchor="start">{cap(ylab)}</text>')
    o.append("</g>")
    return "".join(o)


def bars(rows, xlab, ylab, ymax=None, colour=A, note=None, w=560, h=250, fmt="{:g}",
         labfmt="{:.0f}", xfmt="{:g}"):
    """One series of labelled bars: (x label, value). Direct labels, no legend needed."""
    pad = (58, 16, 40, 46)
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
        o.append(f'<text class="note" x="{w-pad[1]}" y="15" text-anchor="end">{note[0].upper() + note[1:]}</text>')
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
    pad = (62, 24, 40, 50)
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
    out = args.out or (args.chain / "20260903-real_v1_chain_and_countersink.html")

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
            (f"{k_lead}/{n_lead}", "Chains completed, flat floor"),
            (f"{mean(endur, 'turns'):.2f}", "Turns in 40 cycles"),
            (f"{rate(ur)[0]}/{rate(ur)[1]}", "Chains completed on a UR5e"),
            ("0.97 mm", "Largest palm droop that completes"),
            (f"{rate(sw_seat)[0]}/{rate(sw_seat)[1]}", "Tool seated and turned"),
            ("6 mm", "Insertion capture radius"),
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
                      "measured corrections during the set-down", "chains complete, of 6",
                      ymax=6.6, colour=A, fmt="{:g}", w=360, h=230)
    rps = group(arm(CH, "repose"), ["repose_iters"])
    repose_chart = bars([(f"{k[0]}", rate(v)[0]) for k, v in sorted(rps.items())],
                        "corrections bringing it upright", "chains complete, of 6",
                        ymax=6.6, colour=B, fmt="{:g}", w=360, h=230)

    # ---------------------------------------------------------------- droop cliff
    stf = group(arm(AR, "stiffness"), ["droop_mm"])
    droop_rows = sorted(((k[0], rate(v)[0]) for k, v in stf.items()), reverse=True)
    droop_chart = bars([(f"{d:.2g}" if d < 10 else f"{d:.3g}", k) for d, k in droop_rows],
                       "palm droop under load (mm)", "chains complete, of 6",
                       ymax=6.6, colour=B)

    # ---------------------------------------------------------------- seat vs plane
    seat_turn = bars([("flat plane", mean(sw_flat, "gain_mean_deg")),
                      ("45&#176; seat", mean(sw_seat, "gain_mean_deg"))],
                     "", "degrees of shaft per cycle", ymax=58, colour=A, labfmt="{:.1f}", w=330, h=250)
    seat_drift = bars([("flat plane", mean(sw_flat, "drift_mm")),
                       ("45&#176; seat", mean(sw_seat, "drift_mm"))],
                      "", "lateral walk over 8 cycles (mm)", ymax=1.6, colour=B,
                      labfmt="{:.2f}", fmt="{:g}", w=330, h=250)

    # ---------------------------------------------------------------- insertion window
    pr = group(arm(SW, "press"), ["press_mm"])
    press_chart = bars([(f"{k[0]:g}", rate(v)[0]) for k, v in sorted(pr.items())],
                       "press through the grip (mm)", "seated and turned, of 6",
                       ymax=6.6, colour=A, w=330, h=250)
    cap = group(arm(SW, "capture"), ["place_err_mm"])
    cap_chart = bars([(f"{k[0][0]:g}", rate(v)[0]) for k, v in sorted(cap.items())],
                     "Lateral error in the target (mm)", "Seated and turned, of 6",
                     ymax=6.6, colour=B, w=330, h=250)
    rg = group(arm(SW, "regrip"), ["carry_squeeze_mm"])
    regrip_chart = bars([(f"{k[0]:g}", rate(v)[0]) for k, v in sorted(rg.items())],
                        "Re-grip interference (mm)",
                        "Seated and turned, of 6", ymax=6.6, colour=A, w=330, h=250)
    ang = group(arm(SW, "angle"), ["tip_len_mm"])
    ang_lab = {5.77: "60&#176;", 10.0: "45&#176;", 17.32: "30&#176;"}
    ang_rows = sorted(((ang_lab[k[0]], rate(v)[0], mean(v, "gain_mean_deg")) for k, v in ang.items()),
                      key=lambda r: -float(r[0].split("&")[0]))
    ang_ok = bars([(a, k) for a, k, _ in ang_rows], "countersink half-angle",
                  "seated and turned, of 6", ymax=6.6, colour=B, w=330, h=250)
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
    out.write_text(retro_style.apply(html), encoding="utf-8")  # plain page style (owner, 2026-10-09)
    print(f"-> {out}  ({out.stat().st_size / 1e6:.2f} MB)")
    return 0


BODY = r"""
<section class="col" id="terms">
  <h2>Terms and metrics</h2>
  <dl class="terms">
    <dt><b>Carry</b></dt><dd>The open-loop turn that stands the shaft upright in the hand, about a pivot raised above the
    contacts.</dd>
    <dt><b>Set-down, release, re-index</b></dt><dd>Lowering the standing tool to the floor or seat, opening the hand, and
    moving the palm over the gait&#8217;s ring grasp.</dd>
    <dt><b>Gait</b></dt><dd>Four-phase cycles of grasp, twist, release and return that turn the standing tool about its own
    axis.</dd>
    <dt><b>Chains</b></dt><dd>Seeds, of six spawn-jittered rollouts, that pass all four gates of the chain.</dd>
    <dt><b>Carry exit tilt</b></dt><dd>Angle of the tool from vertical when the carry hands over to the set-down, in
    degrees.</dd>
    <dt><b>Palm droop</b></dt><dd>Distance of the palm below its commanded pose at the top of the lift, in mm.</dd>
    <dt><b>Stroke</b></dt><dd>Azimuth the pads twist the tool per gait cycle, in degrees.</dd>
    <dt><b>Capture radius</b></dt><dd>Largest lateral error of the insertion target at which the tool still seats, in
    mm.</dd>
  </dl>
</section>

<section class="col">
  <h2>Carry and gait before this study</h2>
  <p><code>probe_real_v1_carry.py</code> takes the shaft off the table and stands it vertical in
  the hand: a fixed grasp, a rotation about a pivot raised above the contacts, cos 0.996, no
  policy anywhere. <code>probe_real_v1_gait.py</code> spins a shaft that is <em>already</em>
  standing on the floor through five revolutions in forty four-phase cycles, again from a
  precomputed set-point table. The first ends with the tool in the air at 113&nbsp;mm; the
  second begins with it planted at 50&nbsp;mm, put there by a line of code.</p>
  <p>This page joins the two across the 63&nbsp;mm in between, then replaces the floating palm
  with a UR5e and the flat floor with a countersink.</p>
</section>

<section>
  <div class="col">
    <h2>Chain from grasp to gait</h2>
    <p class="sub">grasp &rarr; lift &rarr; reorient &rarr; set down &rarr; release &rarr; re-index &rarr; gait</p>
    <p>One rollout runs in one <code>MjData</code>, with no state passed through a file. The re-pose
    is a rigid transfer: with the tool fixed in the palm frame, a desired object pose determines the
    palm pose, a form an arm can execute. Each phase below lists the time since the grasp closed,
    the pad force and the tool's tilt from vertical.</p>
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
    <figcaption>The same run, one frame per checkpoint. In frame five the hand is fully open and
    the tool stands by itself.</figcaption>
  </figure>
  <div class="col">
    <p>A chain completes when four gates pass: the carry held, the tool stood within 10&nbsp;mm of
    its resting height with support contact and under 14&deg; of tilt, the gait grasp took on at
    least two pads, and all requested cycles ran without losing the tool. A run that turns the tool
    and drops it on the way down fails.</p>
    <div class="tw"><table><thead><tr>
      <th>Design</th><th class="num">Carry exit tilt (&deg;)</th><th class="num">Chains</th>
      <th class="num">Turns / 8 cycles</th></tr></thead><tbody>{des_rows}</tbody></table></div>
    <p>Only <code>rv05_manual_stored</code>, the hand the bench was built from, completes. The other
    nine fail at the carry with rv05's carry settings (axis height, turn duration and residual
    budget), which were not re-tuned per design.</p>
  </div>
</section>

<section>
  <div class="col">
    <h2>Grip decay during the set-down</h2>
    <p>At the top of a raised-pivot turn the tool is often cradled against the middle phalanges with
    the pads barely on it (about 0.5&nbsp;mm of commanded interference), and the hold leaks. The
    shaft creeps down through the pads at about 1.5&nbsp;mm/s, and the pad force decays to nothing in
    under two seconds, as <code>probe_real_v1_carry</code> measured from the other direction
    (16.1&nbsp;N to zero, tool on the table by 1.6&nbsp;s).</p>
  </div>
  <figure>{grip_chart}
    <figcaption>Pad force at each checkpoint of one rollout, to the last held phase. The shipped
    chain has 12.0&nbsp;N at the top of the turn and 0.4&nbsp;N by the press; a single 0.3&nbsp;mm
    radial re-squeeze, measured from the <em>commanded</em> pad position, keeps 7.7&ndash;9.0&nbsp;N
    over a longer sequence.</figcaption>
  </figure>
  <div class="col">
    
    <p><strong>Height feedback.</strong> The set-down re-measures the tool and recomputes the palm
    pose. Done once, as a single continuous move, it stands the tool on 6 of 6 spawn-jittered seeds;
    done eight times, on 1. The extra corrections add dwells, each of which costs grip, and the
    height error they correct is the slip itself.</p>
    <p><strong>Tilt feedback.</strong> The same loop applied to orientation completes at any number of
    corrections, since a rotation about the tool's own foot costs no grip.</p>
  </div>
  <div class="pair">
    <figure>{desc_chart}<figcaption>Corrections during the set-down.</figcaption></figure>
    <figure>{repose_chart}<figcaption>Corrections bringing it upright.</figcaption></figure>
  </div>
  <div class="col">
    <h3>Control conditions</h3>
    <div class="tw"><table><thead><tr><th>Condition</th><th class="num">Chains</th>
      <th class="num">Turns</th><th>Note</th></tr></thead><tbody>{ctrls}</tbody></table></div>
    <p>The mid-air re-grasp fails although the gait's tripod ring solves to 1.1&nbsp;mm at the carry's
    own palm pose. A linear joint-space move from one grasp to the other passes through a
    configuration that holds neither, and in mid-air nothing supports the tool. With the tool set
    on the floor first the same transition works on every seed, so the floor lets the hand change
    its grasp.</p>
    <h3>Tolerance of the set-down to the carry's exit tilt</h3>
    <p>The carry's exit tilt depends mostly on the length of the hold after the turn (cos 0.837 at
    the last commanded step, 0.946 after 0.6&nbsp;s, 0.996 after 1.0&nbsp;s). Any exit tilt
    from 6&deg; to 25&deg; completes the chain.</p>
    <div class="tw"><table><thead><tr><th class="num">Hold after the turn (s)</th>
      <th class="num">Carry exit tilt (&deg;)</th><th class="num">Chains</th>
      <th class="num">Turns</th></tr></thead><tbody>{hold_rows}</tbody></table></div>
    <p>The gait keeps the limits it had on a bare floor. A 30&deg; stroke per cycle is reliable,
    45&deg; roughly doubles the turn rate and loses seeds, and 60&deg; walks the tool over.</p>
    <div class="tw"><table><thead><tr><th class="num">Stroke (&deg;)</th><th class="num">Chains</th>
      <th class="num">Turns / 8 cycles</th><th class="num">&deg; per cycle</th>
      <th class="num">Final tilt (&deg;)</th></tr></thead><tbody>{stroke_rows}</tbody></table></div>
    <p>The pad ring acts as a gear: the pads ride at 21.05&nbsp;mm about a 12.5&nbsp;mm shaft, so
    slip-free transmission is 1.684 and a 45&deg; stroke drives at most 75.8&deg;. The measured
    {st45_deg:.0f}&deg; per cycle is {st45_over:.2f} times that, so at 45&deg; the tool is flicked
    and coasts, which is where the seeds are lost.</p>
  </div>
</section>

<section>
  <div class="col">
    <h2>Chain on a UR5e</h2>
    <p>Every earlier result used a floating, gravity-compensated palm with six position-actuated
    joints. <code>build_real_v1_arm_scene.py</code> replaces them with a UR5e flange, and
    <code>palm_driver.py</code> puts the gantry and the arm behind one
    <code>read / write / solve / cmd_pose</code> interface, so the chain probe is the same for both
    wrists. Differential IK is mink 1.1 (1.2 requires MuJoCo 3.12; every result here used 3.6),
    solved on a separate arm-only model because mink integrates over every degree of freedom of a
    model.</p>
    <div class="tw"><table><thead><tr><th>Wrist</th><th class="num">Chains</th>
      <th class="num">Turns / 8 cycles</th><th class="num">&deg; per cycle</th>
      <th class="num">Carry tilt (&deg;)</th><th class="num">Drift (mm)</th>
      <th class="num">Turns / 40 cycles</th><th class="num">Worst IK (mm)</th>
      </tr></thead><tbody>{wrist_rows}</tbody></table></div>
    <p>The arm completes 6 of 6 with a worst IK residual of 0.001&nbsp;mm and no unreachable pose, so
    a 6R arm makes every palm move of the chain, including the rigid re-pose and the lateral
    re-index over a standing tool.</p>
  </div>
  <figure>
    <video src="{vid_arm}" controls muted loop playsinline preload="metadata"></video>
    <figcaption>The same controller, the same contact parameters, a UR5e instead of the
    floating palm.</figcaption>
  </figure>
  <div class="col">
    <p>The arm's cost is stiffness, with a sharp threshold. The menagerie UR5e has no gravity
    feedforward. Its shoulder sits 21&nbsp;mrad below its set-point under the arm's own 12&nbsp;kg,
    and the palm droops 12.7&nbsp;mm at the top of the lift. This turns a 5&deg; carry exit into
    29&deg; and breaks the set-down. Compensating the <em>arm's</em> links (the hand stays an
    undeclared payload) reduces the droop from 12.7&nbsp;mm to 0.35&nbsp;mm on the same robot.</p>
  </div>
  <figure>{droop_chart}
    <figcaption>Chains completed against palm droop under load: every seed completes at
    0.97&nbsp;mm and none from 1.6 to 12.7&nbsp;mm.</figcaption>
  </figure>
  <div class="callout col">
    <span class="tag">Wrist specification</span>
    <p>A wrist for this task must hold the commanded palm pose to about 1&nbsp;mm under the hand's
    weight plus a 25&nbsp;g tool, a requirement an arm can be checked against before
    purchase.</p>
  </div>
  <figure>
    <img src="{film_arm}" alt="Seventeen frames of the arm-mounted chain, from grasp to gait.">
    <figcaption>The arm run, one frame per checkpoint; the wrist rotates through the
    turn.</figcaption>
  </figure>
</section>

<section>
  <div class="col">
    <h2>Countersink seating</h2>
    <p class="sub">A 45&deg; frustum tip and a matching countersink</p>
    <p>On a flat plane the ground acts only as a friction brake and the tool walks
    (<code>drift_mm</code>). The tool now has a 45&deg; frustum tip, a convex mesh that MuJoCo&#8217;s
    hull represents exactly, and the floor has a matching countersink. A conical hole is not
    convex, so the seat is built from 32 facets for the wall, a disc for the relieved bottom and a flat
    annulus around the mouth.</p>
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
    <p>The seat lowers the turn from <strong>{flat_deg:.1f} to {seat_deg:.1f} degrees per
    cycle</strong>, about a sixth, and the walk from <strong>{flat_drift_mm:.2f} to
    {seat_drift_mm:.2f}&nbsp;mm</strong>, sixteen-fold. The wedge explains the loss: the normal
    force on a cone flank is the axial load divided by sin&nbsp;&alpha;, so the same press gives
    more resisting torque than on a plane. Over forty cycles the seated tool turns
    {seat_endur:.2f} revolutions against {flat_endur:.2f} on the plane and stays seated.</p>
    <p>The cone angle trades insertion against turn by the wedge factor read off the data: 2.00,
    1.41 and 1.15 for half-angles of 30, 45 and 60&deg;.</p>
  </div>
  <div class="pair">
    <figure>{ang_ok}<figcaption>A shallower cone inserts more reliably.</figcaption></figure>
    <figure>{ang_turn}<figcaption>&hellip;and turns less, by the wedge factor.</figcaption></figure>
  </div>
  <div class="col">
    <h3>Press depth and capture radius</h3>
    <p>Below about 8&nbsp;mm of commanded press the tip stops above its seat and the gait has
    nothing to push against. From 10 to 18&nbsp;mm the press makes no further difference. The
    capture radius is 6&nbsp;mm against 10.5&nbsp;mm of geometric clearance, because the tool
    arrives with 2&ndash;5&deg; of tilt and a tilted cone binds in a cone of the same angle. An
    8&deg; entry stops 3&nbsp;mm into a 10&nbsp;mm seat.</p>
  </div>
  <div class="pair">
    <figure>{press_chart}<figcaption>Press through the grip.</figcaption></figure>
    <figure>{cap_chart}<figcaption>Deliberate lateral error in the insertion target.</figcaption></figure>
  </div>
  <div class="col">
    <p>The re-grip before insertion has a narrow window: too little and the grasp is gone before
    the tip is home, too much and the pads, at different stations along the shaft, torque it out of
    line so that the cone binds.</p>
  </div>
  <figure>{regrip_chart}
    <figcaption>Extra radial interference commanded once after the turn. 0.3&nbsp;mm is the
    working value; 1&nbsp;mm never seats.</figcaption>
  </figure>
  <div class="col">
    <p>Transport distance makes no difference: standing the tool up and carrying it across are one
    pose command and one ramp, so 30&nbsp;mm and 90&nbsp;mm cost the same.</p>
    <div class="tw"><table><thead><tr><th class="num">Transport (mm)</th><th class="num">Seated</th>
      <th class="num">&deg; per cycle</th><th class="num">Final offset from the seat (mm)</th>
      </tr></thead><tbody>{dist_rows}</tbody></table></div>
  </div>
  <figure>
    <img src="{film_screw}" alt="Frames of the screw chain: pick up beside the countersink, reorient, carry across, insert, release, gait.">
    <figcaption>The insertion run. The hand releases a tool standing in its seat, moves, and
    takes it again in the gait's ring grasp.</figcaption>
  </figure>
</section>

<section class="col">
  <h2>Limitations and measurement errors</h2>
  <p>All results are simulation on the shipped servo model (kp 30), with one hand and one tool; the carry is rv05&#8217;s
  published configuration, under which the other nine designs fail. Nothing here has passed the trajectory-clearance or
  servo-limit gates. The countersink resists with 0.0072&nbsp;N&middot;m and has no thread, so the tool does not descend as
  it turns.</p>
  <p>Three measurement errors read as physics before a control caught them. A re-grip measured from the pad&#8217;s
  <em>achieved</em> position returns the deflection that held the load, and bled a constant grip from 18.4&nbsp;N to
  0.25&nbsp;N over seven reissues with the tool still. A proportional tilt correction tracking a <em>scheduled</em> reference
  changes sign whenever the plant lags, and with a 20&nbsp;N grip and 150 steps of lag the palm oscillated to
  &minus;0.49&nbsp;rad and threw a tool that had reached 3&deg;. <code>probe_real_v1_gait._ground</code> counted only contacts
  with the body named <code>world</code>, so it reported no support for a tool seated in a countersink, which is its own
  body.</p>
</section>

<section class="col">
  <h2>Future work</h2>
  <ul>
    <li>Load the seat in simulation: sweep the resisting torque until the gait stalls and report the stall torque, which
    decides whether the gait can drive a fastener.</li>
    <li>Close the gait loop on the tool&#8217;s rotation about its own axis, which the bench&#8217;s AprilTag rig measures at
    0.017&deg; rms.</li>
    <li>Run the export and clearance gates on the chained trajectory: the gait&#8217;s ring grasp and the re-index move have
    not been checked for finger&ndash;finger clearance or servo travel, and three of four plans on the control station once
    failed that check.</li>
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
  <p>Each study takes under a minute of CPU on six workers; the arm study is serial because mink
  holds a model per process.</p>
</section>
"""


if __name__ == "__main__":
    raise SystemExit(main())
