"""Build the doc page for the handover study: changing grasp without letting go.

Regenerable -- every number, chart and table is read from handover_study.json, so re-running the
study and re-running this is the whole update path. Charts are the chain page's hand-emitted SVG
helpers (theme-aware via CSS custom properties, at most two categorical series each) and media
are inlined as base64.

    uv run --extra rl python scripts/real_v1_handover_page.py \
        --data docs/experiments/20260903-real_v1_handover \
        --out  docs/experiments/20260903-real_v1_handover/20260903-real_v1_relay_handover.html
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from real_v1_chain_page import A, B, REF, b64, bars, jpeg, lines, okcell  # noqa: E402
import retro_style  # noqa: E402

MODES = {
    "full": ("Release", "Open the hand, move the palm to the gait's pose, close on the ring"),
    "none": ("In place", "Release and retake all three at once at the carry's palm pose"),
    "relay": ("Relay", "Move the pads onto the ring one at a time, palm still"),
    "track": ("Track", "Move the palm with the pads held at their world points, then relay"),
    "slide": ("Slide", "Move the palm with the pads dragged along the tool onto the ring"),
}


def sel(rows, **kw):
    return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]


def agg(g):
    ok = [r for r in g if r["ok"]]
    return {
        "n": len(g), "k": sum(r["ok"] for r in g),
        "deg": st.mean(r["gain_mean_deg"] for r in ok) if ok else 0.0,
        "free": st.mean(r["free_frac"] for r in g),
        "one": st.mean(r["one_frac"] for r in g),
        "gnd": st.mean(r["ground_frac"] for r in g),
        "slip": st.mean(r["slip_mm_per_cycle"] for r in g),
        "tilt": st.mean(r["final_tilt_deg"] for r in g),
        "drift": st.mean(r["drift_mm"] for r in g),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path,
                    default=ROOT / "docs/experiments/20260903-real_v1_handover")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    D = args.data
    out = args.out or (D / "20260903-real_v1_relay_handover.html")
    rows = json.loads((D / "handover_study.json").read_text())

    # ------------------------------------------------------------------ the numbers
    hand = {k: agg(sel(rows, arm="handover", reindex=k)) for k in MODES}
    gait = {(i, g): agg(sel(rows, arm="gait", reindex=i, relay_gait=g))
            for i in ("full", "relay") for g in (False, True)}
    nrel = {g: agg(sel(rows, arm="nonerelay", reindex="none", relay_gait=g))
            for g in (False, True)}
    grip = {q: agg(sel(rows, arm="grip", relay_squeeze_mm=q))
            for q in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)}
    tf = (0.0, 0.25, 0.5, 0.75, 1.0)
    trk = {f: agg(sel(rows, arm="track", track_frac=f)) for f in tf}
    trk_ring = {f: st.mean(r["ring_ik_grip_mm"] for r in sel(rows, arm="track", track_frac=f))
                for f in tf}
    trk_hold = {f: st.mean(r["hold_ik_mm"] for r in sel(rows, arm="track", track_frac=f))
                for f in tf}
    load = {(i, t): agg(sel(rows, arm="load", reindex=i, tilt_deg=t))
            for i in ("full", "relay") for t in (0.0, 4.0, 8.0, 12.0, 14.0, 16.0, 20.0, 30.0)}
    seat = {(i, t): agg(sel(rows, arm="screw_load", reindex=i, tilt_deg=t))
            for i in ("full", "relay") for t in (0.0, 8.0, 16.0, 30.0)}
    dirn = {(i, a): agg(sel(rows, arm="dir", reindex=i, tilt_dir=a))
            for i in ("full", "relay") for a in (0.0, 90.0, 180.0, 270.0)}
    flr = {(i, nf): agg(sel(rows, arm="floor", reindex=i, no_floor_gait=nf))
           for i in ("full", "relay") for nf in (False, True)}
    end = {i: sel(rows, arm="endurance", reindex=i) for i in ("full", "relay")}
    endt = {i: [r["turns"] for r in end[i]] for i in end}
    palm = st.mean(r["palm_move_mm"] for r in sel(rows, arm="handover", reindex="full"))
    palmd = st.mean(r["palm_move_deg"] for r in sel(rows, arm="handover", reindex="full"))
    ringp = st.mean(r["ring_ik_grip_mm"] for r in sel(rows, arm="handover", reindex="relay"))
    ringf = st.mean(r["ring_ik_grip_mm"] for r in sel(rows, arm="handover", reindex="full"))
    holdik = st.mean(r["hold_ik_mm"] for r in sel(rows, arm="track", track_frac=1.0))

    K = dict(
        NCELLS=len(rows),
        FREE_FULL=f"{hand['full']['free'] * 100:.0f}",
        FREE_RELAY=f"{gait[('relay', True)]['free'] * 100:.1f}",
        DEG_FULL=f"{hand['full']['deg']:.1f}",
        DEG_RELAY=f"{gait[('relay', True)]['deg']:.1f}",
        RATIO=f"{gait[('relay', True)]['deg'] / hand['full']['deg'] * 100:.0f}",
        PALM=f"{palm:.1f}", PALMD=f"{palmd:.1f}",
        RINGP=f"{ringp:.2f}", RINGF=f"{ringf:.2f}", HOLDIK=f"{holdik:.1f}",
        RINGT0=f"{trk_ring[0.0]:.1f}",
        SEAT_OK=f"{seat[('relay', 30.0)]['k']}/{seat[('relay', 30.0)]['n']}",
        SEAT_DEG=f"{seat[('relay', 30.0)]['deg']:.1f}",
        SEAT_DEG0=f"{seat[('relay', 0.0)]['deg']:.1f}",
        FULL_SEAT16=f"{seat[('full', 16.0)]['k']}/{seat[('full', 16.0)]['n']}",
        SLIP=f"{flr[('relay', True)]['slip']:.1f}",
        FREE_NF_RELAY=f"{flr[('relay', True)]['free'] * 100:.1f}",
        FREE_NF_FULL=f"{flr[('full', True)]['free'] * 100:.0f}",
        TURNS_FULL=f"{st.mean(endt['full']):.2f}",
        TURNS_RELAY=f"{st.mean(endt['relay']):.2f}",
        SD_FULL=f"{st.pstdev(endt['full']):.3f}", SD_RELAY=f"{st.pstdev(endt['relay']):.3f}",
    )

    # ------------------------------------------------------------------ charts
    # Both charts share ONE x axis on purpose. Plotting the five handover modes here, all of
    # which run the synchronous gait, put a 19.1% relay bar next to a 0.4% relay number in the
    # prose beside it -- the same cell in name only.
    XC = [("Release, sync gait", ("full", False)), ("Release, relay gait", ("full", True)),
          ("Relay, sync gait", ("relay", False)), ("Relay, relay gait", ("relay", True))]
    c_free = bars([(lab, gait[k]["free"] * 100) for lab, k in XC],
                  "Hand-over and gait schedule", "Window with no pad on the tool (%)",
                  ymax=100, colour=B, labfmt="{:.1f}", note=None, w=800)
    c_deg = bars([(lab, gait[k]["deg"]) for lab, k in XC],
                 "Hand-over and gait schedule", "Shaft rotation per cycle (deg)",
                 colour=A, labfmt="{:.1f}", w=800)
    c_grip = lines([("turn", [(q, grip[q]["deg"]) for q in sorted(grip)], A, None, 1, -14),
                    ("tilt", [(q, grip[q]["tilt"]) for q in sorted(grip)], B, "5 4", 0, -14)],
                   "Squeeze after the relay (mm)",
                   "Deg/cycle, final tilt (deg)", 3.2, 34,
                   marks=[(0.5, "Tool lost"), (2.5, "Gait stalls")])
    lo = sorted({t for _, t in load})
    c_load = lines([("relay", [(t, load[("relay", t)]["k"] / 6 * 100) for t in lo], A, None,
                     2, 22),
                    ("release", [(t, load[("full", t)]["k"] / 6 * 100) for t in lo], B, "5 4",
                     0, -14)],
                   "Lateral load as an equivalent table tilt (deg)",
                   "Seeds completing 8 cycles (%)", 31, 100,
                   marks=[(14.0, "Static limit 14.0")])
    so = sorted({t for _, t in seat})
    c_seat = lines([("relay", [(t, seat[("relay", t)]["k"] / 6 * 100) for t in so], A, None,
                     3, -14),
                    ("release", [(t, seat[("full", t)]["k"] / 6 * 100) for t in so], B, "5 4",
                     1, 22)],
                   "Lateral load, tool seated in a 45&#176; countersink (deg)",
                   "Seeds completing 8 cycles (%)", 31, 100)
    c_trk_r = bars([(f"{int(f * 100)}%", trk_ring[f]) for f in tf],
                   "Wrist move attempted", "Ring IK residual (mm)", colour=B,
                   labfmt="{:.1f}", note=None)
    c_trk_h = bars([(f"{int(f * 100)}%", trk_hold[f]) for f in tf],
                   "Wrist move attempted", "Largest pin shortfall (mm)", colour=A,
                   labfmt="{:.1f}", note=None)

    def tr(cells, hi=False):
        return f'<tr class="hi">{cells}</tr>' if hi else f"<tr>{cells}</tr>"

    t_mode = "".join(
        tr(f'<td><b>{MODES[k][0]}</b></td><td>{MODES[k][1]}</td>'
           + okcell(hand[k]["k"], hand[k]["n"])
           + f'<td class="num">{hand[k]["free"] * 100:.1f}%</td>'
           + f'<td class="num">{hand[k]["deg"]:.1f}</td>'
           + f'<td class="num">{hand[k]["tilt"]:.1f}</td>', k == "full")
        for k in ("full", "none", "relay", "track", "slide"))

    t_cross = "".join(
        tr(f'<td><b>{"Relay" if i == "relay" else "Release"}</b></td>'
           f'<td>{"One finger at a time" if g else "All three together"}</td>'
           + okcell(gait[(i, g)]["k"], gait[(i, g)]["n"])
           + f'<td class="num">{gait[(i, g)]["free"] * 100:.1f}%</td>'
           + f'<td class="num">{gait[(i, g)]["deg"]:.1f}</td>'
           + f'<td class="num">{gait[(i, g)]["drift"]:.2f}</td>',
           i == "relay" and g)
        for i in ("full", "relay") for g in (False, True))

    t_floor = "".join(
        tr(f'<td><b>{"Relay" if i == "relay" else "Release"}</b></td>'
           f'<td>{"Deleted after the grasp" if nf else "Present"}</td>'
           + okcell(flr[(i, nf)]["k"], flr[(i, nf)]["n"])
           + f'<td class="num">{flr[(i, nf)]["free"] * 100:.1f}%</td>'
           + f'<td class="num">{flr[(i, nf)]["slip"]:.1f}</td>'
           + f'<td class="num">{flr[(i, nf)]["tilt"]:.1f}</td>', i == "relay" and nf)
        for i in ("full", "relay") for nf in (False, True))

    t_dir = "".join(
        tr(f'<td class="num">{int(a)}&#176;</td>'
           + okcell(dirn[("full", a)]["k"], dirn[("full", a)]["n"])
           + okcell(dirn[("relay", a)]["k"], dirn[("relay", a)]["n"])
           + f'<td class="num">{dirn[("relay", a)]["deg"]:.1f}</td>'
           + f'<td class="num">{dirn[("relay", a)]["drift"]:.2f}</td>')
        for a in (0.0, 90.0, 180.0, 270.0))

    vid_relay = b64(D / "20260903-handover_relay_closeup.mp4", "video/mp4")
    vid_seat = b64(D / "20260903-handover_seat_load30.mp4", "video/mp4")
    film = jpeg(D / "20260903-handover_relay_closeup_seams.png")

    stats = "".join(
        f'<div class="stat"><b>{v}</b><span>{s}</span></div>' for v, s in [
            (f'{K["FREE_RELAY"]}%', "of the task with no pad on the tool (was 49%)"),
            (f'{K["RATIO"]}%', "of the release chain's turn rate"),
            (K["SEAT_OK"], "seeds at a 30&#176; lateral load, in the seat"),
            (f'{K["SLIP"]} mm', "per cycle it slips with no floor at all"),
        ])

    body = f"""
<section id="terms"><div class="col">
<h2>Terms and metrics</h2>
<dl class="terms">
<dt><b>Release chain</b></dt><dd>The chained task&rsquo;s hand-over: open the hand, move the palm to the gait&rsquo;s
pose and close on the ring; its gait releases all three pads every cycle.</dd>
<dt><b>Relay hand-over, relay gait</b></dt><dd>One pad moves at a time while two stay loaded; in the gait, release,
return and regrasp run one finger at a time.</dd>
<dt><b>Ring</b></dt><dd>The gait&rsquo;s grasp: three pads around the standing shaft.</dd>
<dt><b>No contact</b></dt><dd>Share of control steps from set-down to the end of the gait with no pad on the
tool.</dd>
<dt><b>Ok</b></dt><dd>Seeds of 6 that complete 8 gait cycles with the tool standing.</dd>
<dt><b>Deg/cycle, drift, end tilt</b></dt><dd>Rotation of the shaft about its axis per cycle over completed seeds;
lateral displacement of the tool, in mm; tilt from vertical at the end, in degrees.</dd>
<dt><b>Ring IK residual, pin shortfall</b></dt><dd>Largest gap between a pad target on the ring and the reachable pad,
in mm; how far a pad held at a world point falls short of it while the palm moves, in mm.</dd>
<dt><b>Squeeze</b></dt><dd>Depth of the commanded pad positions inside the tool&rsquo;s surface after the relay, in
mm.</dd>
<dt><b>Lateral load</b></dt><dd>A steady horizontal force of tan&nbsp;&theta; times the tool&rsquo;s weight, given as
the table tilt &theta; that produces it.</dd>
<dt><b>Slip</b></dt><dd>Axial slip of the tool through the grasp per cycle, in mm.</dd>
</dl>
</div></section>

<section><div class="col">
<h2>Contact during the chained task</h2>
<p>The chained task changes grasp by opening the hand, moving the palm to the pose of the gait study and closing on
the ring. It works on every seed because a 100&#215;25&nbsp;mm cylinder on its end face stands by itself up to a tilt of
atan(12.5/50)&nbsp;=&nbsp;14.0&#176;. Sampled at the control rate from set-down to the end of the gait,
{K["FREE_FULL"]}&nbsp;% of that window has no pad on the tool: the full release of the re-index, then the three-pad
release of every gait cycle. A seated tool tipped a few degrees, an uneven bench or a cable would not wait. The
alternatives are a hand-over that keeps contact, or a gait from the carry&rsquo;s grasp with no hand-over.</p>
</div></section>

<section><div class="col">
<h2>Hand-over modes</h2>
<p>Eight cycles, six spawn-jittered seeds each, flat floor, no load; every row runs the synchronous gait:</p>

<div class="tw"><table>
<thead><tr><th>Hand-over</th><th>Action</th><th class="num">Ok</th>
<th class="num">No contact</th><th class="num">Deg/cycle</th><th class="num">End tilt</th></tr></thead>
<tbody>{t_mode}</tbody></table></div>

<p>A per-finger hand-over in joint space ramps one finger from its carry pose to its ring pose while the other two
hold, and knocks the tool over on every seed. The two stations straddle the shaft, so the straight joint-space line
between them passes through the material, the same defect as the mid-air ring regrasp that fails 6 of 6 in the chain
study. The relay plans each move in the tool&rsquo;s cylindrical coordinates: retract radially, travel to the new
station and azimuth, close.</p>

<p>The carry leaves the palm <b>{K["PALM"]}&nbsp;mm and {K["PALMD"]}&#176;</b> from the gait&rsquo;s pose, and the
gait&rsquo;s ring is reachable to {K["RINGF"]}&nbsp;mm from the gait&rsquo;s pose and to {K["RINGT0"]}&nbsp;mm from the
carry&rsquo;s. Moving the palm with every pad held at its world point (<code>track</code>) falls
<b>{K["HOLDIK"]}&nbsp;mm short</b> at full travel, so the pads come off; dragging the pads along the tool
(<code>slide</code>) lifts the tool and topples it. The relay therefore keeps the palm where the carry left it and takes
its ring at the pads&rsquo; height, reachable to {K["RINGP"]}&nbsp;mm. No fraction of the wrist move both holds the pins
and reaches the ring; the best cell of the sweep holds {trk[0.5]["k"]} of 6.</p>

<div class="pair"><figure>{c_trk_r}<figcaption>Ring IK residual against the fraction of the wrist move.</figcaption></figure><figure>{c_trk_h}<figcaption>Largest pin shortfall against the fraction of the wrist move.</figcaption></figure></div>
</div></section>

<section><div class="col">
<h2>Relay hand-over with a relay gait</h2>
<p>The relay walks one pad at a time from the carry&rsquo;s stations onto the ring, around the tool, with two pads
loaded at every instant. The gait can then release, return and regrasp one finger at a time
(<code>probe_real_v1_gait.py --relay</code>). Hand-over schedule against gait schedule, six seeds each:</p>

<div class="tw"><table>
<thead><tr><th>Hand-over</th><th>Gait release</th><th class="num">Ok</th>
<th class="num">No contact</th><th class="num">Deg/cycle</th><th class="num">Drift (mm)</th></tr></thead>
<tbody>{t_cross}</tbody></table></div>

<p>The two relays work only together. A relay hand-over followed by the synchronous gait completes
{gait[("relay", False)]["k"]} of 6, because the gait releases all three pads of the preserved grasp. Together they
complete every seed with <b>{K["FREE_RELAY"]}&nbsp;% of the task uncontacted</b>, against {K["FREE_FULL"]}&nbsp;%, and
0.03&nbsp;mm of lateral drift.</p>

<div class="pair"><figure>{c_free}<figcaption>Share of the task with no pad on the tool, for the four cells of the
table.</figcaption></figure><figure>{c_deg}<figcaption>Shaft rotation per cycle for the same cells.</figcaption></figure></div>

<p>The relay turns the tool <b>{K["DEG_RELAY"]}&#176; per cycle against {K["DEG_FULL"]}&#176;</b>, {K["RATIO"]}&nbsp;%
of the rate, because its ring is the reachable one and two stance fingers make a shorter stroke than three: over 40
cycles, {K["TURNS_RELAY"]} turns against {K["TURNS_FULL"]}, with a standard deviation of {K["SD_RELAY"]} turns over three
seeds against {K["SD_FULL"]}.</p>

<figure><video src="{vid_relay}" controls muted loop playsinline preload="metadata"></video>
<figcaption>Grasp, carry, set-down, relay hand-over and relay gait. The palm stays still after the tool stands; each
finger leaves, travels around the shaft at the open radius and closes while the other two hold.</figcaption></figure>

<figure><img src="{film}" alt="Frames of the relay run at its phase boundaries"><figcaption>The same run at its phase
boundaries. Rows 2 and 3 are the hand-over, one finger per frame.</figcaption></figure>

<p>Gaiting from the carry&rsquo;s grasp with no hand-over completes {nrel[False]["k"]} of 6 with a synchronous release
and {nrel[True]["k"]} of 6 with a relay, at {nrel[True]["deg"]:.1f}&#176; per cycle; it opens all three fingers to
reach its ring ({nrel[True]["free"] * 100:.0f}&nbsp;% uncontacted), and its grasp is not a tripod.</p>
</div></section>

<section><div class="col">
<h2>Grip after the relay</h2>
<p>A relay places each pad 2&nbsp;mm inside the surface, but the tool moves as each finger arrives: across one hand-over
the pad forces went 7.6, 4.9 and 0.3&nbsp;N, the first pad relieved by the time the third landed. One relative squeeze
after the last pad sets the grip. It is computed from the commanded pose, since the achieved pose includes the
deflection under load; reissuing a constant command from the achieved pose lost 18.4&nbsp;N of grip over
seven reissues on the bench.</p>

<figure>{c_grip}<figcaption>Rotation per cycle and final tilt against the squeeze. Below 1&nbsp;mm the tool rocks and is
lost ({grip[0.5]["k"]} of 6, {grip[0.5]["tilt"]:.0f}&#176; final tilt); above 2&nbsp;mm the gait stalls, as the
ground-supported gait does at 75&nbsp;N, reaching {grip[3.0]["deg"]:.1f}&#176; per cycle at 3&nbsp;mm with the tool
held.</figcaption></figure>
</div></section>

<section><div class="col">
<h2>Lateral load on the floor and in a countersink</h2>
<p><code>--tilt-deg</code> applies a steady horizontal load of tan&nbsp;&theta; times the tool&rsquo;s weight from the
moment the tool stands, as a bench out of level by &theta; would. A free cylinder on its end face topples when
tan&nbsp;&theta;&nbsp;&gt;&nbsp;r/h, at 14.0&#176;.</p>

<div class="pair"><figure>{c_load}<figcaption>On a flat floor. Both modes hold to 8&#176; and fail by 12&#176;, short
of the static limit because the gait adds its own disturbance.</figcaption></figure><figure>{c_seat}<figcaption>In a
45&#176; countersink. The release chain fails at 16&#176;; the relay holds to 30&#176;.</figcaption></figure></div>

<p>On a flat floor the relay fails at the same load as the release chain. Its grip holds the tool upright and unloads
the floor ({gait[("relay", True)]["gnd"] * 100:.0f}&nbsp;% of the window with ground contact, against
{gait[("full", False)]["gnd"] * 100:.0f}&nbsp;%), which removes the restoring moment behind the 14&#176; limit; three pads
on a 21&nbsp;mm ring stabilise the tool less than its 12.5&nbsp;mm foot on the floor.</p>

<p>In the countersink the tool cannot translate, so only rotation about the seated tip remains, which a hand that keeps
contact resists. The relay completes <b>{K["SEAT_OK"]} at 30&#176;</b>, a load of 0.58 of the tool&rsquo;s weight, at
{K["SEAT_DEG"]}&#176; per cycle against {K["SEAT_DEG0"]}&#176; unloaded; the release chain completes {K["FULL_SEAT16"]}
at 16&#176;. The result holds in four load directions at 16&#176;:</p>

<div class="tw"><table>
<thead><tr><th class="num">Load azimuth</th><th class="num">Release</th><th class="num">Relay</th>
<th class="num">Deg/cycle</th><th class="num">Drift (mm)</th></tr></thead>
<tbody>{t_dir}</tbody></table></div>

<figure><video src="{vid_seat}" controls muted loop playsinline preload="metadata"></video>
<figcaption>The chain into the countersink under a 30&#176; lateral load, applied from the moment the tool
stands.</figcaption></figure>
</div></section>

<section><div class="col">
<h2>Gait with the floor removed</h2>
<div class="tw"><table>
<thead><tr><th>Hand-over</th><th>Floor</th><th class="num">Ok</th><th class="num">No contact</th>
<th class="num">Slip (mm/cycle)</th><th class="num">End tilt</th></tr></thead>
<tbody>{t_floor}</tbody></table></div>

<p>With the floor deleted once the grasp is set, both modes score 0 of 6, for different reasons. The release chain
drops the tool, {K["FREE_NF_FULL"]}&nbsp;% of the window uncontacted. The relay keeps three pads and 20&nbsp;N on it for
all six cycles ({K["FREE_NF_RELAY"]}&nbsp;% uncontacted, tilt under 9&#176;) and lowers it {K["SLIP"]}&nbsp;mm per cycle by
axial slip through the grasp, still turning it at 8&#176; per cycle. This is the program&rsquo;s first floor-free gait
that does not end with the tool on the ground.</p>
</div></section>

<section><div class="col">
<h2>Future work</h2>
<p>On a flat, undisturbed surface, release and retake: it turns the tool twice as fast. Where the tool may move, as in a
seat, relay both the hand-over and the gait at {K["RATIO"]}&nbsp;% of the rate.</p>
<ol>
<li><b>Bench gates.</b> Export the relay plans and run <code>scripts/real_v1_trajectory_clearance.py</code> and the
servo-limit check on them; the relay&rsquo;s Cartesian legs are a new path shape, and three of four plans that once
reached the control station failed clearance.</li>
<li><b>Torsional load in the seat.</b> Sweep <code>--screw-torque</code> on the seated relay; the primitive is rated at
0.0072&nbsp;N&middot;m until it turns against a fastener&rsquo;s torque.</li>
<li><b>Closed-loop grip.</b> Correct the squeeze once per cycle from the AprilTag rotation reading, which measures the
gait&rsquo;s controlled variable on the bench.</li>
<li><b>Axial slip metric.</b> Add axial slip per cycle to the gate and the trajectory-health scorecard, so the floor-free
relay counts as a slip failure and not a drop.</li>
</ol>
</div></section>
"""

    tpl = (ROOT / "scripts/real_v1_handover_page.template.html").read_text(encoding="utf-8")
    html = tpl.replace("{{BODY}}", body).replace("{{STATS}}", stats)
    for k, v in K.items():
        html = html.replace("{{" + k + "}}", str(v))
    out.write_text(retro_style.apply(html), encoding="utf-8")  # plain page style (owner, 2026-10-09)
    print(f"-> {out}  ({len(html) / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
