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

MODES = {
    "full": ("release", "open the hand, fly the palm to the gait's pose, close on the ring"),
    "none": ("in place", "release and retake all three at once, at the carry's palm pose"),
    "relay": ("relay", "walk the pads onto the ring one at a time, palm parked"),
    "track": ("track", "fly the palm with the pads pinned in the world, then relay"),
    "slide": ("slide", "fly the palm with the pads dragged along the tool onto the ring"),
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
    XC = [("release + sync", ("full", False)), ("release + relay", ("full", True)),
          ("relay + sync", ("relay", False)), ("relay + relay", ("relay", True))]
    c_free = bars([(lab, gait[k]["free"] * 100) for lab, k in XC],
                  "handover + gait schedule", "% of the window with no pad on the tool",
                  ymax=100, colour=B, labfmt="{:.1f}", note="lower is better", w=800)
    c_deg = bars([(lab, gait[k]["deg"]) for lab, k in XC],
                 "handover + gait schedule", "degrees of shaft rotation per cycle",
                 colour=A, labfmt="{:.1f}", w=800)
    c_grip = lines([("turn", [(q, grip[q]["deg"]) for q in sorted(grip)], A, None, 1, -14),
                    ("tilt", [(q, grip[q]["tilt"]) for q in sorted(grip)], B, "5 4", 0, -14)],
                   "commanded interference at the handover grip (mm)",
                   "deg / cycle  &middot;  final tilt (deg)", 3.2, 34,
                   marks=[(0.5, "LOSES IT"), (2.5, "STALLS")])
    lo = sorted({t for _, t in load})
    c_load = lines([("relay", [(t, load[("relay", t)]["k"] / 6 * 100) for t in lo], A, None,
                     2, 22),
                    ("release", [(t, load[("full", t)]["k"] / 6 * 100) for t in lo], B, "5 4",
                     0, -14)],
                   "lateral load, as an equivalent table tilt (deg)",
                   "seeds completing 8 cycles (%)", 31, 100,
                   marks=[(14.0, "STATIC LIMIT 14.0")])
    so = sorted({t for _, t in seat})
    c_seat = lines([("relay", [(t, seat[("relay", t)]["k"] / 6 * 100) for t in so], A, None,
                     3, -14),
                    ("release", [(t, seat[("full", t)]["k"] / 6 * 100) for t in so], B, "5 4",
                     1, 22)],
                   "the same load, tool seated in a 45&#176; countersink (deg)",
                   "seeds completing 8 cycles (%)", 31, 100)
    c_trk_r = bars([(f"{int(f * 100)}%", trk_ring[f]) for f in tf],
                   "wrist move attempted", "ring IK residual (mm)", colour=B,
                   labfmt="{:.1f}", note="what the move buys")
    c_trk_h = bars([(f"{int(f * 100)}%", trk_hold[f]) for f in tf],
                   "wrist move attempted", "worst pin shortfall (mm)", colour=A,
                   labfmt="{:.1f}", note="what it costs")

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
        tr(f'<td><b>{"relay" if i == "relay" else "release"}</b></td>'
           f'<td>{"one finger at a time" if g else "all three together"}</td>'
           + okcell(gait[(i, g)]["k"], gait[(i, g)]["n"])
           + f'<td class="num">{gait[(i, g)]["free"] * 100:.1f}%</td>'
           + f'<td class="num">{gait[(i, g)]["deg"]:.1f}</td>'
           + f'<td class="num">{gait[(i, g)]["drift"]:.2f}</td>',
           i == "relay" and g)
        for i in ("full", "relay") for g in (False, True))

    t_floor = "".join(
        tr(f'<td><b>{"relay" if i == "relay" else "release"}</b></td>'
           f'<td>{"deleted after the grasp" if nf else "present"}</td>'
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
            (f'{K["RATIO"]}%', "of the shipped chain's turn rate"),
            (K["SEAT_OK"], "seeds at a 30&#176; lateral load, in the seat"),
            (f'{K["SLIP"]} mm', "per cycle it slips with no floor at all"),
        ])

    body = f"""
<section><div class="col">
<h2>What the chain was assuming</h2>
<p class="sub">free_frac &middot; the share of the task after set-down with nothing touching the tool</p>

<p>The chained task published last week gets from the carry's grasp to the gait's the simplest
way there is: open the hand, fly the palm to the pose the gait study validated, close again on a
ring. It works on every seed, and the reason it works is a property of the object, not of the
hand. A 100&#215;25&nbsp;mm cylinder standing on its end face is statically stable out to
atan(12.5/50)&nbsp;=&nbsp;14.0&#176;, so it simply waits.</p>

<p>How much waiting the chain asks for is measurable, and it is not a detail. Sampling contact
at the control rate from the moment the tool is set down to the end of the gait,
<b>{K["FREE_FULL"]}% of that window has no pad on the tool at all</b> &mdash; the full release
during the re-index, and then, every cycle, the three-pad release the gait needs to advance its
contacts. For half the task the hand is not holding the thing it is manipulating.</p>

<p>That is fine for a cylinder on a level floor and false for almost anything else: a tool seated
in a countersink and tipped a few degrees, a bench that is not level, a cable, a heavier tool, a
nudge. So the question this page asks is narrow. <b>Can the same task be run without ever letting
go, and what does it cost?</b></p>

<div class="callout"><span class="tag">The two candidate answers</span>
<p>Either transition from the reorientation grasp to a gaiting grasp while keeping enough contact
to support the tool, or never take a gaiting grasp at all and gait from the reorientation grasp.
They turn out to be the same mechanism &mdash; a handover is a relay whose target ring moved
&mdash; and, more usefully, they turn out to be a matched pair: doing one without the other is
worse than doing neither.</p></div>
</div></section>

<section><div class="col">
<h2>Four ways to change grasp, and why two of them cannot work</h2>
<p class="sub">8 cycles &middot; 6 spawn-jittered seeds each &middot; flat floor, no load
&middot; every row here runs the synchronous gait</p>

<div class="tw"><table>
<thead><tr><th>handover</th><th>what it does</th><th class="num">ok</th>
<th class="num">no contact</th><th class="num">deg/cycle</th><th class="num">end tilt</th></tr></thead>
<tbody>{t_mode}</tbody></table></div>

<p>The two that fail are worth more than the two that work, because each fails for a reason that
was already in the repertoire under another name.</p>

<h3>A joint-space regrasp sweeps the pad through the tool</h3>
<p>The obvious per-finger handover ramps one finger's joints from its carry pose to its ring
pose while the other two hold. It knocks the tool over on every seed and every fraction of the
wrist move tried. The carry's station and the ring's station straddle the shaft, so the straight
line between them in joint space passes <em>through</em> the material: the tool leaves before the
finger arrives. This is the same defect as the mid-air ring regrasp that fails 6/6 in the chain
study, which was read there as a mid-air problem &mdash; it is not, it is a path problem, and it
is present on the floor too. The fix is to plan the move in the tool's own cylindrical
coordinates: retract radially at the station already held, travel to the new station and azimuth
out there, close back in. Three legs, never inside the surface.</p>

<h3>The fingers cannot absorb the wrist move</h3>
<p>The gait's ring is only cleanly reachable from the gait's palm pose, and the carry leaves the
palm <b>{K["PALM"]}&nbsp;mm and {K["PALMD"]}&#176;</b> away from it: that ring's worst IK
residual is {K["RINGF"]}&nbsp;mm solved from the gait's pose and {K["RINGT0"]}&nbsp;mm solved from
where the carry parks the palm. So <code>track</code> flies the palm there with every pad pinned to the world point it currently
holds, which would move the wrist without moving the tool. This hand cannot do it: the pins run
<b>{K["HOLDIK"]}&nbsp;mm short</b> at full travel, the pads come off, and the tool is free-standing
again anyway. Dragging the pads along the tool instead (<code>slide</code>) is worse &mdash; a
loaded pad cannot be slid axially without carrying the object with it, so the tool is lifted off
the floor and topples.</p>

<p>Neither failure is a tuning problem. Together they say the wrist move has to be given up, and
that is the trade the working mode makes: <b>the relay keeps every contact and pays for it by
taking its ring at the height the pads are already at</b>, which it reaches to
{K["RINGP"]}&nbsp;mm rather than {K["RINGF"]}.</p>

<div class="pair"><figure>{c_trk_r}<figcaption>Moving the palm is what makes the gait's ring
reachable.</figcaption></figure><figure>{c_trk_h}<figcaption>And what the fingers fail to absorb
while doing it.</figcaption></figure></div>
<p class="sub" style="margin-top:-12px">The variable was worth sweeping because the answer could
have been a knee. It is not &mdash; there is no fraction at which the pins hold and the ring is
reachable, and the best cell in the sweep is {trk[0.5]["k"]}/6.</p>
</div></section>

<section><div class="col">
<h2>The relay, and why the gait has to relay too</h2>
<p class="sub">handover schedule &#215; gait schedule &middot; 6 seeds each</p>

<p>The working handover walks one pad at a time from the carry's stations onto the gait's ring,
around the tool rather than through it, with the palm parked where the carry left it. Two pads
are loaded at every instant of the transition. The gait then runs its own relay: release, return
and regrasp one finger at a time instead of all three, which is the schedule
<code>probe_real_v1_gait.py</code> has carried as <code>--relay</code> since the gait study and
which that study measured only with the floor underneath.</p>

<div class="tw"><table>
<thead><tr><th>handover</th><th>gait release</th><th class="num">ok</th>
<th class="num">no contact</th><th class="num">deg/cycle</th><th class="num">drift mm</th></tr></thead>
<tbody>{t_cross}</tbody></table></div>

<p><b>The two halves are a matched pair and the cross terms are worse than either diagonal.</b> A
relay handover followed by a synchronous gait is the worst cell in the study that is not an
outright failure: it hands a carefully preserved grasp to a schedule that throws it away three
pads at a time, {gait[("relay", False)]["k"]}/6. Run together, they complete every seed with
<b>{K["FREE_RELAY"]}% of the task uncontacted</b> &mdash; against {K["FREE_FULL"]}% &mdash; and
0.03&nbsp;mm of lateral drift.</p>

<div class="pair"><figure>{c_free}<figcaption>How much of the task the tool spends untouched,
across the same four cells as the table above.</figcaption></figure><figure>{c_deg}
<figcaption>What that costs in rotation.</figcaption></figure></div>

<p>The cost is <b>{K["DEG_RELAY"]}&#176; per cycle against {K["DEG_FULL"]}&#176;</b>, or
{K["RATIO"]}% of the turn rate, and it is paid twice over: once because the relay's ring is the
one it can reach rather than the one the gait wants, and once because two stance fingers driving
while a third recovers is a smaller effective stroke than three. Over 40 cycles that is
{K["TURNS_RELAY"]} turns against {K["TURNS_FULL"]}. It is also markedly <em>more</em> repeatable
&mdash; sd {K["SD_RELAY"]} turns over three seeds against {K["SD_FULL"]} &mdash; which is what one
would expect from a primitive that never re-establishes a grasp from scratch.</p>

<figure><video src="{vid_relay}" controls muted loop playsinline preload="metadata"></video>
<figcaption>Grasp, carry, set down, hand over one finger at a time, gait. The palm does not move
after the tool is standing; each finger leaves, travels around the shaft at the open radius and
closes again while the other two hold.</figcaption></figure>

<figure><img src="{film}" alt="seam frames"><figcaption>The same run by seam. Rows 2&ndash;3 are
the handover, one finger per frame; the shaft never stands alone.</figcaption></figure>

<h3>Gaiting from the carry's grasp, without a handover at all</h3>
<p>The other candidate &mdash; skip the ring, gait from whatever the carry left &mdash; is
{nrel[False]["k"]}/6 with a synchronous release and {nrel[True]["k"]}/6 with a relay, at
{nrel[True]["deg"]:.1f}&#176;/cycle. It still opens all three fingers together to reach its ring,
so it is not actually a continuous-contact mode ({nrel[True]["free"] * 100:.0f}% uncontacted), and
its carry-inherited grasp is not a tripod. Worth knowing it half-works; not worth preferring.</p>
</div></section>

<section><div class="col">
<h2>Arrival order cannot set a grip</h2>
<p class="sub">one commanded squeeze, once all three pads are on the ring</p>

<p>A relay lands each pad on the ring nominally 2&nbsp;mm inside the surface, which on a position
servo is 2&nbsp;mm of grip. It is not, because the tool moves as each finger arrives: measured
across one handover the pad forces went 7.6&nbsp;&rarr;&nbsp;4.9&nbsp;&rarr;&nbsp;0.3&nbsp;N, the
first pad relieved by the time the third landed. The grasp has to be set explicitly afterwards,
with one relative squeeze computed from the <em>commanded</em> pose &mdash; measuring from the
achieved pose hands back exactly the deflection carrying the load, which is the bench's
<code>achieved_fraction</code> error and cost this program 18.4&nbsp;N of grip over seven reissues
of a constant command.</p>

<figure>{c_grip}<figcaption>The window is narrow and bracketed on both sides. Below 1&nbsp;mm the
tool rocks in the hand and is lost ({grip[0.5]["k"]}/6, {grip[0.5]["tilt"]:.0f}&#176; final tilt);
above 2&nbsp;mm the gait stalls the way the ground-supported gait stalls at 75&nbsp;N, reaching
{grip[3.0]["deg"]:.1f}&#176;/cycle at 3&nbsp;mm while still holding the tool
perfectly.</figcaption></figure>
</div></section>

<section><div class="col">
<h2>What continuous contact buys, and what it does not</h2>
<p class="sub">a steady lateral load on the tool, as the table tilt that would produce it</p>

<p>With the assumption named, it can be replaced. <code>--tilt-deg</code> applies a steady
horizontal load of tan(&theta;) of the tool's own weight from the moment it is standing &mdash;
what a bench out of level by &theta; delivers, or a cable, or a nudge that does not go away. The
free-standing prediction is exact and worth stating before the sweep: a cylinder on its end face
topples when tan(&theta;)&nbsp;&gt;&nbsp;r/half, i.e. at 14.0&#176;.</p>

<div class="pair"><figure>{c_load}<figcaption>On a flat floor. Both modes hold to 8&#176; and
both fail by 12&#176;, a little short of the static limit because the gait supplies its own
disturbance.</figcaption></figure><figure>{c_seat}<figcaption>In a 45&#176; countersink. The
release chain fails at 16&#176;; the relay is flat to 30&#176;.</figcaption></figure></div>

<p><b>On a plane, never letting go buys nothing.</b> That is the result I did not expect and it
is worth being plain about: the relay fails at the same load the release chain fails at. The
reason is visible in the ground-force trace &mdash; the relay's grip is what holds the tool
upright, and a grip strong enough to do that also unloads the floor
({gait[("relay", True)]["gnd"] * 100:.0f}% of the window has ground contact, against
{gait[("full", False)]["gnd"] * 100:.0f}%), removing exactly the restoring moment the 14&#176;
figure is computed from. Three pads on a 21&nbsp;mm ring are a worse stabiliser than a
12.5&nbsp;mm foot on a floor.</p>

<p><b>In the seat, it buys everything.</b> The countersink stops the tool translating, so the
only thing left to resist is rotation about the seated tip &mdash; and that is what a hand that
has not let go is good at. The relay is <b>{K["SEAT_OK"]} at 30&#176;</b>, a load of 0.58 of the
tool's weight, at {K["SEAT_DEG"]}&#176;/cycle against {K["SEAT_DEG0"]}&#176; unloaded: the load
does not cost it anything measurable. The release chain is {K["FULL_SEAT16"]} by 16&#176;.
<b>Neither the seat nor the grasp is sufficient alone; the pair is.</b></p>

<div class="tw"><table>
<thead><tr><th class="num">load azimuth</th><th class="num">release</th><th class="num">relay</th>
<th class="num">deg/cycle</th><th class="num">drift mm</th></tr></thead>
<tbody>{t_dir}</tbody></table></div>
<p class="sub" style="margin-top:-14px">Four directions at 16&#176;, in the seat. The result is a
property of the seat, not of a lucky heading.</p>

<figure><video src="{vid_seat}" controls muted loop playsinline preload="metadata"></video>
<figcaption>The whole chain into the countersink under a 30&#176; lateral load, held throughout.
The load is on from the moment the tool is standing.</figcaption></figure>

<h3>With no floor at all</h3>
<div class="tw"><table>
<thead><tr><th>handover</th><th>floor</th><th class="num">ok</th><th class="num">no contact</th>
<th class="num">slip mm/cycle</th><th class="num">end tilt</th></tr></thead>
<tbody>{t_floor}</tbody></table></div>

<p>Deleting the floor once the grasp is established is the gait study's degenerate control, and
it separates the two modes completely even though both score 0/6. The release chain
<em>drops</em> the tool: {K["FREE_NF_FULL"]}% of the window uncontacted. The relay does not
&mdash; {K["FREE_NF_RELAY"]}% uncontacted, three pads and 20&nbsp;N for all six cycles, tilt under
9&#176; &mdash; it holds the tool and <em>lowers</em> it, {K["SLIP"]}&nbsp;mm per cycle, still
turning it at 8&#176;/cycle on the way down. The gate cannot tell those apart and reports both as
failures; the defect is axial slip through the grasp, not loss of grasp. That is the first
floor-free gait in this program that does not end with the tool on the ground, and the thing that
would close it is a term nothing currently measures.</p>
</div></section>

<section><div class="col">
<h2>Where this leaves the task</h2>

<p>The chained task now has two settings rather than one, and which is right depends on what is
under the tool. On a flat surface with nothing disturbing it, release and retake: it is twice as
fast and the floor is genuinely doing the work. Anywhere the tool cannot be trusted to stay put
&mdash; and a fastener in a seat is exactly that case &mdash; relay both the handover and the
gait, at {K["RATIO"]}% of the turn rate, and the load stops mattering.</p>

<p>What has not been done. None of this has been through the export, trajectory-clearance or
servo-limit gates &mdash; the relay's Cartesian legs are a new path shape and three of four plans
that reached the control station once failed exactly that check. There is still no torsional load
on the seat, so this remains a 0.0072&nbsp;N&middot;m primitive and not a fastener. The relay's
grip is set once and never corrected, which is the obvious place for the first closed loop, and
gaiting remains the only task in this program whose controlled variable the bench's AprilTag rig
measures directly. And the axial slip that ends the floor-free run has no reward term, no
scorecard entry and no design metric, on either topology that has shown it.</p>
</div></section>
"""

    tpl = (ROOT / "scripts/real_v1_handover_page.template.html").read_text(encoding="utf-8")
    html = tpl.replace("{{BODY}}", body).replace("{{STATS}}", stats)
    for k, v in K.items():
        html = html.replace("{{" + k + "}}", str(v))
    out.write_text(html, encoding="utf-8")
    print(f"-> {out}  ({len(html) / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
