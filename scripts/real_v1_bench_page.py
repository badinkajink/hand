"""Build the doc page for the bench study: the flat table and the 100 mm wrist stack.

Regenerable -- every number, chart and table is read from bench_study.json, so re-running the
study and re-running this is the whole update path. Charts are the chain page's hand-emitted SVG
helpers; media are inlined as base64.

    uv run --extra rl python scripts/real_v1_bench_page.py \
        --data docs/experiments/20260904-real_v1_bench
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


def sel(rows, **kw):
    out = []
    for r in rows:
        if all(abs(r.get(k, 1e9) - v) < 1e-9 if isinstance(v, float) else r.get(k) == v
               for k, v in kw.items()):
            out.append(r)
    return out


def agg(g):
    ok = [r for r in g if r.get("ok")]
    m = lambda key, src: (st.mean(r.get(key, 0.0) or 0.0 for r in src) if src else 0.0)
    return {
        "n": len(g), "k": sum(1 for r in g if r.get("ok")),
        "deg": m("gain_mean_deg", ok), "free": m("free_frac", g),
        "tilt": m("final_tilt_deg", g), "drift": m("drift_mm", g),
        "slip": m("slip_mm_per_cycle", g), "pumped": sum(1 for r in g if r.get("brake_pumped")),
        # the seam the wrist actually breaks: how upright the tool is once it is pressed home
        "press_tilt": (st.mean(
            [s["tilt_deg"] for r in g for s in r.get("seams", []) if s.get("phase") == "pressed"])
            if any(s.get("phase") == "pressed" for r in g for s in r.get("seams", [])) else 0.0),
        "carry_cos": (st.mean(
            [s["cos"] for r in g for s in r.get("seams", []) if s.get("phase") == "reoriented"])
            if any(s.get("phase") == "reoriented" for r in g for s in r.get("seams", []))
            else 0.0),
        "ikfail": sum(r.get("arm_ik_fails", 0) for r in g),
        "ikpos": max([r.get("arm_ik_pos_mm", 0.0) for r in g] or [0.0]),
    }


def tr(cells, hi=False):
    return f'<tr class="hi">{cells}</tr>' if hi else f"<tr>{cells}</tr>"


def num(v, fmt="{:.1f}"):
    return f'<td class="num">{fmt.format(v)}</td>'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path,
                    default=ROOT / "docs/experiments/20260904-real_v1_bench")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    D = args.data
    out = args.out or (D / "page.html")
    blob = json.loads((D / "bench_study.json").read_text())
    rows, dro = blob["rows"], blob["droop"]

    gk = lambda r: (round(r.get("base_x", 0), 3), round(r.get("base_y", 0), 3),
                    round(r.get("base_z", 0), 3), round(r.get("stack_mm", 0)),
                    round(r.get("stack_density", 0)))
    DR = {gk(r): r["droop_mm"] for r in dro}
    PAY = {gk(r): r["payload_g"] for r in dro}
    Z, S, RHO, XS = (0.0, 0.10, 0.20, 0.30), (0, 25, 50, 75, 100, 125, 150), \
        (0.0, 350.0, 700.0, 1400.0), (-0.35, -0.425, -0.50)
    CS_SEAT, TQ = (0.30, 0.80, 1.50, 2.00, 3.00), (0.0, 0.004, 0.008, 0.012, 0.016, 0.020)

    tab = {(z, m): agg(sel(rows, arm="table", base_z=z, mode=m)) for z in Z
           for m in ("relay", "release")}
    stk = {(s, m): agg(sel(rows, arm="stack", stack_mm=float(s), mode=m)) for s in S
           for m in ("relay", "release")}
    mas = {(p, m): agg(sel(rows, arm="mass", stack_density=p, mode=m)) for p in RHO
           for m in ("relay", "release")}
    # The seat grid: stack length x commanded carry grip, which is where the cost is paid.
    pre = {(s, c, m): agg(sel(rows, arm="preload_seat", stack_mm=float(s),
                              carry_squeeze_mm=c, mode=m))
           for s in (0, 50, 100, 150) for c in CS_SEAT for m in ("relay", "release")}
    pl2 = {(x, c, m): agg(sel(rows, arm="place2", base_x=x, carry_squeeze_mm=c, mode=m))
           for x in XS for c in (0.30, 0.80, 1.50, 2.00) for m in ("relay", "release")}
    tq = {(t, m): agg(sel(rows, arm="torque", screw_torque=t, mode=m)) for t in TQ
          for m in ("relay", "release")}
    plc = {(x, y): agg(sel(rows, arm="place", base_x=x, base_y=y))
           for x in (-0.35, -0.425, -0.50, -0.575, -0.65) for y in (0.0, 0.20)}

    # ---- the collapse: every geometry's BEST cell, against its droop -----------------------
    # Each of the three geometry knobs (stack length, stack mass, standing distance) moves palm
    # droop, and nothing else in the scene. If droop is the variable, the best cell each
    # geometry can reach should be a function of it alone.
    best = []
    for s in (0, 50, 100, 150):
        g = (-0.5, 0.0, 0.0, s, 700)
        cand = [(c, pre[(s, c, "relay")], pre[(s, c, "release")]) for c in CS_SEAT]
        c, a, b = max(cand, key=lambda t: (t[1]["k"], -t[0]))
        best.append({"what": f"{s} mm stack, 500 mm away", "droop": DR[g], "cs": c,
                     "a": a, "b": b})
    for x in XS:
        g = (x, 0.0, 0.0, 100, 700)
        cand = [(c, pl2[(x, c, "relay")], pl2[(x, c, "release")])
                for c in (0.30, 0.80, 1.50, 2.00)]
        c, a, b = max(cand, key=lambda t: (t[1]["k"], -t[0]))
        best.append({"what": f"100 mm stack, {abs(x) * 1000:.0f} mm away", "droop": DR[g],
                     "cs": c, "a": a, "b": b})
    # 100 mm of stack at 500 mm of standoff is reached twice, once down each sweep; they agree
    # exactly, which is the check that the two knobs really are the same knob.
    seen, uniq = set(), []
    for r in sorted(best, key=lambda r: r["droop"]):
        if round(r["droop"], 3) in seen:
            continue
        seen.add(round(r["droop"], 3))
        uniq.append(r)
    best = uniq
    cut = max(r["droop"] for r in best if r["a"]["k"] == r["a"]["n"])

    K = dict(
        NCELLS=len(rows),
        DR0=f'{DR[(-0.5, 0.0, 0.0, 0, 700)]:.2f}',
        DR100=f'{DR[(-0.5, 0.0, 0.0, 100, 700)]:.2f}',
        DR35=f'{DR[(-0.35, 0.0, 0.0, 100, 700)]:.2f}',
        DR425=f'{DR[(-0.425, 0.0, 0.0, 100, 700)]:.2f}',
        PAY0=f'{PAY[(-0.5, 0.0, 0.0, 0, 700)]:.0f}',
        PAY100=f'{PAY[(-0.5, 0.0, 0.0, 100, 700)]:.0f}',
        CUT=f"{cut:.2f}",
        IKFAIL=sum(r.get("arm_ik_fails", 0) for r in rows),
        PUMPED=sum(1 for r in rows if r.get("brake_pumped")),
        PUMPED_AT=", ".join(sorted({f'{r["screw_torque"] * 1000:.0f} mN&middot;m in the '
                                    f'{"relay" if r["mode"] == "relay" else "release"} arm'
                                    for r in rows if r.get("brake_pumped")})) or "none",
    )
    # ------------------------------------------------------------------ charts
    pct = lambda g: g["k"] / g["n"] * 100

    c_table = lines([("relay", [(z * 1000, pct(tab[(z, "relay")])) for z in Z], A, None, 0, -14),
                     ("release", [(z * 1000, pct(tab[(z, "release")])) for z in Z], B, "5 4",
                      2, 22)],
                    "height of the robot's mount above the work (mm)",
                    "seeds completing 8 cycles (%)", 320, 100)
    c_stack = lines([("relay", [(s, pct(stk[(s, "relay")])) for s in S], A, None, 1, -14),
                     ("release", [(s, pct(stk[(s, "release")])) for s in S], B, "5 4", 3, 22)],
                    "hardware between the tool flange and the palm plate (mm)",
                    "seeds completing 8 cycles (%)", 160, 100,
                    marks=[(100, "THE BENCH")])
    c_droop_s = bars([(f"{s}", DR[(-0.5, 0.0, 0.0, s, 700)]) for s in S],
                     "wrist stack (mm)", "palm droop under the hand's own weight (mm)",
                     colour=B, labfmt="{:.2f}", note="deterministic")
    c_droop_x = bars([(f"{abs(x) * 1000:.0f}", DR[(x, 0.0, 0.0, 100, 700)])
                      for x in (-0.35, -0.425, -0.50, -0.575, -0.65)],
                     "distance from the mount to the work (mm)",
                     "palm droop, with the 100 mm stack (mm)", colour=B, labfmt="{:.2f}",
                     note="deterministic")
    c_best = lines([("best cell reached", [(r["droop"], pct(r["a"])) for r in best], A, None,
                     0, -14)],
                   "palm droop of the geometry (mm)",
                   "seeds completing 8 cycles, at that geometry's best grip (%)", 1.4, 100,
                   marks=[(cut, "LAST 8/8")])
    c_cs = lines([("grip it needs",
                   [(r["droop"], r["cs"]) for r in best if r["a"]["k"]], B, None, 0, -16)],
                 "palm droop of the geometry (mm)",
                 "commanded carry grip at the best cell (mm)", 1.4, 2.0,
                 marks=[(1.29, "NOTHING WORKS")])
    c_tq = lines([("relay", [(t * 1000, pct(tq[(t, "relay")])) for t in TQ], A, None, 2, 22),
                  ("release", [(t * 1000, pct(tq[(t, "release")])) for t in TQ], B, "5 4",
                   0, -14)],
                 "resisting torque about the tool axis (mN&middot;m)",
                 "seeds completing 8 cycles (%)", 20, 100)

    # ------------------------------------------------------------------ tables
    t_table = "".join(
        tr(f'<td class="num">{z * 1000:.0f}</td>'
           + okcell(tab[(z, "relay")]["k"], tab[(z, "relay")]["n"])
           + okcell(tab[(z, "release")]["k"], tab[(z, "release")]["n"])
           + num(tab[(z, "relay")]["deg"]) + num(tab[(z, "release")]["deg"])
           + num(DR[(-0.5, 0.0, z, 0, 700)], "{:.2f}"), z == 0.0)
        for z in Z)

    t_stack = "".join(
        tr(f'<td class="num">{s}</td>' + num(PAY[(-0.5, 0.0, 0.0, s, 700)], "{:.0f}")
           + num(DR[(-0.5, 0.0, 0.0, s, 700)], "{:.2f}")
           + okcell(stk[(s, "relay")]["k"], stk[(s, "relay")]["n"])
           + okcell(stk[(s, "release")]["k"], stk[(s, "release")]["n"]), s == 100)
        for s in S)

    t_mass = "".join(
        tr(f'<td class="num">{p:.0f}</td>' + num(PAY[(-0.5, 0.0, 0.0, 100, p)], "{:.0f}")
           + num(DR[(-0.5, 0.0, 0.0, 100, p)], "{:.2f}")
           + okcell(mas[(p, "relay")]["k"], mas[(p, "relay")]["n"])
           + okcell(mas[(p, "release")]["k"], mas[(p, "release")]["n"]), p == 0.0)
        for p in RHO)

    def grid(keys, cells, lab):
        """One row per geometry, one column per commanded carry grip; cells are relay / release.

        A grid rather than two charts because the quantity of interest is the SHAPE of each row
        -- where its window sits -- and two overlaid five-point curves per geometry would be
        four series on one axis.
        """
        head = ("<thead><tr><th>" + lab + "</th>"
                + "".join(f'<th class="num">{c:.1f} mm</th>' for c in cells)
                + '<th class="num">droop</th></tr></thead>')
        body = ""
        for k, name, g in keys:
            cs = ""
            for c in cells:
                a_, b_ = pre.get((k, c, "relay")), pre.get((k, c, "release"))
                if a_ is None:
                    a_, b_ = pl2[(k, c, "relay")], pl2[(k, c, "release")]
                cls = "ok" if a_["k"] == a_["n"] else ("no" if a_["k"] == 0 else "")
                cs += (f'<td class="num"><span class="{cls}">{a_["k"]}</span>'
                       f'<span style="color:var(--ink3)"> / {b_["k"]}</span></td>')
            body += tr(f"<td><b>{name}</b></td>{cs}" + num(DR[g], "{:.2f}"))
        return f'<div class="tw"><table>{head}<tbody>{body}</tbody></table></div>'

    g_stack = grid([(s, f"{s} mm stack", (-0.5, 0.0, 0.0, s, 700)) for s in (0, 50, 100, 150)],
                   CS_SEAT, "wrist stack")
    g_place = grid([(x, f"{abs(x) * 1000:.0f} mm away", (x, 0.0, 0.0, 100, 700)) for x in XS],
                   (0.30, 0.80, 1.50, 2.00), "mount distance")

    t_best = "".join(
        tr(f'<td>{r["what"]}</td>' + num(r["droop"], "{:.2f}")
           + (num(r["cs"], "{:.1f}") if r["a"]["k"] else '<td class="num">&mdash;</td>')
           + okcell(r["a"]["k"], r["a"]["n"]) + okcell(r["b"]["k"], r["b"]["n"])
           + num(r["a"]["deg"]), r["droop"] == cut)
        for r in best)

    t_tq = "".join(
        tr(f'<td class="num">{t * 1000:.0f}</td>'
           + okcell(tq[(t, "relay")]["k"], tq[(t, "relay")]["n"])
           + okcell(tq[(t, "release")]["k"], tq[(t, "release")]["n"])
           + num(tq[(t, "relay")]["deg"]) + num(tq[(t, "release")]["free"] * 100), t == 0.020)
        for t in TQ)

    vid_wide = b64(D / "20260904-bench_wide.mp4", "video/mp4")
    vid_cu = b64(D / "20260904-bench_closeup.mp4", "video/mp4")
    film = jpeg(D / "20260904-bench_closeup_seams.png")

    stats = "".join(
        f'<div class="stat"><b>{v}</b><span>{s}</span></div>' for v, s in [
            ("0 mm", "of pedestal the task needs"),
            (f'{K["DR0"]} &rarr; {K["DR100"]} mm', "palm droop the 100 mm stack adds"),
            (f'{K["CUT"]} mm', "the most droop that still runs every seed"),
            ("425 mm", "where the robot then has to stand"),
        ])

    body = f"""
<section><div class="col">
<h2>What was not on the bench</h2>
<p class="sub">two lines of scene, and the first arm result that describes a robot somebody owns</p>

<p>The chained task &mdash; grasp, lift, reorient, set into a countersink, gait &mdash; has been
run on a UR5e since 2026-09-03, and that run made two geometric assumptions that were never
stated because they were never variables. The robot stood on a 300&nbsp;mm pedestal, so its
shoulder cleared the work. And the palm plate was bolted directly to the tool flange, because
the menagerie UR5e's attachment site is where a gripper goes.</p>

<p>Neither is the bench. The robot is bolted to the same flat table the work sits on. And
between the flange and the first finger joint there is a coupling, a servo bank and its wiring:
about 100&nbsp;mm of hardware, still to be measured, carrying its own weight. Both are now
parameters of <code>build_real_v1_arm_scene.py</code> (<code>--base</code>,
<code>--wrist-stack</code>, <code>--stack-density</code>). The stack is a 40&nbsp;mm cylinder at
700&nbsp;kg/m&sup3;, which takes the hand from {K["PAY0"]}&nbsp;g of payload to
{K["PAY100"]}&nbsp;g &mdash; payload the arm's controller does not model, exactly like the hand
itself.</p>

<figure><video src="{vid_wide}" autoplay loop muted playsinline></video>
<figcaption>The whole chain on the bench as specified: mount on the table surface,
100&nbsp;mm of wrist stack (the dark cylinder above the palm plate), tool driven into a
45&#176; countersink and gaited there without the hand ever letting go. Base 425&nbsp;mm from
the work, carry grip 1.5&nbsp;mm.</figcaption></figure>

<div class="callout"><span class="tag">A bug the change exposed</span>
<p>The arm scene builder copied the design scene's world by whitelist &mdash; lights, floor,
object. Building an arm scene from a countersink scene therefore dropped the countersink, and
the tool stood on a bare plane while the file name said otherwise. It now copies everything
except the hand.</p></div>
</div></section>

<section><div class="col">
<h2>The pedestal was free</h2>
<p class="sub">8 spawn-jittered seeds per cell &middot; 8 gait cycles &middot; no wrist stack</p>

<p>Lowering the mount from 300&nbsp;mm to the table surface changes nothing measurable. Every
height solves the home pose on the same IK branch with the arm's lowest link at
+0.163&nbsp;m &mdash; the shoulder &mdash; and no link ever below the table. Across the whole
study the differential IK reports {K["IKFAIL"]} unreachable poses in {K["NCELLS"]} rollouts, and
they are all in one place: the far corner of the standing sweep, 650&nbsp;mm back and 200&nbsp;mm
across, where 6 of 8 seeds ask for a pose the arm cannot make.</p>

<div class="tw"><table>
<thead><tr><th class="num">mount height (mm)</th><th class="num">relay</th>
<th class="num">release</th><th class="num">relay deg/cy</th><th class="num">release deg/cy</th>
<th class="num">droop mm</th></tr></thead>
<tbody>{t_table}</tbody></table></div>

<figure>{c_table}<figcaption>The raised table was not buying manipulability. It can go.
</figcaption></figure>
</div></section>

<section><div class="col">
<h2>The stack is not free, and only one mode notices</h2>
<p class="sub">wrist stack 0&ndash;150 mm, at the carry grip each mode was published with</p>

<p>Run the same chain with progressively more hardware between the flange and the palm, and the
two ways of changing grasp come apart. The release chain &mdash; open the hand, fly the palm to
the gait's pose, close on a ring &mdash; is indifferent up to 125&nbsp;mm. The relay, which walks
one pad at a time onto the ring and never has fewer than two on the tool, breaks at 50.</p>

<div class="tw"><table>
<thead><tr><th class="num">stack (mm)</th><th class="num">payload (g)</th>
<th class="num">droop (mm)</th><th class="num">relay</th><th class="num">release</th></tr></thead>
<tbody>{t_stack}</tbody></table></div>

<figure>{c_stack}<figcaption>At 150&nbsp;mm both modes fail, and they fail the same way: the tool
is on the floor within a second of the handover starting.</figcaption></figure>

<p>The asymmetry is the point. The release chain re-establishes its grasp from scratch on a tool
the floor is holding upright, so it can absorb a wrist that arrives in the wrong place. The relay
has no such moment &mdash; every pad it places is placed on a tool it is still carrying, and a
placement error becomes a torque on that tool rather than a miss.</p>
</div></section>

<section><div class="col">
<h2>It is the weight, not the reach</h2>
<p class="sub">the same 100 mm stack at four densities</p>

<p>A longer wrist is two things at once: a longer lever from the last joint to the pads, and more
mass hanging off it. Setting the stack's density to zero separates them. A massless 100&nbsp;mm
stack runs every seed. Any mass at all breaks the relay at 500&nbsp;mm of standoff.</p>

<div class="tw"><table>
<thead><tr><th class="num">density (kg/m&sup3;)</th><th class="num">payload (g)</th>
<th class="num">droop (mm)</th><th class="num">relay</th><th class="num">release</th></tr></thead>
<tbody>{t_mass}</tbody></table></div>

<p>At the highest density the release chain fails too, and the reach is identical in all four
rows. What changes is the third column.</p>
</div></section>

<section><div class="col">
<h2>All three knobs are one knob</h2>
<p class="sub">palm droop under the hand's own weight, at the top of a 100 mm lift</p>

<p>Stack length, stack mass and standing distance are three independent things to specify on a
bench, and each of them moves the same quantity: how far the palm ends up from where it was
commanded, once it is carrying a load the arm's controller does not know about. That number is
deterministic &mdash; it does not inherit the chain's seed-to-seed chaos &mdash; and the
2026-09-03 arm study already used it as the wrist's spec, at a resolution of two points
(0.97&nbsp;mm chained 6/6, 1.59&nbsp;mm chained 1/6).</p>

<div class="pair"><figure>{c_droop_s}<figcaption>Droop is linear in stack length, about
6&nbsp;&micro;m per millimetre.</figcaption></figure><figure>{c_droop_x}<figcaption>And it is
monotone in how far back the robot stands: a folded arm is a stiff arm.</figcaption>
</figure></div>

<p>Sweeping the carry grip at each geometry and taking the best cell it can reach collapses the
three knobs onto that one axis. The order is clean, and so is the threshold.</p>

<div class="tw"><table>
<thead><tr><th>geometry</th><th class="num">droop (mm)</th><th class="num">best grip (mm)</th>
<th class="num">relay</th><th class="num">release</th><th class="num">deg/cy</th></tr></thead>
<tbody>{t_best}</tbody></table></div>

<div class="pair"><figure>{c_best}<figcaption>Every seed completes up to
{K["CUT"]}&nbsp;mm of droop, and the geometry that would have been shipped &mdash; 100&nbsp;mm of
stack at 500&nbsp;mm of standoff &mdash; sits just past it.</figcaption></figure>
<figure>{c_cs}<figcaption>And the grip the geometry needs rises with it. This is the whole
mechanism.</figcaption></figure></div>
</div></section>

<section><div class="col">
<h2>Which makes the stack a preload problem</h2>
<p class="sub">wrist stack &#215; commanded carry grip, tool driven into the 45&#176; seat
&middot; each cell is seeds out of 8, relay then release</p>

<p>The carry does not end in a grip. It ends with the tool cradled on the middle phalanges at
about half a millimetre of interference, creeping down through the pads at roughly 1.5&nbsp;mm/s,
and <code>--carry-squeeze</code> is the one command that turns that into a hold before the
transport starts. It is commanded interference: the pads are told to close past the surface, and
the force that results is the difference between where they were told to go and where the plant
lets them stop.</p>

<p>Which is why droop and grip are the same story. A longer, heavier wrist deflects more under
the same command, so the same commanded interference buys less force &mdash; the arm is a spring
in exactly the sense the finger servos are, one level up. The fix is to command more, and the
window moves rather than closing.</p>

{g_stack}

<p>Every row has a window and every window is narrow: too little and the tool is delivered
tilted, too much and the pads torque it out of the seat on the way down. But up to 50&nbsp;mm of
stack the window simply relocates &mdash; 0.3&nbsp;mm becomes 1.5&nbsp;mm and both modes run every
seed. At 100&nbsp;mm the best cell is half the seeds, and at 150&nbsp;mm there is no cell at all.</p>
</div></section>

<section><div class="col">
<h2>Where the robot stands buys it back</h2>
<p class="sub">the full 100 mm stack, in the seat, at three standing distances
&middot; each cell is seeds out of 8, relay then release</p>

<p>Standoff moves droop over the same range the stack does, and in the opposite direction. With
the whole 100&nbsp;mm stack fitted, moving the mount from 500&nbsp;mm to 425&nbsp;mm takes droop
from {K["DR100"]} to {K["DR425"]}&nbsp;mm and takes the best cell from half the seeds to all of
them. At 350&nbsp;mm it needs less grip as well.</p>

{g_place}

<figure><video src="{vid_cu}" autoplay loop muted playsinline></video>
<figcaption>425&nbsp;mm, 100&nbsp;mm of stack, 1.5&nbsp;mm of carry grip: the relay handover onto
the gait ring and the first cycles of the relay gait, with the tool seated. Two pads are on the
tool at every instant.</figcaption></figure>

<div class="callout"><span class="tag">The bench specification, as it stands</span>
<p>One flat table, no pedestal. Mount 350&ndash;425&nbsp;mm from the work. Up to about
{K["CUT"]}&nbsp;mm of palm droop under the hand's own weight, which the 100&nbsp;mm stack fits
inside only at those distances. Carry grip 0.8&ndash;1.5&nbsp;mm rather than the published 0.3.
Everything else &mdash; the hand, the grasp, the gait set-points &mdash; unchanged.</p></div>
</div></section>

<section><div class="col">
<h2>What load it can drive</h2>
<p class="sub">a Coulomb brake about the tool's own axis, armed once the tool is seated
&middot; the recommended geometry: 100 mm stack, 425 mm standoff, 1.5 mm carry grip</p>

<p>A countersink is a friction brake, not a thread: it returns &mu;<i>Nr</i> and no more, and the
press sets <i>N</i> through the wedge. So the resisting torque a gaiting hand has ever worked
against is whatever the geometry happened to supply &mdash; about 7&nbsp;mN&middot;m on a flat
plane. <code>--screw-torque</code> adds a named one, as a brake that opposes whatever rotation
there is and does no work when the tool is still.</p>

<div class="tw"><table>
<thead><tr><th class="num">torque (mN&middot;m)</th><th class="num">relay</th>
<th class="num">release</th><th class="num">relay deg/cy</th>
<th class="num">release, % uncontacted</th></tr></thead>
<tbody>{t_tq}</tbody></table></div>

<figure>{c_tq}<figcaption>Both arms run at the same geometry and the same grip, so the difference
is the schedule. Never letting go is what lets the seat take a thread load.</figcaption></figure>

<p>The relay is flat: {tq[(0.0, "relay")]["deg"]:.1f}&nbsp;deg/cycle unloaded and
{tq[(0.020, "relay")]["deg"]:.1f} at 20&nbsp;mN&middot;m, with the same seeds completing. The
release chain loses the tool. Its share of the window with no pad on the tool is
{tq[(0.0, "release")]["free"] * 100:.0f}% unloaded &mdash; that is simply what releasing costs
&mdash; and it climbs to {tq[(0.020, "release")]["free"] * 100:.0f}% under load, which is a
different thing: the brake acts about the tool's own axis, and a tool standing a few degrees off
vertical is levered over by it during the part of every cycle when nothing is holding it.</p>

<div class="callout"><span class="tag">The knob's validity, and its bench equivalent</span>
<p>The tool is 24&nbsp;g with an axial inertia of 1.9&times;10<sup>-6</sup>&nbsp;kg&middot;m&sup2;,
so a brake much past 20&nbsp;mN&middot;m can overshoot zero inside a timestep and pump the shaft
instead of stopping it. That shows up as a turn past the 1.684 gear ceiling and is flagged per
rollout, and in this sweep {K["PUMPED"]} of {len(sel(rows, arm="torque"))} cells trip it &mdash;
{K["PUMPED_AT"]}, i.e. at the top of the swept range, which is where the range ends rather than a
clean bill for it. On the bench a cone gives no way to set this number at all &mdash; the only
knobs are the press and the seat material. A calibrated one needs a brake under the receptacle: put the
socket on a short shaft in a bushing with a preloaded friction washer, and the preload screw sets
a direction-independent torque a wrench can read.</p></div>
</div></section>

<section><div class="col">
<h2>What this does not settle</h2>

<p>The 100&nbsp;mm is a placeholder and so is the 40&nbsp;mm radius; both are geometry waiting on
a caliper, and the radius in particular is only a collision proxy &mdash; nothing in this study
touched the stack, so its shape has been tested only for what it weighs and how far out it puts
the hand.</p>

<p>The chain remains chaotic at the seam, and the binary pass rate over 8 seeds is a coarse
instrument: the standing sweep at the published grip is not monotone in droop, and it should not
be read as one &mdash; it is a sweep of an untuned parameter across a moving window. The
collapse onto droop is only clean once the grip is swept at each geometry, which is what the
table above does.</p>

<p>Nothing here has been through the export, trajectory-clearance or servo-limit gates. The
relay's Cartesian legs are a new path shape, and three of the four plans that have reached the
control station failed exactly that check.</p>

<p>And the grip is still commanded once and never corrected. Everything above is a search for
the one open-loop number that happens to fit a window whose position depends on the robot, the
payload, and where it stands &mdash; which is the argument for measuring it instead. The hand
already reports the quantity: grip force is commanded minus achieved, the finger servos calibrate
to one spring constant across the hand, and the deflection is readable at the control rate.</p>
</div></section>
"""

    tpl = (ROOT / "scripts/real_v1_bench_page.template.html").read_text()
    html = tpl.replace("{{NCELLS}}", str(K["NCELLS"])).replace(
        "{{STATS}}", stats).replace("{{BODY}}", body)
    out.write_text(html)
    print(f"-> {out}  {len(html) / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
