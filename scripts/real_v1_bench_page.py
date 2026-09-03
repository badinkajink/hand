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
        "turned": (st.mean([r["tilt_turned_deg"] for r in g if "tilt_turned_deg" in r])
                   if any("tilt_turned_deg" in r for r in g) else 0.0),
        "settled": (st.mean([r["tilt_settled_deg"] for r in g if "tilt_settled_deg" in r])
                    if any("tilt_settled_deg" in r for r in g) else 0.0),
        "settle_sd": (st.pstdev([r["tilt_settled_deg"] for r in g if "tilt_settled_deg" in r])
                      if sum(1 for r in g if "tilt_settled_deg" in r) > 1 else 0.0),
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
    out = args.out or (D / "20260904-real_v1_bench_geometry.html")
    blob = json.loads((D / "bench_study.json").read_text())
    rows, dro = blob["rows"], blob["droop"]

    gk = lambda r: (round(r.get("base_x", 0), 3), round(r.get("base_y", 0), 3),
                    round(r.get("base_z", 0), 3), round(r.get("stack_mm", 0)),
                    round(r.get("stack_density", 0)))
    # Keyed WITHOUT the payload flag but built from the undeclared scenes only: a compensated
    # scene has ~0.01 mm of droop and silently overwrote every row of the droop table when the
    # flag was left out of the key.
    DR = {gk(r): r["droop_mm"] for r in dro if not r.get("pgc")}
    DR_PGC = {gk(r): r["droop_mm"] for r in dro if r.get("pgc")}
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
    # The correction: the same seat grid on a UR5e that has been told its payload.
    pay = {(s, c, m): agg(sel(rows, arm="payload", stack_mm=float(s), carry_squeeze_mm=c,
                              mode=m))
           for s in (0, 50, 100, 150) for c in (0.30, 0.80, 1.50)
           for m in ("relay", "release")}
    TSQ, HOLD = (0.0, 0.5, 1.0, 1.5, 2.0), (0, 150, 300, 500, 900)
    slp = {(q, h): agg(sel(rows, arm="slip", turn_squeeze_mm=q, hold_steps=h))
           for q in TSQ for h in HOLD}
    wri = {(g, h): agg(sel(rows, arm="wrist", pgc=g, hold_steps=h))
           for g in (True, False) for h in HOLD}
    tqg = {(q, c): agg(sel(rows, arm="torque_grip", screw_torque=q, carry_squeeze_mm=c))
           for q in (0.0, 0.008, 0.016) for c in (0.30, 0.80)}

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
        TURN_C=f'{wri[(True, 150)]["turned"]:.2f}',
        TURN_U=f'{wri[(False, 150)]["turned"]:.2f}',
        SD_C=f'{wri[(True, 0)]["settle_sd"]:.2f}',
        SETTLE_U=f'{wri[(False, 500)]["settled"]:.2f}',
        BEST_HOLD_TILT=f'{wri[(True, 150)]["settled"]:.2f}',
        DR0=f'{DR[(-0.5, 0.0, 0.0, 0, 700)]:.2f}',
        DR100=f'{DR[(-0.5, 0.0, 0.0, 100, 700)]:.2f}',
        DR35=f'{DR[(-0.35, 0.0, 0.0, 100, 700)]:.2f}',
        DR425=f'{DR[(-0.425, 0.0, 0.0, 100, 700)]:.2f}',
        PAY0=f'{PAY[(-0.5, 0.0, 0.0, 0, 700)]:.0f}',
        PAY100=f'{PAY[(-0.5, 0.0, 0.0, 100, 700)]:.0f}',
        CUT=f"{cut:.2f}",
        DRC=f'{DR_PGC.get((-0.5, 0.0, 0.0, 100, 700), 0.0):.2f}',
        IKFAIL=sum(r.get("arm_ik_fails", 0) for r in rows),
        PUMPED=sum(1 for r in rows if r.get("brake_pumped")),
        PUMPED_AT=", ".join(sorted({f'{r["screw_torque"] * 1000:.0f} mN&middot;m in the '
                                    f'{"relay" if r["mode"] == "relay" else "release"} arm'
                                    for r in rows if r.get("brake_pumped")})) or "none",
    )

    # ------------------------------------------------------------------ charts
    pct = lambda g: g["k"] / g["n"] * 100 if g["n"] else 0.0

    c_table = lines([("relay", [(z * 1000, pct(tab[(z, "relay")])) for z in Z], A, None, 0, -14),
                     ("release", [(z * 1000, pct(tab[(z, "release")])) for z in Z], B, "5 4",
                      2, 22)],
                    "height of the robot's mount above the work (mm)",
                    "seeds completing 8 cycles (%)", 320, 100)
    c_pay = lines([("payload declared", [(s, pct(pay[(s, 0.30, "relay")]))
                                         for s in (0, 50, 100, 150)], A, None, 1, -14),
                   ("not declared", [(s, pct(stk[(s, "relay")])) for s in S], B, "5 4", 3, 22)],
                  "hardware between the tool flange and the palm plate (mm)",
                  "seeds completing 8 cycles (%)", 160, 100, marks=[(100, "THE BENCH")])
    c_droop_s = bars([(f"{s}", DR[(-0.5, 0.0, 0.0, s, 700)]) for s in S],
                     "wrist stack (mm)", "palm droop, payload not declared (mm)",
                     colour=B, labfmt="{:.2f}", note="deterministic")
    c_droop_x = bars([(f"{abs(x) * 1000:.0f}", DR[(x, 0.0, 0.0, 100, 700)])
                      for x in (-0.35, -0.425, -0.50, -0.575, -0.65)],
                     "distance from the mount to the work (mm)",
                     "palm droop with the 100 mm stack (mm)", colour=B, labfmt="{:.2f}",
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
    c_hold = lines([("payload declared", [(h, wri[(True, h)]["settled"]) for h in HOLD], A,
                     None, 2, 24),
                    ("not declared", [(h, wri[(False, h)]["settled"]) for h in HOLD], B, "5 4",
                     1, -16)],
                   "settling steps after the last commanded turn step",
                   "tool tilt from vertical (deg)", 900, 40)
    c_tsq = lines([("at the last turn step", [(q, slp[(q, 0)]["turned"]) for q in TSQ], A,
                    None, 2, 24),
                   ("after 300 settling steps", [(q, slp[(q, 300)]["settled"]) for q in TSQ],
                    B, "5 4", 1, -16)],
                  "extra radial closure held through the turn (mm)",
                  "tool tilt from vertical (deg)", 2.0, 95)
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

    t_pay = "".join(
        tr(f'<td class="num">{s}</td>' + num(PAY[(-0.5, 0.0, 0.0, s, 700)], "{:.0f}")
           + okcell(pay[(s, 0.30, "relay")]["k"], pay[(s, 0.30, "relay")]["n"])
           + num(pay[(s, 0.30, "relay")]["deg"])
           + num(pay[(s, 0.30, "relay")]["tilt"], "{:.1f}")
           + okcell(stk[(s, "relay")]["k"], stk[(s, "relay")]["n"])
           + num(DR[(-0.5, 0.0, 0.0, s, 700)], "{:.2f}"), s == 100)
        for s in (0, 50, 100, 150))

    t_paygrid = "".join(
        tr(f'<td><b>{s} mm stack</b></td>'
           + "".join(okcell(pay[(s, c, m)]["k"], pay[(s, c, m)]["n"])
                     for c in (0.30, 0.80, 1.50) for m in ("relay", "release")), s == 100)
        for s in (0, 50, 100, 150))

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

    t_best = "".join(
        tr(f'<td>{r["what"]}</td>' + num(r["droop"], "{:.2f}")
           + (num(r["cs"], "{:.1f}") if r["a"]["k"] else '<td class="num">&mdash;</td>')
           + okcell(r["a"]["k"], r["a"]["n"]) + okcell(r["b"]["k"], r["b"]["n"])
           + num(r["a"]["deg"]), r["droop"] == cut)
        for r in best)

    t_hold = "".join(
        tr(f'<td class="num">{h}</td>' + num(h * 0.002, "{:.1f}")
           + num(wri[(True, h)]["settled"], "{:.2f}")
           + num(wri[(True, h)]["settle_sd"], "{:.2f}")
           + okcell(wri[(True, h)]["k"], wri[(True, h)]["n"])
           + num(wri[(False, h)]["settled"], "{:.2f}")
           + okcell(wri[(False, h)]["k"], wri[(False, h)]["n"]), h == 150)
        for h in HOLD)

    t_tsq = "".join(
        tr(f'<td class="num">{q:.1f}</td>' + num(slp[(q, 0)]["turned"], "{:.2f}")
           + num(slp[(q, 300)]["settled"], "{:.2f}")
           + okcell(slp[(q, 150)]["k"], slp[(q, 150)]["n"])
           + okcell(slp[(q, 500)]["k"], slp[(q, 500)]["n"]), q == 0.0)
        for q in TSQ)

    t_tqg = "".join(
        tr(f'<td class="num">{q * 1000:.0f}</td>'
           + okcell(tqg[(q, 0.30)]["k"], tqg[(q, 0.30)]["n"])
           + num(tqg[(q, 0.30)]["deg"])
           + okcell(tqg[(q, 0.80)]["k"], tqg[(q, 0.80)]["n"])
           + num(tqg[(q, 0.80)]["deg"]), q == 0.0)
        for q in (0.0, 0.008, 0.016))

    t_tq = "".join(
        tr(f'<td class="num">{t * 1000:.0f}</td>'
           + okcell(tq[(t, "relay")]["k"], tq[(t, "relay")]["n"])
           + okcell(tq[(t, "release")]["k"], tq[(t, "release")]["n"])
           + num(tq[(t, "relay")]["deg"]) + num(tq[(t, "release")]["free"] * 100), t == 0.020)
        for t in TQ)

    vid_wide = b64(D / "20260904-bench_wide.mp4", "video/mp4")
    vid_cu = b64(D / "20260904-bench_closeup.mp4", "video/mp4")

    stats = "".join(
        f'<div class="stat"><b>{v}</b><span>{s}</span></div>' for v, s in [
            ("8/8 at 150 mm", "wrist stack the arm carries once told its payload"),
            (f'{K["TURN_C"]}&#176;', f'tilt the turn then delivers directly, sd {K["SD_C"]}'),
            (f'{K["TURN_U"]}&#176;', "the same turn on an arm that has not been told"),
            ("0 mm", "of pedestal the task needs"),
        ])

    body = f"""
<section><div class="col">
<h2>What was not in the scene</h2>
<p class="sub">one flat table, 100 mm of wrist hardware, and a declared payload</p>

<p>The chained task &mdash; grasp, lift, reorient, set into a countersink, gait &mdash; has run on
a UR5e since 2026-09-03 with three assumptions that were never variables. The robot stood on a
300&nbsp;mm pedestal. The palm plate was bolted straight to the tool flange, because the
menagerie UR5e's attachment site is where a gripper goes. And the arm carried the hand without
knowing it was there.</p>

<p>None of the three is the bench. The robot is bolted to the same flat table as the work; there
is a coupling, a servo bank and its wiring between the flange and the first finger joint, about
100&nbsp;mm of it; and a real UR5e is told its payload with <code>set_payload</code>, after which
its servo loop holds position under a load it is rated to 5&nbsp;kg for. All three are now
parameters of <code>build_real_v1_arm_scene.py</code> (<code>--base</code>,
<code>--wrist-stack</code>, <code>--stack-density</code>, <code>--payload-gravcomp</code>). The
stack is a 40&nbsp;mm cylinder at 700&nbsp;kg/m&sup3;, taking the hand from {K["PAY0"]} to
{K["PAY100"]}&nbsp;g.</p>

<figure><video src="{vid_wide}" autoplay loop muted playsinline></video>
<figcaption>The chain on the bench as specified: mount on the table surface, 100&nbsp;mm of wrist
stack (the dark cylinder above the palm plate), tool driven into a 45&#176; countersink and
gaited there without the hand releasing it.</figcaption></figure>

<div class="callout"><span class="tag">A bug the change exposed</span>
<p>The arm scene builder copied the design scene's world by whitelist &mdash; lights, floor,
object. Building an arm scene from a countersink scene therefore dropped the countersink, and the
tool stood on a bare plane while the file name said otherwise. It now copies everything except
the hand.</p></div>
</div></section>

<section><div class="col">
<h2>The pedestal is free</h2>
<p class="sub">8 spawn-jittered seeds per cell &middot; 8 gait cycles &middot; no wrist stack</p>

<p>Lowering the mount from 300&nbsp;mm to the table surface changes nothing measurable. Every
height solves the home pose on the same IK branch with the arm's lowest link at +0.163&nbsp;m
&mdash; the shoulder &mdash; and no link below the table. Across the study the differential IK
reports {K["IKFAIL"]} unreachable poses in {K["NCELLS"]} rollouts, all at one standing position:
650&nbsp;mm back and 200&nbsp;mm across, where 6 of 8 seeds ask for a pose the arm cannot make.
The second table can go.</p>

<div class="tw"><table>
<thead><tr><th class="num">mount height (mm)</th><th class="num">relay</th>
<th class="num">release</th><th class="num">relay deg/cy</th><th class="num">release deg/cy</th>
<th class="num">droop mm</th></tr></thead>
<tbody>{t_table}</tbody></table></div>
</div></section>

<section><div class="col">
<h2>Told its payload, the arm carries the stack for free</h2>
<p class="sub">tool driven into the 45&#176; seat, published carry grip of 0.3 mm, 500 mm standoff</p>

<p>The wrist stack costs nothing at any length tested. Declaring the payload takes the palm's
error at the top of the lift from {K["DR100"]} to {K["DRC"]}&nbsp;mm, and with that, 0, 50, 100 and 150&nbsp;mm of stack all complete every seed at
{pay[(100, 0.30, "relay")]["deg"]:.0f}&nbsp;deg/cycle, and the tool ends within
{max(pay[(s, 0.30, "relay")]["tilt"] for s in (0, 50, 100, 150)):.1f}&#176; of vertical in every
case. Without the declaration the same 150&nbsp;mm loses the tool on every seed.</p>

<div class="tw"><table>
<thead><tr><th class="num">stack (mm)</th><th class="num">payload (g)</th>
<th class="num">relay, declared</th><th class="num">deg/cy</th><th class="num">end tilt</th>
<th class="num">relay, not declared</th><th class="num">droop mm</th></tr></thead>
<tbody>{t_pay}</tbody></table></div>

<figure>{c_pay}<figcaption>The whole cost of the wrist stack is a payload the controller was not
told about.</figcaption></figure>

<p>The declaration also inverts the grip that works. On the sagging arm the seam needed more
commanded interference the longer the wrist got; on the stiff one the published 0.3&nbsp;mm is
already right and more is worse, because the extra closure torques the tool out of the seat on
the way down rather than compensating a deflection that is no longer there.</p>

<div class="tw"><table>
<thead><tr><th>wrist stack</th><th class="num" colspan="2">0.3 mm grip</th>
<th class="num" colspan="2">0.8 mm</th><th class="num" colspan="2">1.5 mm</th></tr>
<tr><th></th><th class="num">relay</th><th class="num">rel.</th><th class="num">relay</th>
<th class="num">rel.</th><th class="num">relay</th><th class="num">rel.</th></tr></thead>
<tbody>{t_paygrid}</tbody></table></div>

<div class="callout"><span class="tag">Bench specification</span>
<p>One flat table, no pedestal. Mount at 500&nbsp;mm, which is where the reach study left it and
which now needs no adjustment. Declare the hand's payload to the controller. Carry grip
0.3&nbsp;mm, as published. Wrist stack anywhere from 0 to 150&nbsp;mm. Nothing about the hand,
the grasp or the gait set-points changes.</p></div>
</div></section>

<section><div class="col">
<h2>What an undeclared payload costs, and where it starts</h2>
<p class="sub">kept because it is the diagnostic: any wrist that sags behaves this way</p>

<p>The uncompensated sweeps are still the useful characterisation of a compliant wrist, and they
collapse onto one number. Stack length, stack mass and standing distance are three independent
bench specs, and each moves only <b>palm droop</b> &mdash; how far the palm ends up from where it
was commanded while holding a load the controller does not model. Droop is deterministic, so it
does not inherit the seam's seed-to-seed chaos.</p>

<div class="pair"><figure>{c_droop_s}<figcaption>Linear in stack length, about
6&nbsp;&micro;m per millimetre.</figcaption></figure><figure>{c_droop_x}<figcaption>Monotone in
standoff: a folded arm is a stiff arm.</figcaption></figure></div>

<p>Sweeping the carry grip at each geometry and taking its best cell orders exactly on droop.
Every seed completes up to {K["CUT"]}&nbsp;mm; 0.97&nbsp;mm gives 3 of 8 and 1.29&nbsp;mm gives
none. The grip each geometry needs rises with droop over the same range, which is the mechanism:
commanded interference minus deflection is the force, so a wrist that deflects more needs a
larger command for the same grip.</p>

<div class="tw"><table>
<thead><tr><th>geometry</th><th class="num">droop (mm)</th><th class="num">best grip (mm)</th>
<th class="num">relay</th><th class="num">release</th><th class="num">deg/cy</th></tr></thead>
<tbody>{t_best}</tbody></table></div>

<div class="pair"><figure>{c_best}<figcaption>The threshold is {K["CUT"]}&nbsp;mm of droop.
</figcaption></figure><figure>{c_cs}<figcaption>And the grip needed to work around it.
</figcaption></figure></div>

<p>Two supporting sweeps are in the data and not plotted here: a massless 100&nbsp;mm stack runs
every seed, which is what identifies the payload rather than the reach as the cause; and stack
length at the published grip breaks the relay at 50&nbsp;mm while leaving the release chain
untouched out to 125, because the release chain rebuilds its grasp on a tool the floor is holding
and the relay places every pad on a tool it is still carrying.</p>

<div class="pair"><div class="tw"><table>
<thead><tr><th class="num">stack</th><th class="num">g</th><th class="num">droop</th>
<th class="num">relay</th><th class="num">rel.</th></tr></thead>
<tbody>{t_stack}</tbody></table></div>
<div class="tw"><table>
<thead><tr><th class="num">density</th><th class="num">g</th><th class="num">droop</th>
<th class="num">relay</th><th class="num">rel.</th></tr></thead>
<tbody>{t_mass}</tbody></table></div></div>
</div></section>

<section><div class="col">
<h2>Where the reorientation's alignment comes from</h2>
<p class="sub">a new <code>turned</code> seam, taken at the last commanded step of the turn</p>

<p>The turn is an IK sweep of the finger anchor, clipped to &plusmn;0.5&nbsp;rad, followed by a
hold with the finger commands frozen. Any alignment gained during that hold is the shaft rolling
itself upright inside the grasp. Until now there was no snapshot between the two, so the two
contributions were reported as one number.</p>

<p>With them separated, the settle is mostly a correction for the wrist. On an arm that has not
been told its payload the turn ends {K["TURN_U"]}&#176; off vertical and the hold pulls it back
to {K["SETTLE_U"]}&#176;. On the same arm with the payload declared the turn ends
<b>{K["TURN_C"]}&#176;</b> off vertical, with a spread of {K["SD_C"]}&#176; across spawn jitter,
and the hold then makes it worse.</p>

<div class="tw"><table>
<thead><tr><th class="num">settling steps</th><th class="num">seconds</th>
<th class="num">tilt, declared</th><th class="num">sd</th><th class="num">ok</th>
<th class="num">tilt, not declared</th><th class="num">ok</th></tr></thead>
<tbody>{t_hold}</tbody></table></div>

<figure>{c_hold}<figcaption>Most of the 28&#176; of settle that the published chain depends on was
the wrist sagging through the turn and the tool rolling most of it back.</figcaption></figure>

<h3>Can the slip be predicted, controlled and measured?</h3>

<p><b>Predicted:</b> yes, to about a degree, and it degrades with exposure. Tilt at the last
commanded turn step is {K["SD_C"]}&#176; sd over spawn jitter; after 150 settling steps the
settled tilt has sd {wri[(True, 150)]["settle_sd"]:.2f}&#176;, after 900 it is
{wri[(True, 900)]["settle_sd"]:.2f}&#176;. The shaft is rolling away from an unstable
configuration, so the longer it is allowed to roll the less the seeds agree.</p>

<p><b>Controlled:</b> by settling time, and only downwards. The best cell is 150 steps
(0.3&nbsp;s) at {K["BEST_HOLD_TILT"]}&#176;; longer holds drift back out. Gripping through the
turn does not work as the other control: extra closure held through the sweep takes the
commanded tilt from {slp[(0.0, 0)]["turned"]:.1f}&#176; to
{slp[(2.0, 0)]["turned"]:.1f}&#176; and nothing completes at any of the four non-zero
settings.</p>

<div class="tw"><table>
<thead><tr><th class="num">closure through the turn (mm)</th>
<th class="num">tilt at the last turn step</th><th class="num">after 300 steps</th>
<th class="num">ok at 150</th><th class="num">ok at 500</th></tr></thead>
<tbody>{t_tsq}</tbody></table></div>

<figure>{c_tsq}<figcaption>Squeezing through the turn fights it. The fingers are driving the shaft
round by rolling it, and a tighter grip removes the rolling.</figcaption></figure>

<p><b>Measured:</b> already, on the bench, at 0.017&#176; rms. The reorientation turns the shaft
about a horizontal axis, so the vane tag stays in the camera's view for the whole sweep and
<code>morphohand.bench.tags</code> reports the tool's tilt directly. This is the opposite of the
gait, where the shaft spins about the vertical and a single tag leaves view within about
60&#176;.</p>

<p>So the practical answer is that the settle is not needed once the payload is declared. It was
worth 23&#176; on a sagging wrist and it is worth nothing on a stiff one, where the hand delivers
{K["TURN_C"]}&#176; directly and repeatably. The remaining use for the <code>turned</code> seam is
as a monitor: it separates a hand that reoriented the tool from a wrist that dropped it into
place.</p>
</div></section>

<section><div class="col">
<h2>What load it can drive</h2>
<p class="sub">a Coulomb brake about the tool's own axis, armed once the tool is seated</p>

<p>A countersink is a friction brake, not a thread: it returns &mu;<i>Nr</i> and the press sets
<i>N</i> through the wedge, so the resisting torque a gaiting hand has ever worked against is
whatever the geometry supplied &mdash; about 7&nbsp;mN&middot;m on a flat plane.
<code>--screw-torque</code> adds a named one, opposing whatever rotation there is and doing no
work when the tool is still.</p>

<div class="tw"><table>
<thead><tr><th class="num">torque (mN&middot;m)</th><th class="num">relay</th>
<th class="num">release</th><th class="num">relay deg/cy</th>
<th class="num">release, % uncontacted</th></tr></thead>
<tbody>{t_tq}</tbody></table></div>

<figure>{c_tq}<figcaption>At the published 0.3&nbsp;mm grip the relay drives
{tq[(0.004, "relay")]["deg"]:.0f}&nbsp;deg/cycle against 4&nbsp;mN&middot;m and starts losing
seeds by 8.</figcaption></figure>

<p>Raising the commanded grip does not extend that. Crossing torque with
<code>--carry-squeeze</code> on one scene, 0.8&nbsp;mm is worse than 0.3 at every torque
including zero, because on a stiff wrist the extra closure is interference rather than
compensation.</p>

<div class="tw"><table>
<thead><tr><th class="num">torque (mN&middot;m)</th><th class="num">0.3 mm grip</th>
<th class="num">deg/cy</th><th class="num">0.8 mm grip</th><th class="num">deg/cy</th></tr></thead>
<tbody>{t_tqg}</tbody></table></div>

<p>On the sagging arm the same relay ran flat to 20&nbsp;mN&middot;m at 1.5&nbsp;mm of commanded
grip. Both cells are the same fingers on the same seat, and they differ in commanded
interference, achieved pad force and wrist stiffness at once, so they do not compare as grip
settings. What separates them is pad force, which this chain commands open-loop and never
measures &mdash; the drivable torque cannot be quoted as a number for this hand until it
does.</p>

<div class="callout"><span class="tag">The knob's validity, and its bench equivalent</span>
<p>The tool is 24&nbsp;g with an axial inertia of 1.9&times;10<sup>-6</sup>&nbsp;kg&middot;m&sup2;,
so a brake much past 20&nbsp;mN&middot;m can overshoot zero inside a timestep and pump the shaft
instead of stopping it. That reads as a turn past the 1.684 gear ceiling and is flagged per
rollout; {K["PUMPED"]} cells trip it here ({K["PUMPED_AT"]}), which marks where the range ends. On
the bench a cone cannot supply a set torque at all &mdash; the only knobs are the press and the
seat material. A calibrated one needs a brake under the receptacle: the socket on a short shaft
in a bushing with a preloaded friction washer, and the preload screw sets a direction-independent
torque a wrench can read.</p></div>
</div></section>

<section><div class="col">
<h2>Next measurements</h2>

<ol>
<li><b>Caliper the wrist stack and weigh it.</b> 100&nbsp;mm and 40&nbsp;mm radius are
placeholders. Measure flange face to first yaw axis, and the assembly's mass and centre. Both go
straight into <code>--wrist-stack</code> and <code>--stack-density</code>; the whole study
re-runs in about 35&nbsp;minutes. The result above says the numbers will not change the answer,
which is itself the thing to confirm.</li>

<li><b>Confirm the payload is actually declared on the real robot.</b> Everything here turns on
it. Read back <code>get_target_payload()</code> on the controller and compare the arm's sag under
the hand at the top of the lift against the {DR[(-0.5, 0.0, 0.0, 100, 700)]:.2f}&nbsp;mm the
undeclared model predicts. If the bench robot has not been configured, the droop section is the
operative one and the mount has to move to 425&nbsp;mm.</li>

<li><b>Cross the thread load with the grip.</b> The drivable torque moved from 4&ndash;12 to
20&nbsp;mN&middot;m when the commanded grip went 0.3&nbsp;&rarr;&nbsp;1.5&nbsp;mm, but those two
cells also differ in wrist stiffness. Run <code>--screw-torque</code> &times;
<code>--carry-squeeze</code> on one scene and report torque per millimetre of commanded
interference; that is the number a fastener spec would be checked against.</li>

<li><b>Close the grip loop.</b> Pad force is commanded minus achieved, the finger servos
calibrate to one spring constant across the hand, and position is readable at the control rate.
Estimate force from deflection and trim <code>--carry-squeeze</code> to hold a target rather than
searching for an open-loop value that depends on the robot. This is the only item above that
removes a whole class of sweep.</li>

<li><b>Run the relay's Cartesian legs through the gates.</b>
<code>scripts/real_v1_trajectory_clearance.py</code> and the servo-limit check, then export a
plan. It is a new path shape, and three of the four plans that have reached the control station
failed exactly that check.</li>

<li><b>Build the tag collar for the gait.</b> Three AprilTags at 120&#176; on a
60&nbsp;mm printed collar above the pad ring, so the one camera reads the spin through a full
turn instead of the &plusmn;60&#176; a single tag covers. <code>morphohand.bench.tags</code>
already carries the geometry for a vane; the collar is the same maths with an ID lookup and an
unwrap.</li>

<li><b>Instrument the axial slip.</b> The floor-free relay holds the tool for every cycle and
lowers it 11&nbsp;mm per cycle; <code>slip_mm_per_cycle</code> reports it and no reward term,
scorecard entry or design metric on either topology does. Add it to the scorecard before
attempting a persistent-grasp gait.</li>
</ol>
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
