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
import retro_style  # noqa: E402


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
    ap.add_argument("--slip", type=Path,
                    default=ROOT / "docs/experiments/20260904-real_v1_slip/slip_study.json")
    ap.add_argument("--ranked", type=Path,
                    default=ROOT / "docs/experiments/20260904-real_v1_ranked/ranked_slip.json")
    args = ap.parse_args()
    D = args.data
    out = args.out or (D / "20260904-real_v1_bench_geometry.html")
    blob = json.loads((D / "bench_study.json").read_text())
    rows, dro = blob["rows"], blob["droop"]
    # The multi-hand replication of the slip claim lives in its own dated folder because it is
    # its own study; the page reads it because it qualifies a claim this page makes.
    srows = json.loads(args.slip.read_text())["rows"] if args.slip.exists() else []
    # The Sobol population the design study ranks, run at each hand's own operating point in the
    # maneuver it was screened on.
    rk = json.loads(args.ranked.read_text()) if args.ranked.exists() else {"rows": [], "hands": []}
    rrows, rhands = rk["rows"], rk["hands"]

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

    # ---- the multi-hand replication -------------------------------------------------------
    # `held_turn` is two pads on the tool and the tool still in the air at the last commanded
    # turn step. Without that gate a hand that DROPS the tool into the countersink reports a
    # near-perfect tilt, because a tool standing in its seat is vertical.
    def hagg(g):
        h = [r for r in g if r.get("held_turn")]
        f = lambda k: [float(r.get(k) or 0.0) for r in h]
        sd = lambda v: st.pstdev(v) if len(v) > 1 else 0.0
        return {"n": len(g), "held": len(h), "k": sum(1 for r in g if r.get("ok")),
                "turned": st.mean(f("tilt_turned_deg")) if h else float("nan"),
                "turned_sd": sd(f("tilt_turned_deg")) if h else float("nan"),
                "settled": st.mean(f("tilt_settled_deg")) if h else float("nan"),
                "settled_sd": sd(f("tilt_settled_deg")) if h else float("nan"),
                "force": st.mean(f("force_turned_N")) if h else float("nan")}

    # ---- the ranked population ------------------------------------------------------------
    import glob as _glob
    POPD = ROOT / "docs/experiments/20260831-real_v1-sobol8192"

    def _screen(design, budget):
        """What the confirmation scored this hand at THIS clip. Keyed on the budget as well as
        the design: two promoted tags are the same hand at a different clip."""
        for f in sorted(_glob.glob(str(POPD / "confirm_b*.json"))):
            for r in json.loads(Path(f).read_text()):
                if r["design"] == design and abs(r["budget_rad"] - budget) < 1e-9:
                    return r["nom_cos"]
        return None

    def ragg(g):
        h = [r for r in g if r.get("held_turn")]
        f = lambda k: [float(r.get(k) or 0.0) for r in h]
        sdv = lambda v: st.pstdev(v) if len(v) > 1 else float("nan")
        mv = lambda v: st.mean(v) if v else float("nan")
        return {"n": len(g), "held": len(h), "k": sum(1 for r in g if r.get("ok")),
                "turn": mv(f("turn_tilt_deg")), "turn_sd": sdv(f("turn_tilt_deg")),
                "fin": mv(f("final_tilt_deg")), "fin_sd": sdv(f("final_tilt_deg")),
                "force": mv(f("turn_force_hand_N"))}

    RH = [h["tag"] for h in rhands]
    RHOLD = (0, 150, 300, 500, 900, 2500)
    RQ = (0.5, 1.0, 2.0)
    rk_s = {(t, hs): ragg([r for r in rrows if r.get("arm") == "settle" and r["tag"] == t
                           and r.get("hold_steps") == hs]) for t in RH for hs in RHOLD}
    rk_o = {(t, hs): ragg([r for r in rrows if r.get("arm") == "open" and r["tag"] == t
                           and r.get("hold_steps") == hs]) for t in RH for hs in RHOLD}
    rk_q = {(t, q): ragg([r for r in rrows if r.get("arm") == "squeeze" and r["tag"] == t
                          and abs((r.get("hold_squeeze_mm") or 0.0) - q) < 1e-9])
            for t in RH for q in RQ}
    rk_v = {}
    for h in rhands:
        g = [r for r in rrows if r.get("arm") == "validate" and r["tag"] == h["tag"]]
        held = [abs(r["final_cos"]) if r.get("ok") else 0.0 for r in g]
        rk_v[h["tag"]] = {"screen": _screen(h["design"], h["budget"]),
                          "replay": st.mean(held) if held else float("nan"), "n": len(g)}
    # Ordered by what the turn itself delivers, best first.
    RORD = sorted(RH, key=lambda t: rk_s[(t, 0)]["turn"])

    DSN = sorted({r["design"] for r in srows}) if srows else []
    SH = (0, 150, 300, 500, 900)
    mh = {(d, h): hagg([r for r in srows if r["arm"] == "settle" and r["design"] == d
                        and r["hold_steps_m"] == h]) for d in DSN for h in SH}
    mq = {(d, q): hagg([r for r in srows if r["arm"] == "squeeze" and r["design"] == d
                        and abs(r["turn_squeeze_m"] - q) < 1e-9])
          for d in DSN for q in (0.0005, 0.0010, 0.0020)}
    # A hand counts as reaching the turn if it holds the tool at that seam in every seed.
    mw = {(d, h): hagg([r for r in srows if r["arm"] == "wrist" and r["design"] == d
                        and r["hold_steps_m"] == h]) for d in DSN for h in (0, 300, 900)}
    HOLDERS = [d for d in DSN if mh[(d, 0)]["held"] == mh[(d, 0)]["n"] and mh[(d, 0)]["n"]]
    CHAINERS = [d for d in DSN if max(mh[(d, h)]["k"] for h in SH) > 0]

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
        RK_N=len(RH), RK_NDES=len({h["design"] for h in rhands}),
        RK_LO=f'{min(rk_s[(t, 0)]["turn"] for t in RH):.1f}',
        RK_HI=f'{max(rk_s[(t, 0)]["turn"] for t in RH):.1f}',
        RK_SDLO=f'{min(rk_s[(t, 0)]["turn_sd"] for t in RH):.2f}',
        RK_SDHI=f'{max(rk_s[(t, 0)]["turn_sd"] for t in RH):.2f}',
        RK_MATCH=sum(1 for t in RH if rk_v[t]["screen"] is not None
                     and abs(rk_v[t]["screen"] - rk_v[t]["replay"]) < 0.0015),
        RK_HASCELL=sum(1 for t in RH if rk_v[t]["screen"] is not None),
        RK_SETTLE_HI=f'{max(rk_s[(t, 0)]["turn"] - rk_s[(t, 2500)]["fin"] for t in RH):+.1f}',
        RK_SETTLE_LO=f'{min(rk_s[(t, 0)]["turn"] - rk_s[(t, 2500)]["fin"] for t in RH):+.1f}',
        RK_HELPED=sum(1 for t in RH if rk_s[(t, 0)]["turn"] - rk_s[(t, 2500)]["fin"] > 0),
        RK_REGMAX=f'{max(abs(rk_o[(t, 2500)]["fin"] - rk_s[(t, 2500)]["fin"]) for t in RH):.2f}',
        RK_ROWS=len(rrows),
        MH_N=len(DSN), MH_HELD=len(HOLDERS), MH_CHAIN=len(CHAINERS),
        MH_FALSE_MIN=f'{min([r["tilt_turned_deg"] for r in srows if r["arm"] == "settle" and r["design"] == "rv04_mid_sp40" and r["hold_steps_m"] == 300 and (r.get("pads_turned") or 0) == 0 and r["tilt_turned_deg"] < 20]):.1f}' if srows else "",
        MH_FALSE=f'{st.mean([r["tilt_turned_deg"] for r in srows if r["arm"] == "settle" and r["design"] == "rv04_mid_sp40" and r["hold_steps_m"] == 300 and (r.get("pads_turned") or 0) == 0 and r["tilt_turned_deg"] < 20]):.1f}' if srows else "",
        MH_FALSE_N=sum(1 for r in srows if r["arm"] == "settle" and r["design"] == "rv04_mid_sp40" and r["hold_steps_m"] == 300 and (r.get("pads_turned") or 0) == 0 and r["tilt_turned_deg"] < 20),
        MH_FALSE_Z=f'{st.mean([(r.get("z_turned") or 0) * 1000 for r in srows if r["arm"] == "settle" and r["design"] == "rv04_mid_sp40" and r["hold_steps_m"] == 300 and (r.get("pads_turned") or 0) == 0 and r["tilt_turned_deg"] < 20]):.0f}' if srows else "",
        MH_LIST=" and ".join(f'<code>{d.replace("_stored", "")}</code> at '
                             f'{mh[(d, 0)]["turned"]:.0f}&#176;'
                             for d in HOLDERS if "rv05" not in d),
        MH_REP=sum(1 for d in DSN if max(mh[(d, h)]["k"] for h in SH) == mh[(d, 0)]["n"]),
        MH_SDW=f'{max(mh[(d, 0)]["turned_sd"] for d in HOLDERS if "rv05" not in d):.1f}',
        MH_SDB=f'{min(mh[(d, 0)]["turned_sd"] for d in HOLDERS if "rv05" not in d):.1f}',
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
    COL = {0: A, 1: B, 2: REF}
    # label anchors chosen per curve: the three settle curves cross, and parity-picked offsets
    # put two labels 17 px apart on the same x.
    MHLAB = [(3, 22), (1, 22), (3, -14)]
    c_mh = lines([(d.replace("_stored", ""),
                   [(h, mh[(d, h)]["settled"]) for h in SH], COL[i % 3],
                   None if i == 0 else "5 4", *MHLAB[i % 3])
                  for i, d in enumerate(sorted(HOLDERS, key=lambda d: mh[(d, 0)]["turned"]))],
                 "settling steps after the last commanded turn step",
                 "tool tilt from vertical (deg)", 900, 95) if HOLDERS else ""

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

    nan = lambda v, f="{:.2f}": ('<td class="num">&mdash;</td>' if v != v else num(v, f))

    # ---- ranked-population tables and chart -------------------------------------------------
    t_rkv = "".join(
        tr(f'<td><code>{t}</code></td>'
           + (nan(rk_v[t]["screen"] if rk_v[t]["screen"] is not None else float("nan"), "{:.3f}"))
           + num(rk_v[t]["replay"], "{:.3f}")
           + (f'<td class="num">{abs(rk_v[t]["screen"] - rk_v[t]["replay"]):.3f}</td>'
              if rk_v[t]["screen"] is not None
              else '<td class="num">no cell at this clip</td>'))
        for t in RORD)

    t_rk = "".join(
        tr(f'<td><code>{t}</code></td>'
           + f'<td class="num">{rk_s[(t, 0)]["held"]}/{rk_s[(t, 0)]["n"]}</td>'
           + num(rk_s[(t, 0)]["force"], "{:.1f}")
           + num(rk_s[(t, 0)]["turn"]) + num(rk_s[(t, 0)]["turn_sd"])
           + num(rk_s[(t, 2500)]["fin"]) + num(rk_s[(t, 2500)]["fin_sd"])
           + num(rk_s[(t, 0)]["turn"] - rk_s[(t, 2500)]["fin"], "{:+.1f}")
           + okcell(rk_s[(t, 2500)]["k"], rk_s[(t, 2500)]["n"]))
        for t in RORD)

    t_rkh = "".join(
        tr(f'<td><code>{t}</code></td>'
           + "".join(num(rk_s[(t, hs)]["fin"]) for hs in RHOLD)
           + num(rk_s[(t, 0)]["turn"] - rk_s[(t, 2500)]["fin"], "{:+.1f}"))
        for t in RORD)

    t_rko = "".join(
        tr(f'<td><code>{t}</code></td>' + num(rk_s[(t, 2500)]["fin"])
           + num(rk_o[(t, 2500)]["fin"])
           + num(rk_o[(t, 2500)]["fin"] - rk_s[(t, 2500)]["fin"], "{:+.2f}"))
        for t in RORD)

    t_rkq = "".join(
        tr(f'<td><code>{t}</code></td>' + num(rk_s[(t, 300)]["fin"])
           + "".join(num(rk_q[(t, q)]["fin"]) for q in RQ))
        for t in RORD)

    TH_RKH = "".join(f'<th class="num">{hs}</th>' for hs in RHOLD)
    # 8 curves in 3 colours is a tangle. Plot the three hands that bracket the behaviour --
    # the biggest settle toward vertical, the biggest fall back out, and the flattest -- and
    # leave the other five to the table.
    _set = {t: rk_s[(t, 0)]["turn"] - rk_s[(t, 2500)]["fin"] for t in RORD}
    RPICK = [max(RORD, key=lambda t: _set[t]), min(RORD, key=lambda t: abs(_set[t])),
             min(RORD, key=lambda t: _set[t])]
    c_rk = lines([(t.replace("sv1_", ""), [(hs, rk_s[(t, hs)]["fin"]) for hs in RHOLD],
                   (A, B, REF)[i], None if i == 0 else "5 4",
                   (4, 2, 2)[i], (22, -16, 22)[i])
                  for i, t in enumerate(RPICK)],
                 "settling steps after the last commanded turn step",
                 "tool tilt from vertical (deg)", 2500, 55) if RORD else ""

    HORD = sorted(HOLDERS, key=lambda d: mh[(d, 0)]["turned"])
    TH_MHH = "".join(f'<th class="num">{d.replace("_stored", "")}</th><th class="num">sd</th>'
                     for d in HORD)
    TH_MHQ = "".join(f'<th class="num">{d.replace("_stored", "")}</th><th class="num">ok</th>'
                     for d in HORD)
    t_mh = "".join(
        tr(f'<td><code>{d.replace("_stored", "")}</code></td>'
           + f'<td class="num">{mh[(d, 0)]["held"]}/{mh[(d, 0)]["n"]}</td>'
           + nan(mh[(d, 0)]["force"], "{:.1f}") + nan(mh[(d, 0)]["turned"])
           + nan(mh[(d, 0)]["turned_sd"])
           + nan(min((mh[(d, h)]["settled"] for h in SH), default=float("nan")))
           + okcell(max(mh[(d, h)]["k"] for h in SH), mh[(d, 0)]["n"]), "rv05" in d)
        for d in DSN)

    t_mhh = "".join(
        tr(f'<td class="num">{h}</td>'
           + "".join(nan(mh[(d, h)]["settled"]) + nan(mh[(d, h)]["settled_sd"])
                     for d in sorted(HOLDERS, key=lambda d: mh[(d, 0)]["turned"])), h == 150)
        for h in SH)

    t_mhq = "".join(
        tr(f'<td class="num">{q * 1000:.1f}</td>'
           + "".join(nan(mq[(d, q)]["turned"])
                     + okcell(mq[(d, q)]["k"], mq[(d, q)]["n"])
                     for d in sorted(HOLDERS, key=lambda d: mh[(d, 0)]["turned"])))
        for q in (0.0005, 0.0010, 0.0020))

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
            ("8/8 at 150 mm", "Chains completed with a 150 mm wrist stack, payload declared"),
            (f'{K["TURN_C"]}&#176;', f'Tilt at the end of the turn, payload declared (SD {K["SD_C"]}&#176;)'),
            (f'{K["TURN_U"]}&#176;', "Tilt at the end of the same turn, payload not declared"),
            ("0 mm", "Pedestal height the task needs"),
        ])

    body = f"""
<section id="terms"><div class="col">
<h2>Terms and metrics</h2>
<dl class="terms">
<dt><b>Chain</b></dt><dd>The simulated task on the UR5e: grasp the screwdriver, lift it, turn it upright, set it into a
45&#176; countersink and gait it (8 cycles of re-grasps that rotate it about its own axis).</dd>
<dt><b>Relay and release</b></dt><dd>The two gaits. The relay passes the tool between finger pairs without letting go;
the release lets go and re-grasps while the countersink holds the tool.</dd>
<dt><b>Completed (k/n)</b></dt><dd>Seeds, of n spawn-jittered rollouts, in which the chain finishes its 8 gait cycles
with the tool in the seat.</dd>
<dt><b>Wrist stack</b></dt><dd>Hardware between the tool flange and the palm plate (coupling, servo bank, wiring),
modelled as a 40&nbsp;mm cylinder at 700&nbsp;kg/m&#179;.</dd>
<dt><b>Payload declaration</b></dt><dd>Telling the arm controller the mass of the hand and the stack
(<code>set_payload</code> on the UR5e, gravity compensation in simulation).</dd>
<dt><b>Palm droop</b></dt><dd>Distance of the palm from its commanded position at the top of the lift, in mm.</dd>
<dt><b>Carry grip</b></dt><dd>Commanded interference of the pads with the tool during the carry, in mm; the grip force
follows from the commanded minus the achieved closure.</dd>
<dt><b>Tilt</b></dt><dd>Angle between the tool axis and vertical, in degrees, 0 upright; counted only while two pads hold
the tool clear of the seat.</dd>
<dt><b>Post-turn drift</b></dt><dd>Tilt at the last commanded step of the turn minus tilt after the hold that follows it,
in degrees; positive is toward vertical.</dd>
<dt><b>Deg/cy</b></dt><dd>Rotation of the tool about its own axis per gait cycle, in degrees.</dd>
<dt><b>Uncontacted</b></dt><dd>Share of the gait during which no pad touches the tool, in %.</dd>
<dt><b>Ranked hands</b></dt><dd>The eight hands of the 8,192-hand Sobol population promoted to exported plans; a tag
such as <code>sv1_u5860_b070</code> names one.</dd>
<dt><b>Confirmation cosine</b></dt><dd>Cosine of the tool axis with vertical in the screen&rsquo;s confirmation run; +1 is
upright.</dd>
</dl>
</div></section>

<section><div class="col">
<h2>Mount, wrist stack and payload in the arm scene</h2>

<p>Until 2026-09-04 the simulated chain ran with three fixed assumptions: the UR5e stood on a 300&nbsp;mm pedestal, the
palm plate was bolted to the tool flange, and the arm controller did not know the hand&rsquo;s mass. On the bench the robot
stands on the table that holds the work, about 100&nbsp;mm of coupling, servo bank and wiring sit between the flange and
the first finger joint, and the UR5e is told its payload with <code>set_payload</code>. All three are
now parameters of <code>build_real_v1_arm_scene.py</code> (<code>--base</code>, <code>--wrist-stack</code>,
<code>--stack-density</code>, <code>--payload-gravcomp</code>). A 100&nbsp;mm stack takes the hand from {K["PAY0"]} to
{K["PAY100"]}&nbsp;g.</p>

<figure><video src="{vid_wide}" autoplay loop muted playsinline></video>
<figcaption>The chain with the bench geometry: mount on the table surface, 100&nbsp;mm of wrist stack (the dark cylinder
above the palm plate), tool pressed into a 45&#176; countersink and gaited there without the hand releasing it.
</figcaption></figure>
</div></section>

<section><div class="col">
<h2>Mount height</h2>
<p class="sub">8 spawn-jittered seeds per cell, 8 gait cycles, no wrist stack</p>

<p>Mount heights from 0 to 300&nbsp;mm give the same results within one seed. At every height the IK solves the home
pose on the same branch, with the arm&rsquo;s lowest link, the shoulder, at +0.163&nbsp;m and no link below the table.
The differential IK reports {K["IKFAIL"]} unreachable poses in {K["NCELLS"]} rollouts, all at one standing
position (650&nbsp;mm back, 200&nbsp;mm across). The bench needs no pedestal.</p>

<div class="tw"><table>
<thead><tr><th class="num">Mount height (mm)</th><th class="num">Relay</th>
<th class="num">Release</th><th class="num">Relay deg/cy</th><th class="num">Release deg/cy</th>
<th class="num">Droop (mm)</th></tr></thead>
<tbody>{t_table}</tbody></table></div>
</div></section>

<section><div class="col">
<h2>Wrist stack with the payload declared</h2>
<p class="sub">Tool pressed into the 45&#176; seat, carry grip 0.3&nbsp;mm, robot base 500&nbsp;mm from the work</p>

<p>Declaring the payload takes the palm droop at the top of the lift from {K["DR100"]} to {K["DRC"]}&nbsp;mm at
100&nbsp;mm of stack. With the declaration, 0, 50, 100 and 150&nbsp;mm of stack all complete every seed at
{pay[(100, 0.30, "relay")]["deg"]:.0f}&nbsp;deg/cycle, and the tool ends within
{max(pay[(s, 0.30, "relay")]["tilt"] for s in (0, 50, 100, 150)):.1f}&#176; of vertical. Without it a 150&nbsp;mm stack
loses the tool on every seed.</p>

<div class="tw"><table>
<thead><tr><th class="num">Stack (mm)</th><th class="num">Payload (g)</th>
<th class="num">Relay, declared</th><th class="num">Deg/cy</th><th class="num">End tilt (&#176;)</th>
<th class="num">Relay, undeclared</th><th class="num">Droop (mm)</th></tr></thead>
<tbody>{t_pay}</tbody></table></div>

<figure>{c_pay}<figcaption>Chains completed against wrist-stack length, with the payload declared and without.
</figcaption></figure>

<p>The declaration also changes which grip works. On the sagging arm the set-down into the countersink needed more
commanded interference the longer the stack; on the compensated arm the published 0.3&nbsp;mm works and more fails,
because the extra closure torques the tool out of the seat on the way down and there is no deflection left for it to
make up.</p>

<div class="tw"><table>
<thead><tr><th>Wrist stack</th><th class="num" colspan="2">0.3 mm grip</th>
<th class="num" colspan="2">0.8 mm</th><th class="num" colspan="2">1.5 mm</th></tr>
<tr><th></th><th class="num">Relay</th><th class="num">Release</th><th class="num">Relay</th>
<th class="num">Release</th><th class="num">Relay</th><th class="num">Release</th></tr></thead>
<tbody>{t_paygrid}</tbody></table></div>

</div></section>

<section><div class="col">
<h2>Palm droop on an uncompensated wrist</h2>

<p>Without the declaration, stack length, stack mass and the robot&rsquo;s distance from the work each act on the chain
only through palm droop, and so would any compliant wrist. Droop is the same in every seed, so it orders the geometries
without the seed-to-seed scatter of the set-down.</p>

<div class="pair"><figure>{c_droop_s}<figcaption>Droop against stack length: about 6&nbsp;&micro;m per millimetre of
stack.</figcaption></figure><figure>{c_droop_x}<figcaption>Droop against the robot&rsquo;s distance from the work; a
folded arm deflects less.</figcaption></figure></div>

<p>With the carry grip swept at each geometry, the best cell orders by droop. Every seed completes up to
{K["CUT"]}&nbsp;mm of droop; 0.97&nbsp;mm gives 3 of 8 and 1.29&nbsp;mm none. The grip each geometry needs rises with
droop over the same range, because the grip force is commanded interference minus deflection: a wrist that deflects
more needs a larger command for the same force.</p>

<div class="tw"><table>
<thead><tr><th>Geometry</th><th class="num">Droop (mm)</th><th class="num">Best grip (mm)</th>
<th class="num">Relay</th><th class="num">Release</th><th class="num">Deg/cy</th></tr></thead>
<tbody>{t_best}</tbody></table></div>

<div class="pair"><figure>{c_best}<figcaption>Chains completed at the best grip against droop; every seed completes up
to {K["CUT"]}&nbsp;mm.</figcaption></figure><figure>{c_cs}<figcaption>Best carry grip against droop.
</figcaption></figure></div>

<p>Two further sweeps are in the data and not plotted. A massless 100&nbsp;mm stack completes every seed, so the
payload causes the failures and the reach does not. At the published grip, stack length breaks the relay at
50&nbsp;mm and leaves the release gait intact out to 125&nbsp;mm: the release rebuilds its grasp on a tool the seat
holds, and the relay places every pad on a tool the hand still holds.</p>

<div class="pair"><div class="tw"><table>
<thead><tr><th class="num">Stack (mm)</th><th class="num">Mass (g)</th><th class="num">Droop (mm)</th>
<th class="num">Relay</th><th class="num">Release</th></tr></thead>
<tbody>{t_stack}</tbody></table></div>
<div class="tw"><table>
<thead><tr><th class="num">Density (kg/m&#179;)</th><th class="num">Mass (g)</th><th class="num">Droop (mm)</th>
<th class="num">Relay</th><th class="num">Release</th></tr></thead>
<tbody>{t_mass}</tbody></table></div></div>
</div></section>

<section><div class="col">
<h2>Alignment from the turn and from the hold</h2>
<p class="sub">A new checkpoint, <code>turned</code>, taken at the last commanded step of the turn</p>

<p>The turn is an IK sweep of the finger anchor, clipped to &plusmn;0.5&nbsp;rad, followed by a hold with the finger
commands frozen. Alignment gained during the hold is the shaft rolling upright inside the grasp.</p>

<p>Without the declaration the turn ends {K["TURN_U"]}&#176; from vertical and the hold brings the tool back to
{K["SETTLE_U"]}&#176;. With the payload declared the turn ends <b>{K["TURN_C"]}&#176;</b> from vertical, SD
{K["SD_C"]}&#176; across spawn jitter, and the hold makes it worse. This section uses <code>rv05_manual</code>; the next repeats the measurement on
the ranked hands, where both numbers differ.</p>

<div class="tw"><table>
<thead><tr><th class="num">Hold steps</th><th class="num">Seconds</th>
<th class="num">Tilt, declared (&#176;)</th><th class="num">SD</th><th class="num">Completed</th>
<th class="num">Tilt, undeclared (&#176;)</th><th class="num">Completed</th></tr></thead>
<tbody>{t_hold}</tbody></table></div>

<figure>{c_hold}<figcaption>Tilt against hold length with the payload declared and without. Most of the 28&#176; of
post-turn drift the published chain relied on was the tool rolling back after the wrist sagged during the turn.
</figcaption></figure>

<h3>Prediction, control and measurement of the post-turn drift</h3>

<p><b>Prediction.</b> The tilt at the last commanded step has an SD of {K["SD_C"]}&#176; over spawn jitter, after 150
hold steps {wri[(True, 150)]["settle_sd"]:.2f}&#176; and after 900 steps {wri[(True, 900)]["settle_sd"]:.2f}&#176;: the
shaft rolls away from an unstable configuration and the seeds diverge.</p>

<p><b>Control.</b> The hold length lowers the tilt only up to a point: the best cell is 150 steps (0.3&nbsp;s) at
{K["BEST_HOLD_TILT"]}&#176;, and longer holds drift back out. Extra closure held through the sweep raises the tilt at the
last turn step from {slp[(0.0, 0)]["turned"]:.1f}&#176; to {slp[(2.0, 0)]["turned"]:.1f}&#176;, and no seed completes at
any of the four non-zero settings.</p>

<div class="tw"><table>
<thead><tr><th class="num">Closure through the turn (mm)</th>
<th class="num">Tilt at the last turn step (&#176;)</th><th class="num">After 300 steps (&#176;)</th>
<th class="num">Completed, 150-step hold</th><th class="num">Completed, 500-step hold</th></tr></thead>
<tbody>{t_tsq}</tbody></table></div>

<figure>{c_tsq}<figcaption>Tilt against the closure held through the turn.</figcaption></figure>

<p><b>Measurement.</b> The bench tracker reads this tilt at 0.017&#176; rms: the turn rotates the shaft about a
horizontal axis, so the vane tag stays in the camera&rsquo;s view (<code>morphohand.bench.tags</code>). The
<code>turned</code> checkpoint stays as a monitor that separates a hand that turned the tool from a wrist that sagged it
into place.</p>
</div></section>

<section><div class="col">
<h2>The same turn on the ranked hands</h2>
<p class="sub">The {K["RK_N"]} promoted hands of the Sobol&#8209;8192 population, each at its own operating point, in the
maneuver they were screened on; {K["RK_ROWS"]} rollouts</p>

<p>The design study ranks the population <code>20260831-real_v1-sobol8192</code>: 8,192 hands sampled, 535 selected,
227 confirmed and {K["RK_N"]} promoted to exported plans ({K["RK_NDES"]} distinct designs). Each runs here at its
screened operating point (straddle, thumb-axial offset, fitted grip depth and residual clip, from
<code>promotion.json</code>) through the screen&rsquo;s own planner and rollout: the tool on a 100&nbsp;mm platform,
palm fixed, close, pause, turn and hold, with the load regulator and servo torque ceilings of the retention screen. The
replay reproduces the confirmation cosine to three decimals on {K["RK_MATCH"]} of the {K["RK_HASCELL"]} tags confirmed
at their clip; a flat-table rebuild of the maneuver had put every hand at 78&ndash;89&#176; of tilt.</p>

<div class="tw"><table>
<thead><tr><th>Tag</th><th class="num">Confirmation cosine</th><th class="num">This replay</th>
<th class="num">Difference</th></tr></thead>
<tbody>{t_rkv}</tbody></table></div>

<p>On this population the turn ends between <b>{K["RK_LO"]}&#176;</b> and <b>{K["RK_HI"]}&#176;</b> from vertical, with a
seed SD of {K["RK_SDLO"]}&ndash;{K["RK_SDHI"]}&#176; against 0.42&#176; for <code>rv05_manual</code>. None of the eight
reaches 8&#176;, and three are repeatable within a degree. Every hand keeps the tool through the turn and the hold; the
ranked hands fall short on alignment.</p>

<div class="tw"><table>
<thead><tr><th>Tag</th><th class="num">Held at the turn</th><th class="num">Pad force (N)</th>
<th class="num">Tilt at the turn (&#176;)</th><th class="num">SD</th>
<th class="num">After the hold (&#176;)</th><th class="num">SD</th>
<th class="num">Post-turn drift (&#176;)</th><th class="num">Held at the end</th></tr></thead>
<tbody>{t_rk}</tbody></table></div>

<h3>Post-turn drift on the ranked hands</h3>

<p>On the ranked hands the post-turn drift ranges from {K["RK_SETTLE_HI"]}&#176; to {K["RK_SETTLE_LO"]}&#176;: the hold improves the alignment on
{K["RK_HELPED"]} of {K["RK_N"]} hands, and on the rest the shaft falls back out of the alignment the turn produced. The
tilt is still changing at 900 steps on every hand, so the screened hold of 2,500 steps matters and the 150-step optimum
of <code>rv05_manual</code> does not transfer.</p>

<div class="tw"><table>
<thead><tr><th>Tag</th>{TH_RKH}<th class="num">Post-turn drift</th></tr></thead>
<tbody>{t_rkh}</tbody></table></div>

<p class="sub">Tilt from vertical after N hold steps, in degrees, mean of 5 seeds</p>

<figure>{c_rk}<figcaption>Tilt against hold length for three of the {K["RK_N"]} hands: the largest drift toward
vertical, the flattest, and the one that falls back out.</figcaption>
</figure>

<h3>Load regulator and squeeze</h3>

<p>The deployed maneuver has one feedback loop: a regulator that trims each finger&rsquo;s flexion to hold its share of
the servo load, active only in the hold. Switching it off moves the tilt after the hold by at most
{K["RK_REGMAX"]}&#176; on any hand. The post-turn drift is gravity acting through a grip that has weakened, and the
regulator does not change it.</p>

<div class="pair"><div class="tw"><table>
<thead><tr><th>Tag</th><th class="num">Regulator on</th><th class="num">Off</th>
<th class="num">&Delta;</th></tr></thead>
<tbody>{t_rko}</tbody></table></div>
<div class="tw"><table>
<thead><tr><th>Tag</th><th class="num">No squeeze</th><th class="num">0.5&nbsp;mm</th>
<th class="num">1&nbsp;mm</th><th class="num">2&nbsp;mm</th></tr></thead>
<tbody>{t_rkq}</tbody></table></div></div>

<p>Re-gripping at the top of the turn gives the result it gave on <code>rv05_manual</code>: 2&nbsp;mm of extra closure
leaves every hand further from vertical than no squeeze. On the chain the squeeze stopped every seed from completing;
here all five seeds complete on every hand, with worse alignment. The fingers turn the shaft by letting it roll, and a
tighter grip stops the rolling.</p>

<h3>Five-design family and the contact gate on tilt</h3>

<p>An earlier version of this section ran the five-design family <code>results/phase1/real_v1/rv0*</code> through
the chain: {K["MH_HELD"]} of {K["MH_N"]} designs still held the tool at the last turn step and {K["MH_CHAIN"]} finished
the chain in any seed. In {K["MH_FALSE_N"]} of 8 seeds <code>rv04_mid_sp40</code> reaches the <code>turned</code> checkpoint with no pad on the
tool and the tool at z&nbsp;=&nbsp;{K["MH_FALSE_Z"]}&nbsp;mm, the depth of the countersink. It dropped the shaft during
the turn and the shaft landed in its seat, which is vertical by construction, so the checkpoint reads
{K["MH_FALSE_MIN"]}&#176; of tilt at best and {K["MH_FALSE"]}&#176; on average, better than the hand that turned the
tool. Every tilt on this page is therefore counted only while two pads hold the tool clear, and the bench tracker needs
the same gate: a seated tool and a held one look equally vertical to the camera.</p>

<div class="tw"><table>
<thead><tr><th>Design</th><th class="num">Held at the turn</th><th class="num">Pad force (N)</th>
<th class="num">Tilt (&#176;)</th><th class="num">SD</th>
<th class="num">Best after the hold (&#176;)</th><th class="num">Chain</th></tr></thead>
<tbody>{t_mh}</tbody></table></div>
</div></section>

<section><div class="col">
<h2>Load torque the gait can drive</h2>
<p class="sub">A Coulomb brake about the tool&rsquo;s own axis, active once the tool is seated</p>

<p>A countersink acts as a friction brake: it returns &mu;<i>Nr</i>, and the press sets <i>N</i> through the wedge, so
the resisting torque in every earlier gait was whatever the geometry supplied, about 7&nbsp;mN&middot;m on a flat plane.
<code>--screw-torque</code> adds a set torque that opposes any rotation and does no work when the tool is still.</p>

<div class="tw"><table>
<thead><tr><th class="num">Torque (mN&middot;m)</th><th class="num">Relay</th>
<th class="num">Release</th><th class="num">Relay deg/cy</th>
<th class="num">Release, % uncontacted</th></tr></thead>
<tbody>{t_tq}</tbody></table></div>

<figure>{c_tq}<figcaption>At the published 0.3&nbsp;mm grip the relay turns the tool
{tq[(0.004, "relay")]["deg"]:.0f}&nbsp;deg/cycle against 4&nbsp;mN&middot;m and starts losing seeds at
8&nbsp;mN&middot;m.</figcaption></figure>

<p>A larger commanded grip does not raise that limit. With torque crossed with <code>--carry-squeeze</code> on one
scene, 0.8&nbsp;mm completes fewer seeds than 0.3&nbsp;mm at every torque including zero, as in the grip sweep
above.</p>

<div class="tw"><table>
<thead><tr><th class="num">Torque (mN&middot;m)</th><th class="num">0.3 mm grip</th>
<th class="num">Deg/cy</th><th class="num">0.8 mm grip</th><th class="num">Deg/cy</th></tr></thead>
<tbody>{t_tqg}</tbody></table></div>

<p>On the sagging arm the same relay kept its rate up to 20&nbsp;mN&middot;m at 1.5&nbsp;mm of commanded grip. The
two arms differ in pad force and wrist stiffness at once, and the chain does not measure pad force, so the drivable
torque of this hand has no number yet.</p>

<div class="callout"><span class="tag">Range of the brake and a bench equivalent</span>
<p>The tool is 24&nbsp;g with an axial inertia of 1.9&times;10<sup>&#8722;6</sup>&nbsp;kg&middot;m&sup2;, so a brake
much above 20&nbsp;mN&middot;m can overshoot zero inside a timestep and pump the shaft. A pumped rollout (shaft turning up to
&plusmn;450&nbsp;deg/cycle) is flagged; {K["PUMPED"]} cells trip the flag here ({K["PUMPED_AT"]}), the end of the valid
range. On the bench a cone cannot supply a
set torque, and the press and the seat material are the only settings. A calibrated brake needs the socket on a short
shaft in a bushing with a preloaded friction washer, whose preload screw sets a torque a wrench can read.</p></div>
</div></section>

<section><div class="col">
<h2>Future work</h2>

<ol>
<li><b>Bench the post-turn drift on <code>w5120_b070</code> and <code>u5860_b070</code></b>, the two ends of its range:
{rk_s[(RPICK[0], 0)]["turn"]:.0f}&#176; &rarr; {rk_s[(RPICK[0], 2500)]["fin"]:.0f}&#176; over the hold on one and
{rk_s[(RPICK[2], 0)]["turn"]:.0f}&#176; &rarr; {rk_s[(RPICK[2], 2500)]["fin"]:.0f}&#176; on the other, from the same
commanded schedule. Both have exported plans, and the vane tag reads this rotation at 0.017&#176; rms. The simulation
predicts a signed difference of 30&#176;; a bench difference near zero refutes the mechanism.</li>

<li><b>Sweep the post-turn drift over the 227 confirmed hands</b> with the screen (about 3&nbsp;s a rollout, one
overnight job). The eight promoted hands span 46&#176; of drift; the sweep shows whether the drift can be a design
variable.</li>

<li><b>Gate every tilt report on contact</b>: add <code>held_turn</code> (two pads, tool above 80&nbsp;mm, one line in
<code>probe_real_v1_chain</code>) to the scorecard and to the bench tracker&rsquo;s tilt read.</li>

<li><b>Cross <code>--screw-torque</code> with <code>--carry-squeeze</code></b> on one scene and report the drivable
torque per millimetre of commanded interference; the change from 4&ndash;12 to 20&nbsp;mN&middot;m between the two arms
mixes grip with wrist stiffness.</li>

<li><b>Close the grip loop.</b> The finger servos share one spring constant and their positions are readable at the
control rate, so pad force can be estimated from deflection and <code>--carry-squeeze</code> trimmed to a force
target.</li>

<li><b>Run the relay&rsquo;s Cartesian legs through <code>scripts/real_v1_trajectory_clearance.py</code></b> and the
servo-limit check before exporting a plan; three of the four plans that reached the control station failed the
clearance check.</li>

<li><b>Add <code>slip_mm_per_cycle</code> to the scorecard</b> before attempting a gait that keeps the grasp; the
floor-free relay lowers the tool 11&nbsp;mm per cycle and no metric reports it.</li>
</ol>
</div></section>
"""

    tpl = (ROOT / "scripts/real_v1_bench_page.template.html").read_text()
    html = tpl.replace("{{NCELLS}}", str(K["NCELLS"])).replace(
        "{{STATS}}", stats).replace("{{BODY}}", body)
    out.write_text(retro_style.apply(html))  # plain page style (owner, 2026-10-09)
    print(f"-> {out}  {len(html) / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
