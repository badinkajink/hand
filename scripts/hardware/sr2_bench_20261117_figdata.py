#!/usr/bin/env python3
"""Figure data of the SR2 hand bench session of 2026-11-17 (page: scripts/hardware/sr2_bench_20261117_page.py), from the
committed trial records. Runs on corvus under /usr/bin/python3: the finger model comes from the bench study's
lab/hom_setup.py (finger_pts: the tip pad, the tip's end and the PIP joint as balls on SR2Hand's chain).

    /usr/bin/python3 scripts/hardware/sr2_bench_20261117_figdata.py

Writes docs/experiments/20261117-sr2_bench_session/data/20261117-figdata.json:
  gaps    per twist trial, the least distance between the index's and the middle's parts on the hand's model (balls:
          tip pad radius 8 mm, tip end 6 mm, PIP joint 8 mm, at the plan's mounts), at the measured joints and at HOM's
          targets, 10 Hz
  press   per press trial, the middle finger's MCP and PIP, target and measured, 50 Hz, and the chatter: the RMS about a
          0.5 s moving mean and the residual's zero crossings per second over two
  slip    per pick-up, each squeeze step of the slip with the object's tilt at the step's end
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / "docs/experiments/20261115-hom_bench_trials"
sys.path[:0] = [str(ROOT / "src"), str(B / "lab")]
sys.argv = sys.argv[:1]
import hom_setup as HS  # noqa: E402

OUT = ROOT / "docs/experiments/20261117-sr2_bench_session/data/20261117-figdata.json"
SIGN = np.array([1, 1, 1, -1, 1, 1, -1, 1, 1.0])          # servo -> simulator degrees (the index's and middle's yaw)
PLAN_G13 = "plans/20261116-film_x105_y70_twist_gr_plan.json"
PLAN_G0 = "plans/20261116-film_x70_y110_twist_gr_plan.json"
TWISTS = [  # record, plan for the mounts, label
    ("20261116-130318", PLAN_G13, "11-16 13:03, g13, 5 V, tips off"),
    ("20261117-091512", PLAN_G13, "11-17 09:15, g13, 5 V, tips off"),
    ("20261117-092101", PLAN_G13, "11-17 09:21, g13, 5 V, tips on"),
    ("20261117-170910", PLAN_G13, "11-17 17:09, g13, 7.5 V, tips off"),
    ("20261117-171148", PLAN_G13, "11-17 17:11, g13, 7.5 V, tips on"),
    ("20261117-181327", PLAN_G13, "11-17 18:13, g13, 7.5 V, tips off"),
    ("20261117-182114", PLAN_G0, "11-17 18:21, g0, 7.5 V, tips off"),
    ("20261117-182434", PLAN_G0, "11-17 18:24, g0, 7.5 V, tips on"),
]
PRESSES = [
    ("20261116-131927", "11-16 13:19, lat0, 5 V, HOM gain 0.55"),
    ("20261117-183028", "11-17 18:30, lat0, 7.5 V, HOM gain 0.55"),
    ("20261117-184432", "11-17 18:44, lat0 re-planned open pose, 7.5 V, HOM gain 0.825"),
    ("20261117-184907", "11-17 18:49, lat0 re-planned open pose, 7.5 V, HOM gain 0.55"),
]
SLIPS = [
    (B / "data/20261116-pinch_pickup/20261116-154534_pinch_slip_telemetry.jsonl", 5.0, "11-16 15:45, thumb-middle, 5 V"),
    (B / "data/20261117-pinch_pickup/20261117-180709_pinch_slip_telemetry.jsonl", 7.5, "11-17 18:07, thumb-middle, 7.5 V"),
    (B / "data/20261117-pinch_pickup/20261117-184007_pinch_slip_telemetry.jsonl", 7.5, "11-17 18:40, thumb-index, 7.5 V"),
]


def jl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def q3(v, i):
    return {"yaw": float(v[i]), "mcp": float(v[i + 1]), "pip": float(v[i + 2])}


def gap(plan, qi, qm):
    pi, pm = HS.finger_pts(plan, "index", qi), HS.finger_pts(plan, "middle", qm)
    r = lambda part, rad: 8.0 if part == "tip pad" else rad          # noqa: E731
    return min(float(np.linalg.norm(a - b)) - r(na, ra) - r(nb, rb)
               for (a, ra), na in zip(pi, HS.PARTS) for (b, rb), nb in zip(pm, HS.PARTS))


def gaps():
    out = []
    for stem, plan_f, label in TWISTS:
        rows = jl(B / f"runs_corvus/{stem}_twist_corvus.jsonl")
        plan = HS.load_plan(str(B / plan_f))
        st = [r for r in rows if "seq" in r and r.get("joints")]
        t, gm, gt = [], [], []
        for r in st[::5]:
            js = np.asarray(r["joints"], float) * SIGN
            t.append(round(float(r["t"]), 3))
            gm.append(round(gap(plan, q3(js, 3), q3(js, 6)), 2))
            tg = r.get("target_sim_deg")
            gt.append(round(gap(plan, q3(tg, 3), q3(tg, 6)), 2) if tg else None)
        end = rows[-1] if rows[-1].get("kind") == "end" else {}
        out.append({"record": stem, "label": label, "t": t, "measured_mm": gm, "target_mm": gt,
                    "least_measured_mm": min(gm), "least_target_mm": min(x for x in gt if x is not None),
                    "states": end.get("states")})
        print(f"{label}: least {min(gm):.1f} mm measured, {out[-1]['least_target_mm']:.1f} mm at the targets")
    return out


def press():
    out = []
    for stem, label in PRESSES:
        rows = jl(B / f"runs_corvus/{stem}_press_corvus.jsonl")
        st = [r for r in rows if "seq" in r and r.get("joints") and r.get("target_sim_deg")]
        t = np.array([r["t"] for r in st])
        J = np.array([r["joints"] for r in st], float) * SIGN
        T = np.array([r["target_sim_deg"] for r in st], float)

        def osc(x, k=25):
            m = np.convolve(x, np.ones(k) / k, mode="same")
            res = (x - m)[k:-k]
            return float(res.std()), float(np.sum(np.diff(np.sign(res)) != 0) / (t[-k - 1] - t[k]) / 2.0)
        rec = {"record": stem, "label": label, "t": np.round(t, 3).tolist()}
        for i, n in ((7, "mcp"), (8, "pip")):
            rec[f"{n}_target"] = np.round(T[:, i], 2).tolist()
            rec[f"{n}_measured"] = np.round(J[:, i], 2).tolist()
            (rt, ft), (rm, _) = osc(T[:, i]), osc(J[:, i])
            rec[f"{n}_chatter"] = {"target_rms_deg": round(rt, 2), "target_hz": round(ft, 1), "measured_rms_deg": round(rm, 2)}
        out.append(rec)
        print(f"{label}: middle MCP target {rec['mcp_chatter']['target_rms_deg']} deg rms at "
              f"{rec['mcp_chatter']['target_hz']} Hz, measured {rec['mcp_chatter']['measured_rms_deg']}")
    return out


def slip():
    out = []
    for path, volts, label in SLIPS:
        rows = jl(path)
        marks = [(i, r) for i, r in enumerate(rows) if r.get("mark")]
        steps = []
        for k, (i, r) in enumerate(marks):
            sq = float(str(r["mark"][0]).split()[1])
            end = marks[k + 1][0] - 1 if k + 1 < len(marks) else len(rows) - 1
            tilts = [rows[x]["obj"]["tilt"] for x in range(i, end + 1) if rows[x].get("obj") and "tilt" in rows[x]["obj"]]
            if tilts:
                steps.append({"squeeze_deg": sq, "squeeze_5v_deg": round(sq * volts / 5.0, 2),
                              "tilt_start_deg": round(tilts[0], 1), "tilt_end_deg": round(tilts[-1], 1)})
        out.append({"label": label, "volts": volts, "file": str(path.relative_to(ROOT)), "steps": steps})
        print(f"{label}: {len(steps)} steps, tilt {steps[0]['tilt_start_deg']} -> {steps[-1]['tilt_end_deg']} deg")
    return out


def tracking():
    """Every trial of 2026-11-17 scored by scripts/hardware/hom_trial_tracking.py (the bench's tracking measure) from
    the committed server and corvus records; its table goes to data/20261117-trial_tracking.json beside the figure data."""
    sys.path[:0] = [str(ROOT / "scripts/hardware"), str(ROOT / "scripts")]
    import hom_trial_tracking as T
    T.OUT = OUT.parent / "20261117-trial_tracking.json"
    argv = sys.argv
    sys.argv = [argv[0], "--since", "20261117-000000"]
    try:
        T.main()
    finally:
        sys.argv = argv
    return str(T.OUT.relative_to(ROOT))


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d = {"made_by": "scripts/hardware/sr2_bench_20261117_figdata.py", "gaps": gaps(), "press": press(), "slip": slip(),
         "tracking_file": tracking()}
    OUT.write_text(json.dumps(d))
    print(f"-> {OUT} ({OUT.stat().st_size / 1e3:.0f} kB)")
