#!/usr/bin/env python3
"""Collect the 2026-09-16 turn-mechanism study into one JSON for the page builder.

    python3 scripts/turn_mechanism_data.py
    python3 scripts/turn_mechanism_page.py

Sources (all under docs/experiments/):
  20260916-turn_mechanism/rv05_manual_b85_mechanism.json      plant x post x gravity, rv05 bench replay
  20260916-turn_mechanism/rv05_manual_b85_mechanism_kv.json   kv sweep at kp 0.5
  20260916-turn_mechanism/bench_plants.json                   8 hands x 3 plants vs the bench's turn
  20260916-turn_mechanism/kp_refit_kv0.02.json                kp re-fit with the servo's speed
  20260829-real_v1_bench_gripwindow/*-stepped.jsonl           hardware per-step cmd vs achieved
  20260916-plant_kv/chain_hands.json                          4 hands x 3 plants x 2 pivots x 2 seeds
  20260916-plant_kv8/chain_hands.json                         16 hands on the fast plant
  20260916-plant_angle0/chain_hands.json                      8 hands, arm only
  20260916-plant_relief_fast/chain_hands.json                 8 hands, turn relief 4 / 8 mm
  20260916-plant_planturn/chain_hands.json                    8 hands at each plan's own turn
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "docs/experiments"
OUT = E / "20260916-turn_mechanism/turn_mechanism.json"
D = {"sv1_w6689_b060": "D1", "sv1_w2360_b075": "D2", "sv1_u1364_b080": "D3", "g12_b095": "D4",
     "sv1_u0060_b75": "D5", "sv1_u0308_b050": "D6", "rv05_manual_b85": "D7", "sv1_w0099_b100": "D8"}
HW = {"D1": ("6/7", 51.9), "D2": ("10/10", 48.2), "D3": ("4/10", 42.1), "D4": ("9/10", 52.8),
      "D5": ("3/10", 69.6), "D6": ("2/10", 35.5), "D7": ("10/10", 33.3), "D8": ("4/10", 44.2)}
WEIGHT = 0.240


def seam(r, name):
    for s in r.get("seams") or []:
        if s.get("phase") == name:
            return s
    return {}


def held_staged(r):
    s = seam(r, "staged")
    return bool(s and s.get("cos", 0) > 0.85 and s.get("pad_contacts", 0) >= 2
                and s.get("pad_force_N", 0) >= WEIGHT)


def turn_deg(s):
    c = s.get("cos")
    return None if c is None else round(90.0 - float(np.degrees(np.arccos(np.clip(c, -1, 1)))), 1)


def chain_rows(path, key=lambda r: (r["plant"], r["axis_k"])):
    j = json.load(open(path))
    out = {}
    for r in j["rows"]:
        if r.get("error"):
            continue
        hand = D.get(r["tag"], r["tag"])
        k = (hand,) + tuple(key(r))
        L, T, S = seam(r, "lifted"), seam(r, "turned"), seam(r, "staged")
        out.setdefault(k, []).append({
            "seed": r["seed"], "arm": r["arm"],
            "lift_N": L.get("pad_force_N"), "lift_pads": L.get("pad_contacts"), "lift_cos": L.get("cos"),
            "turn_cos": T.get("cos"), "turn_deg": turn_deg(T), "turn_N": T.get("pad_force_N"),
            "turn_pads": T.get("pad_contacts"), "turn_slide": T.get("slide_mm"),
            "q_err": T.get("q_err_deg"),
            "staged_cos": S.get("cos"), "staged_N": S.get("pad_force_N"), "staged_pads": S.get("pad_contacts"),
            "held_staged": held_staged(r), "turns": r.get("turns"), "drop_stage": r.get("drop_stage"),
            "ok": r.get("ok")})
    return out


def summarise(cells):
    rows = []
    for k, rs in sorted(cells.items()):
        def med(f):
            v = [r[f] for r in rs if r[f] is not None]
            return round(float(np.median(v)), 3) if v else None
        qe = [r["q_err"] for r in rs if r["q_err"]]
        rows.append({"key": list(k), "n": len(rs),
                     "lift_N": med("lift_N"), "lift_pads": med("lift_pads"),
                     "turn_deg": med("turn_deg"), "turn_N": med("turn_N"), "turn_pads": med("turn_pads"),
                     "turn_slide": med("turn_slide"),
                     "q_err_max": round(float(np.median([max(q.values()) for q in qe])), 1) if qe else None,
                     "staged_cos": med("staged_cos"), "staged_N": med("staged_N"), "staged_pads": med("staged_pads"),
                     "held_staged": sum(r["held_staged"] for r in rs),
                     "turns": med("turns"),
                     "drop": sorted({str(r["drop_stage"]) for r in rs})})
    return rows


def hw_lag():
    """Per-step commanded vs achieved on the three 2026-08-29 stepped g12 runs."""
    out = []
    for run, label in (("205525", "free air, no shaft"), ("211644", "shaft, grip relieved (dropped)"),
                       ("212231", "shaft, thumb firm / middle free (held)")):
        f = E / f"20260829-real_v1_bench_gripwindow/20260829-{run}-g12-stepped.jsonl"
        rows = [json.loads(l) for l in open(f)]
        meta = rows[0]
        steps = [r for r in rows if r["kind"] == "step"]
        t = np.array([s["t_s"] for s in steps])
        joints = []
        for fn in ("thumb", "index", "middle"):
            for jn in ("yaw", "mcp", "pip"):
                c = np.array([s["joints"][fn]["cmd"][jn] for s in steps])
                g = np.array([s["joints"][fn]["got"][jn] for s in steps])
                if abs(c[-1] - c[0]) < 5:
                    continue
                e = g - c
                joints.append({"joint": f"{fn}_{jn}", "travel": round(float(c[-1] - c[0]), 1),
                               "rate": round(float((c[-1] - c[0]) / (t[-1] - t[0])), 2),
                               "err_med": round(float(np.median(e)), 1), "err_last": round(float(e[-1]), 1),
                               "load_med": round(float(np.median([s["joints"][fn].get("yaw_load", 0) for s in steps])))})
        out.append({"run": run, "label": label, "dwell_s": meta["dwell_s"], "gate_deg": meta["gate_deg"],
                    "step_s": round(float(np.median(np.diff(t))), 3), "total_s": round(float(t[-1] - t[0]), 1),
                    "joints": joints})
    return out


def main() -> int:
    tm = E / "20260916-turn_mechanism"
    mech = json.load(open(tm / "rv05_manual_b85_mechanism.json"))
    mech_kv = json.load(open(tm / "rv05_manual_b85_mechanism_kv.json"))
    mech_kv002 = json.load(open(tm / "rv05_manual_b85_mechanism_kv002.json"))
    mech_kp = json.load(open(tm / "rv05_manual_b85_mechanism_kp025.json"))

    def strip(r):
        return {k: v for k, v in r.items() if k != "trace"}

    def trace(r, every=4):
        tr = r["trace"]
        return [{"t": s["t"], "cos": s["cos"], "z": s["z"], "f_pad": round(sum(s["f_pad"].values()), 3),
                 "n_pad": sum(1 for v in s["n_pad"].values() if v), "f_post": s["f_post"],
                 "qe_mid_pip": s["q_err_deg"]["middle_pip"], "qe_mid_yaw": s["q_err_deg"]["middle_yaw"],
                 "qe_idx_yaw": s["q_err_deg"]["index_yaw"], "phase": s["phase"]}
                for i, s in enumerate(tr) if i % every == 0 or s["phase"] != "turn"]

    data = {
        "bench_mechanism": [strip(r) for r in mech],
        "bench_kv": [strip(r) for r in mech_kv],
        "bench_kv002": [strip(r) for r in mech_kv002 + mech_kp],
        "bench_trace": {r["plant"]: trace(r) for r in mech if r["post"] and r["gravity"]
                        and r["plant"] in ("shipped", "corrected")},
        "bench_trace_kv": {r["plant"]: trace(r) for r in mech_kv if r["plant"] == "kv0.01"},
        "hw_lag": hw_lag(),
        "chain_kv": summarise(chain_rows(E / "20260916-plant_kv/chain_hands.json")),
        "chain16": summarise(chain_rows(E / "20260916-plant_kv8/chain_hands.json",
                                        key=lambda r: (r["axis_k"],))),
        "chain_angle0": summarise(chain_rows(E / "20260916-plant_angle0/chain_hands.json",
                                             key=lambda r: ())),
        "chain_soft": summarise(chain_rows(E / "20260916-plant_soft/chain_hands.json",
                                           key=lambda r: ())),
        "chain_relief": summarise(chain_rows(E / "20260916-plant_relief_fast/chain_hands.json",
                                             key=lambda r: (r.get("turn_relief") or float(r["arm"].split("_l")[-1]) / 1000.0,))),
        "hw": HW,
    }
    p = E / "20260916-plant_planturn/chain_hands.json"
    if p.exists():
        data["chain_planturn"] = summarise(chain_rows(p, key=lambda r: (r["angle_deg"], r["axis_k"])))
    p = tm / "bench_plants.json"
    if p.exists():
        data["bench_plants"] = json.load(open(p))
    p = tm / "kp_refit_kv0.02.json"
    if p.exists():
        data["kp_refit"] = json.load(open(p))
    p = E / "20260902-servo-sysid/kp_calibration.json"
    if p.exists():
        data["kp_fit_kv0.6"] = json.load(open(p))
    n = sum(len(v) for v in (chain_rows(E / "20260916-plant_kv/chain_hands.json"),))
    data["n_rollouts"] = {
        "chain": sum(len(json.load(open(E / f"20260916-{d}/chain_hands.json"))["rows"])
                     for d in ("plant_kv", "plant_kv8", "plant_angle0", "plant_relief_fast", "plant_planturn", "plant_soft")
                     if (E / f"20260916-{d}/chain_hands.json").exists()),
        "bench": len(mech) + len(mech_kv) + len(mech_kv002) + len(mech_kp) + (len(data.get("bench_plants", [])) * 4),
    }
    OUT.write_text(json.dumps(data, indent=1))
    print(f"wrote {OUT}  chain rollouts {data['n_rollouts']['chain']}  bench replays {data['n_rollouts']['bench']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
