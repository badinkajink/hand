#!/usr/bin/env python3
"""Contact-model comparison bed, task 5 (T5 brake): the pinch as a friction hinge that lowers the tool.

A wrapper around hom_contact_rig.exp_brake, which it runs unchanged with the settings of the 10-01 study
(docs/experiments/20261001-hom_contact_patch/brake.jsonl): gravity on, CG offset d = 15 mm from the pinch line,
the tool horizontal and weight-compensated while the pads settle at 6 N, then released, held at 6 N for 0.5 s,
the pinch lowered geometrically to 0.2 N over 4 s and held 1 s. The wrapper patches `make_rig` so the rig
times every physics step and records the tool's angle and rate after each one (exp_brake's own trace is
every 10 steps, which is 50 ms at a 5 ms step), and patches `FilmRig` with the bed's film renderer.

Row metrics: swing end angle and maximum (phi, 0 horizontal, 90 hanging), time and pinch force at which the
swing passes 10, 45 and 80 deg (time from the release of the weight compensation), peak swing speed, slip of
the tool through the pinch, and whether the tool stays pinched (both pads above half the final 0.2 N) at
its grasp station (slip under 5 mm). At 1 ms each row carries the 10-01 row of the same rig spec and the
differences from it; `mj_pads05_tr02` (the 10-01 0.5 mm pad, relaxation 0.02 s) runs only for that check,
since the bed's mj_pads05 uses relaxation 0.03 s.

    PY=logs/20261001-hom_contact/venv/bin/python
    $PY scripts/contact_bed_brake.py --models drake_hydro --dt 1        # one case
    $PY scripts/contact_bed_brake.py                                    # the protocol grid, films, tile
"""
from __future__ import annotations

import argparse
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contact_bed_common as B  # noqa: E402

H = B.H
D_CG, N0, N1 = 0.015, 6.0, 0.2
T_HOLD, T_RAMP, T_END = 0.5, 4.0, 1.0
OUT = B.BED / "brake.jsonl"
REF = B.ROOT / "docs/experiments/20261001-hom_contact_patch/brake.jsonl"
REPRO = {"mj_pads05_tr02": "mj:spheres:s0.5:rs0.75:ir100:tr0.02"}
VIEW = ((0.0, 0.012, -0.015), 0.16, 18.0, -12.0)
FILM_DT = 1.0
STATION_MM = 5.0


def n_sched(t):
    if t < T_HOLD:
        return N0
    if t < T_HOLD + T_RAMP:
        return N0 * (N1 / N0) ** ((t - T_HOLD) / T_RAMP)
    return N1


def ref_rows():
    out = {}
    for r in B.read_rows(REF):
        if (r.get("d_mm"), r.get("N0"), r.get("N1"), r.get("T_ramp")) == (D_CG * 1e3, N0, N1, T_RAMP):
            out[r["spec"]] = r
    return out


def crossing(t, phi, a):
    """First time phi reaches a, interpolated between samples."""
    idx = np.where(phi >= a)[0]
    if not len(idx):
        return None
    i = int(idx[0])
    if i == 0:
        return float(t[0])
    f = (a - phi[i - 1]) / max(phi[i] - phi[i - 1], 1e-12)
    return float(t[i - 1] + f * (t[i] - t[i - 1]))


def run_case(model, dt_ms, film=False):
    spec = B.MODELS[model][0] if model in B.MODELS else REPRO[model]
    dt = dt_ms * 1e-3
    t_wall = time.time()
    H.DT = dt
    fine = []

    def on_step(rig, k):
        st = rig.tool_state()
        fine.append((rig.t, st["theta"], st["omega_x"]))
    films = []
    orig_film = H.FilmRig
    if film:
        def make_film(spec_, d_, w, h):
            f = B.BedFilm(spec_, d_, w, h, model=model, view=VIEW, inset_side="bottom-right", speed=round(40 * dt * 25, 3))
            films.append(f)
            return f
        H.FilmRig = make_film
    try:
        with B.StepProbe(on_step) as P:
            res = H.exp_brake(spec, D_CG, N0, N1, T_hold=T_HOLD, T_ramp=T_RAMP, T_end=T_END, traces=True,
                              keep_frames=film, film_wh=(480, 360))
    finally:
        H.FilmRig = orig_film
    frames = res.pop("_frames", None)
    for f in films:
        f.close()

    F = np.array(fine)
    t = F[:, 0] - P.settle_t
    phi = -np.degrees(F[:, 1] - P.settle_state["theta"])
    rate = np.degrees(-F[:, 2])                              # swing speed, positive toward hanging
    tr = np.array(res["trace"])
    last = tr[-1]
    NL, NR, nL, nR, dz_end = float(last[5]), float(last[6]), int(last[7]), int(last[8]), float(last[9])
    both = NL > 0.5 * N1 and NR > 0.5 * N1 and nL > 0 and nR > 0
    row = {"task": "brake", "model": model, "rig_spec": spec,
           "chain_spec": B.MODELS[model][1] if model in B.MODELS else None, "N": None, "dt_ms": dt_ms,
           "d_mm": D_CG * 1e3, "N0": N0, "N1": N1, "T_hold_s": T_HOLD, "T_ramp_s": T_RAMP, "T_end_s": T_END,
           "mu": H.MU, "gravity": True, "role": "repro_1001" if model in REPRO else "bed",
           "phi_end_deg": float(phi[-1]), "phi_max_deg": float(phi.max()),
           "overshoot_deg": float(phi.max() - phi[-1]),
           "peak_rate_deg_s": float(np.abs(rate).max()), "t_peak_rate_s": float(t[int(np.argmax(np.abs(rate)))])}
    for a in (10, 45, 80):
        ta = crossing(t, phi, a)
        row[f"t_{a}_s"] = ta
        row[f"N_at_{a}_N"] = n_sched(ta) if ta is not None else None
    row.update(slip_end_mm=res["slip_end_mm"], slip_max_mm=res["slip_max_mm"], N_L_end=NL, N_R_end=NR,
               n_L_end=nL, n_R_end=nR, dz_end_mm=dz_end,
               dz_rigid_swing_mm=-D_CG * math.sin(math.radians(float(phi[-1]))) * 1e3,
               pinched_end=bool(both), held_station=bool(both and abs(res["slip_end_mm"]) < STATION_MM),
               us_per_step_median=float(np.median(P.W) * 1e6), us_per_step_mean=float(np.mean(P.W) * 1e6),
               n_steps=len(P.W))
    # exp_brake's own (10-step) numbers, and the 10-01 row of the same spec for the reproduction check
    row["exp_brake"] = {k: res[k] for k in ("phi_end", "phi_max", "dropped", "slip_end_mm", "slip_max_mm",
                                            "N_at_10", "t_at_10", "N_at_45", "t_at_45", "N_at_80", "t_at_80",
                                            "peak_rate_dps")}
    ref = ref_rows().get(spec) if dt_ms == 1.0 else None
    if ref:
        e = row["exp_brake"]
        row["ref_1001"] = {k: ref[k] for k in ("phi_end", "phi_max", "slip_end_mm", "N_at_80", "t_at_80",
                                                "peak_rate_dps")}
        row["diff_vs_1001"] = {k: (e[k] - ref[k]) if (e[k] is not None and ref[k] is not None) else None
                               for k in ("phi_end", "phi_max", "slip_end_mm", "N_at_80", "t_at_80", "peak_rate_dps")}
    stride = max(1, int(round(0.010 / dt)))
    row["fine_cols"] = ["t", "N", "phi_deg", "rate_deg_s"]
    row["fine"] = [[round(float(a), 4), round(n_sched(float(a) - dt), 4), float(b), float(c)]
                   for a, b, c in zip(t[::stride], phi[::stride], rate[::stride])]
    row["trace_cols"] = res["trace_cols"]
    row["trace"] = res["trace"]
    row["film"] = None
    if frames:
        rel = f"media/brake_{model}.mp4"
        B.write_h264(frames, B.BED / rel)
        row["film"] = rel
    row["status"] = "complete" if both else "ejected"
    row["script"] = "scripts/contact_bed_brake.py"
    row["git_rev"] = B.git_rev()
    row["wall_s"] = time.time() - t_wall
    row["when"] = time.strftime("%Y-%m-%d %H:%M")
    return row


def make_tile(out):
    rows = {r["model"]: r for r in B.read_rows(out) if r.get("film") and r.get("dt_ms") == FILM_DT}
    models = [m for m in B.FILM_ORDER if m in rows and (B.BED / rows[m]["film"]).exists()]
    if not models:
        return None
    n = B.tile([B.BED / rows[m]["film"] for m in models], B.MEDIA / "brake_models.mp4", B.MEDIA / "brake_models.jpg",
               cols=3, poster_at=3.2 / (T_HOLD + T_RAMP + T_END))
    return models, n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--dt", nargs="+", type=float, help="physics step, ms (default: the protocol grid)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--no-film", action="store_true")
    ap.add_argument("--no-repro", action="store_true", help="skip the 10-01 0.5 mm pad (relaxation 0.02 s) row")
    ap.add_argument("--tile", action="store_true", help="only assemble the tile from existing clips")
    a = ap.parse_args()
    if a.tile:
        print(make_tile(a.out))
        return
    have = B.done(a.out, key=("model", "dt_ms"))
    cases = B.grid(a.models, a.dt)
    if not a.no_repro and (not a.models or set(a.models) & set(REPRO)) and (not a.dt or 1.0 in a.dt):
        cases += [(m, 1.0) for m in REPRO]
    for model, dt in cases:
        if (model, dt) in have:
            continue
        film = (not a.no_film) and dt == FILM_DT and model in B.MODELS
        try:
            r = run_case(model, dt, film=film)
        except Exception as e:
            r = {"task": "brake", "model": model, "rig_spec": B.MODELS.get(model, (REPRO.get(model),))[0],
                 "dt_ms": dt, "status": "failed", "error": repr(e), "traceback": traceback.format_exc()[-2000:],
                 "film": None, "script": "scripts/contact_bed_brake.py", "git_rev": B.git_rev()}
        H.append_row(a.out, r)
        keys = ("phi_end_deg", "phi_max_deg", "t_80_s", "N_at_80_N", "peak_rate_deg_s", "slip_end_mm",
                "us_per_step_median", "wall_s")
        print(model, dt, r["status"], {k: round(r[k], 4) for k in keys if isinstance(r.get(k), float)},
              r.get("pinched_end"), r.get("held_station"), r.get("diff_vs_1001"), flush=True)
    if not a.no_film:
        print("tile", make_tile(a.out))


if __name__ == "__main__":
    main()
