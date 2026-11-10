#!/usr/bin/env python3
"""The pin-push with an unsensed pusher (owner, 2026-11-09, an exception to the pin-push rule, exploratory: "push with
the unsensorized back of the pusher's fingertip"): where and how hard the pusher touches the cylinder is estimated from
its joint kinematics, the tracked object pose and the thumb's and the index's boards (hom_tasks.pusher_estimate), and
the controller runs on the estimate in place of a pusher board.

Variants of hom_tasks.VARIANTS, each on the CPU plant with every contact and the object held until the grasp closes:
  hold      exact sensing at 100 Hz (the pin-push of the grids);
  tip       boards on all three fingers, 50 Hz, 40 ms late, the tag tracker's noise;
  tip_pe    as tip, the pusher's board replaced by the estimate;
  tipfl, tip_pefl   the same two with the pusher's force loop (10 mm/s per N) closed on its board or on the estimate;
  tip0, tip0_pe     the boards' noise alone, at 100 Hz with no delay and the exact object pose (this study's variants,
                    LOCAL), with the pusher's board or the estimate: the estimate's own cost.
Layouts: the pin-push grid's five best layouts on the SCS0009 at 2.5 MPa (rows grid2dff_v4b_sv25, earlier score) with
their best grasps, and layout (70, 110) mm with its best grasp; seeds 0-4.
Rows: docs/experiments/20261110-hom_pusher_estimate/rows/20261110-pusher_estimate.jsonl (fsynced, resumable).

    HOS_PLANT=servo HOS_SCENES=logs/20261102-hom_servo_plant/scenes HOM_ROWS_TAG=_sv25 \\
        .venv/bin/python scripts/hom/hom_pusher_estimate.py run [--part k --parts n]
    .venv/bin/python scripts/hom/hom_pusher_estimate.py report
"""
from __future__ import annotations

import os

os.environ.setdefault("SCALE_HAND", "cad_v2")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path
import hom_torque as Q  # noqa: E402

T, HG, L = Q.T, Q.HG, Q.L
ROWS = Q.ROOT / "docs/experiments/20261110-hom_pusher_estimate/rows/20261110-pusher_estimate.jsonl"
VARS = ("hold", "tip0", "tip0_pe", "tip", "tip_pe", "tipfl", "tip_pefl")
LOCAL = {"tip0": ({"contact_force_min": T.TIP_F_MIN}, {"tip_model": 1.0}),
         "tip0_pe": ({"contact_force_min": T.TIP_F_MIN}, {"tip_model": 1.0, "pusher_est": True})}
SEEDS = (0, 1, 2, 3, 4)
N_BEST = 5
KEEP = ("held", "theta_rmse_deg", "track_err_rms_deg", "track_ref_rms_deg", "contact_all_frac", "pusher_loc_err_mm",
        "pusher_force_err_N", "pusher_force_true_N", "pusher_est_frac")


def layouts():
    """(call, grasp, which): the grid's N_BEST best layouts with their best grasps, and layout (70, 110) mm."""
    import hom_plant_transfer as TR
    g = TR.grasp_means("pinpush", TR.ROWS / "grid2dff_v4b_sv25" / "pinpush_d25.jsonl")
    best = sorted(g, key=lambda c: -max(g[c].values()))[:N_BEST]
    out = [(c, max(g[c], key=g[c].get), "best") for c in best]
    gen = Q.call_at("pinpush", 25.0, *Q.GENERALIST)
    if gen in g and gen not in best:
        out.append((gen, max(g[gen], key=g[gen].get), "generalist"))
    return out


def run(call, grasp, seed, var, render=False):
    """One CPU rollout of the pin-push; render=True: (result, frames of hom_films.Film every fourth period)."""
    import mujoco
    _k, mxy, setup = Q.find("pinpush", 25.0, call, grasp)
    scene = HG.Scene("cylinder", 25.0)
    bm, q_open = scene.prepare(mxy, setup)
    pl = T.Plant(scene.pads_path, setup)
    m_pt = mujoco.MjModel.from_xml_path(str(scene.pt_path))
    for m in (pl.m, m_pt):
        for f in T.FINGERS:
            m.body_pos[m.body(f"{f}_mount").id] = bm[f]
    T.S.set_palm(m_pt, setup["palm"])
    fs, os_ = T.sr2_specs(setup)
    mirror = T.H.Mirror(m_pt, fs, os_)
    over, noise = {**T.VARIANTS, **LOCAL}[var]
    film = None
    if render:
        import hom_films as HF
        film = HF.Film(pl, setup)
    try:
        res, _rows, frames = T.rollout("pinpush", setup, pl, mirror, q_open, seed, T.PRM,
                                       replace(T.H.HomParams(dt=T.DT_C), **over), film, noise,
                                       T.PLANT_VARIANTS["hold"]["hold_s"])
    finally:
        if film is not None:
            film.close()
    return (res, frames) if render else res


def load():
    return [json.loads(x) for x in ROWS.read_text().splitlines() if x.strip()] if ROWS.exists() else []


def cmd_run(a):
    done = {(r["call"], r["grasp"], r["seed"], r["variant"]) for r in load()}
    jobs = [(c, g, w, sd, v) for c, g, w in layouts() for sd in SEEDS for v in VARS][a.part::a.parts]
    for call, grasp, which, seed, var in jobs:
        if (call, grasp, seed, var) in done:
            continue
        t0 = time.time()
        x, y, _r = L.grid_poses("pinpush", 25.0, "grid2dff")[call]
        row = dict(task="pinpush", size=25.0, call=call, grasp=grasp, layout=[x, y], which=which, seed=seed, variant=var)
        try:
            res = run(call, grasp, seed, var)
            row.update({k: res.get(k) for k in KEEP}, track=round(L.track_score("pinpush", res), 3),
                       old=round(L.rollout_score("pinpush", res), 3))
        except np.linalg.LinAlgError as e:
            row.update(diverged=str(e), track=0.0, old=0.0)
        row["wall_s"] = round(time.time() - t0, 1)
        ROWS.parent.mkdir(parents=True, exist_ok=True)
        with ROWS.open("a") as fh:
            fh.write(json.dumps(row, default=float) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        print(f"({x:.0f}, {y:.0f}) g{grasp} s{seed} {var:8s} track {row['track']:.2f} old {row['old']:.2f} theta "
              f"{row.get('theta_rmse_deg') or 0:5.1f} loc {row.get('pusher_loc_err_mm')} F err "
              f"{row.get('pusher_force_err_N')} F true {row.get('pusher_force_true_N')}", flush=True)
    return 0


def summary():
    rows = load()
    out = {}
    for v in VARS:
        rs = [r for r in rows if r["variant"] == v]
        if not rs:
            continue
        held = [r for r in rs if r.get("held")]
        med = (lambda k: float(np.median([r[k] for r in held if r.get(k) is not None]))
               if any(r.get(k) is not None for r in held) else None)
        out[v] = {"n": len(rs), "held": len(held), "track": float(np.mean([r["track"] for r in rs])),
                  "old": float(np.mean([r["old"] for r in rs])), "theta_rmse": med("theta_rmse_deg"),
                  "loc_err": med("pusher_loc_err_mm"), "force_err": med("pusher_force_err_N"),
                  "force_true": med("pusher_force_true_N"), "est_frac": med("pusher_est_frac")}
    return out


def cmd_report(_a):
    for v, s in summary().items():
        print(f"{v:9s} n {s['n']:3d} held {s['held']:3d} track {s['track']:.3f} old {s['old']:.3f} theta "
              f"{s['theta_rmse']} loc {s['loc_err']} F err {s['force_err']} F true {s['force_true']} est {s['est_frac']}")
    return 0


FILM = {"layout": (75.0, 55.0), "seed": 0, "vars": ("hold", "tip0", "tip0_pe")}
FNAME = {"hold": "Exact sensing", "tip0": "Boards' noise", "tip0_pe": "Boards' noise, pusher estimated"}


def cmd_film(_a):
    """The pin-push on one layout under exact sensing, the boards' noise and the estimate, side by side (MUJOCO_GL=egl)."""
    import hom_contact_rig as HR
    import hom_films as HF
    call, grasp, _w = next((c, g, w) for c, g, w in layouts()
                           if tuple(L.grid_poses("pinpush", 25.0, "grid2dff")[c][:2]) == FILM["layout"])
    clips = []
    for v in FILM["vars"]:
        res, frames = run(call, grasp, FILM["seed"], v, render=True)
        label = (f"{FNAME[v]}: tracking {L.track_score('pinpush', res):.2f}"
                 + ("" if res.get("held") else ", dropped"))
        clips.append([HR.annotate(f, label, size=13) for f in frames])
    out = ROWS.parent.parent / "media" / "20261110-film_pusher_estimate.mp4"
    HF.compose(clips, out)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--part", type=int, default=0)
    r.add_argument("--parts", type=int, default=1)
    sub.add_parser("report")
    sub.add_parser("film")
    a = ap.parse_args()
    return {"run": cmd_run, "report": cmd_report, "film": cmd_film}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
