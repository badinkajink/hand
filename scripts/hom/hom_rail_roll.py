#!/usr/bin/env python3
"""The roll with the thumb held still as a rail, after the HOM paper's video (owner, 2026-11-09: "the roll is the weakest
demonstration because it departs from the HOM paper's video, where the thumb stays extended and rigid and the two other
fingers roll the ball up along it").

hom_tasks.Schedule's roll with prm rail 1: the thumb keeps the servo targets it had when the task began, the sphere
rides on it and may touch its links (the CPU plant has every contact), and the index and the middle roll the sphere under
HOM. With rail 2 the index alone rolls, after the middle backs off 8 mm. The tracking score's contact share is taken over
the rolling fingers' pads (hom_tasks.metrics, rows' "rollers"). The search on the 2-D lattice is a CPU sample in two
stages:
  sample   every layout that reaches the tripod at the size, with its foundational pose (grasp 0), seed 0;
  refine   the K best layouts of the sample by the tracking score, every grasp of their sets, seeds 0-2;
  film     each case's best layout and grasp by the mean tracking score, at the seed closest to its mean, from below
           (HOM_FILM_VIEW 60,55,0.16); ties within 0.02 go to the larger contact share, as on the tracking page;
  report   the tables of the tracking page's section.
Cases: rail 1 at 25, 30, 63 and 80 mm; rail 2 at 25 and 30 mm.
Rows: docs/experiments/20261029-foundational_pose_review/rows/20261110-rail_roll<tag>.jsonl, one fsynced line per
rollout, resumable.

    HOS_PLANT=servo HOS_SCENES=logs/20261102-hom_servo_plant/scenes HOM_ROWS_TAG=_sv25 \\
        .venv/bin/python scripts/hom/hom_rail_roll.py sample --part 0 --parts 6
"""
from __future__ import annotations

import os

os.environ.setdefault("SCALE_HAND", "cad_v2")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path
import hom_torque as Q  # noqa: E402

T, HG, L = Q.T, Q.HG, Q.L
TAG = HG.ROWS_TAG
ROWS = HG.ROWS / f"20261110-rail_roll{TAG}.jsonl"
MEDIA = HG.ROWS.parent / "media"
CASES = ((25.0, 1), (30.0, 1), (63.0, 1), (80.0, 1), (25.0, 2), (30.0, 2))
K = 10
TIE = 0.02
VIEW = "60,55,0.16"


def load():
    return [json.loads(x) for x in ROWS.read_text().splitlines() if x.strip()] if ROWS.exists() else []


def write(row):
    with ROWS.open("a") as fh:
        fh.write(json.dumps(row, default=float) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def grasps(size):
    """{call: [grasp, ...]} of the tripod grid at the size (hom_gpu.layout_list)."""
    out = {}
    for key, _m, _s in HG.layout_list("roll", "grid2dff_size", 0, size):
        out.setdefault(int(key["call"]), []).append(int(key.get("grasp", 0)))
    return out


def run_jobs(jobs, stage, part, parts):
    done = {(r["size"], r["rail"], r["call"], r["grasp"], r["seed"]) for r in load()}
    for size, rail, call, grasp, seed in jobs[part::parts]:
        if (size, rail, call, grasp, seed) in done:
            continue
        try:
            row = Q.run("roll", size, call, grasp, seed, prm={"rail": float(rail)})
        except np.linalg.LinAlgError as e:               # the state diverged and HOM's least squares failed
            x, y, _r = L.grid_poses("roll", size, "grid2dff")[call]
            row = dict(task="roll", size=float(size), call=call, grasp=grasp, layout=[x, y], seed=seed,
                       diverged=str(e), track=0.0, old=0.0)
        row.update(rail=rail, stage=stage)
        write(row)
        print(f"{stage} {size:4.0f} mm rail {rail} {str(tuple(int(v) for v in row['layout'])):>11s} g{grasp:2d} s{seed} "
              f"track {row['track']:.2f} old {row['old']:.2f} rollers' contact {row.get('contact_all_frac') or 0:.2f}",
              flush=True)


def cmd_sample(a):
    jobs = [(size, rail, call, 0, 0) for size, rail in CASES for call in sorted(grasps(size))]
    run_jobs(jobs, "sample", a.part, a.parts)
    return 0


def best_calls(rows, size, rail, k=K):
    sm = [r for r in rows if r["stage"] == "sample" and r["size"] == size and r["rail"] == rail]
    return [r["call"] for r in sorted(sm, key=lambda r: (-r["track"], -(r.get("contact_all_frac") or 0.0)))[:k]]


def cmd_refine(a):
    rows = load()
    jobs = []
    for size, rail in CASES:
        gs = grasps(size)
        for call in best_calls(rows, size, rail):
            jobs += [(size, rail, call, g, sd) for g in gs[call] for sd in (0, 1, 2)]
    run_jobs(jobs, "refine", a.part, a.parts)
    return 0


def picks(rows):
    """Per case the best (call, grasp) of the refine stage by the mean tracking score over seeds 0-2 (ties within TIE
    to the larger mean contact share of the rollers), with its means and per-seed scores."""
    out = {}
    for size, rail in CASES:
        g = {}
        for r in rows:
            if r["stage"] == "refine" and r["size"] == size and r["rail"] == rail:
                g.setdefault((r["call"], r["grasp"]), []).append(r)
        g = {k: v for k, v in g.items() if len(v) == 3}
        if not g:
            continue
        mean = {k: float(np.mean([r["track"] for r in v])) for k, v in g.items()}
        c3 = {k: float(np.mean([r.get("contact_all_frac") or 0.0 for r in v])) for k, v in g.items()}
        top = max(mean.values())
        best = max((k for k in g if mean[k] >= top - TIE), key=lambda k: (c3[k], mean[k]))
        v = sorted(g[best], key=lambda r: r["seed"])
        foll = [max(0.0, 1.0 - r["track_err_rms_deg"] / r["track_ref_rms_deg"]) if r.get("held") else 0.0 for r in v]
        out[(size, rail)] = {"call": best[0], "grasp": best[1], "layout": v[0]["layout"], "track_mean": mean[best],
                             "old_mean": float(np.mean([r["old"] for r in v])), "contact_mean": c3[best],
                             "followed_mean": float(np.mean(foll)), "seeds": [r["track"] for r in v],
                             "film_seed": min(v, key=lambda r: (abs(r["track"] - mean[best]), r["seed"]))["seed"],
                             "candidates": len(g), "layouts": len({k[0] for k in g})}
    return out


def cmd_film(_a):
    import hom_contact_rig as HR
    import hom_films as HF
    os.environ["HOM_FILM_VIEW"] = VIEW
    P = picks(load())
    for rail, sizes in ((1, (25.0, 30.0, 63.0, 80.0)), (2, (25.0, 30.0))):
        clips = []
        for size in sizes:
            p = P.get((size, rail))
            if p is None:
                continue
            row, frames = Q.run("roll", size, p["call"], p["grasp"], p["film_seed"], render=True,
                                prm={"rail": float(rail)})
            lay = tuple(int(v) for v in p["layout"])
            label = (f"{size:g} mm, layout {lay}: tracking {row['track']:.2f} (mean {p['track_mean']:.2f}), seed "
                     f"{p['film_seed']}")
            clips.append([HR.annotate(f, label, size=12) for f in frames])
        if clips:
            HF.compose(clips, MEDIA / f"20261110-film_rail{rail}_roll{TAG}.mp4", grid=(2, 2) if len(clips) == 4 else None)
    return 0


def cmd_report(_a):
    rows = load()
    P = picks(rows)
    for size, rail in CASES:
        sm = [r for r in rows if r["stage"] == "sample" and r["size"] == size and r["rail"] == rail]
        if not sm:
            continue
        held = sum(bool(r.get("held")) for r in sm)
        p = P.get((size, rail))
        print(f"{size:4.0f} mm rail {rail}: sample {len(sm)} layouts, held {held}, best sample track "
              f"{max(r['track'] for r in sm):.2f}, share > 0.2 {np.mean([r['track'] > 0.2 for r in sm]):.2f}"
              + (f"; pick {tuple(int(v) for v in p['layout'])} g{p['grasp']} track {p['track_mean']:.2f} "
                 f"(seeds {', '.join(f'{x:.2f}' for x in p['seeds'])}) old {p['old_mean']:.2f} followed "
                 f"{p['followed_mean']:.2f} rollers' contact {p['contact_mean']:.2f} of {p['candidates']} candidates"
                 if p else ""), flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("sample", "refine"):
        s = sub.add_parser(name)
        s.add_argument("--part", type=int, default=0)
        s.add_argument("--parts", type=int, default=1)
    sub.add_parser("film")
    sub.add_parser("report")
    a = ap.parse_args()
    return {"sample": cmd_sample, "refine": cmd_refine, "film": cmd_film, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
