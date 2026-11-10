#!/usr/bin/env python3
"""The tracking score of the HOM tasks on the films of hand-object scale (owner, 2026-11-09: "you have my go ahead on the
rescoring"; "i value simplicity/interpretability/capturing the essence of the motion/success monotonically and smoothly
over the 0-1 range"; "i would prefer getting a usable set of videos over a full run").

The score (hom_landscape.track_score) is 1 - the RMS angle between the object's orientation and the commanded one over
the RMS commanded rotation, at least 0, times the share of the task with all three pads loaded; 0 when the object
dropped. The films of 2026-11-09 chose each size's layout by the earlier score (hom_landscape.rollout_score), whose twist
counted the largest yaw at any instant on three pads and whose roll gave 0.5 to a sphere held still.

  labels   the filmed rollouts again (seed 0, CPU, no frames): both scores against the owner's verdicts on the clips
  rerun T  per size, a short list of the stored live-EA candidates (rows ea_live<tag>) by a proxy of the stored metrics,
           the earlier pick and layout (70, 110) mm, each at seeds 0-2 on the CPU -> rows/20261109-track_rerun<tag>.jsonl
  film T   each size's best candidate by the mean tracking score, filmed at the seed closest to its mean: clips and a
           two-by-two film

  HOS_PLANT=servo HOS_SCENES=logs/20261102-hom_servo_plant/scenes HOM_ROWS_TAG=_sv25 MUJOCO_GL=egl \\
      .venv/bin/python scripts/hom/hom_track_films.py rerun twist
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path
import hom_films as HF  # noqa: E402
import hom_gpu as HG  # noqa: E402
import hom_landscape as L  # noqa: E402
import hom_rows  # noqa: E402
import hom_tasks as T  # noqa: E402

TAG = HG.ROWS_TAG
SIZES = {"twist": (25.0, 30.0, 63.0, 80.0), "roll": (25.0, 30.0, 63.0, 80.0), "press": (10.0, 13.0, 40.0, 50.0)}
NAMES = {"twist": "Twist", "roll": "Roll", "press": "Press"}
GENERALIST = (70.0, 110.0)
K = int(os.environ.get("TRACK_K", "6"))                 # candidates per size from the proxy
SEEDS = (0, 1, 2)
OUT = HF.ROWS / f"20261109-track_rerun{TAG}.jsonl"
PICKS = HF.ROWS / f"20261109-track_picks{TAG}.json"
L.GRID_SRC, L.GRID_ROWS = "grid2dff", L.ROWS / f"grid2dff_v4b{TAG}"

_found = {}


def _find(task, kind, d_mm, call, grasp):
    """hom_films.find with the layout set built once per task and size."""
    k = (task, kind, float(d_mm))
    if k not in _found:
        _found[k] = {(int(key["call"]), int(key.get("grasp", 0))): (key, mxy, setup)
                     for key, mxy, setup in HG.layout_list(task, kind, 0, d_mm)}
    if (int(call), int(grasp)) not in _found[k]:
        raise KeyError(f"{task} {kind} {d_mm:g} mm: call {call} grasp {grasp} not found")
    return copy.deepcopy(_found[k][(int(call), int(grasp))])


HF.find = _find


def var(task):
    return "hold" if T.SHAPE[task] == "cylinder" else "base"


def obj(task, size):
    return f"{size:g} mm {'sphere' if T.SHAPE[task] == 'sphere' else 'cylinder'}"


def run(task, size, call, grasp, seed, render=False, label=""):
    """One CPU rollout: (frames, a row with both scores), or (None, a row marked diverged)."""
    x, y, _r = L.grid_poses(task, size)[call]
    row = dict(task=task, size=size, call=call, grasp=grasp, seed=seed, layout=[x, y])
    try:
        fr, res = HF.clip(task, "grid2dff_size", size, call, grasp, label, seed=seed, var=var(task), render=render)
    except np.linalg.LinAlgError as e:                   # the state diverged and HOM's least squares failed
        return None, dict(row, diverged=str(e), old=0.0, track=0.0, track_nc=0.0)
    keep = ("held", "dropped", "fingers_end", "track_err_rms_deg", "track_ref_rms_deg", "contact_all_frac",
            "ramp_three_pads_max_deg", "ramp_end_deg", "w_rmse_ratio", "theta_rmse_deg")
    row.update({k: res.get(k) for k in keep}, old=round(L.rollout_score(task, res), 3),
               track=round(L.track_score(task, res), 3), track_nc=round(L.track_score(task, res, contact=False), 3))
    return fr, row


def write(row):
    with OUT.open("a") as fh:
        fh.write(json.dumps(row, default=float) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def filmed():
    """The (task, size, call, grasp, set) of the 2026-11-09 films, from rows/films_v4b.json."""
    info = json.loads(HF.INFO.read_text())
    out = []
    for task in SIZES:
        for name, kind in ((f"sizes4_{task}", "best"), (f"one_{task}", "one")):
            for s, v in info.get(name, {}).items():
                if isinstance(v, dict) and v.get("grasp") is not None and "score" in v:
                    out.append((task, float(s), int(v["call"]), int(v["grasp"]), kind))
    return out


def labels():
    """The filmed rollouts again at seed 0: the earlier score, the tracking score with and without the contact share."""
    print(f"{'task':6s} {'mm':>4s} {'set':4s} {'layout':>11s} {'old':>5s} {'track':>6s} {'no-contact':>10s} "
          f"{'contact':>7s} {'err/ref deg':>12s}", flush=True)
    for task, size, call, g, kind in filmed():
        _fr, r = run(task, size, call, g, 0)
        write(dict(r, set=f"labels_{kind}"))
        print(f"{task:6s} {size:4.0f} {kind:4s} {str(tuple(int(v) for v in r['layout'])):>11s} {r['old']:5.2f} "
              f"{r['track']:6.2f} {r['track_nc']:10.2f} {r.get('contact_all_frac') or 0:7.2f} "
              f"{r.get('track_err_rms_deg') or 0:5.1f}/{r.get('track_ref_rms_deg') or 0:<5.1f}", flush=True)


def proxy(task, rows):
    """Lower is better: the stored metrics closest to the tracking error (the EA's GPU rows hold no orientations)."""
    if task == "twist":
        return float(np.mean([abs((r.get("ramp_end_deg") or 0.0) - 90.0) for r in rows]))
    if task == "roll":
        return float(np.mean([np.sqrt(np.mean([(s_["angle_deg"] - s_["ref_deg"]) ** 2 for s_ in r.get("steps") or []]
                                              or [1e3])) for r in rows]))
    return float(np.mean([r.get("theta_rmse_deg") or 1e3 for r in rows]))


def shortlist(task, size):
    """(call, grasp, why) per candidate: the K best by the proxy among those held at all three GPU seeds, the earlier
    pick by the old score, and the generalist."""
    p = HF.ROWS / f"ea_live{TAG}" / f"{task}_d{size:g}_s0.jsonl"
    groups = {}
    for x in p.read_text().splitlines():
        if x.strip():
            r = hom_rows.rescale_metrics(json.loads(x))
            if r.get("status", "ok") == "ok":
                groups.setdefault((int(r["call"]), int(r.get("grasp", 0))), []).append(r)
    ok = {k: v for k, v in groups.items() if len(v) >= 3 and all(r.get("held") for r in v)}
    best = sorted(ok, key=lambda k: proxy(task, ok[k]))[:K]
    old = max((k for k, v in groups.items() if len(v) >= 3),
              key=lambda k: float(np.mean([L.rollout_score(task, r) for r in groups[k]])))
    out = [(c, g, "proxy") for c, g in best]
    if old not in best:
        out.append((old[0], old[1], "old pick"))
    call = HF.call_at(task, size, *GENERALIST)
    grasps = sorted(int(k.get("grasp", 0)) for k, _m, _s in HG.layout_list(task, "grid2dff_size", 0, size)
                    if int(k["call"]) == call)
    if grasps:
        out.append((call, grasps[0], "generalist"))
    return out


def rerun(task):
    done = set()
    if OUT.exists():
        for x in OUT.read_text().splitlines():
            if x.strip():
                r = json.loads(x)
                if r.get("set") == "rerun":
                    done.add((r["task"], r["size"], r["call"], r["grasp"], r["seed"]))
    for size in SIZES[task]:
        for call, g, why in shortlist(task, size):
            for seed in SEEDS:
                if (task, size, call, g, seed) in done:
                    continue
                t0 = time.time()
                _fr, r = run(task, size, call, g, seed)
                write(dict(r, set="rerun", why=why, wall_s=round(time.time() - t0, 1)))
            print(task, size, call, g, why, flush=True)


def means(task):
    """Per size: [(mean track, mean old, call, grasp, why, layout, n)] from the rerun rows, best first."""
    acc = {}
    for x in OUT.read_text().splitlines():
        r = json.loads(x) if x.strip() else {}
        if r.get("set") == "rerun" and r["task"] == task:
            acc.setdefault((r["size"], r["call"], r["grasp"]), []).append(r)
    out = {}
    for (size, call, g), rs in acc.items():
        out.setdefault(size, []).append((float(np.mean([r["track"] for r in rs])), float(np.mean([r["old"] for r in rs])),
                                         call, g, rs[0]["why"], rs[0]["layout"], len(rs)))
    return {s: sorted(v, key=lambda t: -t[0]) for s, v in out.items()}


def film(task):
    """Each size's best candidate by the mean tracking score over seeds 0-2, filmed at seed 0, and a two-by-two film."""
    import hom_contact_rig as HR
    from PIL import Image
    picks = json.loads(PICKS.read_text()) if PICKS.exists() else {}
    clips, info = [], {}
    rerun_rows = [json.loads(x) for x in OUT.read_text().splitlines() if x.strip()]
    for size, cands in sorted(means(task).items()):
        m, mo, call, g, why, (x, y), n = cands[0]
        seeds = {r["seed"]: r["track"] for r in rerun_rows if r.get("set") == "rerun" and r["task"] == task
                 and r["size"] == size and (r["call"], r["grasp"]) == (call, g)}
        seed = min(seeds, key=lambda s_: (abs(seeds[s_] - m), s_))      # the seed closest to the mean is filmed
        fr, r = run(task, size, call, g, seed, render=True,
                    label=f"{NAMES[task]}, {obj(task, size)}: layout ({x:.0f}, {y:.0f}) mm")
        out = HF.MEDIA / f"20261109-clip_track_{task}_d{size:g}{TAG}.mp4"
        HR.write_mp4(fr, out, fps=25)
        Image.fromarray(fr[int(0.55 * len(fr))]).convert("RGB").save(out.with_suffix(".jpg"), quality=85)
        clips.append(fr)
        gen = next((c for c in cands if c[4] == "generalist"), None)
        info[f"{size:g}"] = dict(layout=[x, y], call=call, grasp=g, why=why, track_mean=round(m, 3),
                                 old_mean=round(mo, 3), seeds=n, film_seed=seed, film=r, clip=str(out.relative_to(HF.ROOT)),
                                 generalist_track_mean=round(gen[0], 3) if gen else None,
                                 candidates=[dict(layout=c[5], call=c[2], grasp=c[3], why=c[4], track=round(c[0], 3),
                                                  old=round(c[1], 3)) for c in cands])
        print(task, size, info[f"{size:g}"]["layout"], why, f"track {m:.2f} (seed {seed} {r['track']:.2f})", flush=True)
    HF.compose(clips, HF.MEDIA / f"20261109-film_track_{task}{TAG}.mp4", grid=(2, 2))
    picks[task] = info
    PICKS.write_text(json.dumps(picks, indent=1, default=float))


def main():
    mode, task = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else None)
    {"labels": lambda: labels(), "rerun": lambda: rerun(task), "film": lambda: film(task)}[mode]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
