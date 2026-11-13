#!/usr/bin/env python3
"""A film of HOM and of the trained UHAS policy side by side on one layout, grasp and seed of the inner-loop benchmark
(scripts/hom/uhas_sr2.py), for its page. Both clips run in the same MuJoCo-Warp batch (one world, HOM driving the
approach, the closing, the squeeze and the lift, then HOM or the policy driving the task phase); the joint and object
states of every period are replayed in CPU MuJoCo and drawn by hom_films.Film (the finger links see-through, the
object's axes as rods), a frame every fourth period.

    .venv/bin/python scripts/hom/uhas_sr2_film.py --run results/uhas_sr2/<run> [--layout 120 65] [--grasp G]
Writes docs/experiments/20261112-inner_loop_benchmark/media/20261112-film_hom_uhas_<task>.mp4 (+ .jpg poster) and the
clips' scores to rows/film_hom_uhas_<task>.json.
"""
from __future__ import annotations

import os

os.environ.setdefault("SCALE_HAND", "cad_v2")
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path
import hom_gpu as HG  # noqa: E402
import uhas_sr2 as UH  # noqa: E402
import uhas_sr2_cik as U  # noqa: E402

T, S = HG.T, HG.S
OUT = ROOT / "docs/experiments/20261112-inner_loop_benchmark"


class Replay:
    """The pads scene as the GPU batch compiles it (palm and post as mocap bodies), with one world's mounts, for drawing
    recorded states."""

    def __init__(self, scene, world):
        import mujoco
        self.m = HG.mocap_model(scene.pads_path)
        for f in HG.FINGERS:
            self.m.body_pos[self.m.body(f"{f}_mount").id] = world["body_mounts"][f]
        self.d = mujoco.MjData(self.m)
        self.tb = self.m.body(S.OBJ).id
        self.post = self.m.geom("post").id
        self.mj = mujoco

    def show(self, qpos, mocap, t):
        self.d.qpos[:] = qpos
        self.d.mocap_pos[:] = mocap
        self.d.time = t
        self.mj.mj_forward(self.m, self.d)


def run_clip(B, world, policy=None, F=None, start="handover"):
    """One episode of one world: HOM throughout, or HOM to the task and the policy after. Returns the per-period states
    (qpos, mocap positions, time, each finger's pad force and the force through its rigid parts on the object, N) and
    the episode's row with J, the task phase's mean rigid-part force (sum over the fingers) and the pads' share of the
    finger forces on the object."""
    import hom_landscape as L
    B.begin([world], record=True, hom=policy is None or start == "handover")
    trace = []

    def keep():
        trace.append((B.vw["qpos"][0].cpu().numpy().copy(), B.vw["mocap_pos"][0].cpu().numpy().copy(),
                      B.clock[min(B.k, len(B.clock) - 1)], B.k))
    if policy is None:
        while not B.done():
            keep()
            B.tick(None)
    else:
        from tensordict import TensorDict
        task = UH.Task(B)
        mounts = torch.tensor(np.array([[world["body_mounts"][f] for f in HG.FINGERS]]), dtype=torch.float32,
                              device="cuda")
        C = U.SphereController(F, mounts)
        tr = UH.Tracker(1)
        tr.reset(torch.arange(1, device="cuda"), B, C, F, mounts, B.n_out[:, 0].float())
        with torch.inference_mode():
            while not B.done():
                keep()
                pol, cri = tr.observe(task, B, C, F, mounts)
                a = policy(TensorDict({"policy": pol, "critic": cri}, batch_size=[1], device="cuda")).clamp(-1, 1)
                B.tick(C.solve(a))
                tr.last_a = a
                tr.tau += 1
                tr.advance(task, B.scene.r)
    row = dict(B.finish()[0])
    row["J"] = L.track_score(B.task, row)
    ft = B.rec["ft"][:, 0].cpu().numpy()
    fl = np.maximum(B.rec["fo"][:, 0].cpu().numpy() - ft, 0.0)
    task = [k for k, t in enumerate(B.clock) if t >= B.t_task - 1e-9]
    row["link_force_mean_N"] = round(float(fl[task].sum(1).mean()), 3)
    row["pad_share"] = round(float(ft[task].sum() / max(1e-9, ft[task].sum() + fl[task].sum())), 3)
    return [(q, m, t, ft[k], fl[k]) for q, m, t, k in trace], row


def frames_of(trace, scene, world, label):
    import hom_contact_rig as HR
    import hom_films as HF
    rp = Replay(scene, world)
    film = HF.Film(rp, world["setup"])
    out = []
    try:
        for i, (qpos, mocap, t, fp, fl) in enumerate(trace):
            if i % 4:
                continue
            rp.show(qpos, mocap, t)
            out.append(film.frame(rp, label, f"t = {t:.2f} s   pads {fp[0]:.1f} {fp[1]:.1f} {fp[2]:.1f} N   "
                                             f"links {fl[0]:.1f} {fl[1]:.1f} {fl[2]:.1f} N"))
    finally:
        film.close()
    return [HR.annotate(f, label, size=14) for f in out]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--layout", nargs=2, type=float, default=None, help="x_sep y_sep (mm); default the test set's best "
                    "layout by HOM's J")
    ap.add_argument("--grasp", type=int, default=None, help="default the layout's best grasp by HOM's J")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="", help="appended to the film's and the row's names, e.g. _x50y110")
    a = ap.parse_args()
    run = Path(a.run)
    sp = json.loads((run / "split.json").read_text())
    task = sp["task"]
    look = UH.split(task, sp["split_seed"])[2]
    xy = tuple(a.layout) if a.layout else max(map(tuple, sp["test"]), key=lambda k: look[k])
    import inner_loop_benchmark_page as BP
    jr = BP.j_rows(task, UH.D_MM[task])
    grasps = {g: np.mean([j for j, _h in v]) for (x, y, g), v in jr.items() if (x, y) == xy}
    g = a.grasp if a.grasp is not None else max(grasps, key=grasps.get)
    entry = [e for e in UH.grid_entries(task) if UH.xy_of(e[0]) == xy and int(e[0].get("grasp", 0)) == g][0]
    scene = HG.Scene(T.SHAPE[task], UH.D_MM[task])
    key, mxy, setup = entry
    bm, q_open = scene.prepare(mxy, setup)
    world = {"key": key, "seed": a.seed, "setup": setup, "body_mounts": bm, "q_open": q_open,
             "var": HG.variant_world(UH.VARIANT[task], a.seed)}
    B = UH.UBatch(scene, task, [world], T.DT_C, gen_seed=a.seed)
    F = U.Fingers(B.m, "cuda")
    actor, ckpt = UH.load_policy(run)
    tr_h, row_h = run_clip(B, world)
    tr_u, row_u = run_clip(B, world, actor, F, sp.get("start", "handover"))
    lab = f"({xy[0]:g}, {xy[1]:g}) mm, grasp {g}, seed {a.seed}"
    fh = frames_of(tr_h, scene, world, f"HOM: J {row_h['J']:.2f}; {lab}")
    fu = frames_of(tr_u, scene, world, f"UHAS, trained {Path(ckpt).stem}: J {row_u['J']:.2f}")
    import hom_films as HF
    tag = ("" if sp.get("start", "handover") == "handover" else "_" + sp["start"]) + a.tag
    out = OUT / "media" / f"20261112-film_hom_uhas_{task}{tag}.mp4"
    HF.compose([fh, fu], out)
    keys = ("J", "held", "track_err_rms_deg", "track_ref_rms_deg", "contact_all_frac", "link_force_max_N",
            "link_force_mean_N", "pad_share", "drift_max_mm")
    info = {"task": task, "layout": list(xy), "grasp": g, "seed": a.seed, "run": run.name, "checkpoint": Path(ckpt).name,
            "hom": {k: row_h.get(k) for k in keys}, "uhas": {k: row_u.get(k) for k in keys},
            "grid_J_of_grasp": float(grasps[g])}
    (OUT / "rows" / f"film_hom_uhas_{task}{tag}.json").write_text(json.dumps(info, indent=1, default=float))
    print(json.dumps(info, default=float))


if __name__ == "__main__":
    main()
