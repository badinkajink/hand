#!/usr/bin/env python3
"""Films of HOM through the twist on the plants of steel_hand_compare.py, two by two on one pair and seed (2026-11-13):
MuJoCo with the sphere pads (the benchmark's model, CPU), SolverSteel with compliant hydroelastic rows, SolverSteel with
point contact, and Newton's SolverMuJoCo with hydroelastic contact (each one world on the GPU, --contact-pairs obj).
Every clip is drawn by hom_films.Film from a MuJoCo replay of the point-contact scene (the pads scene for mjc_pads)
with the plant's joint and object state copied in at each filmed period: the finger links see-through, the sphere's
axes as rods. A clip's label carries its own rollout's J; a single world runs a different trajectory from the same
world in the 40-world batch of the rows (2026-11-13: J 0.618 alone, 0.205 in a batch, from a 10 um difference at the
first contact), so the films show behaviour and the rows give the scores.

    WARP_CACHE_PATH=$(mktemp -d) logs/20261113-steel/venv/bin/python scripts/contact/steel_hand_film.py --pairs 0 4
Writes docs/experiments/20261113-steel_solver/media/20261113-film_twist_p<i>_s<seed>.mp4 (+ .jpg poster) and
rows/film_twist_p<i>_s<seed>.json (each clip's J, held and metrics).
"""
from __future__ import annotations

import os

os.environ.setdefault("SCALE_HAND", "cad_v2")
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import traceback  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _paths  # noqa: E402,F401
import steel_hand_compare as C  # noqa: E402

HG, T, L = C.HG, C.T, C.L
FINGERS, JOINTS = C.FINGERS, C.JOINTS
LABELS = {"mjc_pads": "MuJoCo, sphere pads", "mjc_pt": "MuJoCo, point contact",
          "steel_hydro": "Steel, compliant hydroelastic", "steel_pt": "Steel, point contact",
          "nt_mjc_hydro": "Newton-MuJoCo, hydroelastic", "steel_hydro_hard": "Steel, hard hydroelastic rows",
          "steel_pt_pf": "Steel, point contact and point friction"}


class Replay:
    """A MuJoCo replay of one world (the palm and the post as mocap bodies, hom_gpu.mocap_model) into which a Newton
    world's state is copied; it has the attributes hom_films.Film reads from a plant (m, d, tb, post)."""

    def __init__(self, path, mounts, palm):
        import mujoco
        self.mj = mujoco
        self.m = HG.mocap_model(path)
        for f in FINGERS:
            self.m.body_pos[self.m.body(f"{f}_mount").id] = mounts[f]
        T.S.set_palm(self.m, palm)
        self.d = mujoco.MjData(self.m)
        self.tb = self.m.body(T.S.OBJ).id
        self.post = self.m.geom("post").id
        self.qadr = {(f, j): self.m.jnt_qposadr[self.m.joint(f"{f}_{j}").id] for f in FINGERS for j in JOINTS}
        self.oadr = self.m.jnt_qposadr[self.m.body_jntadr[self.tb]]

    def show(self, s):
        for f in FINGERS:
            for k, j in enumerate(JOINTS):
                self.d.qpos[self.qadr[(f, j)]] = s["q"][f][k]
        self.d.qpos[self.oadr:self.oadr + 3] = s["pos"]
        self.d.qpos[self.oadr + 3:self.oadr + 7] = s["quat"]
        self.d.time = s["t"]
        self.mj.mj_forward(self.m, self.d)


class KeepFilm:
    """T.rollout's film that keeps its frames, so that a rollout stopped by an error still has its clip."""

    def __init__(self, plant, setup):
        import hom_films as HF
        self.film = HF.Film(plant, setup)
        self.frames = []

    def frame(self, plant, title, line):
        f = self.film.frame(plant, title, line)
        self.frames.append(f)
        return f

    def close(self):
        self.film.close()


class StateFilm:
    """T.rollout's film for a Newton plant, whose rollout runs in a worker thread: each filmed period keeps the plant's
    state and caption, and draw() renders them in the replay from the main thread (an EGL context is current in the
    thread that made it)."""

    def __init__(self):
        self.states = []

    def frame(self, plant, title, line):
        self.states.append((plant.state(), line))

    def draw(self, replay, setup):
        film = KeepFilm(replay, setup)
        try:
            for s, line in self.states:
                replay.show(s)
                film.frame(replay, "", line)
        finally:
            film.close()
        return film.frames


def rollout(scene, p, mxy, setup, q_open, sd, plant, film):
    """hom_one of steel_hand_compare.py with a film."""
    import mujoco
    bm = scene.body_mounts(mxy)
    m_pt = mujoco.MjModel.from_xml_path(str(scene.pt_path))
    for f in FINGERS:
        m_pt.body_pos[m_pt.body(f"{f}_mount").id] = bm[f]
    T.S.set_palm(m_pt, setup["palm"])
    fs, os_ = T.sr2_specs(setup)
    mirror = HG.H.Mirror(m_pt, fs, os_)
    over, noise = T.VARIANTS[C.VAR]
    return T.rollout(C.TASK, setup, plant, mirror, q_open, sd, T.PRM, replace(HG.H.HomParams(dt=T.DT_C), **over), film,
                     noise, T.PLANT_VARIANTS.get(C.VAR, {}).get("hold_s", 0.0))


def clip(plant_name, scene, item):
    """One filmed rollout: (frames, result row)."""
    import hom_films as HF
    p, key, mxy, setup, q_open, sd = item
    bm = scene.body_mounts(mxy)
    out = {}
    if plant_name.startswith("mjc_"):
        plant = T.Plant(scene.pads_path if plant_name == "mjc_pads" else scene.pt_path, setup)
        for f in FINGERS:
            plant.m.body_pos[plant.m.body(f"{f}_mount").id] = bm[f]
        film = KeepFilm(plant, setup)
        try:
            out["r"] = rollout(scene, p, mxy, setup, q_open, sd, plant, film)
        except Exception as e:
            out["e"] = e
        film.close()
    else:
        B = C.NewtonBatch([C.pair_xml(scene, bm, setup["palm"])], plant_name, pairs="obj")
        sync = C.Sync(1)
        film = StateFilm()

        def job():
            try:
                out["r"] = rollout(scene, p, mxy, setup, q_open, sd, C.World(B, 0, setup, sync), film)
            except Exception as e:
                out["e"] = e
                traceback.print_exc()
            finally:
                sync.finish(0)
        th = threading.Thread(target=job, daemon=True)
        th.start()
        sync.serve(B)
        th.join()
        film.frames = film.draw(Replay(scene.pt_path, bm, setup["palm"]), setup)
    if "e" in out:
        # HOM's least squares fails once the sphere has left the hand (its witness frame has no finite normal,
        # 2026-11-13): the clip ends there and counts as a drop
        print(plant_name, "stopped:", out["e"], flush=True)
        return film.frames, {"held": False, "stopped": f"{type(out['e']).__name__}: {out['e']}"}
    res, _trace, _frames = out["r"]
    return film.frames, res


def main():
    import hom_contact_rig as HR
    import hom_films as HF
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pairs", type=int, nargs="+", default=[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plants", nargs="+", default=["mjc_pads", "steel_hydro", "steel_pt", "nt_mjc_hydro"])
    a = ap.parse_args()
    media = C.OUT / "media"
    media.mkdir(parents=True, exist_ok=True)
    for i in a.pairs:
        scene, ws = C.worlds([a.seed], i + 1)
        item = ws[i]
        clips, info = [], {}
        for pl in a.plants:
            frames, res = clip(pl, scene, item)
            J = 0.0 if "stopped" in res else L.track_score(C.TASK, res)
            lab = f"{LABELS[pl]}: J {J:.2f}" + ("" if res.get("held") else ", dropped")
            clips.append([HR.annotate(f, lab, size=14) for f in frames])
            info[pl] = {"J": J, **{k: v for k, v in res.items() if isinstance(v, (int, float, bool, str))}}
            print(pl, json.dumps(info[pl], default=float), flush=True)
        name = f"film_twist_p{i}_s{a.seed}"
        HF.compose(clips, media / f"20261113-{name}.mp4", grid=(2, 2) if len(clips) == 4 else None)
        (C.ROWS / f"{name}.json").write_text(json.dumps({"pair": C.pair_name(item[0]), "seed": a.seed,
                                                         "plants": a.plants, "clips": info}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
