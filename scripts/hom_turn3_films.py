#!/usr/bin/env python3
"""Films of the three-finger turn on the fitted plant: the deployed open-loop plan, the relative contact-velocity (HOM)
controller with its governor, and without it. MuJoCo, TPU tip with 2.7 mm fillets as 1 mm pads (pads touching the
tool drawn red), mu 1.0, servo kp 4 N m/rad, 0.02 s, 1 N m; one fixed camera on the tool's start (as
reorient_backends_films.py). Clips in docs/experiments/20261006-hom_turn3/media/, tiled three across.

    PY=logs/20261001-hom_contact/venv/bin/python
    MUJOCO_GL=egl $PY scripts/hom_turn3_films.py --hands D7 D2 D5 --seed 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import hom_turn3 as T  # noqa: E402
import reorient_backends as RB  # noqa: E402
import reorient_backends_films as F  # noqa: E402

MEDIA = ROOT / "docs/experiments/20261006-hom_turn3/media"


def film_hom(hand, seed, prm, title):
    import mujoco
    p_pads, p_mesh, meta, _ = T.scenes(hand, T.PLANT, 1.0)
    plant = T.MjPlant(p_pads, meta)
    mirror = T.Mirror(p_mesh)
    m, d = plant.m, plant.d
    rend, cam = F.renderer_for(m, np.asarray(meta["tool7"], float)[:3])
    pad_rgba = {i: m.geom_rgba[i].copy() for i in range(m.ngeom) if "_pad" in (m.geom(i).name or "")}
    frames = []
    state = {"k": 0}

    def cb(pl, t, info):
        state["k"] += 1
        if state["k"] % 4:
            return
        for i, c in pad_rgba.items():
            m.geom_rgba[i] = c
        for i in range(d.ncon):
            g1, g2 = int(d.contact[i].geom[0]), int(d.contact[i].geom[1])
            for g, o in ((g1, g2), (g2, g1)):
                if g in pad_rgba and o in pl.tool_geoms:
                    m.geom_rgba[g] = (0.9, 0.1, 0.1, 1.0)
        rend.update_scene(d, cam)
        cosv = float(d.xmat[pl.tb][8])
        frames.append(F.label(rend.render().copy(), [title, f"{hand} seed {seed}  t {t:4.2f} s  cos {cosv:+.2f}  "
                                                            f"{info.get('phase', '')}"]))
    T.rollout(plant, mirror, meta, seed, prm, frame_cb=cb)
    for _ in range(25):                     # half a second of the final frame
        frames.append(frames[-1])
    rend.close()
    return frames


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hands", nargs="+", default=["D7", "D2", "D5"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import imageio
    MEDIA.mkdir(parents=True, exist_ok=True)
    for hand in a.hands:
        clips = []
        for slug, title, kind in (("plan", "Deployed open-loop plan", "plan"),
                                  ("hom_gov", "HOM controller, governor on", 1.0),
                                  ("hom_free", "HOM controller, governor off", 0.0)):
            out = MEDIA / f"{hand}_s{a.seed}_{slug}.mp4"
            if not out.exists():
                if kind == "plan":
                    fr = F.film_mujoco(hand, a.seed, "tpu2.7", "pads", "bed", 100.0, title, plant=T.PLANT, mu=1.0)
                else:
                    fr = film_hom(hand, a.seed, dict(T.DEFAULTS, governor=kind), title)
                imageio.mimwrite(str(out), fr, fps=25, codec="libx264", quality=6, macro_block_size=8)
                print(f"{out.name}: {len(fr)} frames", flush=True)
            clips.append(out)
        n = F.tile_stream(clips, MEDIA / f"{hand}_s{a.seed}_three.mp4", MEDIA / f"{hand}_s{a.seed}_three.jpg", cols=3)
        print(f"{hand}: tiled {n} frames", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
