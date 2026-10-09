#!/usr/bin/env python3
"""Films of the HOM turn on representative cells of the hand-object scale landscape: one object on three layouts,
each at the grasp that turned it furthest (landscape.json), tiled three across. MuJoCo on the servo plant, the TPU
tip as 1 mm pads (pads touching the object drawn red), one fixed camera on the object's start.

Renders on the CPU through Mesa's software EGL, so it runs beside a GPU training queue:
    __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json MUJOCO_GL=egl LP_NUM_THREADS=1 \\
        .venv/bin/python scripts/hand_object_scale_films.py --object cylinder:25 --tags x085.0_y095.0 x100.0_y110.0 x122.5_y132.5
Clips in docs/experiments/20261008-hand_object_scale/media/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import hand_object_scale_sim as S  # noqa: E402
import hom_turn3 as H  # noqa: E402
import reorient_backends as RB  # noqa: E402
import reorient_backends_films as F  # noqa: E402

MEDIA = S.OUT_DIR / "media"


def best_turn_candidate(tag, shape, d):
    """Index of the candidate grasp with the largest median held turn, from turn.jsonl."""
    by = {}
    for line in (S.OUT_DIR / "turn.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("status") == "ok" and r["tag"] == tag and r["shape"] == shape and r["d_mm"] == d \
                and r.get("model", "pads") == "pads":
            by.setdefault(r["cand"], []).append(r["held_turn_deg"])
    if not by:
        return None, None
    ci = max(by, key=lambda c: float(np.median(by[c])))
    return ci, float(np.median(by[ci]))


def film_scene(path: Path, r: float) -> Path:
    """The cell's pad scene with a visual-only stalk along the object's marked axis (body +z), so the turn of a
    sphere is visible: no collision, no mass, the same dynamics."""
    import xml.etree.ElementTree as ET
    out = path.with_name(path.stem + "_film.xml")
    root = ET.parse(path).getroot()
    body = root.find(f".//body[@name='{S.OBJ}']")
    ET.SubElement(body, "geom", name="marker", type="capsule", fromto=f"0 0 {-1.6 * r:.6f} 0 0 {1.6 * r:.6f}",
                  size=f"{max(0.002, 0.1 * r):.6f}", rgba="0.92 0.42 0.20 1", contype="0", conaffinity="0",
                  density="0", group="0")
    ET.ElementTree(root).write(out)
    return out


def film_cell(tag, shape, d, seed, title):
    import mujoco
    R_ = json.loads((S.OUT_DIR / "reach.json").read_text())
    cell = next(c for c in R_["cells"] if c["tag"] == tag and c["shape"] == shape and c["d_mm"] == d)
    ci, med = best_turn_candidate(tag, shape, d)
    cands = S.candidates(cell, shape, d)
    cand = cands[ci if ci is not None else len(cands) // 2]
    sc = S.cell_scenes(cell, shape, d)
    m_pt = mujoco.MjModel.from_xml_path(str(sc["pt"][0]))
    gg = S.grasp_geometry(cell, shape, d, cand["h"], cand["spread"])
    gp = S.grip_poses(m_pt, gg, RB.plant_args(S.PLANT)["kp"])
    meta = S.candidate_meta(sc["pads"][1], gg, gp)
    plant = H.MjPlant(film_scene(sc["pads"][0], d / 2000.0), meta)
    S.set_palm(plant.m, gg["palm"])
    mirror = S.Mirror(Path(sc["pt"][0]))
    S.set_palm(mirror.m, gg["palm"])
    m, dd = plant.m, plant.d
    for i in range(m.ngeom):                       # the pads draw as the block; contacts are recoloured below
        if "_pad" in (m.geom(i).name or ""):
            m.geom_rgba[i] = (0.75, 0.4, 0.25, 0.0)
    rend, cam = F.renderer_for(m, np.asarray(meta["tool7"], float)[:3])
    cam.distance = 0.24 + 1.4 * d / 1000.0
    if shape == "sphere":            # look along the pinch axis from behind the thumb: the rod turns in the image plane
        cam.azimuth, cam.elevation = 0.0, -12.0
    pads = [i for i in range(m.ngeom) if "_pad" in (m.geom(i).name or "")]
    frames, k = [], [0]

    def cb(pl, t, info):
        k[0] += 1
        if k[0] % 4:
            return
        for i in pads:
            m.geom_rgba[i, 3] = 0.0
        for i in range(dd.ncon):
            g1, g2 = int(dd.contact[i].geom[0]), int(dd.contact[i].geom[1])
            for g, o in ((g1, g2), (g2, g1)):
                if o in pl.tool_geoms and m.geom_bodyid[g] != pl.tb and "_pad" in (m.geom(g).name or ""):
                    m.geom_rgba[g] = (0.9, 0.1, 0.1, 1.0)
        rend.update_scene(dd, cam)
        frames.append(F.label(rend.render().copy(), [title, f"t {t:4.2f} s  turn {info.get('th_deg', 0):+5.1f} deg  "
                                                            f"{info.get('phase', '')}"]))

    res = S.turn_rollout(plant, mirror, meta, seed, dict(H.DEFAULTS), frame_cb=cb)
    for _ in range(25):
        frames.append(frames[-1])
    rend.close()
    return frames, res, ci, cand


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--object", required=True, help="shape:d_mm, e.g. cylinder:25")
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import imageio
    shape, d = a.object.split(":")
    d = float(d)
    MEDIA.mkdir(parents=True, exist_ok=True)
    clips, log = [], []
    for tag in a.tags:
        out = MEDIA / f"turn_{S.obj_tag(shape, d)}_{tag}_s{a.seed}.mp4"
        R_ = json.loads((S.OUT_DIR / "reach.json").read_text())
        lay = next(l for fam in R_["families"].values() for l in fam if l["tag"] == tag)
        title = f"{shape} {d:g} mm, thumb-pair {lay['x_sep']:g} mm, pair {lay['y_sep']:g} mm"
        fr, res, ci, cand = film_cell(tag, shape, d, a.seed, title)
        imageio.mimwrite(str(out), fr, fps=25, codec="libx264", quality=6, macro_block_size=8)
        log.append({"clip": out.name, "tag": tag, "shape": shape, "d_mm": d, "seed": a.seed, "cand": ci,
                    "h_mm": round(cand["h"] * 1000, 1), "spread_mm": round(cand["spread"] * 1000, 1),
                    "held": res["held"], "turn_end_deg": res["turn_end_deg"], "held_turn_deg": res["held_turn_deg"],
                    "limit_joint": res["limit_joint"], "frames": len(fr)})
        print(json.dumps(log[-1]), flush=True)
        clips.append(out)
    stem = f"turn_{S.obj_tag(shape, d)}_three_s{a.seed}"
    n = F.tile_stream(clips, MEDIA / f"{stem}.mp4", MEDIA / f"{stem}.jpg", cols=len(clips))
    with open(MEDIA / "films.jsonl", "a") as fh:
        for row in log:
            fh.write(json.dumps({**row, "tile": f"{stem}.mp4"}) + "\n")
    print(f"tiled {n} frames -> {stem}.mp4", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
