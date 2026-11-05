#!/usr/bin/env python3
"""Films of the fingertip-shape rerun (docs/handoff/20261103-fingertip_shape_rerun.md, ledger step 7): the HOM turn on a
cad_v2 cell per tip, rendered on the CPU through Mesa's software EGL with the film pieces of scripts/tip_codesign_film.py
(640 x 480 panels with a label bar, tiles at crf 20), as scripts/curved_tips_film.py hom did on the cad_v1 hand.

  hom   the HOM turn on the cell (default cylinder:16:0.625) per tip at friction 0.7 on the SCS0009, pads at 2.5 MPa, the
        grasp each tip chose on seed 0 of the screen (screen/summary.json), at a fresh seed; pads touching the cylinder
        drawn red and the rigid parts touching it (DIP link, cassette, cap) blue; clips, tile and poster in media/

    export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json MUJOCO_GL=egl LP_NUM_THREADS=1
    SCALE_HAND=cad_v2 HOS_PLANT=servo .venv/bin/python scripts/fingertip_shape_rerun_film.py hom --tips <tips>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fingertip_shape_rerun_common as C  # noqa: E402
import tip_codesign_film as TF  # noqa: E402

TF.MEDIA = C.D / "media"
TF.tip_name = lambda k: C.label(k)[:1].upper() + C.label(k)[1:]       # label bars in sentence case


def film_hom(tip, cell_spec, seed, cand_index, mu=0.7):
    import mujoco
    import fingertip_shape_rerun_rig as RR
    _, S = RR.scale_env()
    import hand_object_scale_films as SF
    import hom_turn3 as HT
    import reorient_backends_films as RBF
    shape, dd, t = cell_spec.split(":")
    dd, t = float(dd), float(t)
    S.TIP, S.MU, S.PLANT = tip, mu, C.SERVO
    S.SCENES = C.LOGS / "scale" / f"{tip}_mu{mu:g}"
    cell = next(c for c, sh, d_ in S.cell_list("diag") if sh == shape and d_ == dd and c.get("t") == t)
    cand = S.candidates(cell, shape, dd)[cand_index]
    sc = S.cell_scenes(cell, shape, dd, models=("pads", "pt"))
    m_pt = mujoco.MjModel.from_xml_path(str(sc["pt"][0]))
    gg = S.grasp_geometry(cell, shape, dd, cand["h"], cand["spread"])
    gp = S.grip_poses(m_pt, gg, S.RB.plant_args(C.SERVO)["kp"])
    meta = S.candidate_meta(sc["pads"][1], gg, gp)
    plant = HT.MjPlant(SF.film_scene(Path(sc["pads"][0]), dd / 2000.0), meta)
    S.set_palm(plant.m, gg["palm"])
    mirror = S.Mirror(Path(sc["pt"][0]))
    S.set_palm(mirror.m, gg["palm"])
    m, dat = plant.m, plant.d
    fingers = ("thumb", "index", "middle")
    pads = [i for i in range(m.ngeom) if "_pad" in (m.geom(i).name or "")]
    rigid = [i for i in range(m.ngeom) if (m.geom(i).name or "").split("_")[0] in fingers
             and any(p in (m.geom(i).name or "") for p in ("_cap_s", "_cassette_s", "_dip_link_s"))]
    rgba0 = {i: m.geom_rgba[i].copy() for i in rigid}
    for i in pads:
        m.geom_rgba[i] = (0.75, 0.4, 0.25, 0.0)
    RBF.W, RBF.H = TF.PW, TF.PH
    rend, cam = RBF.renderer_for(m, np.asarray(meta["tool7"], float)[:3])
    cam.distance = 0.24 + 1.4 * dd / 1000.0
    frames, k = [], [0]
    title = TF.tip_name(tip)

    def cb(pl, tt, info):
        k[0] += 1
        if k[0] % 4:
            return
        for i in pads:
            m.geom_rgba[i, 3] = 0.0
        for i in rigid:
            m.geom_rgba[i] = rgba0[i]
        for i in range(dat.ncon):
            g1, g2 = int(dat.contact[i].geom[0]), int(dat.contact[i].geom[1])
            for g, o in ((g1, g2), (g2, g1)):
                if o in pl.tool_geoms and m.geom_bodyid[g] != pl.tb:
                    nm = m.geom(g).name or ""
                    if "_pad" in nm:
                        m.geom_rgba[g] = (0.9, 0.1, 0.1, 1.0)
                    elif g in rgba0:
                        m.geom_rgba[g] = (0.13, 0.4, 0.67, 1.0)
        rend.update_scene(dat, cam)
        frames.append(TF.label_bar(rend.render().copy(), [title, f"t {tt:4.2f} s   turn {info.get('th_deg', 0):+5.1f}°   "
                                                                 f"{info.get('phase', '')}"]))
    prm = dict(HT.DEFAULTS, kp_model=0.72, post_off=1.0)
    res = S.turn_rollout(plant, mirror, meta, seed, prm, frame_cb=cb)
    frames += [frames[-1]] * 25
    rend.close()
    return frames, res


def cmd_hom(a):
    import contact_bed_common as CB
    summ = json.loads((C.D / "screen" / "summary.json").read_text())
    key = f"{a.cell.split(':')[1]}:{a.cell.split(':')[2]}"
    clips = []
    for tip in a.tips:
        ci = summ["tips"][tip]["conditions"][f"0.7|{C.SERVO}"]["cells"][key]["cand"]
        out = TF.MEDIA / f"hom_{a.cell.replace(':', '_')}_s{a.seed}_{tip}.mp4"
        t0 = time.time()
        fr, res = film_hom(tip, a.cell, a.seed, ci)
        TF.MEDIA.mkdir(parents=True, exist_ok=True)
        CB.write_h264(fr, out, crf=20)
        TF._log({"film": "hom", "clip": out.name, "tip": tip, "cell": a.cell, "seed": a.seed, "cand": ci,
                 "held": res["held"], "turn_end_deg": res["turn_end_deg"], "held_turn_deg": res.get("held_turn_deg"),
                 "frames": len(fr), "when": time.strftime("%Y-%m-%d %H:%M")})
        print(f"{out.name}: cand {ci}, held {res['held']}, turn {res['turn_end_deg']:.1f} deg, {len(fr)} frames, "
              f"{time.time() - t0:.0f} s", flush=True)
        clips.append(out)
    if len(clips) > 1:
        print("tile", TF.tile(clips, f"hom_{a.cell.replace(':', '_')}_s{a.seed}_tile"), "frames", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hom")
    h.add_argument("--tips", nargs="+", required=True)
    h.add_argument("--cell", default="cylinder:16:0.625")
    h.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    {"hom": cmd_hom}[a.cmd](a)


if __name__ == "__main__":
    main()
