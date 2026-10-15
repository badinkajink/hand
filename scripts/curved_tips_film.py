#!/usr/bin/env python3
"""Films of the curved-tip study (docs/handoff/20261015-curved_tips.md, step 5), rendered on the CPU through Mesa's
software EGL with the 2026-10-14 study's film pieces (scripts/tip_codesign_film.py: 640 x 480 panels with a label bar,
2 x 2 tiles at crf 20). Clips, tiles, posters and films.jsonl under docs/experiments/20261015-curved_tips/media/.

  bed   the bed spin at 1 N and 1 rad/s with builder-v2 pads (curved_tips_bed.py's model), one frame per 0.01 rad of
        turn (4x slow motion), seen from the tool side through the tool drawn transparent: each left-tip pad touching the
        tool coloured by its normal force over K times its penetration, white at 1 and red at 10 or more; the label gives
        the running spin arm and the law's (Drake's field on the 0.25 mm mesh)
  hom   the HOM turn on a CAD-hand cell (default cylinder:16:0.625) per tip at mu 0.7 on the servo plant, builder-v2
        pads, the grasp each tip chose on seed 0 of the screen (screen/summary.json), at a fresh seed; pads touching the
        cylinder drawn red

    export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json MUJOCO_GL=egl LP_NUM_THREADS=1
    PY=logs/20261001-hom_contact/venv/bin/python
    $PY scripts/curved_tips_film.py bed --members tcd_rs3_re3_p0.2 tcd_cap6_u0.1 ...
    SCALE_HAND=cad_v1 .venv/bin/python scripts/curved_tips_film.py hom --tips tcd_rs3_re3_p0.2 ...
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
import curved_tips_common as C  # noqa: E402
import tip_codesign_film as TF  # noqa: E402

TF.MEDIA = C.D / "media"
TF.tip_name = C.label


def cmd_bed(a):
    import contact_bed_common as CB
    import curved_tips_bed as CTB
    import tip_codesign_sweep as SW
    SW.D = C.D / "bed"                             # film_bed reads the law's spin arm from SW.D / law.jsonl
    SW.model_of = lambda member, variant: CTB.model(member, "pads")
    clips = []
    for mem in a.members:
        out = TF.MEDIA / f"bed_spin_{mem}.mp4"
        t0 = time.time()
        fr, info = TF.film_bed(mem, a.N, a.omega)
        CB.write_h264(fr, out, crf=20)
        TF._log({"film": "bed", "clip": out.name, **info, "frames": len(fr), "when": time.strftime("%Y-%m-%d %H:%M")})
        print(f"{out.name}: arm {info['arm_mm']:.2f} mm, law {info['law_mm']}, max f/(K d) {info['max_ratio_1um']:.1f}, "
              f"{len(fr)} frames, {time.time() - t0:.0f} s", flush=True)
        clips.append(out)
    if len(clips) > 1:
        print("tile", TF.tile(clips, a.stem), "frames", flush=True)


def film_hom(tip, cell_spec, seed, cand_index, mu=0.7):
    """tip_codesign_film.film_hom with builder-v2 pads and this study's scene caches."""
    import argparse as _ap
    import mujoco
    import curved_tips_rig  # noqa: F401  (this study's scene folders)
    import hand_object_scale_films as SF
    import hand_object_scale_sim as S
    import hom_turn3 as HT
    import reorient_backends as RB
    import reorient_backends_films as RBF
    import tip_codesign_rig as R
    shape, dd, t = cell_spec.split(":")
    dd, t = float(dd), float(t)
    ns = _ap.Namespace(E=RB.E_TPU, pf=None, tr=RB.PAD_TR, tag=None, h="local", K="each")
    R.install(ns, force_tip=tip)
    S.SCENES = R.SCN.parent / "scale" / f"{tip}_{R.tag_of(ns)}_mu{mu:g}"
    S.MU, S.PLANT = mu, C.SERVO
    cell = next(c for c, sh, d_ in S.cell_list("diag") if sh == shape and d_ == dd and c.get("t") == t)
    cand = S.candidates(cell, shape, dd)[cand_index]
    sc = S.cell_scenes(cell, shape, dd, models=("pads", "pt"))
    m_pt = mujoco.MjModel.from_xml_path(str(sc["pt"][0]))
    gg = S.grasp_geometry(cell, shape, dd, cand["h"], cand["spread"])
    gp = S.grip_poses(m_pt, gg, RB.plant_args(C.SERVO)["kp"])
    meta = S.candidate_meta(sc["pads"][1], gg, gp)
    plant = HT.MjPlant(SF.film_scene(Path(sc["pads"][0]), dd / 2000.0), meta)
    S.set_palm(plant.m, gg["palm"])
    mirror = S.Mirror(Path(sc["pt"][0]))
    S.set_palm(mirror.m, gg["palm"])
    m, dat = plant.m, plant.d
    pads = [i for i in range(m.ngeom) if "_pad" in (m.geom(i).name or "")]
    for i in pads:
        m.geom_rgba[i] = (0.75, 0.4, 0.25, 0.0)
    RBF.W, RBF.H = TF.PW, TF.PH
    rend, cam = RBF.renderer_for(m, np.asarray(meta["tool7"], float)[:3])
    cam.distance = 0.24 + 1.4 * dd / 1000.0
    frames, k = [], [0]
    title = C.label(tip)

    def cb(pl, tt, info):
        k[0] += 1
        if k[0] % 4:
            return
        for i in pads:
            m.geom_rgba[i, 3] = 0.0
        for i in range(dat.ncon):
            g1, g2 = int(dat.contact[i].geom[0]), int(dat.contact[i].geom[1])
            for g, o in ((g1, g2), (g2, g1)):
                if o in pl.tool_geoms and m.geom_bodyid[g] != pl.tb and "_pad" in (m.geom(g).name or ""):
                    m.geom_rgba[g] = (0.9, 0.1, 0.1, 1.0)
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
    if os.environ.get("SCALE_HAND") != "cad_v1":
        raise SystemExit("set SCALE_HAND=cad_v1")
    summ = json.loads((C.D / "screen" / "summary.json").read_text())
    key = f"{a.cell.split(':')[1]}:{a.cell.split(':')[2]}"
    clips = []
    for tip in a.tips:
        ci = summ["tips"][tip]["conditions"][f"0.7|{C.SERVO}"]["cells"][key]["cand"]
        out = TF.MEDIA / f"hom_{a.cell.replace(':', '_')}_s{a.seed}_{tip}.mp4"
        t0 = time.time()
        fr, res = film_hom(tip, a.cell, a.seed, ci)
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
    b = sub.add_parser("bed")
    b.add_argument("--members", nargs="+", required=True)
    b.add_argument("--N", type=float, default=1.0)
    b.add_argument("--omega", type=float, default=1.0)
    b.add_argument("--stem", default="bed_spin_tile")
    h = sub.add_parser("hom")
    h.add_argument("--tips", nargs="+", required=True)
    h.add_argument("--cell", default="cylinder:16:0.625")
    h.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    {"bed": cmd_bed, "hom": cmd_hom}[a.cmd](a)


if __name__ == "__main__":
    main()
