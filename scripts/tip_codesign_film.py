#!/usr/bin/env python3
"""Films of the fingertip co-design study (docs/notes/20261014-tip_codesign_plan.md, step 12; ledger step 23), rendered
on the CPU through Mesa's software EGL so they never wait for the GPU. Panels 640 x 480 with a label bar (24 and 22 px
type, 15 and 14 px when a 1280 px tile is shown 828 px wide), tiled 2 x 2 with reorient_backends_films.tile_stream at
crf 20; clips, tiles, posters and films.jsonl (the numbers each label shows) under
docs/experiments/20261014-tip_codesign/media/.

  bed     (a) the bed spin at 1 N and 1 rad/s (tip_codesign_bed.case_spin's drive), one frame per 0.01 rad of turn
          (4x slow motion), seen from the tool side through the tool drawn transparent: each left-block pad touching
          the tool is coloured by its normal force over K times its penetration, white at 1 and red at 10 or more
  hom     (d) the HOM turn on a CAD-hand cell (default cylinder:16:0.625) per tip, the cell's grasp chosen on seed 0 of
          the stage-1/2 rows at mu 0.7 on the servo plant, at a stage-3 seed; pads touching the object drawn red
  replay  (e) the D7 seed-0 plan replay on the servo plant at mu 1.0 per tip, pads touching the tool drawn red

    export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json MUJOCO_GL=egl LP_NUM_THREADS=1
    PY=logs/20261001-hom_contact/venv/bin/python
    $PY scripts/tip_codesign_film.py bed --members tpu1 tcd_rs3_re3 tcd_rs3_re3_p0.2 tpu6
    $PY scripts/tip_codesign_film.py replay --tips tcd_rs3_re3_p0.2 <finalists>
    SCALE_HAND=cad_v1 .venv/bin/python scripts/tip_codesign_film.py hom --tips tcd_rs3_re3_p0.2 <finalists>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

D = ROOT / "docs/experiments/20261014-tip_codesign"
MEDIA = D / "media"
PW, PH = 640, 480
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
SERVO = "kp0.72_kv0_fr0.18_fl0_dp0.0144"


def tip_name(key: str) -> str:
    """Plain label of a tip key."""
    if key.startswith("tpu"):
        return f"block r {key[3:]} mm"
    import re
    m = re.match(r"tcd_rs([0-9.]+)_re([0-9.]+)(_p([0-9.]+))?", key)
    s = f"r_side {m[1]}, r_end {m[2]} mm"
    if key == "tcd_rs3_re3":
        s = "as-built, design surface"
    elif key == "tcd_rs3_re3_p0.2":
        s = "as-built, as printed"
    elif m[4]:
        s += f", printed at {m[4]} mm"
    return s


def label_bar(frame, lines, legend=False):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(frame)
    dr = ImageDraw.Draw(im, "RGBA")
    fb, fr = ImageFont.truetype(FONT_B, 24), ImageFont.truetype(FONT, 22)
    dr.rectangle([0, 0, im.width, 64], fill=(255, 255, 255, 235))
    dr.line([0, 64, im.width, 64], fill=(0, 0, 0, 255), width=1)
    dr.text((10, 4), lines[0], fill=(10, 10, 10), font=fb)
    if len(lines) > 1:
        dr.text((10, 34), lines[1], fill=(40, 40, 40), font=fr)
    if legend:                                     # white (1) to red (10 or more), log scale
        x0, y0, w, h = im.width - 250, im.height - 58, 200, 16
        for k in range(w):
            t = k / (w - 1)
            dr.line([x0 + k, y0, x0 + k, y0 + h], fill=_ratio_rgb(10 ** t, 255))
        dr.rectangle([x0, y0, x0 + w, y0 + h], outline=(0, 0, 0))
        f2 = ImageFont.truetype(FONT, 20)
        for v in (1, 3, 10):
            xx = x0 + int(round(math.log10(v) * (w - 1)))
            dr.text((xx - 6, y0 + h + 2), f"{v}", fill=(10, 10, 10), font=f2)
        dr.text((x0 - 150, y0 - 4), "f / (K δ)", fill=(10, 10, 10), font=f2)
    return np.asarray(im.convert("RGB"))


def _ratio_rgb(r, scale=1.0):
    t = min(1.0, max(0.0, math.log10(max(r, 1e-9))))
    c = (1.0, 1.0 - 0.88 * t, 1.0 - 0.88 * t)
    return tuple(int(round(x * scale)) for x in c) if scale != 1.0 else c


def _log(row):
    MEDIA.mkdir(parents=True, exist_ok=True)
    with open(MEDIA / "films.jsonl", "a") as fh:
        fh.write(json.dumps(row) + "\n")


def tile(clips, stem, poster_at=0.62):
    import reorient_backends_films as RBF
    RBF.W, RBF.H = PW, PH
    return RBF.tile_stream(clips, MEDIA / f"{stem}.mp4", MEDIA / f"{stem}_poster.jpg", cols=2, poster_at=poster_at, crf=20)


# ---------------------------------------------------------------------------------- (a) bed spin

def film_bed(member, N=1.0, omega=1.0):
    import mujoco
    import contact_bed_common as CB
    import hom_contact_rig as H
    import tip_codesign_bed as B
    import tip_codesign_sweep as SW
    import tpu_tip_rig as T
    name = SW.model_of(member, "base")
    law = None                                       # the Drake-field law's spin arm (psi 23/46/69 deg) at N, mm
    for r in map(json.loads, open(SW.D / "law.jsonl")):
        if r["tip"] == member and r["field"] == "drake":
            law = r["spin_arm_mm"][f"{N:g}"]
    spec = CB.MODELS[name][0]
    with T.tpu_rigs():
        H.DT = 1e-3
        rig = H.make_rig(spec, d_cg=0.0, gravity=False)
    H.settle(rig, N, T=1.0, weight_comp=False)
    m, d = rig.m, rig.d
    K = float(rig.info["K_sphere"])
    padL = {i for i in range(m.ngeom) if (m.geom(i).name or "").startswith("padL_s")}
    bR, tb = m.body("padR").id, m.body("tool").id
    base = {i: np.array([0.50, 0.52, 0.56, 1.0]) for i in padL}   # untouched pads grey, so the white-red scale
    for i, c in base.items():                                     # belongs to the touching pads only
        m.geom_rgba[i] = c
    for i in range(m.ngeom):
        if m.geom_bodyid[i] == bR:
            m.geom_rgba[i, 3] = 0.0
        elif m.geom_bodyid[i] == tb:
            m.geom_rgba[i, 3] = 0.22
    pts = [d.contact[i].pos.copy() for i in range(d.ncon) if int(d.contact[i].geom[0]) in padL or int(d.contact[i].geom[1]) in padL]
    look = np.mean(pts, axis=0) if pts else np.array([-H.R_TOOL, 0.0, 0.0])
    m.vis.global_.offwidth, m.vis.global_.offheight = max(PW, m.vis.global_.offwidth), max(PH, m.vis.global_.offheight)
    rend = mujoco.Renderer(m, PH, PW)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = look
    cam.distance, cam.azimuth, cam.elevation = 0.034, 180.0, -18.0
    st = rig.tool_state()
    th0, t0 = st["theta"], rig.t
    kp, kd = 0.5, 0.005
    n = int(round(1.2 / omega / H.DT))
    k_s = max(1, int(round(0.01 / (omega * H.DT))))
    f6 = np.zeros(6)
    taus, frames, mx_all = [], [], 0.0
    title = f"{tip_name(member)}, 1 mm pads"
    for k in range(n):
        st = rig.tool_state()
        tt = rig.t - t0
        tau = kp * (th0 + omega * tt - st["theta"]) + kd * (omega - st["omega_x"])
        rig.set_tool_wrench(np.zeros(3), np.array([tau, 0.0, 0.0]))
        rig.step(H.DT)
        th = st["theta"] - th0
        if th >= 0.4:
            taus.append(tau)
        if k % k_s:
            continue
        for i, c in base.items():
            m.geom_rgba[i] = c
        mx = 0.0
        for j in range(d.ncon):
            c = d.contact[j]
            g = [int(c.geom[0]), int(c.geom[1])]
            pad = [x for x in g if x in padL]
            if not pad or c.dist >= 0:
                continue
            mujoco.mj_contactForce(m, d, j, f6)
            r = f6[0] / (K * -c.dist)
            if -c.dist >= 1e-6:                      # the maximum over pads at least 1 um deep
                mx = max(mx, r)
            m.geom_rgba[pad[0]] = (*_ratio_rgb(r), 1.0)
        mx_all = max(mx_all, mx)
        arm = np.mean(taus) / (2 * H.MU * N) * 1e3 if taus else None
        rend.update_scene(d, cam)
        frames.append(label_bar(rend.render().copy(), [
            title, f"turn {th:4.2f} rad   arm {arm:4.2f} mm" + (f" (law {law:.2f})" if law else "") if arm else
            f"turn {th:4.2f} rad   arm from 0.4 rad" + (f" (law {law:.2f} mm)" if law else "")], legend=True))
    rend.close()
    arm = float(np.mean(taus) / (2 * H.MU * N) * 1e3)
    return frames, {"member": member, "model": name, "N": N, "omega": omega, "arm_mm": arm, "law_mm": law,
                    "arm_over_law": arm / law if law else None, "max_ratio_1um": mx_all, "K_sphere": K}


def cmd_bed(a):
    import contact_bed_common as CB
    clips = []
    for mem in a.members:
        out = MEDIA / f"bed_spin_{mem}.mp4"
        t0 = time.time()
        fr, info = film_bed(mem, a.N, a.omega)
        CB.write_h264(fr, out, crf=20)
        _log({"film": "bed", "clip": out.name, **info, "frames": len(fr), "when": time.strftime("%Y-%m-%d %H:%M")})
        print(f"{out.name}: arm {info['arm_mm']:.2f} mm, law {info['law_mm']}, max f/(K d) {info['max_ratio_1um']:.1f}, "
              f"{len(fr)} frames, {time.time() - t0:.0f} s", flush=True)
        clips.append(out)
    if len(clips) > 1:
        print("tile", tile(clips, "bed_spin_tile"), "frames", flush=True)


# ---------------------------------------------------------------------------------- (e) D7 replay

def _install(tip):
    import pad_usability_backends as PUB
    import reorient_backends as RB
    import tip_codesign_rig as R
    ns = argparse.Namespace(E=RB.E_TPU, pf=None, tr=RB.PAD_TR, tag=None, h="fixed", K="median")
    R.install(ns, force_tip=tip)
    PUB.SCN = R.SCN / f"film_{tip}"


def cmd_replay(a):
    import contact_bed_common as CB
    import pad_usability_backends as PUB
    import reorient_backends_films as RBF
    RBF.W, RBF.H = PW, PH
    orig = PUB.label_bar
    PUB.label_bar = lambda fr, lines: label_bar(fr, lines)
    clips = []
    for tip in a.tips:
        _install(tip)
        out = MEDIA / f"replay_{a.hand}_s{a.seed}_{tip}.mp4"
        t0 = time.time()
        fr = PUB.film_mujoco(a.hand, a.seed, "pads", SERVO, tip_name(tip), mu="1.0")
        CB.write_h264(fr, out, crf=20)
        _log({"film": "replay", "clip": out.name, "tip": tip, "hand": a.hand, "seed": a.seed, "frames": len(fr),
              "when": time.strftime("%Y-%m-%d %H:%M")})
        print(f"{out.name}: {len(fr)} frames, {time.time() - t0:.0f} s", flush=True)
        clips.append(out)
    PUB.label_bar = orig
    if len(clips) > 1:
        print("tile", tile(clips, f"replay_{a.hand}_s{a.seed}_tile"), "frames", flush=True)


# ---------------------------------------------------------------------------------- (d) HOM on a CAD-hand cell

def film_hom(tip, cell_spec, seed, cand_index):
    import mujoco
    import hand_object_scale_films as SF
    import hand_object_scale_sim as S
    import hom_turn3 as HT
    import reorient_backends as RB
    import reorient_backends_films as RBF
    import tip_codesign_rig as R
    shape, dd, t = cell_spec.split(":")
    dd, t = float(dd), float(t)
    ns = argparse.Namespace(E=RB.E_TPU, pf=None, tr=RB.PAD_TR, tag=None, h="fixed", K="median")
    R.install(ns, force_tip=tip)
    S.SCENES = R.SCN.parent / "scale" / f"{tip}_{R.tag_of(ns)}_mu0.7"
    S.MU, S.PLANT = 0.7, SERVO
    cell = next(c for c, sh, d_ in S.cell_list("diag") if sh == shape and d_ == dd and c.get("t") == t)
    cands = S.candidates(cell, shape, dd)
    cand = cands[cand_index]
    sc = S.cell_scenes(cell, shape, dd, models=("pads", "pt"))
    m_pt = mujoco.MjModel.from_xml_path(str(sc["pt"][0]))
    gg = S.grasp_geometry(cell, shape, dd, cand["h"], cand["spread"])
    gp = S.grip_poses(m_pt, gg, RB.plant_args(SERVO)["kp"])
    meta = S.candidate_meta(sc["pads"][1], gg, gp)
    plant = HT.MjPlant(SF.film_scene(Path(sc["pads"][0]), dd / 2000.0), meta)
    S.set_palm(plant.m, gg["palm"])
    mirror = S.Mirror(Path(sc["pt"][0]))
    S.set_palm(mirror.m, gg["palm"])
    m, dat = plant.m, plant.d
    pads = [i for i in range(m.ngeom) if "_pad" in (m.geom(i).name or "")]
    for i in pads:
        m.geom_rgba[i] = (0.75, 0.4, 0.25, 0.0)
    RBF.W, RBF.H = PW, PH
    rend, cam = RBF.renderer_for(m, np.asarray(meta["tool7"], float)[:3])
    cam.distance = 0.24 + 1.4 * dd / 1000.0
    frames, k = [], [0]
    title = tip_name(tip)

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
        frames.append(label_bar(rend.render().copy(), [title, f"t {tt:4.2f} s   turn {info.get('th_deg', 0):+5.1f}°   "
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
    s3 = json.loads((D / "final/summary_stage3.json").read_text()) if (D / "final/summary_stage3.json").exists() else {}
    key = f"{a.cell.split(':')[1]}:{a.cell.split(':')[2]}"
    clips = []
    for tip in a.tips:
        ci = (((s3.get("candidates") or {}).get(tip) or {}).get(f"mu0.7_{SERVO}") or {}).get("chosen_grasp", {}).get(key)
        if ci is None:
            import tip_codesign_analysis as AN
            sel = [r for r in AN.rows(AN._sel_dir(tip, "0.7", SERVO) / "turn.jsonl") if r.get("status") == "ok"
                   and int(r["seed"]) == 0 and f"{float(r['d_mm']):g}:{r.get('t'):g}" == key and r["shape"] == a.cell.split(":")[0]]
            ci = max((AN.held_turn(r), int(r["cand"])) for r in sel)[1]
        out = MEDIA / f"hom_{a.cell.replace(':', '_')}_s{a.seed}_{tip}.mp4"
        t0 = time.time()
        fr, res = film_hom(tip, a.cell, a.seed, ci)
        CB.write_h264(fr, out, crf=20)
        _log({"film": "hom", "clip": out.name, "tip": tip, "cell": a.cell, "seed": a.seed, "cand": ci,
              "held": res["held"], "turn_end_deg": res["turn_end_deg"], "held_turn_deg": res.get("held_turn_deg"),
              "frames": len(fr), "when": time.strftime("%Y-%m-%d %H:%M")})
        print(f"{out.name}: cand {ci}, held {res['held']}, turn {res['turn_end_deg']:.1f} deg, {len(fr)} frames, "
              f"{time.time() - t0:.0f} s", flush=True)
        clips.append(out)
    if len(clips) > 1:
        print("tile", tile(clips, f"hom_{a.cell.replace(':', '_')}_s{a.seed}_tile"), "frames", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bed")
    b.add_argument("--members", nargs="+", default=["tpu1", "tcd_rs3_re3", "tcd_rs3_re3_p0.2", "tpu6"])
    b.add_argument("--N", type=float, default=1.0)
    b.add_argument("--omega", type=float, default=1.0)
    r = sub.add_parser("replay")
    r.add_argument("--tips", nargs="+", required=True)
    r.add_argument("--hand", default="D7")
    r.add_argument("--seed", type=int, default=0)
    h = sub.add_parser("hom")
    h.add_argument("--tips", nargs="+", required=True)
    h.add_argument("--cell", default="cylinder:16:0.625")
    h.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    {"bed": cmd_bed, "replay": cmd_replay, "hom": cmd_hom}[a.cmd](a)


if __name__ == "__main__":
    main()
