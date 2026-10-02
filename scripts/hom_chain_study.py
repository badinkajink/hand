#!/usr/bin/env python3
"""Driver for docs/experiments/20261002-hom_chain: Exp 1 of arXiv 2609.25619 on D8, the nominal chain filmed
in six contact models, a perturbed MuJoCo batch (seeds 1-10), and a close-up of the sphere-packed pad.
Resumable: rows already in the jsonl files are skipped.

    PY=logs/20261001-hom_contact/venv/bin/python
    MUJOCO_GL=egl $PY scripts/hom_chain_study.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import hom_chain as K  # noqa: E402
import hom_contact_rig as H  # noqa: E402
import hom_control as C  # noqa: E402
import hom_hand_brake as B  # noqa: E402

OUT = ROOT / "docs/experiments/20261002-hom_chain"
MEDIA = OUT / "media"
NOMINAL = {"p4s_closed": ("mj:point4s", "closed"), "p4s_open": ("mj:point4s", "open"),
           "s1_closed": ("mj:spheres:s1:rs0.75:tr0.02", "closed"), "mp3_closed": ("mj:point3", "closed"),
           "dhy_closed": ("drake:hydro:rt0.01", "closed"), "dpt_closed": ("drake:point", "closed")}
BATCH = {"p4s": "mj:point4s", "s1": "mj:spheres:s1:rs0.75:tr0.02"}
SEEDS = range(1, 11)


def done(path, key):
    if not path.exists():
        return set()
    return {key(json.loads(l)) for l in open(path) if l.strip()}


def main():
    MEDIA.mkdir(parents=True, exist_ok=True)
    e1 = OUT / "exp1.jsonl"
    have = done(e1, lambda r: r["plant"])
    for plant in ("cal", "stiff"):
        if plant not in have:
            r = C.exp1(plant, 500.0, film=str(MEDIA / f"20261002-exp1_{plant}.mp4"))
            H.append_row(e1, r)
            print("exp1", plant, round(r["rmse_ang_dps"], 1), round(r["rmse_lin_mmps"], 1), flush=True)
    ch = OUT / "chain.jsonl"
    have = done(ch, lambda r: (r["spec"], r["brake"], r["trial"]["seed"]))
    for key, (spec, brake) in NOMINAL.items():
        if (spec, brake, 0) in have:
            continue
        r = K.run_chain(spec, 0, brake, film=str(MEDIA / f"20261002-chain_{key}.mp4"))
        r["key"] = key
        H.append_row(ch, r)
        print("nominal", key, r["phi_end"], r["chain_ok"], round(r["wall_s"], 1), flush=True)
    for bk, spec in BATCH.items():
        for brake in ("open", "closed"):
            for seed in SEEDS:
                if (spec, brake, seed) in have:
                    continue
                r = K.run_chain(spec, seed, brake)
                r["key"] = f"{bk}_{brake}"
                r.pop("trace", None)
                H.append_row(ch, r)
                print("batch", bk, brake, seed, r["phi_end"], r["chain_ok"], flush=True)
    tile_films()
    closeup()


def tile_films():
    import imageio.v2 as imageio
    from PIL import Image
    order = ["dhy_closed", "s1_closed", "p4s_closed", "dpt_closed", "mp3_closed", "p4s_open"]
    fr = {k: [f for f in imageio.get_reader(str(MEDIA / f"20261002-chain_{k}.mp4"))] for k in order}
    n = max(len(v) for v in fr.values())
    get = lambda k, i: fr[k][min(i, len(fr[k]) - 1)]  # noqa: E731
    grid = [np.concatenate([np.concatenate([get(k, i) for k in order[:3]], 1),
                            np.concatenate([get(k, i) for k in order[3:]], 1)], 0) for i in range(n)]
    H.write_mp4(grid, MEDIA / "20261002-chain_six_models.mp4", fps=25)
    Image.fromarray(grid[min(n - 1, 130)]).save(MEDIA / "20261002-chain_six_models_poster.png")


def closeup():
    """Thumb pad at 1, 0.5 and 2 mm sphere spacing pressed 0.4 mm into the tool, contacts drawn."""
    import mujoco
    from PIL import Image
    path, _ = K.postures([0.0, -0.0004])
    q = path[-0.0004]
    dirs = K.contact_dirs(path[0.0])
    trial = K.make_trial(0)
    imgs = []
    for s_mm in (2, 1, 0.5):
        xml, info, lay = K.chain_scene(f"mj:spheres:s{s_mm}:rs0.75:tr0.02", trial, dirs)
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        P, u, a = K.pinch_frame()
        s_hat = lay["c"] - trial["d_cg"] * lay["axis"]
        d.qpos[[m.jnt_qposadr[m.joint(n).id] for f in B.FINGERS for n in B.JOINTS[f]]] = q
        d.qpos[[m.jnt_qposadr[m.joint(n).id] for n in ("palm_x", "palm_y", "palm_z", "palm_yaw")]] = np.r_[s_hat - P, 0]
        mujoco.mj_forward(m, d)
        tool = m.body("tool").id
        ncon = sum(1 for i in range(d.ncon) if tool in (m.geom_bodyid[d.contact[i].geom[0]], m.geom_bodyid[d.contact[i].geom[1]])
                   and m.body(m.geom_bodyid[d.contact[i].geom[0]]).name.startswith("thumb"))
        m.geom_rgba[m.geom("tool").id] = [0.55, 0.6, 0.68, 0.25]
        r = mujoco.Renderer(m, 480, 480)
        opt = mujoco.MjvOption()
        opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        m.vis.scale.contactwidth, m.vis.scale.contactheight = 0.08, 0.02
        cam = mujoco.MjvCamera()
        tip = d.xpos[m.body("thumb_tip").id]
        cam.lookat[:] = tip + 0.006 * u
        cam.distance, cam.azimuth, cam.elevation = 0.07, math.degrees(math.atan2(-u[1], -u[0])) + 35, -20
        r.update_scene(d, cam, opt)
        img = H.annotate(r.render().copy(), f"{s_mm:g} mm spacing: {info['n_spheres']} spheres, {ncon} touching", size=14)
        imgs.append(img)
    Image.fromarray(np.concatenate(imgs, 1)).save(MEDIA / "20261002-sphere_pad_closeup.png")
    print("closeup done", flush=True)


if __name__ == "__main__" and len(sys.argv) == 1:
    main()


PERF = {"mp3": "mj:point3", "p4s": "mj:point4s", "s2": "mj:spheres:s2:rs0.75:tr0.02",
        "s1": "mj:spheres:s1:rs0.75:tr0.02", "s05": "mj:spheres:s0.5:rs0.75:tr0.02",
        "dpt": "drake:point", "dhy": "drake:hydro:rt0.01"}


def perf():
    """Nominal chain with the closed-loop brake, no film, one core: wall time per simulated second."""
    import time
    p = OUT / "perf.jsonl"
    have = done(p, lambda r: r["spec"])
    for key, spec in PERF.items():
        if spec in have:
            continue
        try:
            r = K.run_chain(spec, 0, "closed")
        except ValueError as e:                     # pad stiffness/relaxation outside MuJoCo's soft-contact range
            H.append_row(p, {"key": key, "spec": spec, "error": str(e)})
            print("perf", key, "skipped:", e, flush=True)
            continue
        H.append_row(p, {"key": key, "spec": spec, "wall_s": r["wall_s"], "sim_s": r["sim_s"], "phi_end": r["phi_end"],
                         "chain_ok": r["chain_ok"], "n_spheres": (r.get("info") or {}).get("n_spheres"),
                         "when": time.strftime("%Y-%m-%d %H:%M")})
        print("perf", key, round(r["wall_s"], 2), round(r["sim_s"], 2), r["phi_end"], flush=True)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "perf":
    perf()
