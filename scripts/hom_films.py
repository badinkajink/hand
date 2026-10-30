#!/usr/bin/env python3
"""Films of the HOM tasks on the SR2 hand with the v4b fingertip, for the two 10-29 pages and the code walkthrough
(2026-10-30; owner: "some of the videos on these pages use the incorrect older fingertips ... more videos would be good").

Every clip is a CPU rollout of scripts/hom_tasks.py on the v4b pads scene (SCALE_HAND=cad_v2) from a layout and grasp
of the GPU evaluator's layout sets (scripts/hom_gpu.py layout_list: grid2d_size, grid2dval_<task>_size, local_twist40_size),
filmed close up with the finger links see-through. Clips of one film run side by side (two or three) or two by two
(four), each under a label bar.

  ~/.claude/bin/resguard.sh run --mem 3G --cpu 200 -- .venv/bin/python scripts/hom_films.py [name ...]
Names: grasp_twist, grasp_pinpush, post, local, sizes, four, reanchor (all when none given). Writes
docs/experiments/20261029-foundational_pose_review/media/20261030-film_<name>.mp4 (+ .jpg poster) and, for reanchor,
docs/experiments/20261029-hom_control_review/media/20261030-film_reanchor.mp4; each film's clips and their results in
rows/films_v4b.json of the design page folder.
"""
from __future__ import annotations

import os

os.environ.setdefault("SCALE_HAND", "cad_v2")
os.environ.setdefault("MUJOCO_GL", "egl")

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import hom_gpu as HG  # noqa: E402
import hom_grasp_overlap as OV  # noqa: E402
import hom_landscape as L  # noqa: E402

T, S, H = HG.T, HG.S, HG.H
FINGERS = HG.FINGERS
FPD = ROOT / "docs/experiments/20261029-foundational_pose_review"
MEDIA, ROWS = FPD / "media", FPD / "rows"
WALK_MEDIA = ROOT / "docs/experiments/20261029-hom_control_review/media"
INFO = ROWS / "films_v4b.json"
NAMES = {"twist": "twist", "roll": "roll", "pinpush": "pin-push", "press": "press"}


AXES = ((0, (0.85, 0.1, 0.1, 1.0)), (1, (0.1, 0.65, 0.1, 1.0)), (2, (0.1, 0.25, 0.9, 1.0)))


def arrow(scn, a, b, rgba, width=0.0012, kind="arrow"):
    """An arrow (or a capsule rod) from a to b added to the scene."""
    import mujoco
    if scn.ngeom >= scn.maxgeom:
        return
    t = mujoco.mjtGeom.mjGEOM_ARROW if kind == "arrow" else mujoco.mjtGeom.mjGEOM_CAPSULE
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, t, np.zeros(3), np.zeros(3), np.zeros(9), np.asarray(rgba, np.float32))
    mujoco.mjv_connector(g, t, width, np.asarray(a, float), np.asarray(b, float))
    scn.ngeom += 1


class Film:
    """Frames of a rollout every fourth period: the finger links see-through, the support post faded once removed, and
    the object's axes drawn as rods through it (x red, y green; z blue on the sphere), so that its turns show."""

    def __init__(self, plant, setup, w=480, h=400):
        import mujoco
        self.r = mujoco.Renderer(plant.m, h, w)
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = [0.0, 0.0, S.Z_OBJ + 0.004]
        cyl = setup["shape"] == "cylinder"
        self.cam.distance = 0.16 if cyl else 0.15
        self.cam.azimuth, self.cam.elevation = (340.0, 14.0) if cyl else (60.0, 22.0)
        m = plant.m
        for g in range(m.ngeom):
            bn = m.body(m.geom_bodyid[g]).name
            if any(bn.startswith(f + "_") for f in FINGERS) and "_pad" not in (m.geom(g).name or ""):
                m.geom_rgba[g, 3] = min(float(m.geom_rgba[g, 3]), 0.45)
        self.post = plant.post
        self.sphere = not cyl
        self.r_obj = float(setup["r"])

    def frame(self, plant, title, line):
        import hom_contact_rig as HR
        if plant.d.time >= T.T_LIFT:
            plant.m.geom_rgba[self.post, 3] = 0.12
        self.r.update_scene(plant.d, self.cam)
        o = plant.d.xpos[plant.tb]
        R = plant.d.xmat[plant.tb].reshape(3, 3)
        L_ = self.r_obj + 0.012                           # rods through the object, 12 mm out on both sides
        for k, c in AXES[:3 if self.sphere else 2]:
            arrow(self.r.scene, o - L_ * R[:, k], o + L_ * R[:, k], c, width=0.0015, kind="capsule")
        return HR.annotate_bottom(self.r.render().copy(), line, size=11)

    def close(self):
        self.r.close()


def find(task, kind, d_mm, call, grasp):
    """The (key, mounts, setup) of one layout and grasp in a layout set of hom_gpu.layout_list."""
    for key, mxy, setup in HG.layout_list(task, kind, 0, d_mm):
        if int(key["call"]) == int(call) and int(key.get("grasp", 0)) == int(grasp):
            return key, mxy, setup
    raise KeyError(f"{task} {kind} {d_mm:g} mm: call {call} grasp {grasp} not found")


def clip(task, kind, d_mm, call, grasp, label, seed=0, var="base", gpu_filter=False):
    """One CPU rollout filmed: (frames with the label on top, the rollout's result with the fingers' smallest gap at the
    touch pose, gap_mm). In the label, {cpu} becomes the filmed run's score. gpu_filter: the GPU evaluator's collision
    filter (pads and rigid spheres under 2 mm meet the object only)."""
    import re
    import mujoco
    import hom_contact_rig as HR
    _key, mxy, setup = find(task, kind, d_mm, call, grasp)
    scene = HG.Scene(T.SHAPE[task], d_mm)
    m_g = mujoco.MjModel.from_xml_path(str(scene.pads_path))
    gaps = OV.touch_gaps(scene, m_g, mujoco.MjData(m_g), setup, mxy)
    bm, q_open = scene.prepare(mxy, setup)
    plant = T.Plant(scene.pads_path, setup)
    if gpu_filter:
        m = plant.m
        fine = re.compile(r"_(dip_link|cassette|cap)_s\d+$")
        filt = np.array([("_pad" in (m.geom(g).name or "")) or bool(fine.search(m.geom(g).name or "")
                                                                     and m.geom_size[g][0] < 0.002) for g in range(m.ngeom)])
        tool = np.flatnonzero(m.geom_bodyid == plant.tb)
        m.geom_contype[filt], m.geom_conaffinity[filt] = 2, 0
        m.geom_conaffinity[tool] = m.geom_conaffinity[tool] | 2
    m_pt = mujoco.MjModel.from_xml_path(str(scene.pt_path))
    for m in (plant.m, m_pt):
        for f in FINGERS:
            m.body_pos[m.body(f"{f}_mount").id] = bm[f]
    S.set_palm(m_pt, setup["palm"])
    fs, os_ = T.sr2_specs(setup)
    mirror = H.Mirror(m_pt, fs, os_)
    film = Film(plant, setup)
    over, noise = T.VARIANTS[var]
    try:
        res, _trace, frames = T.rollout(task, setup, plant, mirror, q_open, seed, T.PRM,
                                        replace(H.HomParams(dt=T.DT_C), **over), film, noise,
                                        T.PLANT_VARIANTS.get(var, {}).get("hold_s", 0.0))
    finally:
        film.close()
    label = label.replace("{cpu}", f"{score(task, res):.2f}")
    res = dict(res, gap_mm=min(gaps.values()), gaps_mm=gaps)
    return [HR.annotate(f, label, size=14) for f in frames], res


def compose(clips, out, grid=None):
    """Clips (lists of equal-sized frames) side by side, or two by two for four; the shorter ones hold their last frame.
    Writes out (.mp4) and a poster (.jpg) at 55 % of the film."""
    import hom_contact_rig as HR
    from PIL import Image
    n = max(len(c) for c in clips)
    clips = [c + [c[-1]] * (n - len(c)) for c in clips]
    if grid == (2, 2):
        frames = [np.concatenate([np.concatenate([a, b], 1), np.concatenate([c, d], 1)], 0)
                  for a, b, c, d in zip(*clips)]
    else:
        frames = [np.concatenate(fs, 1) for fs in zip(*clips)]
    out.parent.mkdir(parents=True, exist_ok=True)
    HR.write_mp4(frames, out, fps=25)
    Image.fromarray(frames[int(0.55 * len(frames))]).convert("RGB").save(out.with_suffix(".jpg"), quality=85)
    print("wrote", out.relative_to(ROOT), f"({len(frames)} frames)", flush=True)


def score(task, r):
    return L.rollout_score(task, r)


def gpu_mean(task, size, call, grasp, rows_dir=L.GRID_ROWS, name=None):
    sc, _ = L.grid_scores(task, size, rows_dir, name)
    v = sc.get((call, grasp))
    return float(np.mean(v)) if v else float("nan")


def peak_call(task, size):
    """The grid call at the peak of the largest basin of the smoothed best-of-16 map (design page, Table 11)."""
    xs, ys, M = L.grid_maps(task, size)
    sm = L.smooth_map(M["best"])
    pk, pks = L.peaks(sm)
    top = max(pks, key=lambda p: int((pk == p).sum()))
    x, y = float(xs[top % len(xs)]), float(ys[top // len(xs)])
    return next(c for c, (a, b, _r) in L.grid_poses(task, size).items() if a == x and b == y), (x, y)


def call_at(task, size, x, y):
    return next(c for c, (a, b, _r) in L.grid_poses(task, size).items() if a == x and b == y)


def best_grasp(task, size, call, clear=False):
    """The layout's best grasp by the GPU's seeds 0-2; clear=True: among the grasps whose fingers stay L.GAP_OK apart."""
    if not clear:
        return next(r["best_grasp"] for r in L.grid_table(task, size) if r["call"] == call)
    ok = {k for k, g in L.overlap_gaps(task, size).items() if g >= L.GAP_OK}
    _p, b, _f = L.masked_table(task, size, ok)
    return b[call][1]


def clear_peak(task, size):
    """The call and (x_sep, y_sep) of the largest basin's peak with the overlapping grasps left out."""
    ok = {k for k, g in L.overlap_gaps(task, size).items() if g >= L.GAP_OK}
    poses, b, _f = L.masked_table(task, size, ok)
    x, y = L.peak_of(poses, b)
    return call_at(task, size, x, y), (x, y)


def res_brief(task, r):
    keys = ("held", "ramp_three_pads_max_deg", "w_rmse_ratio", "theta_rmse_deg", "F_mean", "gap_mm")
    return dict({k: r.get(k) for k in keys if k in r}, score=round(score(task, r), 3))


def film_grasp(task, size):
    """The foundational pose against the best grasp on the task's peak layout of the grid, overlapping grasps left out,
    seed 0."""
    call, (x, y) = clear_peak(task, size)
    bg = best_grasp(task, size, call, clear=True)
    clips, info = [], {"layout": [x, y], "call": call}
    for name, g in (("foundational pose", 0), ("best grasp", bg)):
        fr, res = clip(task, "grid2d_size", size, call, g,
                       f"{name}: score {{cpu}} here, {gpu_mean(task, size, call, g):.2f} on the GPU")
        clips.append(fr)
        info[name] = dict(res_brief(task, res), grasp=g)
    compose(clips, MEDIA / f"20261030-film_grasp_{task}.mp4")
    return info


def film_post():
    """The pin-push example (grid call 280, foundational pose) at seed 7, at seed 8, and at seed 8 with the object held
    at its start until the forces come on."""
    clips, info = [], {}
    for name, sd, var in (("seed 7", 7, "base"), ("seed 8", 8, "base"), ("seed 8, object held to 0.8 s", 8, "hold")):
        fr, res = clip("pinpush", "grid2d_size", 25.0, 280, 0, f"{name}: score {{cpu}}", seed=sd, var=var)
        clips.append(fr)
        info[name] = res_brief("pinpush", res)
    compose(clips, MEDIA / "20261030-film_post.mp4")
    return info


def film_local():
    """The local refinement's start with the largest gain on seeds 12-14 among those whose fingers stay clear, and its best
    neighbour, seed 12."""
    summ = json.loads((ROWS / "local" / "summary_twist40.json").read_text())
    rows = {r["call"]: r for r in (json.loads(x) for x in (ROWS / "local_twist40.jsonl").read_text().splitlines())}
    gaps = {json.loads(x)["call"]: json.loads(x)["min_mm"] for p_ in sorted(L.OVERLAP.glob("local_twist40_twist_d40_p*.jsonl"))
            for x in p_.read_text().splitlines() if x.strip()}
    start = {o["center"]: next(c for c, r in rows.items() if r["center"] == o["center"] and r["radius_mm"] == 0.0)
             for o in summ["per_start"]}
    clear = [o for o in summ["per_start"] if gaps.get(start[o["center"]], -1) >= L.GAP_OK
             and gaps.get(o["best_call"], -1) >= L.GAP_OK]          # fingers 2 mm apart at both touch poses
    o = max(clear, key=lambda r: r["best_check"] - r["start_check"])
    c0 = start[o["center"]]
    clips, info = [], {"center": o["center"]}
    for name, c, sc in (("start", c0, o["start_check"]), (f"best neighbour, {o['best_radius_mm']:.1f} mm away",
                                                            o["best_call"], o["best_check"])):
        fr, res = clip("twist", "local_twist40_size", 40.0, c, 0, f"{name}: score {{cpu}} here, {sc:.2f} on the GPU",
                       seed=12)
        clips.append(fr)
        info[name] = dict(res_brief("twist", res), call=c, d=rows[c]["d"])
    compose(clips, MEDIA / "20261030-film_local.mp4")
    return info


def film_sizes():
    """The twist of the 30 mm and the 63 mm sphere, each on its size's peak layout with its best grasp, seed 0."""
    clips, info = [], {}
    for size in (30.0, 63.0):
        call, (x, y) = clear_peak("twist", size)
        bg = best_grasp("twist", size, call, clear=True)
        fr, res = clip("twist", "grid2d_size", size, call, bg,
                       f"{size:g} mm sphere, ({x:.0f}, {y:.0f}) mm: score {{cpu}} here, "
                       f"{gpu_mean('twist', size, call, bg):.2f} on the GPU")
        clips.append(fr)
        info[f"{size:g}"] = dict(res_brief("twist", res), layout=[x, y], grasp=bg)
    compose(clips, MEDIA / "20261030-film_sizes.mp4")
    return info


def film_four():
    """The four tasks on the grid layout whose worst task score is highest with each task's best grasp, overlapping grasps
    left out (rows/grid2d_v4b/overlap/summary.json)."""
    x, y = json.loads((L.OVERLAP / "summary.json").read_text())["cross"]["clear"]["maximin_xy"]
    clips, info = [], {"layout": [x, y]}
    for task, size in (("twist", 40.0), ("roll", 40.0), ("pinpush", 25.0), ("press", 25.0)):
        call = call_at(task, size, x, y)
        bg = best_grasp(task, size, call, clear=True)
        obj = f"{size:g} mm {'sphere' if T.SHAPE[task] == 'sphere' else 'cylinder'}"
        fr, res = clip(task, "grid2d_size", size, call, bg,
                       f"{NAMES[task]}, {obj}: score {{cpu}} here, {gpu_mean(task, size, call, bg):.2f} on the GPU")
        clips.append(fr)
        info[task] = dict(res_brief(task, res), grasp=bg)
    compose(clips, MEDIA / "20261030-film_four.mp4", grid=(2, 2))
    return info


def film_reanchor():
    """The pin-push example without and with the re-anchor, seed 0 (code walkthrough, Section 10)."""
    clips, info = [], {}
    for name, var in (("without the re-anchor", "base"), ("with the re-anchor", "reanchor")):
        fr, res = clip("pinpush", "grid2d_size", 25.0, 280, 0, f"{name}: score {{cpu}}", var=var)
        clips.append(fr)
        info[name] = res_brief("pinpush", res)
    compose(clips, WALK_MEDIA / "20261030-film_reanchor.mp4")
    return info


def film_overlap():
    """The best grasp of the twist's peak layout on the GPU grid, overlapping grasps included, seed 0: the CPU plant with
    all contacts against the same plant with the GPU's collision filter."""
    call, (x, y) = peak_call("twist", 40.0)
    bg = best_grasp("twist", 40.0, call)
    clips, info = [], {"layout": [x, y], "call": call, "grasp": bg, "gpu": gpu_mean("twist", 40.0, call, bg)}
    for name, gf in (("all contacts", False), ("pads and small rigid spheres meet the object only", True)):
        fr, res = clip("twist", "grid2d_size", 40.0, call, bg, f"{name}: score {{cpu}}", gpu_filter=gf)
        clips.append(fr)
        info[name] = res_brief("twist", res)
    compose(clips, MEDIA / "20261030-film_overlap.mp4")
    return info


FILMS = {"overlap": film_overlap, "grasp_twist": lambda: film_grasp("twist", 40.0), "grasp_pinpush": lambda: film_grasp("pinpush", 25.0),
         "post": film_post, "local": film_local, "sizes": film_sizes, "four": film_four, "reanchor": film_reanchor}


def main():
    names = sys.argv[1:] or list(FILMS)
    info = json.loads(INFO.read_text()) if INFO.exists() else {}
    for n in names:
        t0 = time.time()
        info[n] = dict(FILMS[n](), when=time.strftime("%Y-%m-%d %H:%M"))
        INFO.write_text(json.dumps(info, indent=1, default=float))
        print(n, f"{time.time() - t0:.0f} s", json.dumps(info[n], default=float)[:300], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
