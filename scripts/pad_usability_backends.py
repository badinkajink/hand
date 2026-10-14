#!/usr/bin/env python3
"""Plan replays and HOM closed-loop turns of the eight deployed hands D1-D8 in six MuJoCo fingertip models and in
Drake hydroelastic (experiment U2 of the 2026-10-14 pad-usability study; plan in
docs/notes/20261014-contact_model_assessment_plan.md).

Per rollout it measures the tool's turn at the end of the replay (change of the tool axis' angle from the vertical
between the end of the grip and the end of the run, deg), whether the hand keeps the tool (held_end: tool less than
20 mm below its grip height and touched by at least two fingers), the summed normal force of the fingertips on the
tool at the hold (N), and the wall time per physics step (us). It wraps scripts/reorient_backends.py (RB) and
scripts/hom_turn3.py (HT) without editing them:

  model keys  pads   1 mm sphere pads (radius 0.75 mm) on the printed TPU block, colliding with every geom (RB 'pads')
              padsT  the same pads touching the tool only; the block's convex mesh takes every other contact
              pads2  pads at 2 mm spacing and 1.5 mm radius, tool only (the 2 mm pads of the RL scenes)
              pt     point contact on the block's convex mesh, condim 3
              ptmc   pt with multiccd (up to four contacts per convex pair)
              pt4    pt with condim 4, priority 1 and friction '1 <spin> 0.0001' on each {f}_tipgeom; spin is the pads'
                     twist-onset arm 2.527 F^0.070 mm at the plant's per-finger grip: 2.8 mm on the servo plant,
                     3.0 mm on the working plant (--spin overrides)
  numerics    bed    1 ms, implicitfast, elliptic cone, Newton, 100 iterations, tolerance 1e-10 (RB.NUMERICS['bed'])
              rl     the RL trainer's: 2 ms, implicitfast, elliptic, Newton, 10 iterations, 20 line-search iterations,
                     tolerance 1e-8 (env_build.MujocoCfg); run it with --ir 10
  --dt        physics step in ms: the MuJoCo timestep, or Drake's discrete step (RB.DT before DrakeBench is built);
              default 1 ms ('bed', Drake) or 2 ms ('rl'); it enters the case_id and the scene name

    PY=logs/20261001-hom_contact/venv/bin/python
    $PY scripts/pad_usability_backends.py run --hands D7 --tips tpu2.7 --models pads pt4 \
        --plants kp0.72_kv0_fr0.18_fl0_dp0.0144 --mu 1.0 --ir 100 --seeds 0 --out <rows.jsonl>
    $PY scripts/pad_usability_backends.py run --sim drake --hands D7 --tips tpu2.7 \
        --plants kp0.72_kv0_fr0.18_fl0_dp0.0144 --mu 1.0 --seeds 0 1 2 --out <rows.jsonl>
    $PY scripts/pad_usability_backends.py hom --hands D7 --sims mujoco drake --seeds 0 \
        --plant kp0.72_kv0_fr0.18_fl0_dp0.0144 --mu 1.0 --set kp_model=0.72 --out <rows.jsonl>
    MUJOCO_GL=egl $PY scripts/pad_usability_backends.py film --hand D7 --seed 0 --plant kp0.72_kv0_fr0.18_fl0_dp0.0144

Rows: one fsynced JSON line per rollout in --out (U2: docs/experiments/20261014-pad_usability/agree/*.jsonl);
`run` resumes by case_id, `hom` by (hand, sim, plant, seed). Scenes: logs/20261014-pad_usability/scenes/<key>_<numerics>_dt<ms>/.
Films: docs/experiments/20261014-pad_usability/agree/media/ (one clip per model, a 3 x 2 tile and a PNG poster).
"""
from __future__ import annotations

import argparse
import functools
import gc
import json
import math
import os
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fingertip_geometry as G  # noqa: E402
import reorient_backends as RB  # noqa: E402

_RB_BUILD = RB.build_scene
_G_WRITE_OBJ = G.write_obj


def _write_obj_atomic(path, V, F):
    """fingertip_geometry.write_obj through a temporary file: a process compiling a scene never reads a mesh that a
    parallel process is rewriting (every build rewrites the block's OBJ files)."""
    path = Path(path)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    _G_WRITE_OBJ(tmp, V, F)
    os.replace(tmp, path)
    return path


G.write_obj = _write_obj_atomic

SCRIPT = "scripts/pad_usability_backends.py"
SCN = ROOT / "logs/20261014-pad_usability/scenes"
AGREE = ROOT / "docs/experiments/20261014-pad_usability/agree"
MEDIA = AGREE / "media"
SERVO, WORK = "kp0.72_kv0_fr0.18_fl0_dp0.0144", "kp4_kv0_fr1_fl0_dp0.08"
SPIN = {SERVO: 0.0028, WORK: 0.0030}          # m: the pads' onset arm 2.527 F^0.070 mm at 4.3 and 11.6 N per finger
MODELS = ("pads", "padsT", "pads2", "pt", "ptmc", "pt4")
PADS2_S, PADS2_RS = 0.002, 0.0015
RB.NUMERICS.setdefault("rl", dict(timestep="0.002", integrator="implicitfast", cone="elliptic", solver="Newton",
                                  iterations="10", ls_iterations="20", tolerance="1e-8"))
DT_DEFAULT = {"bed": 1.0, "rl": 2.0}
TITLES = {"drake": "Drake hydroelastic", "pads": "MuJoCo, 1 mm pads", "padsT": "MuJoCo, 1 mm pads, tool only",
          "pads2": "MuJoCo, 2 mm pads", "pt": "MuJoCo, point contact", "ptmc": "MuJoCo, multiccd",
          "pt4": "MuJoCo, condim 4"}


# ---------------------------------------------------------------------------------- scenes

def build(hand: str, tip: str, key: str, plant: str, numerics: str = "bed", ir: float = 100.0, mu="1.0",
          dt_ms: float | None = None, spin: float | None = None, prefix: str = ""):
    """MJCF of the bench scene for model `key` (see the module docstring) and its metadata. RB.build_scene writes the
    base scene ('pads', 'padsT' or 'pt') into a folder of its own per prefix, key, numerics and step, so processes
    building other keys or steps never overwrite each other's scenes; this function then edits it and writes the
    final scene with the key and the step in its name."""
    if key not in MODELS:
        raise ValueError(f"unknown model key {key!r}")
    dt_ms = float(dt_ms or DT_DEFAULT[numerics])
    base = {"pads": "pads", "padsT": "padsT", "pads2": "padsT"}.get(key, "pt")
    out_dir = SCN / f"{prefix}{key}_{numerics}_dt{dt_ms:g}"
    saved_num, saved_rt = RB.NUMERICS[numerics], RB.replace_tips
    if saved_num is None:
        raise ValueError("numerics 'scene' keeps the deploy scene's 2 ms step; use 'bed' or 'rl'")
    RB.NUMERICS[numerics] = dict(saved_num, timestep=f"{dt_ms * 1e-3:g}")
    if key == "pads2":
        RB.replace_tips = functools.partial(saved_rt, rs=PADS2_RS)
    try:
        path, meta = _RB_BUILD(hand, tip, base, plant, numerics, ir, mu, out_dir=out_dir,
                               pad_s=PADS2_S if key == "pads2" else RB.PAD_S)
    finally:
        RB.NUMERICS[numerics], RB.replace_tips = saved_num, saved_rt
    tree = ET.parse(path)
    root = tree.getroot()
    if key == "ptmc":
        opt = root.find("option")
        fl = opt.find("flag")
        if fl is None:
            fl = ET.SubElement(opt, "flag")
        fl.set("multiccd", "enable")
    if key == "pt4":
        spin = float(spin if spin is not None else SPIN[plant])
        for f in RB.FINGERS:
            g = root.find(f".//body[@name='{f}_tip']/geom[@name='{f}_tipgeom']")
            g.set("condim", "4")
            g.set("priority", "1")
            g.set("friction", f"{meta['mu']:g} {spin:g} 0.0001")
        meta["spin_m"] = spin
    if key == "pads2":
        meta["pad_rs_m"] = PADS2_RS
    elif key in ("pads", "padsT"):
        meta["pad_rs_m"] = RB.PAD_RS
    name = (f"{hand}_{tip}_{key}_{plant}_{numerics}_ir{ir:g}_mu{meta['mu']:g}_dt{dt_ms:g}"
            + (f"_spin{meta['spin_m'] * 1e3:g}" if key == "pt4" else "") + ".xml")
    final = out_dir / name
    tmp = final.with_name(final.name + f".{os.getpid()}.tmp")
    tree.write(tmp)
    os.replace(tmp, final)
    meta.update(model_key=key, dt_ms=dt_ms, numerics=numerics, scene_file=str(final))
    return final, meta


def append_row(path, row):
    """One JSON line written by a single write() on an O_APPEND descriptor, then fsynced: processes that share an
    output file never interleave their lines."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(row, default=float) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        n = os.write(fd, data)
        if n != len(data):
            raise OSError(f"short write {n} of {len(data)} bytes to {path}")
        os.fsync(fd)
    finally:
        os.close(fd)


def drake_scene(hand, tip, plant, mu, prefix="drake_"):
    """The single-geom MuJoCo scene that DrakeBench parses (RB.main builds 'pt' at 'bed' numerics for Drake)."""
    return build(hand, tip, "pt", plant, "bed", 100.0, mu, dt_ms=1.0, prefix=prefix)


# ---------------------------------------------------------------------------------- plan replays

def cmd_run(a):
    RB.JITTER_MODE = a.jitter
    have = RB.done(a.out)
    models = a.models if a.sim == "mujoco" else ["hydro"]
    for hand in a.hands:
        plan, traj = RB.load_plan(hand)
        for tip in a.tips:
            for model in models:
                for plant in a.plants:
                    for numerics in (a.numerics if a.sim == "mujoco" else ["bed"]):
                        for ir in (a.ir if a.sim == "mujoco" else [0.0]):
                            for mu in a.mu:
                                for dt_ms in (a.dt or [DT_DEFAULT[numerics]]):
                                    run_cases(a, hand, plan, traj, tip, model, plant, numerics, ir, mu, float(dt_ms),
                                              have)


def run_cases(a, hand, plan, traj, tip, model, plant, numerics, ir, mu, dt_ms, have):
    scene = meta = db = None
    gc.collect()
    mkey = model
    if model == "pt4":                              # the torsional coefficient is part of the model
        mkey = f"pt4s{(a.spin if a.spin is not None else SPIN[plant]) * 1e3:g}"
    for seed in a.seeds:
        cid = f"{a.sim}|{hand}|{tip}|{mkey}|{plant}|{numerics}|{ir:g}|{mu}|{a.jitter}|dt{dt_ms:g}|{seed}"
        if cid in have:
            continue
        t0 = time.time()
        row = {"case_id": cid, "sim": a.sim, "hand": hand, "tag": RB.HANDS[hand][0], "tip": tip, "model": model,
               "plant": plant, "numerics": numerics, "impratio": ir if a.sim == "mujoco" else None, "mu": mu,
               "jitter_mode": a.jitter, "dt_ms": dt_ms, "seed": seed, "jitter": RB.jitter(seed)}
        try:
            if a.sim == "mujoco":
                if scene is None:
                    scene, meta = build(hand, tip, model, plant, numerics, ir, mu, dt_ms, a.spin)
                res, trace = RB.run_mujoco(scene, meta, plan, traj, seed, None)
            else:
                if scene is None:
                    scene, meta = drake_scene(hand, tip, plant, mu)
                if db is None:
                    RB.DT = dt_ms * 1e-3            # DrakeBench reads RB.DT for its discrete step
                    db = RB.DrakeBench(scene, meta)
                RB.DT = dt_ms * 1e-3                # run_drake reads it for the per-step wall time and n_steps
                res, trace = RB.run_drake(db, meta, plan, traj, seed)
            row.update(res)
            row["scene"] = str(Path(scene).relative_to(ROOT))
            row["n_pads"] = meta.get("n_pads")
            row["pad_d0"] = meta.get("pad_d0")
            row["pad_spacing_m"] = meta.get("pad_spacing_m")
            row["pad_rs_m"] = meta.get("pad_rs_m")
            row["spin_m"] = meta.get("spin_m")
            if a.traces:
                row["trace"] = trace
            else:
                row["trace_coarse"] = [[s_["t"], round(s_["cos"], 4), round(s_["z"], 5),
                                        round(sum(s_["F"].values()), 3), sum(1 for f in RB.FINGERS if s_["n"][f] > 0)]
                                       for s_ in trace[::5]]
        except Exception as e:                      # a failed case is a row
            row.update(status="failed", error=repr(e), traceback=traceback.format_exc()[-2000:])
        row.update(script=SCRIPT, git_rev=RB.git_rev(), wall_s=time.time() - t0, when=time.strftime("%Y-%m-%d %H:%M"))
        append_row(a.out, row)
        have.add(cid)
        print(f"{cid:78s} {row.get('status'):9s} turn {row.get('turn_hold_deg', float('nan')):+6.1f}"
              f"/{row.get('turn_end_deg', float('nan')):+6.1f} deg held {row.get('held_hold')}/{row.get('held_end')} "
              f"F {row.get('F_pads_hold_N')} N  {row.get('us_per_step_median', 0):.0f} us/step "
              f"wall {row['wall_s']:.1f} s", flush=True)


# ---------------------------------------------------------------------------------- HOM closed loop

def cmd_hom(a):
    """hom_turn3's run loop with a resume check: scenes under logs/20261014-pad_usability/scenes/hom."""
    import hom_turn3 as HT
    HT.SCENES = SCN / "hom"
    prm = dict(HT.DEFAULTS)
    for kv in a.set:
        k, v = kv.split("=")
        if k not in prm:
            raise SystemExit(f"unknown controller parameter {k}")
        prm[k] = float(v)
    mu = a.mu if a.mu == "scene" else float(a.mu)
    have = set()
    if a.out.exists():
        for line in open(a.out):
            try:
                r = json.loads(line)
                have.add((r["hand"], r["sim"], r["plant"], int(r["seed"])))
            except Exception:
                pass
    a.out.parent.mkdir(parents=True, exist_ok=True)
    for hand in a.hands:
        todo = {sim: [s for s in a.seeds if (hand, sim, a.plant, s) not in have] for sim in a.sims}
        if not any(todo.values()):
            continue
        p_pads, p_mesh, meta, meta_mesh = HT.scenes(hand, a.plant, mu)
        mirror = HT.Mirror(p_mesh)
        for sim in a.sims:
            if not todo[sim]:
                continue
            plant = HT.MjPlant(p_pads, meta) if sim == "mujoco" else HT.DrakePlant(p_mesh, meta_mesh)
            for seed in todo[sim]:
                row = {"case_id": f"hom|{hand}|{sim}|{a.plant}|{a.mu}|{seed}", "hand": hand, "sim": sim, "seed": seed,
                       "plant": a.plant, "mu": a.mu, "tip": "tpu2.7", "controller": "hom_turn3",
                       **{f"prm_{k}": v for k, v in prm.items()}, "when": time.strftime("%Y-%m-%d %H:%M")}
                t0 = time.time()
                try:
                    res, trace = HT.rollout(plant, mirror, meta, seed, prm)
                    row.update(res, status="ok")
                    row["trace_coarse"] = [[s["t"], s["th_deg"], s["z_mm"], round(sum(s["F"].values()), 3),
                                            sum(1 for f in RB.FINGERS if s["n"][f] > 0)] for s in trace[::5]]
                except Exception as e:
                    row.update(status="error", error=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-1500:])
                row.update(script=SCRIPT, git_rev=RB.git_rev(), wall_total_s=round(time.time() - t0, 2))
                append_row(a.out, row)
                print(f"{hand} {sim:<6} seed {seed}: " + (
                    f"turn {row['turn_end_deg']:+.1f} deg, frozen {row['frozen_deg']}, held {row['held']}, "
                    f"fingers {row['fingers_end']}, F {row['F_end']}, {row['wall_s']} s wall"
                    if row["status"] == "ok" else row["error"]), flush=True)
            del plant
            gc.collect()


# ---------------------------------------------------------------------------------- films

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def label_bar(frame, lines):
    """Label bar across the top of a 480 x 360 panel: the condition in bold, then hand, seed, time and the tilt of the
    tool axis above the horizontal. 24 and 22 px type stay 13 px or larger when the 1440 px tile is shown 828 px wide."""
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(frame)
    dr = ImageDraw.Draw(im, "RGBA")
    fb, fr = ImageFont.truetype(FONT_B, 24), ImageFont.truetype(FONT, 22)
    dr.rectangle([0, 0, im.width, 62], fill=(255, 255, 255, 235))
    dr.line([0, 62, im.width, 62], fill=(0, 0, 0, 255), width=1)
    dr.text((10, 4), lines[0], fill=(10, 10, 10), font=fb)
    if len(lines) > 1:
        dr.text((10, 33), lines[1], fill=(40, 40, 40), font=fr)
    return np.asarray(im.convert("RGB"))


def _label_line(hand, seed, t, c):
    """Hand, seed, simulated time and the tilt of the tool axis above the horizontal (deg) from its vertical cosine."""
    return f"{hand} seed {seed}   t {t:4.2f} s   tilt {math.degrees(math.asin(max(-1.0, min(1.0, c)))):+3.0f}\u00b0"


def film_mujoco(hand, seed, key, plant, title, mu="1.0", numerics="bed", ir=100.0, dt_ms=None, spin=None):
    """reorient_backends_films.film_mujoco on a scene built by build(): same camera, a frame every 40 ms of simulated
    time; pad spheres touching the tool are drawn red, and so is the block of a single-geom tip while it touches."""
    import mujoco
    import reorient_backends_films as RBF
    plan, traj = RB.load_plan(hand)
    scene, meta = build(hand, "tpu2.7", key, plant, numerics, ir, mu, dt_ms, spin, prefix="film_")
    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)
    tb = m.body(RB.OBJ).id
    qa = m.jnt_qposadr[m.body_jntadr[tb]]
    tool7 = np.asarray(meta["tool7"], float)
    dx, dy, dyaw = RB.jitter(seed)
    d.qpos[qa:qa + 3] = tool7[:3] + [dx, dy, 0.0]
    d.qpos[qa + 3:qa + 7] = RB._yaw_quat(tool7[3:], dyaw)
    act = {}
    for f in RB.FINGERS:
        for j in RB.JOINTS:
            n = f"{f}_{j}"
            d.qpos[m.jnt_qposadr[m.joint(n).id]] = meta["q0"][n]
            act[n] = m.actuator(f"a_{n}").id
    mujoco.mj_forward(m, d)
    rend, cam = RBF.renderer_for(m, tool7[:3])
    touch = {i: m.geom_rgba[i].copy() for i in range(m.ngeom)
             if "_pad" in (m.geom(i).name or "") or (m.geom(i).name or "").endswith("_tipgeom")}
    tool_geoms = {i for i in range(m.ngeom) if m.geom_bodyid[i] == tb}
    segs = RB.schedule(plan, traj)
    k_film = max(1, int(round(1 / 25 / m.opt.timestep)))
    frames, step = [], 0
    for t_end, tgt, _ in segs:
        for n_, a_ in act.items():
            d.ctrl[a_] = tgt[n_]
        while d.time < t_end - 1e-9:
            mujoco.mj_step(m, d)
            step += 1
            if step % k_film == 0:
                for i, c in touch.items():
                    m.geom_rgba[i] = c
                for i in range(d.ncon):
                    g1, g2 = int(d.contact[i].geom[0]), int(d.contact[i].geom[1])
                    for g, o in ((g1, g2), (g2, g1)):
                        if g in touch and o in tool_geoms:
                            m.geom_rgba[g] = (0.9, 0.1, 0.1, 1.0)
                rend.update_scene(d, cam)
                R9 = np.zeros(9)
                mujoco.mju_quat2Mat(R9, d.qpos[qa + 3:qa + 7])
                frames.append(label_bar(rend.render().copy(), [title, _label_line(hand, seed, d.time, R9[8])]))
    rend.close()
    return frames


def film_drake(hand, seed, plant, title, mu="1.0"):
    """reorient_backends_films.film_drake with the scene from this study's folder and the larger label bar."""
    import reorient_backends_films as RBF
    orig_build, orig_label = RB.build_scene, RBF.label

    def build_here(h, tip, model, pl, numerics, ir, mu_, **kw):
        return drake_scene(h, tip, pl, mu_, prefix="film_drake_")

    def label_here(frame, lines):
        import re                                   # RBF's second line ends 'cos +0.62': show the tilt instead
        mt = re.search(r"t\s+([0-9.]+) s\s+cos ([+-]?[0-9.]+)", lines[1])
        if mt:
            lines = [lines[0], _label_line(hand, seed, float(mt.group(1)), float(mt.group(2)))]
        return label_bar(frame, lines)
    RB.build_scene, RBF.label = build_here, label_here
    RB.DT = 0.001
    try:
        return RBF.film_drake(hand, seed, "tpu2.7", title, plant=plant, mu=mu)
    finally:
        RB.build_scene, RBF.label = orig_build, orig_label


def cmd_film(a):
    import contact_bed_common as CB
    import reorient_backends_films as RBF
    MEDIA.mkdir(parents=True, exist_ok=True)
    pshort = "servo" if a.plant == SERVO else ("work" if a.plant == WORK else a.plant)
    clips = []
    for key in a.conds:
        title = TITLES[key] + (f" ({SPIN.get(a.plant, 0) * 1e3:g} mm)" if key == "pt4" else "")
        out = MEDIA / f"{a.hand}_s{a.seed}_{pshort}_{key}.mp4"
        if not out.exists():
            t0 = time.time()
            fr = film_drake(a.hand, a.seed, a.plant, title) if key == "drake" else \
                film_mujoco(a.hand, a.seed, key, a.plant, title)
            CB.write_h264(fr, out)
            print(f"wrote {out.relative_to(ROOT)} ({len(fr)} frames, {time.time() - t0:.0f} s)", flush=True)
        clips.append(out)
    if len(clips) > 1:
        stem = MEDIA / f"{a.hand}_s{a.seed}_{pshort}_models"
        n = RBF.tile_stream(clips, stem.with_suffix(".mp4"), stem.with_name(stem.name + "_poster.png"), cols=3,
                            poster_at=a.poster_at)
        print("tile", n, "frames ->", stem.with_suffix(".mp4").relative_to(ROOT), flush=True)


# ---------------------------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="plan replays")
    r.add_argument("--sim", default="mujoco", choices=["mujoco", "drake"])
    r.add_argument("--hands", nargs="+", default=list(RB.HANDS))
    r.add_argument("--tips", nargs="+", default=["tpu2.7"])
    r.add_argument("--models", nargs="+", default=["pads"], choices=MODELS)
    r.add_argument("--plants", nargs="+", default=[SERVO])
    r.add_argument("--numerics", nargs="+", default=["bed"], choices=["bed", "rl"])
    r.add_argument("--ir", nargs="+", type=float, default=[100.0])
    r.add_argument("--mu", nargs="+", default=["1.0"])
    r.add_argument("--dt", nargs="+", type=float, default=None, help="physics step, ms")
    r.add_argument("--spin", type=float, default=None, help="pt4 torsional coefficient, m (default by plant)")
    r.add_argument("--seeds", nargs="+", type=int, default=[0])
    r.add_argument("--jitter", default="j2", choices=["j2", "j0916"])
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--traces", action="store_true", help="keep the 50 Hz trace in the row")
    h = sub.add_parser("hom", help="HOM closed-loop turn (hom_turn3) with a resume check")
    h.add_argument("--hands", nargs="+", default=list(RB.HANDS))
    h.add_argument("--sims", nargs="+", default=["mujoco", "drake"])
    h.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    h.add_argument("--plant", default=SERVO)
    h.add_argument("--mu", default="1.0")
    h.add_argument("--set", nargs="*", default=[], help="controller parameters k=v (hom_turn3.DEFAULTS)")
    h.add_argument("--out", type=Path, required=True)
    f = sub.add_parser("film", help="one clip per condition and a 3 x 2 tile")
    f.add_argument("--hand", default="D7")
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--plant", default=SERVO)
    f.add_argument("--conds", nargs="+", default=["drake", "pads", "pads2", "pt4", "pt", "ptmc"])
    f.add_argument("--poster-at", type=float, default=0.62)
    a = ap.parse_args()
    {"run": cmd_run, "hom": cmd_hom, "film": cmd_film}[a.cmd](a)


if __name__ == "__main__":
    main()
