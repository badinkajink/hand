#!/usr/bin/env python3
"""CAD-hand control tasks for the printable 85A fingertips (docs/handoff/20261016-sim2real_fingertip_readiness.md, steps
3, 4 and 6).

Each variant of scripts/sim2real_fingertip_scenes.py runs the scale study's cells (scripts/hand_object_scale_sim.py,
SCALE_HAND=cad_v1) with its own fingers: the CAD links, the variant's printed tip mass and inertia, the magnetometer board
on the DIP link (a collision box), and the tip's 1 mm pads at the variant's modulus, on the servo plant
(kp0.72_kv0_fr0.18_fl0_dp0.0144), friction 1.0 unless --mu says otherwise. Cell scenes, generated designs and base scenes
are cached per variant under logs/20261016-sim2real_fingertip_readiness/.

  hom    HOM (hom_turn3, kp_model 0.72, the support post removed when the squeeze ends) on every candidate grasp of the
         cells, seeds 0-2; per run the scale study's row plus: peak joint torque and the share of ticks at the 0.18 N m
         limit, RMS and peak servo-command rate, the share of ticks after the squeeze with three fingers' pads each
         pressing with the cylinder's weight, and contacts of non-pad finger geoms (board, DIP-link and link spheres)
         with the cylinder
  icem   pinned iCEM (hand_object_scale_planners icem_pin) on each cell's grasp with the best median held HOM turn
  film   one HOM or iCEM run rendered (640 x 480 panels, pads touching the cylinder red, the board green)

Cells (the screens' four): cylinder:25:0.125, cylinder:16:0.625, cylinder:10:0.75, cylinder:16:-0.125.
Rows: docs/experiments/20261016-sim2real_fingertip_readiness/cells/<what>.jsonl, one fsynced row per run, resumable.

    SCALE_HAND=cad_v1 ~/.claude/bin/resguard.sh run --mem 3G --cpu 200 -- .venv/bin/python \\
        scripts/sim2real_fingertip_cells.py hom --variants block4_g30 cap_g30
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

os.environ["SCALE_HAND"] = "cad_v1"
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

D = ROOT / "docs/experiments/20261016-sim2real_fingertip_readiness"
OUT = D / "cells"
ASSETS = ROOT / "assets/mjcf/experimental/20261016-sim2real_fingertip_readiness"
LOGS = ROOT / "logs/20261016-sim2real_fingertip_readiness"
SERVO = "kp0.72_kv0_fr0.18_fl0_dp0.0144"
CELLS = ["cylinder:25:0.125", "cylinder:16:0.625", "cylinder:10:0.75", "cylinder:16:-0.125"]
TAU_LIM = 0.18
BUILDER_HK = {"v1": ("fixed", "median"), "v2": ("local", "each")}
SCRIPT = "scripts/sim2real_fingertip_cells.py"


def setup(variant, builder="v2", mu=1.0):
    """hand_object_scale_sim pointed at the variant: its CAD fingers, its tip's pads at its modulus, its own caches."""
    import hand_object_scale_sim as S
    import reorient_backends as RB
    import tip_codesign_rig as R
    spec = json.loads((ASSETS / variant / "README.json").read_text())
    h, K = BUILDER_HK[builder]
    ns = argparse.Namespace(E=spec["pad_E_Pa"], pf=None, tr=RB.PAD_TR, tag=None, h=h, K=K)
    R.install(ns, force_tip=spec["key"])
    S.CAD_DIR = ASSETS / variant / "cad_v1"
    S.GEN = LOGS / "gen" / variant
    S.SCENES = LOGS / "cells" / f"{variant}_{builder}_mu{mu:g}"
    S.MU, S.PLANT, S.BUILDER = mu, SERVO, builder
    return S, spec


def cell_of(S, spec_str):
    shape, dd, t = spec_str.split(":")
    dd, t = float(dd), float(t)
    cell = next(c for c, sh, d_ in S.cell_list("diag") if sh == shape and d_ == dd and c.get("t") == t)
    return cell, shape, dd


class Probe:
    """Per control tick: joint torques, servo commands, fingers loaded through their pads, non-pad contacts."""

    def __init__(self, plant, mass_kg, t_squeeze_end):
        m = plant.m
        self.plant, self.w, self.t0 = plant, mass_kg * 9.81, t_squeeze_end
        self.pad = np.array([("_pad" in (m.geom(g).name or "")) for g in range(m.ngeom)])
        self.fing = np.full(m.ngeom, -1)
        self.body = {}
        for g in range(m.ngeom):
            bn = m.body(m.geom_bodyid[g]).name
            for k, f in enumerate(("thumb", "index", "middle")):
                if bn.startswith(f + "_"):
                    self.fing[g] = k
                    nm = m.geom(g).name or ""
                    self.body[g] = "board" if "board" in nm else bn[len(f) + 1:]
        self.tool = np.zeros(m.ngeom, bool)
        self.tool[list(plant.tool_geoms)] = True
        self.tau, self.cmd, self.t = [], [], []
        self.three, self.nonpad, self.nonpad_F = [], {}, 0.0
        self.f6 = np.zeros(6)

    def __call__(self, pl, t, info):
        import mujoco
        m, d = pl.m, pl.d
        self.t.append(t)
        self.tau.append(np.abs(d.actuator_force[pl.act]).copy())
        self.cmd.append(d.ctrl[pl.act].copy())
        F = np.zeros(3)
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom[0]), int(c.geom[1])
            if self.tool[g1] == self.tool[g2]:
                continue
            g = g2 if self.tool[g1] else g1
            k = self.fing[g]
            if k < 0:
                continue
            mujoco.mj_contactForce(m, d, i, self.f6)
            if self.f6[0] <= 1e-6:
                continue
            if self.pad[g]:
                F[k] += self.f6[0]
            else:
                b = self.body.get(g, "?")
                self.nonpad[b] = self.nonpad.get(b, 0) + 1
                self.nonpad_F = max(self.nonpad_F, float(self.f6[0]))
        if t >= self.t0:
            self.three.append(int(np.count_nonzero(F >= self.w)))

    def summary(self, dt):
        tau = np.asarray(self.tau)
        cmd = np.asarray(self.cmd)
        rate = np.diff(cmd, axis=0) / dt if len(cmd) > 1 else np.zeros((1, 9))
        after = np.asarray(self.t[1:]) >= self.t0 if len(cmd) > 1 else np.zeros(1, bool)
        r = rate[after] if after.any() else rate
        th = np.asarray(self.three) if self.three else np.zeros(1)
        return dict(peak_tau_Nm=round(float(tau.max()), 4), tau_at_limit_frac=round(float((tau >= 0.98 * TAU_LIM).mean()), 4),
                    cmd_rate_rms=round(float(np.sqrt((r ** 2).mean())), 4), cmd_rate_peak=round(float(np.abs(r).max()), 3),
                    three_pad_frac=round(float((th == 3).mean()), 4), pads_loaded_end=int(th[-1]),
                    nonpad_contact_ticks=self.nonpad, nonpad_F_max_N=round(self.nonpad_F, 3))


def _append(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _done(path, keys):
    out = set()
    if path.exists():
        for line in open(path):
            r = json.loads(line)
            if r.get("status") == "ok":
                out.add(tuple(r.get(k) for k in keys))
    return out


def cmd_hom(a):
    import mujoco
    import hom_turn3 as H
    out = OUT / f"{a.tag}.jsonl"
    keys = ("variant", "builder", "mu", "tag", "shape", "d_mm", "cand", "seed")
    done = _done(out, keys)
    for variant in a.variants:
        S, spec = setup(variant, a.builder, a.mu)
        kp = S.RB.plant_args(SERVO)["kp"]
        prm = dict(H.DEFAULTS, kp_model=kp, post_off=1.0)
        for kv in a.set:
            k, v = kv.split("=")
            prm[k] = float(v)
        for cs in a.cells:
            cell, shape, dd = cell_of(S, cs)
            cands = S.candidates(cell, shape, dd)
            sc = None
            for ci, cand in enumerate(cands):
                if a.cands and ci not in a.cands:
                    continue
                seeds = [s for s in a.seeds if (variant, a.builder, a.mu, cell["tag"], shape, dd, ci, s) not in done]
                if not seeds:
                    continue
                if sc is None:
                    sc = S.cell_scenes(cell, shape, dd, models=("pads", "pt"))
                    m_pt = mujoco.MjModel.from_xml_path(str(sc["pt"][0]))
                gg = S.grasp_geometry(cell, shape, dd, cand["h"], cand["spread"])
                gp = S.grip_poses(m_pt, gg, kp)
                path, meta0 = sc["pads"]
                meta = S.candidate_meta(meta0, gg, gp)
                plant = H.MjPlant(path, meta)
                S.set_palm(plant.m, gg["palm"])
                if a.board == "off":                       # the board keeps its mass; only its collisions go
                    for g in range(plant.m.ngeom):
                        if "board" in (plant.m.geom(g).name or ""):
                            plant.m.geom_contype[g] = plant.m.geom_conaffinity[g] = 0
                mirror = S.Mirror(Path(sc["pt"][0]))
                S.set_palm(mirror.m, gg["palm"])
                for s in seeds:
                    row = {"variant": variant, "tip": spec["key"], "E_Pa": spec["pad_E_Pa"], "builder": a.builder,
                           "mu": a.mu, "plant": SERVO, "hand_model": "cad_v1", "tag": cell["tag"], "t": cell.get("t"),
                           "shape": shape, "d_mm": dd, "model": "pads", "cand": ci, "n_cand": len(cands),
                           "h_mm": round(cand["h"] * 1000, 2), "spread_mm": round(cand["spread"] * 1000, 2), "seed": s,
                           "controller": "hom_turn3", "board_collides": a.board == "on",
                           **{f"prm_{k}": v for k, v in prm.items()},
                           "when": time.strftime("%Y-%m-%d %H:%M")}
                    probe = Probe(plant, meta["mass_kg"], prm["t_grip"] + prm["t_squeeze"])
                    try:
                        res = S.turn_rollout(plant, mirror, meta, s, prm, frame_cb=probe)
                        row.update(res, **probe.summary(prm["dt_c"]), status="ok")
                        row["trace"] = res.pop("trace", [])
                    except Exception as e:
                        row.update(status="error", error=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-1200:])
                    row.update(script=SCRIPT, n_pads=meta0.get("n_pads"))
                    _append(out, row)
                    print(f"{variant} {a.builder} {cs} c{ci} s{s}: " + (
                        f"held {row['held']} turn {row['turn_end_deg']} pads {row['pads_loaded_end']} tau {row['peak_tau_Nm']} "
                        f"rate {row['cmd_rate_rms']} nonpad {row['nonpad_contact_ticks']} {row['wall_s']} s"
                        if row["status"] == "ok" else row["error"]), flush=True)


def hom_best(path, variant, builder, mu):
    from collections import defaultdict
    by = defaultdict(lambda: defaultdict(list))
    for line in open(path):
        r = json.loads(line)
        if r.get("status") == "ok" and r["variant"] == variant and r["builder"] == builder and r["mu"] == mu:
            by[(r["tag"], r["shape"], r["d_mm"])][r["cand"]].append(r["held_turn_deg"])
    return {k: max(sorted(cc), key=lambda c: float(np.median(cc[c]))) for k, cc in by.items()}


def cmd_icem(a):
    import hand_object_scale_mpc as M
    import hand_object_scale_planners as HP
    out = OUT / f"{a.tag}.jsonl"
    keys = ("variant", "builder", "mu", "config", "tag", "shape", "d_mm", "cand", "seed")
    done = _done(out, keys)
    for variant in a.variants:
        S, spec = setup(variant, a.builder, a.mu)
        M.S = S
        best = hom_best(OUT / f"{a.hom_tag}.jsonl", variant, a.builder, a.mu)
        prm = HP.prm_of(a.config, a.set)
        for cs in a.cells:
            cell, shape, dd = cell_of(S, cs)
            key = (cell["tag"], shape, dd)
            if key not in best:
                print(f"{variant} {cs}: no HOM rows", flush=True)
                continue
            ci = best[key]
            seeds = [s for s in a.seeds if (variant, a.builder, a.mu, a.config, cell["tag"], shape, dd, ci, s) not in done]
            if not seeds:
                continue
            plant, meta, geo, _, _ = M.setup(cell, shape, dd, ci)
            for s in seeds:
                row = {"variant": variant, "tip": spec["key"], "E_Pa": spec["pad_E_Pa"], "builder": a.builder, "mu": a.mu,
                       "plant": SERVO, "config": a.config, "tag": cell["tag"], "t": cell.get("t"), "shape": shape,
                       "d_mm": dd, **geo, "seed": s, **{f"prm_{k}": v for k, v in prm.items()},
                       "when": time.strftime("%Y-%m-%d %H:%M")}
                probe = Probe(plant, meta["mass_kg"], prm["t_grip"] + prm["t_squeeze"])
                try:
                    res = M.mpc_turn(plant, meta, s, prm, nthread=a.nthread, frame_cb=probe)
                    row.update(res, **probe.summary(0.01), status="ok")
                except Exception as e:
                    row.update(status="error", error=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-1200:])
                row["script"] = SCRIPT
                _append(out, row)
                print(f"{variant} {a.config} {cs} c{ci} s{s}: " + (
                    f"held {row['held']} turn {row['turn_end_deg']} loaded {row['fingers_loaded_end']} three "
                    f"{row.get('three_frac')} tau {row['peak_tau_Nm']} rate {row.get('cmd_rate_rms')} {row['wall_s']} s"
                    if row["status"] == "ok" else row["error"]), flush=True)


def cmd_film(a):
    """One HOM run of a variant on a cell, rendered: pads touching the cylinder red, the board green and drawn red while it
    touches the cylinder; stills at the end of the grip, mid-turn and the end beside the clip."""
    import mujoco
    import contact_bed_common as CB
    import hand_object_scale_films as SF
    import hom_turn3 as H
    import reorient_backends_films as RBF
    import tip_codesign_film as TF
    from PIL import Image
    S, spec = setup(a.variant, a.builder, a.mu)
    cell, shape, dd = cell_of(S, a.cell)
    cands = S.candidates(cell, shape, dd)
    ci = a.cand
    if ci is None:
        best = hom_best(OUT / "hom.jsonl", a.variant, a.builder, a.mu)
        ci = best[(cell["tag"], shape, dd)]
    cand = cands[ci]
    kp = S.RB.plant_args(SERVO)["kp"]
    sc = S.cell_scenes(cell, shape, dd, models=("pads", "pt"))
    m_pt = mujoco.MjModel.from_xml_path(str(sc["pt"][0]))
    gg = S.grasp_geometry(cell, shape, dd, cand["h"], cand["spread"])
    gp = S.grip_poses(m_pt, gg, kp)
    meta = S.candidate_meta(sc["pads"][1], gg, gp)
    plant = H.MjPlant(SF.film_scene(Path(sc["pads"][0]), dd / 2000.0), meta)
    S.set_palm(plant.m, gg["palm"])
    mirror = S.Mirror(Path(sc["pt"][0]))
    S.set_palm(mirror.m, gg["palm"])
    m, dat = plant.m, plant.d
    pads = [i for i in range(m.ngeom) if "_pad" in (m.geom(i).name or "")]
    boards = [i for i in range(m.ngeom) if "board" in (m.geom(i).name or "")]
    for i in pads:
        m.geom_rgba[i] = (0.75, 0.4, 0.25, 0.0)
    RBF.W, RBF.H = a.width, a.height
    m.vis.global_.offwidth, m.vis.global_.offheight = max(a.width, 640), max(a.height, 480)
    rend, cam = RBF.renderer_for(m, np.asarray(meta["tool7"], float)[:3])
    cam.distance = 0.24 + 1.4 * dd / 1000.0
    cam.elevation += a.elev
    cam.azimuth += a.azim
    frames, k, stills, hits = [], [0], {}, {"board": 0}
    title = f"{a.variant}: {spec['key']}, E {spec['pad_E_Pa'] / 1e6:.2f} MPa, {a.cell}, grasp {ci}, seed {a.seed}"
    prm = dict(H.DEFAULTS, kp_model=kp, post_off=1.0)
    t_sq = prm["t_grip"] + prm["t_squeeze"]

    def cb(pl, tt, info):
        k[0] += 1
        if k[0] % a.every:
            return
        for i in pads:
            m.geom_rgba[i, 3] = 0.0
        for i in boards:
            m.geom_rgba[i] = (0.1, 0.45, 0.2, 1.0)
        for i in range(dat.ncon):
            g1, g2 = int(dat.contact[i].geom[0]), int(dat.contact[i].geom[1])
            for g, o in ((g1, g2), (g2, g1)):
                if o in pl.tool_geoms and m.geom_bodyid[g] != pl.tb:
                    if g in pads:
                        m.geom_rgba[g] = (0.9, 0.1, 0.1, 1.0)
                    elif g in boards:
                        m.geom_rgba[g] = (0.95, 0.1, 0.1, 1.0)
                        hits["board"] += 1
        rend.update_scene(dat, cam)
        fr = TF.label_bar(rend.render().copy(), [title, f"t {tt:4.2f} s   turn {info.get('th_deg', 0):+5.1f} deg   "
                                                        f"{info.get('phase', '')}"])
        frames.append(fr)
        for name, tw in (("grip", prm["t_grip"]), ("squeeze", t_sq), ("mid", t_sq + 1.0), ("late", t_sq + 2.5)):
            if name not in stills and tt >= tw - 1e-9:
                stills[name] = fr
    res = S.turn_rollout(plant, mirror, meta, a.seed, prm, frame_cb=cb)
    stills.setdefault("end", frames[-1])
    stills["end"] = frames[-1]
    frames += [frames[-1]] * 25
    rend.close()
    media = D / "media"
    media.mkdir(parents=True, exist_ok=True)
    stem = f"hom_{a.cell.replace(':', '_')}_c{ci}_s{a.seed}_{a.variant}"
    CB.write_h264(frames, media / f"{stem}.mp4", crf=20)
    for name, fr in stills.items():
        Image.fromarray(fr).save(media / f"{stem}_{name}.png")
    row = {"film": "hom", "clip": f"{stem}.mp4", "variant": a.variant, "cell": a.cell, "cand": ci, "seed": a.seed,
           "held": res["held"], "turn_end_deg": res["turn_end_deg"], "fingers_loaded_end": res["fingers_loaded_end"],
           "board_contact_frames": hits["board"], "frames": len(frames), "stills": sorted(stills),
           "when": time.strftime("%Y-%m-%d %H:%M")}
    _append(media / "films.jsonl", row)
    print(json.dumps(row), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("hom", "icem"):
        p = sub.add_parser(name)
        p.add_argument("--variants", nargs="+", required=True)
        p.add_argument("--cells", nargs="+", default=CELLS)
        p.add_argument("--builder", default="v2", choices=list(BUILDER_HK))
        p.add_argument("--mu", type=float, default=1.0)
        p.add_argument("--set", nargs="*", default=[])
    h = sub.choices["hom"]
    h.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    h.add_argument("--cands", nargs="*", type=int, default=[])
    h.add_argument("--tag", default="hom")
    h.add_argument("--board", default="on", choices=["on", "off"],
                   help="off: the board's collisions disabled (its mass stays), rows to --tag (e.g. hom_noboard)")
    i = sub.choices["icem"]
    i.add_argument("--seeds", nargs="+", type=int, default=[0])
    i.add_argument("--config", default="icem_pin")
    i.add_argument("--nthread", type=int, default=6)
    i.add_argument("--hom-tag", default="hom")
    i.add_argument("--tag", default="icem")
    f = sub.add_parser("film")
    f.add_argument("--variant", required=True)
    f.add_argument("--cell", default="cylinder:16:0.625")
    f.add_argument("--cand", type=int, default=None)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--builder", default="v2", choices=list(BUILDER_HK))
    f.add_argument("--mu", type=float, default=1.0)
    f.add_argument("--width", type=int, default=960)
    f.add_argument("--height", type=int, default=720)
    f.add_argument("--every", type=int, default=4)
    f.add_argument("--elev", type=float, default=10.0)
    f.add_argument("--azim", type=float, default=0.0)
    a = ap.parse_args()
    {"hom": cmd_hom, "icem": cmd_icem, "film": cmd_film}[a.cmd](a)


if __name__ == "__main__":
    main()
