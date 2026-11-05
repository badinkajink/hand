#!/usr/bin/env python3
"""Hand-level runs of the fingertip-shape rerun (docs/handoff/20261103-fingertip_shape_rerun.md, ledger step 5): the task
screen of 2026-10-14 and 10-15 on the v4b hand, at 2.5 MPa and builder v2, with each tip in place of the v4b pad.

  reach   the four cells of the stage score on cad_v2 (hand_object_scale_kin.grasp_scan and summarise on the v4b finger's
          kinematics, the diagonal family's layouts at t 0.125, 0.625, 0.75 and -0.375) -> <run>/kin/reach.json
  scale   hand_object_scale_sim's hold test and HOM turn (hom_turn3) on those cells with SCALE_HAND=cad_v2 and the tip in
          place of hand_object_scale_sim.TIP; cell scenes in logs/<run>/scale/<tip>_mu<mu>/ (a fresh cache per tip and
          friction, the plant in each file name), base and design scenes in logs/<run>/gen/; rows in --out-dir
  hom     HOM on the deployed real_v1 hands D1-D8 (tip_codesign_rig hom, its scenes under logs/<run>/rig/)

Every turn row carries rigid_share: of the normal force that the fingers press on the object, summed over the rollout's
control samples, the share through finger geoms other than the tip's pads (on cad_v2 the DIP link, cassette and cap
spheres, on real_v1 the distal capsules). CLAUDE.md asks that pads, not rigid parts, carry the contact on the v4b hand.

    SCALE_HAND=cad_v2 HOS_PLANT=servo .venv/bin/python scripts/fingertip/fingertip_shape_rerun_rig.py reach
    SCALE_HAND=cad_v2 HOS_PLANT=servo .venv/bin/python scripts/fingertip/fingertip_shape_rerun_rig.py scale --tip tcd_cap6_u0.1 \\
        --mu 0.7 --plant kp0.72_kv0_fr0.18_fl0_dp0.0144 --what hold,turn --seeds 0 1 2 --out-dir <dir>
    PY=logs/20261001-hom_contact/venv/bin/python
    $PY scripts/fingertip/fingertip_shape_rerun_rig.py hom --tip tcd_cap6_u0.1 --mu 0.7 --seeds 0 1 2 3 4 --out <rows.jsonl>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path
import fingertip_shape_rerun_common as C  # noqa: E402

SCRIPT = "scripts/fingertip/fingertip_shape_rerun_rig.py"
T_OF_CELL = {c: float(c.split(":")[2]) for c in C.CELLS}
STUDY = f"{C.DATE}-fingertip_shape_rerun"


def rigid_hook(H):
    """hom_turn3.MjPlant with each sample's normal force split into pads and other finger geoms (rigid_share)."""
    cls = H.MjPlant
    if getattr(cls, "_rigid_hook", False):
        return
    init, reset = cls.__init__, cls.reset

    def __init__(self, path, meta):
        init(self, path, meta)
        m = self.m
        self.pad_geoms = {i for i in range(m.ngeom) if any(m.geom(i).name.startswith(f"{f}_pad") for f in H.FINGERS)}
        self.F_pad = self.F_rigid = 0.0
        self.F_body = {}
        cls.last = self

    def reset_(self, seed):
        reset(self, seed)
        self.F_pad = self.F_rigid = 0.0
        self.F_body = {}

    def state(self):
        m, d = self.m, self.d
        F = {f: 0.0 for f in H.FINGERS}
        n = {f: 0 for f in H.FINGERS}
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom[0]), int(c.geom[1])
            if (g1 in self.tool_geoms) == (g2 in self.tool_geoms):
                continue
            g = g2 if g1 in self.tool_geoms else g1
            f = self.finger_of.get(g)
            if f is None:
                continue
            self.mj.mj_contactForce(m, d, i, self.f6)
            if self.f6[0] > 1e-6:
                F[f] += float(self.f6[0])
                n[f] += 1
                if g in self.pad_geoms:
                    self.F_pad += float(self.f6[0])
                else:
                    self.F_rigid += float(self.f6[0])
                    b = m.geom(g).name.rsplit("_s", 1)[0] or m.body(m.geom_bodyid[g]).name
                    self.F_body[b] = self.F_body.get(b, 0.0) + float(self.f6[0])
        return {"t": float(d.time), "q": d.qpos[self.qadr].copy(), "qd": d.qvel[self.vadr].copy(),
                "p": d.qpos[self.qa:self.qa + 3].copy(), "quat": d.qpos[self.qa + 3:self.qa + 7].copy(),
                "R": d.xmat[self.tb].reshape(3, 3).copy(), "F": F, "n": n}

    cls.__init__, cls.reset, cls.state, cls._rigid_hook, cls.last = __init__, reset_, state, True, None


def rigid_share(H):
    """(rigid share of the last rollout, its rigid force by part: geom name less its sphere index, or the body's name)."""
    p = getattr(H.MjPlant, "last", None)
    if p is None:
        return None, None
    tot = p.F_pad + p.F_rigid
    by = {k: round(v / tot, 4) for k, v in sorted(p.F_body.items(), key=lambda kv: -kv[1])[:4]} if tot > 0 else {}
    return (round(p.F_rigid / tot, 5) if tot > 0 else None), by


def scale_env():
    """hand_object_scale_sim on cad_v2 with this run's folders: kinematics (reach.json), generated scenes."""
    if os.environ.get("SCALE_HAND") != "cad_v2":
        raise SystemExit("set SCALE_HAND=cad_v2")
    if os.environ.get("HOS_PLANT") != "servo":
        raise SystemExit("set HOS_PLANT=servo (the SCS0009)")
    import hand_object_scale_kin as K
    import hand_object_scale_sim as S
    K.OUT_DIR = S.OUT_DIR = C.D / "kin"
    S.GEN = C.LOGS / "gen"
    return K, S


def cmd_reach(a):
    import numpy as np
    K, _ = scale_env()
    lim = K.limits(False)
    fam = [r for r in K.family("diag") if r["t"] in set(T_OF_CELL.values())]
    out = {"meta": {"date": "2026-11-04", "script": f"{SCRIPT} reach (hand_object_scale_kin.grasp_scan, summarise)",
                    "hand": "cad_v2", "limits": "servo calibrated range meet scene ROM",
                    "limits_deg": {f: np.round(np.degrees(lim[f]), 2).tolist() for f in K.FINGERS},
                    "cyl_length_mm": 100.0, "spreads_mm": [s * 1000 for s in K.cylinder_spreads(0.100)]},
           "families": {"diag": fam}, "cells": []}
    t0 = time.time()
    for spec in C.CELLS:
        shape, dd, t = spec.split(":")
        lay = next(r for r in fam if r["t"] == float(t))
        xs, ys = lay["x_sep"] / 1000.0, lay["y_sep"] / 1000.0
        rows = K.grasp_scan(xs, ys, shape, float(dd) / 1000.0, lim, K.cylinder_spreads(0.100), length=0.100)
        cell = K.summarise(rows, xs, ys, shape, float(dd) / 1000.0)
        cell["tag"], cell["palm_radius_mm"] = lay["tag"], lay["palm_radius"]
        out["cells"].append(cell)
        print(f"{spec}: layout {lay['tag']} feasible {cell['feasible']} ({cell['n_feasible']} grasps)"
              + (f" fail {cell.get('fail')}" if not cell["feasible"] else ""), flush=True)
    out["meta"]["seconds"] = round(time.time() - t0, 1)
    p = C.D / "kin" / "reach.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, separators=(",", ":")))
    print("->", p.relative_to(ROOT), flush=True)


def cmd_prebuild(a):
    """The cells' base, design and plant scenes in logs/<run>/gen/, written once before the workers start (design_scene
    and base_scene write their files in place, so two workers that build one cell at once could read a partial file)."""
    _, S = scale_env()
    for spec in C.CELLS:
        shape, dd, t = spec.split(":")
        cell = next(c for c, sh, d_ in S.cell_list("diag") if sh == shape and d_ == float(dd) and c.get("t") == float(t))
        design = S.design_scene(cell["x_sep_mm"], cell["y_sep_mm"], shape, float(dd))
        for pl in sorted({p for _, p in C.CONDS}):
            S.PLANT = pl
            print(spec, S.plant_scene(design).relative_to(ROOT), flush=True)


def cmd_scale(a):
    K, S = scale_env()
    import hom_turn3 as H
    rigid_hook(H)
    S.TIP = a.tip
    S.MU = a.mu
    S.SCENES = C.LOGS / "scale" / f"{a.tip}_mu{a.mu:g}"
    orig = S._append
    extra = {"tip": a.tip, "mu": a.mu, "E_Pa": C.E_PA, "pad_tr_s": C.TR_S, "pad_builder": "v2", "study": STUDY,
             "script_wrapper": SCRIPT}

    def append(path, row):
        row.update(extra)
        if "turn_end_deg" in row:
            row["rigid_share"], row["rigid_by_part"] = rigid_share(H)
        orig(path, row)
    S._append = append
    ns = argparse.Namespace(cells="diag", what=a.what, model="pads", seeds=a.seeds, set=a.set, shard="0/1",
                            plant=a.plant, out_dir=str(a.out_dir), only=a.only or C.CELLS, traces=False, builder=None)
    return S.cmd_run(ns)


def cmd_hom(a):
    import hom_turn3 as H
    import pad_usability_backends as PUB
    import tip_codesign_rig as R
    rigid_hook(H)
    R.SCN = C.LOGS / "rig" / "scenes"
    R.SCRIPT = f"{SCRIPT} (tip_codesign_rig)"
    extra = R._row_extra
    R._row_extra = lambda ns: dict(extra(ns), study=STUDY)
    orig = PUB.append_row

    def append(path, row):
        if "turn_end_deg" in row:
            row["rigid_share"], row["rigid_by_part"] = rigid_share(H)
        orig(path, row)
    PUB.append_row = append
    ns = argparse.Namespace(hands=a.hands, seeds=a.seeds, E=C.E_PA, pf=None, tr=C.TR_S, tag=None, h="local",
                            K="each", mesh_eps=None, out=a.out, tip=a.tip, sims=["mujoco"], plant=a.plant, mu=a.mu,
                            set=a.set, ir=100.0)
    return R.cmd_hom(ns)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("reach")
    sub.add_parser("prebuild")
    sc = sub.add_parser("scale")
    sc.add_argument("--tip", required=True)
    sc.add_argument("--mu", type=float, default=0.7)
    sc.add_argument("--plant", default=C.SERVO)
    sc.add_argument("--what", default="hold,turn")
    sc.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    sc.add_argument("--only", nargs="*", default=None)
    sc.add_argument("--set", nargs="*", default=["kp_model=0.72", "post_off=1"])
    sc.add_argument("--out-dir", type=Path, required=True)
    h = sub.add_parser("hom")
    h.add_argument("--tip", required=True)
    h.add_argument("--hands", nargs="+", default=[f"D{i}" for i in range(1, 9)])
    h.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    h.add_argument("--mu", default="0.7")
    h.add_argument("--plant", default=C.SERVO)
    h.add_argument("--set", nargs="*", default=["kp_model=0.72"])
    h.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    {"reach": cmd_reach, "prebuild": cmd_prebuild, "scale": cmd_scale, "hom": cmd_hom}[a.cmd](a)


if __name__ == "__main__":
    main()
