#!/usr/bin/env python3
"""Pinned iCEM on the 1 mm pad plant with a cheaper rollout model of the fingertip (U4 of the 2026-10-14 pad-usability
study; plan in docs/notes/20261014-contact_model_assessment_plan.md).

The plant of every run is the hand-object scale study's servo-limited real_v1 plant with the 1 mm sphere pads
(hand_object_scale_mpc.setup). The planner is pinned iCEM (hand_object_scale_planners config icem_pin: 64 rollouts of
0.5 s every 20 ms in two iterations, three knots, the first knot pinned to the command in force, the three-pad cost).
Only the model the planner rolls out changes:
  icem_pin       the plant's own pad model (the reference planner of 2026-10-13)
  icem_pin_pt    the cell's point-contact scene: the TPU block as one convex mesh, condim 3
  icem_pin_pt4   the same with condim 4, priority 1 and friction (1, 0.0028, 1e-4) on the three {f}_tipgeom geoms
                 (torsional coefficient 2.8 mm, the pads' twist-onset arm at the cells' grips; priority 1 because every
                 cell-scene geom has torsional friction 0.005 and MuJoCo takes the larger at equal priority)
The planner's model is compiled from that scene with hand_object_scale_mpc.sensor_model's per-finger contact sensors and
the plant's palm pose, and must have the plant's nq, nv and FULLPHYSICS state size, since every replanning starts from the
plant's state. The post is removed from the planner's model as in hand_object_scale_mpc.Planner. No module of the
scale study is edited: this script installs its Planner subclass as hand_object_scale_mpc.Planner, wraps
hand_object_scale_mpc.setup to keep the cell's scene paths on the plant, registers the two configurations in
hand_object_scale_planners.CONFIGS and runs that module's run or film subcommand.

    PY=.venv/bin/python
    $PY scripts/pad_usability_planner_model.py run --configs icem_pin_pt icem_pin_pt4 \\
        --cells cylinder:25:0.125 cylinder:16:0.625 cylinder:10:0.75 cylinder:16:-0.375 --seeds 0 1 2 3 --nthread 3 \\
        --out docs/experiments/20261014-pad_usability/planner/turn.jsonl
    MUJOCO_GL=egl $PY scripts/pad_usability_planner_model.py film --configs icem_pin icem_pin_pt4 icem_pin_pt \\
        --cells cylinder:25:0.125 cylinder:16:-0.375 --seed 0 --cols 3 --stem planner_models \\
        --out-dir docs/experiments/20261014-pad_usability/planner/media
Rows (fsynced, resumable by config, cell, seed and variant): docs/experiments/20261014-pad_usability/planner/turn.jsonl;
the pads planner's seeds 0 and 1 are also the icem_pin rows of docs/experiments/20261013-scale_turn_controllers/planners/.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import hand_object_scale_mpc as M  # noqa: E402
import hand_object_scale_planners as HP  # noqa: E402
import hand_object_scale_sim as S  # noqa: E402

PT4_FRICTION = (1.0, 0.0028, 1e-4)
HP.CONFIGS["icem_pin_pt"] = dict(HP.CONFIGS["icem_pin"], plan_tip="pt")
HP.CONFIGS["icem_pin_pt4"] = dict(HP.CONFIGS["icem_pin"], plan_tip="pt4")
HP.TITLES.update(icem_pin="Pads planner", icem_pin_pt="Point-contact planner", icem_pin_pt4="Condim-4 planner")

_setup = M.setup


def setup(cell, shape, dd, ci):
    """hand_object_scale_mpc.setup, with the cell's scene paths kept on the plant for the planner's model."""
    plant, meta, geo, sc, gg = _setup(cell, shape, dd, ci)
    plant.scene_paths = {k: str(v[0]) for k, v in sc.items()}
    return plant, meta, geo, sc, gg


class ModelPlanner(M.Planner):
    """hand_object_scale_mpc.Planner whose rollout model comes from the scene of prm['plan_tip']."""

    def __init__(self, plant, prm, nthread, rng):
        tip = prm.get("plan_tip", "pads")
        if tip == "pads":
            super().__init__(plant, prm, nthread, rng)
            return
        if prm.get("w_contact", 0.0) <= 0 or prm.get("pad_filter", 0.0) > 0:
            raise ValueError("plan_tip needs the contact-sensor cost and no pad filter")
        xml0 = plant.xml_path
        plant.xml_path = plant.scene_paths["pt"]          # sensor_model compiles plant.xml_path
        try:
            super().__init__(plant, prm, nthread, rng)
        finally:
            plant.xml_path = xml0
        mj, m = self.mj, self.m
        if tip == "pt4":
            for f in S.FINGERS:
                g = m.geom(f"{f}_tipgeom").id
                m.geom_condim[g] = 4
                m.geom_priority[g] = 1
                m.geom_friction[g] = PT4_FRICTION
        elif tip != "pt":
            raise ValueError(f"plan_tip {tip}")
        if (m.nq, m.nv, m.na) != (plant.m.nq, plant.m.nv, plant.m.na) or \
                mj.mj_stateSize(m, self.spec) != mj.mj_stateSize(plant.m, self.spec):
            raise RuntimeError("the planner's model and the plant differ in state size")
        if sum(1 for g in range(m.ngeom) if "_pad" in (m.geom(g).name or "")):
            raise RuntimeError("the point-contact scene has pad geoms")


M.setup = setup
M.Planner = ModelPlanner

if __name__ == "__main__":
    sys.exit(HP.main())
