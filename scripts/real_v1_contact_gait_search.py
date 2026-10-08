#!/usr/bin/env python3
"""Search short release-and-recontact plans on the deployed D1 hand.

Each plan has three yaw targets in three time bins, followed by two finger
opening amplitudes and two closing biases. CPU finite differences test whether
local trajectory gradients help after a contact sequence has been sampled.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import real_v1_diffmjx_gate as gate  # noqa: E402

DT_MS = 1
HORIZON_MS = 400
PHASE_EDGES_MS = (120, 240, 400)
OPEN_UNTIL_MS = 180
N_PARAM = 13
FINGERS = gate.rb.FINGERS
YAW_SCALE_RAD = 0.35
CLOSE_SCALE_RAD = 0.30


class Scene:
    def __init__(self, tip: str, contact_model: str, seed: int = 1,
                 forced_release: bool = False, pad_spacing_mm: float = 1.0):
        import mujoco

        self.mujoco = mujoco
        self.scene, self.meta = gate.make_scene("D1", "bed", 100.0, no_post=True,
                                               tip=tip, contact_model=contact_model,
                                               pad_spacing_m=pad_spacing_mm*0.001)
        supported, _ = gate.make_scene("D1", "bed", 100.0,
                                       tip=tip, contact_model=contact_model,
                                       pad_spacing_m=pad_spacing_mm*0.001)
        self.model = mujoco.MjModel.from_xml_path(str(self.scene))
        self.held = gate.clone_data(self.model, gate.grip_state(
            mujoco.MjModel.from_xml_path(str(supported)), self.meta, "D1", seed))
        body = self.model.body(gate.rb.OBJ).id
        self.qadr = int(self.model.jnt_qposadr[self.model.body_jntadr[body]])
        self.yaw_ids = np.array([self.model.actuator(f"a_{finger}_yaw").id
                                 for finger in FINGERS])
        self.flex_ids = {finger: np.array([self.model.actuator(f"a_{finger}_{joint}").id
                                           for joint in ("mcp", "pip")])
                         for finger in ("index", "middle")}
        self.ctrl0 = self.held.ctrl.copy()
        self.z0 = float(self.held.qpos[self.qadr + 2])
        self.cos0 = self.vertical_cos(self.held)
        self.low = self.model.actuator_ctrlrange[:, 0].copy()
        self.high = self.model.actuator_ctrlrange[:, 1].copy()
        self.calls = 0
        self.forced_release = forced_release
        self.pad_spacing_mm = pad_spacing_mm

    def vertical_cos(self, data):
        quat = data.qpos[self.qadr + 3:self.qadr + 7]
        return float(1 - 2 * (quat[1] ** 2 + quat[2] ** 2))

    def evaluate(self, x, trace: bool = False):
        x = np.asarray(x, float)
        assert x.shape == (N_PARAM,)
        if self.forced_release:
            x = x.copy()
            x[9] = 1.0
        self.calls += 1
        data = gate.clone_data(self.model, self.held)
        phase = -1
        min_z = self.z0
        contact_trace = []
        for k in range(HORIZON_MS):
            phase_next = int(k >= PHASE_EDGES_MS[0]) + int(k >= PHASE_EDGES_MS[1])
            if phase_next != phase or k == OPEN_UNTIL_MS:
                phase = phase_next
                data.ctrl[:] = self.ctrl0
                data.ctrl[self.yaw_ids] = np.clip(
                    self.ctrl0[self.yaw_ids] + YAW_SCALE_RAD * x[3*phase:3*phase+3],
                    self.low[self.yaw_ids], self.high[self.yaw_ids])
                for i, finger in enumerate(("index", "middle")):
                    ids = self.flex_ids[finger]
                    if k < OPEN_UNTIL_MS:
                        amplitude = 0.5 * (x[9+i] + 1.0)
                        data.ctrl[ids] = self.ctrl0[ids] + amplitude * (self.low[ids] - self.ctrl0[ids])
                    else:
                        data.ctrl[ids] = np.clip(self.ctrl0[ids] + CLOSE_SCALE_RAD*x[11+i],
                                                 self.low[ids], self.high[ids])
            self.mujoco.mj_step(self.model, data)
            min_z = min(min_z, float(data.qpos[self.qadr + 2]))
            if trace and (k+1) % 10 == 0:
                contact = gate.contact_summary(self.model, data)
                contact_trace.append({"t_s": round((k+1)*0.001, 3),
                                      "vertical_cos": self.vertical_cos(data),
                                      "tool_z_m": float(data.qpos[self.qadr+2]),
                                      "finger_normal_force_N": contact["finger_normal_force_N"]})
        contact = gate.contact_summary(self.model, data)
        forces = contact["finger_normal_force_N"]
        final_cos = self.vertical_cos(data)
        height_deficit = max(0.0, self.z0 - min_z - 0.003)
        force_deficit = sum(max(0.0, 0.5 - forces[finger]) for finger in FINGERS)
        score = final_cos - self.cos0 - 70.0*height_deficit - 0.7*force_deficit
        row = {"score": float(score), "vertical_cos": final_cos,
               "cos_gain": float(final_cos-self.cos0),
               "tool_z_m": float(data.qpos[self.qadr+2]), "min_tool_z_m": min_z,
               "finger_normal_force_N": forces,
               "held_final": bool(min_z >= self.z0-0.008 and min(forces.values()) >= 0.5),
               "x": x.tolist()}
        if trace:
            row["contact_trace"] = contact_trace
            row["index_release_ms"] = int(sum(item["finger_normal_force_N"]["index"] < 0.05
                                              for item in contact_trace)*10)
            row["middle_release_ms"] = int(sum(item["finger_normal_force_N"]["middle"] < 0.05
                                               for item in contact_trace)*10)
            row["index_recontact"] = bool(row["index_release_ms"] >= 20 and
                                          forces["index"] >= 0.5)
            row["middle_recontact"] = bool(row["middle_release_ms"] >= 20 and
                                           forces["middle"] >= 0.5)
        return row


def hold_vector(forced_release: bool = False):
    x = np.zeros(N_PARAM)
    x[9:11] = -1.0
    if forced_release:
        x[9] = 1.0
    return x


def optimize_gradient(scene: Scene, start, budget: int):
    t0 = time.perf_counter()
    n0 = scene.calls
    history = []
    best = {"score": -float("inf"), "x": np.asarray(start, float).tolist()}

    class BudgetReached(Exception):
        pass

    def loss(x):
        if scene.calls-n0 >= budget:
            raise BudgetReached
        row = scene.evaluate(x)
        history.append(row["score"])
        if row["score"] > best["score"]:
            best.update(score=row["score"], x=row["x"])
        return -row["score"]

    try:
        bounds = [(-1.0, 1.0)]*N_PARAM
        if scene.forced_release:
            bounds[9] = (1.0, 1.0)
        result = minimize(loss, np.asarray(start, float), method="L-BFGS-B",
                          jac="2-point", bounds=bounds,
                          options={"maxfun": budget, "maxiter": budget,
                                   "ftol": 1e-9})
        message = str(result.message)
    except BudgetReached:
        message = "evaluation budget reached"
    return {"x": best["x"], "score": float(best["score"]),
            "evals": scene.calls-n0, "wall_s": time.perf_counter()-t0,
            "all_scores": history, "optimizer_message": message}


def optimize_cem(scene: Scene, budget: int, seed: int):
    rng = np.random.default_rng(seed)
    mean = hold_vector(scene.forced_release)
    std = np.full(N_PARAM, 0.62)
    std[9:11] = 0.95
    best = None
    all_scores = []
    t0 = time.perf_counter()
    n0 = scene.calls
    popsize = 64
    while scene.calls-n0 < budget:
        count = min(popsize, budget-(scene.calls-n0))
        X = np.clip(rng.normal(mean, std, size=(count,N_PARAM)), -1.0, 1.0)
        if scene.forced_release:
            X[:,9] = 1.0
        if best is not None:
            X[0] = best["x"]
        rows = [scene.evaluate(x) for x in X]
        scores = np.array([row["score"] for row in rows])
        all_scores.extend(scores.tolist())
        top = np.argsort(scores)[-max(2, count//8):]
        if best is None or scores[top[-1]] > best["score"]:
            best = rows[top[-1]]
        mean = 0.5*mean + 0.5*X[top].mean(axis=0)
        std = np.maximum(0.12, 0.5*std + 0.5*X[top].std(axis=0))
    return {"x": best["x"], "score": best["score"],
            "evals": scene.calls-n0, "wall_s": time.perf_counter()-t0,
            "all_scores": all_scores}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tip", choices=("sphere", "tpu6"), default="tpu6")
    parser.add_argument("--contact-model", choices=("pt", "padsT"), default="padsT")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--budget", type=int, default=256)
    parser.add_argument("--forced-release", action="store_true")
    parser.add_argument("--pad-spacing-mm", type=float, default=1.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    scene = Scene(args.tip, args.contact_model, args.seed, args.forced_release,
                  args.pad_spacing_mm)
    hold = hold_vector(args.forced_release)
    hold_row = scene.evaluate(hold, trace=True)
    direct = optimize_gradient(scene, hold, args.budget)
    sampled = optimize_cem(scene, args.budget, 20261007)
    half = optimize_cem(scene, args.budget//2, 20261007)
    refined = optimize_gradient(scene, half["x"], args.budget//2)
    candidate_rows = {"hold": hold_row,
                      "direct_gradient": scene.evaluate(direct["x"], trace=True),
                      "cem": scene.evaluate(sampled["x"], trace=True),
                      "hybrid": scene.evaluate(refined["x"], trace=True),
                      "hybrid_seed": scene.evaluate(half["x"], trace=True)}
    output = {"scene": str(scene.scene.relative_to(ROOT)),
              "mujoco_version": scene.mujoco.__version__,
              "tip": args.tip, "contact_model": args.contact_model,
              "seed": args.seed, "horizon_ms": HORIZON_MS,
              "phase_edges_ms": PHASE_EDGES_MS, "open_until_ms": OPEN_UNTIL_MS,
              "parameter_count": N_PARAM, "search_budget_evals": args.budget,
              "forced_release":args.forced_release,
              "pad_spacing_mm":args.pad_spacing_mm,
              "initial_vertical_cos": scene.cos0, "initial_tool_z_m": scene.z0,
              "initial_force_N": gate.contact_summary(scene.model,scene.held)["finger_normal_force_N"],
              "score": "cos_gain - 70 max(0,z0-min_z-0.003 m) - 0.7 sum_f max(0,0.5 N-F_f)",
              "methods": {"direct_gradient": direct, "cem": sampled,
                          "hybrid_cem": half, "hybrid_gradient": refined},
              "candidates": candidate_rows}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")
    compact = {key: {k:row[k] for k in ("score","cos_gain","min_tool_z_m",
                                          "held_final","index_release_ms","index_recontact")}
               for key,row in candidate_rows.items()}
    print(json.dumps({"tip":args.tip,"contact_model":args.contact_model,
                      "mujoco_version":scene.mujoco.__version__,"candidates":compact,
                      "evals":{k:v["evals"] for k,v in output["methods"].items()},
                      "wall_s":{k:round(v["wall_s"],3) for k,v in output["methods"].items()}},
                     indent=2,allow_nan=False),flush=True)


if __name__ == "__main__":
    main()
