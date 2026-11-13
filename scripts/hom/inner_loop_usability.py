#!/usr/bin/env python3
"""Usability of an inner-loop controller on the SR2 hand: the forces, joint states and commands a controller needs for
the task, scored the same way for HOM and for a trained UHAS policy (scripts/hom/uhas_sr2.py) on the same worlds of the
inner-loop benchmark (owner, 2026-11-12: "can we codify/quantify" why UHAS is less usable).

Every world runs HOM through the approach, the closing, the squeeze and the lift. In the task phase (t >= 2.4 s) HOM or
the policy drives the servos. Each control period of the task phase is recorded: each finger's pad force and the force
through its rigid parts on the object (hom_gpu's F and F_link), the servo targets, the joint angles and the actuator
torques. Per rollout, over the task phase:
    link_mean_N    mean over the periods of the rigid-part force, summed over the fingers (N)
    link_max_N     the largest value of that sum (N)
    link_frac_1N   share of the periods in which that sum exceeds 1 N
    pad_share      the pads' force over the pads' and rigid parts' force, summed over the task
    grip_mean_N    mean over the periods of the total finger force on the object, summed over the fingers (N)
    pad_max_N      the largest force on one finger's pads (N)
    sat_frac       share of the joint-periods with the actuator torque at 95 % of its limit or more
    limit_frac     share of the periods with a finger joint within 1 deg of its range limit
    cmd_rate_dps   RMS rate of the servo targets over the joints and periods (deg/s)
    reversals_ps   sign changes of a servo target's rate, counted where the rate exceeds 1 deg/s, per joint per second
with the rows' J (hom_landscape.track_score), held, drift and tracking error.

    .venv/bin/python scripts/hom/inner_loop_usability.py --run results/uhas_sr2/<run> [--split test] [--seeds 0]
Writes docs/experiments/20261112-inner_loop_benchmark/rows/usability_<task>_<run>.jsonl, one row per world and
controller, fsynced per batch, and prints the medians over the held rollouts.
"""
from __future__ import annotations

import os

os.environ.setdefault("SCALE_HAND", "cad_v2")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path
import inner_loop_plant  # noqa: E402,F401  the talk's plant (SCS0009, 2.5 MPa pads), set before hom_gpu reads it
import hom_gpu as HG  # noqa: E402
import uhas_sr2 as UH  # noqa: E402
import uhas_sr2_cik as U  # noqa: E402

T = HG.T
ROWS = ROOT / "docs/experiments/20261112-inner_loop_benchmark/rows"
METRICS = ("link_mean_N", "link_max_N", "link_frac_1N", "pad_share", "grip_mean_N", "pad_max_N", "sat_frac",
           "limit_frac", "cmd_rate_dps", "reversals_ps")
KEEP = ("J", "held", "drift_max_mm", "track_err_rms_deg", "contact_all_frac")


def worlds_of(task, xys, seeds):
    scene = HG.Scene(T.SHAPE[task], UH.D_MM[task])
    out = []
    for key, mxy, setup in UH.grid_entries(task):
        if UH.xy_of(key) not in xys:
            continue
        bm, q_open = scene.prepare(mxy, setup)
        for sd in seeds:
            out.append({"key": key, "seed": sd, "setup": setup, "body_mounts": bm, "q_open": q_open,
                        "var": HG.variant_world(UH.VARIANT[task], sd)})
    return scene, out


class Limits:
    """The finger joints' ranges and the servos' torque limits of the batch's model, in the batch's joint order."""

    def __init__(self, B):
        m = B.m
        qadr = np.asarray(B.qadr).reshape(-1)
        jid = [int(np.flatnonzero(m.jnt_qposadr == a)[0]) for a in qadr]
        self.lo = torch.tensor(m.jnt_range[jid, 0], device="cuda", dtype=torch.float32)
        self.hi = torch.tensor(m.jnt_range[jid, 1], device="cuda", dtype=torch.float32)
        self.limited = torch.tensor(m.jnt_limited[jid].astype(bool), device="cuda")
        act = np.asarray(B.act).reshape(-1)
        self.tau = torch.tensor(np.abs(m.actuator_forcerange[act]).max(1), device="cuda", dtype=torch.float32)
        self.tau_limited = torch.tensor(m.actuator_forcelimited[act].astype(bool), device="cuda")


def targets(B):
    import warp as wp
    if HG.DEADBAND > 0:
        return wp.to_torch(B.target).float()
    return B.vw["ctrl"][:, B.act_t].float()


def metrics(B, k0, Q, C, A, lim, dt):
    """The usability metrics of every world over the task phase (periods k0 on): Q, C, A are the joint angles, servo
    targets and actuator torques after each period, (n, W, 9)."""
    ft = B.rec["ft"][k0:].float()
    fl = (B.rec["fo"][k0:].float() - ft).clamp_min(0.0)
    link = fl.sum(-1)                                                      # (n, W)
    out = {"link_mean_N": link.mean(0), "link_max_N": link.max(0).values, "link_frac_1N": (link > 1.0).float().mean(0),
           "pad_share": ft.sum((0, 2)) / (ft.sum((0, 2)) + fl.sum((0, 2))).clamp_min(1e-9),
           "grip_mean_N": B.rec["fo"][k0:].float().sum(-1).mean(0), "pad_max_N": ft.amax((0, 2))}
    at = (A.abs() >= 0.95 * lim.tau) & lim.tau_limited
    out["sat_frac"] = at.float().mean((0, 2))
    tol = math.radians(1.0)
    near = ((Q - lim.lo).abs() < tol) | ((lim.hi - Q).abs() < tol)
    out["limit_frac"] = (near & lim.limited).any(-1).float().mean(0)
    rate = torch.diff(C, dim=0) / dt                                       # (n-1, W, 9) rad/s
    out["cmd_rate_dps"] = torch.rad2deg(rate.pow(2).mean((0, 2)).sqrt())
    s = torch.sign(rate) * (rate.abs() > math.radians(1.0))
    flips = ((s[1:] * s[:-1]) < 0).float().sum(0)                          # (W, 9)
    out["reversals_ps"] = flips.mean(-1) / (rate.shape[0] * dt)
    return {k: v.cpu().numpy() for k, v in out.items()}


def run(task, scene, worlds, W, policy=None, label="hom"):
    """Rows with the usability metrics: HOM throughout (policy None) or the policy in the task phase."""
    import hom_landscape as L
    from tensordict import TensorDict
    rows, B = [], None
    for c0 in range(0, len(worlds), W):
        chunk = worlds[c0:c0 + W]
        pad = chunk + [chunk[-1]] * (W - len(chunk))
        if B is None:
            B = UH.UBatch(scene, task, pad, T.DT_C, gen_seed=0)
            F = U.Fingers(B.m, "cuda")
            lim = Limits(B)
        t0 = time.time()
        B.begin(pad, record=True, hom=True)
        if float(B.cal_q.abs().max()) > 0:          # set by the controller's reset in begin()
            raise RuntimeError("calibration offsets in the plant: the policy's targets bypass them")
        k0 = B.k
        if policy is not None:
            task_ref = UH.Task(B)
            mounts = torch.tensor(np.array([[wd_["body_mounts"][f] for f in HG.FINGERS] for wd_ in pad]),
                                  dtype=torch.float32, device="cuda")
            C = U.SphereController(F, mounts)
            tr = UH.Tracker(W, "cuda")
            tr.reset(torch.arange(W, device="cuda"), B, C, F, mounts, B.n_out[:, 0].float())
        Q, Cm, A = [], [], []
        import warp as wp
        with torch.inference_mode():
            while not B.done():
                if policy is None:
                    B.tick(None)
                else:
                    pol, cri = tr.observe(task_ref, B, C, F, mounts)
                    a = policy(TensorDict({"policy": pol, "critic": cri}, batch_size=[W], device="cuda")).clamp(-1, 1)
                    B.tick(C.solve(a))
                    tr.last_a = a
                    tr.tau += 1
                    tr.advance(task_ref, scene.r)
                Q.append(B.vw["qpos"][:, B.qadr_t].reshape(W, 9).float().clone())
                Cm.append(targets(B).reshape(W, 9).clone())
                A.append(wp.to_torch(B.wd.actuator_force)[:, B.act_t].float().clone())
        res = B.finish()[:len(chunk)]
        M = metrics(B, k0, torch.stack(Q), torch.stack(Cm), torch.stack(A), lim, B.prm.dt)
        for i, (wd_, r) in enumerate(zip(chunk, res)):
            x, y = UH.xy_of(wd_["key"])
            row = {"controller": label, "x_sep": x, "y_sep": y, "grasp": int(wd_["key"].get("grasp", 0)),
                   "seed": wd_["seed"], "J": L.track_score(task, r), **inner_loop_plant.describe()}
            row.update({k: r.get(k) for k in KEEP if k != "J"})
            row.update({k: round(float(M[k][i]), 4) for k in METRICS})
            rows.append(row)
        print(f"{label}: {len(rows)} of {len(worlds)} rollouts; batch {time.time() - t0:.0f} s", flush=True)
    return rows


def summary(rows):
    out = {}
    for lab in sorted({r["controller"] for r in rows}):
        R = [r for r in rows if r["controller"] == lab]
        H = [r for r in R if r["held"]]
        out[lab] = {"rollouts": len(R), "held": round(len(H) / len(R), 3), "J_mean": round(float(np.mean([r["J"] for r in R])), 3)}
        for k in METRICS + ("drift_max_mm",):
            v = np.array([r[k] for r in H if r.get(k) is not None], float)
            out[lab][k] = [round(float(np.median(v)), 3), round(float(np.percentile(v, 90)), 3)] if len(v) else None
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--worlds", type=int, default=2048)
    a = ap.parse_args()
    run_dir = Path(a.run)
    sp = json.loads((run_dir / "split.json").read_text())
    task = sp["task"]
    scene, worlds = worlds_of(task, set(map(tuple, sp[a.split])), a.seeds)
    actor, ckpt = UH.load_policy(run_dir)
    out = ROWS / f"usability_{task}_{run_dir.name}.jsonl"
    rows = []
    for label, pol in (("hom", None), ("uhas", actor)):
        R = run(task, scene, worlds, a.worlds, pol, label)
        for r in R:
            r["checkpoint"] = Path(ckpt).name if pol is not None else None
        rows += R
        with open(out, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r, default=float) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
    S = summary(rows)
    print(json.dumps(S, indent=1))
    print(f"-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
