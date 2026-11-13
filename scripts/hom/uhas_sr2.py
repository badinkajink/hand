#!/usr/bin/env python3
"""UHAS (Casas et al., arXiv 2607.03570, 2026) trained and evaluated on the SR2 hand's layouts in MuJoCo-Warp with the
sphere-pad fingertips: the learned inner-loop controller of the inner-loop benchmark (owner, 2026-11-12: "If we have to
train our own uhas, that's fine"; "We can let the other methods have the sphere packed model and warp").

Environment. The worlds of hom_gpu.Batch: the hand-design talk's scenes, plant (SCS0009, 0.72 N m/rad) and 2.5 MPa pads,
one layout, grasp and seed per world. HOM's controller drives the approach, the closing, the squeeze and the lift (the
first 2.0 s), as in the HOM grids. From the start of the task phase the policy drives the fingers through UHAS's sphere
controller (uhas_sr2_cik.py) at HOM's control period of 10 ms, against the task's commanded rotation
(hom_gpu.Batch.schedule). One batch is built once and reset in place for every episode.

Observation (UHAS appendix; in the sphere's frame, lengths in sphere radii): the object's position, orientation, linear and
angular velocity; the goal orientation (the start orientation turned by the commanded rotation so far) and the
orientation error; the previous action; the middle point and the tip of each finger, positions and velocities; and the
commanded angular velocity, since the goal moves. The policy sees the simulator's state without noise. The critic also
sees each pad's normal force and the share of the task elapsed.

Reward (UHAS appendix, multi_env_cfg.py): -10 times the object's distance from its reference position (m), plus
1 / (orientation error + 0.1 rad), minus 0.016 times the lateral joints' and 0.004 times the encompassing joints' squared
deviation from the undeformed sphere's posture, and -100 when the object drops 15 mm below its start (J's rule). A drop
ends the world's episode. UHAS's bonus of 250 per goal reached within 0.1 rad has no counterpart in a continuous
reference and is left out. J's contact factor is added as +2 per period with all three pads above 0.05 N.

PPO (UHAS appendix, table of hyperparameters; RSL-RL 5.0.1): 16 steps per world per iteration, empirical observation
normalization, actor and critic of [512, 512, 256, 128] ELU units, initial action std 1.0, learning rate 5e-4 with the
adaptive schedule (desired KL 0.016), clip 0.2, entropy 0.005, 5 epochs, 4 mini-batches, value coefficient 1.0 with the
clipped value loss, gamma 0.99, lambda 0.95, gradient norm at most 1.0.

Episodes. The worlds of a batch start and end together. A world whose object dropped stays at its terminal state,
reporting a zero observation, zero reward and done until the batch ends. Each episode draws its layouts and grasps from
the training layouts and seeds from 100 on; evaluation uses seeds 0-2, as the HOM grids.

    .venv/bin/python scripts/hom/uhas_sr2.py smoke --task twist --worlds 64
    .venv/bin/python scripts/hom/uhas_sr2.py train --task twist --worlds 2048 --minutes 120
    .venv/bin/python scripts/hom/uhas_sr2.py train --task twist --only 105 70 --minutes 30     # one layout
    .venv/bin/python scripts/hom/uhas_sr2.py train --task twist --start open --minutes 120     # its own grasp
With --start open the policy acts from time 0, the fingers open around the object on its post, as UHAS's policy acts
from the start of its episode; each world's post goes away at the lift (1.6 s).
    .venv/bin/python scripts/hom/uhas_sr2.py eval --task twist --run <run dir> [--split test]
Runs: results/uhas_sr2/<date-time>_<task>_s<seed>/ (checkpoints, tensorboard, episodes.jsonl, split.json); evaluation rows:
docs/experiments/20261112-inner_loop_benchmark/rows/uhas_<task>_<run>.jsonl.
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
import hom_gpu as HG  # noqa: E402
import uhas_sr2_cik as U  # noqa: E402
import warp as wp  # noqa: E402
from sr2hand.geometry.rotations import mat_to_quat  # noqa: E402
from sr2hand.rl.math import quat_conj, quat_from_axis_angle, quat_mul  # noqa: E402

T = HG.T
RUNS = ROOT / "results/uhas_sr2"
OUT = ROOT / "docs/experiments/20261112-inner_loop_benchmark"
D_MM = {"twist": 40.0, "roll": 40.0, "pinpush": 25.0, "press": 25.0}
VARIANT = {"twist": "base", "roll": "base", "pinpush": "hold", "press": "hold"}
REW = {"dist": -10.0, "rot": 1.0, "rot_eps": 0.1, "lat": -0.016, "rad": -0.004, "fall": -100.0, "contact": 2.0,
       "fall_dist": 0.2, "drop_m": 0.015, "f_on": 0.05}
# train --limits (owner, 2026-11-12: "run uhas training with joint and actuator limits built into training"): per period,
# -0.2 sum_j clip((servo demand_j / torque limit - 0.8) / 0.2, 0, 3), the demand kp |target - q| and 0.8 the SCS0009's
# overload threshold; -0.1 sum_j clip((3 deg - distance to the joint's range limit) / 3 deg, 0, 1); -0.04 |a - a_prev|^2.
# The period's reward is then clipped at zero before the drop penalty (legged_gym's only_positive_rewards): at -0.5 for the
# torque term a 64-world smoke run's untrained policy paid 8 per period, more than a held sphere earns, so dropping the
# sphere would have paid. inner_loop_mpc.py's planners have the same terms in their cost.
REW_LIMITS = {"tau": -0.2, "tau_on": 0.8, "q": -0.1, "q_band_deg": 3.0, "rate": -0.04}
LIMITS = False
PPO_CFG = {
    "num_steps_per_env": 16, "save_interval": 50, "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
    "logger": "tensorboard",
    "algorithm": {"class_name": "PPO", "num_learning_epochs": 5, "num_mini_batches": 4, "clip_param": 0.2,
                  "gamma": 0.99, "lam": 0.95, "value_loss_coef": 1.0, "entropy_coef": 0.005, "learning_rate": 5e-4,
                  "max_grad_norm": 1.0, "use_clipped_value_loss": True, "schedule": "adaptive", "desired_kl": 0.016,
                  "rnd_cfg": None, "symmetry_cfg": None},
    "actor": {"class_name": "MLPModel", "hidden_dims": [512, 512, 256, 128], "activation": "elu",
              "obs_normalization": True,
              "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"}},
    "critic": {"class_name": "MLPModel", "hidden_dims": [512, 512, 256, 128], "activation": "elu",
               "obs_normalization": True}}


# ------------------------------------------------------------------------------------------------ layouts
def xy_of(key):
    """(x_sep, y_sep) mm of a grid row's six coordinates (hom_gpu.sym6)."""
    d = key["d"]
    return (round(float(d[2] - d[0]), 3), round(float(d[3] - d[5]), 3))


def grid_entries(task):
    """Every grasp of every lattice layout that reaches the task's pose: [(key, mounts xy, setup)] (hom_gpu.layout_list
    on the grids' grasp sets, the plant's finger clearance)."""
    return HG.layout_list(task, "grid2dff_size", 0, D_MM[task])


def split(task, seed=0, test_share=0.2, keep_best=5):
    """Training and test layouts: a fifth of the reached layouts held out at random, with the best five by J (the HOM
    grid's) always in the test set."""
    import hom_search_qd as Q
    look = Q.lookup(task, D_MM[task])[0]
    run = sorted(k for k, (_f, n) in look.items() if n > 0)
    rng = np.random.default_rng(seed)
    perm = [run[i] for i in rng.permutation(len(run))]
    best = sorted(run, key=lambda k: look[k][0], reverse=True)[:keep_best]
    test = set(perm[:int(round(test_share * len(run)))]) | set(best)
    return sorted(set(run) - test), sorted(test), {k: look[k][0] for k in run}


# ------------------------------------------------------------------------------------------------ the batch
class UBatch(HG.Batch):
    """hom_gpu.Batch reset in place for each episode, with HOM driving the phases before the task and a joint command
    from outside driving the task phase."""

    CAPTURE = False

    def __init__(self, scene, task, worlds, dt, gen_seed=0):
        super().__init__(scene, task, worlds, dt, gen_seed=gen_seed)
        self.mount_bid = {f: self.m.body(f"{f}_mount").id for f in HG.FINGERS}
        self.t_task = T.T_LIFT + 0.4
        self.first = True

    def begin(self, worlds, record=False, hom=True):
        """Reset every world to its layout, grasp and seed, and run HOM up to the task phase (hom=False: stay at the
        start, the fingers open and the object on its post)."""
        if not self.first:
            self.worlds = worlds
            bp = np.repeat(self.m.body_pos[None].astype(np.float32), self.W, 0)
            for w, wd_ in enumerate(worlds):
                for f in HG.FINGERS:
                    bp[w, self.mount_bid[f]] = wd_["body_mounts"][f]
            self.wm.body_pos.assign(bp)
            self._init_state()
            self._launch_forces()
            wp.synchronize()
        self.first = False
        self.reset_controller()
        self.sch = T.Schedule(self.task, worlds[0]["setup"], T.PRM)
        self.vw, self.mv = self.views(self.wd), self.views(self.wdm)
        self.ft_t, self.fo_t, self.cop_t = wp.to_torch(self.ftip), wp.to_torch(self.ftot), wp.to_torch(self.cops)
        self.clock, t = [], 0.0
        while t < self.sch.t_end - 1e-9:
            self.clock.append(t)
            t += self.prm.dt
        self.k, self.post_off, self.record = 0, False, record
        n = len(self.clock)
        th = self.torch
        if record:
            self.rec = {k: th.zeros((n, self.W, 3), device=self.dev) for k in ("pos", "w", "ft", "fo", "vr", "ax")}
            self.rec["cop"] = th.zeros((n, self.W, 3, 3), device=self.dev)
            self.phases, self.wrefs, self.times, self.thref = [], [], [], []
        while hom and self.clock[self.k] < self.t_task - 1e-9:
            self.tick(None)

    def tick(self, q_policy):
        """One control period: HOM's command when q_policy is None, else the joint angles q_policy (W, 3, 3)."""
        th = self.torch
        k, t, vw, W = self.k, self.clock[self.k], self.vw, self.W
        qpos = vw["qpos"].to(self.f64)
        R = vw["xmat"][:, self.ob].to(self.f64)
        pos = qpos[:, self.qa:self.qa + 3]
        F = self.ft_t.to(self.f64)
        cop = self.cop_t.to(self.f64) / F.clamp_min(1e-12)[..., None]
        cop = th.where((F > 0)[..., None], cop, th.zeros_like(cop))
        w_ref, s_push, th_cmd, ph = self.schedule(t)
        w_ref_t = th.tensor(w_ref, dtype=self.f64, device=self.dev)
        v_ref = th.zeros((W, 3), dtype=self.f64, device=self.dev)
        if self.task == "roll":
            v_ref = th.cross(w_ref_t.expand(W, 3), -self.scene.r * self.n_out[:, 0], dim=-1)
        if self.record:
            wl = vw["qvel"][:, self.va + 3:self.va + 6].to(self.f64)
            self.rec["pos"][k] = pos.float()
            self.rec["w"][k] = th.einsum("wij,wj->wi", R, wl).float()
            self.rec["ft"][k] = self.ft_t
            self.rec["fo"][k] = self.fo_t
            self.rec["ax"][k] = R[:, :, 2].float()
            self.rec["cop"][k] = th.einsum("wji,wfj->wfi", R, cop - pos[:, None, :]).float()
            self.rec["vr"][k] = v_ref.float()
            self.phases.append(ph)
            self.wrefs.append(w_ref)
            self.times.append(t)
            self.thref.append(th_cmd)
        if t >= T.T_LIFT and not self.post_off:
            vw["mocap_pos"][:, self.mo_post, 2] -= 1.0                              # the post goes away
            self.post_off = True
        if q_policy is None:
            true = {"q": qpos[:, self.qadr_t], "pos": pos, "quat": qpos[:, self.qa + 3:self.qa + 7], "cop": cop, "F": F}
            meas = self.measure(k, true)
            mv = self.mirror(meas)
            Rd = self._quat_mat(meas["quat_d"])
            RdT = Rd.transpose(-1, -2)
            V = th.cat([th.einsum("wij,j->wi", RdT, w_ref_t), th.einsum("wij,wj->wi", RdT, v_ref)], -1)
            ctrl = self.control(t, meas, mv, V, s_push) - self.cal_q
        else:
            ctrl = q_policy.to(self.f64)
        if HG.DEADBAND > 0:
            wp.to_torch(self.target).copy_(ctrl.reshape(W, 9).float())
        else:
            vw["ctrl"][:, self.act_t] = ctrl.reshape(W, 9).float()
        wp.capture_launch(self.graph)
        dtc = self.prm.dt
        if t + dtc <= self.hold_max + 1e-9:          # variant hold: the object back at its start pose, at rest
            hm = (self.hold_t >= t + dtc - 1e-9)[:, None]
            q_o = vw["qpos"][:, self.qa:self.qa + 7]
            q_o.copy_(th.where(hm, self.obj0_t, q_o))
            v_o = vw["qvel"][:, self.va:self.va + 6]
            v_o.copy_(th.where(hm, th.zeros_like(v_o), v_o))
        self.k += 1
        return w_ref_t, v_ref, th_cmd

    def done(self):
        return self.k >= len(self.clock)

    def state(self):
        """Torch views of the per-world state a handover snapshot keeps."""
        to = wp.to_torch
        st = {"qpos": to(self.wd.qpos), "qvel": to(self.wd.qvel), "ctrl": to(self.wd.ctrl),
              "mocap_pos": to(self.wd.mocap_pos), "body_pos": to(self.wm.body_pos)}
        if hasattr(self.wd, "qacc_warmstart"):
            st["warm"] = to(self.wd.qacc_warmstart)
        return st

    def physics(self, q):
        """One control period of the plant with the joint angles q (W, 3, 3) as the servo targets."""
        if HG.DEADBAND > 0:
            wp.to_torch(self.target).copy_(q.reshape(self.W, 9).float())
        else:
            self.vw["ctrl"][:, self.act_t] = q.reshape(self.W, 9).float()
        wp.capture_launch(self.graph)

    def finish(self):
        """The rows of a recorded episode (hom_gpu.Batch.score): one per world."""
        wp.synchronize()
        out = {k: v.cpu().numpy() for k, v in self.rec.items()}
        final = {"pos": self.vw["qpos"][:, self.qa:self.qa + 3].cpu().numpy(), "ftot": self.fo_t.cpu().numpy(),
                 "n_ra": self.n_ra.cpu().numpy()}
        return self.score(out, final, self.phases, np.array(self.wrefs), self.times, self.thref)


# ------------------------------------------------------------------------------------------------ the task
class Task:
    """The task phase's references against the task's clock (hom_gpu.Batch.schedule): the commanded world angular
    velocity (n, 3) and, for the cylinders, the commanded pinch angle (n,) in rad, one row per control period."""

    def __init__(self, B):
        n0 = B.k
        w, th = [], []
        for t in B.clock[n0:]:
            w_ref, _s, th_cmd, _ph = B.schedule(t)
            w.append(w_ref)
            th.append(math.radians(th_cmd) if th_cmd is not None else 0.0)
        self.n = len(w)
        self.k0 = max(0, int(round((B.t_task - B.clock[n0]) / B.prm.dt)))    # the task's first row in the table
        self.k_lift = int(round((T.T_LIFT - B.clock[n0]) / B.prm.dt)) if B.clock[n0] < T.T_LIFT else None
        self.w = torch.tensor(np.array(w), dtype=torch.float32, device="cuda")
        self.th = torch.tensor(np.array(th), dtype=torch.float32, device="cuda")
        self.pinch = B.task in ("pinpush", "press")
        self.roll = B.task == "roll"


class Tracker:
    """Per-world references, rewards and observations for W worlds (UHAS's terms, module docstring): the start pose
    and its turned goal, the reference position, the previous action and the finger points' last positions, and J's
    ingredients accumulated over each world's task."""

    def __init__(self, W, device="cuda"):
        self.W, self.dev = W, device
        z = (lambda *sh: torch.zeros(*sh, device=device))
        self.tau = torch.zeros(W, dtype=torch.long, device=device)
        self.q0, self.q_ref, self.p0, self.p_ref, self.a0 = z(W, 4), z(W, 4), z(W, 3), z(W, 3), z(W, 3)
        self.palm, self.n0, self.qRsT, self.last_a, self.pts_prev = z(W, 3), z(W, 3), z(W, 4), z(W, 9), z(W, 3, 2, 3)
        self.th_ref, self.alive = z(W), torch.ones(W, dtype=torch.bool, device=device)
        self.acc = {k: z(W) for k in ("err2", "ref2", "contact", "n")}

    def reset(self, idx, B, C, F, mounts, n_out0, qRsT=None):
        """Worlds idx start their task now, from the batch's current state. qRsT: the quaternions of their sphere frames'
        transposes, when known."""
        vw = B.vw
        self.palm[idx] = vw["mocap_pos"][idx, B.mo_palm].float()
        self.p0[idx] = vw["qpos"][idx, B.qa:B.qa + 3].float() - self.palm[idx]
        self.p_ref[idx] = self.p0[idx]
        self.q0[idx] = vw["qpos"][idx, B.qa + 3:B.qa + 7].float()
        self.a0[idx] = vw["xmat"][idx, B.ob].float().reshape(-1, 3, 3)[:, :, 2]
        self.q_ref[idx] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.dev)
        self.th_ref[idx] = 0.0
        self.tau[idx] = 0
        self.last_a[idx] = 0.0
        self.n0[idx] = n_out0
        self.alive[idx] = True
        for v in self.acc.values():
            v[idx] = 0.0
        self.qRsT[idx] = qRsT if qRsT is not None else torch.tensor(
            np.array([mat_to_quat(R.T) for R in C.R[idx].cpu().numpy()]), dtype=torch.float32, device=self.dev)
        q = B.vw["qpos"][:, B.qadr_t.reshape(-1)].float().reshape(self.W, 3, 3)
        self.pts_prev[idx] = C.to_sphere(F.points(F.fk(mounts, q)))[idx]

    def advance(self, task, r_obj):
        """Integrate the references over one period at each world's own task time."""
        i = self.tau.clamp(max=task.n - 1)
        w = task.w[i]
        dt = T.DT_C
        ang = w.norm(dim=-1)
        dq = quat_from_axis_angle(w / ang.clamp_min(1e-12)[:, None], ang * dt)
        self.q_ref = quat_mul(dq, self.q_ref)
        self.q_ref = self.q_ref / self.q_ref.norm(dim=-1, keepdim=True)
        if task.roll:
            self.p_ref = self.p_ref + torch.cross(w, -r_obj * self.n0, dim=-1) * dt
        self.th_ref = task.th[i]
        self.w_now = w

    def goal(self, task):
        if task.pinch:
            ax = torch.tensor([1.0, 0.0, 0.0], device=self.dev).expand(self.W, 3)
            return quat_mul(quat_from_axis_angle(ax, -self.th_ref), self.q0)
        return quat_mul(self.q_ref, self.q0)

    def rot_error(self, task, B):
        if task.pinch:
            a = B.vw["xmat"][:, B.ob].float().reshape(-1, 3, 3)[:, :, 2]
            c, s_ = torch.cos(-self.th_ref), torch.sin(-self.th_ref)
            a0 = self.a0
            a_ref = torch.stack([a0[:, 0], c * a0[:, 1] - s_ * a0[:, 2], s_ * a0[:, 1] + c * a0[:, 2]], -1)
            return torch.arccos((a * a_ref).sum(-1).clamp(-1.0, 1.0))
        dq = quat_mul(B.vw["qpos"][:, B.qa + 3:B.qa + 7].float(), quat_conj(self.goal(task)))
        return 2.0 * torch.asin(dq[:, 1:].norm(dim=-1).clamp(max=1.0))

    def commanded(self, task):
        if task.pinch:
            return self.th_ref.abs()
        return 2.0 * torch.asin(self.q_ref[:, 1:].norm(dim=-1).clamp(max=1.0))

    def observe(self, task, B, C, F, mounts):
        W, vw = self.W, B.vw
        R_s, r = C.R, C.r
        p = vw["qpos"][:, B.qa:B.qa + 3].float() - self.palm
        q_obj = vw["qpos"][:, B.qa + 3:B.qa + 7].float()
        v = vw["qvel"][:, B.va:B.va + 3].float()
        Ro = vw["xmat"][:, B.ob].float().reshape(-1, 3, 3)
        w = (Ro @ vw["qvel"][:, B.va + 3:B.va + 6].float()[..., None])[..., 0]
        to_s = (lambda x: (x[:, None] @ R_s)[:, 0])
        q_s = quat_mul(self.qRsT, q_obj)
        g_s = quat_mul(self.qRsT, self.goal(task))
        q = vw["qpos"][:, B.qadr_t.reshape(-1)].float().reshape(W, 3, 3)
        pts = C.to_sphere(F.points(F.fk(mounts, q)))
        vel = (pts - self.pts_prev) / T.DT_C
        self.pts_prev = pts
        w_now = getattr(self, "w_now", torch.zeros(W, 3, device=self.dev))
        pol = torch.cat([to_s(p - C.c) / r[:, None], q_s, to_s(v) / r[:, None], to_s(w), g_s,
                         quat_mul(q_s, quat_conj(g_s)), self.last_a, pts.reshape(W, -1), vel.reshape(W, -1),
                         to_s(w_now)], -1)
        cri = torch.cat([pol, B.ft_t.float(), (self.tau.float() / task.n)[:, None]], -1)
        return pol, cri

    def reward(self, task, B, C, q_cmd=None, a=None, a_prev=None, lim=None):
        """UHAS's reward terms with the task's reference, the drop test, and J's ingredients; with LIMITS, the limit
        terms of REW_LIMITS (q_cmd: the servo targets just applied, a and a_prev: this and the last action, lim: the
        joints' kp, torque limits and ranges)."""
        p = B.vw["qpos"][:, B.qa:B.qa + 3].float() - self.palm
        dist = (p - self.p_ref).norm(dim=-1).clamp(max=REW["fall_dist"])
        err = self.rot_error(task, B)
        qj = B.vw["qpos"][:, B.qadr_t.reshape(-1)].float().reshape(self.W, 3, 3)
        lat = ((qj[..., 0] - C.q_ref[..., 0]) ** 2).sum(-1)
        rad = ((qj[..., 1:] - C.q_ref[..., 1:]) ** 2).sum((-1, -2))
        on3 = (B.ft_t > REW["f_on"]).all(-1).float()
        drop = self.alive & (p[:, 2] < self.p0[:, 2] - REW["drop_m"])
        rew = (REW["dist"] * dist + REW["rot"] / (err + REW["rot_eps"]) + REW["lat"] * lat + REW["rad"] * rad
               + REW["contact"] * on3)
        if LIMITS and q_cmd is not None:
            q9, L_ = qj.reshape(self.W, 9), REW_LIMITS
            dmd = lim["kp"] * (q_cmd.reshape(self.W, 9).float() - q9).abs() / lim["tau"]
            band = math.radians(L_["q_band_deg"])
            near = torch.minimum(q9 - lim["lo"], lim["hi"] - q9)
            pen = torch.stack([((dmd - L_["tau_on"]) / (1.0 - L_["tau_on"])).clamp(0.0, 3.0).sum(-1),
                               ((band - near) / band).clamp(0.0, 1.0).sum(-1), ((a - a_prev) ** 2).sum(-1)], -1)
            rew = (rew + pen @ torch.tensor([L_["tau"], L_["q"], L_["rate"]], device=self.dev)).clamp_min(0.0)
            self.pen = pen
        rew = torch.where(drop, rew + REW["fall"], rew)
        live = self.alive.float() * (self.tau >= task.k0).float()
        ref = self.commanded(task)
        self.acc["err2"] += live * err ** 2
        self.acc["ref2"] += live * ref ** 2
        self.acc["contact"] += live * on3
        self.acc["n"] += live
        return rew, drop

    def j_proxy(self, idx, dropped):
        """J's proxy of worlds idx: 1 - RMS error / RMS commanded rotation, times the three-pad share; 0 if dropped."""
        n = self.acc["n"][idx].clamp_min(1.0)
        s = (1.0 - (self.acc["err2"][idx] / n).sqrt() / (self.acc["ref2"][idx] / n).sqrt().clamp_min(1e-6)).clamp_min(0)
        return torch.where(dropped, torch.zeros_like(s), s * self.acc["contact"][idx] / n)


# ------------------------------------------------------------------------------------------------ training
class TrainEnv:
    """W worlds of one task under UHAS's sphere controller, an RSL-RL vectorized environment with per-world resets.
    A world starts from a handover state drawn from a pool (HOM's approach, closing, squeeze and lift on a training
    layout and grasp, kept when the object is held: no lower than 15 mm below its start, two pads loaded) and ends when
    the object drops or the task's references end (a time-out)."""

    def __init__(self, task, entries, W, pool_batches=4, seed=0, device="cuda", start="handover"):
        from tensordict import TensorDict
        self.TD = TensorDict
        self.start = start
        self.task_name, self.entries, self.W = task, entries, W
        self.num_envs, self.num_actions, self.device = W, 9, device
        self.rng = np.random.default_rng(seed)
        self.scene = HG.Scene(T.SHAPE[task], D_MM[task])
        self.prep, self.seed_next = {}, 100 + 100000 * seed
        self.cfg = {"task": task, "worlds": W, "reward": REW, "pool_batches": pool_batches, "start": start}
        worlds = self.draw()
        self.B = UBatch(self.scene, task, worlds, T.DT_C, gen_seed=seed)
        self.F = U.Fingers(self.B.m, device)
        self.build_pool(worlds, pool_batches)
        self.task = Task(self.B)
        self.max_episode_length = self.task.n
        self.episode_length_buf = torch.zeros(W, dtype=torch.long, device=device)
        self.tr = Tracker(W, device)
        self.snap = torch.zeros(W, dtype=torch.long, device=device)
        self.C = self.pool_C.subset(torch.zeros(W, dtype=torch.long, device=device))
        self.mounts = self.C.mounts
        m, B = self.B.m, self.B
        t9 = (lambda x: torch.tensor(np.asarray(x, np.float32).reshape(-1), device=device))
        self.lim = {"kp": t9(B.kp), "tau": t9(np.abs(m.actuator_forcerange[B.act.reshape(-1)]).max(1)),
                    "lo": t9(B.lo), "hi": t9(B.hi)}
        self.pen_sum, self.pen_n = torch.zeros(3, device=device), 0
        self.done_j, self.done_n = [], 0
        self.reset_worlds(torch.arange(W, device=device))
        self.obs = self.observe()

    def prepared(self, key, mxy, setup):
        k = (int(key["call"]), int(key.get("grasp", 0)))
        if k not in self.prep:
            self.prep[k] = self.scene.prepare(mxy, setup)
        return self.prep[k]

    def draw(self):
        out = []
        for i in self.rng.integers(0, len(self.entries), self.W):
            key, mxy, setup = self.entries[i]
            bm, q_open = self.prepared(key, mxy, setup)
            out.append({"key": key, "seed": int(self.seed_next), "setup": setup, "body_mounts": bm, "q_open": q_open,
                        "var": HG.variant_world(VARIANT[self.task_name], int(self.seed_next))})
            self.seed_next += 1
        return out

    def build_pool(self, worlds, n):
        """The start states of n batches of W worlds, with their sphere controllers: HOM's handovers in which the object
        is held, or (start 'open') every world at time 0, the fingers open and the object on its post."""
        B, keep = self.B, {k: [] for k in ("qpos", "qvel", "ctrl", "mocap_pos", "body_pos", "mounts", "n_out")}
        t0, kept, total = time.time(), 0, 0
        for b in range(n):
            if b:
                worlds = self.draw()
            B.begin(worlds, hom=self.start == "handover")
            st = B.state()
            pos = st["qpos"][:, B.qa:B.qa + 3]
            held = ((B.ft_t > REW["f_on"]).sum(-1) >= 2) & (pos[:, 2] > self.B.obj0_t[:, 2] - REW["drop_m"])
            if self.start == "open":
                held = torch.ones_like(held)
            idx = torch.nonzero(held).squeeze(-1)
            for k in ("qpos", "qvel", "ctrl", "mocap_pos", "body_pos"):
                keep[k].append(st[k][idx].clone())
            keep["mounts"].append(torch.tensor(np.array([[wd_["body_mounts"][f] for f in HG.FINGERS] for wd_ in worlds]),
                                               dtype=torch.float32, device=self.device)[idx])
            keep["n_out"].append(B.n_out[idx, 0].float())
            kept += len(idx)
            total += len(worlds)
        self.pool = {k: torch.cat(v) for k, v in keep.items()}
        self.pool_C = U.SphereController(self.F, self.pool["mounts"])
        self.pool_qRsT = torch.tensor(np.array([mat_to_quat(R.T) for R in self.pool_C.R.cpu().numpy()]),
                                      dtype=torch.float32, device=self.device)
        self.P = int(self.pool["qpos"].shape[0])
        print(f"pool ({self.start}): {kept} of {total} start states kept ({100 * kept / max(1, total):.0f} %), "
              f"{time.time() - t0:.0f} s", flush=True)

    def reset_worlds(self, idx):
        if len(idx) == 0:
            return
        src = torch.as_tensor(self.rng.integers(0, self.P, len(idx)), device=self.device)
        st = self.B.state()
        for k in ("qpos", "qvel", "ctrl", "mocap_pos", "body_pos"):
            st[k][idx] = self.pool[k][src]
        if "warm" in st:
            st["warm"][idx] = 0.0
        self.C.copy_rows(self.pool_C, idx, src)
        self.snap[idx] = src
        self.B.mjw.kinematics(self.B.wm, self.B.wd)
        wp.synchronize()
        self.tr.reset(idx, self.B, self.C, self.F, self.C.mounts, self.pool["n_out"][src], self.pool_qRsT[src])
        self.episode_length_buf[idx] = 0

    def observe(self):
        pol, cri = self.tr.observe(self.task, self.B, self.C, self.F, self.C.mounts)
        return self.TD({"policy": pol, "critic": cri}, batch_size=[self.W], device=self.device)

    def get_observations(self):
        return self.obs

    def step(self, actions):
        a = actions.clamp(-1.0, 1.0)
        q = self.C.solve(a)
        self.B.physics(q)
        tr = self.tr
        a_prev = tr.last_a
        tr.last_a = a
        tr.tau += 1
        if self.task.k_lift is not None:                  # each world's post goes away at its own lift time
            lift = torch.nonzero(tr.tau == self.task.k_lift).squeeze(-1)
            if len(lift):
                self.B.vw["mocap_pos"][lift, self.B.mo_post, 2] -= 1.0
        tr.advance(self.task, self.scene.r)
        rew, drop = tr.reward(self.task, self.B, self.C, q, a, a_prev, self.lim)
        if LIMITS:
            self.pen_sum += tr.pen.mean(0)
            self.pen_n += 1
        tout = (tr.tau >= self.task.n) & ~drop
        done = drop | tout
        self.episode_length_buf += 1
        extras = {"time_outs": tout.float()}
        idx = torch.nonzero(done).squeeze(-1)
        if len(idx):
            j = tr.j_proxy(idx, drop[idx])
            self.done_j.append(j)
            self.done_n += len(idx)
            extras["log"] = {"J_proxy": j.mean().item(), "dropped": drop[idx].float().mean().item()}
            self.reset_worlds(idx)
        self.obs = self.observe()
        return self.obs, rew, done.float(), extras

    def take_log(self):
        """Mean J proxy and count of the world episodes finished since the last call."""
        if not self.done_j:
            return {"J_proxy": None, "world_episodes": 0}
        j = torch.cat(self.done_j)
        self.done_j = []
        out = {"J_proxy": round(j.mean().item(), 4), "J_proxy_p90": round(j.quantile(0.9).item(), 4),
               "world_episodes": int(len(j))}
        if self.pen_n:                                    # the limit terms' means per world and period, unweighted
            out.update(zip(("pen_tau", "pen_q", "pen_rate"), (self.pen_sum / self.pen_n).round(decimals=4).tolist()))
            self.pen_sum.zero_()
            self.pen_n = 0
        return out


# ------------------------------------------------------------------------------------------------ evaluation
def evaluate(task, policy, worlds_list, W=2048, seed=0, device="cuda", start="handover"):
    """The policy (deterministic) on the given worlds, HOM driving the phases before the task as in the grids; the
    rows of hom_gpu.Batch.score, one per world, with J."""
    import hom_landscape as L
    scene = HG.Scene(T.SHAPE[task], D_MM[task])
    rows = []
    B = None
    for c0 in range(0, len(worlds_list), W):
        chunk = worlds_list[c0:c0 + W]
        pad = chunk + [chunk[-1]] * (W - len(chunk))
        if B is None:
            B = UBatch(scene, task, pad, T.DT_C, gen_seed=seed)
            F = U.Fingers(B.m, device)
        t0 = time.time()
        B.begin(pad, record=True, hom=start == "handover")
        task_ref = Task(B)
        mounts = torch.tensor(np.array([[wd_["body_mounts"][f] for f in HG.FINGERS] for wd_ in pad]),
                              dtype=torch.float32, device=device)
        C = U.SphereController(F, mounts)
        tr = Tracker(W, device)
        allw = torch.arange(W, device=device)
        tr.reset(allw, B, C, F, mounts, B.n_out[:, 0].float())
        with torch.inference_mode():
            while not B.done():
                pol, _cri = tr.observe(task_ref, B, C, F, mounts)
                from tensordict import TensorDict
                a = policy(TensorDict({"policy": pol, "critic": _cri}, batch_size=[W], device=device)).clamp(-1, 1)
                B.tick(C.solve(a))
                tr.last_a = a
                tr.tau += 1
                tr.advance(task_ref, scene.r)
        res = B.finish()[:len(chunk)]
        for wd_, r in zip(chunk, res):
            row = dict(wd_["key"], task=task, d_mm=D_MM[task], seed=wd_["seed"], variant=wd_["var"]["var"], **r)
            row["J"] = L.track_score(task, row)
            rows.append(row)
        print(f"evaluated {len(rows)} of {len(worlds_list)} rollouts; batch {time.time() - t0:.0f} s", flush=True)
    return rows


# ------------------------------------------------------------------------------------------------ commands
def train_entries(task, split_seed=0):
    train, test, J = split(task, split_seed)
    tr = set(train)
    return [e for e in grid_entries(task) if xy_of(e[0]) in tr], train, test, J


def cmd_smoke(a):
    """A few hundred steps with random actions: the pool, timing and the observations."""
    entries, _train, _test, _J = train_entries(a.task)
    t0 = time.time()
    env = TrainEnv(a.task, entries, a.worlds, pool_batches=a.pool, start=a.start)
    print(f"build and pool: {time.time() - t0:.1f} s; policy obs {tuple(env.obs['policy'].shape)}, critic obs "
          f"{tuple(env.obs['critic'].shape)}; task steps {env.task.n}", flush=True)
    t1 = time.time()
    for i in range(a.steps):
        act = torch.rand(env.W, 9, device=env.device) * 2 - 1
        obs, rew, done, ex = env.step(act)
        if not torch.isfinite(obs["policy"]).all():
            raise RuntimeError(f"non-finite observation at step {i}")
    dt = time.time() - t1
    print(f"{a.steps} steps: {a.steps * env.W / dt:.0f} transitions/s; {env.take_log()}; reward mean {rew.mean():.2f}",
          flush=True)


def cmd_train(a):
    from rsl_rl.runners import OnPolicyRunner
    entries, train, test, _J = train_entries(a.task, a.split_seed)
    global LIMITS
    LIMITS = bool(a.limits)
    tag = ("" if a.start == "handover" else f"_{a.start}") + ("_limits" if LIMITS else "")
    if a.only:                                        # UHAS's single-hand baseline: one layout, its own grasps
        one = (float(a.only[0]), float(a.only[1]))
        entries = [e for e in grid_entries(a.task) if xy_of(e[0]) == one]
        train = test = [one]
        tag += f"_x{one[0]:g}_y{one[1]:g}"
    stamp = time.strftime("%Y%m%d-%H%M")
    run = RUNS / f"{stamp}_{a.task}{tag}_s{a.seed}"
    run.mkdir(parents=True, exist_ok=True)
    (run / "split.json").write_text(json.dumps({"task": a.task, "split_seed": a.split_seed, "train": train,
                                                 "test": test, "grasps": len(entries), "start": a.start,
                                                 "limits": REW_LIMITS if LIMITS else None}, indent=1))
    torch.manual_seed(a.seed)
    t0 = time.time()
    env = TrainEnv(a.task, entries, a.worlds, pool_batches=a.pool, seed=a.seed, start=a.start)
    cfg = json.loads(json.dumps(PPO_CFG))
    runner = OnPolicyRunner(env, cfg, log_dir=str(run), device="cuda")
    log = run / "progress.jsonl"
    print(f"{run}: {len(train)} training layouts ({len(entries)} grasps), {len(test)} test; {a.worlds} worlds; pool "
          f"{env.P}; {env.task.n} task steps; setup {time.time() - t0:.0f} s; budget {a.minutes} min", flush=True)
    save0 = runner.save

    def save(path, infos=None):                       # every checkpoint also appends a progress row, fsynced
        save0(path, infos)
        it = runner.current_learning_iteration
        rec = {"iteration": it, "checkpoint": Path(path).name, "minutes": round((time.time() - t0) / 60.0, 2),
               "transitions": (it + 1) * PPO_CFG["num_steps_per_env"] * env.W, **env.take_log()}
        with open(log, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        print(json.dumps(rec), flush=True)
    runner.save = save
    t1 = time.time()
    runner.learn(a.calib)                              # the iteration time, then one run to the budget
    runner.logger.init_logging_writer = (lambda: None)
    per_it = (time.time() - t1) / a.calib
    n = int(max(0.0, a.minutes * 60.0 - (time.time() - t0)) / per_it)
    print(f"{per_it:.2f} s per iteration; {n} more iterations", flush=True)
    runner.current_learning_iteration += 1
    if n > 0:
        runner.learn(n)
    print(f"done: {runner.current_learning_iteration + 1} iterations in {(time.time() - t0) / 60:.1f} min -> {run}",
          flush=True)


def load_policy(run, ckpt=None):
    """The actor of a training run's last (or given) checkpoint, as a function of the observations."""
    import glob
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict
    path = ckpt or max(glob.glob(str(Path(run) / "model_*.pt")), key=lambda p_: int(Path(p_).stem.split("_")[1]))
    sd = torch.load(path, map_location="cuda", weights_only=False)
    dims = {"policy": 69, "critic": 73}
    obs = TensorDict({k: torch.zeros(1, v, device="cuda") for k, v in dims.items()}, batch_size=[1])
    cfg = dict(PPO_CFG["actor"])
    cfg.pop("class_name")
    actor = MLPModel(obs, {"actor": ["policy"], "critic": ["critic"]}, "actor", 9, **cfg).to("cuda")
    actor.load_state_dict(sd["actor_state_dict"])
    actor.eval()
    return actor, path


def cmd_eval(a):
    run = Path(a.run)
    sp = json.loads((run / "split.json").read_text())
    task = sp["task"]
    want = set(map(tuple, sp[a.split][:a.limit] if a.limit else sp[a.split]))
    entries = [e for e in grid_entries(task) if xy_of(e[0]) in want]
    actor, path = load_policy(run, a.ckpt)
    env_scene = HG.Scene(T.SHAPE[task], D_MM[task])
    worlds = []
    for key, mxy, setup in entries:
        bm, q_open = env_scene.prepare(mxy, setup)
        for sd in (0, 1, 2):
            worlds.append({"key": key, "seed": sd, "setup": setup, "body_mounts": bm, "q_open": q_open,
                           "var": HG.variant_world(VARIANT[task], sd)})
    t0 = time.time()
    rows = evaluate(task, actor, worlds, W=a.worlds, start=sp.get("start", "handover"))
    out = OUT / "rows" / f"uhas_{task}_{run.name}_{a.split}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for r in rows:
            r["checkpoint"] = Path(path).name
            fh.write(json.dumps(r, default=float) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    held = np.mean([bool(r.get("held")) for r in rows])
    print(f"{len(rows)} rollouts in {time.time() - t0:.0f} s; held {held:.2f}; mean J {np.mean([r['J'] for r in rows]):.3f}"
          f" -> {out.relative_to(ROOT)}", flush=True)


def fit_actions(C, F, q_target, iters=8, pop=128, elite=16, seed=0):
    """The sphere actions (W, 9) whose cascade-IK postures come nearest the joint angles q_target (W, 3, 3): the
    cross-entropy method on the pads' contact points (mean distance over the three fingers, m). Returns the actions,
    their postures and the contact-point and joint errors."""
    W, dev = C.W, F.device
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    rep = torch.arange(W, device=dev).repeat_interleave(pop)
    Cr = C.subset(rep)
    e_t = F.fk(C.mounts, q_target)["contact"]                                   # (W, 3, 3)
    mu, sd = torch.zeros(W, 9, device=dev), torch.full((W, 9), 0.6, device=dev)
    best_a, best_e = torch.zeros(W, 9, device=dev), torch.full((W,), float("inf"), device=dev)
    for _ in range(iters):
        A = (mu[:, None] + sd[:, None] * torch.randn(W, pop, 9, device=dev, generator=g)).clamp(-1.0, 1.0)
        q = Cr.solve(A.reshape(W * pop, 9))
        e = (F.fk(Cr.mounts, q)["contact"] - e_t[rep]).norm(dim=-1).mean(-1).reshape(W, pop)
        order = e.argsort(1)[:, :elite]
        El = torch.gather(A, 1, order[..., None].expand(-1, -1, 9))
        mu, sd = El.mean(1), El.std(1).clamp_min(0.02)
        b = e.gather(1, order[:, :1])[:, 0]
        better = b < best_e
        best_e = torch.where(better, b, best_e)
        best_a = torch.where(better[:, None], A[torch.arange(W, device=dev), order[:, 0]], best_a)
    q_fit = C.solve(best_a)
    dq = (q_fit - q_target).abs()
    return best_a, q_fit, best_e, dq


def cmd_express(a):
    """How near UHAS's sphere actions come to the grasp postures HOM hands over at the start of the task."""
    entries, _train, _test, _J = train_entries(a.task)
    env_rng = np.random.default_rng(a.seed)
    scene = HG.Scene(T.SHAPE[a.task], D_MM[a.task])
    worlds = []
    for i in env_rng.integers(0, len(entries), a.worlds):
        key, mxy, setup = entries[i]
        bm, q_open = scene.prepare(mxy, setup)
        worlds.append({"key": key, "seed": 100 + int(i), "setup": setup, "body_mounts": bm, "q_open": q_open,
                       "var": HG.variant_world(VARIANT[a.task], 100 + int(i))})
    B = UBatch(scene, a.task, worlds, T.DT_C)
    B.begin(worlds)
    F = U.Fingers(B.m, "cuda")
    mounts = torch.tensor(np.array([[wd_["body_mounts"][f] for f in HG.FINGERS] for wd_ in worlds]),
                          dtype=torch.float32, device="cuda")
    C = U.SphereController(F, mounts)
    q_t = B.vw["qpos"][:, B.qadr_t.reshape(-1)].float().reshape(a.worlds, 3, 3)
    held = ((B.ft_t > REW["f_on"]).sum(-1) >= 2).cpu().numpy()
    _a, _q, err, dq = fit_actions(C, F, q_t, iters=a.iters, pop=a.pop)
    e = err.cpu().numpy() * 1e3
    d = np.degrees(dq.cpu().numpy())
    out = {"task": a.task, "worlds": a.worlds, "held_at_handover": int(held.sum()), "iters": a.iters, "pop": a.pop,
           "contact_err_mm": {"median": float(np.median(e[held])), "p90": float(np.percentile(e[held], 90)),
                              "max": float(e[held].max())},
           "joint_err_deg_median": {j: float(np.median(d[held][:, :, k])) for k, j in enumerate(U.JOINTS)},
           "share_within_2mm": float(np.mean(e[held] <= 2.0)), "share_within_5mm": float(np.mean(e[held] <= 5.0))}
    path = OUT / "rows" / f"uhas_express_{a.task}_{a.iters}x{a.pop}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("smoke")
    s.add_argument("--task", default="twist", choices=list(D_MM))
    s.add_argument("--worlds", type=int, default=64)
    s.add_argument("--pool", type=int, default=1)
    s.add_argument("--steps", type=int, default=200)
    s.add_argument("--start", default="handover", choices=("handover", "open"))
    t = sub.add_parser("train")
    t.add_argument("--task", default="twist", choices=list(D_MM))
    t.add_argument("--worlds", type=int, default=2048)
    t.add_argument("--pool", type=int, default=4, help="batches of handover states")
    t.add_argument("--minutes", type=float, default=120.0)
    t.add_argument("--calib", type=int, default=20, help="iterations that time the run before the budget is set")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--split-seed", type=int, default=0)
    t.add_argument("--only", nargs=2, type=float, default=None, help="train on this layout alone (x_sep y_sep, mm)")
    t.add_argument("--start", default="handover", choices=("handover", "open"),
                   help="the policy takes over at HOM's handover, or acts from the open hand with the object on its post")
    t.add_argument("--limits", action="store_true", help="the joint and actuator limit terms (REW_LIMITS) in the "
                   "reward")
    e = sub.add_parser("eval")
    e.add_argument("--run", required=True)
    e.add_argument("--split", default="test", choices=("test", "train"))
    e.add_argument("--ckpt", default=None)
    e.add_argument("--worlds", type=int, default=2048)
    e.add_argument("--limit", type=int, default=0, help="the first N layouts of the split only")
    x = sub.add_parser("express")
    x.add_argument("--task", default="twist", choices=list(D_MM))
    x.add_argument("--worlds", type=int, default=64)
    x.add_argument("--iters", type=int, default=8)
    x.add_argument("--pop", type=int, default=128)
    x.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    {"smoke": cmd_smoke, "train": cmd_train, "eval": cmd_eval, "express": cmd_express}[a.cmd](a)


if __name__ == "__main__":
    main()
