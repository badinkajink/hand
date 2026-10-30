"""Batched CPU MuJoCo environment for the residual grasping policy (manta-hand final paper, III).

One episode = one grasp phase on a fresh cylinder:
    diameter d ~ U[20, 40] mm, mass m ~ U[20, 150] g, placement error +-2 mm (x, y), +-2 deg (yaw).
The open-loop plan (plan.py; keyframes interpolated between a table of radii) pre-shapes, closes,
squeezes and lifts the palm 80 mm by T_g = 2.4 s. The policy only nudges the finger targets:

    q_target_t = q_plan(t) + dq_t,   dq_t = BETA * dq_{t-1} + (1 - BETA) * DQ_MAX * a_t,   a_t in [-1, 1]^9
With EnvCfg.palm_action (default) a 10th action nudges the UR5's palm height the same way:
    z_palm_t = z_plan(t) + dz_t,     dz_t = BETA * dz_{t-1} + (1 - BETA) * DZ_MAX * a_t[9]
and any finger-table contact is penalized (the fingertips must never press into the table).

At T_g the fixed reorient judge (judge.py) runs on a copy-free continuation of the env's own state
and its verdict is the terminal reward. The cylinder dropping during the grasp ends the episode.

Observations (actor; only what the hardware measures):
    last 3 joint positions (27), tracking error q_target_{t-1} - q_t (9), previous action (9),
    phase t / T_g (1)                                                    -> 46
    condition C adds the last 5 servo loads (45)                          -> 91
Critic (privileged, training only): actor obs + object position rel. palm (3), quaternion (4),
linear and angular velocity (6), per-finger contact (3), diameter and mass (2).

Actuators: "default" = MuJoCo position servo (scene.py: kp 1.5 N m/rad, 0.23 N m limit, no friction),
the paper's condition A. The identified (BAM) actuator for B / C plugs in through `actuator=`.

Physics 500 Hz, control 50 Hz, envs stepped in a thread pool (MuJoCo releases the GIL in mj_step).
"""
from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import mujoco
import numpy as np

from . import judge, kin, plan, scene

DT = 0.02
N_STEPS = int(round(plan.T_GRASP_END / DT))      # 120 control steps
DQ_MAX = 0.15
DZ_MAX = 0.005           # m, palm-height residual (UR5 z offset), 10th action
BETA = 0.5
D_RANGE = (0.020, 0.040)
M_RANGE = (0.020, 0.150)
PLACE_XY, PLACE_YAW = 0.002, np.radians(2.0)
PLAN_DIAMETERS = tuple(np.round(np.arange(0.0175, 0.04251, 0.0025), 5))   # table, padded past the range
N_HIST_Q, N_HIST_LOAD = 3, 5


@dataclass
class Weights:
    """Reward weights (paper III-D). Per-step terms are small next to the judge's terminal verdict."""
    contact: float = 0.02        # w1 * n_t / 3
    carry: float = 0.03          # w2 * h_t
    slip: float = 0.5            # w3 * |v_slip| (m/s)
    residual: float = 0.5        # w4 * |dq|^2 (rad^2)
    smooth: float = 0.01         # w5 * |a_t - a_{t-1}|^2
    torque: float = 0.05         # w6 * sum relu(|tau|/tau_max - kappa)^2
    kappa: float = 0.8
    success: float = 10.0        # w_s, judge succeeds
    tilt: float = 5.0            # w_phi * tilt / 90 deg
    drop: float = 5.0            # w_d, dropped during the grasp
    floor: float = 0.2           # per step with any finger touching the table


@dataclass
class EnvCfg:
    n_envs: int = 256
    threads: int = 12
    load_obs: bool = False       # condition C
    actuator: str = "default"    # "default" | "bam"
    weights: Weights = field(default_factory=Weights)
    seed: int = 0
    palm_action: bool = True     # 10th action: UR5 palm-height residual
    mu_range: tuple = (1.5, 2.5)  # pad-object friction DR, foam-taped cylinder on TPU pads (assumed)


class _Plans:
    """Keyframes for any radius, interpolated linearly between the cached plan table."""

    def __init__(self):
        radii = tuple(float(d / 2) for d in PLAN_DIAMETERS)
        tab = plan.plan_table(radii)
        self.r = np.array(radii)
        self.q_pre = np.stack([tab[r].q_pre for r in radii])
        self.q_grip = np.stack([tab[r].q_grip for r in radii])
        self.palm = np.array([tab[r].palm_z for r in radii])
        self.feasible = np.array([tab[r].feasible for r in radii])

    def get(self, radius):
        f = lambda a: np.array([np.interp(radius, self.r, a[:, j]) for j in range(a.shape[1])])
        return f(self.q_pre), f(self.q_grip), float(np.interp(radius, self.r, self.palm))


def plan_q(q_pre, q_grip, t):
    a = np.clip((t - plan.T_CLOSE) / (plan.T_GRIP - plan.T_CLOSE), 0, 1)
    a = a * a * (3 - 2 * a)
    return q_pre + (q_grip - q_pre) * a


def plan_palm(palm_z, t):
    a = np.clip((t - plan.T_LIFT) / plan.T_LIFT_DUR, 0, 1)
    return palm_z + plan.LIFT_DZ * a * a * (3 - 2 * a)


class _Env:
    """One environment: its own MjModel copy (object size / mass) and MjData."""

    def __init__(self, proto, plans: _Plans, palm_action: bool = True):
        self.na = 10 if palm_action else 9
        self.m = copy.copy(proto)
        self.d = mujoco.MjData(self.m)
        self.ix = scene.indices(self.m)
        self.plans = plans
        self.nsub = int(round(DT / self.m.opt.timestep))

    def reset(self, rng, diameter=None, mass=None, place=None, mu=None, mu_range=None):
        r = (diameter if diameter is not None else rng.uniform(*D_RANGE)) / 2
        mass = mass if mass is not None else rng.uniform(*M_RANGE)
        if mu is None:
            mu = rng.uniform(*mu_range) if (rng is not None and mu_range) else scene.MU_PAD_OBJ
        scene.set_object(self.m, r, mass, friction=mu)
        self.r, self.mass, self.mu = r, mass, mu
        self.q_pre, self.q_grip, self.palm_z = self.plans.get(r)
        m, d, ix = self.m, self.d, self.ix
        mujoco.mj_resetData(m, d)
        dx, dy, dyaw = place if place is not None else (rng.uniform(-PLACE_XY, PLACE_XY, 2).tolist()
                                                        + [rng.uniform(-PLACE_YAW, PLACE_YAW)])
        p = scene.object_rest_pose(m, r)
        qo = np.zeros(4)
        mujoco.mju_mulQuat(qo, [np.cos(dyaw / 2), 0, 0, np.sin(dyaw / 2)], p[3:])
        oq = ix["obj_qpos"]
        d.qpos[oq:oq + 7] = [p[0] + dx, p[1] + dy, p[2], *qo]
        d.qpos[ix["joint_qpos"]] = self.q_pre
        d.qpos[ix["palm_qpos"]] = self.palm_z
        d.ctrl[ix["joint_act"]] = self.q_pre
        d.ctrl[ix["palm_act"]] = self.palm_z
        mujoco.mj_forward(m, d)
        self.dq = np.zeros(9)
        self.dz = 0.0
        self.q_hist = [d.qpos[ix["joint_qpos"]].copy()] * N_HIST_Q
        self.load_hist = [np.zeros(9)] * N_HIST_LOAD
        self.q_target = self.q_pre.copy()
        self.prev_a = np.zeros(self.na)
        self.pads_prev = d.site_xpos[ix["pad_site"]].copy()
        self.alive, self.k = True, 0
        self.verdict = None

    # ------------------------------------------------------------------ one control step
    def step(self, a, w: Weights, K: kin.Kinematics):
        """Returns (reward, done, info). `a` in [-1, 1]^9."""
        m, d, ix = self.m, self.d, self.ix
        t = self.k * DT
        a = np.clip(a, -1, 1)
        self.dq = BETA * self.dq + (1 - BETA) * DQ_MAX * a[:9]
        if self.na == 10:
            self.dz = BETA * self.dz + (1 - BETA) * DZ_MAX * a[9]
        lo, hi = ix["jnt_range"][:, 0], ix["jnt_range"][:, 1]
        q_tgt = np.clip(plan_q(self.q_pre, self.q_grip, t + DT) + self.dq, lo, hi)
        d.ctrl[ix["joint_act"]] = q_tgt
        d.ctrl[ix["palm_act"]] = plan_palm(self.palm_z, t + DT) + self.dz
        for _ in range(self.nsub):
            mujoco.mj_step(m, d)
        self.k += 1
        t = self.k * DT

        q = d.qpos[ix["joint_qpos"]].copy()
        self.q_hist = self.q_hist[1:] + [q]
        tau = d.actuator_force[ix["joint_act"]]
        self.load_hist = self.load_hist[1:] + [tau / scene.SERVO_TORQUE]   # default actuator: torque fraction
        fc = _finger_contacts(m, d, ix)
        n = int(fc.sum())
        oq, od = ix["obj_qpos"], ix["obj_dof"]
        z = float(d.qpos[oq + 2])
        pads = d.site_xpos[ix["pad_site"]].copy()
        v_pad = (pads - self.pads_prev).mean(0) / DT
        self.pads_prev = pads
        v_slip = d.qvel[od:od + 3] - v_pad
        lifted = plan_palm(self.palm_z, t) - self.palm_z
        carry = t >= plan.T_LIFT and n >= 2 and (z - self.r) > 0.5 * lifted
        r = (w.contact * n / 3 + w.carry * carry - w.slip * float(np.linalg.norm(v_slip))
             - w.residual * (float(self.dq @ self.dq) + (self.dz / DZ_MAX * DQ_MAX) ** 2)
             - w.floor * _finger_floor(m, d, ix) - w.smooth * float((a - self.prev_a) @ (a - self.prev_a))
             - w.torque * float(np.sum(np.maximum(0, np.abs(tau) / scene.SERVO_TORQUE - w.kappa) ** 2)))
        self.prev_a = a
        self.q_target = q_tgt
        info = {}
        dropped = t >= plan.T_LIFT + plan.T_LIFT_DUR and (z - self.r) < 0.5 * plan.LIFT_DZ
        if dropped:
            r -= w.drop
            self.alive = False
            info = dict(dropped=True, success=False, tilt=0.0, held=False)
            return r, True, info
        if self.k >= N_STEPS:
            v = judge.run(m, d, ix, K)
            self.verdict = v
            r += w.success * v.success + w.tilt * max(0.0, v.tilt_deg) / 90.0
            self.alive = False
            info = dict(dropped=False, success=bool(v.success), tilt=float(v.tilt_deg), held=bool(v.held))
            return r, True, info
        return r, False, info

    # ------------------------------------------------------------------ observations
    def obs(self, load_obs: bool):
        t = self.k * DT
        q = self.q_hist[-1]
        o = [np.concatenate(self.q_hist), self.q_target - q, self.prev_a, [t / plan.T_GRASP_END]]
        if self.na == 10:
            o.append([self.dz / DZ_MAX])
        if load_obs:
            o.append(np.concatenate(self.load_hist))
        return np.concatenate(o)

    def priv(self):
        d, ix = self.d, self.ix
        oq, od = ix["obj_qpos"], ix["obj_dof"]
        palm = d.xpos[ix["palm_body"]]
        return np.concatenate([d.qpos[oq:oq + 3] - palm, d.qpos[oq + 3:oq + 7], d.qvel[od:od + 6],
                               _finger_contacts(self.m, d, ix).astype(float),
                               [self.r * 2 / 0.04, self.mass / 0.15, self.mu]])


def _finger_floor(m, d, ix) -> float:
    """1.0 if any finger geom touches the table this step."""
    fg = ix["floor_geom"]
    fingers = set().union(*ix["finger_bodies"].values())
    for c in d.contact[:d.ncon]:
        if fg in (c.geom1, c.geom2):
            other = c.geom2 if c.geom1 == fg else c.geom1
            if m.geom_bodyid[other] in fingers:
                return 1.0
    return 0.0


def _finger_contacts(m, d, ix) -> np.ndarray:
    og = ix["obj_geom"]
    hit = np.zeros(3, bool)
    for c in d.contact[:d.ncon]:
        if og not in (c.geom1, c.geom2):
            continue
        b = m.geom_bodyid[c.geom2 if c.geom1 == og else c.geom1]
        for i, f in enumerate(scene.FINGERS):
            if b in ix["finger_bodies"][f]:
                hit[i] = True
    return hit


class GraspEnv:
    """N envs in lockstep. Every episode is exactly one grasp phase; dead envs idle until reset."""

    def __init__(self, cfg: EnvCfg):
        if cfg.actuator != "default":
            raise NotImplementedError("BAM actuator: next step")
        self.cfg = cfg
        self.proto = scene.build()
        self.plans = _Plans()
        self.envs = [_Env(self.proto, self.plans, cfg.palm_action) for _ in range(cfg.n_envs)]
        self.act_dim = 10 if cfg.palm_action else 9
        self.pool = ThreadPoolExecutor(cfg.threads)
        self.chunks = np.array_split(np.arange(cfg.n_envs), cfg.threads)
        self.K = [kin.Kinematics(self.proto) for _ in range(cfg.threads)]
        self.rng = np.random.default_rng(cfg.seed)
        self.obs_dim = 46 + (2 if cfg.palm_action else 0) + (45 if cfg.load_obs else 0)
        self.priv_dim = self.obs_dim + 19

    def reset(self, cells=None):
        """cells: optional list of (diameter, mass, (dx, dy, dyaw)) per env, for fixed-grid evaluation."""
        seeds = self.rng.integers(1 << 31, size=len(self.envs))
        for i, e in enumerate(self.envs):
            if cells is None:
                e.reset(np.random.default_rng(seeds[i]), mu_range=self.cfg.mu_range)
            else:
                dm, ms, pl, *mu = cells[i]          # optional 4th entry: friction
                e.reset(None, dm, ms, pl, mu=mu[0] if mu else None)
        return self.observe()

    def observe(self):
        o = np.stack([e.obs(self.cfg.load_obs) for e in self.envs]).astype(np.float32)
        p = np.concatenate([o, np.stack([e.priv() for e in self.envs])], 1).astype(np.float32)
        return o, p

    def step(self, actions):
        n = len(self.envs)
        rew, done = np.zeros(n, np.float32), np.zeros(n, bool)
        infos = [None] * n
        w = self.cfg.weights

        def work(ti):
            for i in self.chunks[ti]:
                e = self.envs[i]
                if not e.alive:
                    done[i] = True
                    continue
                rew[i], done[i], inf = e.step(actions[i], w, self.K[ti])
                if done[i]:
                    infos[i] = inf
        list(self.pool.map(work, range(len(self.chunks))))
        o, p = self.observe()
        return o, p, rew, done, infos
