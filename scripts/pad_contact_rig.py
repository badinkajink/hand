#!/usr/bin/env python3
"""Rig for the contact laws of the production fingertip pads (reorient_backends.replace_tips, the default 1 mm TPU-tip
pads), shared by tests/test_pad_contact_laws.py and the measurements of the 2026-10-13 contact-feedback page.

The pads discretise a Winkler foundation, pressure p = (E/h) x overlap with h = 8.5 mm, as one sphere per ~1 mm2 of tip
surface. Each sphere is a spring K_s = E a_s / h set through MuJoCo's soft contact: at rest a contact row's force over its
penetration is 1 / ((1 - d) tc^2 A), where A is the row's approximate inverse inertia from body_invweight0 of the tip and
the tool, and the builder solves d per finger so that this equals the median K_s. The stiffness is a physical value while
A is the one the pads were built with, so a tool of another mass needs its own build.

The rig is the tracked real_v1 screwdriver scene (the tool a 25 mm cylinder of 24.5 g) with the production pads. The hand
is made rigid (every joint but the tool's removed, the tips' inverse weights restored to the jointed scene's values),
gravity is off and only the thumb's pads collide with the tool, whose axis lies across the palmar face. `measure` applies
known forces and torques to the tool and writes what the page quotes:

    .venv/bin/python scripts/pad_contact_rig.py        # -> docs/experiments/20261013-contact_feedback/pad_laws.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fingertip_geometry as G  # noqa: E402
import reorient_backends as RB  # noqa: E402

SRC = ROOT / "assets/mjcf/real_v1/scenes/scene_screwdriver_medium.xml"
OUT = ROOT / "docs/experiments/20261013-contact_feedback/pad_laws.json"
SERVO_RUN = ROOT / "results/phase1/real_v1/20261011-sv1_u0308_b050_servo_tip_tpu2.7pads1"   # the servo-plant pad scene
TOOL, R_CYL, DENSITY = RB.OBJ, 0.0125, 500.0
FILLET = 0.0027                                  # the tpu2.7 tip
TC = RB.PAD_TR / 2.0                             # solref time constant of the pads
MU = 1.0                                         # pad friction; the tool's own 2.4 loses to the pads' priority


def realised_stiffness(m, finger: str) -> float:
    """Static force per metre of penetration of `finger`'s pads against the tool in compiled model `m`, N/m."""
    g = next(i for i in range(m.ngeom) if m.geom(i).name == f"{finger}_pad0")
    d0, tc = m.geom_solimp[g, 1], m.geom_solref[g, 0]
    A = m.body_invweight0[m.body(f"{finger}_tip").id, 0] + m.body_invweight0[m.body(TOOL).id, 0]
    return 1.0 / ((1.0 - d0) * tc * tc * A)


class Rig:
    """The thumb's pads against the tool. `mass_scale` scales the tool before the pads are built; `stale_scale`
    scales it after, so the pads keep the d0 solved for the unscaled tool."""

    def __init__(self, out: Path, mass_scale: float = 1.0, impratio: float = 10.0, stale_scale: float = 1.0,
                 E: float = RB.E_TPU):
        self.E = E
        root = ET.parse(SRC).getroot()
        geom = root.find(f".//body[@name='{TOOL}']/geom")
        geom.set("density", f"{DENSITY * mass_scale:g}")
        opt = root.find("option")
        opt.set("cone", "elliptic")
        opt.set("impratio", f"{impratio:g}")
        _, self.meta = RB.replace_tips(root, "tpu2.7", "pads", MU, out, tool=TOOL, tool_only=True, E=E)
        geom.set("density", f"{DENSITY * mass_scale * stale_scale:g}")
        self.jointed = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        for b in root.iter("body"):
            if b.get("name") != TOOL:
                for j in b.findall("joint"):
                    b.remove(j)
        for tag in ("actuator", "keyframe", "sensor", "equality", "tendon"):
            for e in root.findall(tag):
                root.remove(e)
        opt.set("gravity", "0 0 0")
        for b in [root.find("worldbody"), *root.iter("body")]:
            if b.get("name") == TOOL:
                continue
            for g in b.findall("geom"):
                if not g.get("name", "").startswith("thumb_pad"):
                    g.set("contype", "0")
                    g.set("conaffinity", "0")
        m = self.m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        for f in RB.FINGERS:
            m.body_invweight0[m.body(f"{f}_tip").id] = self.jointed.body_invweight0[self.jointed.body(f"{f}_tip").id]
        self.d = mujoco.MjData(m)
        self.tool, self.tip = m.body(TOOL).id, m.body("thumb_tip").id
        self.qadr = m.jnt_qposadr[m.body_jntadr[self.tool]]
        self.vadr = m.jnt_dofadr[m.body_jntadr[self.tool]]
        mujoco.mj_kinematics(m, self.d)
        self.R = self.d.xmat[self.tip].reshape(3, 3).copy()      # thumb tip frame; its +x is the palmar normal
        self.p = self.d.xpos[self.tip].copy()
        self.n, self.t = self.R[:, 0], self.R[:, 1]

    def press(self, F: float, steps: int = 300):
        """The tool touching the palmar face with its axis along the tip's y, then pushed onto it with F newtons."""
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        Rc = np.column_stack([self.R[:, 0], np.cross(self.R[:, 1], self.R[:, 0]), self.R[:, 1]])
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, Rc.ravel())
        d.qpos[self.qadr:self.qadr + 3] = self.p + self.R @ np.array([G.REACH + R_CYL, 0.0, G.CENTRE[2]])
        d.qpos[self.qadr + 3:self.qadr + 7] = q
        d.xfrc_applied[self.tool, :3] = -F * self.n
        self.run(steps)

    def load(self, F: float, Ft: float = 0.0, tau: float = 0.0):
        """Normal push F, tangential force Ft along the tool's axis acting at the contact line, torque tau about n."""
        w = self.d.xfrc_applied[self.tool]
        w[:3] = -F * self.n + Ft * self.t
        w[3:] = np.cross(-R_CYL * self.n, Ft * self.t) + tau * self.n

    def run(self, steps: int):
        for _ in range(steps):
            mujoco.mj_step(self.m, self.d)

    def contacts(self):
        """(normal force N, penetration m, contact point in the tip frame, contact normal in world) per active pad."""
        m, d = self.m, self.d
        f6, out = np.zeros(6), []
        for i in range(d.ncon):
            c = d.contact[i]
            pad = any(m.geom(int(g)).name.startswith("thumb_pad") for g in c.geom)
            if pad and c.dist < 0:
                mujoco.mj_contactForce(m, d, i, f6)
                out.append((f6[0], -c.dist, self.R.T @ (c.pos - self.p), c.frame[:3].copy()))
        return out

    def approach(self) -> float:
        """How far the tool has moved into the palmar face from first touch, m."""
        return G.REACH + R_CYL - float((self.R.T @ (self.d.xpos[self.tool] - self.p))[0])

    def twist_limit(self) -> float:
        """mu sum f_i rho_i: the torque at which every pad slides, about the normal through the patch centre, N m."""
        return MU * sum(f * math.hypot(p[1], p[2] - G.CENTRE[2]) for f, _, p, _ in self.contacts())

    def inverse_weight(self) -> float:
        """A of the thumb's pad rows against the tool, 1/kg."""
        return float(self.m.body_invweight0[self.tip, 0] + self.m.body_invweight0[self.tool, 0])

    def creep_law(self, Ft: float, n_active: int) -> float:
        """Steady tangential speed of the tool under Ft below the cone: R_t f_t = b v on every pad row, m/s."""
        d0 = self.meta["pad_d0"]["thumb"]
        Rt = (1 - d0) / d0 * self.inverse_weight() / self.m.opt.impratio
        return Ft * Rt / (2.0 / (d0 * TC) * n_active)

    def tool_speed(self) -> float:
        return float(self.d.qvel[self.vadr:self.vadr + 3] @ self.t)

    def turned_deg(self, q0) -> float:
        return 2 * math.degrees(math.acos(min(1.0, abs(float(q0 @ self.d.xquat[self.tool])))))


_P, _, _A = G.surface_samples(FILLET, 0.0001, 1.0)     # the block's surface at 0.1 mm


def continuum(approach: float, E: float = RB.E_TPU):
    """Normal force (N) and torque per unit friction (N m) of p = (E/h) overlap over the tip surface at `approach`."""
    cx = G.REACH + R_CYL - approach
    dx, dz = _P[:, 0] - cx, _P[:, 2] - G.CENTRE[2]
    r = np.hypot(dx, dz)
    p = E / G.FOUNDATION * np.maximum(R_CYL - r, 0.0)
    return float(np.sum(p * _A * -dx / np.maximum(r, 1e-12))), float(np.sum(p * _A * np.hypot(_P[:, 1], dz)))


def continuum_approach(F: float, E: float = RB.E_TPU) -> float:
    lo, hi = 0.0, 1e-3
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if continuum(mid, E)[0] < F else (lo, mid)
    return 0.5 * (lo + hi)


def servo_stiffness(run: Path = SERVO_RUN) -> dict:
    """Each fingertip's translational stiffness along its palmar normal through the finger servos (position gain kp per
    joint, N m/rad), at the CEM grasp command of `run`: 1 / (n^T J K^-1 J^T n), J the palmar-face centre's Jacobian."""
    m = mujoco.MjModel.from_xml_path(str(run / "frozen_scene.xml"))
    d = mujoco.MjData(m)
    summ = json.loads((run / "summary.json").read_text())
    d.qpos[:] = m.key_qpos[m.key(summ["keyframe"]).id]
    names = [f"{f}_{j}" for f in RB.FINGERS for j in ("yaw", "mcp", "pip")]
    for nm, v in zip(names, summ["best_finger_ctrl"]):
        d.qpos[m.jnt_qposadr[m.joint(nm).id]] = v
    mujoco.mj_forward(m, d)
    act = {m.joint(int(m.actuator_trnid[a, 0])).name: a for a in range(m.nu)}
    out = {}
    for f in RB.FINGERS:
        b = m.body(f"{f}_tip").id
        R = d.xmat[b].reshape(3, 3)
        sgn = G.FACE_SIGN[f]
        jp = np.zeros((3, m.nv))
        mujoco.mj_jac(m, d, jp, None, d.xpos[b] + R @ np.array([sgn * G.REACH, 0, G.CENTRE[2]]), b)
        J = jp[:, [m.jnt_dofadr[m.joint(f"{f}_{j}").id] for j in ("yaw", "mcp", "pip")]]
        kp = np.array([m.actuator_gainprm[act[f"{f}_{j}"], 0] for j in ("yaw", "mcp", "pip")])
        n = R[:, 0] * sgn
        out[f] = dict(kp=kp.tolist(), stiffness_N_m=float(1.0 / (n @ J @ np.diag(1 / kp) @ J.T @ n)),
                      lever_m=np.linalg.norm(J, axis=0).round(4).tolist())
    return dict(run=str(run.relative_to(ROOT)), fingers=out)


def measure(out: Path = OUT) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="pad_rig_"))
    t0 = time.perf_counter()
    base = Rig(tmp / "base")
    res = dict(pad_K=base.meta["pad_K"], pad_d0=base.meta["pad_d0"]["thumb"], E_Pa=RB.E_TPU, h_m=G.FOUNDATION,
               spacing_m=RB.PAD_S, pad_radius_m=RB.PAD_RS, relaxation_s=RB.PAD_TR, mu=MU,
               tool_mass_kg=float(base.m.body_mass[base.tool]), inverse_weight=base.inverse_weight(),
               tip_inverse_weight=float(base.m.body_invweight0[base.tip, 0]))
    static = []
    for F in (0.5, 1.0, 2.0, 3.0):
        base.press(F)
        c = base.contacts()
        k = [f / pen for f, pen, _, _ in c]
        a_ref = continuum_approach(F)
        static.append(dict(F=F, n_active=len(c), k_min=min(k), k_max=max(k),
                           push=sum(f * abs(float(nrm @ base.n)) for f, _, _, nrm in c),
                           approach_m=base.approach(), approach_law_m=a_ref, twist_Nm=base.twist_limit(),
                           twist_law_Nm=MU * continuum(a_ref)[1]))
    res["static"] = static
    creep = []
    for ir in (10.0, 100.0):
        r = base if ir == 10.0 else Rig(tmp / f"ir{ir:g}", impratio=ir)
        r.press(1.0)
        n_act = len(r.contacts())
        r.load(1.0, Ft=0.5 * MU)
        r.run(200)
        v = r.tool_speed()
        r.press(1.0)
        y0 = float(r.d.xpos[r.tool] @ r.t)
        r.load(1.0, Ft=1.2 * MU)
        r.run(50)
        creep.append(dict(impratio=ir, n_active=n_act, v_m_s=v, v_law_m_s=r.creep_law(0.5 * MU, n_act),
                          slide_m_in_0p1s=float(r.d.xpos[r.tool] @ r.t) - y0))
    res["creep"] = creep
    twist = []
    for frac in (0.5, 0.9, 1.2):
        base.press(1.0)
        tau = frac * base.twist_limit()
        q0 = base.d.xquat[base.tool].copy()
        base.load(1.0, tau=tau)
        base.run(100)
        twist.append(dict(frac=frac, tau_Nm=tau, turned_deg_in_0p2s=base.turned_deg(q0)))
    res["twist"] = twist
    mass = []
    for s in (0.25, 1.0, 4.0):
        r = base if s == 1.0 else Rig(tmp / f"m{s:g}", mass_scale=s)
        r.press(1.0)
        k = [f / pen for f, pen, _, _ in r.contacts()]
        r.load(1.0, Ft=0.5 * MU)
        r.run(200)
        row = dict(scale=s, tool_mass_kg=float(r.m.body_mass[r.tool]), d0=r.meta["pad_d0"]["thumb"],
                   k_rebuilt=float(np.median(k)), creep_m_s=r.tool_speed())
        if s != 1.0:
            st = Rig(tmp / f"stale{s:g}", stale_scale=s)
            st.press(1.0)
            row["k_stale"] = float(np.median([f / pen for f, pen, _, _ in st.contacts()]))
        mass.append(row)
    res["mass"] = mass
    sweep = []
    for E in (5e6, 7.3e6, 1e7, 1.46e7, 2e7):
        r = base if E == RB.E_TPU else Rig(tmp / f"E{E:g}", E=E)
        row = dict(E_Pa=E, pad_K=r.meta["pad_K"])
        for F in (0.5, 1.0, 1.5):
            r.press(F)
            row[f"approach_m_{F:g}N"] = r.approach()
            row[f"arm_m_{F:g}N"] = r.twist_limit() / (MU * F)
        r.press(1.0)
        a1 = r.approach()
        r.press(1.05)
        row["tangent_N_m_1N"] = 0.05 / (r.approach() - a1)
        sweep.append(row)
    res["modulus"] = sweep
    if (SERVO_RUN / "frozen_scene.xml").exists():
        res["servo"] = servo_stiffness()
    res.update(mujoco=mujoco.__version__, wall_s=round(time.perf_counter() - t0, 1), when=time.strftime("%Y-%m-%d %H:%M"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    res = measure(a.out)
    for s in res["static"]:
        print(f"{s['F']:g} N: {s['n_active']} pads, k {s['k_min']:.1f}-{s['k_max']:.1f} N/m, approach "
              f"{s['approach_m'] * 1e6:.1f} um (law {s['approach_law_m'] * 1e6:.1f}), twist {s['twist_Nm'] * 1e3:.3f} mN m "
              f"(law {s['twist_law_Nm'] * 1e3:.3f})")
    print(f"wrote {a.out} in {res['wall_s']} s")


if __name__ == "__main__":
    main()
