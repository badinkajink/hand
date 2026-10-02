#!/usr/bin/env python3
"""Relative contact-velocity control of arXiv 2609.25619 (Wang, Oh, Pollard) on the real_v1 hand.

The controller reads joint angles, joint velocities and the tool pose, builds a generalized contact
frame (GCF) per fingertip and returns joint-velocity references from the bounded least squares of the
paper's Eq. 2. It runs on a kinematic MuJoCo copy of the scene (`Mirror`), so one controller drives a
MuJoCo plant or a Drake plant (scripts/hom_chain.py).

GCF (paper Sec. III-A): origins at the witness points of the fingertip sphere and the tool cylinder,
x = the normal (gradient of the tool's signed distance at the tip centre, pointing tool -> finger),
y = the finger's flexion axis projected on the tangent plane, z = x cross y. For a sphere against a
cylinder both witness points and the normal are exact in closed form and stay smooth through contact,
so the paper's geodesic complementary filter on the hydroelastic centre of pressure (its Eqs. 4-6)
has nothing to smooth and is left out.

Relative contact twist (Eq. 1), the fingertip's witness point with respect to the tool's, in the GCF:
v_spin = w.x, v_roll = (w.y, w.z), v_sep = v.x, v_slide = (v.y, v.z). The paper writes object minus
finger; here it is finger minus object, so v_sep > 0 opens the gap.

Eq. 2: u = argmin sum_i w_i |v_i,ref - v_i(u)|^2 + w_damp |u|^2 + w_healthy |k_h (q_healthy - q) - u|^2
subject to q_min <= q + dt u <= q_max and |u| <= u_max. Linear rows are divided by ELL = 20 mm so that
20 mm/s weighs like 1 rad/s; w_healthy << w_damp << w_i as in the paper. Joint targets integrate u.

    PY=logs/20261001-hom_contact/venv/bin/python
    $PY scripts/hom_control.py exp1 --plant cal --rate 500 --film /tmp/exp1.mp4
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import hom_contact_rig as H  # noqa: E402
import hom_hand_brake as B  # noqa: E402

COMPONENTS = ("spin", "roll_y", "roll_z", "sep", "slide_y", "slide_z")
ANGULAR = (0, 1, 2)
LINEAR = (3, 4, 5)
ELL = 0.02                       # m; linear rows are scaled by 1/ELL
W_DAMP, W_HEALTHY, K_HEALTHY = 1e-3, 1e-5, 1.0
U_MAX = 2.0                      # rad/s
LEAD = 0.3                       # rad; joint targets stay within this of the measured angle
PLANTS = {
    # calibrated servo of the brake study (calfast without frictionloss)
    "cal": dict(kp=0.5, kv=0.02, forcerange=0.35, frictionloss=0.0, damping=0.0),
    # the template servo every pre-2026-09-16 result used
    "stiff": dict(kp=30.0, kv=0.5, forcerange=10.0, frictionloss=0.0, damping=0.5),
}


def flex_joint(f):
    return f"{f}_pip"


# ------------------------------------------------------------------------------------ geometry

def gcf_sphere_cylinder(c, r_tip, p_tool, R_tool, r_cyl, hl, flex_axis):
    """GCF between a fingertip sphere (centre c, radius r_tip) and a solid cylinder (centre p_tool,
    axis R_tool[:, 2], radius r_cyl, half-length hl). Returns dict(dist, n, p_fing, p_obj, E) with
    E = [x y z] as columns, x = n pointing from the tool to the finger."""
    a = R_tool[:, 2]
    v = c - p_tool
    s = float(v @ a)
    rad = v - s * a
    rho = float(np.linalg.norm(rad))
    if rho > 1e-12:
        e_r = rad / rho
    else:
        e_r = np.cross(a, [1.0, 0, 0])
        e_r /= np.linalg.norm(e_r)
    if rho <= r_cyl and abs(s) <= hl:                    # centre inside the solid (never in practice)
        if r_cyl - rho < hl - abs(s):
            n, q, depth = e_r, p_tool + s * a + r_cyl * e_r, r_cyl - rho
        else:
            n = a * math.copysign(1.0, s)
            q, depth = p_tool + math.copysign(hl, s) * a + rad, hl - abs(s)
        dist = -depth - r_tip
    else:
        q = p_tool + min(max(s, -hl), hl) * a + min(rho, r_cyl) * e_r
        dv = c - q
        dn = float(np.linalg.norm(dv))
        n = dv / dn
        dist = dn - r_tip
    y = flex_axis - (flex_axis @ n) * n
    ny = np.linalg.norm(y)
    if ny < 1e-6:
        y = np.cross(n, a)
        ny = np.linalg.norm(y)
    y /= ny
    z = np.cross(n, y)
    return {"dist": float(dist), "n": n, "p_fing": c - r_tip * n, "p_obj": q, "E": np.column_stack([n, y, z])}


# --------------------------------------------------------------------------------- the mirror

class Mirror:
    """Kinematic MuJoCo copy of a scene: joint state in, GCFs and twist Jacobians out."""

    def __init__(self, xml, fingers=B.FINGERS):
        import mujoco
        self.mj = mujoco
        self.m = mujoco.MjModel.from_xml_string(xml)
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.fingers = fingers
        self.fq = {f: np.array([m.jnt_qposadr[m.joint(n).id] for n in B.JOINTS[f]]) for f in fingers}
        self.fv = {f: np.array([m.jnt_dofadr[m.joint(n).id] for n in B.JOINTS[f]]) for f in fingers}
        self.lo = {f: np.array([m.jnt_range[m.joint(n).id][0] for n in B.JOINTS[f]]) for f in fingers}
        self.hi = {f: np.array([m.jnt_range[m.joint(n).id][1] for n in B.JOINTS[f]]) for f in fingers}
        self.tip = {f: m.body(f"{f}_tip").id for f in fingers}
        self.flex = {f: m.joint(flex_joint(f)).id for f in fingers}
        self.tool = m.body("tool").id
        self.tool_free = m.body_jntnum[self.tool] > 0
        tg = [g for g in range(m.ngeom) if m.geom_bodyid[g] == self.tool and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER]
        self.r_cyl, self.hl = float(m.geom_size[tg[0]][0]), float(m.geom_size[tg[0]][1])
        self.palm = [m.joint(n).id for n in ("palm_x", "palm_y", "palm_z", "palm_yaw") if _has_joint(m, n)]
        self.jacp, self.jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        self.jacpo, self.jacro = np.zeros((3, m.nv)), np.zeros((3, m.nv))

    def set_state(self, q_f, v_f, tool_pos=None, tool_quat=None, tool_v=None, tool_w_world=None, palm_q=None,
                  palm_v=None):
        """q_f, v_f: 9 finger joint angles / rates (thumb, index, middle x yaw, mcp, pip)."""
        m, d = self.m, self.d
        for i, f in enumerate(self.fingers):
            d.qpos[self.fq[f]] = q_f[3 * i:3 * i + 3]
            d.qvel[self.fv[f]] = v_f[3 * i:3 * i + 3]
        if self.palm and palm_q is not None:
            for j, q, v in zip(self.palm, palm_q, palm_v if palm_v is not None else np.zeros(4)):
                d.qpos[m.jnt_qposadr[j]] = q
                d.qvel[m.jnt_dofadr[j]] = v
        if self.tool_free and tool_pos is not None:
            ja = m.body_jntadr[self.tool]
            qa, va = m.jnt_qposadr[ja], m.jnt_dofadr[ja]
            d.qpos[qa:qa + 3] = tool_pos
            d.qpos[qa + 3:qa + 7] = tool_quat
            R9 = np.zeros(9)
            self.mj.mju_quat2Mat(R9, np.asarray(tool_quat, float))
            d.qvel[va:va + 3] = tool_v if tool_v is not None else 0.0
            d.qvel[va + 3:va + 6] = R9.reshape(3, 3).T @ tool_w_world if tool_w_world is not None else 0.0
        self.mj.mj_kinematics(m, d)
        self.mj.mj_comPos(m, d)

    def gravity(self):
        """Joint torques that hold the nine finger joints against their own weight at the current posture
        (qfrc_bias at zero velocity), for a feedforward offset tau_g / kp on the servo targets."""
        m, d = self.m, self.d
        qv = d.qvel.copy()
        d.qvel[:] = 0.0
        self.mj.mj_comVel(m, d)
        out = np.zeros(m.nv)
        self.mj.mj_rne(m, d, 0, out)
        d.qvel[:] = qv
        self.mj.mj_comVel(m, d)
        return np.concatenate([out[self.fv[f]] for f in self.fingers])

    def tool_pose(self):
        return self.d.xpos[self.tool].copy(), self.d.xmat[self.tool].reshape(3, 3).copy()

    def frame(self, f):
        p, R = self.tool_pose()
        return gcf_sphere_cylinder(self.d.xpos[self.tip[f]].copy(), B.R_TIP, p, R, self.r_cyl, self.hl,
                                   self.d.xaxis[self.flex[f]].copy())

    def twist(self, f, fr=None):
        """A (6x3), b (6): relative twist rows (spin, roll_y, roll_z, sep/ELL, slide_y/ELL, slide_z/ELL)
        of finger f's witness point w.r.t. the tool's = A u + b, u = finger f's joint rates; b carries
        the palm's and the tool's motion at the current qvel (finger f's own rates zeroed)."""
        m, d = self.m, self.d
        fr = fr or self.frame(f)
        self.mj.mj_jac(m, d, self.jacp, self.jacr, fr["p_fing"], self.tip[f])
        self.mj.mj_jac(m, d, self.jacpo, self.jacro, fr["p_obj"], self.tool)
        E = fr["E"]
        cols = self.fv[f]
        A = np.vstack([E.T @ self.jacr[:, cols], E.T @ self.jacp[:, cols] / ELL])
        qv = d.qvel.copy()
        qv[cols] = 0.0
        b = np.concatenate([E.T @ (self.jacr @ qv - self.jacro @ d.qvel),
                            E.T @ (self.jacp @ qv - self.jacpo @ d.qvel) / ELL])
        return A, b, fr

    def achieved(self, f, fr=None):
        """The relative twist at the mirror's full qvel (6, same scaling as `twist`)."""
        A, b, fr = self.twist(f, fr)
        return A @ self.d.qvel[self.fv[f]] + b, fr


def _has_joint(m, name):
    try:
        m.joint(name)
        return True
    except KeyError:
        return False


# --------------------------------------------------------------------------------------- Eq. 2

def solve_u(A, b, vref, w, q, lo, hi, dt, q_healthy=None, w_damp=W_DAMP, w_h=W_HEALTHY, k_h=K_HEALTHY,
            u_max=U_MAX):
    """Bounded least squares of Eq. 2 for one finger (3 joint rates)."""
    from scipy.optimize import lsq_linear
    sw = np.sqrt(np.asarray(w, float))
    rows = [sw[:, None] * A, math.sqrt(w_damp) * np.eye(3)]
    rhs = [sw * (vref - b), np.zeros(3)]
    if q_healthy is not None and w_h > 0:
        rows.append(math.sqrt(w_h) * np.eye(3))
        rhs.append(math.sqrt(w_h) * k_h * (q_healthy - q))
    lb = np.maximum(-u_max, (lo - q) / dt)
    ub = np.minimum(u_max, (hi - q) / dt)
    lb, ub = np.minimum(lb, ub - 1e-9), np.maximum(ub, lb + 1e-9)
    return lsq_linear(np.vstack(rows), np.concatenate(rhs), bounds=(lb, ub), method="bvls").x


# ---------------------------------------------------------------------------- Exp 1 of the paper

EXP1_SEQ = ("sep", "slide_y", "slide_z", "spin", "roll_y", "roll_z")
STEP_ANG, STEP_LIN = math.radians(40.0), 0.020          # the paper's 40 deg/s and 20 mm/s


def exp1_scene(plant, gap):
    """D8 hand (palm fixed, calibrated or template servo), a fixed standing cylinder whose axis passes
    through the pinch point; thumb and index pre-shaped `gap` from its surface."""
    import mujoco
    saved = dict(B.PLANT)
    B.PLANT.update(PLANTS[plant])
    try:
        off = B.d8_offsets()
        root = B.hand_tree(off)
    finally:
        B.PLANT.clear()
        B.PLANT.update(saved)
    import hom_chain as K                      # the chain's pinch line: 45 deg off the mount line, 50 mm deep
    P, u, a = K.pinch_frame()
    P = P + np.array([0.0, 0.0, B.PALM_Z])
    hl = 0.040
    zc = P[2] - 0.020
    wb = root.find("worldbody")
    tool = ET.SubElement(wb, "body", name="tool", pos=f"{P[0]:.6f} {P[1]:.6f} {zc:.6f}")
    ET.SubElement(tool, "geom", name="tool", type="cylinder", size=f"{H.R_TOOL} {hl}", rgba="0.55 0.6 0.68 1")
    asset = root.find("asset")
    ET.SubElement(asset, "texture", name="sky", type="skybox", builtin="gradient", rgb1="1 1 1", rgb2=".86 .89 .93",
                  width="64", height="64")
    g = root.find("visual").find("global")
    g.set("offwidth", "960")
    g.set("offheight", "720")
    xml = ET.tostring(root, encoding="unicode")
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    r = H.R_TOOL + B.R_TIP + gap
    q = {}
    q["thumb"], _ = B.solve_tip(m, d, "thumb", P - r * u, np.array([-0.2, 0.35, 0.8]))
    q["index"], _ = B.solve_tip(m, d, "index", P + r * u, np.array([0.2, 0.35, 0.8]))
    q["middle"] = np.array(K.MIDDLE_PARK)
    return xml, np.concatenate([q[f] for f in B.FINGERS]), (P, u, a)


def exp1(plant="cal", rate=500.0, T_step=0.5, T_rest=0.3, gap=0.008, film=None, mover="index"):
    """Finger `mover` moves around the standing cylinder without contact; each scalar element of its
    reference relative contact velocity is stepped in turn (+ for T_step, - for T_step, rest T_rest)."""
    import mujoco
    t0w = time.time()
    xml, q0, (P, u, a) = exp1_scene(plant, gap)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mir = Mirror(xml)
    ja = [m.joint(n).id for f in B.FINGERS for n in B.JOINTS[f]]
    qa = np.array([m.jnt_qposadr[j] for j in ja])
    va = np.array([m.jnt_dofadr[j] for j in ja])
    d.qpos[qa] = q0
    d.ctrl[:] = q0
    mujoco.mj_forward(m, d)
    fi = B.FINGERS.index(mover)
    sl = slice(3 * fi, 3 * fi + 3)
    q_ref = q0.copy()
    q_healthy = q0[sl].copy()
    dt_c = 1.0 / rate
    n_sub = max(1, int(round(dt_c / m.opt.timestep)))
    sched = []
    for k, comp in enumerate(EXP1_SEQ):
        mag = STEP_ANG if COMPONENTS.index(comp) in ANGULAR else STEP_LIN / ELL
        sched += [(comp, +mag, T_step), (comp, -mag, T_step), (comp, 0.0, T_rest)]
    T_tot = sum(s[2] for s in sched) + 0.2
    rows, frames = [], []
    renderer = Exp1Renderer(m, P, u) if film else None
    t_edges = np.cumsum([0.2] + [s[2] for s in sched])
    n_ctrl = int(round(T_tot / dt_c))
    min_gap = 1.0
    ncon_max = 0
    for k in range(n_ctrl):
        t = d.time
        i = int(np.searchsorted(t_edges, t, side="right")) - 1
        vref = np.zeros(6)
        label = "rest"
        if 0 <= i < len(sched):
            comp, val, _ = sched[i]
            vref[COMPONENTS.index(comp)] = val
            label = f"{comp} {'+' if val > 0 else '-' if val < 0 else ''}"
        mir.set_state(d.qpos[qa], d.qvel[va])
        A, b, fr = mir.twist(mover)
        q_meas = d.qpos[qa][sl]
        uu = solve_u(A, b, vref, np.ones(6), q_meas, mir.lo[mover], mir.hi[mover], dt_c, q_healthy)
        q_ref[sl] = np.clip(q_ref[sl] + uu * dt_c, q_meas - LEAD, q_meas + LEAD)
        q_ref[sl] = np.clip(q_ref[sl], mir.lo[mover], mir.hi[mover])
        d.ctrl[:] = q_ref
        for _ in range(n_sub):
            mujoco.mj_step(m, d)
        ncon_max = max(ncon_max, d.ncon)
        # achieved relative twist after the step, at the new state
        mir.set_state(d.qpos[qa], d.qvel[va])
        ach, fr2 = mir.achieved(mover)
        min_gap = min(min_gap, fr2["dist"])
        rows.append([round(float(d.time), 4), *vref.tolist(), *ach.tolist(), fr2["dist"], *uu.tolist()])
        if renderer is not None and k % max(1, int(round(rate / 25))) == 0:
            frames.append(renderer.frame(d, mir, mover, f"Exp 1 on D8, {plant} servo, {rate:g} Hz",
                                         f"step {label:10s} gap {fr2['dist'] * 1e3:5.1f} mm  t {d.time:4.2f} s"))
    tr = np.array(rows)
    ref, ach = tr[:, 1:7], tr[:, 7:13]
    scale = np.array([180 / math.pi] * 3 + [ELL * 1e3] * 3)       # deg/s and mm/s
    err = (ach - ref) * scale
    res = {"exp": "exp1", "plant": plant, "rate": rate, "gap_mm": gap * 1e3, "mover": mover,
           "T_step": T_step, "rmse_all": np.sqrt((err ** 2).mean(0)).tolist(),
           "rmse_ang_dps": float(np.sqrt((err[:, :3] ** 2).sum(1).mean())),
           "rmse_lin_mmps": float(np.sqrt((err[:, 3:] ** 2).sum(1).mean())),
           "min_gap_mm": min_gap * 1e3, "ncon_max": int(ncon_max), "wall_s": time.time() - t0w,
           "per_step": {}}
    for comp in EXP1_SEQ:
        k = COMPONENTS.index(comp)
        on = np.abs(ref[:, k]) > 0
        e = err[on]
        res["per_step"][comp] = {"rmse_ang_dps": float(np.sqrt((e[:, :3] ** 2).sum(1).mean())),
                                 "rmse_lin_mmps": float(np.sqrt((e[:, 3:] ** 2).sum(1).mean())),
                                 "commanded_rmse": float(np.sqrt((e[:, k] ** 2).mean())),
                                 "commanded_gain": float(np.median(ach[on, k] / ref[on, k]))}
    res["trace_cols"] = ["t"] + [f"ref_{c}" for c in COMPONENTS] + [f"ach_{c}" for c in COMPONENTS] + ["gap", "u0", "u1", "u2"]
    res["trace"] = tr[::max(1, int(round(rate / 100)))].round(6).tolist()
    if film:
        H.write_mp4(frames, film, fps=25)
        res["film"] = str(film)
    return res


class Exp1Renderer:
    def __init__(self, m, P, u, w=480, h=360):
        import mujoco
        self.mj = mujoco
        self.m = m
        self.r = mujoco.Renderer(m, h, w)
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = [P[0], P[1], P[2]]
        self.cam.distance = 0.20
        self.cam.azimuth = math.degrees(math.atan2(u[1], u[0])) + 90.0
        self.cam.elevation = -25.0

    def frame(self, d, mir, f, title, line):
        self.r.update_scene(d, self.cam)
        for g in ("thumb", "index"):
            fr = mir.frame(g)
            draw_gcf(self.r.scene, fr)
        return H.annotate_bottom(H.annotate(self.r.render().copy(), title), line)


def add_connector(scn, kind, p0, p1, width, rgba):
    import mujoco
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, kind, np.zeros(3), np.zeros(3), np.eye(3).reshape(-1), np.asarray(rgba, np.float32))
    mujoco.mjv_connector(g, kind, width, np.asarray(p0, float), np.asarray(p1, float))
    scn.ngeom += 1


def add_sphere(scn, p, r, rgba):
    import mujoco
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([r, r, r]), np.asarray(p, float),
                        np.eye(3).reshape(-1), np.asarray(rgba, np.float32))
    scn.ngeom += 1


def draw_gcf(scn, fr, L=0.012, w=0.0007):
    import mujoco
    cols = ((0.85, 0.15, 0.15, 1), (0.15, 0.65, 0.2, 1), (0.15, 0.3, 0.85, 1))
    for p in (fr["p_obj"], fr["p_fing"]):
        for k in range(3):
            add_connector(scn, mujoco.mjtGeom.mjGEOM_ARROW, p, p + L * fr["E"][:, k], w, cols[k])
    add_connector(scn, mujoco.mjtGeom.mjGEOM_LINE, fr["p_obj"], fr["p_fing"], 2.0, (0.1, 0.1, 0.1, 1))


# ------------------------------------------------------------------------------------------ CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("exp1")
    e.add_argument("--plant", default="cal", choices=sorted(PLANTS))
    e.add_argument("--rate", type=float, default=500.0)
    e.add_argument("--gap", type=float, default=8.0, help="mm")
    e.add_argument("--film")
    e.add_argument("--rows")
    args = ap.parse_args()
    if args.cmd == "exp1":
        r = exp1(args.plant, args.rate, gap=args.gap * 1e-3, film=args.film)
        if args.rows:
            H.append_row(args.rows, r)
        print(json.dumps({k: v for k, v in r.items() if k not in ("trace",)}, default=H._json_default, indent=1))


if __name__ == "__main__":
    main()
