#!/usr/bin/env python3
"""The open-loop geometric carry, drawn: from the fitted grasp to the hold pose, on one hand.

    MUJOCO_GL=egl uv run python scripts/real_v1_carry_construction.py --plan sv1_w2360_b075
    MUJOCO_GL=egl uv run python scripts/real_v1_carry_construction.py --plan g12_b095 --video

The deployed controller (paper Sec. II.C, `real_v1_deploy_envelope.make_plan`) is three finger
set-points and two ramps. This script rebuilds one shipped plan from its own metadata, keeps
every intermediate the construction passes through, and draws them on MuJoCo renders of the
plan's own scene:

  grasp        the CEM/IK grip, settled in physics on the bench post; the three pad centres,
               their centroid C and the straddle 2S between the index and middle pads
  construction the pivot P = C + k S z (k = axis_k), the rotation axis through P along the
               pinch direction (world X, end-on in this view), each pad's arc through the
               angle theta about P, and the shaft's intended pose after the turn
  hold pose    per-finger IK to the rotated pad targets, with the grasp as a ghost; the joint
               deltas are what the plan ships, clipped to +-b per joint
  execution    the joint-space ramp grip -> hold replayed in physics: pad trails against the
               construction arcs, the shaft's signed alignment, the commanded joint deltas
               with the clip band

The Y-Z view is orthographic and looks along the rotation axis, so the arcs are true circles
and every length in it is to scale. Nothing is drawn that the plan does not compute.

Outputs, in docs/experiments/20260921-carry_construction/:
  20260921-carry_construction_<D>_<plan>.png/.pdf      all three rows
  ..._construction.png   ..._execution.png   ..._schedule.png      one row each, for slides
  ..._.json              the numbers on the figure
  ..._.mp4               with --video: front view + perspective + the schedule with a cursor
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import mujoco  # noqa: E402

import probe_real_v1_carry as pc  # noqa: E402
import real_v1_deploy_envelope as de  # noqa: E402
from morphohand.tools.keyframe_ik import FINGERS, TIPS, ik_finger  # noqa: E402
from real_v1_plan_band import _depth_req  # noqa: E402
from real_v1_transfer_ranking import DESIGN_ID  # noqa: E402

OUT = ROOT / "docs/experiments/20260921-carry_construction"
DT = 0.002                      # sim step; 10 steps = one 50 Hz control step
REC = 10                        # record every control step
CLOSE_STEPS, SETTLE_STEPS, HOLD_STEPS = 250, 400, 800

#: finger colours follow the paper's Fig. 1c (thumb red, index blue, middle green); the
#: construction is amber, the only colour that belongs to no finger.
COL = {"thumb": "#C24A3F", "index": "#3F74BF", "middle": "#4E8F52"}
AMBER, INK, MUTE = "#BE7514", "#1A2127", "#77848D"
OBJECT_RGBA = (0.24, 0.26, 0.30, 1.0)
PAPER = (0.965, 0.970, 0.976)
ORDER = ("index", "thumb", "middle")
SHORT = {"thumb": "T", "index": "I", "middle": "M"}
FS = {"scale": 1.0}             # font scale for the schedule plots; the video draws them larger


def _fs(x):
    return x * FS["scale"]


def _hex(h, a=1.0):
    return tuple(int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)) + (a,)


# --------------------------------------------------------------------------------------------
# the construction, with its intermediates kept
# --------------------------------------------------------------------------------------------
def find_plan(tag: str) -> Path:
    hits = sorted(ROOT.glob(f"docs/experiments/*/deploy/{tag}_plan.json"))
    if not hits:
        sys.exit(f"no shipped plan named {tag}")
    # the plan with a recorded budget is the exported one; older copies lack it
    hits.sort(key=lambda p: json.loads(p.read_text())["meta"].get("budget_rad") is None)
    return hits[0]


class Recorder:
    """One control-step record per REC sim steps: pose, command, pads, object, contacts."""

    def __init__(self, m, d, acts):
        self.m, self.d, self.acts, self.rec, self.t = m, d, acts, [], 0.0

    def snap(self, phase, u=None):
        m, d = self.m, self.d
        n, f = pc._contacts_hand(m, d, de.OBJ)
        self.rec.append(dict(t=self.t, phase=phase, u=u, qpos=d.qpos.copy(),
                             ctrl={j: float(d.ctrl[a]) for j, a in self.acts.items()},
                             tips={k: d.body(TIPS[k]).xpos.copy() for k in FINGERS},
                             obj=d.body(de.OBJ).xpos.copy(),
                             axis=d.body(de.OBJ).xmat.reshape(3, 3)[:, 2].copy(),
                             cos=pc._cos(m, d, de.OBJ), ncon=n, force=f))

    def run(self, n, phase, before=None):
        for k in range(n):
            if before is not None:
                before(k)
            mujoco.mj_step(self.m, self.d)
            self.t += DT
            if (k + 1) % REC == 0:
                self.snap(phase)


def _carry_from_state(m, d, scene, acts, anchor, R_, *, axis_k, angle_deg, budget,
                      turn_steps, hold_steps, q_open, depth_mm=float("nan"),
                      hold_squeeze=0.0, squeeze_steps=200) -> dict:
    """The construction and the execution, from a settled grasp. Verbatim `make_plan` /
    `carry(linear_anchor=True)`: pads -> centroid -> raised pivot -> rotated targets ->
    per-finger IK -> joint deltas, then the clipped joint-space ramp and the hold."""
    rec = R_.rec
    tip0 = {k: d.body(TIPS[k]).xpos.copy() for k in FINGERS}
    centroid = np.mean([tip0[k] for k in FINGERS], axis=0)
    span = abs(tip0["index"][1] - tip0["middle"][1]) / 2.0
    pivot = centroid.copy()
    pivot[2] += axis_k * span
    q0 = {j: float(d.qpos[m.jnt_qposadr[m.joint(j).id]]) for j in acts}
    R = pc._rotx(np.radians(angle_deg))
    targets = {k: pivot + R @ (tip0[k] - pivot) for k in FINGERS}
    mik = mujoco.MjModel.from_xml_path(str(scene))
    dik = mujoco.MjData(mik)
    dik.qpos[:] = d.qpos
    dik.qvel[:] = 0.0
    mujoco.mj_forward(mik, dik)
    resid = {k: ik_finger(mik, dik, k, targets[k], iters=400) for k in FINGERS}
    end = {j: float(dik.qpos[mik.jnt_qposadr[mik.joint(j).id]]) for j in acts}
    reached = {k: dik.body(TIPS[k]).xpos.copy() for k in FINGERS}
    q_hold = dik.qpos.copy()
    delta = {j: end[j] - q0[j] for j in acts}
    clipped = {j: float(np.clip(delta[j], -budget, budget)) for j in acts}
    obj_grip = (d.body(de.OBJ).xpos.copy(), d.body(de.OBJ).xmat.reshape(3, 3).copy())
    q_grip = d.qpos.copy()
    t_turn0 = R_.t
    # the hold pose the servos are actually sent: anchor + clipped delta, kinematically
    dik.qpos[:] = d.qpos
    for j in acts:
        dik.qpos[mik.jnt_qposadr[mik.joint(j).id]] = q0[j] + clipped[j]
    mujoco.mj_forward(mik, dik)
    q_sent = dik.qpos.copy()
    sent_tips = {k: dik.body(TIPS[k]).xpos.copy() for k in FINGERS}

    # execution: the ramp, then the hold, on the same live state
    for k in range(1, turn_steps + 1):
        u = k / turn_steps
        for j, a in acts.items():
            d.ctrl[a] = anchor[j] + float(np.clip(delta[j] * u, -budget, budget))
        mujoco.mj_step(m, d)
        R_.t += DT
        if k % REC == 0:
            R_.snap("turn", u)
    # THE RE-SQUEEZE (`make_plan --hold-squeeze`, `carry(hold_squeeze=)`): position servos hold
    # whatever commanded-minus-actual error is left, and after the turn that is a fraction of a
    # newton. One more set-point puts each pad `hold_squeeze` back inside the rotated shaft's
    # surface, solved from the live end-of-turn state and ramped over `squeeze_steps`.
    sq_delta = None
    if hold_squeeze > 0.0:
        o = d.body(de.OBJ).xpos.copy()
        ax = d.body(de.OBJ).xmat.reshape(3, 3)[:, 2]
        dik.qpos[:] = d.qpos
        dik.qvel[:] = 0.0
        mujoco.mj_forward(mik, dik)
        for f in FINGERS:
            tp = d.body(TIPS[f]).xpos.copy()
            v = (tp - o) - float((tp - o) @ ax) * ax
            n = float(np.linalg.norm(v))
            if n > 1e-6:
                ik_finger(mik, dik, f, tp - (v / n) * hold_squeeze, iters=200)
        sq_end = {j: float(dik.qpos[mik.jnt_qposadr[mik.joint(j).id]]) for j in acts}
        sq_delta = {j: sq_end[j] - q0[j] for j in acts}
        sq_start = {j: float(d.ctrl[a]) for j, a in acts.items()}
        for k in range(1, squeeze_steps + 1):
            u = k / squeeze_steps
            for j, a in acts.items():
                tgt = anchor[j] + sq_delta[j]
                d.ctrl[a] = float(np.clip(sq_start[j] + (tgt - sq_start[j]) * u,
                                          anchor[j] - budget, anchor[j] + budget))
            mujoco.mj_step(m, d)
            R_.t += DT
            if k % REC == 0:
                R_.snap("squeeze", 1.0)
    R_.run(hold_steps, "hold")
    for r in rec:
        if r["phase"] == "hold":
            r["u"] = 1.0
    return dict(m=m, scene=scene, acts=acts, rec=rec, tip0=tip0, centroid=centroid, span=span,
                pivot=pivot, R=R, theta=float(angle_deg), k=float(axis_k),
                targets=targets, reached=reached, resid=resid, q_grip=q_grip, q_hold=q_hold,
                q_sent=q_sent, sent_tips=sent_tips, q_open=q_open,
                q0=q0, end=end, anchor=anchor, delta=delta, clipped=clipped, budget=budget,
                obj_grip=obj_grip, depth_mm=float(depth_mm), turn_s=turn_steps * DT,
                hold_s=hold_steps * DT, t_turn0=t_turn0, hold_squeeze=hold_squeeze,
                squeeze_s=(squeeze_steps * DT if hold_squeeze > 0 else 0.0), sq_delta=sq_delta)


def construct(meta: dict, design: str, budget: float, scene: Path | None = None,
              turn_steps: int | None = None, hold_steps: int = HOLD_STEPS,
              close_ramp: bool = False) -> dict:
    """A shipped bench plan: `make_plan`, step for step. Fixed palm, tool on the post.
    `scene` replaces the plan's own (shipped-plant) scene, e.g. with its bench-calibrated
    rewrite from `apply_measured_plant.py`; the construction is then run on that plant."""
    scene = Path(meta["scene"]) if scene is None else Path(scene)
    built = pc._grip_from_fit(scene, meta["straddle_mm"] / 1000, 0.0, meta["squeeze_mm"] / 1000,
                              de.OBJ, _depth_req(meta, design), meta["thumb_axial_mm"] / 1000)
    if built is None:
        sys.exit("the fitter found no reachable grasp for this plan")
    m, open_qpos, grip, depth_mm = built
    d = mujoco.MjData(m)
    d.qpos[:] = open_qpos
    d.qvel[:] = 0.0
    d.ctrl[:] = grip
    acts = pc._finger_act(m)
    anchor = {j: float(grip[a]) for j, a in acts.items()}
    # the CB1 ramps open -> grip over the plan's 0.5 s (poses[grip].ramp_s) and holds 0.8 s;
    # `make_plan` steps to the grip and settles 650 steps. `close_ramp` picks the former.
    q_open_j = {j: float(open_qpos[m.jnt_qposadr[m.joint(j).id]]) for j in acts}
    if close_ramp:
        for j, a in acts.items():
            d.ctrl[a] = q_open_j[j]
    mujoco.mj_forward(m, d)
    R_ = Recorder(m, d, acts)
    R_.snap("open")
    if close_ramp:
        def ramp(k):
            u = (k + 1) / CLOSE_STEPS
            for j, a in acts.items():
                d.ctrl[a] = q_open_j[j] + (anchor[j] - q_open_j[j]) * u
        R_.run(CLOSE_STEPS, "close", ramp)
    else:
        R_.run(CLOSE_STEPS, "close")
    R_.run(SETTLE_STEPS, "settle")
    return _carry_from_state(m, d, scene, acts, anchor, R_, axis_k=float(meta["axis_k"]),
                             angle_deg=float(meta["angle_deg"]), budget=budget,
                             turn_steps=int(turn_steps or meta["turn_steps"]),
                             hold_steps=hold_steps, q_open=open_qpos.copy(),
                             depth_mm=float(depth_mm))


def construct_morph(run: Path, *, lift: float, axis_k: float, angle_deg: float, budget: float,
                    turn_steps: int, hold_steps: int, jitter: float = 0.0, seed: int = 0,
                    scene: Path | None = None, hold_squeeze: float = 0.0) -> dict:
    """A CEM morphology run's own grip on its frozen scene, lifted off the floor and turned in
    the air: `probe_real_v1_carry.carry(..., linear_anchor=True)`, step for step. `scene`
    substitutes a rewrite of the run's scene (same bodies and keyframes, another plant)."""
    scene = run / "frozen_scene.xml" if scene is None else Path(scene)
    m = pc._load_model(scene)
    d = mujoco.MjData(m)
    key = m.key("open_ik").id
    mujoco.mj_resetDataKeyframe(m, d, key)
    d.ctrl[:] = m.key_ctrl[key]
    if jitter > 0.0:                       # the carry probe's spawn jitter, on the object's xy
        rng = np.random.default_rng(seed)
        adr = int(m.jnt_qposadr[m.body(de.OBJ).jntadr[0]])
        d.qpos[adr + 0] += float(rng.normal(0.0, jitter))
        d.qpos[adr + 1] += float(rng.normal(0.0, jitter))
    q_open = d.qpos.copy()
    closed = np.load(run / "best_rollout.npz")["best_finger_ctrl"]
    anchor = {j: float(closed[i * 3 + k])
              for i, (f, js) in enumerate(FINGERS.items()) for k, j in enumerate(js)}
    acts = pc._finger_act(m)
    pz_a = next(k for k in range(m.nu) if m.actuator(k).name == "a_palm_pz")
    for j, a in acts.items():
        d.ctrl[a] = anchor[j]
    mujoco.mj_forward(m, d)
    R_ = Recorder(m, d, acts)
    R_.snap("open")
    R_.run(250, "close")
    pz0 = float(d.ctrl[pz_a])
    R_.run(200, "lift", lambda k: d.ctrl.__setitem__(pz_a, pz0 + lift * (k + 1) / 200))
    R_.run(200, "settle")
    C = _carry_from_state(m, d, scene, acts, anchor, R_, axis_k=axis_k, angle_deg=angle_deg,
                          budget=budget, turn_steps=turn_steps, hold_steps=hold_steps,
                          q_open=q_open, hold_squeeze=hold_squeeze)
    C["lift"] = lift
    return C


# --------------------------------------------------------------------------------------------
# rendering: a studio version of the plan's scene, plus a world -> pixel projection
# --------------------------------------------------------------------------------------------
class Studio:
    def __init__(self, m: mujoco.MjModel, width: int, height: int):
        self.m, self.W, self.H = m, width, height
        self.finger_geoms = {f: [] for f in FINGERS}
        for g in range(m.ngeom):
            b = m.body(m.geom_bodyid[g]).name
            rgba = None
            for f in FINGERS:
                if b.startswith(f + "_"):
                    rgba = _hex(COL[f])
                    self.finger_geoms[f].append(g)
            if b == de.OBJ:
                rgba = OBJECT_RGBA
            if b.startswith("palm"):
                rgba = (0.40, 0.43, 0.47, 1.0)
            if m.geom(g).name == "floor":
                rgba = (0.905, 0.915, 0.925, 1.0)     # a plain ground; shadows are off
            if m.geom(g).name == "tool_post":
                rgba = (0.66, 0.69, 0.73, 1.0)
            if rgba is not None:
                m.geom_rgba[g] = rgba
                m.geom_matid[g] = -1                  # the material's texture would win
        m.site_rgba[:, 3] = 0.0
        for tx in range(m.ntex):
            if m.tex_type[tx] == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
                adr = m.tex_adr[tx]
                n = m.tex_width[tx] * m.tex_height[tx] * m.tex_nchannel[tx]
                m.tex_data[adr:adr + n] = np.tile(
                    np.array([int(c * 255) for c in PAPER], dtype=np.uint8), n // 3)
        m.vis.global_.offwidth = max(m.vis.global_.offwidth, width)
        m.vis.global_.offheight = max(m.vis.global_.offheight, height)
        m.vis.map.znear = 0.005
        m.vis.headlight.ambient[:] = 0.42
        m.vis.headlight.diffuse[:] = 0.58
        m.vis.headlight.specular[:] = 0.30
        self.ren = mujoco.Renderer(m, height=height, width=width)
        self.d = mujoco.MjData(m)
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(m, self.cam)
        self.opt = mujoco.MjvOption()
        self.opt.sitegroup[:] = 0

    def view(self, name: str, lookat, height_m: float = 0.165):
        """Free-camera presets. The free camera is orthographic only when the MODEL says so
        (`vis.global.orthographic`), and then `fovy` is the vertical extent in metres."""
        c, g = self.cam, self.m.vis.global_
        c.lookat[:] = lookat
        if name == "front":            # along the rotation axis: the Y-Z plane, to scale
            g.orthographic, g.fovy = 1, height_m
            c.azimuth, c.elevation, c.distance = 180, 0, 0.5
        elif name == "iso":
            g.orthographic, g.fovy = 0, 30
            c.azimuth, c.elevation, c.distance = 135, -10, 0.31
        elif name == "side":
            g.orthographic, g.fovy = 1, height_m
            c.azimuth, c.elevation, c.distance = 90, 0, 0.5
        return self

    def render(self, qpos, extras=()) -> np.ndarray:
        self.d.qpos[:] = qpos
        self.d.qvel[:] = 0.0
        mujoco.mj_forward(self.m, self.d)
        self.ren.update_scene(self.d, self.cam, self.opt)
        scn = self.ren.scene
        scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        scn.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 1
        for fn in extras:
            fn(scn)
        return self.ren.render().copy()

    def project(self, p) -> tuple[float, float]:
        """World point -> pixel (u right, v down) for the camera of the last render.
        Mirrors mjr_render: the two GL cameras are averaged, and the horizontal extent is
        the vertical one scaled by the viewport aspect."""
        c0, c1 = self.ren.scene.camera[0], self.ren.scene.camera[1]
        pos = (np.array(c0.pos) + np.array(c1.pos)) / 2
        fwd = np.array(c0.forward) + np.array(c1.forward)
        fwd /= np.linalg.norm(fwd)
        up = np.array(c0.up) + np.array(c1.up)
        up /= np.linalg.norm(up)
        right = np.cross(fwd, up)
        dd = np.asarray(p, float) - pos
        x, y, z = dd @ right, dd @ up, dd @ fwd
        top, bot = c0.frustum_top, c0.frustum_bottom
        center = (c0.frustum_center + c1.frustum_center) / 2
        half_h = (top - bot) / 2
        half_w = half_h * self.W / self.H
        if not c0.orthographic:
            x, y = x * c0.frustum_near / z, y * c0.frustum_near / z
        return (self.W / 2 + (x - center) / half_w * self.W / 2,
                self.H / 2 - (y - (top + bot) / 2) / half_h * self.H / 2)

    def ghost(self, qpos, alpha=0.30, fingers=FINGERS):
        """The fingers at another pose, as translucent copies of their own geoms."""
        dg = mujoco.MjData(self.m)
        dg.qpos[:] = qpos
        mujoco.mj_forward(self.m, dg)
        items = []
        for f in fingers:
            for g in self.finger_geoms[f]:
                items.append((int(self.m.geom_type[g]), self.m.geom_size[g].copy(),
                              dg.geom_xpos[g].copy(), dg.geom_xmat[g].copy(),
                              _hex(COL[f], alpha)))

        def fn(scn):
            for typ, size, pos, mat, rgba in items:
                geom = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(geom, typ, size, pos, mat, np.array(rgba, np.float32))
                scn.ngeom += 1
        return fn

    # ---- extra geoms drawn inside the MuJoCo pass, so they shade and occlude properly ----
    @staticmethod
    def cylinder(pos, mat, radius, half_len, rgba):
        def fn(scn):
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CYLINDER,
                                np.array([radius, half_len, 0.0]), np.asarray(pos, float),
                                np.asarray(mat, float).flatten(), np.array(rgba, np.float32))
            scn.ngeom += 1
        return fn

    @staticmethod
    def sphere(pos, radius, rgba):
        def fn(scn):
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius, 0.0, 0.0]),
                                np.asarray(pos, float), np.eye(3).flatten(),
                                np.array(rgba, np.float32))
            scn.ngeom += 1
        return fn

    @staticmethod
    def line(a, b, width, rgba):
        def fn(scn):
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                                np.eye(3).flatten(), np.array(rgba, np.float32))
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width,
                                 np.asarray(a, float), np.asarray(b, float))
            scn.ngeom += 1
        return fn


def object_geom(m):
    """(radius, half-length) of the tool's cylinder."""
    bid = m.body(de.OBJ).id
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == bid and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER:
            return float(m.geom_size[g][0]), float(m.geom_size[g][1])
    raise KeyError("tool cylinder")


def ghost_tool(C, alpha=0.22):
    """The shaft where the construction sends it: the grip pose rotated by theta about P."""
    r, hl = object_geom(C["m"])
    pos, mat = C["obj_grip"]
    return Studio.cylinder(C["pivot"] + C["R"] @ (pos - C["pivot"]), C["R"] @ mat, r, hl,
                           OBJECT_RGBA[:3] + (alpha,))


def palm_z(C):
    """Height of the palm plate's underside at the grip."""
    m = C["m"]
    d = mujoco.MjData(m)
    d.qpos[:] = C["q_grip"]
    mujoco.mj_forward(m, d)
    return float(d.body("palm_pose").xpos[2])


def frame_at(C, tips=None):
    """Where the orthographic front view looks: the pads' y, and 15 mm above the pads, which
    keeps the palm plate, the pads, their targets and the standing shaft inside a 165 mm
    window. With `tips` (a record's) the frame follows the hand through a lift."""
    c = C["centroid"] if tips is None else np.mean([tips[f] for f in FINGERS], axis=0)
    return np.array([C["centroid"][0], C["centroid"][1], c[2] + 0.015])


def arc_points(C, finger, n=60, frac=1.0):
    th = np.radians(C["theta"]) * frac
    P, v = C["pivot"], C["tip0"][finger] - C["pivot"]
    return np.array([P + pc._rotx(a) @ v for a in np.linspace(0.0, th, n)])


# --------------------------------------------------------------------------------------------
# panels
# --------------------------------------------------------------------------------------------
def _panel(ax, img):
    ax.imshow(img, interpolation="lanczos")
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def _title(ax, letter, text, size=11):
    ax.set_title(f"({letter})  {text}", loc="left", fontsize=size, pad=7, color=INK)


def _dot(ax, uv, color, size=52, hollow=False, z=6, lw=1.7):
    ax.scatter([uv[0]], [uv[1]], s=size, facecolors="white" if hollow else color,
               edgecolors=color, linewidths=lw, zorder=z)


def _label(ax, uv, text, color, dx=10, dy=0, size=10, ha="left", va="center", weight="bold",
           box=True):
    ax.text(uv[0] + dx, uv[1] + dy, text, fontsize=size, color=color, ha=ha, va=va,
            weight=weight, zorder=9,
            bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.4) if box else None)


def _dim(ax, a, b, text, color=INK, offset=(0, 0), fontsize=9, ha="center", va="center"):
    """A dimension line between two pixel points with a label beside its middle."""
    ax.annotate("", xy=b, xytext=a, zorder=6,
                arrowprops=dict(arrowstyle="<->", color=color, lw=1.0, shrinkA=0, shrinkB=0))
    mid = ((a[0] + b[0]) / 2 + offset[0], (a[1] + b[1]) / 2 + offset[1])
    ax.text(*mid, text, fontsize=fontsize, color=color, ha=ha, va=va, zorder=9,
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.4))


def _pivot(ax, P, size=90, label=True):
    _dot(ax, P, AMBER, size=size, z=8)
    ax.scatter([P[0]], [P[1]], s=size * 0.11, color="white", zorder=9)
    if label:
        _label(ax, P, "P", AMBER, dx=13, dy=-13, size=11)


def draw_grasp(ax, st, C):
    st.view("front", frame_at(C))
    img = st.render(C["q_grip"])
    _panel(ax, img)
    for f in FINGERS:
        uv = st.project(C["tip0"][f])
        _dot(ax, uv, COL[f])
    c = st.project(C["centroid"])
    _dot(ax, c, INK, size=34)
    _label(ax, c, "C", INK, dx=12, dy=12, size=11)
    i, mm = st.project(C["tip0"]["index"]), st.project(C["tip0"]["middle"])
    y = max(i[1], mm[1]) + 120
    for f in ("index", "middle"):
        p = st.project(C["tip0"][f])
        ax.plot([p[0], p[0]], [p[1] + 12, y + 6], color=MUTE, lw=0.7, ls=":", zorder=5)
    _dim(ax, (i[0], y), (mm[0], y), f"2S = {2 * C['span'] * 1000:.0f} mm", offset=(0, 18))
    _label(ax, st.project(C["tip0"]["index"]), "index pad", COL["index"], dx=14, dy=-16,
           size=9, weight="normal")
    _label(ax, st.project(C["tip0"]["middle"]), "middle pad", COL["middle"], dx=-14, dy=-16,
           size=9, ha="right", weight="normal")
    _label(ax, st.project(C["tip0"]["thumb"]), "thumb pad\n(behind the shaft)", COL["thumb"],
           dx=0, dy=30, size=9, ha="center", weight="normal")
    plate = st.project(C["m"].body("palm_pose").subtree_com if False else
                       [C["centroid"][0], C["centroid"][1] + 0.07, palm_z(C)])
    _label(ax, plate, "palm plate", MUTE, dx=0, dy=-14, size=8.5, ha="center", weight="normal",
           box=False)
    rec = C["rec"]
    s_end = [r for r in rec if r["phase"] == "settle"][-1]
    dz = (s_end["obj"][2] - rec[0]["obj"][2]) * 1000
    tilt = np.degrees(np.arcsin(np.clip(abs(s_end["cos"]), 0, 1)))
    if C.get("lift"):
        first = (f"CEM grasp from the morphology run, palm lifted {C['lift'] * 1000:.0f} mm\n"
                 f"after the lift: ")
    else:
        first = (f"grip depth {C['depth_mm']:.1f} mm, pads driven 10 mm into the shaft\n"
                 f"after the close: ")
    ax.text(0.02, 0.03, first + f"{s_end['ncon']} hand contacts, {s_end['force']:.1f} N; "
            f"shaft {dz:+.0f} mm, tilted {tilt:.0f}°",
            transform=ax.transAxes, fontsize=8.5, color=INK, va="bottom",
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=2))


def draw_construction(ax, st, C):
    st.view("front", frame_at(C))
    img = st.render(C["q_grip"], [ghost_tool(C)])
    _panel(ax, img)
    P = st.project(C["pivot"])
    c = st.project(C["centroid"])
    for f in FINGERS:
        uv = np.array([st.project(p) for p in arc_points(C, f)])
        ax.plot(uv[:, 0], uv[:, 1], color=COL[f], lw=1.8, ls=(0, (4, 3)), zorder=5)
        for q in (C["tip0"][f], C["targets"][f]):
            uvq = st.project(q)
            ax.plot([P[0], uvq[0]], [P[1], uvq[1]], color=COL[f], lw=0.7, alpha=0.6, zorder=4)
        _dot(ax, st.project(C["tip0"][f]), COL[f])
        _dot(ax, st.project(C["targets"][f]), COL[f], hollow=True)
    _dot(ax, c, INK, size=30)
    _label(ax, c, "C", INK, dx=16, dy=8, size=10)
    _pivot(ax, P, label=False)
    _label(ax, P, "P", AMBER, dx=16, dy=-12, size=11)
    # the pivot's lift above the centroid: a bracket beside the two points, and the arithmetic
    # in the free corner under the index finger, with a leader back to P
    import matplotlib.patheffects as pe
    x = c[0] - 34
    halo = [pe.withStroke(linewidth=3.2, foreground="white")]
    ax.plot([x, x], [c[1], P[1]], color=AMBER, lw=1.3, zorder=7, path_effects=halo)
    for q in (c, P):
        ax.plot([x - 6, x + 6], [q[1], q[1]], color=AMBER, lw=1.3, zorder=7, path_effects=halo)
    ax.text(0.02, 0.96, f"P = C + k S ẑ   (the bracket)\nk S = {C['k']:.2f} × "
            f"{C['span'] * 1000:.0f} mm = {C['k'] * C['span'] * 1000:.1f} mm",
            transform=ax.transAxes, fontsize=9, color=AMBER, ha="left", va="top", zorder=9,
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=2))
    # theta, on the longer of the two pair arcs, labelled on the outside of the arc
    f = max(("index", "middle"), key=lambda k: np.linalg.norm(C["tip0"][k] - C["pivot"]))
    mid = np.array(st.project(arc_points(C, f, n=3, frac=0.5)[1]))
    outward = mid - np.array(P)
    outward /= max(np.linalg.norm(outward), 1e-9)
    ax.annotate(f"θ = {C['theta']:.0f}°", xy=mid, xytext=mid + outward * 95,
                fontsize=11.5, color=COL[f], ha="center", va="center", weight="bold", zorder=9,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.6),
                arrowprops=dict(arrowstyle="-", color=COL[f], lw=0.8, shrinkB=4))
    for f in FINGERS:
        t = np.array(st.project(C["targets"][f]))
        if f == "thumb":                       # T′ sits a few mm from T, C and P: keep it left
            _label(ax, t, "T′", COL[f], dx=-16, dy=6, size=10.5, ha="right")
            continue
        away = t - np.array(P)
        away /= max(np.linalg.norm(away), 1e-9)
        pos = t + away * 24
        _label(ax, pos, f"{SHORT[f]}′", COL[f], dx=0, dy=0, size=10.5, ha="center")
    # the intended shaft pose
    gp = C["pivot"] + C["R"] @ (C["obj_grip"][0] - C["pivot"])
    gax = C["R"] @ C["obj_grip"][1][:, 2]
    r, hl = object_geom(C["m"])
    tip = st.project(gp + gax * hl * (1 if gax[2] < 0 else -1))
    _label(ax, tip, "shaft after the turn\n(intended)", MUTE, dx=26, dy=-6, size=8.5,
           weight="normal")
    ax.text(0.02, 0.03, "rotation axis through P, along x (into the page)",
            transform=ax.transAxes, fontsize=8.5, color=AMBER, va="bottom",
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=2))


def draw_hold(ax, st, C):
    st.view("front", frame_at(C))
    img = st.render(C["q_sent"], [st.ghost(C["q_grip"], 0.26)])
    _panel(ax, img)
    P = st.project(C["pivot"])
    for f in FINGERS:
        uv = np.array([st.project(p) for p in arc_points(C, f)])
        ax.plot(uv[:, 0], uv[:, 1], color=COL[f], lw=1.0, ls=(0, (4, 3)), alpha=0.55, zorder=4)
        t = np.array(st.project(C["targets"][f]))
        _dot(ax, t, COL[f], hollow=True)
        s = np.array(st.project(C["sent_tips"][f]))
        _dot(ax, s, COL[f], size=52)
        miss = np.hypot(*(s - t))
        if miss > 8:
            ax.annotate("", xy=s, xytext=t, zorder=7,
                        arrowprops=dict(arrowstyle="->", color=COL[f], lw=1.2, shrinkA=6,
                                        shrinkB=6))
            mid = (s + t) / 2
            side = np.array([-(s - t)[1], (s - t)[0]])
            side /= max(np.linalg.norm(side), 1e-9)
            short = np.linalg.norm(C["sent_tips"][f] - C["targets"][f]) * 1000
            far = C["resid"][f] * 1000 > 6.0
            clip = any(abs(C["delta"][j]) > C["budget"] for j in FINGERS[f])
            why = ("out of reach" if far and not clip else "clipped at ±b" if clip and not far
                   else "out of reach, clipped")
            _label(ax, mid + side * 22, f"{short:.0f} mm short: {why}", COL[f], dx=0, dy=0,
                   size=8.5, ha="center", weight="normal")
    _pivot(ax, P, size=60, label=False)
    ax.text(0.02, 0.96, "solid: the hold pose the servos are sent\nghost: the grasp",
            transform=ax.transAxes, fontsize=8.5, color=INK, va="top",
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=2))
    lines = []
    for f in ORDER:
        js = FINGERS[f]
        dd = [np.degrees(C["delta"][j]) for j in js]
        clip = ["*" if abs(C["delta"][j]) > C["budget"] else "" for j in js]
        lines.append(f"{f:6s} Δ yaw {dd[0]:+5.0f}°{clip[0]}  mcp {dd[1]:+5.0f}°{clip[1]}  "
                     f"pip {dd[2]:+5.0f}°{clip[2]}   IK miss {C['resid'][f] * 1000:4.1f} mm")
    ax.text(0.02, 0.03, "\n".join(lines) + f"\n* clipped to ±b = ±{np.degrees(C['budget']):.0f}°",
            transform=ax.transAxes, fontsize=8, color=INK, va="bottom", family="DejaVu Sans Mono",
            bbox=dict(fc="white", ec="none", alpha=0.88, pad=2.5))


def draw_perspective(ax, st, C):
    st.view("iso", C["centroid"] + np.array([0, 0, 0.02]))
    ax_len = 0.075
    xhat = np.array([1.0, 0, 0])
    extras = [ghost_tool(C),
              Studio.line(C["pivot"] - ax_len * xhat, C["pivot"] + ax_len * xhat, 0.0012,
                          _hex(AMBER)),
              Studio.sphere(C["pivot"], 0.004, _hex(AMBER))]
    for f in FINGERS:
        pts = arc_points(C, f, n=40)
        for a, b in zip(pts[:-1], pts[1:]):
            extras.append(Studio.line(a, b, 0.0014, _hex(COL[f])))
        extras.append(Studio.sphere(C["targets"][f], 0.0042, _hex(COL[f], 0.9)))
        extras.append(Studio.sphere(C["tip0"][f], 0.0042, _hex(COL[f])))
    img = st.render(C["q_grip"], extras)
    _panel(ax, img)
    P = st.project(C["pivot"])
    _label(ax, P, "P", AMBER, dx=12, dy=-12, size=11)
    e = st.project(C["pivot"] - ax_len * xhat)
    _label(ax, e, "rotation axis ∥ x", AMBER, dx=-8, dy=-6, size=9, weight="normal", ha="right")


def draw_execution(axes, st, C, n_turn=4):
    rec = C["rec"]
    turn = [r for r in rec if r["phase"] == "turn"]
    hold = [r for r in rec if r["phase"] == "hold"]
    us = np.linspace(0, 1, n_turn)
    picks = [turn[int(round(u * (len(turn) - 1)))] for u in us]
    picks.append(hold[len(hold) // 2])
    picks.append(hold[-1])
    labels = [f"turn ramp {r['u'] * 100:.0f} %  ({r['t'] - C['t_turn0']:.2f} s)" for r in picks[:-2]]
    labels += [f"hold  +{(picks[-2]['t'] - C['t_turn0'] - C['turn_s']):.1f} s",
               f"hold  +{(picks[-1]['t'] - C['t_turn0'] - C['turn_s']):.1f} s  (end)"]
    st.view("front", frame_at(C))
    i0 = rec.index(turn[0])
    for ax, r, lab in zip(axes, picks, labels):
        img = st.render(r["qpos"], [ghost_tool(C, 0.10)])
        _panel(ax, img)
        i_r = rec.index(r)
        for f in FINGERS:
            uv = np.array([st.project(p) for p in arc_points(C, f)])
            ax.plot(uv[:, 0], uv[:, 1], color=COL[f], lw=1.0, ls=(0, (3, 3)), alpha=0.55, zorder=4)
            trail = np.array([st.project(rr["tips"][f]) for rr in rec[i0:i_r + 1]])
            if len(trail) > 1:
                ax.plot(trail[:, 0], trail[:, 1], color=COL[f], lw=2.4, zorder=5,
                        solid_capstyle="round")
            _dot(ax, st.project(r["tips"][f]), COL[f], size=36)
            _dot(ax, st.project(C["targets"][f]), COL[f], hollow=True, size=30, lw=1.2)
        _pivot(ax, st.project(C["pivot"]), size=44, label=False)
        ax.set_title(lab, fontsize=9.5, pad=4, color=INK, loc="left")
        ax.text(0.97, 0.03, f"cos {r['cos']:+.2f}\n{r['ncon']} contacts, {r['force']:.1f} N",
                transform=ax.transAxes, fontsize=9, color=INK, ha="right", va="bottom",
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=2))


def draw_schedule(ax_q, ax_c, C, cursor=None, legend=True):
    rec, acts = C["rec"], C["acts"]
    t0 = C["t_turn0"]
    ts = np.array([r["t"] - t0 for r in rec])
    b = np.degrees(C["budget"])
    ax_q.axhspan(-b, b, color=AMBER, alpha=0.08, lw=0)
    for s in (b, -b):
        ax_q.axhline(s, color=AMBER, lw=0.9, ls="--")
    ax_q.text(ts[-1] - 0.02, b, f"+b = {b:.0f}°", fontsize=_fs(8.5), color=AMBER, va="bottom",
              ha="right")
    ax_q.text(ts[-1] - 0.02, -b, f"−b", fontsize=_fs(8.5), color=AMBER, va="top", ha="right")
    styles = {"yaw": "-", "mcp": (0, (5, 2)), "pip": (0, (1.5, 1.5))}
    for f in ORDER:
        for j in FINGERS[f]:
            cmd = np.degrees(np.array([r["ctrl"][j] for r in rec]) - C["anchor"][j])
            ax_q.plot(ts, cmd, color=COL[f], lw=1.7, ls=styles[j.rpartition("_")[2]])
    clipped = [(j, f) for f in ORDER for j in FINGERS[f] if abs(C["delta"][j]) > C["budget"]]
    for i, (j, f) in enumerate(sorted(clipped, key=lambda jf: abs(C["delta"][jf[0]]),
                                      reverse=True)):
        t_sat = C["budget"] / abs(C["delta"][j]) * C["turn_s"]
        y = np.sign(C["delta"][j]) * b
        ax_q.plot([t_sat], [y], marker="o", ms=_fs(4), color=COL[f], zorder=6)
        ax_q.text(C["turn_s"] + 0.03 * (ts[-1] - C["turn_s"]), b * (0.62 - 0.2 * i),
                  f"{j.replace('_', ' ')} asks {np.degrees(C['delta'][j]):+.0f}°, "
                  f"clipped from {t_sat:.2f} s", fontsize=_fs(8.5), color=COL[f], va="center",
                  ha="left")
    for a in (ax_q, ax_c):
        a.axvspan(0, C["turn_s"], color=INK, alpha=0.05, lw=0)
        a.set_xlim(ts[0], ts[-1])
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.tick_params(labelsize=_fs(8.5))
    ax_q.set_ylabel("commanded joint delta from the grip (°)", fontsize=_fs(9))
    ax_q.set_ylim(-b * 1.25, b * 1.25)
    ytop = b * 1.12
    ax_q.text(-t0 / 2, ytop, "close, lift, settle" if any(r["phase"] == "lift" for r in rec)
              else "close, settle", fontsize=_fs(8.5), ha="center", color=MUTE)
    ax_q.text(C["turn_s"] / 2, ytop, "turn ramp", fontsize=_fs(8.5), ha="center", color=INK)
    ax_q.text(C["turn_s"] + C["hold_s"] / 2, ytop, "hold", fontsize=_fs(8.5), ha="center",
              color=MUTE)
    if legend:
        from matplotlib.lines import Line2D
        hs = [Line2D([], [], color=COL[f], lw=2.2, label=f) for f in ORDER]
        hs += [Line2D([], [], color=INK, lw=1.4, ls=styles[k], label=k) for k in styles]
        ax_q.legend(handles=hs, fontsize=_fs(8), ncol=2, frameon=False, loc="lower left")
    ax_q.set_xticklabels([])
    cos = np.array([r["cos"] for r in rec])
    fz = np.array([r["force"] for r in rec])
    ax_c.plot(ts, cos, color=INK, lw=2.0)
    ax_c.axhline(0, color=MUTE, lw=0.6)
    ax_c.set_ylim(-0.12, 1.08)
    ax_c.set_ylabel("cos(shaft, vertical)", fontsize=_fs(9))
    ax_c.set_xlabel("time from the start of the turn (s)", fontsize=_fs(9))
    ax_f = ax_c.twinx()
    ax_f.plot(ts, fz, color=AMBER, lw=1.2, alpha=0.9)
    ax_f.set_ylabel("hand contact force (N)", fontsize=_fs(9), color=AMBER)
    ax_f.tick_params(axis="y", labelsize=_fs(8.5), colors=AMBER)
    ax_f.set_ylim(0, max(1.0, fz.max() * 1.2))
    ax_f.spines["top"].set_visible(False)
    end = rec[-1]
    ax_c.text(ts[-1] - 0.02, end["cos"] + 0.07, f"cos {end['cos']:+.2f} at the end of the hold",
              fontsize=_fs(8.5), ha="right", color=INK)
    if cursor is not None:
        for a in (ax_q, ax_c):
            a.axvline(cursor, color=INK, lw=1.2)
    return ax_f


# --------------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------------
def _rc():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42,
                         "figure.facecolor": "white", "axes.edgecolor": MUTE,
                         "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})
    return plt


PLANT_NAMES = {"shipped": "shipped plant (kp 30)", "corrected": "bench-calibrated plant (kp 0.5, kv 0.6)",
               "fast": "bench-calibrated plant (kp 0.5, kv 0.02)"}


def _who(C, did, tag):
    if C.get("lift"):
        return f"{did} ({tag.split('_k')[0]}), CEM grasp, floor-free"
    plan = tag.split("_corrected")[0].split("_fast")[0]
    if C.get("plant", "shipped") == "shipped":
        return f"{did}, deployed plan {plan}, {PLANT_NAMES['shipped']}"
    return f"{did}, the construction of {plan} run on the {PLANT_NAMES[C['plant']]}"


def fig_construction(C, st, plt, head):
    fig, axs = plt.subplots(1, 4, figsize=(18, 5.6))
    fig.subplots_adjust(left=0.006, right=0.994, top=0.86, bottom=0.01, wspace=0.03)
    draw_grasp(axs[0], st, C)
    _title(axs[0], "a", "grasp: the CEM grip, closed, lifted and settled" if C.get("lift")
           else "grasp: the fitted grip, closed and settled")
    draw_construction(axs[1], st, C)
    _title(axs[1], "b", "construction: rotate the three pads by θ about P")
    draw_hold(axs[2], st, C)
    _title(axs[2], "c", "hold pose: per-finger IK to I′ T′ M′, clipped to ±b")
    draw_perspective(axs[3], st, C)
    _title(axs[3], "d", "the same construction in perspective")
    fig.suptitle(head, fontsize=13, x=0.006, ha="left", y=0.975, color=INK)
    return fig


def fig_execution(C, st, plt, head):
    fig, axs = plt.subplots(1, 6, figsize=(18, 3.9))
    fig.subplots_adjust(left=0.006, right=0.994, top=0.82, bottom=0.01, wspace=0.03)
    draw_execution(axs, st, C)
    fig.suptitle(head, fontsize=13, x=0.006, ha="left", y=0.975, color=INK)
    return fig


def fig_schedule(C, plt, head):
    fig, (ax_q, ax_c) = plt.subplots(2, 1, figsize=(18, 5.2), sharex=False,
                                     gridspec_kw=dict(height_ratios=[1.15, 1.0], hspace=0.08))
    fig.subplots_adjust(left=0.05, right=0.955, top=0.86, bottom=0.11)
    draw_schedule(ax_q, ax_c, C)
    fig.suptitle(head, fontsize=13, x=0.006, ha="left", y=0.975, color=INK)
    return fig


def figures(C, did, tag, stem: Path, dpi=200):
    plt = _rc()
    st = Studio(C["m"], 1200, 1000)
    pars = (f"k = {C['k']:.2f}, θ = {C['theta']:.0f}°, b = {C['budget']:.2f} rad "
            f"({np.degrees(C['budget']):.0f}°), turn {C['turn_s']:.1f} s")
    who = _who(C, did, tag)
    h1 = f"Open-loop geometric carry on {who}: construction.  {pars}"
    h2 = (f"Open-loop geometric carry on {who}: the joint-space ramp replayed in physics "
          f"(one nominal rollout)")
    h3 = f"Open-loop geometric carry on {who}: what the servos are sent and what the shaft does"
    for name, mk in (("construction", lambda: fig_construction(C, st, plt, h1)),
                     ("execution", lambda: fig_execution(C, st, plt, h2)),
                     ("schedule", lambda: fig_schedule(C, plt, h3))):
        fig = mk()
        fig.savefig(f"{stem}_{name}.png", dpi=dpi)
        fig.savefig(f"{stem}_{name}.pdf")
        plt.close(fig)
    # the three rows as one sheet
    from PIL import Image
    rows = [Image.open(f"{stem}_{n}.png") for n in ("construction", "execution", "schedule")]
    W = max(r.width for r in rows)
    sheet = Image.new("RGB", (W, sum(r.height for r in rows)), "white")
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height
    sheet.save(f"{stem}.png")
    st.ren.close()


# --------------------------------------------------------------------------------------------
# the video
# --------------------------------------------------------------------------------------------
def video(C, did, tag, out: Path, speed=0.5, fps=30):
    plt = _rc()
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    rec = C["rec"]
    st = Studio(C["m"], 1200, 1000)
    st_iso = Studio(mujoco.MjModel.from_xml_path(str(C["scene"])), 1200, 900)
    t0 = C["t_turn0"]
    ts = np.array([r["t"] for r in rec])
    lead = 0.5                                  # seconds of the open pose before anything moves
    T = ts[-1] + lead * speed
    n = int(round(T / speed * fps)) + 1
    W, H = 1920, 1080
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    canvas = FigureCanvasAgg(fig)
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba",
         "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "slow",
         "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
        stdin=subprocess.PIPE)
    i_turn0 = next(i for i, r in enumerate(rec) if r["phase"] == "turn")
    pars = (f"k = {C['k']:.2f}, θ = {C['theta']:.0f}°, b = {C['budget']:.2f} rad, "
            f"turn {C['turn_s']:.1f} s")
    who = _who(C, did, tag)
    xhat = np.array([1.0, 0, 0])
    FS["scale"] = 1.45
    for k in range(n):
        t = k / fps * speed - lead * speed
        i = int(np.searchsorted(ts, max(t, 0.0), side="right") - 1)
        r = rec[max(i, 0)]
        show_geo = r["phase"] in ("settle", "turn", "hold")
        fig.clf()
        gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.333], height_ratios=[1.0, 0.42],
                              left=0.008, right=0.992, top=0.935, bottom=0.075, wspace=0.02,
                              hspace=0.16)
        ax = fig.add_subplot(gs[0, 0])
        st.view("front", frame_at(C, r["tips"] if r["phase"] in ("open", "close", "lift")
                                  else None))
        img = st.render(r["qpos"], [ghost_tool(C, 0.14)] if show_geo else [])
        _panel(ax, img)
        if show_geo:
            for f in FINGERS:
                uv = np.array([st.project(p) for p in arc_points(C, f)])
                ax.plot(uv[:, 0], uv[:, 1], color=COL[f], lw=1.4, ls=(0, (4, 3)), alpha=0.75,
                        zorder=4)
                _dot(ax, st.project(C["targets"][f]), COL[f], hollow=True, size=44)
                if r["phase"] in ("turn", "hold"):
                    trail = np.array([st.project(rr["tips"][f]) for rr in rec[i_turn0:i + 1]])
                    if len(trail) > 1:
                        ax.plot(trail[:, 0], trail[:, 1], color=COL[f], lw=3.0, zorder=5,
                                solid_capstyle="round")
                _dot(ax, st.project(r["tips"][f]), COL[f], size=56)
            _pivot(ax, st.project(C["pivot"]), size=90)
        phase = {"open": "open pose", "close": "close onto the tool", "lift": "lift",
                 "settle": "settle", "turn": "turn ramp", "hold": "hold"}[r["phase"]]
        if r["phase"] == "turn":
            phase += f"  {r['u'] * 100:.0f} %"
        ax.text(0.02, 0.975, phase, transform=ax.transAxes, fontsize=20, color=INK, va="top",
                weight="bold")
        ax.text(0.02, 0.915, f"{r['t'] - t0:+.2f} s from the start of the turn   "
                f"({speed:g}× speed)", transform=ax.transAxes, fontsize=12.5, color=MUTE,
                va="top")
        ax.text(0.98, 0.03, f"cos {r['cos']:+.2f}", transform=ax.transAxes, fontsize=30,
                color=INK, ha="right", va="bottom", weight="bold")
        ax.text(0.98, 0.135, f"{r['ncon']} contacts, {r['force']:.1f} N",
                transform=ax.transAxes, fontsize=13, color=MUTE, ha="right", va="bottom")
        ax2 = fig.add_subplot(gs[0, 1])
        st_iso.view("iso", (C["centroid"] if show_geo else
                            np.mean([r["tips"][f] for f in FINGERS], axis=0)) + np.array([0, 0, 0.02]))
        extras = []
        if show_geo:
            extras = [ghost_tool(C, 0.14),
                      Studio.line(C["pivot"] - 0.075 * xhat, C["pivot"] + 0.075 * xhat, 0.0012,
                                  _hex(AMBER)), Studio.sphere(C["pivot"], 0.004, _hex(AMBER))]
            for f in FINGERS:
                pts = arc_points(C, f, n=40)
                for a, b in zip(pts[:-1], pts[1:]):
                    extras.append(Studio.line(a, b, 0.0014, _hex(COL[f])))
                extras.append(Studio.sphere(C["targets"][f], 0.0042, _hex(COL[f], 0.9)))
        _panel(ax2, st_iso.render(r["qpos"], extras))
        ax2.text(0.02, 0.975, "perspective: rotation axis through P in amber, pad arcs, "
                 "intended shaft pose", transform=ax2.transAxes, fontsize=12.5, color=MUTE,
                 va="top")
        sub = gs[1, :].subgridspec(1, 2, wspace=0.16, width_ratios=[1.0, 1.0])
        ax_q = fig.add_subplot(sub[0, 0])
        ax_c = fig.add_subplot(sub[0, 1])
        for a_, dx in ((ax_q, 0.035), (ax_c, 0.0)):       # room for the axis labels
            b_ = a_.get_position()
            a_.set_position([b_.x0 + dx, b_.y0, b_.width - 0.035, b_.height])
        draw_schedule(ax_q, ax_c, C, cursor=max(t, ts[0]) - t0, legend=False)
        ax_q.set_xlabel("time from the start of the turn (s)", fontsize=_fs(9))
        ax_q.set_xticklabels([f"{x:g}" for x in ax_q.get_xticks()])
        ax_q.set_title("commanded joint delta (yaw —, mcp – –, pip ···), clip ±b in amber",
                       loc="left", fontsize=13, color=INK)
        ax_c.set_title("shaft alignment (black) and hand contact force (amber)", loc="left",
                       fontsize=13, color=INK)
        fig.suptitle(f"Open-loop geometric carry on {who}.  {pars}", fontsize=17,
                     x=0.008, ha="left", y=0.985, color=INK)
        canvas.draw()
        ff.stdin.write(np.asarray(canvas.buffer_rgba()).tobytes())
    ff.stdin.close()
    ff.wait()
    plt.close(fig)
    FS["scale"] = 1.0
    st.ren.close()
    st_iso.ren.close()


DEPLOYED = [("D1", "sv1_w6689_b060"), ("D2", "sv1_w2360_b075"), ("D3", "sv1_u1364_b080"),
            ("D4", "g12_b095"), ("D5", "sv1_u0060_b75"), ("D6", "sv1_u0308_b050"),
            ("D7", "rv05_manual_b85"), ("D8", "sv1_w0099_b100")]


def fig_array(out_dir: Path, dpi=200):
    """Panel (b) for each of the eight bench hands, in the order of the bench video array."""
    plt = _rc()
    fig, axs = plt.subplots(2, 4, figsize=(18, 8.6))
    fig.subplots_adjust(left=0.006, right=0.994, top=0.90, bottom=0.01, wspace=0.03,
                        hspace=0.16)
    rows = []
    for ax, (did, tag) in zip(axs.ravel(), DEPLOYED):
        shipped = json.loads(find_plan(tag).read_text())
        meta = shipped["meta"]
        C = construct(meta, shipped["design"], float(meta["budget_rad"]))
        st = Studio(C["m"], 1200, 1000)
        draw_construction(ax, st, C)
        st.ren.close()
        end = C["rec"][-1]
        ax.set_title(f"{did}  {shipped['design']}   k = {C['k']:.2f}, θ = {C['theta']:.0f}°, "
                     f"b = {C['budget']:.2f} rad", loc="left", fontsize=10.5, pad=6, color=INK)
        rows.append(dict(did=did, plan=tag, k=C["k"], theta=C["theta"], budget=C["budget"],
                         span_mm=C["span"] * 1000, ik_residual_mm={f: C["resid"][f] * 1000
                                                                    for f in FINGERS},
                         final_cos=end["cos"], final_force_N=end["force"]))
    fig.suptitle("The same construction on the eight bench hands: pads (filled), targets "
                 "(hollow), pivot P, arcs through θ, and the intended shaft pose",
                 fontsize=13, x=0.006, ha="left", y=0.97, color=INK)
    stem = out_dir / "20260921-carry_construction_eight_hands"
    fig.savefig(f"{stem}.png", dpi=dpi)
    fig.savefig(f"{stem}.pdf")
    plt.close(fig)
    Path(f"{stem}.json").write_text(json.dumps(rows, indent=1))
    print(f"wrote {stem}.png / .pdf / .json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="sv1_w2360_b075")
    ap.add_argument("--plant", default="shipped", choices=["shipped", "corrected", "fast"],
                    help="which plant the shipped plan's construction runs on: the plan's own "
                         "scene (kp 30), or its bench-calibrated rewrite (kp 0.5, kv 0.6 / 0.02) "
                         "from docs/experiments/20260916-turn_mechanism/scenes")
    ap.add_argument("--close-ramp", action="store_true",
                    help="ramp open -> grip over the plan's 0.5 s instead of stepping to it")
    ap.add_argument("--morph-run", type=Path, default=None,
                    help="a CEM morphology run (its frozen scene and grip) instead of a shipped "
                         "plan; the carry is then floor-free, lifted by --lift")
    ap.add_argument("--lift", type=float, default=0.10)
    ap.add_argument("--axis-k", type=float, default=0.25)
    ap.add_argument("--angle-deg", type=float, default=-90.0)
    ap.add_argument("--turn-steps", type=int, default=800)
    ap.add_argument("--hold-steps", type=int, default=500)
    ap.add_argument("--label", default=None, help="hand label on the figures (default: the run's design id)")
    ap.add_argument("--all", action="store_true",
                    help="also the 2x4 array of the construction on the eight bench hands")
    ap.add_argument("--budget", type=float, default=None, help="override the plan's clip")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--speed", type=float, default=0.5)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    if a.all:
        fig_array(a.out_dir)
    if a.morph_run is not None:
        budget = a.budget if a.budget is not None else 0.5
        design = a.morph_run.name
        base = design.replace("_stored", "")
        did = a.label or DESIGN_ID.get(base, base)
        C = construct_morph(a.morph_run, lift=a.lift, axis_k=a.axis_k, angle_deg=a.angle_deg,
                            budget=budget, turn_steps=a.turn_steps, hold_steps=a.hold_steps)
        tag = f"{design}_k{a.axis_k:g}_b{budget:g}_a{abs(a.angle_deg):g}_t{a.turn_steps}"
        p = a.morph_run
    else:
        p = find_plan(a.plan)
        shipped = json.loads(p.read_text())
        meta, design = shipped["meta"], shipped["design"]
        budget = a.budget if a.budget is not None else float(meta["budget_rad"])
        did = DESIGN_ID.get(design, design)
        scene = None
        if a.plant != "shipped":
            scene = ROOT / f"docs/experiments/20260916-turn_mechanism/scenes/{a.plan}__{a.plant}.xml"
            if not scene.exists():
                sys.exit(f"no {a.plant} scene for {a.plan}: run real_v1_bench_plants.py first")
        C = construct(meta, design, budget, scene=scene, close_ramp=a.close_ramp)
        C["plant"] = a.plant
        tag = a.plan + ("" if a.plant == "shipped" else f"_{a.plant}")
    stem = a.out_dir / f"20260921-carry_construction_{did}_{tag}"
    figures(C, did, tag, stem)
    end = C["rec"][-1]
    summary = dict(source=str(p.resolve().relative_to(ROOT)), design=design, did=did, budget=budget,
                   axis_k=C["k"], theta_deg=C["theta"], span_mm=C["span"] * 1000,
                   pivot_lift_mm=C["k"] * C["span"] * 1000, grip_depth_mm=C["depth_mm"],
                   tip0_mm={f: (C["tip0"][f] * 1000).round(2).tolist() for f in FINGERS},
                   targets_mm={f: (C["targets"][f] * 1000).round(2).tolist() for f in FINGERS},
                   ik_residual_mm={f: C["resid"][f] * 1000 for f in FINGERS},
                   delta_deg={j: float(np.degrees(v)) for j, v in C["delta"].items()},
                   clipped={j: bool(abs(C["delta"][j]) > budget) for j in C["delta"]},
                   final_cos=end["cos"], final_contacts=end["ncon"], final_force_N=end["force"],
                   turn_end_cos=[r for r in C["rec"] if r["phase"] == "turn"][-1]["cos"])
    Path(f"{stem}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("tip0_mm", "targets_mm")},
                     indent=1))
    print(f"wrote {stem}.png (+ _construction/_execution/_schedule .png/.pdf) and .json")
    if a.video:
        video(C, did, tag, Path(f"{stem}.mp4"), speed=a.speed)
        print(f"wrote {stem}.mp4")


if __name__ == "__main__":
    main()
