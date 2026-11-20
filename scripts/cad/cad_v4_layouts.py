#!/usr/bin/env python3
"""Two actuator layouts for the SR2 finger (cad_v4), built on the cad_v3 hand, and their checks against the bench trials.

v4a  PIP servo on the yaw link: the PIP servo moves from the MCP link into a housing on the yaw link, behind the MCP
     servo. A parallel four-bar turns an idler on the MCP axis, and a rod parallel to the MCP link turns the DIP link
     with the idler (a parallelogram), so the DIP link keeps the idler's angle MCP + PIP. Below the MCP servo the
     finger shrinks to the sensor cassette's +-10.5 mm.
v4b  Gimbal MCP: the yaw axis passes through the MCP axis at G, v3's MCP point. The MCP servo stands on the yoke with
     its shaft through G, the yaw servo moves behind it on the yaw axis, and the PIP servo hangs from the yoke below the
     yaw servo, driving the same idler, rod and DIP link as v4a.

The dimensions are in assets/cad_v4/concepts/20261118-layout_params.json (mm, each finger's cad_v3 body frames: x
dorsal, y along the MCP axis, z up). New parts are boxes and capsules; parts that do not change keep cad_v3's meshes and
MorphIt collision spheres.

Sub-commands (each loads meshes, so run through resguard):
  build    write assets/mjcf/experimental/20261118-cad_v4/{v4a,v4b}.xml
  check    least distance between each finger's new proximal parts and its moving lower finger over the driver's ranges
  replay   index-middle clearance over the 65 bench trials; v4b by inverse kinematics to the same tip positions
  forces   pinch force at the servo limits, yaw lever, transmission angles of the drive, over the trials' postures
  joints   the trials' yaw-MCP joint space against the serial limits and a two-servo differential (figure)
  render   stills of v3, v4a and v4b
  play     open v4a or v4b in MuJoCo's viewer with the least index-middle gap overlaid (not part of `all`):
           play v4a|v4b [--kinematic] [--frames N]

  ~/.claude/bin/resguard.sh run --mem 3G --cpu 100 -- .venv/bin/python scripts/cad/cad_v4_layouts.py all
Outputs go to docs/experiments/20261118-sr2_actuator_layout/ (data/, media/).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _paths  # noqa: E402,F401

import ast  # noqa: E402
import collections  # noqa: E402
import glob  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / "assets/mjcf/experimental/20261104-cad_v3/scene_screwdriver_medium.xml"
PARAMS = ROOT / "assets/cad_v4/concepts/20261118-layout_params.json"
RUNS = ROOT / "docs/experiments/20261115-hom_bench_trials/runs"
MJCF_OUT = ROOT / "assets/mjcf/experimental/20261118-cad_v4"
OUT = ROOT / "docs/experiments/20261118-sr2_actuator_layout"
DATA, MEDIA = OUT / "data", OUT / "media"
FINGERS = ("index", "middle")
SIGN = {"yaw": -1.0, "mcp": 1.0, "pip": 1.0}  # manta_hand.plan JOINT_SIGN for index and middle
DRIVER = {"yaw": (-85.0, 85.0), "mcp": (-15.0, 92.0), "pip": (-18.0, 92.0)}  # servos.py nominal contract
DRIVER_MCP_CAP = {"index": 64.75, "middle": 69.73}  # servos.py caps from the pot's dead zone
STRIDE = 2
TRIALS_LAST_DAY = "20261117"
MARGIN = 0.02  # m; distances beyond it are reported as 20 mm
P = json.loads(PARAMS.read_text())
SILVER, DARK, STEEL, HOUSING = [0.66, 0.68, 0.72, 1], [0.12, 0.12, 0.13, 1], [0.42, 0.44, 0.50, 1], [0.85, 0.62, 0.38, 0.55]


def mujoco():
    import mujoco as mj
    return mj


# ----------------------------------------------------------------------------------------------- bench trials
def load_trials():
    """Every STRIDE-th period of the bench runs: mounts (mm, P) and sim joint angles (deg) of index and middle."""
    rows, plans = [], {}
    limit = int(os.environ.get("CADV4_LIMIT", "0"))
    for path in sorted(glob.glob(str(RUNS / "*.jsonl"))):
        if Path(path).name[:8] > TRIALS_LAST_DAY:  # the 65 runs of 16 and 17 November; later runs land in the same folder
            continue
        lines = Path(path).read_text().splitlines()
        plan = json.loads(lines[0]).get("plan")
        if plan not in plans:
            try:
                plans[plan] = json.loads((ROOT / plan).read_text())["meta"]["hom"]["d6_P_mm"]
            except (OSError, KeyError, TypeError, json.JSONDecodeError):
                plans[plan] = None
        d6 = plans[plan]
        if d6 is None:
            continue
        for k, line in enumerate(lines[1:]):
            if k % STRIDE:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            j = rec.get("joints")
            if not j:
                continue
            if isinstance(j, str):
                j = ast.literal_eval(j)
            q = {f: [j[f]["aa"] / SIGN["yaw"], j[f]["fe1"] / SIGN["mcp"], j[f]["fe2"] / SIGN["pip"]] for f in FINGERS}
            rows.append({"run": Path(path).name, "mount": {"index": d6[2:4], "middle": d6[4:6]}, "q": q})
    return rows[::max(1, len(rows) // limit)] if limit else rows


def angle_stats(rows):
    out = {}
    for f in FINGERS:
        Q = np.array([r["q"][f] for r in rows])
        th = Q[:, 1] + Q[:, 2]
        out[f] = {name: [round(float(np.percentile(x, 0.5)), 1), round(float(np.percentile(x, 99.5)), 1),
                         round(float(x.min()), 1), round(float(x.max()), 1)]
                  for name, x in (("yaw", Q[:, 0]), ("mcp", Q[:, 1]), ("pip", Q[:, 2]), ("dip_abs", th))}
    Qa = np.array([r["q"][f] for r in rows for f in FINGERS])
    pip_mid = float(np.mean(np.percentile(Qa[:, 2], [0.5, 99.5])))
    th_mid = float(np.mean(np.percentile(Qa[:, 1] + Qa[:, 2], [0.5, 99.5])))
    return out, pip_mid, th_mid


# ----------------------------------------------------------------------------------------------- geometry helpers
def _mm(v):
    return [float(x) * 1e-3 for x in v]


def add_box(mj, body, name, lo, hi, rgba, group=2):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return body.add_geom(name=name, type=mj.mjtGeom.mjGEOM_BOX, size=_mm((hi - lo) / 2), pos=_mm((hi + lo) / 2),
                         rgba=rgba, group=group, contype=1, conaffinity=1, mass=0)


# Distances are computed here, not by mj_geomDistance: it returned 0 for box-box and mesh-mesh pairs 10-15 mm apart
# (MuJoCo 3.6.0, box-box and mesh-mesh). Every collision geom is a box with half-sizes h plus a ball of radius r
# (sphere: h = 0; capsule: h along its axis). The gap is min |cA + RA u - cB - RB v| over |u| <= hA, |v| <= hB, minus
# rA + rB, solved by coordinate descent (exact in one pass when one side is a point).
def boxball(model, ids):
    mj, m, d = model.mj, model.m, model.d
    ids = np.asarray(ids, int)
    c = d.geom_xpos[ids] * 1e3
    R = d.geom_xmat[ids].reshape(-1, 3, 3).copy()
    h, r = np.zeros((len(ids), 3)), np.zeros(len(ids))
    for k, g in enumerate(ids):
        t, sz = m.geom_type[g], m.geom_size[g] * 1e3
        if t == mj.mjtGeom.mjGEOM_SPHERE:
            r[k] = sz[0]
        elif t == mj.mjtGeom.mjGEOM_CAPSULE:
            r[k], h[k, 2] = sz[0], sz[1]
        elif t == mj.mjtGeom.mjGEOM_BOX:
            h[k] = sz[:3]
        else:
            raise ValueError(f"geom type {t} not handled")
    return c, R, h, r


def pair_gap(cA, RA, hA, rA, cB, RB, hB, rB, passes=40):
    u, v = np.zeros_like(cA), np.zeros_like(cB)
    res = cA - cB
    for _ in range(passes):
        for i in range(3):
            a = RA[:, :, i]
            new = np.clip(u[:, i] - np.einsum("ni,ni->n", a, res), -hA[:, i], hA[:, i])
            res += a * (new - u[:, i])[:, None]
            u[:, i] = new
        for i in range(3):
            b = RB[:, :, i]
            new = np.clip(v[:, i] + np.einsum("ni,ni->n", b, res), -hB[:, i], hB[:, i])
            res -= b * (new - v[:, i])[:, None]
            v[:, i] = new
    return np.linalg.norm(res, axis=1) - rA - rB


def set_gap(A, B, cutoff=25.0, every=False):
    """Least gap (mm) between two sets of geoms and the indices of the pair; cutoff if every pair is farther. With
    every=True, the gaps and index pairs of all pairs within the cutoff."""
    cA, RA, hA, rA = A
    cB, RB, hB, rB = B
    bA, bB = np.linalg.norm(hA, axis=1) + rA, np.linalg.norm(hB, axis=1) + rB
    lower = np.linalg.norm(cA[:, None] - cB[None], axis=2) - bA[:, None] - bB[None]
    I, J = np.nonzero(lower < cutoff)
    if len(I) == 0:
        return (np.array([]), I, J) if every else (cutoff, None)
    flat = (np.abs(hA[I]).sum(1) == 0) & (np.abs(hB[J]).sum(1) == 0)
    gap = np.empty(len(I))
    gap[flat] = np.linalg.norm(cA[I[flat]] - cB[J[flat]], axis=1) - rA[I[flat]] - rB[J[flat]]
    k = ~flat
    if k.any():
        gap[k] = pair_gap(cA[I[k]], RA[I[k]], hA[I[k]], rA[I[k]], cB[J[k]], RB[J[k]], hB[J[k]], rB[J[k]])
    if every:
        return gap, I, J
    n = int(np.argmin(gap))
    return float(gap[n]), (int(I[n]), int(J[n]))


def add_capsule(mj, body, name, a, b, r, rgba, group=2):
    g = body.add_geom(name=name, type=mj.mjtGeom.mjGEOM_CAPSULE, size=[r * 1e-3, 0, 0], rgba=rgba, group=group,
                      contype=1, conaffinity=1, mass=0)
    g.fromto = _mm(list(a) + list(b))
    return g


def add_body(parent, name, pos, axis=None):
    import mujoco as mj
    b = parent.add_body(name=name, pos=_mm(pos))
    b.explicitinertial = True
    b.mass, b.ipos, b.inertia = 1e-4, [0.0, 0.0, 0.0], [1e-10, 1e-10, 1e-10]
    if axis is not None:
        b.add_joint(name=name + "_q", type=mj.mjtJoint.mjJNT_HINGE, axis=axis, range=[-math.pi, math.pi])
    return b


def servo_box(mj, body, name, kind, spline, long_dir, h_dir, rgba):
    """Case and tab envelope of a servo whose spline top is at `spline`, shaft along h_dir (base to spline), long end
    along long_dir. Only axis-aligned placements are needed here."""
    s = P["servos"][kind]
    long_dir, h_dir = np.asarray(long_dir, float), np.asarray(h_dir, float)
    t_dir = np.abs(np.cross(long_dir, h_dir))
    sp = np.asarray(spline, float)

    def corners(l_rng, h_rng):
        pts = [sp + l * long_dir + (h - s["h_spline"]) * h_dir + t * s["half_t"] * t_dir
               for l in l_rng for h in h_rng for t in (-1, 1)]
        return np.min(pts, 0), np.max(pts, 0)
    add_box(mj, body, name + "_case", *corners(s["len"], (0.0, s["h_case"])), rgba)
    add_box(mj, body, name + "_tabs", *corners(s["tab_len"], s["h_tab"]), rgba)
    boss = [sp + (h - s["h_spline"]) * h_dir + e1 * 3.0 * long_dir + e2 * 3.0 * t_dir
            for h in (s["h_case"], s["h_spline"]) for e1 in (-1, 1) for e2 in (-1, 1)]
    add_box(mj, body, name + "_boss", np.min(boss, 0), np.max(boss, 0), rgba)


def housing_box(mj, body, name, kind, spline, long_dir, h_dir):
    s, w = P["servos"][kind], P["drive"]["housing_wall"]
    long_dir, h_dir = np.asarray(long_dir, float), np.asarray(h_dir, float)
    t_dir = np.abs(np.cross(long_dir, h_dir))
    sp = np.asarray(spline, float)
    pts = [sp + l * long_dir + (h - s["h_spline"]) * h_dir + t * (s["half_t"] + w) * t_dir
           for l in (s["tab_len"][0] - w, s["tab_len"][1] + w) for h in (-w, s["h_case"]) for t in (-1, 1)]
    return add_box(mj, body, name, np.min(pts, 0), np.max(pts, 0), HOUSING)


def lower_finger(mj, spec, f, mcp_body, pip_body, pip_mid, prox=None):
    """The slim MCP link and DIP link shared by v4a and v4b."""
    L = P["lower_finger"]
    for g in list(mcp_body.geoms) + list(pip_body.geoms):
        nm = g.name or ""
        if (nm.startswith(f"{f}_mcp_frame_s") or nm.startswith(f"{f}_dip_link_s")
                or nm in (f"{f}_mcp_link_visual", f"{f}_servo_pip_visual", f"{f}_dip_link_visual")):
            spec.delete(g)
    x0, x1 = L["mcp_plate_x"]
    z0, z1 = L["mcp_plate_z"]
    cut = L["mcp_plate_cut"]
    top = L["mcp_plate_top"]
    for side, (y0, y1) in (("horn", L["mcp_plate_horn_y"]), ("base", L["mcp_plate_base_y"])):
        add_box(mj, mcp_body, f"{f}_v4_mcplink_plate_{side}_a", (x0, y0, z0), (cut["x_from"], y1, z1), SILVER)
        add_box(mj, mcp_body, f"{f}_v4_mcplink_plate_{side}_b", (cut["x_from"], y0, cut["z_below"]), (x1, y1, z1), SILVER)
        add_box(mj, mcp_body, f"{f}_v4_mcplink_plate_{side}_top", (top["x"][0], y0, top["z"][0]), (top["x"][1], y1, top["z"][1]),
                SILVER)
    add_box(mj, mcp_body, f"{f}_v4_mcplink_web", (L["mcp_web_x"][0], L["mcp_plate_horn_y"][0], L["mcp_web_z"][0]),
            (L["mcp_web_x"][1], L["mcp_plate_base_y"][1], L["mcp_web_z"][1]), SILVER)
    add_box(mj, mcp_body, f"{f}_v4_mcplink_core", (L["mcp_core_x"][0], L["mcp_core_y"][0], L["mcp_core_z"][0]),
            (L["mcp_core_x"][1], L["mcp_core_y"][1], L["mcp_core_z"][1]), SILVER)
    pz = -(prox or P["finger"]["mcp_pip"])
    add_capsule(mj, mcp_body, f"{f}_v4_mcplink_pipboss", (0, L["mcp_core_y"][0], pz), (0, L["mcp_core_y"][1], pz),
                L["pip_boss_r"], SILVER)
    add_box(mj, mcp_body, f"{f}_v4_mcplink_neck", (L["mcp_core_x"][0], L["mcp_core_y"][0], pz),
            (L["mcp_core_x"][1], L["mcp_core_y"][1], L["mcp_core_z"][0]), SILVER)
    for side, sgn in (("pos", 1), ("neg", -1)):
        ya, yb = sorted((sgn * L["dip_plate_y"][0], sgn * L["dip_plate_y"][1]))
        add_box(mj, pip_body, f"{f}_v4_diplink_plate_{side}", (L["dip_plate_x"][0], ya, L["dip_plate_z"][0]),
                (L["dip_plate_x"][1], yb, L["dip_plate_z"][1]), SILVER)
    add_box(mj, pip_body, f"{f}_v4_diplink_web", (L["dip_web_x"][0], -L["dip_plate_y"][1], L["dip_web_z"][0]),
            (L["dip_web_x"][1], L["dip_plate_y"][1], L["dip_web_z"][1]), SILVER)


DRIVE_JOINTS = ("idler_q", "rod_q", "coupler_q", "pipcrank_q")


def drive_angles(mcp, pip):
    """Angles (rad) of the drive's joints at MCP and PIP (rad): the idler and the PIP servo's crank turn at MCP + PIP,
    the rod keeps the MCP link's direction and the coupler the yaw frame's."""
    return {"idler_q": mcp + pip, "rod_q": -pip, "coupler_q": -(mcp + pip), "pipcrank_q": mcp + pip}


def drive(mj, f, yaw_body, mcp_axis, pip_shaft, pip_mid, th_mid, prox=None):
    """Idler on the MCP axis, parallelogram rod, parallel four-bar to the PIP servo's crank (positions in the frame of
    yaw_body, mm). Joint names: {f}_idler_q (MCP + PIP), {f}_rod_q (-PIP), {f}_coupler_q (-(MCP + PIP)), {f}_pipcrank_q."""
    D = P["drive"]
    beta = math.radians(90.0 - pip_mid)
    c = D["crank_r"] * np.array([math.sin(beta), 0.0, math.cos(beta)])
    dx, dz = pip_shaft[0] - mcp_axis[0], pip_shaft[2] - mcp_axis[2]
    alpha = math.atan2(-dz, dx) - math.radians(th_mid)  # arm perpendicular to the coupler, upward, at th_mid
    arm = D["arm_r"] * np.array([math.sin(alpha), 0.0, math.cos(alpha)])
    idler = add_body(yaw_body, f"{f}_idler", mcp_axis, axis=[0, 1, 0])
    add_capsule(mj, idler, f"{f}_v4_idler_hub", (0, D["idler_y"][0], 0), (0, D["idler_y"][1], 0), D["idler_hub_r"], STEEL)
    yr, yc = D["rod_y_upper"], D["coupler_y"]
    add_capsule(mj, idler, f"{f}_v4_idler_rodarm", (0, yr, 0), (c[0], yr, c[2]), 2.0, STEEL)
    add_capsule(mj, idler, f"{f}_v4_idler_fourbararm", (0, yc, 0), (arm[0], yc, arm[2]), 2.0, STEEL)
    rod = add_body(idler, f"{f}_rod", (c[0], 0, c[2]), axis=[0, 1, 0])
    j0, j1 = D["rod_jog_from_pin"]
    L = prox or P["finger"]["mcp_pip"]
    add_capsule(mj, rod, f"{f}_v4_rod_upper", (0, yr, 0), (0, yr, -j0), D["rod_r"], STEEL)
    add_capsule(mj, rod, f"{f}_v4_rod_jog", (0, yr, -j0), (0, D["rod_y_lower"], -j1), D["rod_r"], STEEL)
    add_capsule(mj, rod, f"{f}_v4_rod_lower", (0, D["rod_y_lower"], -j1), (0, D["rod_y_lower"], -L), D["rod_r"], STEEL)
    cp = add_body(idler, f"{f}_coupler", (arm[0], 0, arm[2]), axis=[0, 1, 0])
    add_capsule(mj, cp, f"{f}_v4_coupler", (0, yc, 0), (dx, yc, dz), D["coupler_r"], STEEL)
    crank = add_body(yaw_body, f"{f}_pipcrank", pip_shaft, axis=[0, 1, 0])
    add_capsule(mj, crank, f"{f}_v4_pipcrank", (0, yc, 0), (arm[0], yc, arm[2]), 2.0, STEEL)
    sp = D["support_plate_y"]
    add_box(mj, yaw_body, f"{f}_v4_mcp_support", (mcp_axis[0] + D["support_plate_x"][0], sp[0], mcp_axis[2] + D["support_plate_z"][0]),
            (mcp_axis[0] + D["support_plate_x"][1], sp[1], mcp_axis[2] + D["support_plate_z"][1]), SILVER)
    return {"crank_beta_deg": round(math.degrees(beta), 2), "arm_alpha_deg": round(math.degrees(alpha), 2),
            "coupler_mm": round(math.hypot(dx, dz), 2), "pip_mid_deg": round(pip_mid, 2), "dip_abs_mid_deg": round(th_mid, 2)}


def build_spec(variant, pip_mid, th_mid):
    mj = mujoco()
    spec = mj.MjSpec.from_file(str(SCENE))
    info = {}
    if variant == "v3":
        return spec, info
    sts = spec.add_mesh(name="v4_sts3032", file="../../20261013-cad_v1/meshes/servo_pip.stl")
    sts.scale = [1.0, P["servos"]["sts3032"]["h_spline"] / P["servos"]["scs0009"]["h_spline"], 1.0]
    q_pod = [0.0, math.sqrt(0.5), 0.0, -math.sqrt(0.5)]  # x -> -z, y -> -y, z -> -x: spline toward +y, long end down
    R_pod = np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]], float)
    sts_spline_mesh = np.array([0.0, -19.05 * sts.scale[1], -30.8])
    q_up = [0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5)]  # x -> +z, y -> -y, z -> +x: spline toward +y, long end up
    R_up = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]], float)
    for f in FINGERS:
        mount, yaw = spec.body(f"{f}_mount"), spec.body(f"{f}_yaw_frame")
        mcp, pip = spec.body(f"{f}_mcp_frame"), spec.body(f"{f}_pip_frame")
        if variant == "v4a":
            V = P["v4a"]
            mcp_axis = (0.0, 0.0, -P["finger"]["yaw_link"])
            pip_shaft = (V["pip_shaft_xz"][0], 0.0, V["pip_shaft_xz"][1])
            spl = np.array([V["pip_shaft_xz"][0], V["pip_spline_y"], V["pip_shaft_xz"][1]])
            housing_box(mj, yaw, f"{f}_v4_pod", V["pip_servo"], spl, (0, 0, -1), (0, 1, 0))
            servo_box(mj, yaw, f"{f}_v4_podservo", V["pip_servo"], spl, (0, 0, -1), (0, 1, 0), DARK)
            yaw.add_geom(name=f"{f}_v4vis_podservo", type=mj.mjtGeom.mjGEOM_MESH, meshname="v4_sts3032", quat=q_pod,
                         pos=_mm(spl - R_pod @ sts_spline_mesh), rgba=DARK, group=1, contype=0, conaffinity=0, mass=0)
            lower_finger(mj, spec, f, mcp, pip, pip_mid)
            info[f] = drive(mj, f, yaw, mcp_axis, pip_shaft, pip_mid, th_mid)
        elif variant in ("v4b", "v4b_low"):
            V = P["v4b"]
            G = V["gimbal_z"] if variant == "v4b" else V["low_variant"]["gimbal_z"]
            prox = V["proximal"] if variant == "v4b" else V["low_variant"]["proximal"]
            yaw.pos, mcp.pos = _mm((0, 0, G)), [0.0, 0.0, 0.0]
            spec.body(f"{f}_pip_frame").pos = _mm((0, 0, -prox))
            cut = G + V["carriage_cut_above_G"]
            for g in list(yaw.geoms):
                spec.delete(g)
            for g in list(mount.geoms):
                nm = g.name or ""
                if nm in (f"{f}_servo_yaw_visual", f"{f}_carriage_visual") or (
                        nm.startswith(f"{f}_mount_s") and g.pos[2] * 1e3 < cut):
                    spec.delete(g)
            mount.add_geom(name=f"{f}_v4vis_yawservo", type=mj.mjtGeom.mjGEOM_MESH, meshname="cad_servo_yaw",
                           pos=_mm((V["yaw_spline_x"] + 2.55, 0, G)), rgba=DARK, group=1, contype=0, conaffinity=0, mass=0)
            servo_box(mj, mount, f"{f}_v4_yawservo", V["yaw_servo"], (V["yaw_spline_x"], 0, G), (0, 0, 1), (-1, 0, 0), DARK)
            hb = V["yaw_housing"]
            add_box(mj, mount, f"{f}_v4_yawhousing", (hb["x"][0], hb["y"][0], G + hb["z"][0]),
                    (hb["x"][1], hb["y"][1], G + hb["z"][1]), HOUSING)
            glass = [0.66, 0.68, 0.72, 0.28]
            add_box(mj, mount, f"{f}_v4_carriage", (V["carriage_box_x"][0], -19.5, cut),
                    (V["carriage_box_x"][1], 19.5, 64.95), glass)
            add_box(mj, mount, f"{f}_v4_carriage_arm", (V["carriage_arm_x"][0], -8.55, G + hb["z"][1]),
                    (V["carriage_arm_x"][1], 8.55, 64.95), glass)
            # yoke: MCP servo upright on G, horn plate, lower bar and back plate around the yaw housing, PIP pod
            yaw.add_geom(name=f"{f}_v4vis_mcpservo", type=mj.mjtGeom.mjGEOM_MESH, meshname="cad_servo_mcp",
                         quat=[math.sqrt(0.5), 0, -math.sqrt(0.5), 0], pos=_mm((-20.75, 0, 0)), rgba=DARK, group=1,
                         contype=0, conaffinity=0, mass=0)
            servo_box(mj, yaw, f"{f}_v4_mcpservo", V["mcp_servo"], (0, -19.05, 0), (0, 0, 1), (0, -1, 0), DARK)
            for part in ("yoke_plate", "yoke_bar", "yoke_back", "yoke_front"):
                b = V[part]
                add_box(mj, yaw, f"{f}_v4_{part}", (b["x"][0], b["y"][0], b["z"][0]), (b["x"][1], b["y"][1], b["z"][1]), SILVER)
            pip_shaft = (V["pip_shaft_xz"][0], 0.0, V["pip_shaft_xz"][1])
            spl = np.array([V["pip_shaft_xz"][0], V["pip_spline_y"], V["pip_shaft_xz"][1]])
            housing_box(mj, yaw, f"{f}_v4_pod", V["pip_servo"], spl, (0, 0, 1), (0, 1, 0))
            servo_box(mj, yaw, f"{f}_v4_podservo", V["pip_servo"], spl, (0, 0, 1), (0, 1, 0), DARK)
            yaw.add_geom(name=f"{f}_v4vis_podservo", type=mj.mjtGeom.mjGEOM_MESH, meshname="v4_sts3032", quat=q_up,
                         pos=_mm(spl - R_up @ sts_spline_mesh), rgba=DARK, group=1, contype=0, conaffinity=0, mass=0)
            lower_finger(mj, spec, f, mcp, pip, pip_mid, prox)
            info[f] = drive(mj, f, yaw, (0.0, 0.0, 0.0), pip_shaft, pip_mid, th_mid, prox)
            info[f].update({"gimbal_z": G, "proximal": prox})
        elif variant == "v4c":
            info[f] = build_v4c(mj, spec, f, mount, yaw, mcp, pip, pip_mid)
    return spec, info


# ----------------------------------------------------------------------------------------------- v4c
V4C_RODS = (("p", 1.0), ("n", -1.0))


def v4c_closed_form(f):
    """v4c's rods in amazinghand_differential's geometry form (mm): the base pair in the mount frame (yaw about x, then
    MCP about y, through G; rest at yaw_rest and mcp_rest) and the PIP rod in the MCP link frame (PIP about y through the
    PIP axis; rest at pip_rest). The PIP lug is set square to its rod at rest."""
    import amazinghand_differential as AD
    V = P["v4c"]
    B, Q = V["base"], V["pip"]
    ax = np.array([1.0, 0.0, 0.0])

    def rod(C, u, A, r):
        C, u, A = (np.asarray(x, float) for x in (C, u, A))
        v = np.cross(ax, u)
        L = float(np.linalg.norm(A - (C + r * u)))
        w = A - C
        p_, q_ = w @ u, w @ v
        k = (w @ w + r * r - L * L) / (2 * r)
        acos = math.acos(float(np.clip(k / math.hypot(p_, q_), -1.0, 1.0)))
        branch = min((1.0, -1.0), key=lambda b: abs(float(AD.wrap(math.atan2(q_, p_) + b * acos))))
        return {"axis": ax, "u": u, "v": v, "C": C, "A": A, "r": r, "L": L, "th0": 0.0, "branch": branch}

    G = V["gimbal_z"]
    base = {"O": np.array([0.0, 0.0, G]), "e_a": ax, "e_f": np.array([0.0, 1.0, 0.0]), "f0": 0.0,
            "rest": (math.radians(B["yaw_rest"]), math.radians(B["mcp_rest"])),
            "rods": [rod((B["lug_x"], s * B["shaft_y"], G + B["horn_z"]), (0.0, s, 0.0), (B["lug_x"], s * B["lug_y"], G),
                         B["horn_r"]) for _k, s in V4C_RODS]}
    s = Q["side"][f]
    O = np.array([0.0, 0.0, -V["proximal"]])
    ball = np.array([Q["horn_x"], s * Q["horn_r"], Q["spline_z"]])
    lug = O + np.array([Q["lug_r"], Q["lug_y"], 0.0])
    for _ in range(12):  # the lug arm square to the rod's projection on the flexion plane
        d = (ball - lug) * np.array([1.0, 0.0, 1.0])
        d /= np.linalg.norm(d)
        n = np.array([d[2], 0.0, -d[0]])
        n = n if n[0] > 0 else -n
        lug = O + Q["lug_r"] * n + np.array([0.0, Q["lug_y"], 0.0])
    pip = {"O": O, "e_a": ax, "e_f": np.array([0.0, 1.0, 0.0]), "f0": 0.0, "rest": (0.0, math.radians(Q["pip_rest"])),
           "rods": [rod((Q["horn_x"], 0.0, Q["spline_z"]), (0.0, s, 0.0), lug, Q["horn_r"])]}
    return base, pip


def v4c_state(closed, yaw, mcp, pip):
    """Horn angles (rad), ball-joint quaternions of the rods (wxyz) and whether every rod closes with its horn within
    the limit, at joint angles in rad: [(horn, quat) for the two base rods], (horn, quat) of the PIP rod, ok."""
    import amazinghand_differential as AD
    from sr2hand.geometry.rotations import mat_to_quat, rot_axis_angle, rot_between
    lim = math.radians(P["v4c"]["base"]["horn_limit_deg"])
    out, ok = [], True
    for geom, a, f_ in ((closed[0], yaw, mcp), (closed[1], 0.0, pip)):
        a_, f2 = np.array([a - geom["rest"][0]]), np.array([f_ - geom["rest"][1]])
        th, good = AD.horn_angles(geom, a_, f2)
        R = AD.pose(geom, a_, f2)[0]
        res = []
        for rod, t, g in zip(geom["rods"], th[:, 0], good[:, 0]):
            t = float(np.clip(t, -lim, lim)) if np.isfinite(t) else 0.0
            ok &= bool(g) and abs(t) < lim
            ball = rod["C"] + rod["r"] * (rod["u"] * math.cos(t) + rod["v"] * math.sin(t))
            lug = geom["O"] + R @ (rod["A"] - geom["O"])
            d0 = rod["A"] - (rod["C"] + rod["r"] * rod["u"])
            res.append((t, mat_to_quat(rot_between(d0, rot_axis_angle(rod["axis"], t).T @ (lug - ball)))))
        out.append(res)
    return out[0], out[1][0], ok


def v4c_set_qpos(m, qpos, f, closed, yaw, mcp, pip):
    """Write the horn angles and rod quaternions of finger f into qpos for joint angles in rad; returns whether the
    pose is inside v4c's reach."""
    import mujoco as mj
    base, prod, ok = v4c_state(closed, yaw, mcp, pip)
    adr = lambda n: m.jnt_qposadr[mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, n)]  # noqa: E731
    for (k, _s), (t, q) in zip(V4C_RODS, base):
        qpos[adr(f"{f}_bhorn_{k}_q")] = t
        qpos[adr(f"{f}_brod_{k}_q"):adr(f"{f}_brod_{k}_q") + 4] = q
    qpos[adr(f"{f}_phorn_q")] = prod[0]
    qpos[adr(f"{f}_prod_q"):adr(f"{f}_prod_q") + 4] = prod[1]
    return ok


def build_v4c(mj, spec, f, mount, yaw, mcp, pip, pip_mid):
    """v4c on one finger: the gimbal of v4b driven by two rods from STS3032s in the carriage, and the PIP servo lying
    flat on the MCP link, driving the DIP link through one rod. Rods hang from their horns on ball joints; the
    playable scene closes them on the lugs with connect constraints."""
    from sr2hand.geometry.rotations import rot_y
    V = P["v4c"]
    B, Q, Lk, Y = V["base"], V["pip"], V["mcp_link"], V["yoke"]
    G, prox = V["gimbal_z"], V["proximal"]
    yaw.pos, mcp.pos = _mm((0, 0, G)), [0.0, 0.0, 0.0]
    pip.pos = _mm((0, 0, -prox))
    cut = G + V["carriage_cut_above_G"]
    for g in list(yaw.geoms):
        spec.delete(g)
    for g in list(mount.geoms):
        nm = g.name or ""
        if nm in (f"{f}_servo_yaw_visual", f"{f}_carriage_visual") or (nm.startswith(f"{f}_mount_s") and g.pos[2] * 1e3 < cut):
            spec.delete(g)
    lower_finger(mj, spec, f, mcp, pip, pip_mid, prox)
    for g in list(mcp.geoms):
        if (g.name or "").startswith(f"{f}_v4_mcplink_") and g.name != f"{f}_v4_mcplink_pipboss":
            spec.delete(g)
    base, pr = v4c_closed_form(f)
    for k, s in V4C_RODS:
        servo_box(mj, mount, f"{f}_v4_baseservo_{k}", B["servo"], (B["spline_x"], s * B["shaft_y"], G + B["horn_z"]),
                  (0, 0, -1), (-1, 0, 0), DARK)
    hb, fp = V["housing"], V["frame_post"]
    add_box(mj, mount, f"{f}_v4_basehousing", (hb["x"][0], hb["y"][0], G + hb["z"][0]), (hb["x"][1], hb["y"][1], G + hb["z"][1]),
            HOUSING)
    add_box(mj, mount, f"{f}_v4_framepost", (fp["x"][0], fp["y"][0], G + fp["z"][0]), (fp["x"][1], fp["y"][1], G + fp["z"][1]),
            SILVER)
    for k, s in V4C_RODS:
        ya, yb = sorted((s * Y["cheek_y"][0], s * Y["cheek_y"][1]))
        add_box(mj, yaw, f"{f}_v4_yoke_cheek_{k}", (Y["cheek_x"][0], ya, Y["cheek_z"][0]), (Y["cheek_x"][1], yb, Y["cheek_z"][1]),
                SILVER)
        add_box(mj, yaw, f"{f}_v4_yoke_arm_{k}", (Y["arm_x"][0], ya, Y["arm_z"][0]), (Y["arm_x"][1], yb, Y["arm_z"][1]), SILVER)
    add_box(mj, yaw, f"{f}_v4_yoke_dorsal", (Y["bridge_x"][0], -Y["cheek_y"][1], Y["arm_z"][0]),
            (Y["bridge_x"][1], Y["cheek_y"][1], Y["arm_z"][1]), SILVER)
    add_capsule(mj, yaw, f"{f}_v4_yoke_pin", (Y["pin_x"][0], 0, 0), (Y["pin_x"][1], 0, 0), Y["pin_r"], STEEL)
    for k, s in V4C_RODS:
        ya, yb = sorted((s * Lk["plate_y"][0], s * Lk["plate_y"][1]))
        add_box(mj, mcp, f"{f}_v4_mcplink_plate_{k}", (Lk["plate_x"][0], ya, Lk["plate_z"][0]), (Lk["plate_x"][1], yb, Lk["plate_z"][1]),
                SILVER)
        e = Lk["plate_end_r"] / math.sqrt(2.0)  # the square inscribed in the plate's round end about G
        add_box(mj, mcp, f"{f}_v4_mcplink_plateend_{k}", (-e, ya, -e), (e, yb, e), SILVER)
        add_box(mj, mcp, f"{f}_v4_mcplink_belly_{k}", (Lk["servo_zone_x"][0], ya, Lk["servo_zone_z"][0]),
                (Lk["servo_zone_x"][1], yb, Lk["servo_zone_z"][1]), SILVER)
    servo_box(mj, mcp, f"{f}_v4_pipservo", Q["servo"], (Q["spline_x"], 0.0, Q["spline_z"]),
              (0, 0, 1) if Q.get("long_end") == "up" else (0, 0, -1), (1, 0, 0), DARK)
    lim = math.radians(B["horn_limit_deg"])

    def horn_and_rod(parent, name, rod, ball_r, rod_r, lug_body, lug_local):
        h = add_body(parent, f"{f}_{name[0]}horn{name[1:]}", rod["C"], axis=[1, 0, 0])
        j = spec.joint(f"{f}_{name[0]}horn{name[1:]}_q")
        j.limited, j.range = mj.mjtLimited.mjLIMITED_TRUE, [-lim, lim]
        arm = rod["r"] * rod["u"]
        add_capsule(mj, h, f"{f}_v4_{name[0]}horn{name[1:]}", (0, 0, 0), arm, 2.0, STEEL)
        rb = add_body(h, f"{f}_{name[0]}rod{name[1:]}", arm)
        rb.add_joint(name=f"{f}_{name[0]}rod{name[1:]}_q", type=mj.mjtJoint.mjJNT_BALL)
        end = rod["A"] - (rod["C"] + arm)
        add_capsule(mj, rb, f"{f}_v4_{name[0]}rod{name[1:]}", (0, 0, 0), end, rod_r, STEEL)
        rb.add_site(name=f"{f}_v4c_{name[0]}rod{name[1:]}_end", pos=_mm(end))
        lug_body.add_site(name=f"{f}_v4c_{name[0]}lug{name[1:]}", pos=_mm(lug_local))
        add_capsule(mj, lug_body, f"{f}_v4_{name[0]}lug{name[1:]}", lug_local, lug_local + np.array([0.0, 0.0, 0.5]), ball_r, STEEL)
        return rod["L"]

    R_rest = rot_y(math.radians(B["mcp_rest"]))
    lens = [horn_and_rod(mount, f"b_{k}", rod, B["ball_r"], B["rod_r"], mcp, R_rest.T @ (rod["A"] - base["O"]))
            for (k, _s), rod in zip(V4C_RODS, base["rods"])]
    prod_ = pr["rods"][0]
    R_pip = rot_y(math.radians(Q["pip_rest"]))
    lens.append(horn_and_rod(mcp, "p", prod_, Q["ball_r"], Q["rod_r"], pip, R_pip.T @ (prod_["A"] - pr["O"])))
    return {"gimbal_z": G, "proximal": prox, "rod_mm": [round(x, 2) for x in lens],
            "pip_lug_mm": [round(float(x), 2) for x in prod_["A"]]}


PALM_LIFT = 0.04  # m; the base scene's straight fingers reach 25 mm below the floor


def make_playable(mj, spec, variant="v4a"):
    """What the written scenes need under physics, which kinematic posing (Model.set) does not: the base scene's
    morphology slides (mount x, y and finger length) removed, since contact forces drag them; equality constraints
    that keep the drive on drive_angles (a fixed tendon idler - MCP - PIP held at 0, joint couplings for the rest);
    unlimited drive joints with a little armature; no contacts between a finger's drive parts and the finger's
    other parts or the palm (their clearances are the `check` sub-command's); and the palm raised by PALM_LIFT, so
    that the straight fingers of qpos0, where a viewer starts, clear the floor (playable_key keeps the keys' world
    poses)."""
    for f in ("thumb",) + FINGERS:
        for n in ("x", "y", "len"):
            spec.delete(spec.joint(f"{f}_{n}"))
    spec.key("open").qpos = []  # refilled by playable_key; the old one has the removed slides' entries
    spec.body("palm_pose").pos = np.asarray(spec.body("palm_pose").pos) + [0.0, 0.0, PALM_LIFT]
    have = {frozenset((x.bodyname1, x.bodyname2)) for x in spec.excludes}
    if variant == "v4c":
        stiff = [2.0 * 0.002, 1.0]
        for f in FINGERS:
            new = [f"{f}_{b}" for b in ("bhorn_p", "bhorn_n", "brod_p", "brod_n", "phorn", "prod")]
            for b in new:
                if b.endswith(("horn_p", "horn_n", "phorn")):
                    spec.joint(f"{b}_q").armature = 1e-5
            for rod, lug in (("brod_p", "blug_p"), ("brod_n", "blug_n"), ("prod", "plug")):
                spec.add_equality(name=f"{f}_{rod}_close", type=mj.mjtEq.mjEQ_CONNECT, objtype=mj.mjtObj.mjOBJ_SITE,
                                  name1=f"{f}_v4c_{rod}_end", name2=f"{f}_v4c_{lug}", solref=stiff)
            own = [f"{f}_{b}" for b in ("mount", "yaw_frame", "mcp_frame", "len_frame", "pip_frame", "tip")] + ["palm_pose"]
            for a, b in list(itertools.combinations(new, 2)) + [(a, b) for a in new for b in own]:
                if frozenset((a, b)) not in have:
                    spec.add_exclude(bodyname1=a, bodyname2=b)
                    have.add(frozenset((a, b)))
        return
    for f in FINGERS:
        for n in DRIVE_JOINTS:
            j = spec.joint(f"{f}_{n}")
            j.limited, j.armature = mj.mjtLimited.mjLIMITED_FALSE, 1e-5
        t = spec.add_tendon(name=f"{f}_idler_minus_mcp_pip")
        for n, k in (("idler_q", 1.0), ("mcp", -1.0), ("pip", -1.0)):
            t.wrap_joint(f"{f}_{n}", k)
        stiff = [2.0 * 0.002, 1.0]  # the shortest time constant MuJoCo allows at the scene's 2 ms step
        spec.add_equality(name=f"{f}_idler_eq", type=mj.mjtEq.mjEQ_TENDON, objtype=mj.mjtObj.mjOBJ_TENDON,
                          name1=t.name, data=np.zeros(11), solref=stiff)
        for j1, j2, k in (("rod_q", "pip", -1.0), ("coupler_q", "idler_q", -1.0), ("pipcrank_q", "idler_q", 1.0)):
            data = np.zeros(11)
            data[1] = k
            spec.add_equality(name=f"{f}_{j1}_eq", type=mj.mjtEq.mjEQ_JOINT, objtype=mj.mjtObj.mjOBJ_JOINT,
                              name1=f"{f}_{j1}", name2=f"{f}_{j2}", data=data, solref=stiff)
        new = [f"{f}_{b}" for b in ("idler", "rod", "coupler", "pipcrank")]
        own = [f"{f}_{b}" for b in ("mount", "yaw_frame", "mcp_frame", "len_frame", "pip_frame", "tip")] + ["palm_pose"]
        for a, b in list(itertools.combinations(new, 2)) + [(a, b) for a in new for b in own]:
            if frozenset((a, b)) not in have:
                spec.add_exclude(bodyname1=a, bodyname2=b)
                have.add(frozenset((a, b)))


def playable_key(mj, spec, m, variant="v4a"):
    """The base scene's keyframe "open" mapped onto the new joints by name (MjSpec keeps the old qpos and appends
    zeros, which puts the old values on the wrong joints), with the drive on drive_angles and the palm where the base
    scene has it (its fingertips 3.9 mm into the floor), and a key "lifted" with the palm 40 mm higher."""
    base = mj.MjModel.from_xml_path(str(SCENE))
    kb = base.key("open")
    width = {int(mj.mjtJoint.mjJNT_FREE): 7, int(mj.mjtJoint.mjJNT_BALL): 4}
    jkey = lambda mm, j: mm.joint(j).name or "body " + mm.body(mm.jnt_bodyid[j]).name  # noqa: E731 (free joints are unnamed)
    new = {jkey(m, j): j for j in range(m.njnt)}
    q = m.qpos0.copy()
    for j in range(base.njnt):
        if jkey(base, j) not in new:  # a removed morphology slide
            continue
        a, n = base.jnt_qposadr[j], width.get(int(base.jnt_type[j]), 1)
        b = m.jnt_qposadr[new[jkey(base, j)]]
        q[b:b + n] = kb.qpos[a:a + n]
    adr = lambda f, n: m.jnt_qposadr[m.joint(f"{f}_{n}").id]  # noqa: E731
    for f in FINGERS:
        if variant == "v4c":
            v4c_set_qpos(m, q, f, v4c_closed_form(f), q[adr(f, "yaw")], q[adr(f, "mcp")], q[adr(f, "pip")])
            continue
        for n, v in drive_angles(q[adr(f, "mcp")], q[adr(f, "pip")]).items():
            q[adr(f, n)] = v
    ctrl = kb.ctrl.copy()
    pz, upz = adr("palm", "pz"), m.actuator("a_palm_pz").id
    lifted = spec.add_key(name="lifted", qpos=q.copy(), ctrl=ctrl.copy())
    q[pz] -= PALM_LIFT
    ctrl[upz] -= PALM_LIFT
    spec.key("open").qpos, spec.key("open").ctrl = q, ctrl
    return lifted


# ----------------------------------------------------------------------------------------------- kinematics
class Model:
    def __init__(self, variant, pip_mid, th_mid):
        mj = mujoco()
        self.mj, self.variant = mj, variant
        self.spec, self.info = build_spec(variant, pip_mid, th_mid)
        self.spec.memory = 96 * 1024 * 1024
        self.m = self.spec.compile()
        self.d = mj.MjData(self.m)
        self.jid = lambda n: mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_JOINT, n)
        self.mount0 = {f: self.m.body_pos[mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_BODY, f"{f}_mount")] * 1e3 for f in FINGERS}
        self.v4c = {f: v4c_closed_form(f) for f in FINGERS} if variant == "v4c" else None
        self.v4c_ok = {}

    def set(self, mounts, q):
        m, d = self.m, self.d
        d.qpos[:] = m.qpos0
        for f in FINGERS:
            d.qpos[m.jnt_qposadr[self.jid(f"{f}_x")]] = (mounts[f][0] - self.mount0[f][0]) * 1e-3
            d.qpos[m.jnt_qposadr[self.jid(f"{f}_y")]] = (mounts[f][1] - self.mount0[f][1]) * 1e-3
            yaw, mcp, pip = (math.radians(v) for v in q[f])
            for name, v in (("yaw", yaw), ("mcp", mcp), ("pip", pip), *drive_angles(mcp, pip).items()):
                j = self.jid(f"{f}_{name}")
                if j >= 0:
                    d.qpos[m.jnt_qposadr[j]] = v
            if self.v4c:
                self.v4c_ok[f] = v4c_set_qpos(m, d.qpos, f, self.v4c[f], yaw, mcp, pip)
        self.mj.mj_kinematics(m, d)
        self.mj.mj_comPos(m, d)

    def tip(self, f):
        return self.d.xpos[self.mj.mj_name2id(self.m, self.mj.mjtObj.mjOBJ_BODY, f"{f}_tip")] * 1e3

    def body_pos(self, name):
        return self.d.xpos[self.mj.mj_name2id(self.m, self.mj.mjtObj.mjOBJ_BODY, name)] * 1e3


def ik_gimbal(p, G, q_ref, a=None):
    """Yaw about x through G, then a planar 2R (30.8, 36.36). p, G in mm (mount frame). Returns (yaw, mcp, pip) deg
    on the branch nearest q_ref, and whether the point is within reach."""
    a, b = a or P["finger"]["mcp_pip"], P["finger"]["pip_tip"]
    v = np.asarray(p) - np.asarray(G)
    yaw = math.atan2(v[1], -v[2])
    u, w = v[0], math.hypot(v[1], v[2])
    c2 = (u * u + w * w - a * a - b * b) / (2 * a * b)
    reach = abs(c2) <= 1.0
    c2 = max(-1.0, min(1.0, c2))
    best = None
    for s in (1, -1):
        p2 = s * math.acos(c2)
        p1 = math.atan2(-u, w) - math.atan2(b * math.sin(p2), a + b * math.cos(p2))
        q = (math.degrees(yaw), math.degrees(p1), math.degrees(p2))
        cost = abs(q[1] - q_ref[1]) + abs(q[2] - q_ref[2])
        if best is None or cost < best[0]:
            best = (cost, q)
    return best[1], reach


# ----------------------------------------------------------------------------------------------- commands
def cmd_build(rows):
    _, pip_mid, th_mid = angle_stats(rows)
    MJCF_OUT.mkdir(parents=True, exist_ok=True)
    out = {}
    mj = mujoco()
    for v in ("v4a", "v4b", "v4c"):
        spec, info = build_spec(v, pip_mid, th_mid)
        make_playable(mj, spec, v)
        playable_key(mj, spec, spec.compile(), v)
        spec.compile()
        spec.meshdir = "../20261104-cad_v3/meshes"
        spec.texturedir = "../20261104-cad_v3/meshes"
        spec.modelname = f"cad_v4_{v}_scene_screwdriver_medium"
        head, rest = spec.to_xml().split("\n", 1)
        drive_note = ("     Equality constraints keep the idler, rod, coupler and PIP crank on the MCP and PIP joints; the idler\n"
                      "     and the PIP servo's crank turn at MCP + PIP. The actuators are cad_v3's, in joint angles.\n")
        if v == "v4c":
            drive_note = ("     Connect constraints close the two base rods on the MCP link's lugs and the PIP rod on the DIP link's\n"
                          "     lug; the horns stop at +-90 deg, where the reach ends. The actuators are cad_v3's, in joint angles.\n")
        note = (f"<!-- cad_v4 {v}, written by scripts/cad/cad_v4_layouts.py build from {PARAMS.relative_to(ROOT)}.\n"
                + drive_note + f"     View: .venv/bin/python scripts/cad/cad_v4_layouts.py play {v} -->")
        (MJCF_OUT / f"{v}.xml").write_text(f"{head}\n{note}\n{rest}")
        out[v] = info
        print("wrote", (MJCF_OUT / f"{v}.xml").relative_to(ROOT))
    return out


def geoms_of(model, prefix_list, exclude=()):
    mj, m = model.mj, model.m
    out = []
    for g in range(m.ngeom):
        nm = mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, g) or ""
        if any(nm.startswith(p) for p in prefix_list) and not any(e in nm for e in exclude):
            out.append(g)
    return out


def cmd_check(rows):
    """Least distance between each finger's new proximal parts and its own moving lower finger over a grid of the
    driver's MCP and PIP ranges (and five yaw angles for v4b's mount-side housing)."""
    _, pip_mid, th_mid = angle_stats(rows)
    res = {}
    for v in ("v4a", "v4b"):
        M = Model(v, pip_mid, th_mid)
        f = "index"
        prox_names = [f"{f}_v4_pod", f"{f}_v4_podservo", f"{f}_v4_coupler", f"{f}_v4_pipcrank", f"{f}_v4_mcp_support"]
        if v == "v4b":
            prox_names += [f"{f}_v4_yoke", f"{f}_v4_mcpservo", f"{f}_v4_yawhousing", f"{f}_v4_yawservo", f"{f}_v4_carriage"]
        dist_names = [f"{f}_v4_mcplink", f"{f}_v4_diplink", f"{f}_v4_rod", f"{f}_cassette_s", f"{f}_cap_s"]
        prox = geoms_of(M, prox_names)
        dist = geoms_of(M, dist_names)
        lab = lambda g: label(M.mj, M.m, g)  # noqa: E731
        mounts = {ff: M.mount0[ff][:2] for ff in FINGERS}
        worst, grid = {}, []
        for yaw in ((-85, -45, 0, 45, 85) if v == "v4b" else (0,)):
            for mcp in np.arange(DRIVER["mcp"][0], DRIVER["mcp"][1] + 0.1, 3.0):
                for pip in np.arange(DRIVER["pip"][0], DRIVER["pip"][1] + 0.1, 3.0):
                    M.set(mounts, {ff: (yaw, mcp, pip) for ff in FINGERS})
                    gap, I, J = set_gap(boxball(M, prox), boxball(M, dist), cutoff=10.0, every=True)
                    if len(gap) == 0:
                        grid.append((yaw, float(mcp), float(pip), 10.0, "", ""))
                        continue
                    k = int(np.argmin(gap))
                    grid.append((yaw, float(mcp), float(pip), float(gap[k]), lab(prox[I[k]]), lab(dist[J[k]])))
                    for gg, i, j in zip(gap, I, J):
                        key = f"{lab(prox[i])} | {lab(dist[j])}"
                        if key not in worst or gg < worst[key][0]:
                            worst[key] = (round(float(gg), 2), yaw, float(mcp), float(pip))
        G = np.array([g[3] for g in grid])
        blocked = [g for g in grid if g[3] < 0.5]
        res[v] = {"grid_points": len(grid), "least_mm": round(float(G.min()), 2),
                  "points_under_0p5mm": len(blocked),
                  "blocked_examples": [list(map(lambda x: round(x, 2) if isinstance(x, float) else x, g)) for g in blocked[:12]],
                  "least_by_pair": dict(sorted(worst.items(), key=lambda kv: kv[1][0])[:14])}
        print(v, json.dumps({k: res[v][k] for k in ("grid_points", "least_mm", "points_under_0p5mm")}))
        for k, w in list(res[v]["least_by_pair"].items())[:10]:
            print("   ", k, w)
    res["carriage"] = carriage_check(rows, pip_mid, th_mid)
    res["v4c"] = check_v4c(pip_mid, th_mid)
    res["v4c_limits"] = limits_v4c(pip_mid, th_mid)
    return res


def check_v4c(pip_mid, th_mid):
    """Self-clearance of v4c's index finger over the driver's ranges (yaw at five angles, MCP and PIP on a 3 and 6 deg
    grid), at the poses inside its reach: the base rods, horns, servos, housing, frame post and yoke against the MCP
    link, PIP drive and DIP link, and the PIP rod against its own link. Pairs joined by design (a rod and its lug, a
    rod and the plate that carries its lug, a horn and its servo) are left out."""
    M = Model("v4c", pip_mid, th_mid)
    f = "index"
    mj, m = M.mj, M.m
    names = {g: mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, g) or "" for g in range(m.ngeom)}
    pick = lambda keys: [g for g, n in names.items() if n.startswith(f + "_") and any(f"_v4_{k}" in n for k in keys)]  # noqa: E731
    upper = pick(("baseservo", "basehousing", "framepost", "yoke", "bhorn", "brod"))
    lower = pick(("mcplink", "blug", "pipservo", "phorn", "prod", "plug", "diplink")) + [
        g for g, n in names.items() if n.startswith(f"{f}_cassette_s") or n.startswith(f"{f}_cap_s")]
    joined = {("brod_p", "blug_p"), ("brod_n", "blug_n"), ("brod_p", "mcplink_plate_p"), ("brod_n", "mcplink_plate_n"),
              ("brod_p", "mcplink_plateend_p"), ("brod_n", "mcplink_plateend_n"),
              ("prod", "plug"), ("phorn", "pipservo"), ("prod", "phorn"), ("plug", "diplink")}
    lab = lambda g: names[g].split("_v4_", 1)[-1] if "_v4_" in names[g] else label(mj, m, g)  # noqa: E731

    def pairs(A, B):
        out = []
        for i in A:
            for j in B:
                a, b = lab(i), lab(j)
                if any(a.startswith(x) and b.startswith(y) or a.startswith(y) and b.startswith(x) for x, y in joined) or i == j:
                    continue
                out.append((i, j))
        return out
    P1 = pairs(upper, lower)
    P2 = pairs(pick(("prod",)), pick(("mcplink", "pipservo", "diplink")) + [g for g, n in names.items() if n.startswith(f"{f}_cassette_s")])
    mounts = {ff: M.mount0[ff][:2] for ff in FINGERS}
    worst, least, n_in, n_out = {}, 99.0, 0, 0
    for yaw in (-85, -45, 0, 45, 85):
        for mcp in np.arange(DRIVER["mcp"][0], DRIVER["mcp"][1] + 0.1, 3.0):
            for pip in np.arange(DRIVER["pip"][0], DRIVER["pip"][1] + 0.1, 6.0):
                M.set(mounts, {ff: (yaw, mcp, pip) for ff in FINGERS})
                if not M.v4c_ok[f]:
                    n_out += 1
                    continue
                n_in += 1
                for i, j in P1 + P2:
                    gap, _pair = set_gap(boxball(M, [i]), boxball(M, [j]), cutoff=10.0)
                    key = f"{lab(i)} | {lab(j)}"
                    if gap < worst.get(key, (99.0,))[0]:
                        worst[key] = (round(float(gap), 2), yaw, float(mcp), float(pip))
                    least = min(least, gap)
    out = {"poses_in_reach": n_in, "poses_out_of_reach": n_out, "least_mm": round(float(least), 2),
           "least_by_pair": dict(sorted(worst.items(), key=lambda kv: kv[1][0])[:14])}
    print("v4c", json.dumps({k: out[k] for k in ("poses_in_reach", "poses_out_of_reach", "least_mm")}))
    for k, w in list(out["least_by_pair"].items())[:10]:
        print("   ", k, w)
    return out


def limits_v4c(pip_mid, th_mid):
    """Where v4c's remaining contacts begin: a base rod against the other side's lug, plate, plate end or belly (the
    largest yaw, either sign, at which every MCP angle inside the reach stays clear), and the PIP rod against the PIP
    servo, the PIP boss and the MCP link's plates over the PIP range (yaw and MCP at 0)."""
    M = Model("v4c", pip_mid, th_mid)
    mj, m = M.mj, M.m
    nm = {g: mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, g) or "" for g in range(m.ngeom)}
    gid = lambda s: [g for g, n in nm.items() if n.startswith("index_v4_" + s)]  # noqa: E731
    other = lambda k: gid(f"blug_{k}") + gid(f"mcplink_plate_{k}") + gid(f"mcplink_plateend_{k}") + gid(f"mcplink_belly_{k}")  # noqa: E731
    base_pairs = [(gid("brod_p")[0], g) for g in other("n")] + [(gid("brod_n")[0], g) for g in other("p")]
    pip_pairs = [(gid("prod")[0], g) for g in gid("pipservo") + gid("mcplink_pipboss") + gid("mcplink_plate") + gid("mcplink_belly")]
    mounts = {f: M.mount0[f][:2] for f in FINGERS}
    gap = lambda pairs: min(set_gap(boxball(M, [i]), boxball(M, [j]), cutoff=10.0)[0] for i, j in pairs)  # noqa: E731
    base = {}
    for yaw in np.arange(0.0, 85.01, 2.5):
        worst = 99.0
        for sgn in (1, -1):
            for mcp in np.arange(DRIVER["mcp"][0], DRIVER["mcp"][1] + 0.01, 3.0):
                M.set(mounts, {f: (sgn * yaw, mcp, 30.0) for f in FINGERS})
                if M.v4c_ok["index"]:
                    worst = min(worst, gap(base_pairs))
        base[float(yaw)] = round(float(worst), 2)
    pip = {}
    for p in np.arange(DRIVER["pip"][0], DRIVER["pip"][1] + 0.01, 1.0):
        M.set(mounts, {f: (0.0, 0.0, p) for f in FINGERS})
        pip[float(p)] = round(float(gap(pip_pairs)), 2)
    clear_yaw = 0.0
    for y in sorted(base):
        if base[y] < 0:
            break
        clear_yaw = y
    clear_pip = [p for p, g in pip.items() if g >= 0]
    out = {"base_clear_to_yaw_deg": clear_yaw, "base_gap_mm_by_yaw": base, "pip_clear_deg": [min(clear_pip), max(clear_pip)],
           "pip_gap_mm_by_angle": pip}
    print("v4c limits", json.dumps({k: out[k] for k in ("base_clear_to_yaw_deg", "pip_clear_deg")}))
    return out


def surface_samples(m, geoms, spacing=0.25e-3, seed=0):
    """Vertices plus area-weighted random points of mesh geoms, about one per spacing squared, in each geom's frame."""
    rng = np.random.default_rng(seed)
    out = []
    for g in geoms:
        k = m.geom_dataid[g]
        V = m.mesh_vert[m.mesh_vertadr[k]: m.mesh_vertadr[k] + m.mesh_vertnum[k]]
        A = V[m.mesh_face[m.mesh_faceadr[k]: m.mesh_faceadr[k] + m.mesh_facenum[k]]]
        area = 0.5 * np.linalg.norm(np.cross(A[:, 1] - A[:, 0], A[:, 2] - A[:, 0]), axis=1)
        n = int(area.sum() / spacing ** 2) + 1
        i = rng.choice(len(A), size=n, p=area / area.sum())
        r = rng.random((n, 2))
        r[r.sum(1) > 1] = 1 - r[r.sum(1) > 1]
        out.append((g, np.vstack([V, A[i, 0] + r[:, :1] * (A[i, 1] - A[i, 0]) + r[:, 1:] * (A[i, 2] - A[i, 0])])))
    return out


def points_gap(m, d, samples, geoms):
    """Least distance (mm) from the posed surface samples to box, capsule and sphere geoms, and that geom."""
    mj = mujoco()
    W = np.vstack([d.geom_xpos[g] + p @ d.geom_xmat[g].reshape(3, 3).T for g, p in samples])
    best = (np.inf, None)
    for g in geoms:
        loc, h = (W - d.geom_xpos[g]) @ d.geom_xmat[g].reshape(3, 3), m.geom_size[g]
        if m.geom_type[g] == mj.mjtGeom.mjGEOM_BOX:
            dist = np.linalg.norm(loc - np.clip(loc, -h, h), axis=1)
        elif m.geom_type[g] == mj.mjtGeom.mjGEOM_CAPSULE:  # axis z, half-length h[1], radius h[0]
            axis_pt = np.zeros_like(loc)
            axis_pt[:, 2] = np.clip(loc[:, 2], -h[1], h[1])
            dist = np.linalg.norm(loc - axis_pt, axis=1) - h[0]
        else:  # sphere
            dist = np.linalg.norm(loc, axis=1) - h[0]
        k = float(dist.min()) * 1e3
        if k < best[0]:
            best = (k, g)
    return best


def carriage_check(rows, pip_mid, th_mid):
    """The MCP link against the carriage: the mount's collision spheres (and v4b's new mount parts) in every finger-period
    of the trials, and for v4a also against the carriage's and yaw servo's meshes, which the spheres overstate. v4b's
    gimbal moves the joints, so it is checked over the driver's whole yaw and MCP ranges (5 and 3 deg grid)."""
    out = {}
    for v in ("v3", "v4a", "v4b"):
        M = Model(v, pip_mid, th_mid)
        mj, m, d = M.mj, M.m, M.d
        names = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, g) or "" for g in range(m.ngeom)]
        out[v] = {}
        for f in FINGERS:
            mount = [g for g, n in enumerate(names) if n.startswith(f"{f}_mount_s") or any(
                n.startswith(f"{f}_v4_{p}") for p in ("yawservo", "yawhousing", "carriage"))]
            link = [g for g, n in enumerate(names) if n.startswith(f"{f}_mcp_frame_s") or n.startswith(f"{f}_v4_mcplink")]
            ja = [m.jnt_qposadr[M.jid(f"{f}_{n}")] for n in ("yaw", "mcp")]
            if v == "v4b":
                poses = [(y, c) for y in np.arange(-85.0, 85.1, 5.0) for c in np.arange(-15.0, 92.1, 3.0)]
            else:
                poses = [(r["q"][f][0], r["q"][f][1]) for r in rows]
            least, over = (np.inf, None), []
            for yaw, mcp in poses:
                d.qpos[:] = m.qpos0
                d.qpos[ja[0]], d.qpos[ja[1]] = math.radians(yaw), math.radians(mcp)
                mj.mj_kinematics(m, d)
                gap, pair = set_gap(boxball(M, mount), boxball(M, link), cutoff=5.0)
                if gap < 0:
                    over.append((yaw, mcp))
                if gap < least[0]:
                    least = (gap, (round(yaw, 1), round(mcp, 1), names[mount[pair[0]]], names[link[pair[1]]]) if pair else None)
            res = {"poses": len(poses), "overlapping": len(over), "least_mm": round(least[0], 2), "at": least[1]}
            if over:
                O = np.array(over)
                res["overlap_yaw_deg"] = [round(float(O[:, 0].min()), 1), round(float(O[:, 0].max()), 1)]
                res["overlap_mcp_deg"] = [round(float(O[:, 1].min()), 1), round(float(O[:, 1].max()), 1)]
            if v == "v4a" and over:
                mesh = [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(f"{f}_mount").id
                        and m.geom_type[g] == mj.mjtGeom.mjGEOM_MESH]
                samples = surface_samples(m, mesh)
                worst = (np.inf, None)
                for yaw, mcp in sorted({(round(y), round(c)) for y, c in over}):
                    d.qpos[:] = m.qpos0
                    d.qpos[ja[0]], d.qpos[ja[1]] = math.radians(yaw), math.radians(mcp)
                    mj.mj_kinematics(m, d)
                    k, g = points_gap(m, d, samples, link)
                    if k < worst[0]:
                        worst = (k, (yaw, mcp, names[g]))
                res["mesh_least_mm_where_spheres_overlap"] = round(worst[0], 2)
                res["mesh_at"] = worst[1]
                res["mesh_parts"] = [names[g] for g in mesh]
                res["mesh_samples"] = int(sum(len(p) for _, p in samples))
            out[v][f] = res
            print("carriage", v, f, json.dumps(res, default=str))
    return out


def collision_sets(M, mirror=None):
    """cad_v3 spheres kept (without the mount's, as in the first replay) and v4 primitives of each finger; with
    mirror, that finger's collision geoms are reflected through its own x-z plane (servo bases on the other side)."""
    mj, m = M.mj, M.m
    sets = {}
    for f in FINGERS:
        ids = []
        for g in range(m.ngeom):
            nm = mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, g) or ""
            if not nm.startswith(f + "_") or nm.startswith(f + "_mount_s") or "_v4vis_" in nm:
                continue
            if ("_s" in nm and m.geom_type[g] == mj.mjtGeom.mjGEOM_SPHERE and m.geom_group[g] == 3) or "_v4_" in nm:
                ids.append(g)
        sets[f] = ids
    if mirror:
        for g in sets[mirror]:
            m.geom_pos[g, 1] *= -1
            q = m.geom_quat[g].copy()
            m.geom_quat[g] = [q[0], -q[1], q[2], -q[3]]
    return sets


def label(mj, m, g):
    nm = mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, g) or ""
    f, rest = nm.split("_", 1)
    if "_v4_" in nm:
        return rest.split("v4_", 1)[1].rsplit("_", 1)[0] if rest.count("_") > 1 else rest.split("v4_", 1)[1]
    return rest[: rest.rindex("_s")]


def min_distance(M, sets):
    A, B = boxball(M, sets["index"]), boxball(M, sets["middle"])
    gap, pair = set_gap(A, B, cutoff=25.0)
    if pair is None:
        return gap, None
    return gap, f"{label(M.mj, M.m, sets['index'][pair[0]])} | {label(M.mj, M.m, sets['middle'][pair[1]])}"


def cmd_replay(rows):
    _, pip_mid, th_mid = angle_stats(rows)
    variants = [("as_built", "v3", None), ("as_built_index_mirrored", "v3", "index"), ("v4a", "v4a", None),
                ("v4b_same_tips", "v4b", None), ("v4b_low_same_tips", "v4b_low", None), ("v4c_same_tips", "v4c", None)]
    v3 = Model("v3", pip_mid, th_mid)
    tips = []
    for r in rows:
        v3.set(r["mount"], r["q"])
        tips.append({f: v3.tip(f) for f in FINGERS})
    G0 = {f: None for f in FINGERS}
    summary, per_run, ik_stats = {}, collections.defaultdict(dict), {}
    for name, var, mirror in variants:
        M = Model(var, pip_mid, th_mid)
        sets = collision_sets(M, mirror)
        res, close, unreach, qs, outside_v4c = [], collections.Counter(), 0, [], 0
        for r, tp in zip(rows, tips):
            q = r["q"]
            if var.startswith("v4b") or var == "v4c":
                q, ok = {}, {}
                M.set(r["mount"], {ff: (0, 0, 0) for ff in FINGERS})
                for f in FINGERS:
                    G = M.body_pos(f"{f}_yaw_frame")
                    q[f], ok[f] = ik_gimbal(tp[f], G, r["q"][f], M.info[f]["proximal"])
                    unreach += (not ok[f])
                qs.append([q[f] for f in FINGERS])
            M.set(r["mount"], q)
            if var == "v4c":
                outside_v4c += sum(not M.v4c_ok[f] for f in FINGERS)
            if var.startswith("v4b") or var == "v4c":
                for f in FINGERS:
                    err = np.linalg.norm(M.tip(f) - tp[f])
                    if err > 0.05 and ok[f]:
                        raise SystemExit(f"v4b IK check failed: {err:.3f} mm")
            dm, pair = min_distance(M, sets)
            res.append(dm)
            per_run[r["run"]][name] = min(per_run[r["run"]].get(name, 1e9), dm)
            if dm < 5.0:
                close[pair] += 1
        x = np.array(res)
        summary[name] = {"periods": int(x.size), "min_mm": round(float(x.min()), 2),
                         "p1_mm": round(float(np.percentile(x, 1)), 2), "p10_mm": round(float(np.percentile(x, 10)), 2),
                         "median_mm": round(float(np.median(x)), 2), "share_overlap": round(float(np.mean(x < 0)), 4),
                         "share_under_5mm": round(float(np.mean(x < 5)), 4),
                         "closest_parts_under_5mm (index | middle)": close.most_common(8)}
        if var == "v4c":
            summary[name]["finger_periods_outside_reach"] = int(outside_v4c)
        if var.startswith("v4b") or var == "v4c":
            Q = np.array(qs)  # periods x finger x (yaw, mcp, pip)
            Q0 = np.array([[r["q"][f] for f in FINGERS] for r in rows])
            ik_stats = {"unreachable_finger_periods": int(unreach),
                        "yaw_abs_ratio_median": round(float(np.median(np.abs(Q[..., 0]) / np.maximum(np.abs(Q0[..., 0]), 1.0))), 3),
                        "yaw_range_deg": {f: [round(float(Q[:, i, 0].min()), 1), round(float(Q[:, i, 0].max()), 1)] for i, f in enumerate(FINGERS)},
                        "mcp_range_deg": {f: [round(float(Q[:, i, 1].min()), 1), round(float(Q[:, i, 1].max()), 1)] for i, f in enumerate(FINGERS)},
                        "pip_range_deg": {f: [round(float(Q[:, i, 2].min()), 1), round(float(Q[:, i, 2].max()), 1)] for i, f in enumerate(FINGERS)},
                        "outside_driver_limits": int(np.sum((Q[..., 0] < -85) | (Q[..., 0] > 85) | (Q[..., 1] < -15) | (Q[..., 1] > 92)
                                                             | (Q[..., 2] < -18) | (Q[..., 2] > 92)))}
            summary[name]["ik"] = ik_stats
        print(name, json.dumps(summary[name]))
    return {"summary": summary, "per_run": [{"run": k, **{n: round(v, 2) for n, v in d.items()}} for k, d in per_run.items()]}


def cmd_forces(rows):
    """Pinch force along the pad normal at the servo limits and the yaw lever, over the trials' postures."""
    stats, pip_mid, th_mid = angle_stats(rows)
    a, b, h = P["finger"]["mcp_pip"], P["finger"]["pip_tip"], P["finger"]["yaw_link"]
    Q = np.array([r["q"][f] for r in rows for f in FINGERS])
    p2 = np.radians(Q[:, 2])
    lev_v3 = b + a * np.cos(p2)
    lev_v4_mcp = np.abs(a * np.cos(p2))
    F_v3 = np.minimum(1 / lev_v3, 1 / b) * 1e3  # N per N m
    F_v4 = np.minimum(1 / np.maximum(lev_v4_mcp, 1e-6), 1 / b) * 1e3
    ratio = F_v4 / F_v3
    v3 = Model("v3", pip_mid, th_mid)
    yaw_lever = {"v3": [], "v4b": []}
    for r in rows:
        v3.set(r["mount"], r["q"])
        for f in FINGERS:
            tp = v3.tip(f) - np.r_[v3.d.xpos[v3.mj.mj_name2id(v3.m, v3.mj.mjtObj.mjOBJ_BODY, f"{f}_mount")] * 1e3]
            yaw_lever["v3"].append(math.hypot(tp[1], tp[2]))
            yaw_lever["v4b"].append(math.hypot(tp[1], tp[2] + h))
    yl = {k: np.array(v) for k, v in yaw_lever.items()}
    th = Q[:, 1] + Q[:, 2]
    mu_rod = 90 - np.abs(Q[:, 2] - pip_mid)
    mu_4bar = 90 - np.abs(th - th_mid)
    T = {"scs0009_s3_trip": 0.17, "sts3032_stall_6V": P["servos"]["sts3032"]["stall_Nm_6V"]}
    pct = lambda x: [round(float(np.percentile(x, q)), 3) for q in (5, 50, 95)]  # noqa: E731
    out = {"postures": int(len(Q)), "pip_mid_deg": round(pip_mid, 2), "dip_abs_mid_deg": round(th_mid, 2),
           "pinch_ratio_v4_over_v3_p5_p50_p95": pct(ratio),
           "pinch_N_at": {k: {"v3_p5_p50_p95": pct(F_v3 * t), "v4_p5_p50_p95": pct(F_v4 * t)} for k, t in T.items()},
           "mcp_servo_share_of_v3_load_p50": round(float(np.median(lev_v4_mcp / lev_v3)), 3),
           "yaw_lever_mm_p5_p50_p95": {k: pct(v) for k, v in yl.items()},
           "lateral_force_ratio_v4b_over_v3_p5_p50_p95": pct(yl["v3"] / yl["v4b"]),
           "transmission_deg_min_p1": {"rod": [round(float(mu_rod.min()), 1), round(float(np.percentile(mu_rod, 1)), 1)],
                                       "four_bar": [round(float(mu_4bar.min()), 1), round(float(np.percentile(mu_4bar, 1)), 1)]},
           "driver_range_transmission_deg": {"rod": round(90 - max(abs(DRIVER['pip'][0] - pip_mid), abs(DRIVER['pip'][1] - pip_mid)), 1),
                                             "four_bar": round(90 - max(abs(-33 - th_mid), abs(184 - th_mid)), 1)},
           "angles": stats}
    out["v4c"] = forces_v4c(rows, pip_mid, th_mid)
    print(json.dumps({k: v for k, v in out.items() if k != "angles"}, indent=1))
    return out


def forces_v4c(rows, pip_mid, th_mid):
    """Pinch force along the pad normal at the servo limits for v4c and v3 over the trials' tips, each servo at its
    sheet stall torque at 6 V (v3: SCS0009 on each joint; v4c: STS3032 on each horn). v4c reaches the tips through the
    gimbal (ik_gimbal); its MCP torque is the two rods' with no yaw torque, its PIP torque the PIP rod's."""
    import amazinghand_differential as AD
    v3, M = Model("v3", pip_mid, th_mid), Model("v4c", pip_mid, th_mid)
    a3, a, b = P["finger"]["mcp_pip"], P["v4c"]["proximal"], P["finger"]["pip_tip"]
    T3, T4 = P["servos"]["scs0009"]["stall_Nm_6V"], P["servos"]["sts3032"]["stall_Nm_6V"]
    F3, F4, limit, outside = [], [], collections.Counter(), 0
    lim = math.radians(P["v4c"]["base"]["horn_limit_deg"])
    for r in rows:
        v3.set(r["mount"], r["q"])
        M.set(r["mount"], {ff: (0, 0, 0) for ff in FINGERS})
        for f in FINGERS:
            (yaw, mcp, pip), _ok = ik_gimbal(v3.tip(f), M.body_pos(f"{f}_yaw_frame"), r["q"][f], a)
            base, pr = M.v4c[f]
            ya, mc, pp = (np.array([math.radians(x)]) for x in (yaw, mcp, pip))
            A_, F_ = ya - base["rest"][0], mc - base["rest"][1]
            th, okb = AD.horn_angles(base, A_, F_)
            Fp = pp - pr["rest"][1]
            thp, okp = AD.horn_angles(pr, np.zeros(1), Fp)
            if not (okb.all() and okp.all() and np.all(np.abs(th) <= lim) and np.all(np.abs(thp) <= lim)):
                outside += 1
                continue
            cap_mcp = float(AD.axis_capacity(AD.statics(base, A_, F_, th)[0], 1)[0])
            cap_pip = abs(float(AD.statics(pr, np.zeros(1), Fp, thp)[0][0, 1, 0]))
            p3 = math.radians(r["q"][f][2])
            F3.append(T3 * 1e3 * min(1 / (b + a3 * math.cos(p3)), 1 / b))
            f_mcp, f_pip = cap_mcp / (b + a * math.cos(pp[0])), cap_pip / b
            F4.append(T4 * 1e3 * min(f_mcp, f_pip))
            limit["mcp" if f_mcp < f_pip else "pip"] += 1
    F3, F4 = np.array(F3), np.array(F4)
    pct = lambda x: [round(float(np.percentile(x, q)), 2) for q in (5, 50, 95)]  # noqa: E731
    pr = M.v4c["index"][1]
    pips = np.radians(np.arange(DRIVER["pip"][0], DRIVER["pip"][1] + 0.01, 0.5))
    fp = pips - pr["rest"][1]
    thp, okp = AD.horn_angles(pr, np.zeros_like(fp), fp)
    tp = np.abs(AD.statics(pr, np.zeros_like(fp), fp, thp)[0][:, 1, 0])
    reach = okp[0] & (np.abs(thp[0]) <= lim)
    pip_drive = {"driver_range_reached": bool(reach.all()), "horn_deg_min_max": [round(float(np.degrees(thp[0].min())), 1),
                                                                                round(float(np.degrees(thp[0].max())), 1)],
                 "torque_sts_min_median": [round(float(tp[reach].min()), 3), round(float(np.median(tp[reach])), 3)]}
    return {"pip_drive_over_driver_range": pip_drive, "finger_periods": int(F3.size), "finger_periods_outside_reach": outside, "torque_Nm": {"scs0009": T3, "sts3032": T4},
            "pinch_N_v3_scs0009_p5_p50_p95": pct(F3), "pinch_N_v4c_sts3032_p5_p50_p95": pct(F4),
            "ratio_v4c_over_v3_p5_p50_p95": pct(F4 / F3), "first_limit_v4c": dict(limit)}


def rhombus(c, r):
    return np.array([[c[0] + r, c[1]], [c[0], c[1] + r], [c[0] - r, c[1]], [c[0], c[1] - r], [c[0] + r, c[1]]])


def cmd_joints(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 17})
    Q = {f: np.array([r["q"][f] for r in rows]) for f in FINGERS}
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 8.6), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    for f, col in (("index", "#1f5fbf"), ("middle", "#c0392b")):
        ax.scatter(Q[f][:, 0], Q[f][:, 1], s=3, alpha=0.18, color=col, rasterized=True)
        ax.scatter([], [], s=40, color=col, label=f"{f}, bench trials ({len(Q[f])} periods)")
    yl, ml = DRIVER["yaw"], DRIVER["mcp"]
    ax.plot([yl[0], yl[1], yl[1], yl[0], yl[0]], [ml[0], ml[0], ml[1], ml[1], ml[0]], "k-", lw=1.6,
            label="serial joints: driver range (yaw ±85°, MCP −15° to 92°)")
    c = (0.0, (ml[0] + ml[1]) / 2)
    need = (yl[1] - yl[0]) / 2 + (ml[1] - ml[0]) / 2
    for theta, g, ls, col, nm in ((300, 1.0, "--", "#555555", "SCS0009, 300°"), (360, 1.0, "-.", "#2e8b57", "STS3032, 360°"),
                                 (360, 2.0, ":", "#8e44ad", "STS3032, 360°")):
        R = theta / (2 * g)
        poly = rhombus(c, R)
        ax.plot(poly[:, 0], poly[:, 1], ls=ls, color=col, lw=1.6, label=f"differential, {nm}, ratio {g:g}:1")
    ax.set_xlabel("Yaw angle, simulator convention (deg)")
    ax.set_ylabel("MCP flexion (deg)")
    ax.set_xlim(-190, 190)
    ax.set_ylim(-150, 225)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    h1, l1 = ax.get_legend_handles_labels()
    ax.set_title("(a) Joint space of the yaw and MCP joints", fontsize=17)
    ax = axes[1]
    ax.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], "k-", lw=1.6, label="serial joints, one servo each")
    for g, ls, col in ((1.0, "-.", "#2e8b57"), (1.3, "--", "#d35400")):
        poly = rhombus((0, 0), 2 * g)
        ax.plot(poly[:, 0], poly[:, 1], ls=ls, color=col, lw=1.6, label=f"torques of a differential, ratio {g:g}:1")
    ax.set_xlabel("Yaw torque / servo limit")
    ax.set_ylabel("MCP torque / servo limit")
    ax.set_xlim(-2.9, 2.9)
    ax.set_ylim(-2.9, 2.9)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    h2, l2 = ax.get_legend_handles_labels()
    ax.set_title("(b) Joint torques at the servo limit", fontsize=17)
    fig.legend(h1 + h2[1:], l1 + l2[1:], loc="lower center", ncol=2, fontsize=15, frameon=False)
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    MEDIA.mkdir(parents=True, exist_ok=True)
    path = MEDIA / "20261118-yaw_mcp_joint_space.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    per = {}
    for f in FINGERS:
        lo, hi = np.percentile(Q[f][:, :2], 0.5, axis=0), np.percentile(Q[f][:, :2], 99.5, axis=0)
        half = (hi - lo) / 2
        per[f] = {"yaw_band": [round(float(lo[0]), 1), round(float(hi[0]), 1)],
                  "mcp_band": [round(float(lo[1]), 1), round(float(hi[1]), 1)],
                  "motor_range_needed_at_1to1_deg": round(float(2 * half.sum()), 1),
                  "max_ratio_scs0009": round(float(300 / (2 * half.sum())), 2),
                  "max_ratio_sts3032": round(float(360 / (2 * half.sum())), 2)}
    out = {"figure": str(path.relative_to(ROOT)),
           "driver_rectangle_motor_range_needed_at_1to1_deg": round(2 * need, 1),
           "max_ratio_containing_driver_rectangle": {"scs0009": round(300 / (2 * need), 2), "sts3032": round(360 / (2 * need), 2)},
           "per_finger_trials": per}
    print(json.dumps(out, indent=1))
    return out


def cmd_render(rows):
    os.environ.setdefault("MUJOCO_GL", "egl")
    from PIL import Image, ImageDraw, ImageFont
    _, pip_mid, th_mid = angle_stats(rows)
    poses = {"open": (0, 0, 10), "curl": (0, 70, 80), "yaw 35°": (35, 30, 40)}
    views = {"side": (90, -5), "front": (180, -5)}
    tiles = {}
    for v in ("v3", "v4a", "v4b", "v4c"):
        M = Model(v, pip_mid, th_mid)
        mj, m, d = M.mj, M.m, M.d
        root = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "index_mount")

        def under(bb):
            while bb > 0:
                if bb == root:
                    return True
                bb = m.body_parentid[bb]
            return False
        for g in range(m.ngeom):
            nm = mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, g) or ""
            show = under(m.geom_bodyid[g]) and ("visual" in nm or "_v4_" in nm or "_v4vis_" in nm) and "_tabs" not in nm \
                and ("_case" not in nm or v == "v4c") and "_boss" not in nm and "bridge" not in nm
            m.geom_group[g] = 2 if show else 5
        opt = mj.MjvOption()
        for k in range(6):
            opt.geomgroup[k] = 1 if k == 2 else 0
            opt.sitegroup[k] = 0
        m.vis.headlight.ambient[:] = [0.45, 0.45, 0.45]
        r = mj.Renderer(m, 640, 420)  # height, width
        for pn, q in poses.items():
            M.set({f: M.mount0[f][:2] for f in FINGERS}, {f: q for f in FINGERS})
            c = d.xpos[root].copy() + np.array([0.008, 0, -0.05])
            for vn, (az, el) in views.items():
                cam = mj.MjvCamera()
                cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = c, 0.2, az, el
                r.update_scene(d, camera=cam, scene_option=opt)
                img = r.render().copy()
                r.enable_segmentation_rendering()
                r.update_scene(d, camera=cam, scene_option=opt)
                seg = r.render()[..., 0]
                r.disable_segmentation_rendering()
                img[seg < 0] = 244
                tiles[(v, pn, vn)] = Image.fromarray(img)
        r.close()
    MEDIA.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", 30)
    except OSError:
        font = ImageFont.load_default()
    W, H, top = 420, 640, 54
    paths = {}
    for vn in views:
        sheet = Image.new("RGB", (W * 3, H * 4 + top), "white")
        dr = ImageDraw.Draw(sheet)
        for ci, pn in enumerate(poses):
            dr.text((ci * W + 12, 10), pn, fill="black", font=font)
        for ri, v in enumerate(("v3", "v4a", "v4b", "v4c")):
            for ci, pn in enumerate(poses):
                sheet.paste(tiles[(v, pn, vn)], (ci * W, top + ri * H))
            dr.text((10, top + ri * H + 10), {"v3": "v3 as built", "v4a": "v4a", "v4b": "v4b", "v4c": "v4c"}[v], fill="black",
                    font=font)
        paths[vn] = MEDIA / f"20261118-finger_layouts_{vn}.png"
        sheet.save(paths[vn])
        print("wrote", paths[vn].relative_to(ROOT))
    old = MEDIA / "20261118-finger_layouts.png"
    if old.exists():
        old.unlink()
    return {k: str(v.relative_to(ROOT)) for k, v in paths.items()}


def cmd_prims(rows):
    """Every v4 primitive and moved servo mesh of the index finger at the zero pose, in the index mount's frame (mm), for
    the STEP export (scripts/cad/cad_v4_step.py)."""
    _, pip_mid, th_mid = angle_stats(rows)
    out = {}
    for v in ("v4a", "v4b"):
        M = Model(v, pip_mid, th_mid)
        mj, m, d = M.mj, M.m, M.d
        M.set({f: M.mount0[f][:2] for f in FINGERS}, {f: (0, 0, 0) for f in FINGERS})
        b0 = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "index_mount")
        o, R0 = d.xpos[b0] * 1e3, d.xmat[b0].reshape(3, 3)
        parts = []
        for g in range(m.ngeom):
            nm = mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, g) or ""
            if not nm.startswith("index_v4_"):
                continue
            t = {int(mj.mjtGeom.mjGEOM_BOX): "box", int(mj.mjtGeom.mjGEOM_CAPSULE): "capsule"}.get(int(m.geom_type[g]))
            Rg = R0.T @ d.geom_xmat[g].reshape(3, 3)
            parts.append({"name": nm[len("index_v4_"):], "type": t, "size_mm": (m.geom_size[g] * 1e3).round(4).tolist(),
                          "pos_mm": (R0.T @ (d.geom_xpos[g] * 1e3 - o)).round(4).tolist(), "rot": Rg.round(6).tolist(),
                          "body": mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g]))})
        out[v] = {"frame": "index mount (cad_v3 P axes, origin at the carriage's yaw-axis point), zero pose", "parts": parts}
        print(v, len(parts), "parts")
    return out


def cmd_play(variant, kinematic=False, frames=None):
    """A written scene in MuJoCo's viewer, posed from the Control sliders, with the least index-middle gap of the
    replay's collision sets overlaid. Under physics touching parts push each other apart; with kinematic=True the
    joints take the sliders' values and parts may overlap (negative gap). frames stops after that many frames."""
    import time
    import types
    mj = mujoco()
    from mujoco import viewer as mjv
    m = mj.MjModel.from_xml_path(str(MJCF_OUT / f"{variant}.xml"))
    d = mj.MjData(m)
    mj.mj_resetDataKeyframe(m, d, m.key("lifted").id)
    M = types.SimpleNamespace(mj=mj, m=m, d=d)
    sets = collision_sets(M)
    names = ("yaw", "mcp", "pip") + (() if variant == "v4c" else DRIVE_JOINTS)
    adr = {f: {n: m.jnt_qposadr[m.joint(f"{f}_{n}").id] for n in names} for f in FINGERS}
    closed = {f: v4c_closed_form(f) for f in FINGERS} if variant == "v4c" else None
    acts = [(a, m.jnt_qposadr[m.actuator_trnid[a, 0]]) for a in range(m.nu)
            if m.actuator_trntype[a] == mj.mjtTrn.mjTRN_JOINT]
    steps = max(1, round(1.0 / 60.0 / m.opt.timestep))
    title = f"{variant}, {'kinematic' if kinematic else 'physics'}: least index-middle gap"
    k, gap, pair = 0, float("nan"), None
    with mjv.launch_passive(m, d) as viewer:
        while viewer.is_running() and (frames is None or k < frames):
            t0 = time.time()
            if kinematic:
                for a, i in acts:
                    d.qpos[i] = d.ctrl[a]
                for f in FINGERS:
                    if closed:
                        v4c_set_qpos(m, d.qpos, f, closed[f], *(d.qpos[adr[f][n]] for n in ("yaw", "mcp", "pip")))
                        continue
                    for n, v in drive_angles(d.qpos[adr[f]["mcp"]], d.qpos[adr[f]["pip"]]).items():
                        d.qpos[adr[f][n]] = v
                d.qvel[:] = 0.0
                mj.mj_forward(m, d)
            else:
                for _ in range(steps):
                    mj.mj_step(m, d)
            gap, pair = min_distance(M, sets)
            text = "over 25 mm" if pair is None else f"{gap:.2f} mm, {pair}"
            viewer.set_texts((None, mj.mjtGridPos.mjGRID_BOTTOMLEFT, title, text))
            viewer.sync()
            k += 1
            time.sleep(max(0.0, 1.0 / 60.0 - (time.time() - t0)))
    return {"variant": variant, "kinematic": kinematic, "frames": k, "last_gap_mm": round(float(gap), 3), "pair": pair}


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "play":
        args = sys.argv[2:]
        frames = int(args[args.index("--frames") + 1]) if "--frames" in args else None
        print(cmd_play(next((a for a in args if a in ("v4a", "v4b", "v4c")), "v4a"), "--kinematic" in args, frames), flush=True)
        os._exit(0)  # skips glfw's exit hook, which hung under Xvfb's software GL after the window closed
    rows = load_trials()
    print(f"{len(rows)} periods from the bench runs")
    DATA.mkdir(parents=True, exist_ok=True)
    cmds = {"build": cmd_build, "check": cmd_check, "replay": cmd_replay, "forces": cmd_forces, "joints": cmd_joints,
            "render": cmd_render, "prims": cmd_prims}
    todo = [c for c in cmds if c != "prims"] + ["prims"] if cmd == "all" else [cmd]
    for c in todo:
        res = cmds[c](rows)
        (DATA / f"20261118-{c}.json").write_text(json.dumps({"script": "scripts/cad/cad_v4_layouts.py", "command": c,
                                                             "params": str(PARAMS.relative_to(ROOT)), "result": res},
                                                            indent=1, default=str))
        print("wrote", (DATA / f"20261118-{c}.json").relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
