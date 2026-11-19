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
    b.mass, b.inertia = 1e-4, [1e-10, 1e-10, 1e-10]
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
    return spec, info


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

    def set(self, mounts, q):
        m, d = self.m, self.d
        d.qpos[:] = m.qpos0
        for f in FINGERS:
            d.qpos[m.jnt_qposadr[self.jid(f"{f}_x")]] = (mounts[f][0] - self.mount0[f][0]) * 1e-3
            d.qpos[m.jnt_qposadr[self.jid(f"{f}_y")]] = (mounts[f][1] - self.mount0[f][1]) * 1e-3
            yaw, mcp, pip = (math.radians(v) for v in q[f])
            for name, v in (("yaw", yaw), ("mcp", mcp), ("pip", pip), ("idler_q", mcp + pip), ("rod_q", -pip),
                            ("coupler_q", -(mcp + pip)), ("pipcrank_q", mcp + pip)):
                j = self.jid(f"{f}_{name}")
                if j >= 0:
                    d.qpos[m.jnt_qposadr[j]] = v
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
    for v in ("v4a", "v4b"):
        spec, info = build_spec(v, pip_mid, th_mid)
        spec.compile()
        spec.meshdir = "../20261104-cad_v3/meshes"
        spec.texturedir = "../20261104-cad_v3/meshes"
        spec.modelname = f"cad_v4_{v}_scene_screwdriver_medium"
        (MJCF_OUT / f"{v}.xml").write_text(spec.to_xml())
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
    return res


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
                ("v4b_same_tips", "v4b", None), ("v4b_low_same_tips", "v4b_low", None)]
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
        res, close, unreach, qs = [], collections.Counter(), 0, []
        for r, tp in zip(rows, tips):
            q = r["q"]
            if var.startswith("v4b"):
                q, ok = {}, {}
                M.set(r["mount"], {ff: (0, 0, 0) for ff in FINGERS})
                for f in FINGERS:
                    G = M.body_pos(f"{f}_yaw_frame")
                    q[f], ok[f] = ik_gimbal(tp[f], G, r["q"][f], M.info[f]["proximal"])
                    unreach += (not ok[f])
                qs.append([q[f] for f in FINGERS])
            M.set(r["mount"], q)
            if var.startswith("v4b"):
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
        if var.startswith("v4b"):
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
    print(json.dumps({k: v for k, v in out.items() if k != "angles"}, indent=1))
    return out


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
    for v in ("v3", "v4a", "v4b"):
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
                and "_case" not in nm and "_boss" not in nm and "bridge" not in nm
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
        sheet = Image.new("RGB", (W * 3, H * 3 + top), "white")
        dr = ImageDraw.Draw(sheet)
        for ci, pn in enumerate(poses):
            dr.text((ci * W + 12, 10), pn, fill="black", font=font)
        for ri, v in enumerate(("v3", "v4a", "v4b")):
            for ci, pn in enumerate(poses):
                sheet.paste(tiles[(v, pn, vn)], (ci * W, top + ri * H))
            dr.text((10, top + ri * H + 10), {"v3": "v3 as built", "v4a": "v4a", "v4b": "v4b"}[v], fill="black", font=font)
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


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
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
