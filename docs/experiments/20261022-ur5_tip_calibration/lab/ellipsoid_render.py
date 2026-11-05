#!/usr/bin/env python3
"""MuJoCo renders of the arm's velocity and force ellipsoids at the probe's sphere (owner, 2026-11-02: "your analysis/
visual renders in mujoco should reveal a similarly good manipulability and force ellipsoid at this config").

Two configurations press the same point, seat 2's reference (probe-ref-00 of lab 20261102-plan_probe_table_seat2_v3id5,
at its 1.5 mm rest), with the strip at its two-key frame:
  owner's   the press configuration of 11-02 (IK branch 2, wrist flipped, the stem straight down, joint 6 near 0), the
            joints nearest the action's q_rest;
  old       the 10-30 plans' (the homing's branch, joint 5 near +60 deg): the stem of the 10-30 seat 2 plan's
            probe-ref-00 in the strip's frame, turned about itself for the joints nearest the homing seating's.
At the sphere's centre, with J the arm's 3 x 6 translational Jacobian (mm/rad): the velocity ellipsoid
{J qdot : |qdot| <= 1 rad/s} has semi-axes sqrt(eig(J J^T)) mm/s; the force ellipsoid {f : |J^T f| <= 1 N m} has
semi-axes 1000 / sqrt(eig(J J^T)) N. Both are drawn at the same scale in both panels: velocity 0.1 mm per mm/s (blue),
force 25 mm per N (orange); the press direction (into the pad) is the red rod.

Scene: the robot's URDF visual meshes (scripts/ur5/ur5_cell_render.py), the HEX-E, its port, the probe base and the 5 mm
sphere's tool (the seat block's CAD), the strip and its three held tips at the two-key frame, the 2x4 under the strip
(x_A 50..139, 38 mm), the table top at z_B 0.

    PYTHONPATH=<scratch>/pylib MUJOCO_GL=egl rh56_controller/.venv312/bin/python ellipsoid_render.py OUT_DIR
"""
import json
import math
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("UR5_MOUNT", "v7")
os.environ.setdefault("UR5M_REV", "v7")
import numpy as np

ROOT = Path("/home/will/workspace/hand")
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path
import ur5_inverted_rig as RIG  # noqa: E402
import ur5_cell_render as RN  # noqa: E402

C, CELL = RIG.C, RIG.CELL
L = ROOT / "docs/experiments/20261022-ur5_tip_calibration/lab"
CAD = ROOT / "docs/experiments/20261030-inverted_rig_seat_block/cad"
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
T_BA = np.array(json.load(open(L / "20261102-strip_frame_two_keys.json"))["T_BA"], float)
prog = json.load(open(L / "20261102-plan_probe_table_seat2_v3id5.json"))
old = json.load(open(L / "20261030-plan_probe_seat2_walls_v3id3.json"))
src = json.loads((RIG.ROOT / prog["checks"]["source"]).read_text())
RIG.use_cell("table", src.get("j6_ref"))
RIG.HOME_KEY[0] = int(prog["homing"]["key"])
ctx = RIG.context(dict(orient=src["orient"], seat_h=RIG.SEAT_BLOCK["seat_h"]), tool=src["tool"])
ctx["T_BA"] = T_BA
T_EM = ctx["T_EM"]
g = prog["geometry"]
c_M = np.array(g["probe"]["centre_M"], float)
R_S = float(g["probe"]["radius"])
VSCALE, FSCALE = 0.1, 25.0                      # mm drawn per mm/s, per N


def rest_of(plan, x, T_BA_):
    T_BP = T_BA_ @ np.array(plan["geometry"]["seat"]["T_AP"], float)
    tg = x["target"]
    d = np.array(tg["d"], float) / np.linalg.norm(tg["d"])
    n = np.array(tg["n"], float) / np.linalg.norm(tg["n"])
    c_B = C.apply(T_BP, (np.array(tg["p"], float) + R_S * n - float(x.get("standoff", 1.5)) * d)[None])[0]
    R_BM = T_BP[:3, :3] @ np.array(tg["R_PM"], float)
    return C.Tm(R_BM, c_B - R_BM @ c_M) @ C.inv(T_EM), T_BP[:3, :3] @ d, c_B


ref = next(x for x in prog["actions"] if x["id"] == "probe-ref-00")
T_own, d_B, c_B = rest_of(prog, ref, T_BA)
q_ref = np.array(ref["q_rest"], float)
q_own = RIG.ik_near(T_own[None], q_ref, branch=RIG.branch_of(q_ref))[0]

# the old configuration: the 10-30 stem (z_M in A) at the same sphere centre, turned about itself, nearest the homing's
ref_old = next(x for x in old["actions"] if x["id"] == "probe-ref-00")
R_AM_old = np.array(old["geometry"]["seat"]["T_AP"], float)[:3, :3] @ np.array(ref_old["target"]["R_PM"], float)
z_old_B = T_BA[:3, :3] @ R_AM_old[:, 2]
q_seat = np.array(ctx["q_seat"], float)
best = None
for spin in np.radians(np.arange(0.0, 360.0, 5.0)):
    x0 = np.cross([0.0, 0.0, 1.0], z_old_B)
    x0 /= np.linalg.norm(x0)
    y0 = np.cross(z_old_B, x0)
    xs = math.cos(spin) * x0 + math.sin(spin) * y0
    R_BM = np.column_stack([xs, np.cross(z_old_B, xs), z_old_B])
    T = C.Tm(R_BM, c_B - R_BM @ c_M) @ C.inv(T_EM)
    q = RIG.ik_near(T[None], q_seat, branch=RIG.branch_of(q_seat))[0]
    if not np.isfinite(q).all():
        continue
    j5 = math.degrees(q[4])
    if not (30.0 <= j5 <= 100.0 and abs(math.sin(q[4])) >= 0.5):
        continue
    cost = float(np.abs(q - q_seat).max())
    if best is None or cost < best[0]:
        best = (cost, q, T)
q_old = best[1]


def jacobian(q, p):
    F = [np.eye(4)]
    for i in range(6):
        F.append(F[-1] @ RIG._dh(i, q[i]))
    J = np.zeros((3, 6))
    for i in range(6):
        z, o = F[i][:3, 2], F[i][:3, 3]
        J[:, i] = np.cross(z, p - o)
    return J


def ellipsoids(q):
    """(velocity semi-axes mm/s, force semi-axes N, axes (columns), numbers) at the sphere's centre for joints q."""
    T = RIG.fk_many(q[None])[0]
    p = C.apply(T @ T_EM, c_M[None])[0]
    J = jacobian(q, p)
    lam, V = np.linalg.eigh(J @ J.T)
    down = d_B / np.linalg.norm(d_B)
    M = J @ J.T
    nums = dict(cond=math.sqrt(lam[-1] / lam[0]), yoshikawa=math.sqrt(np.linalg.det(M)) / 1e6,
                f_press=1e3 / math.sqrt(down @ M @ down), v_press=1.0 / math.sqrt(down @ np.linalg.inv(M) @ down),
                p=p, q=np.degrees(q))
    return np.sqrt(lam), 1e3 / np.sqrt(lam), V, nums


def scene_xml(w, h):
    import trimesh
    work = Path(tempfile.mkdtemp(prefix="ellipsoid_render_"))
    S = RN.Scene(work)
    for ln, (T0, path) in RN.urdf_visuals().items():
        groups = {}
        for mat, P, F in RN.read_dae(path):
            cls = next((c for k, c in RN.DAE_CLASS if mat.lower().startswith(k)), "ur_dark")
            groups.setdefault(cls, []).append(RN.tm(P * 1e3, F))
        for cls, ms in groups.items():
            S.mesh(RN.transformed(trimesh.util.concatenate(ms), T0), cls, body=ln, name=f"{ln}_{cls}")
    S.mesh(RN.transformed(CELL.hex_e_mesh(None), T_EM), "hex_e", body="tool", name="hex_e", smooth_deg=40.0)
    S.mesh(RN.port_mesh_E(), "ur_black", body="tool", name="hex_e_port", smooth_deg=50.0)
    S.mesh(RN.transformed(CELL.finger_mesh("adapter"), T_EM), "purple", body="tool", name="tool_adapter")
    for name, f in (("probe_base", "20261030-seat_block_probe_base.stl"), ("tool", "20261030-seat_block_tool_sphere5.stl")):
        S.mesh(RN.transformed(trimesh.load(CAD / f, force="mesh"), T_EM), "purple", body="tool", name=f"probe_{name}")
    S.mesh(RN.transformed(trimesh.load(CAD / "20261030-seat_block_seat_block.stl", force="mesh"), T_BA), "purple",
           name="strip")
    for i, st in enumerate(ctx["g"]["seats"]):
        T = T_BA @ np.asarray(st["T_AF"], float)
        for name, m, mat in RN.tip_parts_F():
            S.mesh(RN.transformed(m, T), mat, name=f"held{i}_{name}")
    b = RIG.TABLE["board"]
    S.mesh(RN.transformed(RN.box_mesh((b["x"][0], -260.0, -b["h"]), (b["x"][1], 260.0, 0.0)), T_BA), "pine",
           name="board_2x4")
    S.raw('<geom type="box" size="0.75 0.62 0.019" pos="-0.25 -0.30 -0.019" rgba="0.80 0.70 0.55 1"/>')
    S.raw('<geom type="plane" size="4 4 0.1" pos="0 0 -0.76" rgba="0.55 0.56 0.55 1"/>')
    apex = C.apply(T_BA, np.asarray(ctx["g"]["seats"][1]["apex"], float)[None])[0]
    cams = (RN.look_at(apex + [520.0, -620.0, 330.0], apex + [-60.0, 60.0, 180.0], 42.0, "side")
            + RN.look_at(apex + [230.0, -300.0, 150.0], c_B, 36.0, "close"))
    return S, cams, work


def ell_geoms(c, axes_mm, V, rgba, name):
    R = V.copy()
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1
    return (f'<geom name="{name}" type="ellipsoid" size="{RN.fmt(np.asarray(axes_mm) * 1e-3)}" pos="{RN.fmt(c * 1e-3)}" '
            f'quat="{RN.fmt(RN.quat(R))}" rgba="{RN.fmt(rgba, 3)}" group="1"/>')


def main():
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    w, h = 900, 700
    S, cams, work = scene_xml(w, h)
    panels, rows = [], {}
    for key, q, label in (("owner", q_own, "owner's press configuration (11-02)"),
                          ("old", q_old, "the 10-30 plans' configuration")):
        va, fa, V, nums = ellipsoids(q)
        rows[key] = dict(q_deg=np.round(nums["q"], 1).tolist(), velocity_axes_mm_s=np.round(va, 1).tolist(),
                         force_axes_N=np.round(fa, 3).tolist(), cond=round(nums["cond"], 3),
                         yoshikawa_1e6mm3=round(nums["yoshikawa"], 2), force_along_press_N_per_Nm=round(nums["f_press"], 3),
                         velocity_along_press_mm_s=round(nums["v_press"], 1))
        p = nums["p"]
        extra = (ell_geoms(p, va * VSCALE, V, (0.20, 0.45, 0.95, 0.35), "vel")
                 + ell_geoms(p, fa * FSCALE, V, (0.98, 0.55, 0.10, 0.45), "force")
                 + f'<geom type="cylinder" fromto="{RN.fmt(p * 1e-3)} {RN.fmt((p + 40.0 * d_B) * 1e-3)}" size="0.0015" '
                   f'rgba="0.85 0.1 0.1 1" group="1"/>')
        xml = S.xml(w, h, cams, RN.lights_xml(), RN.materials_xml({}))
        xml = xml.replace("</asset>", '<material name="pine" rgba="0.86 0.73 0.52 1" specular="0.05" shininess="0.1"/></asset>')
        xml = xml.replace("</worldbody>", extra + "</worldbody>")
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        RN.set_pose(m, d, q)
        ren = mujoco.Renderer(m, h, w)
        ims = []
        for cam in ("side", "close"):
            ren.update_scene(d, camera=cam)
            ims.append(Image.fromarray(ren.render()))
        ren.close()
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", 19)
        except OSError:
            font = ImageFont.load_default()
        text = (f"{label}, joints {np.round(nums['q'], 0).astype(int).tolist()} deg: condition {nums['cond']:.2f}, "
                f"Yoshikawa {nums['yoshikawa']:.0f} (1e6 mm^3); along the press {nums['v_press']:.0f} mm/s per rad/s, "
                f"{nums['f_press']:.2f} N per N m")
        legend = (f"blue: velocity ellipsoid ({VSCALE:g} mm drawn per mm/s at 1 rad/s of joint speed); orange: force "
                  f"ellipsoid ({FSCALE:g} mm drawn per N at 1 N m of joint torque); red: the press, into the pad")
        bar = Image.new("RGB", (2 * w, 64), (255, 255, 255))
        ImageDraw.Draw(bar).text((12, 6), text, fill=(0, 0, 0), font=font)
        ImageDraw.Draw(bar).text((12, 34), legend, fill=(70, 70, 70), font=font)
        pan = Image.new("RGB", (2 * w, h + 64), (255, 255, 255))
        pan.paste(bar, (0, 0))
        pan.paste(ims[0], (0, 64))
        pan.paste(ims[1], (w, 64))
        pan.save(OUT / f"20261102-ellipsoids_{key}.png")
        panels.append(pan)
    full = Image.new("RGB", (2 * w, 2 * (h + 64)), (255, 255, 255))
    full.paste(panels[0], (0, 0))
    full.paste(panels[1], (0, h + 64))
    full.save(OUT / "20261102-ellipsoids_owner_vs_old.png")
    json.dump(dict(what=__doc__.split("\n\n")[0], point=dict(seat=2, action="probe-ref-00", centre_B=np.round(c_B, 2).tolist(),
                                                             press_B=np.round(d_B, 4).tolist()),
                   scale=dict(velocity_mm_per_mm_s=VSCALE, force_mm_per_N=FSCALE), rows=rows),
              open(OUT / "20261102-ellipsoids.json", "w"), indent=1)
    print(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
