"""Independent registration of the cad_v1 carriage and yaw link onto the three v3 fingers in frame A.

Method: sample points on each mesh surface; initial poses from the inertia principal axes (all 8
sign combinations: 4 proper, 4 improper); point-to-point ICP with Kabsch (reflection allowed only
in the improper runs). The best proper fit maps the cad_v1 finger frame F (origin = yaw centre,
x_F = yaw axis toward the servo side, z_F up; assets/mjcf/experimental/20261013-cad_v1/assembly.json,
T_F_from_part_design after the export rotation R0 of scripts/hardware/build_cad_v1_finger.py) into A.
The best improper fit tests chirality: a mirrored copy should not fit as well as the proper one.

Run: ~/.claude/bin/resguard.sh run --mem 3G -- .venv/bin/python <this file>
"""
import json
import numpy as np
import trimesh
from scipy.spatial import cKDTree

ROOT = '/home/humanoid/Programs/hand/'
V1 = ROOT + 'assets/mjcf/experimental/20261013-cad_v1/'
EX = ROOT + 'logs/20261104-cad_v3/extract/hand/'
OUT = ROOT + 'logs/20261104-cad_v3/verify_gantry_identity/'
asm = json.load(open(V1 + 'assembly.json'))

FINGERS = {  # instance -> part numbers in file "hand" (from parts.json chains and volumes)
    'Finger v3 <1>': dict(carriage='028', yaw_link='029'),
    'Finger v3 <2>': dict(carriage='094', yaw_link='088'),
    'Finger v3 <3>': dict(carriage='016', yaw_link='015'),
}


def unit(v):
    return np.asarray(v, float) / np.linalg.norm(v)


def design_rotation(mesh):
    """Copy of scripts/build_cad_v1_finger.design_rotation (export tilt of the carriage)."""
    order = np.argsort(-mesh.facets_area)
    for i in order[:5]:
        n = mesh.facets_normal[i]
        k = int(np.argmax(np.abs(n)))
        target = np.zeros(3)
        target[k] = np.sign(n[k])
        ang = np.degrees(np.arccos(np.clip(n @ target, -1, 1)))
        if ang < 3.0:
            if ang < 1e-4:
                return np.eye(3)
            ax = unit(np.cross(n, target))
            return trimesh.transformations.rotation_matrix(np.radians(ang), ax)[:3, :3]
    return np.eye(3)


def T4(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def kabsch(A, B, allow_reflection):
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    R = Vt.T @ U.T
    if not allow_reflection and np.linalg.det(R) < 0:
        D = np.diag([1, 1, -1])
        R = Vt.T @ D @ U.T
    if allow_reflection and np.linalg.det(R) > 0:
        # force an improper solution so the mirrored hypothesis is tested on its own
        D = np.diag([1, 1, -1])
        R = Vt.T @ D @ U.T
    return R, cb - R @ ca


def icp(src_pts, tgt_pts, R0, t0, improper, iters=80):
    tree = cKDTree(tgt_pts)
    R, t = R0, t0
    for _ in range(iters):
        moved = src_pts @ R.T + t
        d, j = tree.query(moved)
        keep = d < np.percentile(d, 95)
        R, t = kabsch(src_pts[keep], tgt_pts[j[keep]], improper)
    moved = src_pts @ R.T + t
    d, _ = tree.query(moved)
    d2, _ = cKDTree(moved).query(tgt_pts)
    return R, t, float(np.sqrt(np.mean(d ** 2))), float(np.sqrt(np.mean(d2 ** 2))), float(np.percentile(np.r_[d, d2], 99))


def register(src_mesh, tgt_mesh, seed=0):
    rs = np.random.default_rng(seed)
    src_pts = trimesh.sample.sample_surface(src_mesh, 25000, seed=seed)[0]
    tgt_pts = trimesh.sample.sample_surface(tgt_mesh, 25000, seed=seed + 1)[0]
    es = src_mesh.principal_inertia_vectors  # rows
    et = tgt_mesh.principal_inertia_vectors
    cs, ct = src_mesh.center_mass, tgt_mesh.center_mass
    best = {}
    for s in np.array(np.meshgrid([1, -1], [1, -1], [1, -1])).T.reshape(-1, 3):
        R0 = et.T @ np.diag(s) @ es
        improper = np.linalg.det(R0) < 0
        R, t, rms_st, rms_ts, p99 = icp(src_pts, tgt_pts, R0, ct - R0 @ cs, improper)
        key = 'improper' if improper else 'proper'
        score = 0.5 * (rms_st + rms_ts)
        if key not in best or score < best[key]['score']:
            best[key] = dict(R=R, t=t, score=score, rms_src_to_tgt=rms_st, rms_tgt_to_src=rms_ts, p99=p99)
    return best


def main():
    # The MJCF meshes are the parts in their body frames, metres (build_cad_v1_finger.body_meshes):
    # carriage.stl in the mount body = F (origin at the yaw centre), yaw_link.stl in yaw_frame (origin at
    # the yaw centre, zero yaw). Scale to mm; F <- stl is then the identity.
    car_v1 = trimesh.load(V1 + 'meshes/carriage.stl'); car_v1.apply_scale(1000.0)
    yaw_v1 = trimesh.load(V1 + 'meshes/yaw_link.stl'); yaw_v1.apply_scale(1000.0)
    print('cad_v1 carriage volume', round(car_v1.volume, 1), 'mm^3; yaw link', round(yaw_v1.volume, 1))
    T_F_car = np.eye(4)
    T_F_yaw = np.eye(4)
    # cross-check route: the raw CAD carriage (assets/cad_v1, mm) through R0 and T_F_from_part_design
    raw = trimesh.load(ROOT + 'assets/cad_v1/carriage.stl', force='mesh')
    R0 = design_rotation(raw)
    T_raw = np.array(asm['T_F_from_part_design_mm']['carriage']) @ T4(R0, np.zeros(3))
    raw.apply_transform(T_raw)
    d = cKDTree(car_v1.vertices).query(trimesh.sample.sample_surface(raw, 5000, seed=3)[0])[0]
    print('raw CAD carriage through R0 and T_F_from_part_design vs MJCF mesh: rms', round(float(np.sqrt(np.mean(d**2))), 4), 'mm,',
          'export rotation', round(float(np.degrees(np.arccos(np.clip((np.trace(R0) - 1) / 2, -1, 1)))), 4), 'deg')
    mcp_F = np.array(asm['report']['kinematics_mm']['mcp_centre'])
    mcp_axis_F = np.array(asm['report']['kinematics_mm']['axes']['mcp'])
    out = {}
    for name, ids in FINGERS.items():
        car = trimesh.load(EX + ids['carriage'] + '.stl')
        yl = trimesh.load(EX + ids['yaw_link'] + '.stl')
        bc = register(car_v1, car)
        by = register(yaw_v1, yl)
        Rc, tc = bc['proper']['R'], bc['proper']['t']
        T_A_car = T4(Rc, tc)                                  # A <- cad_v1 carriage stl
        T_A_F = T_A_car @ np.linalg.inv(T_F_car)              # A <- F (carriage-borne)
        yaw_c = T_A_F[:3, 3]
        x_F, y_F, z_F = T_A_F[:3, 0], T_A_F[:3, 1], T_A_F[:3, 2]
        # yaw link: carries the yaw centre and the MCP centre/axis at the CAD pose
        Ry, ty = by['proper']['R'], by['proper']['t']
        T_A_Fy = T4(Ry, ty) @ np.linalg.inv(T_F_yaw)
        yaw_c_from_link = T_A_Fy[:3, 3]
        mcp_c = T_A_Fy[:3, :3] @ mcp_F + T_A_Fy[:3, 3]
        mcp_a = T_A_Fy[:3, :3] @ mcp_axis_F
        # foot of the common perpendicular from the MCP axis onto the carriage-borne yaw axis
        n = unit(np.cross(x_F, mcp_a))
        sol = np.linalg.solve(np.column_stack([x_F, -mcp_a, n]), mcp_c - yaw_c)
        foot = yaw_c + sol[0] * x_F
        # yaw angle: rotation of the yaw link's F about x_F relative to the carriage's F
        Rrel = T_A_F[:3, :3].T @ T_A_Fy[:3, :3]
        yaw_deg = float(np.degrees(np.arctan2(Rrel[2, 1], Rrel[1, 1])))
        out[name] = dict(
            carriage_fit=dict(proper_rms_mm=[round(bc['proper']['rms_src_to_tgt'], 4), round(bc['proper']['rms_tgt_to_src'], 4)],
                              proper_p99_mm=round(bc['proper']['p99'], 4),
                              improper_rms_mm=[round(bc['improper']['rms_src_to_tgt'], 4), round(bc['improper']['rms_tgt_to_src'], 4)],
                              improper_p99_mm=round(bc['improper']['p99'], 4)),
            yaw_link_fit=dict(proper_rms_mm=[round(by['proper']['rms_src_to_tgt'], 4), round(by['proper']['rms_tgt_to_src'], 4)],
                              improper_rms_mm=[round(by['improper']['rms_src_to_tgt'], 4), round(by['improper']['rms_tgt_to_src'], 4)]),
            yaw_centre_A_carriage=np.round(yaw_c, 3).tolist(),
            yaw_centre_A_yaw_link=np.round(yaw_c_from_link, 3).tolist(),
            yaw_centre_A_common_perp=np.round(foot, 3).tolist(),
            mcp_offset_along_yaw_axis_mm=round(float(sol[0]), 3),
            yaw_to_mcp_mm=round(float(np.linalg.norm(foot - (mcp_c + sol[1] * mcp_a))), 3),
            x_F=np.round(x_F, 5).tolist(), y_F=np.round(y_F, 5).tolist(), z_F=np.round(z_F, 5).tolist(),
            det_R_carriage=round(float(np.linalg.det(Rc)), 6), yaw_angle_deg=round(yaw_deg, 3))
        print(name, json.dumps(out[name], indent=1))
    json.dump(out, open(OUT + 'register_carriage.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
