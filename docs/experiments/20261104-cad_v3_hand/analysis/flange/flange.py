#!/usr/bin/env python3
"""tool0 (E) of the lab's UR5 CB3 to the hand frame A of the cad_v3 export (owner's Onshape STEP of 2026-11-04), for the
four clockings k of the UR5 adaptor on the flange.

Inputs: logs/20261104-cad_v3/extract/parts.json and its STLs (FreeCAD 0.19 reading of the STEP files), and
wrist3_features.json (probe_wrist3*.py: the UR5 wrist 3 visual mesh of ur_description in tool0). Output: flange.json.

    ~/.claude/bin/resguard.sh run --mem 3G -- .venv/bin/python logs/20261104-cad_v3/flange/flange.py
"""
import itertools
import json
import math
import os
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

REPO = Path("/home/humanoid/Programs/hand")
WORK = REPO / "logs/20261104-cad_v3"
OUT = WORK / "flange"
PARTS = json.load(open(WORK / "extract/parts.json"))["parts"]
WR3 = json.load(open(OUT / "wrist3_features.json"))
Z = np.array([0.0, 0.0, 1.0])


def part(f, n):
    return next(p for p in PARTS if p["file"] == f and p["name"] == n)


def T_place(p):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(p["placement"]["quat_xyzw"]).as_matrix()   # FreeCAD Rotation.Q is (x, y, z, w)
    T[:3, 3] = p["placement"]["pos"]
    return T


def apply(T, P):
    return P @ T[:3, :3].T + T[:3, 3]


def stl(p):
    return trimesh.load(REPO / p["stl"], force="mesh")


def holes(p, r, tol=0.01, axis=Z, near=None, within=None):
    """Distinct axes (x, y of the axis point; z range of the face) of the cylindrical faces of radius r parallel to
    axis."""
    out = []
    for c in p["cylinders"]:
        if abs(c["r"] - r) > tol or abs(abs(np.dot(c["axis"], axis)) - 1) > 1e-6:
            continue
        pt = np.array(c["point"])
        if near is not None and np.hypot(*(pt[:2] - near)) > within:
            continue
        zr = [c["bbox"][0][2], c["bbox"][1][2]]
        for h in out:
            if np.hypot(*(h["xy"] - pt[:2])) < 1e-3:
                h["z"] = [min(h["z"][0], zr[0]), max(h["z"][1], zr[1])]
                break
        else:
            out.append(dict(xy=pt[:2].copy(), z=zr, concave=c["concave"]))
    return out


def circle_fit(P):
    A = np.c_[2 * P[:, 0], 2 * P[:, 1], np.ones(len(P))]
    cx, cy, c = np.linalg.lstsq(A, (P ** 2).sum(1), rcond=None)[0]
    r = math.sqrt(c + cx ** 2 + cy ** 2)
    res = np.hypot(P[:, 0] - cx, P[:, 1] - cy) - r
    return np.array([cx, cy]), r, float(np.sqrt((res ** 2).mean()))


def kabsch2(P, Q, reflect=False):
    """Best 2-D rigid map P -> Q (rotation, or rotation with reflection if reflect): rms residual."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, -d if reflect else d])
    R = Vt.T @ D @ U.T
    if reflect and np.linalg.det(R) > 0:
        return math.inf
    return float(np.sqrt((((P - pc) @ R.T + qc - Q) ** 2).sum(1).mean()))


def mesh_match(ms, mh, T):
    """Distances (mm) between the standalone mesh moved by T and the hand file's mesh: vertex to nearest vertex, both
    ways, and the vertex to surface distance where vertices do not coincide."""
    Vs = apply(T, ms.vertices)
    d1 = cKDTree(mh.vertices).query(Vs)[0]
    d2 = cKDTree(Vs).query(mh.vertices)[0]
    out = dict(n_vertices=[int(len(Vs)), int(len(mh.vertices))], max_vertex_to_vertex_mm=float(max(d1.max(), d2.max())),
               rms_vertex_to_vertex_mm=float(np.sqrt(np.r_[d1, d2].__pow__(2).mean())))
    if out["max_vertex_to_vertex_mm"] > 1e-3:
        far = Vs[d1 > 1e-3]
        dd = []
        for i in range(0, len(far), 4096):
            dd.append(trimesh.proximity.closest_point(mh, far[i:i + 4096])[1])
        dd = np.concatenate(dd) if dd else np.zeros(0)
        out["max_vertex_to_surface_mm"] = float(dd.max()) if len(dd) else 0.0
    return out


def feature_match(ps, ph, T):
    """Every cylindrical face (radius, axis line) and planar face (area, normal, centroid) of the standalone part ps,
    moved by T, against the nearest face of the same radius or area in the hand file's part ph: largest distances (mm)
    and angles (deg)."""
    R, t = T[:3, :3], T[:3, 3]
    cd, ca = 0.0, 0.0
    for c in ps["cylinders"]:
        p, d = R @ np.array(c["point"]) + t, R @ np.array(c["axis"])
        best = (math.inf, 0.0)
        for h in ph["cylinders"]:
            if abs(h["r"] - c["r"]) > 1e-6:
                continue
            q, e = np.array(h["point"]), np.array(h["axis"])
            dist = float(np.linalg.norm(np.cross(p - q, e)))
            if dist < best[0]:
                best = (dist, float(np.degrees(np.arccos(min(1.0, abs(d @ e))))))
        cd, ca = max(cd, best[0]), max(ca, best[1])
    pd, pa = 0.0, 0.0
    for f in ps["planes"]:
        p, n = R @ np.array(f["centre"]) + t, R @ np.array(f["normal"])
        cands = [h for h in ph["planes"] if abs(h["area"] - f["area"]) < 1e-3 * max(1.0, f["area"])]
        h = min(cands, key=lambda h: np.linalg.norm(np.array(h["centre"]) - p))
        pd = max(pd, float(np.linalg.norm(np.array(h["centre"]) - p)))
        pa = max(pa, float(np.degrees(np.arccos(min(1.0, n @ np.array(h["normal"]))))))
    return dict(n_cylinders=len(ps["cylinders"]), n_planes=len(ps["planes"]), cylinder_axis_max_mm=cd,
                cylinder_axis_max_deg=ca, plane_centre_max_mm=pd, plane_normal_max_deg=pa)


def rot_z_angle(R):
    return math.degrees(math.atan2(R[1, 0], R[0, 0]))


def r3(x, n=3):
    return [round(float(v), n) for v in np.ravel(x)]


def main():
    res = {}
    hand_rm, hand_bp, hand_ad = part("hand", "001"), part("hand", "002"), part("hand", "004")
    sa_rm, sa_bp, sa_ad = part("robot_mount", "000"), part("base_plate", "000"), part("adaptor", "000")
    for name, ph, ps in (("robot_mount", hand_rm, sa_rm), ("base_plate", hand_bp, sa_bp), ("adaptor", hand_ad, sa_ad)):
        assert abs(ph["volume"] - ps["volume"]) / ps["volume"] < 1e-6, name
    # ---------------------------------------------------------------- 1. standalone files against the hand frame A
    frames = {}
    no_mesh = os.environ.get("FLANGE_NO_MESH") == "1"
    for name, ph, ps in (("robot_mount", hand_rm, sa_rm), ("base_plate", hand_bp, sa_bp), ("adaptor", hand_ad, sa_ad)):
        T = T_place(ph) @ np.linalg.inv(T_place(ps))
        com = apply(T, np.array([ps["com"]]))[0]
        Ih, Is = np.array(ph["inertia_mm5"]), np.array(ps["inertia_mm5"])
        Irot = T[:3, :3] @ Is @ T[:3, :3].T
        # the best pure translation (centres of mass on each other) and what it leaves
        Tt = np.eye(4)
        Tt[:3, 3] = np.array(ph["com"]) - np.array(ps["com"])
        if no_mesh:
            m = mt = dict(not_run="the STL comparison needs resguard (a mesh probe); its gate stayed closed on swap "
                                  "(15.5 GB in use > 15 GB) on 2026-11-04 17:30-17:50; the face-level check stands in")
        else:
            ms, mh = stl(ps), stl(ph)
            m = mesh_match(ms, mh, T)
            mt = mesh_match(ms, mh, Tt)
        rv = Rotation.from_matrix(T[:3, :3]).as_rotvec()
        ang = float(np.degrees(np.linalg.norm(rv)))
        fm, fmt = feature_match(ps, ph, T), feature_match(ps, ph, Tt)
        print(name, "faces under the placement:", {k: (round(v, 9) if isinstance(v, float) else v) for k, v in fm.items()})
        print(name, "faces under the COM translation only:", {k: (round(v, 3) if isinstance(v, float) else v)
                                                              for k, v in fmt.items()})
        frames[name] = dict(
            feature_match=fm, feature_match_translation_only=fmt,
            T_A_from_file=T.round(6).tolist(),
            rotation_angle_deg=round(ang, 6),
            rotation_axis_A=r3(rv / np.linalg.norm(rv), 6) if ang > 1e-9 else None,
            translation_A_mm=r3(T[:3, 3], 4),
            file_origin_in_A_mm=r3(T[:3, 3], 4),
            com_error_mm=float(np.linalg.norm(com - np.array(ph["com"]))),
            inertia_rel_error=float(np.linalg.norm(Irot - Ih) / np.linalg.norm(Ih)),
            mesh_match=m,
            pure_translation_check=dict(translation_A_mm=r3(Tt[:3, 3], 4), stl=mt))
        print(name, "angle", round(ang, 4), "axis", frames[name]["rotation_axis_A"], "t", r3(T[:3, 3]),
              "mesh", {k: (round(v, 6) if isinstance(v, float) else v) for k, v in m.items()},
              "COM err", round(frames[name]["com_error_mm"], 6), "I err", f'{frames[name]["inertia_rel_error"]:.2e}',
              "| translation only: max", frames[name]["pure_translation_check"])
    res["standalone_frames"] = frames

    # ---------------------------------------------------------------- 2. the adaptor in the robot mount (A)
    c0 = np.array([-111.0, 116.0])
    ad_44 = holes(hand_ad, 2.2)
    rm_44 = holes(hand_rm, 2.2, near=c0, within=45)
    rm_cb = holes(hand_rm, 4.25, near=c0, within=45)
    assert len(ad_44) == 4 and len(rm_44) == 4 and len(rm_cb) == 4
    plate = []
    for h in ad_44:
        d = [np.hypot(*(h["xy"] - g["xy"])) for g in rm_44]
        j = int(np.argmin(d))
        dc = min(np.hypot(*(h["xy"] - g["xy"])) for g in rm_cb)
        plate.append(dict(adaptor_hole_A_mm=r3(h["xy"], 4), adaptor_hole_z_A_mm=r3(h["z"], 3),
                          robot_mount_hole_A_mm=r3(rm_44[j]["xy"], 4), robot_mount_hole_z_A_mm=r3(rm_44[j]["z"], 3),
                          residual_mm=float(d[j]), residual_to_counterbore_mm=float(dc)))
    plate.sort(key=lambda h: (h["adaptor_hole_A_mm"][1], h["adaptor_hole_A_mm"][0]))
    for h in plate:
        print("plate-side hole", h["adaptor_hole_A_mm"], "robot mount", h["robot_mount_hole_A_mm"], "residual",
              f'{h["residual_mm"]:.2e}', "to its 8.5 mm counterbore", f'{h["residual_to_counterbore_mm"]:.2e}')
    Q = np.array([h["adaptor_hole_A_mm"] for h in plate])
    sym = []
    for perm in itertools.permutations(range(4)):
        e_rot, e_ref = kabsch2(Q, Q[list(perm)]), kabsch2(Q, Q[list(perm)], reflect=True)
        sym.append(dict(perm=list(perm), rms_rotation_mm=round(e_rot, 4), rms_mirror_mm=round(e_ref, 4)))
    proper = [s for s in sym if s["rms_rotation_mm"] < 0.5]
    mirror = [s for s in sym if s["rms_mirror_mm"] < 0.5]
    next_best = sorted(s["rms_rotation_mm"] for s in sym)[1]
    print("trapezoid self-maps: proper", proper, "mirror", mirror, "next best proper rms", next_best)
    # the M6 pattern
    m6 = holes(hand_ad, 3.2)
    cb = holes(hand_ad, 5.5)
    st = holes(hand_ad, 3.25)
    M = np.array([h["xy"] for h in m6])
    cM, rM, rmsM = circle_fit(M)
    cC, rC, rmsC = circle_fit(np.array([h["xy"] for h in cb]))
    ang = np.degrees(np.arctan2(M[:, 1] - cM[1], M[:, 0] - cM[0]))
    order = np.argsort(np.mod(ang + 90, 360))
    ang = ang[order]
    M = M[order]
    m6_rows = [dict(centre_A_mm=r3(M[i], 4), angle_A_deg=round(float(ang[i]), 4),
                    z_A_mm=r3(m6[order[i]]["z"], 3)) for i in range(4)]
    # four-fold symmetry of the pattern about its centre
    R90 = np.array([[0, -1], [1, 0.0]])
    rot = (M - cM) @ R90.T + cM
    e90 = max(min(np.hypot(*(p - q)) for q in M) for p in rot)
    # faces: the adaptor's flange face and its hand-side face
    up = [pl for pl in hand_ad["planes"] if np.dot(pl["normal"], Z) > 0.999999]
    dn = [pl for pl in hand_ad["planes"] if np.dot(pl["normal"], Z) < -0.999999]
    face = max(up, key=lambda pl: pl["centre"][2])
    low = min(dn, key=lambda pl: pl["centre"][2])
    rm_top = max((pl for pl in hand_rm["planes"] if np.dot(pl["normal"], Z) > 0.999999), key=lambda pl: pl["centre"][2])
    zmax_all = max(p["bbox"][5] for p in PARTS if p["file"] == "hand")
    over = [p["name"] for p in PARTS if p["file"] == "hand" and p["bbox"][5] > face["centre"][2] + 1e-6]
    cbz = [h["z"] for h in cb]
    print("M6 bolt circle centre", r3(cM, 4), "d", round(2 * rM, 4), "rms", rmsM, "angles", r3(ang, 3),
          "90-deg symmetry max", e90, "counterbore circle centre", r3(cC, 4), "z", cbz[0])
    print("flange face z", face["centre"][2], "area", round(face["area"], 1), "robot mount top", rm_top["centre"][2],
          "highest point of the hand", zmax_all, "parts above the face", over)
    # adaptor outline orientation: its flat sides' normals in A
    sides = [pl for pl in hand_ad["planes"] if abs(np.dot(pl["normal"], Z)) < 1e-6]
    side_ang = sorted(float(np.degrees(np.arctan2(pl["normal"][1], pl["normal"][0]))) for pl in sides)
    pocket = [pl for pl in hand_rm["planes"] if abs(np.dot(pl["normal"], Z)) < 1e-6 and
              np.hypot(pl["centre"][0] - cM[0], pl["centre"][1] - cM[1]) < 40]
    pocket_d = sorted(float(abs(np.dot(np.array(pl["centre"][:2]) - cM, pl["normal"][:2]))) for pl in pocket)
    side_d = sorted(float(abs(np.dot(np.array(pl["centre"][:2]) - cM, pl["normal"][:2]))) for pl in sides)
    print("adaptor side normals (deg in A)", r3(side_ang, 3), "half widths", r3(side_d, 3), "pocket half widths",
          r3(pocket_d, 3))
    res["adaptor_in_A"] = dict(
        plate_side_holes=plate,
        plate_side_hole_pattern_self_maps=dict(proper_rotations=proper, mirror_maps=mirror,
                                               next_best_proper_rms_mm=next_best),
        m6_holes=m6_rows, m6_bolt_circle=dict(centre_A_mm=r3(cM, 4), diameter_mm=round(2 * rM, 4), rms_mm=rmsM),
        m6_counterbores=dict(d_mm=11.0, centre_A_mm=r3(cC, 4), z_A_mm=r3(cbz[0], 3), seat_z_A_mm=13.0,
                             note="11 mm counterbores 7 mm deep from the z_A = 6 face (the hand side); the head seat at "
                                  "z_A = 13, 15 mm below the flange face"),
        m6_pattern_90deg_symmetry_max_mm=e90,
        flange_face=dict(z_A_mm=face["centre"][2], area_mm2=round(face["area"], 2), normal_A=r3(face["normal"], 6)),
        hand_side_face_z_A_mm=low["centre"][2],
        robot_mount_top_z_A_mm=rm_top["centre"][2],
        highest_point_of_hand_z_A_mm=zmax_all,
        parts_above_flange_face=over,
        adaptor_side_normals_A_deg=r3(side_ang, 4),
        adaptor_half_width_mm=r3(side_d, 3), pocket_half_width_mm=r3(pocket_d, 3))
    c_A = np.array([cM[0], cM[1], face["centre"][2]])

    # ---------------------------------------------------------------- 3. the flange in E and the four clockings
    phi_E = sorted(h["angle_E_deg"] % 360 for h in WR3["face_holes"])          # 45, 135, 225, 315 (mesh)
    phi_E_nom = [45.0, 135.0, 225.0, 315.0]
    assert max(abs(a - b) for a, b in zip(phi_E, phi_E_nom)) < 0.01, phi_E
    # an A-plane direction at angle a maps to angle theta - a in E (z_A = -z_E). The holes need theta - phi_A on 45+90m:
    # theta_0 = the representative in (-45, 45] of 45 + phi_A, mod 90, averaged over the four holes on the circle of 4x
    w = np.exp(1j * np.radians(4 * (45.0 + ang)))
    theta0 = math.degrees(np.angle(w.mean())) / 4
    theta0 = ((theta0 + 45) % 90) - 45
    print("theta_0 =", theta0)
    Rx = np.diag([1.0, -1.0, -1.0])
    thumb_dir_A = np.array([0.0, 1.0])
    gantries = {"thumb": part("hand", "045"), "index": part("hand", "074"), "middle": part("hand", "102")}
    g_centre = {k: np.array([(p["bbox"][0] + p["bbox"][3]) / 2, (p["bbox"][1] + p["bbox"][4]) / 2]) for k, p in
                gantries.items()}
    ks = []
    for k in range(4):
        th = theta0 + 90.0 * k
        R = Rotation.from_euler("z", th, degrees=True).as_matrix() @ Rx
        t = -R @ c_A
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, t
        # checks: the A holes land on the E holes; z_A -> -z_E
        HA = np.c_[M, np.full(4, c_A[2])]
        HE = apply(T, HA)
        hole_err = max(min(abs(((math.degrees(math.atan2(p[1], p[0])) - e + 180) % 360) - 180) for e in phi_E_nom)
                       for p in HE)
        rad_err = float(np.abs(np.hypot(HE[:, 0], HE[:, 1]) - 25.0).max())
        dirs = {}
        for name, gc in g_centre.items():
            v = R[:2, :2] @ (gc - c_A[:2])
            dirs[name] = round(math.degrees(math.atan2(v[1], v[0])), 2)
        conn = -90.0
        rel = {name: round(((conn - a + 180) % 360) - 180, 2) for name, a in dirs.items()}
        q = Rotation.from_matrix(R).as_quat()            # x, y, z, w
        ks.append(dict(k=k, theta_deg=round(th, 4), T_E_from_A=T.round(6).tolist(),
                       T_A_from_E=np.linalg.inv(T).round(6).tolist(),
                       quat_wxyz_E_from_A=r3([q[3], q[0], q[1], q[2]], 8), pos_A_origin_in_E_mm=r3(t, 4),
                       gantry_direction_from_flange_axis_E_deg=dirs,
                       connector_minus_gantry_direction_deg=rel,
                       x_A_in_E=r3(R[:, 0], 6), y_A_in_E=r3(R[:, 1], 6), z_A_in_E=r3(R[:, 2], 6),
                       m6_angles_E_deg=r3(np.degrees(np.arctan2(HE[:, 1], HE[:, 0])) % 360, 3),
                       check_hole_angle_err_deg=hole_err, check_hole_radius_err_mm=rad_err,
                       check_flange_centre_to_E_mm=r3(apply(T, c_A[None])[0], 9)))
        print("k", k, "theta", round(th, 3), "t", r3(t), "gantry dirs", dirs, "connector - gantry", rel, "hole err",
              round(hole_err, 6), round(rad_err, 6))
    res["clockings"] = ks
    res["theta0_deg"] = theta0

    # ---------------------------------------------------------------- 4. offsets in A
    bp = hand_bp
    bp_c = np.array([(bp["bbox"][0] + bp["bbox"][3]) / 2, (bp["bbox"][1] + bp["bbox"][4]) / 2])
    rm_c = np.array([(hand_rm["bbox"][0] + hand_rm["bbox"][3]) / 2, (hand_rm["bbox"][1] + hand_rm["bbox"][4]) / 2])
    gc_all = np.array(list(g_centre.values()))
    lo = np.min([[p["bbox"][0], p["bbox"][1]] for p in gantries.values()], 0)
    hi = np.max([[p["bbox"][3], p["bbox"][4]] for p in gantries.values()], 0)
    g_box_c = (lo + hi) / 2
    g_mid = (g_centre["thumb"] + (g_centre["index"] + g_centre["middle"]) / 2) / 2
    lowest = min(p["bbox"][2] for p in PARTS if p["file"] == "hand")
    offs = dict(
        flange_axis_A_mm=r3(c_A[:2], 4),
        base_plate_bbox_centre_A_mm=r3(bp_c, 4), base_plate_com_A_mm=r3(bp["com"][:2], 4),
        robot_mount_bbox_centre_A_mm=r3(rm_c, 4),
        gantry_frame_centres_A_mm={k: r3(v, 4) for k, v in g_centre.items()},
        gantry_frames_bbox_centre_A_mm=r3(g_box_c, 4),
        gantry_frames_P_like_centre_A_mm=r3(g_mid, 4),
        gantry_frames_mean_centre_A_mm=r3(gc_all.mean(0), 4),
        flange_minus_base_plate_bbox_centre_mm=r3(c_A[:2] - bp_c, 4),
        flange_minus_base_plate_com_mm=r3(c_A[:2] - np.array(bp["com"][:2]), 4),
        flange_minus_gantry_frames_centre_mm=r3(c_A[:2] - g_mid, 4),
        flange_minus_gantry_frames_mean_centre_mm=r3(c_A[:2] - gc_all.mean(0), 4),
        flange_face_to_base_plate_underside_mm=float(c_A[2] - bp["bbox"][2]),
        flange_face_to_gantry_frames_underside_mm=float(c_A[2] - min(p["bbox"][2] for p in gantries.values())),
        flange_face_to_lowest_point_mm=float(c_A[2] - lowest),
        thumb_index_middle_frame_dims_mm={k: r3([p["bbox"][3] - p["bbox"][0], p["bbox"][4] - p["bbox"][1],
                                                 p["bbox"][5] - p["bbox"][2]], 2) for k, p in gantries.items()})
    print(json.dumps(offs, indent=None))
    res["offsets"] = offs
    res["c_A"] = r3(c_A, 4)
    json.dump(res, open(OUT / "flange_raw.json", "w"), indent=1)


if __name__ == "__main__":
    main()
