"""Independent check of the UR5-flange-to-hand transform (cad_v3), 2026-11-04.

Sources: the STEP files (read with stepread.py, written for this check) and parts.json (FreeCAD extraction).
Frames: A = hand assembly frame (mm); D = the adaptor's own file frame (mm); E = UR5 CB3 tool0 (mm).
Run: /home/humanoid/Programs/hand/.venv/bin/python verify.py
"""
import json
import sys
from itertools import permutations

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, ".")
from stepread import Step  # noqa: E402

np.set_printoptions(suppress=True, precision=6, linewidth=160)
ROOT = "/home/humanoid/Programs/hand/"
OUT = {}

hand = Step(ROOT + "assets/cad_v2/full_cad/assemblies/hand v3 (anthro config).step")
adap = Step(ROOT + "assets/cad_v2/full_cad/UR5 Adaptor.step")
rmnt = Step(ROOT + "assets/cad_v2/full_cad/robot_mount.step")
W = hand.world_transforms()
parts = [x for x in json.load(open(ROOT + "logs/20261104-cad_v3/extract/parts.json"))["parts"] if x["file"] == "hand"]


def cylinders(S, solid, T=np.eye(4)):
    """Cylindrical faces of a solid in the target frame: r, unit axis, axis point, axial extent of the face's vertices."""
    out = []
    for f in S.faces(solid):
        su = S.surface(f)
        if su["kind"] != "cyl":
            continue
        a = T[:3, :3] @ su["axis"]
        o = T[:3, :3] @ su["origin"] + T[:3, 3]
        V = S.face_vertices(f)
        V = (T[:3, :3] @ V.T).T + T[:3, 3]
        tt = (V - o) @ a
        # foot of the axis nearest the frame origin's projection: report the axis point at t = tt.min()
        out.append(dict(r=su["r"], axis=a, o=o, tmin=tt.min(), tmax=tt.max(),
                        p0=o + a * tt.min(), p1=o + a * tt.max(), concave=(su["same_sense"] == ".F."), face=f))
    return out


def planes(S, solid, T=np.eye(4)):
    out = []
    for f in S.faces(solid):
        su = S.surface(f)
        if su["kind"] != "plane":
            continue
        n = su["normal"] * (1 if su["same_sense"] == ".T." else -1)
        n = T[:3, :3] @ n
        V = S.face_vertices(f)
        V = (T[:3, :3] @ V.T).T + T[:3, 3]
        out.append(dict(n_out=n, V=V, d=float(np.mean(V @ n)), face=f))
    return out


def circle_fit(P):
    """Least-squares circle through 2-D points (Kasa); returns centre, radius, rms."""
    A = np.c_[2 * P, np.ones(len(P))]
    b = (P ** 2).sum(1)
    sol = np.linalg.lstsq(A, b, rcond=None)[0]
    c = sol[:2]
    r = np.sqrt(sol[2] + c @ c)
    res = np.linalg.norm(P - c, axis=1) - r
    return c, r, float(np.sqrt(np.mean(res ** 2)))


# ---------------------------------------------------------------------------------------------------------------
# 1. Identify the adaptor and the robot mount among the hand's solids by their face census, and get their
#    placements from the STEP text.
# ---------------------------------------------------------------------------------------------------------------
adap_solid = [s for s, T, ch in adap.world_transforms()][0]
adap_D_cyl = cylinders(adap, adap_solid)
sig = sorted(round(c["r"], 4) for c in adap_D_cyl)
cand = []
for k, (s, T, ch) in enumerate(W):
    cs = cylinders(hand, s)
    if sorted(round(c["r"], 4) for c in cs) == sig:
        cand.append(k)
print("hand solids with the adaptor's cylinder radii:", cand)
assert len(cand) == 1
KA = cand[0]
s_ad, T_A_ad, ch_ad = W[KA]
print("adaptor solid index", KA, "chain", ch_ad)
print("T_A_from_adaptor-local (STEP):\n", T_A_ad)

# Does the solid inside the hand file equal the standalone file's solid, face for face, in the same local frame?
loc = cylinders(hand, s_ad)


def cylkey(c):
    return (round(c["r"], 4), tuple(np.round(c["p0"], 4)), tuple(np.round(c["p1"], 4)))


ka = sorted(cylkey(c) for c in adap_D_cyl)
kb = sorted(cylkey(c) for c in loc)
same_local = ka == kb
print("adaptor in the hand file has the standalone file's cylinders in the same local frame:", same_local)
OUT["adaptor_hand_local_equals_standalone"] = same_local

# Compare with FreeCAD's placement in parts.json
pj = parts[KA]
q = pj["placement"]["quat_xyzw"]
R_fc = Rotation.from_quat(q).as_matrix()
t_fc = np.array(pj["placement"]["pos"])
dR = Rotation.from_matrix(R_fc.T @ T_A_ad[:3, :3]).magnitude() * 180 / np.pi
dt = np.linalg.norm(t_fc - T_A_ad[:3, 3])
print("FreeCAD placement vs STEP: rotation diff %.2e deg, translation diff %.2e mm" % (dR, dt))
OUT["adaptor_T_A_from_D_step"] = T_A_ad.tolist()
OUT["adaptor_placement_freecad_vs_step"] = dict(rot_deg=dR, trans_mm=dt)

# Robot mount: the solid with 120 cylinders
KR = [k for k, (s, T, ch) in enumerate(W) if len(cylinders(hand, s)) == 120]
print("robot mount candidates", KR)
KR = KR[0]
s_rm, T_A_rm, ch_rm = W[KR]
print("T_A_from_robot-mount-local (STEP):\n", T_A_rm)
OUT["robot_mount_T_A_from_local_step"] = T_A_rm.tolist()

# ---------------------------------------------------------------------------------------------------------------
# 2. The adaptor's features in A.
# ---------------------------------------------------------------------------------------------------------------
cA = cylinders(hand, s_ad, T_A_ad)
pA = planes(hand, s_ad, T_A_ad)


def axis_xy_at(c, z):
    """Where a near-vertical axis crosses the plane z (A)."""
    a, o = c["axis"], c["o"]
    t = (z - o[2]) / a[2]
    return (o + t * a)[:2]


m6 = [c for c in cA if abs(c["r"] - 3.2) < 1e-6]
cb = [c for c in cA if abs(c["r"] - 5.5) < 1e-6]
h44 = [c for c in cA if abs(c["r"] - 2.2) < 1e-6]
side = [c for c in cA if abs(c["r"] - 4.3) < 1e-6]
corner = [c for c in cA if abs(c["r"] - 20.0) < 1e-6]
for nm, grp in (("M6 6.4 mm", m6), ("counterbore 11 mm", cb), ("4.4 mm", h44), ("side 8.6 mm", side), ("corner R20", corner)):
    print(nm, len(grp))
    for c in grp:
        print("   axis", np.round(c["axis"], 6), "from", np.round(c["p0"], 4), "to", np.round(c["p1"], 4))

# axis directions of the M6 holes: tilt from z_A
tilts = [np.degrees(np.arccos(abs(c["axis"][2]))) for c in m6]
print("M6 axis tilt from z_A (deg):", np.round(tilts, 9))
P6 = np.array([axis_xy_at(c, 28.0) for c in m6])
c6, r6, rms6 = circle_fit(P6)
ang6 = np.sort(np.degrees(np.arctan2(*(P6 - c6).T[::-1])))
print("M6 bolt circle in A at z=28: centre", c6, "diameter %.6f rms %.2e" % (2 * r6, rms6), "angles", ang6)
P6b = np.array([axis_xy_at(c, 6.0) for c in m6])
c6b, r6b, _ = circle_fit(P6b)
print("same at z=6: centre", c6b, "diameter %.6f" % (2 * r6b))
OUT["m6_centre_A"] = c6.tolist()
OUT["m6_diameter"] = 2 * r6
OUT["m6_rms"] = rms6
OUT["m6_angles_A_deg"] = ang6.tolist()
# Which end of the M6 holes carries the counterbore? (= screw head side)
z_cb = [sorted([c["p0"][2], c["p1"][2]]) for c in cb]
z_m6 = [sorted([c["p0"][2], c["p1"][2]]) for c in m6]
print("counterbore z ranges", np.round(z_cb, 4), " M6 6.4 mm z ranges", np.round(z_m6, 4))
# counterbore axes coincide with the M6 axes?
for c in cb:
    xy = axis_xy_at(c, 6.0)
    print("   counterbore axis at z=6 is %.2e mm from the nearest M6 axis" % np.min(np.linalg.norm(P6b - xy, axis=1)))
# head-seat plane
for p in pA:
    if abs(abs(p["n_out"][2]) - 1) < 1e-9:
        print("plane n_out", np.round(p["n_out"], 6), "z %.6f" % (p["d"] * np.sign(p["n_out"][2])), "nverts", len(p["V"]),
              "xy-range", np.round(p["V"][:, :2].min(0), 3), np.round(p["V"][:, :2].max(0), 3))
zs_top = [p for p in pA if p["n_out"][2] > 0.999999]
zs_bot = [p for p in pA if p["n_out"][2] < -0.999999]
z_top = max(np.mean(p["V"][:, 2]) for p in zs_top)
print("adaptor top face (outward +z_A) at z_A = %.9f" % z_top)
OUT["adaptor_top_face_z_A"] = z_top

# flat sides (planes with horizontal normals)
flat = [p for p in pA if abs(p["n_out"][2]) < 1e-9]
flat_ang = sorted(np.degrees(np.arctan2(p["n_out"][1], p["n_out"][0])) % 360 for p in flat)
flat_dist = [float(p["n_out"] @ (p["V"].mean(0) - np.r_[c6, 0])) for p in flat]
print("adaptor flat-side normals (deg in A):", np.round(flat_ang, 6), "half width from the M6 centre:", np.round(flat_dist, 6))
OUT["adaptor_flat_normals_A_deg"] = flat_ang

P44 = np.array([axis_xy_at(c, 6.0) for c in h44])
P44_28 = np.array([axis_xy_at(c, 28.0) for c in h44])
print("4.4 mm holes in A at z=6:", np.round(P44, 9), "max xy drift over the thickness %.2e" % np.abs(P44 - P44_28).max())
OUT["adaptor_44_holes_A"] = P44.tolist()

# ---------------------------------------------------------------------------------------------------------------
# 3. Robot mount features near the pocket (A), from the STEP text.
# ---------------------------------------------------------------------------------------------------------------
cR = cylinders(hand, s_rm, T_A_rm)
pR = planes(hand, s_rm, T_A_rm)
vert = [c for c in cR if abs(abs(c["axis"][2]) - 1) < 1e-9]
nonv = [c for c in cR if abs(abs(c["axis"][2]) - 1) >= 1e-9]
print("robot mount: %d vertical and %d non-vertical cylinders" % (len(vert), len(nonv)))
near = []
for c in vert:
    xy = c["o"][:2]
    d = np.linalg.norm(xy - c6)
    zr = sorted([c["p0"][2], c["p1"][2]])
    near.append((d, c["r"], xy, zr, c["concave"]))
near.sort(key=lambda r: (r[0], r[1]))
print("robot mount vertical cylinders within 70 mm of the M6 centre (A):")
for d, r, xy, zr, cc in near:
    if d < 70:
        print("   d %.4f r %.4f xy %s z %s concave %s" % (d, r, np.round(xy, 4), np.round(zr, 4), cc))
for c in nonv:
    print("   non-vertical r %.4f axis %s from %s to %s concave %s" % (c["r"], np.round(c["axis"], 5), np.round(c["p0"], 3), np.round(c["p1"], 3), c["concave"]))
# pocket walls: planes with horizontal normals within 45 mm of the centre
walls = [p for p in pR if abs(p["n_out"][2]) < 1e-9 and np.linalg.norm(p["V"][:, :2].mean(0) - c6) < 40]
for p in walls:
    nn = p["n_out"][:2]
    print("   pocket wall: outward normal (into the pocket) at %.6f deg, distance from the M6 centre %.6f, z %s" % (
        np.degrees(np.arctan2(nn[1], nn[0])) % 360, -float(nn @ (p["V"][:, :2].mean(0) - c6)), np.round([p["V"][:, 2].min(), p["V"][:, 2].max()], 3)))
floor = [p for p in pR if p["n_out"][2] > 0.999 and abs(np.mean(p["V"][:, 2]) - 6) < 1e-6]
print("   pocket floor faces at z=6:", len(floor))
OUT["robot_mount_holes_near_pocket"] = [dict(d=d, r=r, xy=list(xy), z=list(zr)) for d, r, xy, zr, cc in near if d < 40]

# ---------------------------------------------------------------------------------------------------------------
# 4. Mate between the adaptor's 4.4 mm holes and the robot mount's holes.
# ---------------------------------------------------------------------------------------------------------------
rm_holes = np.array([xy for d, r, xy, zr, cc in near if cc and r < 6 and d < 40])
rm_holes_r = [r for d, r, xy, zr, cc in near if cc and r < 6 and d < 40]
print("robot mount small concave vertical cylinders in the pocket region:", len(rm_holes), sorted(set(np.round(rm_holes_r, 4))))
res = [np.min(np.linalg.norm(rm_holes - p, axis=1)) for p in P44]
print("4.4 mm hole axis to nearest robot-mount hole axis (mm):", np.array(res))
OUT["mate_residuals_mm"] = [float(r) for r in res]

# point-set self-symmetry of the trapezoid: all orderings, best rigid map (rotation only, and with reflection)


def best_rigid(P, Q, allow_reflection):
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    H = Pc.T @ Qc
    U, Sv, Vt = np.linalg.svd(H)
    D = np.eye(2)
    if not allow_reflection and np.linalg.det(Vt.T @ U.T) < 0:
        D[1, 1] = -1
    R = Vt.T @ D @ U.T
    r = (R @ Pc.T).T - Qc
    return np.sqrt((r ** 2).sum(1).mean()), R


res_rot, res_ref = [], []
for perm in permutations(range(4)):
    Q = P44[list(perm)]
    e1, R1 = best_rigid(P44, Q, False)
    e2, R2 = best_rigid(P44, Q, True)
    res_rot.append((e1, perm, np.degrees(np.arctan2(R1[1, 0], R1[0, 0]))))
    if np.linalg.det(R2) < 0:
        res_ref.append((e2, perm))
res_rot.sort()
res_ref.sort()
print("trapezoid onto itself, proper rotations: best", [(round(e, 9), p, round(a, 4)) for e, p, a in res_rot[:3]])
print("trapezoid onto itself, reflections: best", [(round(e, 9), p) for e, p in res_ref[:2]])
OUT["trapezoid_rotation_maps"] = [(float(e), list(p), float(a)) for e, p, a in res_rot[:3]]
OUT["trapezoid_reflection_maps"] = [(float(e), list(p)) for e, p in res_ref[:2]]

# The eight placements of a rounded square in a matching pocket: turn by 90 k about the pocket axis, with or
# without turning the adaptor over (a half turn about an in-plane axis through the centre). For each, where do the
# 4.4 mm holes land, and how far is each from a robot-mount hole?  The pocket centre:
pc = c6  # checked below
poses = []
for flip in (False, True):
    for k in range(4):
        th = np.radians(90 * k)
        Rz = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        if flip:
            # half turn about the in-plane axis through the centre along the adaptor's flat normal (24 deg):
            # reflection of the xy pattern about that line.
            phi = np.radians(flat_ang[0])
            F = np.array([[np.cos(2 * phi), np.sin(2 * phi)], [np.sin(2 * phi), -np.cos(2 * phi)]])
        else:
            F = np.eye(2)
        M = Rz @ F
        Q = (M @ (P44 - pc).T).T + pc
        dmin = [float(np.min(np.linalg.norm(rm_holes - qq, axis=1))) for qq in Q]
        # the square's orientation after this move: flats map to themselves for all 8 (dihedral group of the square)
        poses.append(dict(flip=flip, k=k, dmin=dmin, n_bolt=int(sum(d < 0.5 for d in dmin))))
for p in poses:
    print("   placement flip=%s turn=%d x 90 deg: 4.4 mm holes to nearest robot-mount hole %s mm -> %d of 4 bolt" % (
        p["flip"], p["k"], np.round(p["dmin"], 3), p["n_bolt"]))
OUT["eight_placements"] = poses

# the reflection that maps the trapezoid onto itself: its axis and what it does to the square
Pc = P44 - P44.mean(0)
for e, perm in res_ref[:1]:
    Q = P44[list(perm)]
    e2, R2 = best_rigid(P44, Q, True)
    # reflection matrix [[cos2a, sin2a],[sin2a,-cos2a]] -> axis angle a
    a = 0.5 * np.degrees(np.arctan2(R2[1, 0], R2[0, 0]))
    print("trapezoid mirror line direction %.6f deg (A); its centre" % (a % 180), P44.mean(0))
    # flats under that reflection
    fl = sorted(((2 * a - fa) % 360) for fa in flat_ang)
    print("   flats after that flip would face", np.round(fl, 4), "deg; pocket walls face", np.round(flat_ang, 4))
    OUT["trapezoid_mirror_line_deg_A"] = a % 180


def support(phi_deg, half=30.2, rc=20.0):
    """Support function of a rounded square (flats at 0 deg) in direction phi."""
    ph = np.radians(phi_deg)
    return rc + (half - rc) * (abs(np.cos(ph)) + abs(np.sin(ph)))


print("rounded-square extent turned by 42 deg against the pocket wall normal: %.3f mm (pocket half width 30.5)" % support(42))
OUT["flipped_adaptor_extent_vs_pocket_mm"] = [support(42), 30.5]

# ---------------------------------------------------------------------------------------------------------------
# 5. Transforms T_E_from_A for the four clockings, and checks.
#    Convention under test: flange M6 at 45 + 90 j deg from x_E, on a 50 mm circle, in the plane z_E = 0;
#    z_E points out of the flange toward the tool; the hand lies at z_E > 0.
# ---------------------------------------------------------------------------------------------------------------
cA3 = np.r_[c6, z_top]
mine = []
for k in range(4):
    # z_E = -z_A, so R = Rz(theta) Rx(180) and a direction at angle a in A goes to theta - a in E.
    # choose theta so that the M6 hole nearest -21 deg maps to 45 deg, then add 90 k
    a0 = ang6[np.argmin(np.abs(((ang6 + 21 + 180) % 360) - 180))]
    th = np.radians(45 + a0 + 90 * k)
    R = np.array([[np.cos(th), np.sin(th), 0], [np.sin(th), -np.cos(th), 0], [0, 0, -1.0]])
    t = -R @ cA3
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    mine.append(T)
print("theta_k (deg):", [round(np.degrees(np.arctan2(T[1, 0], T[0, 0])) % 360, 9) for T in mine])

claimed = [np.array(v).reshape(4, 4) for v in [
    [0.913545, 0.406737, 0, 54.222098, 0.406737, -0.913545, 0, 151.11904, 0, 0, -1, 28, 0, 0, 0, 1],
    [-0.406737, 0.913545, 0, -151.11904, 0.913545, 0.406737, 0, 54.222098, 0, 0, -1, 28, 0, 0, 0, 1],
    [-0.913545, -0.406737, 0, -54.222098, -0.406737, 0.913545, 0, -151.11904, 0, 0, -1, 28, 0, 0, 0, 1],
    [0.406737, -0.913545, 0, 151.11904, -0.913545, -0.406737, 0, -54.222098, 0, 0, -1, 28, 0, 0, 0, 1]]]
claimed_q = [[0, 0.9781476, 0.20791169, 0], [0, 0.54463904, 0.83867057, 0], [0, -0.20791169, 0.9781476, 0],
             [0, 0.83867057, -0.54463904, 0]]
claimed_pos_m = [[0.0542221, 0.151119, 0.028], [-0.151119, 0.0542221, 0.028], [-0.0542221, -0.151119, 0.028],
                 [0.151119, -0.0542221, 0.028]]

# all hand vertices in A (STEP), and FreeCAD bbox corners
Vall = []
for s, T, ch in W:
    for f in hand.faces(s):
        V = hand.face_vertices(f)
        Vall.append((T[:3, :3] @ V.T).T + T[:3, 3])
Vall = np.vstack(Vall)
bb = np.array([p["bbox"] for p in parts])
corners = np.array([[x, y, z] for b in bb for x in (b[0], b[3]) for y in (b[1], b[4]) for z in (b[2], b[5])])
com_all = sum(np.array(p["com"]) * p["volume"] for p in parts) / sum(p["volume"] for p in parts)

flange_pts = np.array([[25 * np.cos(np.radians(45 + 90 * j)), 25 * np.sin(np.radians(45 + 90 * j)), 0] for j in range(4)])
m6_A3 = np.c_[P6, np.full(4, 28.0)]
m6_dirs_A = np.array([c["axis"] * np.sign(c["axis"][2]) for c in m6])  # +z_A direction along each axis
rep = []
for k in range(4):
    T = claimed[k]
    R, t = T[:3, :3], T[:3, 3]
    orth = np.abs(R.T @ R - np.eye(3)).max()
    det = np.linalg.det(R)
    # element-wise difference with my matrix
    dmat = np.abs(T - mine[k]).max()
    # M6 holes to flange positions
    E6 = (R @ m6_A3.T).T + t
    err = [float(np.min(np.linalg.norm(flange_pts - e, axis=1))) for e in E6]
    angE = sorted((np.degrees(np.arctan2(e[1], e[0])) % 360) for e in E6)
    # hole axis direction in E (should be along -z_E for +z_A)
    axE = (R @ m6_dirs_A.T).T
    # adaptor face at z_E = 0?  hand on the tool side?
    zE_v = (R @ Vall.T).T[:, 2] + t[2]
    zE_c = (R @ corners.T).T[:, 2] + t[2]
    zE_com = (R @ com_all)[2] + t[2]
    # quaternion and position checks
    qc = np.array(claimed_q[k])
    Rq = Rotation.from_quat([qc[1], qc[2], qc[3], qc[0]]).as_matrix()
    dq = Rotation.from_matrix(Rq.T @ R).magnitude() * 180 / np.pi
    dpos = np.linalg.norm(np.array(claimed_pos_m[k]) * 1000 - t)
    # where x_A points in E, where the flange centre lands
    xA = np.degrees(np.arctan2(R[1, 0], R[0, 0])) % 360
    cE = R @ cA3 + t
    r_ = dict(k=k, orth_err=orth, det=det, max_elem_diff_vs_mine=dmat, m6_to_flange_mm=err, m6_angles_E=angE,
              m6_axis_E=axE.round(9).tolist(), zE_min_vertices=float(zE_v.min()), zE_min_bbox=float(zE_c.min()),
              zE_max_bbox=float(zE_c.max()), zE_com=float(zE_com), quat_vs_matrix_deg=dq, pos_vs_t_mm=dpos,
              xA_angle_in_E=xA, flange_centre_in_E=cE.tolist())
    rep.append(r_)
    print("k=%d det %.12f orth %.1e  |claimed-mine| %.2e  M6->flange %s mm  angles %s" % (
        k, det, orth, dmat, np.round(err, 6), np.round(angE, 4)))
    print("     z_E of hand: vertices min %.4f, bbox corners min %.4f max %.4f, volume centroid %.3f mm;  quat vs R %.2e deg, pos vs t %.2e mm, x_A at %.4f deg" % (
        r_["zE_min_vertices"], r_["zE_min_bbox"], r_["zE_max_bbox"], zE_com, dq, dpos, xA))
OUT["T_checks"] = rep
OUT["T_mine"] = [T.tolist() for T in mine]
OUT["hand_volume_centroid_A"] = com_all.tolist()

# finer: the claimed matrices are rounded to 6 decimals; how much does that move points 300 mm away?
for k in range(4):
    R = claimed[k][:3, :3]
    U, _, Vt = np.linalg.svd(R)
    print("k=%d claimed R: largest deviation from the nearest rotation %.2e (x 300 mm = %.2e mm)" % (k, np.abs(R - U @ Vt).max(), 300 * np.abs(R - U @ Vt).max()))

json.dump(OUT, open("verify.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
print("wrote verify.json")
