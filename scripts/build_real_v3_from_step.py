# /// script
# requires-python = ">=3.10"
# dependencies = ["cadquery-ocp>=7.7", "numpy", "trimesh", "fast-simplification", "scipy", "scikit-image", "rtree", "mujoco>=3.1"]
# ///
"""Build the `real_v3` hand MJCF straight from the Onshape STEP export of "hand v3 (anthro config)".

real_v1 was hand-measured: capsules on a 20.75 / 20.75 / 37.16 mm finger, mounts at (+/-50, 55).
The v3 hardware replaced the 2-servo differential finger with the 3-servo independent finger
(docs/servos.md in manta-hand) and moved to the "anthro" gantry layout. Everything below is
DERIVED from the STEP so there is no second copy of the CAD numbers to drift:

    joint axes      the output-spline cylinder (r = 6.05 mm) of each finger's three servos.
                    The one perpendicular to the other two is yaw; of the parallel pair the one
                    nearer yaw is MCP, the other PIP.
    links           parts are welded into rigid groups wherever they share a fastener hole
                    (coaxial cylinders r <= 2.1 mm off the joint axes); each group is assigned
                    to the segment whose servo it carries (mount carries yaw servo, proximal
                    carries MCP servo, middle carries PIP servo, the rest is distal + pad).
    zero pose       yaw 0 = MCP axis horizontal, MCP 0 = PIP axis directly below MCP axis,
                    PIP 0 = distal principal axis vertical. The CAD pose is then the `cad` key.
    mounts          workspace centre on each axis = midpoint of that axis' linear rail, shifted
                    by the (yaw axis - rail block) offset in the CAD pose. Half-ranges are the
                    MEASURED stepper travels (manta-hand host/manta_hand/kinematics.py
                    FULL_EXTENSION_MM), not the CAD rails, which over-state free travel.

FRAME. palm_pose origin = centre of the three gantry frames, on the underside of the base
plate (STEP z = 0). STEP -> MuJoCo is a +90 deg turn about z: x_MJ = -(y_S - y0),
y_MJ = x_S - x0, z_MJ = z_S. That puts the thumb on -x and index on +y, as in real_v1 and the
host {P} frame. Index vs middle is identified by position and cross-checked against the
mirrored rail part in the middle gantry ("Part 3"; J4 is the mirrored axis on the hardware).

INTERFACE. Body / joint / actuator names, the [x, y, yaw, mcp, len, pip] qpos block, the zero-
travel `len` shim and the `<f>_tip` body with its own geom are kept identical to real_v1 so
the shared tooling (morphology_xml, keyframe parsers, ContactMatch on `<f>_tip`) loads it.

NOT FROM THE CAD (assumptions, flagged in the XML too):
    ROM             real_v1's +/-85 / -15..92 / -18..92 deg. The 3-servo finger's endpoints have
                    not been calibrated yet (servos.md); the middle finger's CAD pose has MCP
                    at about -30 deg, so the real lower limit is wider than -15.
    masses          printed parts at 1000 kg/m^3 (real_v1's implicit density); servos at
                    SERVO_MASS each.
    PALM_Z          scaled from real_v1's fit (same fraction of finger reach), NOT re-fitted.

Usage:
    uv run scripts/build_real_v3_from_step.py                       # default STEP path
    uv run scripts/build_real_v3_from_step.py --step path/to.step
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "assets" / "mjcf" / "real_v3"
DEFAULT_STEP = OUT_DIR / "cad" / "hand_v3_anthro_config.step"

FINGERS = ("thumb", "index", "middle")
# Measured home-to-hardstop travel (mm), manta-hand host/manta_hand/kinematics.py.
# STEPPER_JOINTS {thumb: (J0, J1), index: (J2, J3), middle: (J4, J5)}; firmware "x" tracks
# the palm-frame y axis and firmware "y" the palm-frame x axis (the documented swap).
TRAVEL_MM = {  # finger -> (palm x travel, palm y travel)
    "thumb": (56.2, 112.4),
    "index": (56.0, 62.5),
    "middle": (54.1, 62.2),
}
YAW_RANGE = 1.483530                 # +/- 85 deg   (real_v1, uncalibrated for v3)
MCP_RANGE = (-0.261799, 1.605703)    # -15 .. +92   (real_v1, uncalibrated for v3)
PIP_RANGE = (-0.314159, 1.605703)    # -18 .. +92   (real_v1, uncalibrated for v3)
OPEN_MCP = OPEN_PIP = 0.55           # real_v1's placeholder `open` key
SERVO_MASS = 0.0135                  # kg, Feetech SCS0009 class micro servo (datasheet ~13.5 g)
PRINT_DENSITY = 1000.0               # kg/m^3
REAL_V1 = dict(palm_z=0.0625, reach=0.07866)   # for scaling PALM_Z

SPLINE_R = 6.05          # servo output spline radius, mm
FASTENER_R = 2.1         # cylinders at or below this radius are fastener holes / pins
SERVO_VOL = (6400, 6900)  # mm^3, SCS0009 body as modelled
VIS_FACES = 12000        # visual mesh budget per body mesh


# ------------------------------------------------------------------------------------------
# STEP loading
# ------------------------------------------------------------------------------------------
@dataclass
class Part:
    idx: int
    path: tuple[str, ...]
    mesh: trimesh.Trimesh          # STEP world frame, mm
    volume: float
    cyl: list = field(default_factory=list)   # (r, unit dir, point-on-axis)

    @property
    def assembly(self) -> str:
        return self.path[1] if len(self.path) > 2 else ""

    @property
    def bbox(self):
        return self.mesh.bounds


def load_step(path: Path) -> list[Part]:
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepGProp import BRepGProp
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.GeomAbs import GeomAbs_Cylinder
    from OCP.GProp import GProp_GProps
    from OCP.STEPCAFControl import STEPCAFControl_Reader
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.TDataStd import TDataStd_Name
    from OCP.TDF import TDF_Label
    from OCP.TDocStd import TDocStd_Document
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS
    from OCP.XCAFDoc import XCAFDoc_DocumentTool
    try:
        from OCP.collections import Sequence_TDF_Label as LabelSeq   # OCP >= 7.8
    except ImportError:  # pragma: no cover
        from OCP.TDF import TDF_LabelSequence as LabelSeq

    face_cast = getattr(TopoDS, "Face_s", None) or TopoDS.Face
    doc = TDocStd_Document(TCollection_ExtendedString("step"))
    reader = STEPCAFControl_Reader()
    reader.SetNameMode(True)
    if reader.ReadFile(str(path)) != 1:
        raise RuntimeError(f"could not read {path}")
    reader.Transfer(doc)
    st = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())

    def name(lab):
        n = TDataStd_Name()
        return n.Get().ToExtString() if lab.FindAttribute(TDataStd_Name.GetID_s(), n) else "?"

    parts: list[Part] = []

    def tessellate(shp) -> trimesh.Trimesh:
        BRepMesh_IncrementalMesh(shp, 0.15, False, 0.3, True)
        verts, faces = [], []
        ex = TopExp_Explorer(shp, TopAbs_FACE)
        while ex.More():
            f = face_cast(ex.Current())
            loc = TopLoc_Location()
            tri = BRep_Tool.Triangulation_s(f, loc)
            if tri is not None:
                tr = loc.Transformation()
                base = len(verts)
                for i in range(1, tri.NbNodes() + 1):
                    p = tri.Node(i).Transformed(tr)
                    verts.append((p.X(), p.Y(), p.Z()))
                rev = f.Orientation() == TopAbs_REVERSED
                for i in range(1, tri.NbTriangles() + 1):
                    a, b, c = tri.Triangle(i).Get()
                    faces.append((base + a - 1, base + c - 1, base + b - 1) if rev
                                 else (base + a - 1, base + b - 1, base + c - 1))
            ex.Next()
        m = trimesh.Trimesh(np.array(verts), np.array(faces), process=True)
        m.merge_vertices()
        return m

    def cylinders(shp):
        out = {}
        ex = TopExp_Explorer(shp, TopAbs_FACE)
        while ex.More():
            s = BRepAdaptor_Surface(face_cast(ex.Current()))
            if s.GetType() == GeomAbs_Cylinder:
                c = s.Cylinder()
                ax = c.Axis()
                d = np.array([ax.Direction().X(), ax.Direction().Y(), ax.Direction().Z()])
                if d[np.argmax(abs(d))] < 0:
                    d = -d
                p = np.array([ax.Location().X(), ax.Location().Y(), ax.Location().Z()])
                p = p - d * (p @ d)
                key = (round(c.Radius(), 2), tuple(np.round(d, 3)), tuple(np.round(p, 1)))
                out[key] = (c.Radius(), d, p)
            ex.Next()
        return list(out.values())

    def walk(lab, loc, pathn):
        nm = name(lab)
        if st.IsReference_s(lab):
            ref = TDF_Label()
            st.GetReferredShape_s(lab, ref)
            loc = loc.Multiplied(st.GetLocation_s(lab))
            lab = ref
        pathn = pathn + (nm,)
        if st.IsAssembly_s(lab):
            kids = LabelSeq()
            st.GetComponents_s(lab, kids)
            for i in range(1, kids.Length() + 1):
                walk(kids.Value(i), loc, pathn)
            return
        shp = st.GetShape_s(lab).Moved(loc)
        g = GProp_GProps()
        BRepGProp.VolumeProperties_s(shp, g)
        parts.append(Part(len(parts), pathn, tessellate(shp), g.Mass(), cylinders(shp)))

    roots = LabelSeq()
    st.GetFreeShapes(roots)
    for i in range(1, roots.Length() + 1):
        walk(roots.Value(i), TopLoc_Location(), ())
    return parts


# ------------------------------------------------------------------------------------------
# geometry helpers
# ------------------------------------------------------------------------------------------
def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def rot(axis, angle):
    """3x3 rotation about unit `axis` by `angle` (Rodrigues)."""
    a = unit(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def rigid(R, about):
    """4x4 rotation R about point `about`."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = about - R @ about
    return T


def signed_angle(v_from, v_to, axis):
    v_from = v_from - axis * (v_from @ axis)
    v_to = v_to - axis * (v_to @ axis)
    return float(np.arctan2(np.cross(v_from, v_to) @ axis, v_from @ v_to))


def coaxial(c1, c2, tol_ang=2e-3, tol_off=0.6):
    (_, d1, p1), (_, d2, p2) = c1, c2
    return abs(abs(d1 @ d2) - 1) < tol_ang and np.linalg.norm(np.cross(p2 - p1, d1)) < tol_off


def closest_points(p1, d1, p2, d2):
    A = np.array([d1, -d2]).T
    t = np.linalg.lstsq(A, p2 - p1, rcond=None)[0]
    return p1 + t[0] * d1, p2 + t[1] * d2


# STEP (mm) -> MuJoCo palm frame (m): +90 deg about z after re-centring on the palm origin.
R_S2M = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)


# ------------------------------------------------------------------------------------------
# finger analysis
# ------------------------------------------------------------------------------------------
@dataclass
class Finger:
    asm: str
    parts: list[Part]
    axes: dict = field(default_factory=dict)       # 'A','B','C' -> (dir, point)  STEP mm
    segments: dict = field(default_factory=dict)   # 'mount','prox','mid','dist' -> [Part]
    servos: dict = field(default_factory=dict)     # 'A','B','C' -> Part


def is_servo(p: Part) -> bool:
    return SERVO_VOL[0] < p.volume < SERVO_VOL[1] and any(abs(r - SPLINE_R) < 0.05 for r, _, _ in p.cyl)


def analyse_finger(asm: str, parts: list[Part]) -> Finger:
    f = Finger(asm, parts)
    servos = [p for p in parts if is_servo(p)]
    if len(servos) != 3:
        raise RuntimeError(f"{asm}: expected 3 servos, found {len(servos)}")
    spline = {}
    for s in servos:
        r, d, p = next(c for c in s.cyl if abs(c[0] - SPLINE_R) < 0.05)
        spline[s.idx] = (d, p)
    # yaw = the spline axis perpendicular to the other two
    ids = list(spline)
    perp = [i for i in ids if all(abs(spline[i][0] @ spline[j][0]) < 0.1 for j in ids if j != i)]
    if len(perp) != 1:
        raise RuntimeError(f"{asm}: could not single out the yaw servo")
    a = perp[0]
    dA, pA = spline[a]
    def gap(i):
        q1, q2 = closest_points(pA, dA, spline[i][1], spline[i][0])
        return np.linalg.norm(q1 - q2)

    rest = sorted((i for i in ids if i != a), key=gap)
    b, c = rest
    by = {s.idx: s for s in servos}
    f.servos = {"A": by[a], "B": by[b], "C": by[c]}
    f.axes = {"A": spline[a], "B": spline[b], "C": spline[c]}

    # rigid groups via shared fastener holes that are not on a joint axis
    joint_axes = [(SPLINE_R, d, p) for d, p in f.axes.values()]
    parent = {p.idx: p.idx for p in parts}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, p in enumerate(parts):
        for q in parts[i + 1:]:
            hit = False
            for c1 in p.cyl:
                if c1[0] > FASTENER_R or any(coaxial(c1, j) for j in joint_axes):
                    continue
                for c2 in q.cyl:
                    if c2[0] <= FASTENER_R and coaxial(c1, c2):
                        hit = True
                        break
                if hit:
                    break
            if hit:
                parent[find(p.idx)] = find(q.idx)
    groups: dict[int, list[Part]] = {}
    for p in parts:
        groups.setdefault(find(p.idx), []).append(p)

    seg_of_root = {find(f.servos["A"].idx): "mount", find(f.servos["B"].idx): "prox",
                   find(f.servos["C"].idx): "mid"}
    if len(seg_of_root) != 3:
        raise RuntimeError(f"{asm}: two servos landed in the same rigid group")
    # distal = the heaviest remaining group
    others = [r for r in groups if r not in seg_of_root]
    dist_root = max(others, key=lambda r: sum(p.volume for p in groups[r]))
    seg_of_root[dist_root] = "dist"
    f.segments = {k: [] for k in ("mount", "prox", "mid", "dist")}
    for r, g in groups.items():
        if r in seg_of_root:
            f.segments[seg_of_root[r]] += g
    # leftovers (pins, bearings, horns that only touch a joint axis): nearest segment by the
    # distance from their centroid to the segments' own geometry
    seg_pts = {k: np.vstack([p.mesh.vertices for p in v]) for k, v in f.segments.items()}
    for r, g in groups.items():
        if r in seg_of_root:
            continue
        for p in g:
            c = p.mesh.centroid
            best = min(seg_pts, key=lambda k: np.min(np.linalg.norm(seg_pts[k] - c, axis=1)))
            f.segments[best].append(p)
    return f


# ------------------------------------------------------------------------------------------
# build
# ------------------------------------------------------------------------------------------
def decimate(m: trimesh.Trimesh, faces: int) -> trimesh.Trimesh:
    if len(m.faces) <= faces:
        return m
    try:
        return m.simplify_quadric_decimation(face_count=faces)
    except Exception:  # pragma: no cover - fall back to the full mesh
        return m


def mvee(points: np.ndarray, tol: float = 1e-4, iters: int = 2000):
    """Minimum-volume enclosing ellipsoid (Khachiyan). Returns centre, semi-axes, rotation.

    Every input point satisfies (x - c)^T A (x - c) <= 1 up to `tol`; the result is then
    scaled up so that holds exactly, i.e. the ellipsoid truly encloses the geometry.
    """
    P = points
    n, d = P.shape
    Q = np.vstack([P.T, np.ones(n)])
    u = np.full(n, 1.0 / n)
    for _ in range(iters):
        X = Q @ (u[:, None] * Q.T)
        M = np.einsum("ij,ji->i", Q.T, np.linalg.solve(X, Q))
        j = int(np.argmax(M))
        step = (M[j] - d - 1) / ((d + 1) * (M[j] - 1))
        new_u = (1 - step) * u
        new_u[j] += step
        if np.linalg.norm(new_u - u) < tol:
            u = new_u
            break
        u = new_u
    c = P.T @ u
    A = np.linalg.inv(P.T @ (u[:, None] * P) - np.outer(c, c)) / d
    rel = P - c
    A /= max(1.0, float(np.max(np.einsum("ij,jk,ik->i", rel, A, rel))))   # exact enclosure
    w, V = np.linalg.eigh(A)
    if np.linalg.det(V) < 0:
        V[:, 0] = -V[:, 0]
    return c, 1.0 / np.sqrt(w), V


VOXEL = 0.001            # m, interior sampling pitch for the ellipsoid fit
ELL_IOU_TARGET = 0.75    # add a 2nd/3rd ellipsoid to a link only while one cannot reach this
ELL_MAX_PER_LINK = 1     # one ellipsoid per phalanx (and one for the mount)


def solid_voxels(meshes: list[trimesh.Trimesh], pitch: float = VOXEL) -> np.ndarray:
    """Centres of the voxels filling the union of `meshes` (body frame, m)."""
    pts = []
    for m in meshes:
        try:
            v = m.voxelized(pitch).fill()
            pts.append(v.points)
        except Exception:  # pragma: no cover - degenerate sliver part
            pts.append(m.vertices)
    P = np.vstack(pts)
    return np.unique(np.round(P / pitch).astype(np.int64), axis=0) * pitch


def fit_ellipsoids(meshes: list[trimesh.Trimesh]):
    """Ellipsoids that follow the link's outside without inflating it.

    Each ellipsoid is moment-matched to its share of the solid (a solid ellipsoid with semi-
    axis a has variance a^2/5 along it), then all are scaled together to maximise IoU with the
    voxelised solid -- so protrusions are partly cut and hollows partly filled, rather than
    everything being wrapped. Clusters come from k-means on the voxels; k grows only while
    the IoU is below ELL_IOU_TARGET. Returns ([(centre, semi_axes, R)], iou, vol_ratio).
    """
    S = solid_voxels(meshes)
    lo, hi = S.min(0) - 0.01, S.max(0) + 0.01
    g = np.mgrid[lo[0]:hi[0]:VOXEL, lo[1]:hi[1]:VOXEL, lo[2]:hi[2]:VOXEL].reshape(3, -1).T
    keys = set(map(tuple, np.round(S / VOXEL).astype(np.int64)))
    solid = np.array([tuple(k) in keys for k in np.round(g / VOXEL).astype(np.int64)])

    def moment(pts):
        c = pts.mean(0)
        w, V = np.linalg.eigh(np.cov((pts - c).T))
        if np.linalg.det(V) < 0:
            V[:, 0] = -V[:, 0]
        return c, np.sqrt(5 * np.maximum(w, 1e-10)), V

    def inside(ells, s):
        occ = np.zeros(len(g), bool)
        for c, a, V in ells:
            occ |= ((((g - c) @ V) / (a * s)) ** 2).sum(1) <= 1
        return occ

    best = None
    for k in range(1, ELL_MAX_PER_LINK + 1):
        # k-means, seeded along the principal axis so it is deterministic
        c0, _, V0 = moment(S)
        t = (S - c0) @ V0[:, -1]
        cent = np.array([S[np.argmin(abs(t - q))] for q in np.quantile(t, (np.arange(k) + 0.5) / k)])
        for _ in range(50):
            lab = np.argmin(((S[:, None, :] - cent[None]) ** 2).sum(-1), 1)
            new = np.array([S[lab == j].mean(0) if np.any(lab == j) else cent[j] for j in range(k)])
            if np.allclose(new, cent):
                break
            cent = new
        ells = [moment(S[lab == j]) for j in range(k) if np.sum(lab == j) > 20]
        scores = []
        for s in np.linspace(0.8, 1.25, 19):
            occ = inside(ells, s)
            scores.append(((occ & solid).sum() / (occ | solid).sum(), s, occ.sum() / solid.sum()))
        iou, s, vr = max(scores)
        cand = ([(c, a * s, V) for c, a, V in ells], iou, vr)
        if best is None or iou > best[1] + 0.05:      # a new ellipsoid has to earn its keep
            best = cand
        if best[1] >= ELL_IOU_TARGET:
            break
    return best


MORPHIT_DIR = OUT_DIR / "morphit"
# Body each MorphIt link lives on: the distal bracket stays on the PIP body, the pad gets the
# tip body (so the fingertip ContactMatch sees it, and it can carry pad friction).
MORPHIT_LINKS = ("mount", "prox", "mid", "dist")   # the pad is a box, not spheres
PAD_SPHERE_R = 0.001             # m: fingertip pad = a grid of small spheres on the pad's surface
PAD_PITCH = 0.002                # m: centre spacing (2 * radius -> neighbours just touch)
MORPHIT_REF_FINGER = "index"      # fitted once; mapped rigidly onto the identical other fingers
REMESH_PITCH = 0.0008             # m, voxel pitch of the watertight remesh MorphIt is fed


def watertight(meshes: list[trimesh.Trimesh], pitch: float = REMESH_PITCH) -> trimesh.Trimesh:
    """One closed surface for a link: voxelise the union of its parts, then marching cubes.

    MorphIt's inside/outside sampling needs a single watertight mesh; the CAD parts are
    separate (sometimes touching, sometimes open) shells. +/- pitch/2 of surface error.
    """
    vg = concat(meshes).voxelized(pitch).fill()
    m = vg.marching_cubes
    m.apply_transform(vg.transform)
    m.merge_vertices()
    trimesh.repair.fix_normals(m)
    # (quadric decimation would shrink it further but opens the surface; the coarser pitch
    # is what keeps the face count manageable while staying closed)
    return m


def kabsch(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """4x4 rigid transform taking corresponding points src -> dst (least squares)."""
    cs, cd = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((src - cs).T @ (dst - cd))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, cd - R @ cs
    return T


def load_morphit_spheres() -> dict:
    """{(finger, link): [(centre_xyz, radius)]} in each body's frame, from MorphIt's output.

    spheres.json holds the reference finger's fit; correspondence.npz holds, per finger and
    link, the body-frame vertices of the link's largest CAD part (OCC shares one triangulation
    between instances of a part, so vertex i is the same physical point on every finger). The
    reference spheres are carried to the other fingers by the Kabsch fit of those vertices.
    """
    sp = json.loads((MORPHIT_DIR / "spheres.json").read_text())
    corr = np.load(MORPHIT_DIR / "correspondence.npz")
    out = {}
    for link, fit in sp["links"].items():
        C, r = np.asarray(fit["centers"]), np.asarray(fit["radii"])
        ref = corr[f"{MORPHIT_REF_FINGER}_{link}"]
        for f in FINGERS:
            dst = corr[f"{f}_{link}"]
            T = kabsch(ref, dst)
            resid = float(np.sqrt(((ref @ T[:3, :3].T + T[:3, 3] - dst) ** 2).sum(1).mean()))
            if resid > 2e-4:
                raise RuntimeError(f"{f}_{link}: reference link does not map rigidly (rms {resid*1e3:.2f} mm)")
            out[(f, link)] = [(c, float(rr)) for c, rr in zip(C @ T[:3, :3].T + T[:3, 3], r)]
    return out


def pad_sphere_grid(pad: trimesh.Trimesh, inward: np.ndarray, radius: float, pitch: float):
    """Spheres of `radius` on a `pitch` grid, tangent to the pad's contact surface from inside.

    The contact surface is the part of the pad mesh whose normals face `inward` (toward the
    palm centre, i.e. the object). A plane fitted to it gives the grid axes; rays cast from
    outside along -normal find the actual surface at each grid point, so a curved or chamfered
    pad keeps its shape. Each sphere centre sits `radius` behind the surface along the local
    surface normal: the outer envelope of the grid IS the pad surface.
    """
    fn = pad.face_normals
    sel = fn @ inward > 0.7
    pts = pad.triangles_center[sel]
    c0 = pts.mean(0)
    _, _, Vt = np.linalg.svd(pts - c0)
    n = Vt[2] * np.sign(Vt[2] @ inward)
    u = np.cross([0, 1, 0], n)
    u = u / np.linalg.norm(u) if np.linalg.norm(u) > 1e-6 else Vt[0]
    v = np.cross(n, u)
    surf = pad.vertices[np.unique(pad.faces[sel])]
    a, b = (surf - c0) @ u, (surf - c0) @ v
    ga = np.arange(a.min() + radius, a.max() - radius + 1e-9, pitch)
    gb = np.arange(b.min() + radius, b.max() - radius + 1e-9, pitch)
    ga += (a.min() + a.max() - ga.min() - ga.max()) / 2          # centre the grid on the face
    gb += (b.min() + b.max() - gb.min() - gb.max()) / 2
    A, B = np.meshgrid(ga, gb, indexing="ij")
    origins = c0 + A.reshape(-1, 1) * u + B.reshape(-1, 1) * v + n * 0.01
    dirs = np.tile(-n, (len(origins), 1))
    locs, ray_idx, tri = pad.ray.intersects_location(origins, dirs, multiple_hits=False)
    keep = fn[tri] @ inward > 0.5
    centres = locs[keep] - fn[tri[keep]] * radius
    # the contact face for the site: plane through the hits, extent of the grid
    hits = locs[keep]
    hc = hits.mean(0)
    half = [float(np.ptp((hits - hc) @ u) / 2 + radius), float(np.ptp((hits - hc) @ v) / 2 + radius)]
    R = np.column_stack([u, v, n])
    return centres, dict(pos=hc, R=R, half=half)


def pad_side_spheres(pad: trimesh.Trimesh, inward: np.ndarray, radius: float, pitch: float,
                     existing: np.ndarray, frame: dict) -> np.ndarray:
    """A UNIFORM grid of spheres over the fingertip's two SIDES and its END.

    The grids live on the faces of the pad's bounding box in the contact face's own frame
    (`frame["R"]` = [u, v, n]: u along the finger, v across it, n the contact normal), with the
    same `pitch` and the same lattice offsets as the contact-face grid, so rows line up across
    the edges. Each grid point is ray-cast inward onto the real pad surface and the sphere centre
    placed `radius` behind it. The back (-n) and the top (the end facing the bracket) are left
    bare. Points within 0.9 `pitch` of an existing sphere are dropped (edge seams).
    """
    R = frame["R"]
    u, v, n = R[:, 0], R[:, 1], R[:, 2]
    c0 = frame["pos"]
    V = pad.vertices - c0
    A, B, H = V @ u, V @ v, V @ n
    up = np.array([0.0, 0.0, 1.0])                     # proximal, in the straightened tip frame

    def lattice(lo, hi, anchor):
        """`pitch` grid covering [lo+r, hi-r], sharing the contact grid's phase at `anchor`."""
        k0 = np.ceil((lo + radius - anchor) / pitch)
        k1 = np.floor((hi - radius - anchor) / pitch)
        return anchor + pitch * np.arange(k0, k1 + 1)

    # contact-face lattice phase (its grid is centred on the face)
    a_anchor, b_anchor = 0.0, 0.0
    if len(existing):
        Ea, Eb = (existing - c0) @ u, (existing - c0) @ v
        a_anchor, b_anchor = Ea[0], Eb[0]
    h_anchor = -radius                                   # first row level with the face spheres
    faces = []
    for sgn in (+1, -1):                                 # the two sides: normals +-v
        ga, gh = lattice(A.min(), A.max(), a_anchor), lattice(H.min(), H.max(), h_anchor)
        GA, GH = np.meshgrid(ga, gh, indexing="ij")
        o = c0 + GA.reshape(-1, 1) * u + GH.reshape(-1, 1) * n + v * sgn * (np.abs(B).max() + 0.01)
        faces.append((o, -sgn * v))
    for sgn in (+1, -1):                                 # the two ends: normals +-u
        if (sgn * u) @ up > 0.7:
            continue                                     # the top, seated in the bracket
        gb, gh = lattice(B.min(), B.max(), b_anchor), lattice(H.min(), H.max(), h_anchor)
        GB, GH = np.meshgrid(gb, gh, indexing="ij")
        o = c0 + GB.reshape(-1, 1) * v + GH.reshape(-1, 1) * n + u * sgn * (np.abs(A).max() + 0.01)
        faces.append((o, -sgn * u))
    placed = [np.asarray(existing).reshape(-1, 3)]
    fn = pad.face_normals
    for origins, d in faces:
        locs, _, tri = pad.ray.intersects_location(origins, np.tile(d, (len(origins), 1)), multiple_hits=False)
        ok = fn[tri] @ (-d) > 0.5                        # hit a face that looks back at the ray
        centres = locs[ok] - fn[tri[ok]] * radius
        prev = np.vstack(placed)
        if len(prev) and len(centres):
            dmin = np.min(np.linalg.norm(centres[:, None] - prev[None], axis=2), axis=1)
            centres = centres[dmin > pitch * 0.9]
        if len(centres):
            placed.append(centres)
    return np.vstack(placed[1:]) if len(placed) > 1 else np.zeros((0, 3))


def mat2quat(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    import mujoco
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R).ravel())
    return q


def concat(meshes):
    return trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0].copy()


def fmt(v, n=6):
    return " ".join(f"{x:.{n}f}" for x in np.ravel(v))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", type=Path, default=DEFAULT_STEP)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--finger-geom", choices=("overlay", "ellipsoid", "mesh", "morphit"), default=None,
                    help="default: morphit if MORPHIT_DIR/spheres.json exists, else overlay. "
                         "overlay: CAD meshes for display + one fitted ellipsoid per "
                         "phalanx (translucent) for contact; ellipsoid: ellipsoids only; mesh: CAD "
                         "meshes + box/convex-pad collision; morphit: CAD meshes for display + "
                         "MorphIt sphere sets (MORPHIT_DIR/spheres.json) for contact")
    ap.add_argument("--cad-visuals", action="store_true",
                    help="morphit mode: also emit the CAD meshes (fingers + palm) for display. Off by "
                         "default -- the model is then spheres only")
    ap.add_argument("--pad-sphere-radius", type=float, default=PAD_SPHERE_R, help="m, pad grid sphere radius")
    ap.add_argument("--pad-pitch", type=float, default=PAD_PITCH, help="m, pad grid spacing")
    ap.add_argument("--export-links", action="store_true",
                    help=f"also write watertight per-link meshes + correspondence data to "
                         f"{MORPHIT_DIR.relative_to(ROOT)}/ for scripts/run_morphit_real_v3.py")
    args = ap.parse_args()
    if args.finger_geom is None:
        args.finger_geom = "morphit" if (MORPHIT_DIR / "spheres.json").exists() else "overlay"
    sphere_sets = load_morphit_spheres() if args.finger_geom == "morphit" else None
    out = args.out
    mesh_dir = out / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    for old in mesh_dir.glob("*.stl"):   # generator-owned: drop meshes a previous mode wrote
        old.unlink()

    print(f"reading {args.step}")
    parts = load_step(args.step)
    by_asm: dict[str, list[Part]] = {}
    for p in parts:
        by_asm.setdefault(p.assembly, []).append(p)
    finger_asms = sorted(a for a in by_asm if a.startswith("Finger"))
    gantry_asms = sorted(a for a in by_asm if a.startswith("gantry"))
    if len(finger_asms) != 3 or len(gantry_asms) != 3:
        raise RuntimeError(f"unexpected assemblies: {sorted(by_asm)}")

    fingers = [analyse_finger(a, by_asm[a]) for a in finger_asms]

    # ---- palm origin: centre of the three gantry base frames (largest part in each gantry)
    frames = {a: max(by_asm[a], key=lambda p: (p.bbox[1][0] - p.bbox[0][0]) * (p.bbox[1][1] - p.bbox[0][1]))
              for a in gantry_asms}
    fb = np.array([[*frames[a].bbox[0][:2], *frames[a].bbox[1][:2]] for a in gantry_asms])
    origin_S = np.array([(fb[:, 0].min() + fb[:, 2].max()) / 2, (fb[:, 1].min() + fb[:, 3].max()) / 2, 0.0])

    def to_M(pts):   # STEP mm -> palm frame m
        return (np.asarray(pts) - origin_S) @ R_S2M.T / 1000.0

    def dir_M(d):
        return R_S2M @ d

    # ---- name the fingers: thumb is the odd one out in palm x, index has the larger palm y
    yawpt = {f.asm: to_M(f.axes["A"][1]) for f in fingers}
    for f in fingers:
        # use the closest point of A to B as the yaw anchor
        dA, pA = f.axes["A"]
        dB, pB = f.axes["B"]
        qA, qB = closest_points(pA, dA, pB, dB)
        dC, pC = f.axes["C"]
        qC = pC + dC * ((qB - pC) @ dC)
        f.q = {"A": qA, "B": qB, "C": qC}
        yawpt[f.asm] = to_M(qA)
    xs = {a: v[0] for a, v in yawpt.items()}
    thumb = min(xs, key=xs.get)
    pair = sorted((a for a in xs if a != thumb), key=lambda a: yawpt[a][1])
    names = {thumb: "thumb", pair[1]: "index", pair[0]: "middle"}
    F = {names[f.asm]: f for f in fingers}

    # cross-check index/middle against the mirrored rail part ("Part 3") in the middle gantry
    gantry_of = {}
    for a in gantry_asms:
        lo, hi = np.min([p.bbox[0] for p in by_asm[a]], 0), np.max([p.bbox[1] for p in by_asm[a]], 0)
        for n, f in F.items():
            if np.all(f.q["A"][:2] >= lo[:2]) and np.all(f.q["A"][:2] <= hi[:2]):
                gantry_of[n] = a
    mirrored = [a for a in gantry_asms if any(p.path[-1] == "Part 3" for p in by_asm[a])]
    if mirrored and gantry_of.get("middle") != mirrored[0]:
        print(f"WARNING: middle finger is not in the mirrored gantry ({mirrored[0]}) -- check naming")

    # ---- workspace centres from the rails
    def rails(a):
        out = []
        for p in by_asm[a]:
            ext = p.bbox[1] - p.bbox[0]
            k = int(np.argmax(ext))
            if ext[k] > 90 and np.sort(ext)[1] < 8.5 and p.cyl and max(c[0] for c in p.cyl) <= 1.05:
                out.append((k, p))
        return out

    def rail_block(a, k, rail):
        # the carriage block: a 1923 mm^3-class part straddling the rail's cross-section
        lo, hi = rail.bbox
        best = None
        for p in by_asm[a]:
            if p is rail or not (1500 < p.volume < 2500):
                continue
            plo, phi = p.bbox
            other = [j for j in range(3) if j != k]
            if all(plo[j] <= (lo[j] + hi[j]) / 2 <= phi[j] for j in other):
                best = p
        return best

    centre_S, half = {}, {}
    for n, f in F.items():
        a = gantry_of[n]
        c = f.q["A"].copy()
        for k, rail in rails(a):
            blk = rail_block(a, k, rail)
            if blk is None:
                raise RuntimeError(f"{a}: no carriage block on rail {rail.idx}")
            # finger and block are rigidly coupled: centre the BLOCK on the rail
            rail_mid = (rail.bbox[0][k] + rail.bbox[1][k]) / 2
            c[k] = f.q["A"][k] + (rail_mid - blk.mesh.centroid[k])
        centre_S[n] = c
        half[n] = (TRAVEL_MM[n][0] / 2000.0, TRAVEL_MM[n][1] / 2000.0)

    # ---- per finger: snap yaw axis to +x, straighten, collect body-frame meshes
    report = {"origin_step_mm": origin_S.tolist(), "fingers": {}}
    body_xml, keys_cad, excludes, assets = {}, {}, [], []
    mats = {}
    link_corr: dict[str, np.ndarray] = {}
    if args.export_links:
        (MORPHIT_DIR / "links").mkdir(parents=True, exist_ok=True)
    ell_report: dict[str, dict] = {}
    pad_report: dict[str, dict] = {}

    def part_mat(p: Part) -> str:
        return "servo_mat" if is_servo(p) else "print_mat"

    def emit_meshes(prefix, parts_T, frame_T, collide_boxes=True, hull_parts=(), ell_parts="same", ell_name=None):
        """Geoms for one body. parts_T: [(Part, 4x4 straightening transform)]; frame_T: palm->body.

        --finger-geom decides what is emitted:
            mesh       CAD visual meshes + box collision (+ convex pad hull on the tip)
            ellipsoid  only the fitted ellipsoid(s), which both render and collide
            overlay    CAD visual meshes (no contact) + translucent ellipsoids that collide
        `ell_parts` is the geometry the ellipsoid is fitted to ("same" = parts_T, None = no
        ellipsoid on this body) -- the distal phalanx's ellipsoid lives on the tip body but is
        fitted to the whole phalanx.
        """
        mode = args.finger_geom
        tip = bool(hull_parts)
        xml = []

        def body_mesh(p, T):
            vM = (T[:3, :3] @ to_M(p.mesh.vertices).T).T + T[:3, 3]
            return trimesh.Trimesh((frame_T[:3, :3] @ vM.T).T + frame_T[:3, 3], p.mesh.faces, process=False)

        if mode in ("mesh", "overlay") or (mode == "morphit" and args.cad_visuals):
            groups: dict[str, list] = {}
            for p, T in parts_T:
                groups.setdefault(part_mat(p), []).append(body_mesh(p, T))
            for mat, ms in groups.items():
                nm = f"{prefix}_{mat.split('_')[0]}"
                m = decimate(concat(ms), VIS_FACES)
                m.export(mesh_dir / f"{nm}.stl")
                assets.append(f'    <mesh name="{nm}" file="meshes/{nm}.stl" />')
                xml.append(f'<geom class="visual" mesh="{nm}" material="{mat}" />')
        if mode == "mesh":
            for p, T in parts_T:
                vB = body_mesh(p, T).vertices
                if p.idx in hull_parts:
                    nm = f"{prefix}_pad"
                    trimesh.Trimesh(vB).convex_hull.export(mesh_dir / f"{nm}.stl")
                    assets.append(f'    <mesh name="{nm}" file="meshes/{nm}.stl" />')
                    xml.append(f'<geom class="collision" type="mesh" mesh="{nm}" friction="2.4 0.2 0.02" />')
                elif collide_boxes and p.volume > 900:
                    lo, hi = vB.min(0), vB.max(0)
                    xml.append(f'<geom class="collision" type="box" pos="{fmt((lo + hi) / 2)}" '
                               f'size="{fmt(np.maximum((hi - lo) / 2, 5e-4))}" />')
            return xml
        if mode == "morphit":
            fname = prefix.split("_", 1)[0]
            if tip:
                pad_mesh = concat([body_mesh(p, T) for p, T in parts_T])
                inward = np.array([-np.sign(qA[0]), 0.0, 0.0])        # toward the palm centre
                centres, site = pad_sphere_grid(pad_mesh, inward, args.pad_sphere_radius, args.pad_pitch)
                sides = pad_side_spheres(pad_mesh, inward, args.pad_sphere_radius, args.pad_pitch, centres, site)
                pad_report[fname] = dict(n_face=len(centres), n_sides=len(sides),
                                         face_half_mm=(np.asarray(site["half"]) * 1e3).round(2).tolist())
                centres = np.vstack([centres, sides]) if len(sides) else centres
                for k, c in enumerate(centres):
                    xml.append(f'<geom name="{fname}_pad_s{k}" type="sphere" pos="{fmt(c)}" '
                               f'size="{args.pad_sphere_radius:.6f}" material="pad_mat" friction="2.4 0.2 0.02" />')
                # the pad's contact face as a site: centre on the surface, z = outward normal,
                # size = in-plane half-extents (the grasp IK and the reward reason about it)
                xml.append(f'<site name="{fname}_pad_site" type="box" pos="{fmt(site["pos"])}" '
                           f'quat="{fmt(mat2quat(site["R"]))}" size="{site["half"][0]:.6f} {site["half"][1]:.6f} 0.0003" '
                           f'rgba="1 0.2 0.2 0.5" />')
                return xml
            link = prefix.split("_", 1)[1]
            if (fname, link) not in sphere_sets:
                return xml
            fr = ' friction="2.4 0.2 0.02"' if tip else ""
            for k, (c, r) in enumerate(sphere_sets[(fname, link)]):
                xml.append(f'<geom name="{fname}_{link}_s{k}" type="sphere" pos="{fmt(c)}" size="{r:.6f}" '
                           f'material="ell_mat"{fr} />')
            return xml
        if ell_parts is None:
            return xml
        fit_parts = parts_T if ell_parts == "same" else ell_parts
        name = ell_name or prefix
        ells, iou, vr = fit_ellipsoids([body_mesh(p, T) for p, T in fit_parts])
        ell_report[name] = dict(n=len(ells), iou=round(iou, 3), volume_ratio=round(vr, 3))
        fr = ' friction="2.4 0.2 0.02"' if tip else ""
        mat = "ell_mat" if mode == "overlay" else "print_mat"
        for k, (c, semi, R) in enumerate(ells):
            xml.append(f'<geom name="{name}_ell{k}" type="ellipsoid" pos="{fmt(c)}" quat="{fmt(mat2quat(R))}" '
                       f'size="{fmt(semi)}" material="{mat}"{fr} />')
        return xml

    def inertial(parts_T, frame_T):
        mass, com, I = 0.0, np.zeros(3), np.zeros((3, 3))
        items = []
        for p, T in parts_T:
            vM = to_M(p.mesh.vertices)
            vM = (T[:3, :3] @ vM.T).T + T[:3, 3]
            vB = (frame_T[:3, :3] @ vM.T).T + frame_T[:3, 3]
            m = trimesh.Trimesh(vB, p.mesh.faces, process=False)
            vol = p.volume * 1e-9
            mm = SERVO_MASS if is_servo(p) else vol * PRINT_DENSITY
            if not m.is_volume:
                c = vB.mean(0)
                Ic = np.eye(3) * mm * (np.ptp(vB, 0).mean() / 2) ** 2 / 3
            else:
                m.density = mm / max(m.volume, 1e-12)
                c = m.center_mass
                Ic = m.moment_inertia
            items.append((mm, c, Ic))
            mass += mm
            com += mm * c
        com /= mass
        for mm, c, Ic in items:
            d = c - com
            I += Ic + mm * (d @ d * np.eye(3) - np.outer(d, d))
        I = (I + I.T) / 2
        return (f'<inertial pos="{fmt(com)}" mass="{mass:.6f}" '
                f'fullinertia="{I[0,0]:.4e} {I[1,1]:.4e} {I[2,2]:.4e} {I[0,1]:.4e} {I[0,2]:.4e} {I[1,2]:.4e}" />'), mass

    for n, f in F.items():
        qA, qB, qC = (to_M(f.q[k]) for k in "ABC")
        dA, dB, dC = (dir_M(f.axes[k][0]) for k in "ABC")
        # 0. snap: rotate the whole finger about qA so the yaw axis is exactly +/-x
        if dA[0] < 0:
            dA = -dA
        ax = np.cross(dA, [1, 0, 0])
        T_snap = np.eye(4)
        if np.linalg.norm(ax) > 1e-9:
            T_snap = rigid(rot(ax, np.arcsin(np.clip(np.linalg.norm(ax), -1, 1))), qA)
        snap_deg = float(np.degrees(np.arcsin(min(1.0, np.linalg.norm(ax)))))

        def apply(T, p):
            return T[:3, :3] @ p + T[:3, 3]

        qB1, qC1 = apply(T_snap, qB), apply(T_snap, qC)
        dB1, dC1 = T_snap[:3, :3] @ dB, T_snap[:3, :3] @ dC
        X = np.array([1.0, 0, 0])
        down = np.array([0, 0, -1.0])
        # 1. yaw: A->B onto -z
        th_yaw = signed_angle(down, qB1 - qA, X)
        T_yaw = rigid(rot(X, -th_yaw), qA)
        qB2, qC2 = apply(T_yaw, qB1), apply(T_yaw, qC1)
        # flexion axis: positive rotation swings the tip toward the palm centre in x
        pad_side = np.array([-np.sign(qA[0]), 0, 0])
        a_flex = unit(np.cross(down, pad_side))
        dB2 = T_yaw[:3, :3] @ dB1
        if abs(abs(dB2 @ a_flex) - 1) > 2e-2:
            raise RuntimeError(f"{n}: MCP axis not along y after straightening ({dB2})")
        # 2. MCP: B->C onto -z
        th_mcp = signed_angle(down, qC2 - qB2, a_flex)
        T_mcp = rigid(rot(a_flex, -th_mcp), qB2)
        qC3 = apply(T_mcp, qC2)
        # 3. PIP: distal principal axis (in the plane normal to the PIP axis) onto -z
        T_dist_so_far = T_mcp @ T_yaw @ T_snap
        dv = np.vstack([to_M(p.mesh.vertices) for p in f.segments["dist"]])
        dv = (T_dist_so_far[:3, :3] @ dv.T).T + T_dist_so_far[:3, 3] - qC3
        dv2 = dv - np.outer(dv @ a_flex, a_flex)
        w, V = np.linalg.eigh(np.cov(dv2.T))
        long_ax = V[:, np.argmax(w)]
        if long_ax @ dv2.mean(0) < 0:
            long_ax = -long_ax
        th_pip = signed_angle(down, long_ax, a_flex)
        T_pip = rigid(rot(a_flex, -th_pip), qC3)

        T_seg = {"mount": T_snap, "prox": T_yaw @ T_snap, "mid": T_mcp @ T_yaw @ T_snap,
                 "dist": T_pip @ T_mcp @ T_yaw @ T_snap}
        # straight-pose joint anchors (palm frame)
        pA, pB, pC = qA, qB2, qC3
        # body frames are axis-aligned at the straight pose: body-from-palm is a translation
        def frame(p0):
            T = np.eye(4)
            T[:3, 3] = -p0
            return T

        # pad part: the distal-segment part furthest from the PIP axis
        dist_parts = f.segments["dist"]
        pad = max((p for p in dist_parts if p.volume > 2000),
                  key=lambda p: np.linalg.norm(apply(T_seg["dist"], to_M(p.mesh.centroid)) - pC))
        pad_c = apply(T_seg["dist"], to_M(pad.mesh.centroid))
        tip_ext = float(np.max(-(((T_seg["dist"][:3, :3] @ to_M(pad.mesh.vertices).T).T + T_seg["dist"][:3, 3]) - pC)[:, 2]))

        cen = to_M(centre_S[n])
        mount_pos = np.array([cen[0], cen[1], pA[2]])
        # slide offset of the CAD pose from the workspace centre
        cad_xy = pA[:2] - mount_pos[:2]

        seg_parts = {k: [(p, T_seg[k]) for p in v] for k, v in f.segments.items()}
        flex_ax = fmt(a_flex, 0)
        ind = "      "
        mount_inert, m0 = inertial(seg_parts["mount"], frame(pA))
        prox_inert, m1 = inertial(seg_parts["prox"], frame(pA))
        mid_inert, m2 = inertial(seg_parts["mid"], frame(pB))
        dist_rest = [(p, T) for p, T in seg_parts["dist"] if p is not pad]
        if args.export_links:
            link_parts = {"mount": (seg_parts["mount"], frame(pA)), "prox": (seg_parts["prox"], frame(pA)),
                          "mid": (seg_parts["mid"], frame(pB)), "dist": (dist_rest, frame(pC)),
                          "pad": ([(pad, T_seg["dist"])], frame(pad_c))}
            for link, (pl, fT) in link_parts.items():
                if link not in MORPHIT_LINKS:
                    continue
                ms = []
                for p, T in pl:
                    vM = (T[:3, :3] @ to_M(p.mesh.vertices).T).T + T[:3, 3]
                    ms.append(trimesh.Trimesh((fT[:3, :3] @ vM.T).T + fT[:3, 3], p.mesh.faces, process=False))
                key = max(range(len(pl)), key=lambda i: pl[i][0].volume)
                link_corr[f"{n}_{link}"] = ms[key].vertices
                if n == MORPHIT_REF_FINGER:
                    wt = watertight(ms)
                    wt.export(MORPHIT_DIR / "links" / f"{link}.stl")
                    print(f"exported {link}: watertight={wt.is_watertight} volume={wt.volume*1e9:.0f} mm^3")
        dist_inert, m3 = inertial(dist_rest, frame(pC))
        tip_inert, m4 = inertial([(pad, T_seg["dist"])], frame(pad_c))

        def g(lines, depth):
            return "\n".join(f"{ind}{'  ' * depth}{l}" for l in lines)

        hx, hy = half[n]
        xml = f"""{ind}<body name="{n}_mount" pos="{fmt(mount_pos)}">
{ind}  {mount_inert}
{ind}  <joint class="morph" name="{n}_x" axis="1 0 0" range="{-hx:.6f} {hx:.6f}" />
{ind}  <joint class="morph" name="{n}_y" axis="0 1 0" range="{-hy:.6f} {hy:.6f}" />
{g(emit_meshes(f"{n}_mount", seg_parts["mount"], frame(pA)), 1)}
{ind}  <body name="{n}_yaw_frame">
{ind}    {prox_inert}
{ind}    <joint class="ctrl" name="{n}_yaw" axis="1 0 0" range="{-YAW_RANGE:.6f} {YAW_RANGE:.6f}" />
{g(emit_meshes(f"{n}_prox", seg_parts["prox"], frame(pA)), 2)}
{ind}    <body name="{n}_mcp_frame" pos="{fmt(pB - pA)}">
{ind}      {mid_inert}
{ind}      <joint class="ctrl" name="{n}_mcp" axis="{flex_ax}" range="{MCP_RANGE[0]:.6f} {MCP_RANGE[1]:.6f}" />
{g(emit_meshes(f"{n}_mid", seg_parts["mid"], frame(pB)), 3)}
{ind}      <body name="{n}_len_frame" pos="0 0 0">
{ind}        <inertial pos="0 0 0" mass="0.001" diaginertia="1e-6 1e-6 1e-6" />
{ind}        <joint class="morph" name="{n}_len" axis="1 0 0" range="0 0" />
{ind}        <body name="{n}_pip_frame" pos="{fmt(pC - pB)}">
{ind}          {dist_inert}
{ind}          <joint class="ctrl" name="{n}_pip" axis="{flex_ax}" range="{PIP_RANGE[0]:.6f} {PIP_RANGE[1]:.6f}" />
{g(emit_meshes(f"{n}_dist", dist_rest, frame(pC), ell_parts=None), 5)}
{ind}          <body name="{n}_tip" pos="{fmt(pad_c - pC)}">
{ind}            {tip_inert}
{g(emit_meshes(f"{n}_tip", [(pad, T_seg["dist"])], frame(pad_c), collide_boxes=False, hull_parts=(pad.idx,),
              # one ellipsoid for the whole distal phalanx (bracket + pad), on the tip body so
              # the fingertip ContactMatch still sees it
              ell_parts=seg_parts["dist"], ell_name=f"{n}_dist"), 6)}
{ind}          </body>
{ind}        </body>
{ind}      </body>
{ind}    </body>
{ind}  </body>
{ind}</body>"""
        body_xml[n] = xml
        keys_cad[n] = [cad_xy[0], cad_xy[1], th_yaw, th_mcp, 0.0, th_pip]
        report["fingers"][n] = dict(
            assembly=f.asm, gantry=gantry_of[n],
            servo_parts={k: v.idx for k, v in f.servos.items()},
            segment_parts={k: sorted(p.idx for p in v) for k, v in f.segments.items()},
            mount_pos_m=mount_pos.tolist(), workspace_half_m=list(half[n]),
            yaw_to_mcp_mm=float(np.linalg.norm(pB - pA) * 1000),
            mcp_to_pip_mm=float(np.linalg.norm(pC - pB) * 1000),
            pip_to_pad_tip_mm=tip_ext * 1000,
            mount_to_tip_mm=float((pA[2] - (pC[2] - tip_ext)) * 1000),
            yaw_axis_below_plate_mm=float(-pA[2] * 1000),
            yaw_axis_tilt_snapped_deg=snap_deg,
            cad_pose_deg=dict(yaw=np.degrees(th_yaw), mcp=np.degrees(th_mcp), pip=np.degrees(th_pip)),
            cad_slide_mm=(cad_xy * 1000).tolist(),
            masses_kg=dict(mount=m0, prox=m1, mid=m2, dist=m3, tip=m4),
        )
        excludes += [f'    <exclude body1="palm_pose" body2="{n}_{b}" />'
                     for b in ("mount", "yaw_frame", "mcp_frame", "len_frame", "pip_frame", "tip")]
        chain = ["mount", "yaw_frame", "mcp_frame", "len_frame", "pip_frame", "tip"]
        for i, b1 in enumerate(chain):
            for b2 in chain[i + 2:]:
                excludes.append(f'    <exclude body1="{n}_{b1}" body2="{n}_{b2}" />')

    # ---- static palm: everything that is not a finger and not a moving gantry carriage
    finger_ids = {p.idx for f in fingers for p in f.parts}
    moving = set()
    for a in gantry_asms:
        # The fixed rail sits on the frame (nearest z = 0); the bridge rides it, carrying the
        # other rail, its motor and the finger carriage. The bridge is the longest part that
        # straddles the bridge rail; everything inside the bridge's footprint along the fixed
        # rail moves with it and is left out of the static palm visual.
        rs = rails(a)
        k_fixed, rail_f = max(rs, key=lambda kr: kr[1].bbox[1][2])
        bridge_rail = next(r for _, r in rs if r is not rail_f)
        span = 1 - k_fixed
        mid = (bridge_rail.bbox[0][k_fixed] + bridge_rail.bbox[1][k_fixed]) / 2
        bridge = max((p for p in by_asm[a] if p is not frames[a] and p.bbox[0][k_fixed] <= mid <= p.bbox[1][k_fixed]),
                     key=lambda p: p.bbox[1][span] - p.bbox[0][span])
        lo, hi = bridge.bbox[0][k_fixed] - 2, bridge.bbox[1][k_fixed] + 2
        for p in by_asm[a]:
            if p is frames[a] or p is rail_f:
                continue
            if p.bbox[0][k_fixed] >= lo and p.bbox[1][k_fixed] <= hi:
                moving.add(p.idx)
    static = [p for p in parts if p.idx not in finger_ids and p.idx not in moving]
    palm_groups = {"plate": [], "frame": [], "motor": [], "base": []}
    for p in static:
        ext = p.bbox[1] - p.bbox[0]
        if p.volume > 5e5:
            palm_groups["base"].append(p)
        elif ext[2] <= 3.5 and p.volume > 1e5:
            palm_groups["plate"].append(p)
        elif 10000 < p.volume < 12500:
            palm_groups["motor"].append(p)
        else:
            palm_groups["frame"].append(p)
    palm_xml = []
    mat_for = {"plate": "frame_mat", "frame": "frame_mat", "motor": "motor_mat", "base": "base_mat"}
    for k, ps in (palm_groups.items() if (args.finger_geom != "morphit" or args.cad_visuals) else ()):
        if not ps:
            continue
        m = concat([trimesh.Trimesh(to_M(p.mesh.vertices), p.mesh.faces, process=False) for p in ps])
        m = decimate(m, 3 * VIS_FACES)
        m.export(mesh_dir / f"palm_{k}.stl")
        assets.append(f'    <mesh name="palm_{k}" file="meshes/palm_{k}.stl" />')
        palm_xml.append(f'      <geom class="visual" mesh="palm_{k}" material="{mat_for[k]}" />')
    # palm collision slab: from the plate down to the lowest moving-carriage surface
    gantry_parts = [p for a in gantry_asms for p in by_asm[a]]
    under = min(p.bbox[0][2] for p in gantry_parts)
    glo = to_M(np.min([p.bbox[0] for p in gantry_parts], 0))
    ghi = to_M(np.max([p.bbox[1] for p in gantry_parts], 0))
    lo_xy, hi_xy = np.minimum(glo, ghi)[:2], np.maximum(glo, ghi)[:2]
    slab_c = np.array([(lo_xy[0] + hi_xy[0]) / 2, (lo_xy[1] + hi_xy[1]) / 2, under / 2000.0])
    slab_h = np.array([(hi_xy[0] - lo_xy[0]) / 2, (hi_xy[1] - lo_xy[1]) / 2, -under / 2000.0])
    palm_xml.append(f'      <geom name="palm_slab" class="collision" type="box" pos="{fmt(slab_c)}" size="{fmt(slab_h)}" '
                    f'group="{3 if args.finger_geom != "morphit" or args.cad_visuals else 0}" rgba="0.15 0.15 0.17 0.5" />')

    # ---- palm height: same fraction of the flexing reach as real_v1's fitted 0.0625
    reach = np.mean([r["mount_to_tip_mm"] for r in report["fingers"].values()]) / 1000
    yaw_depth = np.mean([r["yaw_axis_below_plate_mm"] for r in report["fingers"].values()]) / 1000
    palm_z = yaw_depth + REAL_V1["palm_z"] / REAL_V1["reach"] * reach
    report["palm_z_m"] = palm_z
    report["moving_gantry_parts_omitted"] = sorted(moving)
    if ell_report:
        report["ellipsoid_fit"] = ell_report
    if pad_report:
        report["pad_grid"] = pad_report

    sites = []
    colors = {"thumb": "0.85 0.30 0.30", "index": "0.30 0.55 0.85", "middle": "0.30 0.75 0.45"}
    for n in FINGERS:
        mp = report["fingers"][n]["mount_pos_m"]
        hx, hy = half[n]
        sites.append(f'      <site name="workspace_{n}" type="box" pos="{mp[0]:.4f} {mp[1]:.4f} {under / 1000 - 0.001:.4f}" '
                     f'size="{hx:.4f} {hy:.4f} 0.0005" rgba="{colors[n]} 0.25" />')

    def key(name, per_finger, ctrl):
        q = [v for n in FINGERS for v in per_finger[n]]
        return f'    <key name="{name}" qpos="{" ".join(f"{v:.6g}" for v in q)}" ctrl="{" ".join(f"{v:.6g}" for v in ctrl)}" />'

    open_q = {n: [0, 0, 0, OPEN_MCP, 0, OPEN_PIP] for n in FINGERS}
    open_c = [0, OPEN_MCP, OPEN_PIP] * 3
    cad_c = [v for n in FINGERS for v in (keys_cad[n][2], keys_cad[n][3], keys_cad[n][5])]
    actuators = "\n".join(
        f'    <position class="ctrl" name="a_{n}_{j}" joint="{n}_{j}" ctrlrange="{lo:.6f} {hi:.6f}" />'
        for n in FINGERS for j, (lo, hi) in
        (("yaw", (-YAW_RANGE, YAW_RANGE)), ("mcp", MCP_RANGE), ("pip", PIP_RANGE)))

    sphere_alpha = 0.35 if (args.finger_geom != "morphit" or args.cad_visuals) else 1.0
    xml = f"""<mujoco model="real_v3_hand">
  <!-- GENERATED by scripts/build_real_v3_from_step.py from {args.step.name}. Do not hand-edit.
       Geometry, joint axes, link offsets, masses of printed parts and mount positions come from
       the STEP; joint ROM is real_v1's (v3 servos not yet calibrated); palm height is scaled
       from real_v1, not re-fitted. See cad_kinematics.json for the extracted numbers. -->
  <compiler angle="radian" autolimits="true" />
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast" />
  <visual>
    <headlight ambient=".4 .4 .4" diffuse=".8 .8 .8" specular="0.1 0.1 0.1" />
    <map znear=".01" />
    <quality shadowsize="2048" />
    <global elevation="-15" />
  </visual>
  <default>
    <geom friction="1.8 0.15 0.01" solref="0.006 1" solimp="0.97 0.995 0.0005" />
    <motor ctrlrange="-1 1" ctrllimited="true" />
    <default class="visual">
      <geom type="mesh" contype="0" conaffinity="0" group="2" />
    </default>
    <default class="collision">
      <geom group="3" rgba="0.9 0.5 0.1 0.35" />
    </default>
    <default class="pose">
      <joint damping="2.0" armature="0.01" />
      <position kp="4000" kv="100" forcerange="-1000 1000" />
    </default>
    <default class="morph">
      <joint type="slide" damping="2000" armature="0.0001" />
    </default>
    <default class="ctrl">
      <joint type="hinge" damping="0.5" armature="0.001" />
      <position kp="30" kv="0.5" gear="1" forcerange="-10 10" ctrlrange="-1 1" />
    </default>
  </default>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072" />
    <texture name="grid" type="2d" builtin="checker" rgb1="0.15 0.15 0.15" rgb2="0.25 0.25 0.25" width="256" height="256" />
    <material name="table_mat" texture="grid" texrepeat="3 3" reflectance="0.2" />
    <material name="print_mat" rgba="0.45 0.33 0.85 1" />
    <material name="ell_mat" rgba="1.0 0.6 0.15 {sphere_alpha}" />
    <material name="pad_mat" rgba="0.2 0.75 0.35 1" />
    <material name="servo_mat" rgba="0.12 0.12 0.13 1" />
    <material name="frame_mat" rgba="0.08 0.08 0.09 1" />
    <material name="motor_mat" rgba="0.55 0.56 0.58 1" />
    <material name="base_mat" rgba="0.75 0.6 0.25 0.35" />
{chr(10).join(assets)}
  </asset>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true" diffuse=".6 .6 .6" specular=".2 .2 .2" />
    <geom name="floor" type="plane" size="1.5 1.5 0.1" material="table_mat" contype="0" conaffinity="0" />
    <body gravcomp="1" name="palm_pose" pos="0 0 {palm_z:.6f}">
{chr(10).join(palm_xml)}
{chr(10).join(sites)}
{chr(10).join(body_xml[n] for n in FINGERS)}
    </body>
  </worldbody>
  <contact>
{chr(10).join(e for e in excludes if e)}
  </contact>
  <actuator>
{actuators}
  </actuator>
  <keyframe>
{key("open", open_q, open_c)}
{key("cad", keys_cad, cad_c)}
  </keyframe>
</mujoco>
"""
    if args.export_links:
        np.savez(MORPHIT_DIR / "correspondence.npz", **link_corr)
    (out / "real_hand.xml").write_text(xml)
    (out / "cad_kinematics.json").write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps(report, indent=2, default=float))

    import mujoco
    m = mujoco.MjModel.from_xml_path(str(out / "real_hand.xml"))
    print(f"compiled: nq={m.nq} nbody={m.nbody} ngeom={m.ngeom} nmesh={m.nmesh}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
