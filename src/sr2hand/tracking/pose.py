"""Detection, pose solves and the bundle adjustment of a tag map (plan: docs/notes/20261105-object_seating_and_tracking/).

Units are mm and pixels. A transform T_AB is the pose of frame B in frame A (4 x 4). A camera frame C is OpenCV's: x
right, y down, z along the optical axis.

solve_body   one body's pose in one camera from every corner of its visible tags at once: SQPnP over all corners (IPPE
             for a single tag), Levenberg-Marquardt, then each tag seen more than max_view_deg from its normal, and
             each tag whose corners reproject worse than max_rpe px, is dropped and the pose solved again from the rest.
solve_multi  one body's pose in a common frame P from the corners seen by several cameras whose poses in P are known.
bundle_adjust  a tag map refined from many views of the body: the views' poses and each tag's pose and size on the body
             (the first tag's pose held as the gauge, measured lengths for the scale), least squares over the
             reprojection error, then the map turned and moved as a whole onto the starting map.
bundle_adjust_multi  the same from views seen by several cameras with known, held poses: one body pose per view shared
             by the cameras, whose baseline sets the scale, so every tag's size is free without measured lengths.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares

from sr2hand.geometry.rotations import exp_so3, kabsch, log_so3, tf, tf_inv
from sr2hand.geometry.vectors import angle

from .tags import TagMap

# Tags seen more steeply than this from their normal are left out of a pose. On the bench (2026-11-09, two D435s at
# 0.28-0.29 m, the printed block) tag 15 at 80 deg decoded in half the frames and moved the solved depth by about 1 mm
# each time it entered: 1.05 mm sd with it, 0.14 mm without. Its corners reprojected under 1 px, so max_rpe kept it. In
# simulation the 13 mm tags decode in 99 % of frames up to 65 deg (scripts/hardware/object_tracking_sim.py tagsize).
MAX_VIEW_DEG = 65.0


@dataclass
class Camera:
    K: np.ndarray
    dist: np.ndarray
    width: int
    height: int

    @property
    def fx(self):
        return float(self.K[0, 0])


def d435_colour(width=1920, height=1080, fx_1080=1397.0) -> Camera:
    """The D435's colour stream as a pinhole without distortion: 69.0 x 42.2 deg at every resolution, fx 1397 px at
    1920 x 1080 (the plan's value), principal point at the centre of the pixel grid ((w - 1) / 2 with pixel centres at
    integers, as a rendered image has it)."""
    f = fx_1080 * width / 1920.0
    K = np.array([[f, 0.0, (width - 1) / 2.0], [0.0, f, (height - 1) / 2.0], [0.0, 0.0, 1.0]])
    return Camera(K, np.zeros(5), width, height)


def detector(family="tag36h11", nthreads=2, quad_decimate=1.0, **kw):
    """pupil_apriltags' detector with the settings of scripts/hardware/real_v1_tag_tracker.py (full resolution)."""
    from pupil_apriltags import Detector
    return Detector(families=family, nthreads=nthreads, quad_decimate=quad_decimate, **kw)


# pupil_apriltags puts pixel centres at +0.5 (the origin at the top-left corner of the first pixel); OpenCV, its
# calibrations and this module put them at integers. On rendered images of known geometry the detector's corners lay
# +0.31 to +0.50 px off in both axes (2026-11-05, scripts/hardware/object_tracking_sim.py), so detect() subtracts 0.5.
CORNER_OFFSET_PX = 0.5


def detect(det, gray, ids=None, min_margin=0.0):
    """{id: (4, 2) corners} of the tags found in a uint8 image, in the detector's corner order and OpenCV's pixel
    convention; `ids` keeps only those, and detections with a decision margin under min_margin are dropped."""
    out = {}
    for d in det.detect(gray):
        if (ids is None or d.tag_id in ids) and d.decision_margin >= min_margin:
            out[int(d.tag_id)] = np.asarray(d.corners, float) - CORNER_OFFSET_PX
    return out


def _rt_to_T(rvec, tvec):
    return tf(exp_so3(np.asarray(rvec, float).ravel()), np.asarray(tvec, float).ravel())


def _T_to_rt(T):
    return log_so3(T[:3, :3]).reshape(3, 1), T[:3, 3].reshape(3, 1).copy()


def project(cam: Camera, T_CB, P_B):
    """Pixels of points P_B (N, 3) of a body at pose T_CB."""
    r, t = _T_to_rt(T_CB)
    uv, _ = cv2.projectPoints(np.asarray(P_B, float).reshape(-1, 1, 3), r, t, cam.K, cam.dist)
    return uv.reshape(-1, 2)


def view_angles(tag_map: TagMap, ids, T_CB):
    """Angles (deg) between each tag's outward normal and its line of sight to the camera, the body at pose T_CB."""
    R, t = T_CB[:3, :3], T_CB[:3, 3]
    return np.array([np.degrees(angle(R @ tag_map.frames[i][3], -(R @ tag_map.frames[i][0] + t))) for i in ids])


def solve_body(obs: dict, tag_map: TagMap, cam: Camera, T0=None, max_rpe=1.0, max_view_deg=MAX_VIEW_DEG):
    """(T_CB, info) of one body from {id: (4, 2) corners}; None when no tag of the map is seen. The steepest tag beyond
    max_view_deg (None: no limit), then the worst tag beyond max_rpe, is dropped and the pose solved again, while more
    than one tag is left. info: the tags used, their view angles (deg), the tags dropped and the reprojection RMS (px)
    of the corners used."""
    use = [i for i in sorted(obs) if i in tag_map.corners]
    if not use:
        return None, {"tags": [], "dropped": [], "rms_px": None}
    dropped = []
    while True:
        P = tag_map.stacked(use)
        uv = np.concatenate([obs[i] for i in use])
        if T0 is not None:
            r, t = _T_to_rt(T0)
            ok = True
        elif len(use) == 1:
            ok, r, t = cv2.solvePnP(P, uv, cam.K, cam.dist, flags=cv2.SOLVEPNP_IPPE)
        else:
            ok, r, t = cv2.solvePnP(P, uv, cam.K, cam.dist, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            return None, {"tags": [], "dropped": use, "rms_px": None}
        r, t = cv2.solvePnPRefineLM(P, uv, cam.K, cam.dist, r, t)
        T = _rt_to_T(r, t)
        e = np.linalg.norm(project(cam, T, P) - uv, axis=1).reshape(len(use), 4)
        per = np.sqrt((e ** 2).mean(1))
        view = view_angles(tag_map, use, T)
        steep = int(view.argmax())
        if max_view_deg is not None and view[steep] > max_view_deg and len(use) > 1:
            dropped.append(use.pop(steep))
            continue
        worst = int(per.argmax())
        if per[worst] > max_rpe and len(use) > 1:
            dropped.append(use.pop(worst))
            continue
        return T, {"tags": use, "view_deg": np.round(view, 1).tolist(), "dropped": dropped,
                   "rms_px": float(np.sqrt((e ** 2).mean()))}


def solve_multi(obs_by_cam: list, tag_map: TagMap, cams: list, T_PC: list, T0_PB, max_rpe=1.0,
                max_view_deg=MAX_VIEW_DEG):
    """(T_PB, info): the body's pose in P from every camera's corners at once. obs_by_cam[k] are camera k's
    {id: corners}, T_PC[k] its pose in P; T0_PB the start (one camera's solve). Tags dropped by each camera's own
    solve_body are left out."""
    rows = []
    for obs, cam, Tpc in zip(obs_by_cam, cams, T_PC):
        T_CB, inf = solve_body(obs, tag_map, cam, T0=tf_inv(Tpc) @ T0_PB, max_rpe=max_rpe, max_view_deg=max_view_deg)
        if T_CB is None:
            continue
        rows.append((cam, tf_inv(Tpc), tag_map.stacked(inf["tags"]), np.concatenate([obs[i] for i in inf["tags"]])))
    if not rows:
        return None, {"n_cams": 0}

    def res(x):
        T = T0_PB @ tf(exp_so3(x[:3]), x[3:])
        return np.concatenate([(project(c, Tcp @ T, P) - uv).ravel() for c, Tcp, P, uv in rows])
    sol = least_squares(res, np.zeros(6), method="lm", x_scale=np.r_[1e-3, 1e-3, 1e-3, 0.1, 0.1, 0.1])
    T = T0_PB @ tf(exp_so3(sol.x[:3]), sol.x[3:])
    e = sol.fun.reshape(-1, 2)
    return T, {"n_cams": len(rows), "n_corners": len(e), "rms_px": float(np.sqrt((e ** 2).sum(1).mean()))}


def bundle_adjust(views: list, tag_map: TagMap, cam: Camera, T0_views: list, fixed=None, sizes=True, lengths=(),
                  px_sd=0.1):
    """(refined TagMap, info) from views [{id: (4, 2) corners}] of one body with starting poses T0_views (T_CB).

    Every tag moves as a square on the body (rotation and translation) except `fixed` (default: the lowest id), whose
    pose holds the gauge. With `sizes` each tag's edge is free as well: a printed tag's black square need not have the
    map's size (the block printed on 2026-11-07 measured 2.5 % large against its across-flats). A camera whose pose is
    free does not see a uniform scale of the body, so the scale comes from `lengths`, measured distances (i, j, mm, sd)
    between the centres of tags i and j; on the hexagonal block the tags of opposite faces lie the across-flats apart.
    Without lengths the fixed tag keeps its size and the others' sizes are relative to it. The residuals are the corners'
    reprojection errors over px_sd and each length's error over its sd. Afterwards the refined corners are fitted
    rigidly onto the starting map's, so the result stays in its frame."""
    ids = tag_map.ids()
    fixed = ids[0] if fixed is None else fixed
    free = [i for i in ids if i != fixed]
    sized = (ids if lengths else free) if sizes else []
    nv, nf = len(views), len(free)
    cen = {i: tag_map.corners[i].mean(0) for i in ids}

    def unpack(x):
        Tv = [T0 @ tf(exp_so3(x[6 * k:6 * k + 3]), x[6 * k + 3:6 * k + 6]) for k, T0 in enumerate(T0_views)]
        s = dict(zip(sized, np.exp(x[6 * (nv + nf):])))
        corners, centres = {}, {}
        for i in ids:
            if i == fixed:
                R, t = np.eye(3), np.zeros(3)
            else:
                p = x[6 * nv + 6 * free.index(i):6 * nv + 6 * free.index(i) + 6]
                R, t = exp_so3(p[:3]), p[3:]
            corners[i] = s.get(i, 1.0) * (tag_map.corners[i] - cen[i]) @ R.T + cen[i] + t
            centres[i] = cen[i] + t
        return Tv, corners, centres

    def res(x):
        Tv, corners, centres = unpack(x)
        out = []
        for T, obs in zip(Tv, views):
            for i, uv in obs.items():
                if i in corners:
                    out.append((project(cam, T, corners[i]) - uv).ravel() / px_sd)
        out.append(np.array([(np.linalg.norm(centres[i] - centres[j]) - L) / sd for i, j, L, sd in lengths]))
        return np.concatenate(out)
    x0 = np.zeros(6 * (nv + nf) + len(sized))
    scale = np.r_[np.tile(np.r_[1e-3, 1e-3, 1e-3, 0.1, 0.1, 0.1], nv + nf), np.full(len(sized), 1e-3)]
    sol = least_squares(res, x0, x_scale=scale, method="trf")
    Tv, corners, centres = unpack(sol.x)
    A = np.concatenate([corners[i] for i in ids])
    B = tag_map.stacked(ids)
    R, t = kabsch(A, B)
    fitted = {i: corners[i] @ R.T + t for i in ids}
    e = sol.fun[:sol.fun.size - len(lengths)].reshape(-1, 2) * px_sd
    return TagMap.from_corners(tag_map.family, tag_map.size, fitted), {
        "views": nv, "corners": len(e), "rms_px": float(np.sqrt((e ** 2).sum(1).mean())), "cost": float(sol.cost),
        "sizes_mm": {i: float(np.linalg.norm(fitted[i][0] - fitted[i][1])) for i in ids},
        "lengths_mm": [(i, j, L, float(np.linalg.norm(centres[i] - centres[j]))) for i, j, L, _sd in lengths],
        "T_views": Tv}


def bundle_adjust_multi(views: list, tag_map: TagMap, cams: list, T_PC: list, T0_views: list, fixed=None, lengths=(),
                        px_sd=0.1, max_view_deg=MAX_VIEW_DEG, loss="linear"):
    """(refined TagMap, info) from views of one body by several cameras whose poses T_PC in a common frame P are known
    and held: views[v][k] = camera k's {id: (4, 2) corners} in view v, T0_views[v] the body's starting pose T_PB.

    The two-camera form of bundle_adjust (lab session of 2026-11-13, the hex block on the UR5's D435s): one body pose per
    view, shared by the cameras. Their baseline fixes the scale, so every tag's edge is free, and every tag's pose except
    `fixed`'s (default: the lowest id), which holds the gauge. `lengths` (i, j, mm, sd) between tag centres are optional
    extra residuals (leave them out to keep a caliper length as an independent check). A camera's tags seen more than
    max_view_deg from their normal at the starting pose are left out, as solve_body leaves them out. The residuals are
    the corners' reprojection errors over px_sd (with scipy's `loss`, f_scale 5, i.e. 0.5 px at px_sd 0.1) and each
    length's error over its sd; a robust loss softens the lengths too, so fit with lengths under loss="linear". On
    the hex block (35 views, 2026-11-13) the map fitted from 28 views and from all 35 differed by 0.03 mm RMS.
    Afterwards the refined corners are fitted rigidly onto the starting map's, so the result
    stays in its frame. info: the reprojection RMS per camera and per view, each tag's size, the views' poses T_PB (in
    the refined map's frame) and the observations used."""
    ids = tag_map.ids()
    fixed = ids[0] if fixed is None else fixed
    free = [i for i in ids if i != fixed]
    nv, nf, nc = len(views), len(free), len(cams)
    T_CP = [tf_inv(T) for T in T_PC]
    cen = {i: tag_map.corners[i].mean(0) for i in ids}
    obs = []                                                  # (view, camera, tag ids, (4n, 2) corners)
    for v, (vw, T0) in enumerate(zip(views, T0_views)):
        for k in range(nc):
            use = sorted(i for i in vw[k] if i in cen)
            if use and max_view_deg is not None:
                ang = view_angles(tag_map, use, T_CP[k] @ T0)
                use = [i for i, a in zip(use, ang) if a <= max_view_deg]
            if use:
                obs.append((v, k, use, np.concatenate([np.asarray(vw[k][i], float) for i in use])))
    o_tag, o_size = 6 * nv, 6 * (nv + nf)

    def unpack(x):
        Tv = [T0 @ tf(exp_so3(x[6 * v:6 * v + 3]), x[6 * v + 3:6 * v + 6]) for v, T0 in enumerate(T0_views)]
        s = np.exp(x[o_size:])
        corners, centres = {}, {}
        for n, i in enumerate(ids):
            if i == fixed:
                R, t = np.eye(3), np.zeros(3)
            else:
                p = x[o_tag + 6 * free.index(i):o_tag + 6 * free.index(i) + 6]
                R, t = exp_so3(p[:3]), p[3:]
            corners[i] = s[n] * (tag_map.corners[i] - cen[i]) @ R.T + cen[i] + t
            centres[i] = cen[i] + t
        return Tv, corners, centres

    def res(x):
        Tv, corners, centres = unpack(x)
        out = [(project(cams[k], T_CP[k] @ Tv[v], np.concatenate([corners[i] for i in use])) - uv).ravel() / px_sd
               for v, k, use, uv in obs]
        out.append(np.array([(np.linalg.norm(centres[i] - centres[j]) - L) / sd for i, j, L, sd in lengths]))
        return np.concatenate(out)

    n_x = o_size + len(ids)
    rows = []                                                 # Jacobian sparsity: which parameters each row reads
    for v, k, use, uv in obs:
        cols = list(range(6 * v, 6 * v + 6))
        for i in use:
            if i != fixed:
                cols += list(range(o_tag + 6 * free.index(i), o_tag + 6 * free.index(i) + 6))
            cols.append(o_size + ids.index(i))
        rows += [cols] * (8 * len(use))
    for i, j, _L, _sd in lengths:
        rows.append([c for t in (i, j) if t != fixed for c in range(o_tag + 6 * free.index(t), o_tag + 6 * free.index(t) + 6)])
    from scipy.sparse import lil_matrix
    sp = lil_matrix((len(rows), n_x), dtype=int)
    for r, cols in enumerate(rows):
        sp[r, cols] = 1
    x0 = np.zeros(n_x)
    scale = np.r_[np.tile(np.r_[1e-3, 1e-3, 1e-3, 0.1, 0.1, 0.1], nv + nf), np.full(len(ids), 1e-3)]
    sol = least_squares(res, x0, x_scale=scale, method="trf", jac_sparsity=sp, loss=loss, f_scale=5.0)
    Tv, corners, centres = unpack(sol.x)
    A = np.concatenate([corners[i] for i in ids])
    R, t = kabsch(A, tag_map.stacked(ids))
    fitted = {i: corners[i] @ R.T + t for i in ids}
    G = tf(R, t)                                              # old body frame -> the refined map's frame
    Tv = [T @ tf_inv(G) for T in Tv]
    e = sol.fun[:sol.fun.size - len(lengths)].reshape(-1, 2) * px_sd
    if loss != "linear":                                      # sol.fun is not the raw residual under a robust loss
        _Tv, _c, _ = unpack(sol.x)
        e = np.concatenate([project(cams[k], T_CP[k] @ _Tv[v], np.concatenate([_c[i] for i in use])) - uv
                            for v, k, use, uv in obs])
    per_cam, per_view, n0 = {k: [] for k in range(nc)}, {v: [] for v in range(nv)}, 0
    for v, k, use, uv in obs:
        ek = e[n0:n0 + len(uv)]
        n0 += len(uv)
        per_cam[k].append(ek)
        per_view[v].append(ek)
    rms = lambda a: float(np.sqrt((np.concatenate(a) ** 2).sum(1).mean())) if a else None  # noqa: E731
    return TagMap.from_corners(tag_map.family, tag_map.size, fitted), {
        "views": nv, "corners": len(e), "rms_px": float(np.sqrt((e ** 2).sum(1).mean())), "cost": float(sol.cost),
        "rms_px_per_cam": [rms(per_cam[k]) for k in range(nc)], "rms_px_per_view": [rms(per_view[v]) for v in range(nv)],
        "sizes_mm": {i: float(np.linalg.norm(fitted[i][0] - fitted[i][1])) for i in ids},
        "lengths_mm": [(i, j, L, float(np.linalg.norm(centres[i] - centres[j]))) for i, j, L, _sd in lengths],
        "T_views": Tv, "obs": [(v, k, use) for v, k, use, _uv in obs], "status": sol.message}


def map_error(est: TagMap, ref: TagMap):
    """RMS and largest distance (mm) between two maps' corners after the rigid fit of est onto ref."""
    ids = [i for i in ref.ids() if i in est.corners]
    A, B = est.stacked(ids), ref.stacked(ids)
    R, t = kabsch(A, B)
    d = np.linalg.norm(A @ R.T + t - B, axis=1)
    return float(np.sqrt((d ** 2).mean())), float(d.max())
