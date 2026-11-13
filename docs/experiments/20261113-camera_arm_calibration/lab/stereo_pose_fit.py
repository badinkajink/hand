"""A moved camera's pose re-solved from the board seen by both D435s at once (board_pair_views.py), the other camera
holding its arm-fitted pose (handeye_fit.py, TCP route).

Unknowns: the moved camera's pose in the fixed camera's frame and the board's pose in each view. Held: both lenses and
the board's fitted shape (lens_board_fit.py), scaled by the arm's board scale (1.0021, handeye_fit.py --fit-scale).
Least squares over both cameras' corner reprojections, soft-L1 with a 1 px knee. The spread comes from 40 bootstrap
draws over the views. Writes the hand-eye JSON with the moved camera's T_CB replaced and a "moved" record.

On 11-13 the method was tried on the hand-held lens views that both cameras happened to save within 0.3 s (26 pairs,
not taken for it): it put 836612071918 within 0.42 mm and 0.14 deg of its arm-fitted pose.

Checks with snap.py's frames taken before and after the move (--before, --after):
  fixed camera  its reference frame's shift (phase correlation, px) and its solve of the untouched hex block
                (fitted map, data/fits/20261113-hex_block_map.json): if either moved, it was bumped;
  hex block     after the move, the block's origin placed through each camera's solve: the distance between the two.

PAIRS is board_pair_views.py's folder or its pairs.jsonl; --before and --after are snap.py's folders (the image shift
needs their PNGs) or their tags.json alone.

    /usr/bin/python3 stereo_pose_fit.py PAIRS LENS.json HANDEYE.json SCALE.json --moved SERIAL --out NEW.json \
        [--before DIR --after DIR]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))
from sr2hand.geometry.rotations import exp_so3, log_so3, tf, tf_inv  # noqa: E402
from sr2hand.tracking import pose as tp  # noqa: E402
from sr2hand.tracking.tags import TagMap  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("pairs")
ap.add_argument("lens")
ap.add_argument("handeye")
ap.add_argument("scale")
ap.add_argument("--moved", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--before")
ap.add_argument("--after")
ap.add_argument("--boot", type=int, default=40)
a = ap.parse_args()
L, H = json.load(open(a.lens)), json.load(open(a.handeye))
sc = json.load(open(a.scale))["board_scale"]
SNS = sorted(H["cams"])
M = a.moved
F = [s for s in SNS if s != M][0]
BOARD = cv2.aruco.CharucoBoard_create(7, 7, 15.0, 10.5, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100))
P = (np.array(BOARD.chessboardCorners, float) + np.array(L["board_offsets_mm"])) * sc
HEX = Path(__file__).resolve().parents[1] / "data/fits/20261113-hex_block_map.json"


def lens(sn):
    l = L["cams"][sn]["lens"]
    return (np.array([[l["fx"], 0, l["cx"]], [0, l["fy"], l["cy"]], [0, 0, 1.0]]),
            np.array([l[k] for k in ("k1", "k2", "p1", "p2", "k3")], float))


K = {sn: lens(sn) for sn in SNS}
pf = Path(a.pairs) / "pairs.jsonl" if Path(a.pairs).is_dir() else Path(a.pairs)
rows = [json.loads(l) for l in open(pf)]
V = [{sn: (np.array(r["cams"][sn]["ids"]), np.array(r["cams"][sn]["corners"], float)) for sn in SNS} for r in rows]


def pnp(ids, uv, sn):
    ok, r, t = cv2.solvePnP(P[ids], uv, *K[sn], flags=cv2.SOLVEPNP_SQPNP)
    r, t = cv2.solvePnPRefineLM(P[ids], uv, *K[sn], r, t)
    return tf(exp_so3(r.ravel()), t.ravel())


def proj(T, ids, sn):
    uv, _ = cv2.projectPoints(P[ids], log_so3(T[:3, :3]), T[:3, 3], *K[sn])
    return uv.reshape(-1, 2)


TF = [pnp(*v[F], F) for v in V]                                # board in the fixed camera
TM = [pnp(*v[M], M) for v in V]                                # board in the moved camera
rel = [tf_ @ tf_inv(tm) for tf_, tm in zip(TF, TM)]            # T_FM per view
med = np.median([r[:3, 3] for r in rel], 0)
T0 = rel[int(np.argmin([np.linalg.norm(r[:3, 3] - med) for r in rel]))]


def solve(idx):
    n = len(idx)

    def unpack(x):
        T_FM = T0 @ tf(exp_so3(x[:3]), x[3:6])
        return T_FM, [TF[v] @ tf(exp_so3(x[6 + 6 * k:9 + 6 * k]), x[9 + 6 * k:12 + 6 * k]) for k, v in enumerate(idx)]

    def res(x):
        T_FM, Tb = unpack(x)
        out = []
        for v, T in zip(idx, Tb):
            out.append((proj(T, V[v][F][0], F) - V[v][F][1]).ravel())
            out.append((proj(tf_inv(T_FM) @ T, V[v][M][0], M) - V[v][M][1]).ravel())
        return np.concatenate(out)
    sp = lil_matrix((sum(2 * (len(V[v][F][0]) + len(V[v][M][0])) for v in idx), 6 + 6 * n), dtype=int)
    r0 = 0
    for k, v in enumerate(idx):                               # a view's rows read its board pose; the moved camera's
        nf, nm = 2 * len(V[v][F][0]), 2 * len(V[v][M][0])     # rows read T_FM as well
        sp[r0:r0 + nf + nm, 6 + 6 * k:12 + 6 * k] = 1
        sp[r0 + nf:r0 + nf + nm, :6] = 1
        r0 += nf + nm
    sol = least_squares(res, np.zeros(6 + 6 * n), loss="soft_l1", f_scale=1.0, jac_sparsity=sp,
                        x_scale=np.tile([1e-3] * 3 + [0.1] * 3, n + 1))
    T_FM, Tb = unpack(sol.x)
    err = {sn: [] for sn in SNS}
    per_view = []
    for v, T in zip(idx, Tb):
        ef = proj(T, V[v][F][0], F) - V[v][F][1]
        em = proj(tf_inv(T_FM) @ T, V[v][M][0], M) - V[v][M][1]
        err[F].append(ef)
        err[M].append(em)
        per_view.append(float(np.sqrt((np.concatenate([ef, em]) ** 2).sum(1).mean())))
    rms = {sn: float(np.sqrt((np.concatenate(e) ** 2).sum(1).mean())) for sn, e in err.items()}
    return T_FM, rms, per_view


idx = list(range(len(V)))
T_FM, rms, per_view = solve(idx)
T_F = np.array(H["cams"][F]["T_CB"])                            # camera <- base
T_M_old = np.array(H["cams"][M]["T_CB"])
T_M = tf_inv(T_FM) @ T_F
c_new, c_old = tf_inv(T_M)[:3, 3], tf_inv(T_M_old)[:3, 3]
turn = np.degrees(np.linalg.norm(log_so3(T_M[:3, :3] @ T_M_old[:3, :3].T)))
rng = np.random.default_rng(0)
boot_c, boot_r = [], []
for _ in range(a.boot):
    Tb, _, _ = solve(sorted(rng.choice(idx, len(idx), replace=True).tolist()))
    Tm = tf_inv(Tb) @ T_F
    boot_c.append(tf_inv(Tm)[:3, 3])
    boot_r.append(log_so3(Tm[:3, :3] @ T_M[:3, :3].T))
sd_c = float(np.sqrt(np.trace(np.cov(np.array(boot_c).T))))
sd_r = float(np.degrees(np.sqrt(np.trace(np.cov(np.array(boot_r).T)))))
print(f"{len(V)} views; reprojection rms {F} (fixed) {rms[F]:.3f} px, {M} (moved) {rms[M]:.3f} px; "
      f"worst views {sorted(range(len(per_view)), key=lambda v: -per_view[v])[:3]} at "
      f"{sorted(per_view)[-1]:.2f} px")
print(f"{M}: centre moved {np.linalg.norm(c_new - c_old):.1f} mm ({np.round(c_new - c_old, 1).tolist()}), turned "
      f"{turn:.1f} deg; bootstrap sd {sd_c:.2f} mm, {sd_r:.3f} deg; cameras now {np.linalg.norm(T_FM[:3, 3]):.1f} mm apart")
out = json.loads(json.dumps(H))
out["cams"][M] = {"T_CB": T_M.tolist(), "rms_px": rms[M]}
out["moved"] = {"camera": M, "fixed": F, "views": len(V), "rms_px": rms, "per_view_rms_px": per_view,
                "T_CB_before": T_M_old.tolist(), "centre_moved_mm": (c_new - c_old).tolist(), "turned_deg": turn,
                "bootstrap_sd_mm": sd_c, "bootstrap_sd_deg": sd_r, "pairs": a.pairs, "board_scale": sc}

if a.before and a.after:
    hm = json.load(open(HEX))
    HEXMAP = TagMap.from_corners(hm["family"], hm["size_mm"], {int(i): c for i, c in hm["corners_mm"].items()})
    cams = {sn: tp.Camera(*K[sn], 1920, 1080) for sn in SNS}
    tb, ta = (json.load(open(Path(d) / "tags.json" if Path(d).is_dir() else d))["tags"] for d in (a.before, a.after))
    chk, dx, dy, resp = {}, np.nan, np.nan, np.nan
    if Path(a.before).is_dir() and (Path(a.before) / f"{F}_ref.png").exists():
        g0 = cv2.imread(str(Path(a.before) / f"{F}_ref.png"), 0).astype(np.float32)
        g1 = cv2.imread(str(Path(a.after) / f"{F}_ref.png"), 0).astype(np.float32)
        (dx, dy), resp = cv2.phaseCorrelate(g0, g1)
        chk = {"fixed_image_shift_px": [dx, dy], "phase_response": resp}

    def block(tags, sn, T_CB):
        obs = {int(i): np.array(c) for i, c in tags[sn].items() if int(i) in HEXMAP.corners}
        T, inf = tp.solve_body(obs, HEXMAP, cams[sn])
        return (None, []) if T is None else (tf_inv(T_CB) @ T, inf["tags"])
    b0, b1 = block(tb, F, T_F), block(ta, F, T_F)
    if b0[0] is not None and b1[0] is not None:
        chk["fixed_block_moved_mm"] = float(np.linalg.norm(b1[0][:3, 3] - b0[0][:3, 3]))
        chk["fixed_block_turned_deg"] = float(np.degrees(np.linalg.norm(log_so3(b1[0][:3, :3] @ b0[0][:3, :3].T))))
    bm = block(ta, M, T_M)
    if b1[0] is not None and bm[0] is not None:
        chk["block_cameras_apart_mm"] = float(np.linalg.norm(b1[0][:3, 3] - bm[0][:3, 3]))
        chk["block_tags"] = {F: b1[1], M: bm[1]}
    print(f"fixed camera {F}: image shift ({dx:+.2f}, {dy:+.2f}) px (response {resp:.2f}); block from it moved "
          f"{chk.get('fixed_block_moved_mm', float('nan')):.2f} mm, {chk.get('fixed_block_turned_deg', float('nan')):.3f} deg")
    obs = [{int(i): np.array(c) for i, c in ta[sn].items() if int(i) in HEXMAP.corners} for sn in SNS]
    if b1[0] is not None and all(obs):
        for name, Tm_ in (("new", T_M), ("before_move", T_M_old)):
            T_PC = [tf_inv(T_F) if sn == F else tf_inv(Tm_) for sn in SNS]
            _, inf = tp.solve_multi(obs, HEXMAP, [cams[sn] for sn in SNS], T_PC, b1[0])
            chk[f"block_joint_rms_px_{name}_pose"] = inf["rms_px"]
        obs0 = [{int(i): np.array(c) for i, c in tb[sn].items() if int(i) in HEXMAP.corners} for sn in SNS]
        if b0[0] is not None and all(obs0):
            T_PC = [tf_inv(T_F) if sn == F else tf_inv(T_M_old) for sn in SNS]
            _, inf = tp.solve_multi(obs0, HEXMAP, [cams[sn] for sn in SNS], T_PC, b0[0])
            chk["block_joint_rms_px_before_move"] = inf["rms_px"]
        print(f"hex block before the move, both cameras at once: {chk.get('block_joint_rms_px_before_move', np.nan):.3f} px")
        print(f"hex block after the move, both cameras at once: {chk['block_joint_rms_px_new_pose']:.3f} px with the new "
              f"pose, {chk['block_joint_rms_px_before_move_pose']:.3f} px with the pose before the move")
    print(f"hex block after the move: cameras {chk.get('block_cameras_apart_mm', float('nan')):.2f} mm apart "
          f"(tags {chk.get('block_tags')})")
    out["moved"]["checks"] = chk
Path(a.out).write_text(json.dumps(out, indent=1))
print(f"wrote {a.out}")
