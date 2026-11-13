"""Two-camera fit of the hex tag block's map from block_views.py's views, and its checks.

The cameras stay at today's poses (handeye_fit.py, TCP route, fitted lens) and lenses (lens_board_fit.py), and their
339 mm baseline sets the scale. sr2hand.tracking.pose.bundle_adjust_multi frees each tag's pose and size on the block
and each view's block pose; tag 10 holds the gauge, and the result is fitted rigidly onto HexBlock(29.985, 13.0)'s map,
the frame in which the objects' placements are defined. Corners have soft-L1 weights with a 0.5 px knee in A; B is
plain least squares, since scipy's robust loss would also soften the lengths.

Fits: A, the baseline alone (the map); B, A plus the caliper across-flats (29.985 mm, sd 0.03) between opposite tags'
centres, to show how far the two scales disagree. A is fitted once on every fifth view left out (A_train, for the
held-out checks) and once on all views (the map written out).

Checks, each with the nominal map (13.0 and 13.33 mm tags) and the fitted one:
  held-out views  each camera's own solve (pose.solve_body) of the block, both put in the base frame: the distance
                  between the two block origins, and the joint solve's (pose.solve_multi) reprojection RMS;
  arm poses       the 17 poses of the arm calibration, the block untouched on the table: the same distance between the
                  cameras' mean block origins (block_depth_check.py's check, 2.7 mm with 13.0 mm tags).

VIEWS is block_views.py's folder or its views.jsonl (data/20261113-block_views.jsonl, the corners without the frames);
ARM_DIR the arm calibration's raw folder with the pNN frames, or - to skip that check.

    /usr/bin/python3 block_map_fit.py VIEWS LENS.json HANDEYE.json ARM_DIR OUT.json [MAP_OUT.json]
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))
from sr2hand.geometry.rotations import log_so3, tf_inv  # noqa: E402
from sr2hand.tracking import pose as tp  # noqa: E402
from sr2hand.tracking.tags import HexBlock  # noqa: E402

views_dir, lens_f, he_f, arm_dir, out_f = sys.argv[1:6]
map_f = sys.argv[6] if len(sys.argv) > 6 else None
LENS = json.load(open(lens_f))["cams"]
HE = json.load(open(he_f))["cams"]
SNS = sorted(HE)
AF = 29.985
NOM = HexBlock(across_flats=AF, tag_size=13.0).tag_map()
N133 = HexBlock(across_flats=AF, tag_size=13.33).tag_map()
OPP = [(10, 13), (11, 14), (12, 15)]


def camera(sn):
    l = LENS[sn]["lens"]
    K = np.array([[l["fx"], 0, l["cx"]], [0, l["fy"], l["cy"]], [0, 0, 1.0]])
    return tp.Camera(K, np.array([l["k1"], l["k2"], l["p1"], l["p2"], l["k3"]], float), 1920, 1080)


CAMS = [camera(sn) for sn in SNS]
T_PC = [tf_inv(np.array(HE[sn]["T_CB"])) for sn in SNS]


def solve_each(obs, M):
    """Each camera's own block pose in P (None where it sees no tag of M) and the tags it used."""
    out = []
    for k in range(len(SNS)):
        T, info = tp.solve_body(obs[k], M, CAMS[k])
        out.append((None, []) if T is None else (T_PC[k] @ T, info["tags"]))
    return out


def start_pose(obs):
    each = [e for e in solve_each(obs, NOM) if e[0] is not None]
    if not each:
        return None
    T0 = max(each, key=lambda e: len(e[1]))[0]
    T, _ = tp.solve_multi(obs, NOM, CAMS, T_PC, T0)
    return T


vf = Path(views_dir) / "views.jsonl" if Path(views_dir).is_dir() else Path(views_dir)
rows = [json.loads(l) for l in open(vf)]
views, T0s = [], []
for r in rows:
    obs = [{int(i): np.array(c, float) for i, c in r["cams"][sn]["corners"].items()} for sn in SNS]
    T0 = start_pose(obs)
    if T0 is not None:
        views.append(obs)
        T0s.append(T0)
test = [v for v in range(len(views)) if v % 5 == 4]
train = [v for v in range(len(views)) if v % 5 != 4]
print(f"{len(views)} views ({len(train)} to fit, {len(test)} held out); tags per camera per view: "
      + ", ".join(f"{sn} {np.mean([len(v[k]) for v in views]):.1f}" for k, sn in enumerate(SNS)))
seen = {i: [sum(i in v[k] for v in views) for k in range(len(SNS))] for i in NOM.ids()}
print("views seeing each tag (" + ", ".join(SNS) + "): " + ", ".join(f"{i}: {a}/{b}" for i, (a, b) in seen.items()))


def fit(idx, lengths=()):
    return tp.bundle_adjust_multi([views[v] for v in idx], NOM, CAMS, T_PC, [T0s[v] for v in idx], fixed=10,
                                  lengths=lengths, loss="linear" if lengths else "soft_l1")


def describe(M, info, name):
    cen = {i: M.frames[i][0] for i in M.ids()}
    d = {"rms_px": info["rms_px"], "rms_px_per_cam": dict(zip(SNS, info["rms_px_per_cam"])),
         "sizes_mm": {str(i): round(s, 4) for i, s in info["sizes_mm"].items()},
         "opposite_mm": {f"{i}-{j}": round(float(np.linalg.norm(cen[i] - cen[j])), 4) for i, j in OPP},
         "vs_nominal_mm": dict(zip(("rms", "max"), np.round(tp.map_error(M, NOM), 4).tolist())), "tags": {},
         "rms_px_per_view": [None if e is None else round(e, 3) for e in info["rms_px_per_view"]]}
    for i in M.ids():
        c0, r0, d0, n0 = NOM.frames[i]
        c, r, dd, n = M.frames[i]
        R0, R = np.c_[r0, d0, n0], np.c_[r, dd, n]
        d["tags"][str(i)] = {"along_normal_mm": round(float((c - c0) @ n0), 3),
                             "in_face_mm": round(float(np.linalg.norm((c - c0) - ((c - c0) @ n0) * n0)), 3),
                             "turn_deg": round(float(np.degrees(np.linalg.norm(log_so3(R0.T @ R)))), 3)}
    print(f"{name}: {info['views']} views, {info['corners']} corners, rms {info['rms_px']:.3f} px "
          f"({', '.join(f'{x:.3f}' for x in info['rms_px_per_cam'])}); sizes "
          + " ".join(f"{i}:{s:.3f}" for i, s in info["sizes_mm"].items())
          + "; opposite centres " + " ".join(f"{k}:{v:.3f}" for k, v in d["opposite_mm"].items())
          + f" (calipers {AF}); vs nominal map rms {d['vs_nominal_mm']['rms']:.3f} max {d['vs_nominal_mm']['max']:.3f} mm")
    print("   per tag (along normal mm, in face mm, turn deg): "
          + " ".join(f"{i}:({t['along_normal_mm']:+.2f},{t['in_face_mm']:.2f},{t['turn_deg']:.2f})" for i, t in d["tags"].items()))
    return d


M_tr, I_tr = fit(train)
D_tr = describe(M_tr, I_tr, "A_train")
M_all, I_all = fit(range(len(views)))
D_all = describe(M_all, I_all, "A (all views)")
M_len, I_len = fit(range(len(views)), [(i, j, AF, 0.03) for i, j in OPP])
D_len = describe(M_len, I_len, "B (with across-flats)")
D_len["map_A_vs_B_mm"] = np.round(tp.map_error(M_all, M_len), 4).tolist()
D_all["map_A_train_vs_A_mm"] = np.round(tp.map_error(M_tr, M_all), 4).tolist()
print(f"   map A_train vs map A: rms {D_all['map_A_train_vs_A_mm'][0]:.3f}, max {D_all['map_A_train_vs_A_mm'][1]:.3f} mm")
print(f"   map A vs map B: rms {D_len['map_A_vs_B_mm'][0]:.3f}, max {D_len['map_A_vs_B_mm'][1]:.3f} mm")
bad = [v for v, e in enumerate(I_all["rms_px_per_view"]) if e is not None and e > 0.6]
print(f"   views over 0.6 px in A: {bad}")


def held_out(M):
    d, rms, both2 = [], [], []
    for v in test:
        each = solve_each(views[v], M)
        if all(e[0] is not None for e in each):
            dist = float(np.linalg.norm(each[0][0][:3, 3] - each[1][0][:3, 3]))
            d.append(dist)
            print(f"   held-out view {v}: tags {[e[1] for e in each]}, cameras apart {dist:.2f} mm") if M is M_tr else None
            if all(len(e[1]) >= 2 for e in each):
                both2.append(dist)
        T, inf = tp.solve_multi(views[v], M, CAMS, T_PC, T0s[v])
        if T is not None:
            rms.append(inf["rms_px"])
    q = lambda a: dict(n=len(a), median=round(float(np.median(a)), 3), max=round(float(np.max(a)), 3)) if a else None  # noqa: E731
    return {"cameras_apart_mm": q(d), "cameras_apart_mm_2tags_each": q(both2), "joint_rms_px": q(rms)}


det = tp.detector()
arm = []
for p in sorted(Path(arm_dir).glob("p[0-9][0-9]")) if arm_dir != "-" else []:
    arm.append([tp.detect(det, cv2.imread(str(p / f"{sn}_color.png"), 0), ids=NOM.ids()) for sn in SNS])


def arm_check(M):
    if not arm:
        return {"cameras_apart_mm": None}
    cen = [[], []]
    for obs in arm:
        for k, (T, _) in enumerate(solve_each(obs, M)):
            if T is not None:
                cen[k].append(T[:3, 3])
    c0, c1 = np.array(cen[0]), np.array(cen[1])
    return {"cameras_apart_mm": round(float(np.linalg.norm(c0.mean(0) - c1.mean(0))), 3),
            "sd_mm": {SNS[0]: np.round(c0.std(0), 3).tolist(), SNS[1]: np.round(c1.std(0), 3).tolist()},
            "n": [len(c0), len(c1)]}


checks = {}
for name, M_ho, M_arm in (("nominal 13.0", NOM, NOM), ("nominal 13.33", N133, N133), ("fitted A", M_tr, M_all),
                          ("fitted B", M_len, M_len)):
    checks[name] = {"held_out": held_out(M_ho), "arm_poses": arm_check(M_arm)}
    h, a = checks[name]["held_out"], checks[name]["arm_poses"]
    print(f"{name:14s} held out: cameras apart {h['cameras_apart_mm']} (2+ tags each {h['cameras_apart_mm_2tags_each']}), "
          f"joint rms {h['joint_rms_px']};  arm poses: cameras apart {a['cameras_apart_mm']} mm")
print("   (held-out checks of 'fitted A' use A_train; 'fitted B' saw the held-out views)")
out = {"views": len(views), "train": train, "test": test, "seen_per_cam": {str(i): s for i, s in seen.items()},
       "fits": {"A_train": D_tr, "A": D_all, "B_across_flats": D_len}, "checks": checks,
       "map": M_all.to_json(), "map_B": M_len.to_json(), "lens": lens_f, "handeye": he_f}
Path(out_f).write_text(json.dumps(out, indent=1))
if map_f:
    Path(map_f).write_text(json.dumps(M_all.to_json(), indent=1))
print(f"wrote {out_f}" + (f" and {map_f}" if map_f else ""))
