"""v3 ID 3 on the inverted rig (seat 1, 11-04) against its tip mode (10-30): tip or rig.
  within   the GP fitted and tested on the inverted session alone, a fifth of its point groups held out (4 seeds):
           location, direction, magnitude on holds of 1 N and over, and on near-normal loads (|f_t| / |f_n| < 0.1)
  tipmode  the same test on ID 3's tip-mode presses alone (kind press, near-normal), for the like-for-like number
  transfer the GP fitted on all of ID 3's tip-mode data predicting the inverted session's holds: location error, its mean
           offset in the pad frame (a registration difference between the rigs shows as an offset) and direction
  refs     the reference presses at 2 N: board spread (|d - mean| over |mean|), uT/N in time order
Usage: id3swap_gp.py HOLDS_DIR"""
import sys
import numpy as np
sys.path.insert(0, "/home/humanoid/Programs/hand/scripts")
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path
import ur5_tip_calibration_v3_data_curve as V  # noqa: E402
floor = V.PR.floors(0)[:15]
ang = lambda a, b: np.degrees(np.arccos(np.clip(np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12), -1, 1)))  # noqa: E731


def tilt(H):
    fm = np.linalg.norm(H["f"], axis=1)
    fn = np.abs(np.sum(H["f"] * H["n"], 1))
    return fm, np.sqrt(np.maximum(fm ** 2 - fn ** 2, 0)) / (fn + 1e-9)


def within(H, name, keep=None):
    Y = np.c_[H["p"], H["f"], H["tn"]]
    fm, tl = tilt(H)
    keep = np.ones(len(Y), bool) if keep is None else keep
    out = {k: [] for k in ("loc", "loc_n", "dir", "dir_n", "mag")}
    G0 = np.unique(H["group"][keep])
    for seed in range(4):
        rng = np.random.default_rng(500 + seed)
        G = rng.permutation(G0)
        test = np.isin(H["group"], G[: len(G) // 5]) & keep
        train = ~np.isin(H["group"], G[: len(G) // 5]) & keep
        m = V.BG.fit(H["d_board"][train], Y[train], floor)
        P = V.BG.predict(m, H["d_board"][test])
        dp = P[:, :3] - H["p"][test]
        n = H["n"][test]
        loc = np.linalg.norm(dp - np.sum(dp * n, 1)[:, None] * n, axis=1)
        d = ang(P[:, 3:6], H["f"][test])
        mg = np.abs(np.linalg.norm(P[:, 3:6], axis=1) - fm[test]) / fm[test]
        hi, nn = fm[test] >= 1.0, tl[test] < 0.1
        out["loc"].append(np.median(loc[hi])); out["loc_n"].append(np.median(loc[hi & nn]))
        out["dir"].append(np.median(d[hi])); out["dir_n"].append(np.median(d[hi & nn])); out["mag"].append(np.median(mg[hi]))
    print(f"within {name}: {keep.sum()} holds, {len(G0)} groups; location {np.median(out['loc']):.3f} mm (near-normal "
          f"{np.median(out['loc_n']):.3f}), direction {np.median(out['dir']):.2f} deg (near-normal {np.median(out['dir_n']):.2f}), "
          f"magnitude {np.median(out['mag']) * 100:.1f} %; location per seed {np.round(out['loc_n'], 3).tolist()}", flush=True)


V.KINDS[:] = ["press", "slide", "roll", "twist"]
T = V.load(V.TIPS[3])                             # ID 3, tip mode
V.KINDS[:] = ["press", "twist", "sweep", "offset", "slide", "roll"]
I = V.load([sys.argv[1]])                         # ID 3, inverted rig
within(I, "inverted")
fmT, tlT = tilt(T)
within(T, "tip mode, presses", keep=(T["kind"].astype(str) == "press"))
within(T, "tip mode, all")
Y = np.c_[T["p"], T["f"], T["tn"]]
m = V.BG.fit(T["d_board"], Y, floor)
P = V.BG.predict(m, I["d_board"])
fm, tl = tilt(I)
dp = P[:, :3] - I["p"]
loc = np.linalg.norm(dp - np.sum(dp * I["n"], 1)[:, None] * I["n"], axis=1)
hi = fm >= 1.0
print(f"transfer tip mode -> inverted: {hi.sum()} holds; location {np.median(loc[hi]):.3f} mm (near-normal "
      f"{np.median(loc[hi & (tl < 0.1)]):.3f}); mean offset in P {np.round(dp[hi].mean(0), 3).tolist()} mm; direction "
      f"{np.median(ang(P[hi, 3:6], I['f'][hi])):.2f} deg; magnitude "
      f"{np.median(np.abs(np.linalg.norm(P[hi, 3:6], axis=1) - fm[hi]) / fm[hi]) * 100:.1f} %", flush=True)
z = np.load(sys.argv[1] + "/holds.npz", allow_pickle=True)
ids, lv = z["id"].astype(str), z["level_N"].astype(float)
k = np.array(["ref" in i for i in ids]) & (np.abs(lv - 2.0) < 1e-6)
o = np.argsort(z["t"][k].astype(float))
d = z["d_board"][k].astype(float)[o]
f = np.linalg.norm(z["f"][k], axis=1)[o]
spread = np.median(np.linalg.norm(d - d.mean(0), axis=1)) / np.linalg.norm(d.mean(0)) * 100
print(f"refs at 2 N: {k.sum()}, board spread {spread:.1f} %; uT/N in order {np.round(np.linalg.norm(d, axis=1) / f, 1).tolist()}")
