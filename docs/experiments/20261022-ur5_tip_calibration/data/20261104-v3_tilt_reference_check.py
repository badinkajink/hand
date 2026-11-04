"""1. The GP's held-out error by load tilt (|f_t| / |f_n|), a fifth of the point groups held out, 3 seeds, holds of 1 N
and over: location (mm), direction (deg) and the tangential force's error (|f_t predicted - f_t measured|, N).
2. The reference presses repeated through each session, at the level nearest 2 N: the spread of their contact label p
(mm, about the mean), of the board reading (|d - mean| over |mean|, %) and of the measured force's direction (deg)."""
import sys
import numpy as np
sys.path.insert(0, "/home/humanoid/Programs/hand/scripts")
import ur5_tip_calibration_v3_data_curve as V  # noqa: E402
H_ = "docs/experiments/20261030-inverted_rig_seat_block/data/holds/"
K4 = ["press", "slide", "roll", "twist"]
K6 = ["press", "sweep", "offset", "slide", "roll", "twist"]
TIPS = {1: (V.TIPS[1], K4), 2: (V.TIPS[2], K4), 3: (V.TIPS[3], K4),
        4: ((H_ + "20261102-220337-s3v4", H_ + "20261103-202017-rod32_dense"), K6),
        5: ((H_ + "20261102-182710-s2v5", H_ + "20261103-144808-s2sph2", H_ + "20261103-172647-rod21_dense"), K6),
        6: ((H_ + "20261102-145010-s1v6b", H_ + "20261103-122020-s1sph2b", H_ + "20261103-214753-rod13_dense"), K6)}
BINS = ((0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 9))
floor = V.PR.floors(0)[:15]
ang = lambda a, b: np.degrees(np.arccos(np.clip(np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12), -1, 1)))  # noqa: E731
print("1. held-out error by tilt: holds, location mm, direction deg, tangential force error N")
for tip, (ss, kinds) in TIPS.items():
    V.KINDS[:] = kinds
    H = V.load(ss)
    Y = np.c_[H["p"], H["f"], H["tn"]]
    fm = np.linalg.norm(H["f"], axis=1)
    n_ = H["n"]
    fn = np.sum(H["f"] * n_, 1)
    tilt = np.sqrt(np.maximum(fm ** 2 - fn ** 2, 0)) / (np.abs(fn) + 1e-9)
    R = {b: [[], [], []] for b in BINS}
    for seed in range(3):
        rng = np.random.default_rng(400 + seed)
        G = rng.permutation(np.unique(H["group"]))
        test = np.isin(H["group"], G[: len(G) // 5])
        m = V.BG.fit(H["d_board"][~test], Y[~test], floor)
        P = V.BG.predict(m, H["d_board"][test])
        p, n, f = H["p"][test], n_[test], H["f"][test]
        dp = P[:, :3] - p
        loc = np.linalg.norm(dp - np.sum(dp * n, 1)[:, None] * n, axis=1)
        dirn = ang(P[:, 3:6], f)
        ft_m = f - np.sum(f * n, 1)[:, None] * n
        fp = P[:, 3:6]
        ft_p = fp - np.sum(fp * n, 1)[:, None] * n
        eft = np.linalg.norm(ft_p - ft_m, axis=1)
        hi = fm[test] >= 1.0
        for b in BINS:
            k = hi & (tilt[test] >= b[0]) & (tilt[test] < b[1])
            for j, v in enumerate((loc, dirn, eft)):
                R[b][j].append(v[k])
    parts = []
    for b in BINS:
        L, D, E = (np.concatenate(x) for x in R[b])
        if len(L) >= 20:
            parts.append(f"{b[0]:g}-{b[1]:g}: {len(L)} at {np.median(L):.2f} mm, {np.median(D):.1f} deg, {np.median(E):.2f} N")
        else:
            parts.append(f"{b[0]:g}-{b[1]:g}: {len(L)}")
    print(f"  ID {tip}: " + "; ".join(parts), flush=True)
print("2. reference presses near 2 N: session, presses, label spread mm, board spread %, force direction spread deg")
for tip, (ss, kinds) in TIPS.items():
    for s in ss:
        z = np.load(V.ROOT / s / "holds.npz", allow_pickle=True)
        ids = z["id"].astype(str)
        isref = np.array([("ref" in i) and not i.endswith("-n") and "hard" not in i and "-p" not in i for i in ids])
        if isref.sum() < 3:
            continue
        lv = z["level_N"].astype(float)
        L0 = min(np.unique(lv[isref]), key=lambda x: abs(x - 2.0))
        k = isref & (np.abs(lv - L0) < 1e-6) & np.isfinite(z["d_board"]).all(1)
        if k.sum() < 3:
            continue
        p, d, f = z["p"][k], z["d_board"][k].astype(float), z["f"][k]
        lab = np.median(np.linalg.norm(p - p.mean(0), axis=1))
        brd = np.median(np.linalg.norm(d - d.mean(0), axis=1)) / np.linalg.norm(d.mean(0)) * 100
        fd = np.median(ang(f, np.tile(f.mean(0), (len(f), 1))))
        print(f"  ID {tip} {s.split('/')[-1]}: {k.sum()} at {L0:g} N; label {lab:.3f} mm, board {brd:.1f} %, direction {fd:.2f} deg",
              flush=True)
