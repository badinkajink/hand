"""Sphere against rod on near-normal loads (|f_t| / |f_n| < 0.1, 1 N and over), per tip, with each tip's sessions pooled
(IDs 4 to 6 with the dense steps): random holds held out (their point stays in training) and a fifth of the point
groups held out (new points), each over 3 seeds."""
import sys
import numpy as np
sys.path.insert(0, "/home/humanoid/Programs/hand/scripts")
import ur5_tip_calibration_v3_data_curve as V  # noqa: E402
H_ = "docs/experiments/20261030-inverted_rig_seat_block/data/holds/"
K4 = ["press", "slide", "roll", "twist"]
K6 = ["press", "sweep", "offset", "slide", "roll", "twist"]
TIPS = {1: (V.TIPS[1], K4, [0]), 2: (V.TIPS[2], K4, [0]), 3: (V.TIPS[3], K4, [0]),
        4: ((H_ + "20261102-220337-s3v4", H_ + "20261103-202017-rod32_dense"), K6, [0]),
        5: ((H_ + "20261102-182710-s2v5", H_ + "20261103-144808-s2sph2", H_ + "20261103-172647-rod21_dense"), K6, [0, 1]),
        6: ((H_ + "20261102-145010-s1v6b", H_ + "20261103-122020-s1sph2b", H_ + "20261103-214753-rod13_dense"), K6, [0, 1])}
floor = V.PR.floors(0)[:15]
for tip, (ss, kinds, sph) in TIPS.items():
    V.KINDS[:] = kinds
    H = V.load(ss)
    src = np.array([int(s.split(":", 1)[0]) for s in H["id"].astype(str)])
    sphere = np.isin(src, sph)
    Y = np.c_[H["p"], H["f"], H["tn"]]
    fm = np.linalg.norm(H["f"], axis=1)
    fn = np.abs(np.sum(H["f"] * H["n"], 1))
    normal = (np.sqrt(np.maximum(fm ** 2 - fn ** 2, 0)) / (fn + 1e-9) < 0.1) & (fm >= 1.0)
    res = {}
    for mode in ("random", "groups"):
        L = {"sphere": [], "rod": []}
        for seed in range(3):
            rng = np.random.default_rng(300 + seed)
            if mode == "random":
                test = rng.random(len(Y)) < 0.2
            else:
                G = rng.permutation(np.unique(H["group"]))
                test = np.isin(H["group"], G[: len(G) // 5])
            m = V.BG.fit(H["d_board"][~test], Y[~test], floor)
            P = V.BG.predict(m, H["d_board"][test])
            dp = P[:, :3] - H["p"][test]
            n = H["n"][test]
            loc = np.linalg.norm(dp - np.sum(dp * n, 1)[:, None] * n, axis=1)
            for k, msk in (("sphere", sphere[test]), ("rod", ~sphere[test])):
                L[k].append(loc[msk & normal[test]])
        res[mode] = {k: (len(np.concatenate(v)), float(np.median(np.concatenate(v)))) for k, v in L.items()}
    print(f"ID {tip}: near-normal holds sphere {int((sphere & normal).sum())}, rod {int((~sphere & normal).sum())}; "
          f"random hold-out sphere {res['random']['sphere'][1]:.3f} rod {res['random']['rod'][1]:.3f} mm; "
          f"new points sphere {res['groups']['sphere'][1]:.3f} rod {res['groups']['rod'][1]:.3f} mm", flush=True)
