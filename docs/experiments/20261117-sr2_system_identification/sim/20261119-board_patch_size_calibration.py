"""Does the fingertip boards' field per newton depend on the contact's size? The UR5 calibrations of the v3 tips
(10-28 to 11-04) pressed each pad with indenters of three sizes at the same places: spheres of R 2 and R 5 mm and a rod
of R 12.5 mm (the owner, 2026-11-19 16:3x: "is there any way to estimate the size of the contact patch and thus get
better performance. can we try to learn that from our existing calibrations?").

For each tip and each indenter other than the R 5 sphere, the near-normal presses (|f_t| < 0.15 |f_n|, 0.5 N and over)
are matched to the R 5 sphere's near-normal presses within --match mm of the same contact point (the pad frame of the
holds tables: x out of the pad, y across, z along from the distal end) and within 30 % of the same normal force. Per
matched hold:
  field ratio  |dB| / f_n of the hold over the median of its matches' (15 board components, uT per N)
  depth ratio  the indentation depth per newton (holds' "depth", mm) over the matches'
  cosine       the cosine between the hold's dB and its matches' mean dB, against the same for the sphere's own
               holds (each against its other matches): how far the pattern differs beyond repeat noise
and the board GP (scripts/ur5/ur5_tip_calibration_board_gp.py) fitted on the R 5 sphere's holds of that tip (all kinds)
predicts the force of the other indenter's near-normal presses: gain = predicted |f| / measured |f| (and the same for a
fifth of the sphere's own groups held out).

Hertz at the apex (pad radii 6.44 mm across, 14.2 distal / 56.9 proximal along; tip_codesign_geometry tcd_ell7.4x8.5x11x22)
puts the patch area of the R 2 sphere at about 0.6 of the R 5 sphere's, the rod's at 1.8-2.2 and a flat plate's at 2.5-3.1:
if the boards sensed the patch's indentation, the rod would read well under the sphere per newton, and the R 2 sphere over.

    /usr/bin/python3 sim/20261119-board_patch_size_calibration.py [--match 0.75] [--out data/s3/20261119-board_patch_size_calibration.json]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "scripts/ur5")]
import _paths  # noqa: E402,F401
import ur5_tip_calibration_board_gp as BG  # noqa: E402
import ur5_tip_calibration_predict as PR  # noqa: E402

RIG = "docs/experiments/20261030-inverted_rig_seat_block/data/holds/"
TIPS = {
    "2 (middle)": {"sphere R5": ["logs/20261029-spherev3id2"], "rod R12.5": ["logs/20261029-rodv3id2"],
                   "tall R5": ["logs/20261029-tallv3id2"]},
    "4": {"sphere R5": [RIG + "20261102-220337-s3v4"], "rod R12.5": [RIG + "20261103-202017-rod32_dense"]},
    "5": {"sphere R5": [RIG + "20261102-182710-s2v5"], "sphere R2": [RIG + "20261103-144808-s2sph2"],
          "rod R12.5": [RIG + "20261103-172647-rod21_dense"]},
    "6": {"sphere R5": [RIG + "20261102-145010-s1v6b"], "sphere R2": [RIG + "20261103-122020-s1sph2b"],
          "rod R12.5": [RIG + "20261103-214753-rod13_dense"]},
}
EXTRA = {"1 (thumb)": {"sphere R5": ["logs/20261028-spherev3id1_twists"], "rod R12.5": ["logs/20261028-rodv3id1"]},
         "3 (index)": {"sphere R5": ["logs/20261030-spherev3id3"], "rod R12.5": ["logs/20261030-rodv3id3"]}}
KINDS_ALL = ("press", "slide", "roll", "twist", "sweep", "offset")


def load(sessions):
    H = {}
    for s in sessions:
        z = np.load(ROOT / s / "holds.npz", allow_pickle=True)
        ok = (np.isin(z["kind"].astype(str), KINDS_ALL) & np.isfinite(z["d_board"]).all(1) & np.isfinite(z["p"]).all(1)
              & np.isfinite(z["f"]).all(1) & np.isfinite(z["n"]).all(1))
        h = {k: z[k][ok] for k in ("d_board", "p", "n", "f", "kind", "depth")}
        for k, v in h.items():
            H[k] = np.concatenate([H[k], v]) if k in H else v
    f, n = H["f"], H["n"]
    fn = np.sum(f * n, 1)
    ft = np.linalg.norm(f - fn[:, None] * n, axis=1)
    H["fn"] = np.abs(fn)
    H["normal"] = (ft < 0.15 * np.abs(fn)) & (np.abs(fn) >= 0.5) & np.isin(H["kind"].astype(str), ("press", "sweep"))
    return H


def med(x):
    return float(np.median(x)) if len(x) else float("nan")


def q(x, p):
    return float(np.percentile(x, p)) if len(x) else float("nan")


def compare(S, O, match):
    """Matched ratios of O's near-normal holds against S's (the R 5 sphere)."""
    si = np.flatnonzero(S["normal"])
    oi = np.flatnonzero(O["normal"])
    Sp, Sf, SB = S["p"][si], S["fn"][si], S["d_board"][si]
    sb_n = np.linalg.norm(SB, axis=1) / Sf
    sd_n = S["depth"][si] / Sf
    fr, dr, cos_o, cos_s, zs = [], [], [], [], []
    for i in oi:
        d = np.linalg.norm(Sp - O["p"][i], axis=1)
        m = (d < match) & (np.abs(Sf / O["fn"][i] - 1.0) < 0.3)
        if m.sum() < 3:
            continue
        fr.append(np.linalg.norm(O["d_board"][i]) / O["fn"][i] / np.median(sb_n[m]))
        if np.isfinite(O["depth"][i]) and np.isfinite(sd_n[m]).sum() >= 3:
            dr.append(O["depth"][i] / O["fn"][i] / np.nanmedian(sd_n[m]))
        mb = SB[m].mean(0)
        cos_o.append(O["d_board"][i] @ mb / (np.linalg.norm(O["d_board"][i]) * np.linalg.norm(mb) + 1e-12))
        zs.append(O["p"][i][2])
    for j in range(len(si)):                                   # the sphere against its own other matches
        d = np.linalg.norm(Sp - Sp[j], axis=1)
        m = (d < match) & (np.abs(Sf / Sf[j] - 1.0) < 0.3)
        m[j] = False
        if m.sum() < 3:
            continue
        mb = SB[m].mean(0)
        cos_s.append(SB[j] @ mb / (np.linalg.norm(SB[j]) * np.linalg.norm(mb) + 1e-12))
    return dict(n=len(fr), field_ratio=[q(fr, 25), med(fr), q(fr, 75)], depth_ratio=[q(dr, 25), med(dr), q(dr, 75)],
                cos_other=[q(cos_o, 25), med(cos_o), q(cos_o, 75)], cos_sphere_self=[q(cos_s, 25), med(cos_s),
                                                                                       q(cos_s, 75)],
                z_mm=[q(zs, 10), q(zs, 90)])


def gp_gain(S, others, floor, seed=0):
    rng = np.random.default_rng(seed)
    Y = np.c_[S["p"], S["f"]]
    # a fifth of the sphere's near-normal holds held out, by contact point (0.75 mm cells)
    cell = np.floor(S["p"] / 0.75).astype(int)
    keys = np.unique(cell, axis=0)
    out_keys = keys[rng.random(len(keys)) < 0.2]
    held = (cell[:, None, :] == out_keys[None, :, :]).all(2).any(1)
    m = BG.fit(S["d_board"][~held], Y[~held], floor)
    res = {}
    t = held & S["normal"] & (S["fn"] >= 1.0)
    P = BG.predict(m, S["d_board"][t])
    g = np.linalg.norm(P[:, 3:6], axis=1) / np.linalg.norm(S["f"][t], axis=1)
    res["sphere R5 (held-out points)"] = dict(n=int(t.sum()), gain=[q(g, 25), med(g), q(g, 75)])
    for name, O in others.items():
        t = O["normal"] & (O["fn"] >= 1.0)
        P = BG.predict(m, O["d_board"][t])
        g = np.linalg.norm(P[:, 3:6], axis=1) / np.linalg.norm(O["f"][t], axis=1)
        dp = np.linalg.norm(P[:, :3] - O["p"][t], axis=1)
        res[name] = dict(n=int(t.sum()), gain=[q(g, 25), med(g), q(g, 75)], loc_err_mm=med(dp))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--match", type=float, default=0.75)
    ap.add_argument("--gp", action="store_true", help="also the GP cross-indenter gains (slower)")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data/s3/20261119-board_patch_size_calibration.json"))
    a = ap.parse_args()
    floor = PR.floors(0)[:15]
    tips = dict(TIPS)
    for k, v in EXTRA.items():
        if all((ROOT / s / "holds.npz").exists() for ss in v.values() for s in ss):
            tips[k] = v
    out = {}
    for tip, sess in tips.items():
        D = {name: load(ss) for name, ss in sess.items()}
        S = D["sphere R5"]
        out[tip] = {"n_normal": {k: int(v["normal"].sum()) for k, v in D.items()}}
        for name, O in D.items():
            if name == "sphere R5":
                continue
            r = compare(S, O, a.match)
            out[tip][name] = r
            print(f"tip {tip}: {name} vs sphere R5 at the same point and force: {r['n']} holds; field per N "
                  f"{r['field_ratio'][1]:.2f} ({r['field_ratio'][0]:.2f}-{r['field_ratio'][2]:.2f}); depth per N "
                  f"{r['depth_ratio'][1]:.2f}; dB cosine to the sphere's {r['cos_other'][1]:.3f} (sphere to itself "
                  f"{r['cos_sphere_self'][1]:.3f}); z {r['z_mm'][0]:.1f}-{r['z_mm'][1]:.1f} mm", flush=True)
        if a.gp:
            g = gp_gain(S, {k: v for k, v in D.items() if k != "sphere R5"}, floor)
            out[tip]["gp_trained_on_sphere_R5"] = g
            print(f"tip {tip}: GP on the sphere R5: " + "; ".join(f"{k} gain {v['gain'][1]:.2f} ({v['gain'][0]:.2f}-"
                                                                    f"{v['gain'][2]:.2f}, n {v['n']})" for k, v in g.items()),
                  flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
