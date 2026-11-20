"""Does a spread contact's field equal the sum of the calibration's point fields over its patch? (2026-11-19; the owner,
16:3x: "is there any way to estimate the size of the contact patch and thus get better performance. can we try to learn
that from our existing calibrations?")

The calibration's sphere presses give each tip's field per newton for a point load at each place on the pad, G(p) (15
components, uT/N): the R 5 sphere's near-normal presses of 1-4 N (20261105-viewer_inputs/correction), smoothed over the
pad's surface by a Gaussian of --sigma mm in (y, z). The boards' field is linear in the pad's small deformation, so a
contact spread over a patch should read d = sum_e G(p_e) f_e over the patch's elements. G changes sign within a few mm
(chip 1's y goes from 0 at the apex to -19 uT/N 6 mm proximal on tip 3), so a long patch cancels much of it.

The patch: the v4b pad's surface (tcd_ell7.4x8.5x11x22: semi-axes 7.4 across, 8.5 deep, 11 distal and 22 proximal of
the apex at z 10.55 mm from the distal end face) on a 0.25 mm grid, a flat plate whose inward normal in the pad frame is
(-1, -t_y, -t_z) pushed --delta mm into it (or a sphere of radius R), each element's force proportional to its overlap
(a Winkler bed, as the simulator's pads). Over a grid of tilts, the prediction's cosine with the measured field per
newton g (sim/20261119-arm_press_boards.py) and its magnitude; the best tilt's patch (its extent along z and y).

    /usr/bin/python3 sim/20261119-board_patch_superposition.py [--arm data/s3/20261119-arm_press_boards.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[2]
CORR = ROOT / "docs/experiments/20261022-ur5_tip_calibration/data/20261105-viewer_inputs/correction"
RIG = ROOT / "docs/experiments/20261030-inverted_rig_seat_block/data/holds"
CAL = {1: [CORR / "20261028-spherev3id1_twists"], 2: [CORR / "20261029-spherev3id2"],
       3: [CORR / "20261030-spherev3id3", RIG / "20261104-145215-id3pts11"]}   # tip 3's rig rod presses reach z < 4 mm
AY, AX, AZD, AZP, ZA = 7.4, 8.5, 11.0, 22.0, 10.55       # the pad's semi-axes (mm) and the apex's z from the end face


def cal_points(tip):
    P, G = [], []
    for s in CAL[tip]:
        z = np.load(s / "holds.npz", allow_pickle=True)
        ok = np.isfinite(z["d_board"]).all(1) & np.isfinite(z["p"]).all(1) & np.isfinite(z["f"]).all(1)
        f, n, p, B, k = z["f"][ok], z["n"][ok], z["p"][ok], z["d_board"][ok], z["kind"][ok].astype(str)
        fn = np.abs(np.sum(f * n, 1))
        ft = np.linalg.norm(f - np.sum(f * n, 1)[:, None] * n, axis=1)
        m = (ft < 0.15 * fn) & (fn >= 1.0) & (fn <= 4.0) & (k == "press")
        P.append(p[m])
        G.append(B[m] / fn[m][:, None])
    return np.concatenate(P), np.concatenate(G)


class Field:
    """G(y, z): the calibration's field per newton, Gaussian-weighted over its press points (support where the weights
    sum over --min-w)."""

    def __init__(self, P, G, sigma):
        self.P, self.G, self.s = P, G, sigma

    def __call__(self, Y, Z):
        D2 = (Y[:, None] - self.P[None, :, 1]) ** 2 + (Z[:, None] - self.P[None, :, 2]) ** 2
        W = np.exp(-0.5 * D2 / self.s ** 2)
        ws = W.sum(1)
        return (W @ self.G) / np.maximum(ws, 1e-9)[:, None], ws


def surface(step=0.25):
    y = np.arange(-AY, AY + 1e-9, step)
    z = np.arange(0.0, 21.41 + 1e-9, step)
    Y, Z = np.meshgrid(y, z)
    az = np.where(Z < ZA, AZD, AZP)
    r = 1.0 - (Y / AY) ** 2 - ((Z - ZA) / az) ** 2
    m = r > 0
    X = AX * np.sqrt(np.where(m, r, 0.0))
    return X[m], Y[m], Z[m]


def plate_patch(X, Y, Z, ty, tz, delta):
    u = np.array([1.0, ty, tz]) / np.linalg.norm([1.0, ty, tz])
    h = X * u[0] + Y * u[1] + Z * u[2]
    ov = h - (h.max() - delta)
    return np.maximum(ov, 0.0)


def sphere_patch(X, Y, Z, R, cy, cz, delta):
    """A sphere of radius R whose centre lies on the pad's outward normal line through (apex x, cy, cz)... approximated
    as the sphere's surface over the pad: overlap where the pad's surface pokes into the sphere moved delta past touch."""
    # the sphere's centre at (x0 + R, cy, cz) with x0 chosen so it first touches, then moved delta toward the pad
    d0 = np.sqrt((Y - cy) ** 2 + (Z - cz) ** 2)
    inside = d0 < R
    xs = np.where(inside, -np.sqrt(np.maximum(R * R - d0 ** 2, 0.0)), np.inf)   # the sphere's lower surface, centre at x=0
    gap = xs - X                   # the sphere's surface minus the pad's along x, centre at x = 0: touches at min gap
    g0 = gap[inside].min()
    ov = (g0 + delta) - gap
    return np.where(inside, np.maximum(ov, 0.0), 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default=str(STUDY / "data/s3/20261119-arm_press_boards.json"))
    ap.add_argument("--sigma", type=float, default=0.8)
    ap.add_argument("--out", default=str(STUDY / "data/s3/20261119-board_patch_superposition.json"))
    a = ap.parse_args()
    X, Y, Z = surface()
    fields = {t: Field(*cal_points(t), a.sigma) for t in CAL}
    arm = json.loads(Path(a.arm).read_text())
    res = []
    for run in arm["runs"]:
        for pr in run["presses"]:
            if "g" not in pr:
                continue
            g = np.array(pr["g"], float)
            Gf = fields[pr["tip"]]
            Ge, ws = Gf(Y, Z)
            sup = ws > 0.5
            best = None
            sphere = "sphere" in pr["indenter"]
            grid = ([(ty, tz, dl) for ty in np.linspace(-0.12, 0.12, 9) for tz in np.linspace(-0.35, 0.35, 29)
                     for dl in (0.15, 0.3, 0.5, 0.8)] if not sphere else
                    [(cy, cz, dl) for cy in np.linspace(-3, 3, 13) for cz in np.linspace(4, 18, 29) for dl in (0.15, 0.3, 0.5)])
            for p1, p2, dl in grid:
                ov = sphere_patch(X, Y, Z, 20.75, p1, p2, dl) if sphere else plate_patch(X, Y, Z, p1, p2, dl)
                if ov.sum() <= 0:
                    continue
                w = ov * sup
                frac = w.sum() / ov.sum()                         # share of the patch where G is known
                if frac < 0.9:
                    continue
                dp = (w[:, None] * Ge).sum(0) / ov.sum()           # field per newton of the patch
                c = float(dp @ g / (np.linalg.norm(dp) * np.linalg.norm(g) + 1e-12))
                if best is None or c > best["cos"]:
                    on = ov > 0
                    best = dict(cos=round(c, 4), p1=round(float(p1), 3), p2=round(float(p2), 3), delta=dl,
                                pred_uT_per_N=round(float(np.linalg.norm(dp)), 2),
                                meas_uT_per_N=pr["field_per_N_uT"],
                                ratio_meas_over_pred=round(pr["field_per_N_uT"] / float(np.linalg.norm(dp)), 3),
                                patch_z=[round(float(Z[on].min()), 2), round(float(Z[on].max()), 2)],
                                patch_y=[round(float(Y[on].min()), 2), round(float(Y[on].max()), 2)],
                                centroid=[round(float((ov * Y).sum() / ov.sum()), 2), round(float((ov * Z).sum() / ov.sum()), 2)],
                                point_cal_at_centroid=None)
            if best is not None:
                Gc, _ = Gf(np.array([best["centroid"][0]]), np.array([best["centroid"][1]]))
                best["point_cal_at_centroid_uT_per_N"] = round(float(np.linalg.norm(Gc[0])), 2)
                best["cos_point_at_centroid"] = round(float(Gc[0] @ g / (np.linalg.norm(Gc[0]) * np.linalg.norm(g))), 4)
            rec = dict(run=run["run"], rep=pr["rep"], finger=pr["finger"], indenter=pr["indenter"], best=best,
                       point_best_cos=pr["best_cos"])
            res.append(rec)
            print(f"{run['run']} rep {pr['rep']} {pr['finger']} ({pr['indenter']}): point-load best cos {pr['best_cos']}; "
                  f"patch best cos {best['cos']} at {'cy, cz' if sphere else 't_y, t_z'} {best['p1']}, {best['p2']}, delta "
                  f"{best['delta']} mm: patch z {best['patch_z']} y {best['patch_y']}, predicted {best['pred_uT_per_N']} "
                  f"uT/N, measured {best['meas_uT_per_N']} (ratio {best['ratio_meas_over_pred']}); a point at the patch's "
                  f"centroid {best['centroid']} would read {best['point_cal_at_centroid_uT_per_N']} uT/N (cos "
                  f"{best['cos_point_at_centroid']})", flush=True)
    Path(a.out).write_text(json.dumps({"what": __doc__.split("\n\n")[0].replace("\n", " "), "sigma_mm": a.sigma,
                                       "presses": res}, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
