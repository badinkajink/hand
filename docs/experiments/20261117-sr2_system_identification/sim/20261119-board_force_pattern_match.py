"""A location-aware linear force estimate against the trials' GP on the arm presses of 2026-11-19 (sim/
20261119-arm_press_boards.py): does the force the GP loses come from the boards or from the GP?

The boards' field is linear in the contact force at a fixed place on the pad (the calibration's presses: the field per
newton does not depend on the indenter, sim/20261119-board_patch_size_calibration.py). So the field of a press at place
p is d = f G(p), G the calibration's field per newton there (the tip's near-normal presses of 1-4 N: the UR5 sphere
session and, for tip 3, the inverted rig's rod presses, which reach the pad's proximal 4 mm). The estimate takes the
calibration point whose G is most nearly parallel to the measured d (the best cosine) and the least-squares force
f = G.d / |G|^2 there; the cosine says how well any calibrated point explains the field (the calibration's own repeats
reach 0.99). Applied to each press's field per newton g (the least-squares line of d against the HEX-E's force), the
estimate is that press's slope against the HEX-E, beside the GP's slope from the same press.

    /usr/bin/python3 sim/20261119-board_force_pattern_match.py [--arm data/s3/20261119-arm_press_boards.json]
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
       3: [CORR / "20261030-spherev3id3", RIG / "20261104-145215-id3pts11"]}


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


def estimate(g, P, G, top=5):
    """(force per newton, best cosine, place): the least-squares force at the best-matching calibration points (the
    median over the `top` best)."""
    cos = G @ g / (np.linalg.norm(G, axis=1) * np.linalg.norm(g) + 1e-12)
    idx = np.argsort(-cos)[:top]
    f = (G[idx] @ g) / np.sum(G[idx] ** 2, axis=1)
    return float(np.median(f)), float(cos[idx[0]]), np.median(P[idx], 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default=str(STUDY / "data/s3/20261119-arm_press_boards.json"))
    ap.add_argument("--out", default=str(STUDY / "data/s3/20261119-board_force_pattern_match.json"))
    a = ap.parse_args()
    cal = {t: cal_points(t) for t in CAL}
    arm = json.loads(Path(a.arm).read_text())
    rows = []
    for run in arm["runs"]:
        for pr in run["presses"]:
            if "g" not in pr:
                continue
            P, G = cal[pr["tip"]]
            f, c, p = estimate(np.array(pr["g"], float), P, G)
            gp = pr["estimators"].get("gp", {}).get("slope")
            gpo = pr["estimators"].get("gp_own_pose", {}).get("slope")
            rows.append(dict(run=run["run"], rep=pr["rep"], finger=pr["finger"], indenter=pr["indenter"],
                             field_per_N_uT=pr["field_per_N_uT"], pattern_slope=round(f, 3), best_cos=round(c, 4),
                             place=np.round(p, 2).tolist(), gp_slope=gp, gp_own_pose_slope=gpo))
            print(f"{run['run']} rep {pr['rep']} {pr['finger']:6s} {pr['indenter']:28s}: |g| {pr['field_per_N_uT']:5.1f} "
                  f"uT/N; pattern-matched force per HEX-E newton {f:.2f} (best cos {c:.3f} at z {p[2]:.1f} mm); GP "
                  f"{gp} (with own-pose {gpo})", flush=True)
    Path(a.out).write_text(json.dumps({"what": __doc__.split("\n\n")[0].replace("\n", " "), "rows": rows}, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
