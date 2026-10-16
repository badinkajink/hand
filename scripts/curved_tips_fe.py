#!/usr/bin/env python3
"""Sensing of the quarter-ellipsoid tips (docs/handoff/20261015-curved_tips.md, ledger step 13): the magnet's travel per
newton from finite elements of the whole printed tip, and the five-chip board's force noise with the board on the tip's
flat back.

Finite elements: scripts/tpu_tip_magnet.py (homogenized gyroid core, walls and skins, the magnet bonded to its pocket)
with tpu_tip_design's geometry hooks replaced by the shape's insert-frame functions, as scripts/tip_codesign_fe.py does,
and the fixed back face moved to the shape's back plane (x = 8.5 - ax in the insert frame): the board is rigid, so every
node on that plane is fixed, with the mount face. The magnet (4.76 x 1.59 mm) lies under the contact point with its top
face `depth` under the palmar apex, at most 5 mm and at least 1 mm of TPU above the board.

Force noise: the magnet's translation and rotation per newton of normal and of shear load at the contact point (the FE
row) move a point dipole (scripts/tpu_tip_magnet_sensing.py, remanence 1.45 T); the field change at the board's five
MLX90393 (scripts/mlx90393_model.py: the plus of chips 7 mm apart, sensing spots 0.42 mm behind the back plane, U1 under
the contact point) at the readout page's setting (GAIN_SEL 7, RES 0, a 141 Hz burst) gives the Cramer-Rao bound of the
normal force from one sample, with the contact point known, and the same with Earth's field estimated alongside.

    ~/.claude/bin/resguard.sh run --mem 5G --cpu 400 -- taskset -c 16-19 ~/miniconda3/envs/femtip/bin/python \\
        scripts/curved_tips_fe.py --tips tcd_qell7.4x8.5x11x22_u0.1
Rows: docs/experiments/20261015-curved_tips/fe/fe.jsonl (tpu_tip_magnet's row with the tip, back plane and noise added).
"""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import tip_codesign_geometry as TG  # noqa: E402

OUT = ROOT / "docs/experiments/20261015-curved_tips/fe/fe.jsonl"
MAG = ("d4.76x1.59", 4.76, 1.59)


def depth_of(sh, t=1.59):
    """Magnet top depth (mm): 5 mm where the pulp allows, else 1 mm of TPU between the magnet and the back plane."""
    ax = (TG.X_FACE - sh.xc) * 1e3 if getattr(sh, "family", "") == "qell" else 17.0
    return round(min(5.0, ax - t - 1.0), 2)


def install(sh):
    """Point tpu_tip_design and tpu_tip_fem_tip at the shape: geometry hooks, and the back plane as the fixed face."""
    import tpu_tip_design as TD
    import tpu_tip_fem_tip as T
    TD.inside_body, TD.surface_x = sh.fe_inside, sh.fe_surface_x
    if getattr(sh, "family", "") == "qell":
        import tpu_tip_magnet as M
        xb = round((sh.xc - (TG.REACH - 0.0085)) * 1e3, 6)
        TD.X_BACK = xb
        T.X_BACK = xb
        # the grid's x zones start at the back plane, so its nodes are a node plane and are fixed (until 2026-10-15
        # 19:00 the default zones had no node at x = 0, which left the 8.5 mm quarter-ellipsoids' backs free)
        fine = [(7.0, 8.5, 0.2)]
        M.GRID = dict(M.GRID, x=([(xb, 1.0, 0.5), (1.0, 7.0, 0.25)] if xb < 1.0 else [(xb, 7.0, 0.25)]) + fine)
    return TD.X_BACK


def noise(row, depth, gap=0.42):
    """Force noise (N) of one 141 Hz sample from the five chips: normal force with the shear forces estimated alongside,
    the contact point known; and the same with a uniform background field (Earth's) estimated too."""
    import tpu_tip_magnet_sensing as TS
    pf = row["per_force"]["1"] if isinstance(row["per_force"], dict) else json.loads(row["per_force"].replace("'", '"'))["1"]
    mot = {"normal": [(pf["magnet_t_um"], pf.get("magnet_w_mrad", [0.0, 0.0, 0.0]))],
           "shear_y": [(row["shear_y_magnet_t_um_per_N"], row["shear_y_magnet_w_mrad_per_N"])],
           "shear_z": [(row["shear_z_magnet_t_um_per_N"], row["shear_z_magnet_w_mrad_per_N"])]}
    r = {"magnets": [(MAG[1], MAG[2], 0.0, TS.TD.Z_C)], "depth_mm": depth}
    ch = TS.chips(r, "board", gap)
    sig = np.tile(TS.NOISE, len(ch))
    S = {k: TS.signal(r, v, ch) for k, v in mot.items()}
    J = np.stack([S["normal"], S["shear_y"], S["shear_z"]], 1)
    U = np.tile(np.eye(3), (len(ch), 1))
    b0 = np.abs(TS.B0(r, ch)).max() * 1e3
    return {"noise_N_known": float(TS.crb(J, sig)[0]), "noise_N_known_earth": float(TS.crb(np.concatenate([J, U], 1), sig)[0]),
            "noise_N_U1": float(TS.crb(J[:3], sig[:3])[0]), "signal_normal_uT_per_N": float(np.linalg.norm(S["normal"])),
            "B0_max_mT": float(b0), "saturated": bool(b0 > TS.B_RANGE_MT), "board_gap_mm": gap}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tips", nargs="+", required=True)
    ap.add_argument("--recipes", nargs="+", default=["w95_20"])
    ap.add_argument("--forces", nargs="+", type=float, default=[0.5, 1.0, 3.0])
    ap.add_argument("--count", action="store_true", help="only the TPU and magnet voxels of the model grid (no solve)")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    have = set()
    if a.out.exists():
        have = {(r["tip"], r["recipe"]) for r in map(json.loads, open(a.out))}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    for key in a.tips:
        sh = TG.shape(key)
        xb = install(sh)
        depth = depth_of(sh)
        import tpu_tip_magnet as M
        if not getattr(M, "_captures", False):                 # keep the built model to count the fixed back nodes
            _build = M.build

            def build_capture(recipe, magnet, depth_):
                model, cc = _build(recipe, magnet, depth_)
                M.LAST_MODEL = model
                return model, cc
            M.build, M._captures = build_capture, True
        if a.count:
            import tpu_tip_design as TD
            occ, _ = TD.occupancy(M.H_AUX, MAG[0], depth)
            print(f"{key}: back plane x {xb:.2f} mm, magnet depth {depth} mm, voxels TPU {int((occ == 1).sum())} "
                  f"magnet {int((occ == 2).sum())} (h {M.H_AUX} mm)", flush=True)
            continue
        for rec in a.recipes:
            if (key, rec) in have:
                continue
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td) / "row.jsonl"
                M.run(rec, M.RECIPES[rec], MAG[0], depth, tuple(a.forces), out=tmp)
                row = json.loads(tmp.read_text().splitlines()[-1])
            row.update(tip=key, back_plane_x_mm=xb, study="20261015-curved_tips", script_wrapper="scripts/curved_tips_fe.py",
                       n_fixed_back_nodes=int(((np.abs(M.LAST_MODEL.X[:, 0] - xb) < 1e-6) & M.LAST_MODEL.fixed_nodes).sum())
                       if getattr(M, "LAST_MODEL", None) is not None else None)
            row.update(noise(row, depth))
            with open(a.out, "a") as fh:
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            pf = row["per_force"]["1"]
            print(f"{key} {rec}: travel {abs(pf['magnet_t_um'][0]):.1f} um/N, noise {row['noise_N_known']:.3f} N "
                  f"(with Earth {row['noise_N_known_earth']:.3f}), {row['n_dof']} dof", flush=True)


if __name__ == "__main__":
    main()
