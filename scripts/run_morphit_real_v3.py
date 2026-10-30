"""Fit MorphIt sphere sets to the real_v3 finger links.

Input:  assets/mjcf/real_v3/morphit/links/<link>.stl   (watertight, body frame, metres) --
        written by `uv run scripts/build_real_v3_from_step.py --export-links`.
Output: assets/mjcf/real_v3/morphit/spheres.json        consumed by
        `uv run scripts/build_real_v3_from_step.py --finger-geom morphit`.

Only the reference finger (index) is fitted: all three fingers are the same CAD parts, and the
builder carries these spheres onto thumb/middle rigidly, so the fingers stay identical.

MorphIt (HIRO-group/MorphIt-1, MIT) lives in external/MorphIt with its own Python 3.10 venv
(torch, Cython triangle_hash). Run it with that interpreter:

    external/MorphIt/.venv/bin/python scripts/run_morphit_real_v3.py
    external/MorphIt/.venv/bin/python scripts/run_morphit_real_v3.py --variant MorphIt-B --spheres mount=24,dist=12

Variants (paper): MorphIt-S fits the SURFACE (contact-rich manipulation, our case), MorphIt-V
covers the VOLUME (conservative collision avoidance), MorphIt-B balances the two.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
MORPHIT_SRC = ROOT / "external" / "MorphIt" / "src"
LINK_DIR = ROOT / "assets" / "mjcf" / "real_v3" / "morphit" / "links"
OUT = ROOT / "assets" / "mjcf" / "real_v3" / "morphit" / "spheres.json"
DEFAULT_SPHERES = {"mount": 32, "prox": 32, "mid": 40, "dist": 32}   # MorphIt-S, ~1 mm mean surface error


def evaluate(mesh: trimesh.Trimesh, C: np.ndarray, r: np.ndarray, n: int = 40000, seed: int = 0) -> dict:
    """Coverage = share of the link's solid inside some sphere; excess = sphere volume outside
    the link, relative to the link volume; surface error = mean |dist| from the link surface
    to the sphere-union surface."""
    rng = np.random.default_rng(seed)
    lo, hi = mesh.bounds
    pad = r.max()
    box = rng.uniform(lo - pad, hi + pad, size=(n, 3))
    in_mesh = np.concatenate([mesh.contains(b) for b in np.array_split(box, 20)])
    in_sph = np.concatenate([((np.linalg.norm(b[:, None, :] - C[None], axis=2) - r[None]) <= 0).any(1)
                             for b in np.array_split(box, 20)])
    vol_box = np.prod(hi - lo + 2 * pad)
    surf = mesh.sample(8000)
    ds = (np.linalg.norm(surf[:, None, :] - C[None], axis=2) - r[None]).min(1)
    return dict(
        coverage=float((in_mesh & in_sph).sum() / max(in_mesh.sum(), 1)),
        excess=float((in_sph & ~in_mesh).sum() * vol_box / n / mesh.volume),
        surface_err_mm=float(np.abs(ds).mean() * 1000),
        surface_err_p95_mm=float(np.quantile(np.abs(ds), 0.95) * 1000),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="MorphIt-S", choices=("MorphIt-S", "MorphIt-B", "MorphIt-V"))
    ap.add_argument("--spheres", default="", help="per-link overrides, e.g. mount=24,dist=12")
    ap.add_argument("--iterations", type=int, default=400)
    ap.add_argument("--links", default=",".join(DEFAULT_SPHERES), help="comma list of links to fit")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    counts = dict(DEFAULT_SPHERES)
    for tok in filter(None, args.spheres.split(",")):
        k, v = tok.split("=")
        counts[k] = int(v)

    sys.path.insert(0, str(MORPHIT_SRC))
    os.chdir(MORPHIT_SRC)          # MorphIt resolves some resources relative to src/
    import torch
    from config import get_config, update_config_from_dict
    from morphit import MorphIt
    from training import train_morphit

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    result = json.loads(OUT.read_text()) if OUT.exists() else {"links": {}}
    for link in args.links.split(","):
        mesh_path = LINK_DIR / f"{link}.stl"
        cfg = update_config_from_dict(get_config(args.variant), {
            "model.num_spheres": counts[link],
            "model.max_spheres": counts[link],
            "model.mesh_path": str(mesh_path),
            "model.device": "cpu",
            "training.iterations": args.iterations,
            "training.verbose_frequency": 10 ** 9,
            "visualization.enabled": False,
            "visualization.save_video": False,
            "random_seed": args.seed,
        })
        t0 = time.time()
        model = MorphIt(cfg)
        train_morphit(model)
        C = model.centers.detach().cpu().numpy().astype(float)
        r = model.radii.detach().cpu().numpy().astype(float)
        keep = r > 1e-4
        C, r = C[keep], r[keep]
        mesh = trimesh.load(mesh_path, force="mesh")
        metrics = evaluate(mesh, C, r)
        result["links"][link] = dict(centers=C.tolist(), radii=r.tolist(), variant=args.variant,
                                     iterations=args.iterations, seed=args.seed, n=len(r),
                                     seconds=round(time.time() - t0, 1), **metrics)
        print(f"{link:6s} n={len(r):3d}  coverage={metrics['coverage']:.3f}  excess={metrics['excess']:.3f}  "
              f"surface err mean={metrics['surface_err_mm']:.2f} mm p95={metrics['surface_err_p95_mm']:.2f} mm  "
              f"({time.time() - t0:.0f}s)", flush=True)
        OUT.write_text(json.dumps(result, indent=1))
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
