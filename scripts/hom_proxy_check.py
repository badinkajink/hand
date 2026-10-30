#!/usr/bin/env python3
"""Finger-to-finger contact on the GPU through a few proxy spheres per fingertip (2026-10-30). The GPU evaluator lets the
pads and the rigid spheres under 2 mm meet the object only; letting the small rigid spheres collide as on the CPU made a
2,016-world batch 13 times slower (rows/ff_timing). After the touch-pose fix (fp_sr2_toy.ff_clearance_v4b) the index and
the middle still touched during 14 of the 40 best twist rollouts on the CPU plant, and 5 of those lost the sphere
(rows/cpu_best/top_twist_d40_ff.jsonl). Here each finger's distal parts (DIP link, cassette, cap and pad spheres, in the
PIP frame) are covered by K bounding spheres, k-means clusters each enlarged to contain its members. They collide with
other fingers' proxies only (contype and conaffinity 4) and carry no mass.

    .venv/bin/python scripts/hom_proxy_check.py timing     # the 2,016-world twist batch of rows/ff_timing with proxies
    .venv/bin/python scripts/hom_proxy_check.py top        # the 40 best twist layouts of grid2dff, seed 0, with proxies
Rows: docs/experiments/20261029-foundational_pose_review/rows/ff_timing/t_proxy_42*.jsonl and rows/cpu_best/
top_twist_d40_ff_gpu_proxy.jsonl.
"""
from __future__ import annotations

import os

os.environ.setdefault("SCALE_HAND", "cad_v2")

import json  # noqa: E402
import sys  # noqa: E402
import xml.etree.ElementTree as ET  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import hom_gpu as HG  # noqa: E402

T, K = HG.T, HG.K
K_PROXY = 12


def proxies(f, k=K_PROXY, seed=0):
    """Finger f's proxy spheres in its PIP frame: (k, 3) centres and (k,) radii containing every distal sphere."""
    import fp_sr2_toy as FP
    from scipy.cluster.vq import kmeans2
    S = FP.spheres(f)
    c = np.vstack([S["pip_frame"][0], S["tip"][0] + np.array([0.0, 0.0, -K.P])])
    r = np.concatenate([S["pip_frame"][1], S["tip"][1]])
    cen, lab = kmeans2(c, k, seed=seed, minit="++")
    rad = np.array([np.max(np.linalg.norm(c[lab == j] - cen[j], axis=1) + r[lab == j]) if np.any(lab == j) else 0.0
                    for j in range(k)])
    keep = rad > 0
    return cen[keep], rad[keep]


def proxy_model(path):
    """hom_gpu.mocap_model with the proxies added to each finger's PIP-frame body."""
    import mujoco
    root = ET.parse(path).getroot()
    for name in ("palm_pose", "post"):
        root.find(f".//worldbody/body[@name='{name}']").set("mocap", "true")
    for f in HG.FINGERS:
        body = root.find(f".//body[@name='{f}_pip_frame']")
        for j, (c, r) in enumerate(zip(*proxies(f))):
            ET.SubElement(body, "geom", name=f"{f}_proxy{j}", type="sphere", size=f"{r:.6f}",
                          pos=" ".join(f"{x:.6f}" for x in c), contype="4", conaffinity="4", group="4", density="0")
    tmp = Path(path).with_name(f"{Path(path).stem}.proxy.{os.getpid()}.xml")
    ET.ElementTree(root).write(tmp)
    try:
        return mujoco.MjModel.from_xml_path(str(tmp))
    finally:
        tmp.unlink()


def child(spec):
    """hom_gpu.batch_child on the model with proxies (a spawned process imports this module afresh)."""
    HG.mocap_model = proxy_model
    HG.batch_child(spec)


def worlds_for(task, d, keys, seeds, var):
    """GPU worlds of the given (call, grasp) pairs of grid2dff at diameter d, as hom_gpu.cmd_run builds them."""
    scene = HG.Scene(T.SHAPE[task], d)
    want = set(keys)
    out = []
    for key, mxy, setup in HG.layout_list(task, "grid2dff_size", 0, d):
        if (int(key["call"]), int(key.get("grasp", 0))) in want:
            bm, q_open = scene.prepare(mxy, setup)
            for sd in seeds:
                out.append({"key": key, "seed": sd, "setup": setup, "body_mounts": bm, "q_open": q_open,
                            "var": HG.variant_world(var, sd)})
    return out


def run(worlds, task, d, path, tpath, layouts, batch=2048):
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    for b0 in range(0, len(worlds), batch):
        spec = {"shape": T.SHAPE[task], "d_mm": float(d), "task": task, "worlds": worlds[b0:b0 + batch],
                "dt": worlds[0]["var"]["dt"], "gen_seed": b0, "batch": b0 // batch, "path": str(path),
                "tpath": str(tpath), "layouts": layouts}
        pr = ctx.Process(target=child, args=(spec,))
        pr.start()
        pr.join()
        if pr.exitcode != 0:
            raise RuntimeError(f"batch {b0 // batch} failed (exit {pr.exitcode})")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "top"
    if cmd == "timing":
        lays = HG.layout_list("twist", "grid2d_size", 42, 40.0)
        scene = HG.Scene("sphere", 40.0)
        worlds = []
        for key, mxy, setup in lays:
            bm, q_open = scene.prepare(mxy, setup)
            for sd in (0, 1, 2):
                worlds.append({"key": key, "seed": sd, "setup": setup, "body_mounts": bm, "q_open": q_open,
                               "var": HG.variant_world("base", sd)})
        out = HG.ROWS / "ff_timing"
        out.mkdir(parents=True, exist_ok=True)
        for p in (out / "t_proxy_42.jsonl", out / "t_proxy_42_timing.jsonl"):
            p.unlink(missing_ok=True)
        run(worlds, "twist", 40.0, out / "t_proxy_42.jsonl", out / "t_proxy_42_timing.jsonl", "grid2d_size")
        return 0
    top = [json.loads(x) for x in (HG.ROWS / "cpu_best" / "top_twist_d40_ff.jsonl").read_text().splitlines() if x.strip()]
    keys = sorted({(r["call"], r["grasp"]) for r in top})
    path = HG.ROWS / "cpu_best" / "top_twist_d40_ff_gpu_proxy.jsonl"
    for p in (path, path.with_name(path.stem + "_timing.jsonl")):
        p.unlink(missing_ok=True)
    run(worlds_for("twist", 40.0, keys, (0,), "base"), "twist", 40.0, path, path.with_name(path.stem + "_timing.jsonl"),
        "grid2dff_size")
    return 0


if __name__ == "__main__":
    sys.exit(main())
