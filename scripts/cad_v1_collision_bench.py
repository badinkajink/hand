#!/usr/bin/env python3
"""Step time of the screwdriver scene under four collision models of the finger (2026-10-13).

real_v1 capsules with 1 mm pads (reorient_backends.replace_tips on real_v1's scene); the CAD finger with pads and no
link spheres; with the balanced MorphIt spheres (spheres.json); with the tight spheres (spheres_tight.json). Every
scene starts at its 'open' key, where the fingers touch nothing and the tool lies on the floor, and holds the key's
commands: the time measures what the extra geometry costs the collision pipeline (broadphase and narrowphase tests),
not a grasp. 1000 steps of 2 ms on one thread, best of three.

    .venv/bin/python scripts/cad_v1_collision_bench.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import build_cad_v1_finger as B  # noqa: E402

OUT = B.OUT


def real_v1_with_pads(tmp: Path) -> str:
    import reorient_backends as RB
    root = ET.parse(REPO / "assets/mjcf/real_v1/scenes/scene_screwdriver_medium.xml").getroot()
    for key in list(root.find("keyframe")):
        if key.get("name") == "open_ik":
            root.find("keyframe").remove(key)
    RB.replace_tips(root, "tpu2.7", "pads", B.PAD_MU, tmp, tool=RB.OBJ)
    return ET.tostring(root, encoding="unicode")


def cad_scene(spheres: Path | None, tmp: Path) -> str:
    A = B.derive()
    raw = json.loads(spheres.read_text()) if spheres else {}
    coll = {b: [(s[:3], s[3]) for s in v] for b, v in raw.items()}
    bm = B.body_meshes(A)
    inert = {b: B.inertial([(p, m) for p, (bb, m) in bm.items() if bb == b]) for b in B.BODY_PARTS}
    out = tmp / "cad"
    (out / "meshes").mkdir(parents=True, exist_ok=True)
    for p, (_, m) in bm.items():
        m.export(out / "meshes" / f"{p}.stl")
    B.write_scene(A, coll, out, inert)
    return (out / "scene_screwdriver_medium.xml").read_text()


def bench(xml: str, label: str, n=1000, reps=3):
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    k = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_KEY, i) for i in range(m.nkey)].index("open")
    best, ncon_max, ncon_mean = 1e9, 0, 0
    for _ in range(reps):
        mujoco.mj_resetDataKeyframe(m, d, k)
        ncs = []
        t = time.perf_counter()
        for s in range(n):
            mujoco.mj_step(m, d)
            ncs.append(d.ncon)
        best = min(best, time.perf_counter() - t)
        ncon_max, ncon_mean = max(ncs), float(np.mean(ncs))
    ngeom_coll = int(sum(1 for i in range(m.ngeom) if m.geom_contype[i] or m.geom_conaffinity[i]))
    row = dict(model=label, ngeom=int(m.ngeom), ngeom_colliding=ngeom_coll, us_per_step=1e6 * best / n,
               steps_per_s=n / best, ncon_mean=ncon_mean, ncon_max=int(ncon_max))
    print(row, flush=True)
    return row


def main():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        rows = [bench(real_v1_with_pads(tmp / "rv1"), "real_v1 capsules and pads"),
                bench(cad_scene(None, tmp / "n"), "CAD finger, pads only"),
                bench(cad_scene(OUT / "spheres.json", tmp / "b"), "CAD finger, balanced spheres and pads"),
                bench(cad_scene(OUT / "spheres_tight.json", tmp / "s"), "CAD finger, tight spheres and pads")]
    (OUT / "collision_bench.json").write_text(json.dumps(dict(rows=rows, steps=1000, dt=0.002, threads=1,
                                                              mujoco=mujoco.__version__), indent=1))
    print("wrote", OUT / "collision_bench.json")


if __name__ == "__main__":
    main()
