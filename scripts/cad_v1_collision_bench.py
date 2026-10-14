#!/usr/bin/env python3
"""Step time of the screwdriver scene under four collision models of the finger (2026-10-13).

real_v1 capsules with 1 mm pads (reorient_backends.replace_tips on real_v1's scene); the CAD finger with pads and no
link spheres; with the balanced MorphIt spheres (spheres.json); with the tight spheres (spheres_tight.json). Two
conditions, each 1000 steps of 2 ms on one thread, best of three:

  open   the scene's 'open' key held: the fingers touch nothing and the tool lies on the floor, so the time measures
         what the extra geometry costs the broadphase and narrowphase tests;
  close  from 'open', every MCP and PIP command ramps +0.9 rad over 1 s (clamped to its range) and holds for 1 s, so
         the fingers squeeze the tool; an untimed repeat records the share of the finger-to-tool normal force that
         goes through the <f>_tip bodies (the pads).

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


def real_v1_with_pads(tmp: Path) -> Path:
    import reorient_backends as RB
    root = ET.parse(REPO / "assets/mjcf/real_v1/scenes/scene_screwdriver_medium.xml").getroot()
    for key in list(root.find("keyframe")):
        if key.get("name") == "open_ik":
            root.find("keyframe").remove(key)
    RB.replace_tips(root, "tpu2.7", "pads", B.PAD_MU, tmp, tool=RB.OBJ)
    (tmp / "scene.xml").write_text(ET.tostring(root, encoding="unicode"))
    return tmp / "scene.xml"


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
    return out / "scene_screwdriver_medium.xml"


FINGERS = ("thumb", "index", "middle")
TOOL = "screwdriver_medium"


def run(m, d, k, condition, n, forces=False):
    """One run of n steps; returns contact counts, and with forces the finger-to-tool normal force by tip / not tip."""
    mujoco.mj_resetDataKeyframe(m, d, k)
    c0 = d.ctrl.copy()
    acts = [m.actuator(f"a_{f}_{j}").id for f in FINGERS for j in ("mcp", "pip")]
    lo, hi = m.actuator_ctrlrange[acts, 0], m.actuator_ctrlrange[acts, 1]
    tool = m.body(TOOL).id
    ncs, f_tip, f_other, fc, first = [], 0.0, 0.0, np.zeros(6), {}
    for s in range(n):
        if condition == "close":
            d.ctrl[acts] = np.clip(c0[acts] + 0.9 * min(1.0, s / 500), lo, hi)
        mujoco.mj_step(m, d)
        ncs.append(d.ncon)
        if forces:
            for i in range(d.ncon):
                c = d.contact[i]
                b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
                if tool not in (b1, b2):
                    continue
                other = m.body(b2 if b1 == tool else b1).name
                if not other.startswith(FINGERS):
                    continue
                mujoco.mj_contactForce(m, d, i, fc)
                kind = "tip" if other.endswith("_tip") else other.split("_", 1)[1]
                first.setdefault(kind, round((s + 1) * m.opt.timestep, 3))
                if other.endswith("_tip"):
                    f_tip += fc[0]
                else:
                    f_other += fc[0]
    return (ncs, f_tip, f_other, first) if forces else (ncs, f_tip, f_other)


def bench(path: Path, label: str, n=1000, reps=3):
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    k = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_KEY, i) for i in range(m.nkey)].index("open")
    ngeom_coll = int(sum(1 for i in range(m.ngeom) if m.geom_contype[i] or m.geom_conaffinity[i]))
    row = dict(model=label, ngeom=int(m.ngeom), ngeom_colliding=ngeom_coll)
    for cond in ("open", "close"):
        best = 1e9
        for _ in range(reps):
            t = time.perf_counter()
            ncs, _, _ = run(m, d, k, cond, n)
            best = min(best, time.perf_counter() - t)
        _, f_tip, f_other, first = run(m, d, k, cond, n, forces=True)
        row[cond] = dict(us_per_step=1e6 * best / n, steps_per_s=n / best, ncon_mean=float(np.mean(ncs)), ncon_max=int(max(ncs)),
                         tip_force_share=float(f_tip / (f_tip + f_other)) if f_tip + f_other > 0 else None,
                         first_tool_contact_s=first)  # per finger body kind (tip, pip_frame, ...): time of its first touch
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
