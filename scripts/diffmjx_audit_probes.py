#!/usr/bin/env python3
"""CPU checks behind the 2026-10-11 audit of the 2026-10-07 DiffMJX and contact-gait studies.

Each subcommand writes one JSON file into docs/experiments/20261011-diffmjx_audit/:

  interfinger  replays the stored 400 ms plans in the packed-pad scene, whose fingertip blocks collide only with
               the tool, and records the smallest distance between fingertip blocks and from each block to the
               other fingers' links (the convex-mesh comparison keeps those contacts)
  fdstep       records the step SciPy's L-BFGS-B uses for its two-point derivatives in the 400 ms search, and the
               central-difference gradient of the search score at five steps, at the hold plan and at the CEM
               plan that the hybrid refines
  cfd_reach    counts, at the held state, the pad-to-tool pairs within the contact-from-distance width that the
               10-07 probes used (cfd_solimp[2] = 0.05 m), next to the pairs in contact
  frames       compares the tool's angular velocity and constraint torque about its body x axis (what the 10-07
               scripts read from the free joint) with the same quantities about world x, 20 ms after the held state
  corrections  collects the corrected 400 ms searches (searches/*.json, real_v1_contact_gait_search.py with
               --fd-step 1e-2 and five CEM seeds) and their eight-jitter replay (transfer-fd1e-2-mj360.json) next to the
               2026-10-07 runs, and times the CPU finite differences of the recontact derivative

    uv run --no-sync python scripts/diffmjx_audit_probes.py interfinger fdstep cfd_reach frames
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import real_v1_d1_continuation as d1  # noqa: E402

OUT = ROOT / "docs/experiments/20261011-diffmjx_audit"
GAIT = ROOT / "docs/experiments/20261007-contact-gait"
CFD_WIDTH_M = 0.05     # cfd_solimp[2] in real_v1_diffmjx_official_gate.py, _pad_jvp.py and _recontact_jvp.py


def write(name: str, payload: dict) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as fh:
        fh.write(json.dumps(payload, indent=1, allow_nan=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    print(path, flush=True)
    return path


def interfinger() -> None:
    import mujoco
    scene = d1.Scene("tpu6", "padsT", seed=1)
    m = scene.model
    blocks = {f: m.geom(f + "_tipvis").id for f in d1.FINGERS}
    links = {f: [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith(f + "_")
                 and m.geom_contype[g] == 1] for f in d1.FINGERS}
    pairs = [("index", "middle"), ("thumb", "index"), ("thumb", "middle")]
    fromto = np.zeros(6)
    failed = {"queries": 0}

    def distance(data, g1, g2):
        """mj_geomDistance, None when it returns zero with a long witness segment (a failed query: MuJoCo 3.6
        returns 0.0 for one step of the release hold plan while the blocks' centres stay 64.8 mm apart)."""
        dist = mujoco.mj_geomDistance(m, data, g1, g2, 0.2, fromto)
        if dist <= 0.0 and np.linalg.norm(fromto[3:] - fromto[:3]) > 1e-3:
            failed["queries"] += 1
            return None
        return dist

    rows = []
    for fname, forced in (("20261007-padsT-seed1-256.json", False),
                          ("20261007-padsT-release-seed1-256.json", True)):
        source = json.loads((GAIT / fname).read_text())
        for method in ("hold", "direct_gradient", "cem", "hybrid_seed", "hybrid"):
            x = d1.apply_forced_release(source["candidates"][method]["x"], forced)
            tip_tip = {f"{a}-{b}": 1.0 for a, b in pairs}
            tip_link = {f"{a} block-{b} links": 1.0 for a in d1.FINGERS for b in d1.FINGERS if a != b}

            def watch(k, data):
                if k % 5:
                    return
                for a, b in pairs:
                    dist = distance(data, blocks[a], blocks[b])
                    if dist is not None:
                        tip_tip[f"{a}-{b}"] = min(tip_tip[f"{a}-{b}"], dist)
                for a in d1.FINGERS:
                    for b in d1.FINGERS:
                        if a != b:
                            for g in links[b]:
                                dist = distance(data, blocks[a], g)
                                key = f"{a} block-{b} links"
                                if dist is not None:
                                    tip_link[key] = min(tip_link[key], dist)

            row = scene.evaluate(x, callback=watch)
            rows.append({"source": fname, "plan": method, "held_final": row["held_final"],
                         "cos_gain": row["cos_gain"],
                         "min_block_block_mm": {k: v * 1e3 for k, v in tip_tip.items()},
                         "min_block_link_mm": {k: v * 1e3 for k, v in tip_link.items()}})
            print(fname, method, {k: round(v * 1e3, 2) for k, v in tip_tip.items()}, flush=True)
    write("interfinger.json", {
        "scene": str(scene.scene.relative_to(ROOT)), "mujoco_version": mujoco.__version__,
        "sample_every_ms": 5, "failed_distance_queries": failed["queries"], "rows": rows,
        "overall_min_block_block_mm": min(min(r["min_block_block_mm"].values()) for r in rows),
        "overall_min_block_link_mm": min(min(r["min_block_link_mm"].values()) for r in rows)})


def fdstep() -> None:
    import mujoco
    import scipy
    import real_v1_contact_gait_search as search
    out = {"scipy_version": scipy.__version__, "mujoco_version": mujoco.__version__,
           "parameter_scale_rad": {"yaw": d1.YAW_SCALE_RAD, "close": d1.CLOSE_SCALE_RAD},
           "steps": [1.49e-8, 1e-6, 1e-4, 1e-3, 1e-2], "tips": {}}
    steps = out["steps"]
    for label, tip, model, fname in (("1 mm packed", "tpu6", "padsT", "20261007-padsT-seed1-256.json"),
                                     ("convex mesh", "tpu6", "pt", "20261007-mesh-seed1-256.json"),
                                     ("single sphere", "sphere", "pt", "20261007-sphere-seed1-256.json")):
        scene = d1.Scene(tip, model, seed=1)
        visited = []
        evaluate = scene.evaluate

        def recording(x, **kw):
            visited.append(np.asarray(x, float).copy())
            return evaluate(x, **kw)

        scene.evaluate = recording
        search.optimize_gradient(scene, d1.hold_vector(), 15)
        scene.evaluate = evaluate
        visited = np.array(visited)
        source = json.loads((GAIT / fname).read_text())
        row = {"ccd_tolerance": float(scene.model.opt.ccd_tolerance),
               "probe_step_per_coordinate": np.abs(visited[1:14] - visited[0]).max(axis=1).tolist(),
               "hybrid_optimizer": {k: source["methods"]["hybrid_gradient"][k] for k in ("score", "evals")},
               "hybrid_seed_score": source["methods"]["hybrid_cem"]["score"], "plans": {}}
        for plan, x0 in (("hold", d1.hold_vector()),
                         ("cem128", np.asarray(source["candidates"]["hybrid_seed"]["x"], float))):
            t0 = time.perf_counter()
            grads = []
            for h in steps:
                g = np.zeros(d1.N_PARAM)
                for j in range(d1.N_PARAM):
                    e = np.zeros(d1.N_PARAM)
                    e[j] = h
                    xp, xm = np.clip(x0 + e, -1, 1), np.clip(x0 - e, -1, 1)
                    g[j] = (scene.evaluate(xp)["score"] - scene.evaluate(xm)["score"]) / (xp[j] - xm[j])
                grads.append(g)
            grads = np.array(grads)
            ref = grads[steps.index(1e-3)]
            row["plans"][plan] = {
                "score": scene.evaluate(x0)["score"], "gradients": grads.tolist(),
                "norm": np.linalg.norm(grads, axis=1).tolist(),
                "cosine_to_1e-3": [float(np.dot(g, ref) / (np.linalg.norm(g) * np.linalg.norm(ref)))
                                   for g in grads],
                "wall_s": time.perf_counter() - t0}
            print(label, plan, "norm", np.round(row["plans"][plan]["norm"], 4),
                  "cos", np.round(row["plans"][plan]["cosine_to_1e-3"], 4), flush=True)
        out["tips"][label] = row
    write("fdstep.json", out)


def cfd_reach() -> None:
    import mujoco
    rows = {}
    fromto = np.zeros(6)
    for spacing in (1.0, 2.0):
        scene = d1.Scene("tpu6", "padsT", seed=1, pad_spacing_mm=spacing)
        m, data = scene.model, d1.clone_data(scene.model, scene.held)
        tool = [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(d1.OBJ).id][0]
        pads = [g for g in range(m.ngeom) if "_pad" in m.geom(g).name]
        dist = np.array([mujoco.mj_geomDistance(m, data, g, tool, 0.2, fromto) for g in pads])
        rows[f"{spacing:g} mm"] = {
            "pads": len(pads), "pad_margin_m": float(m.geom_margin[pads].max()),
            "in_contact": int((dist < 0).sum()), "within_cfd_width": int((dist < CFD_WIDTH_M).sum()),
            "max_pad_tool_distance_mm": float(dist.max() * 1e3),
            "median_pad_tool_distance_mm": float(np.median(dist) * 1e3)}
        print(spacing, rows[f"{spacing:g} mm"], flush=True)
    write("cfd_reach.json", {"cfd_width_m": CFD_WIDTH_M, "rows": rows,
                             "mujoco_version": mujoco.__version__})


def frames() -> None:
    import mujoco
    scene = d1.Scene("tpu6", "padsT", seed=1)
    m, data = scene.model, d1.clone_data(scene.model, scene.held)
    for _ in range(20):
        mujoco.mj_step(m, data)
    body = m.body(d1.OBJ).id
    vadr = int(m.jnt_dofadr[m.body_jntadr[body]])
    rot = data.xmat[body].reshape(3, 3)            # body-to-world rotation
    omega_body = data.qvel[vadr + 3:vadr + 6].copy()
    torque_body = data.qfrc_constraint[vadr + 3:vadr + 6].copy()
    vel = np.zeros(6)
    mujoco.mj_objectVelocity(m, data, mujoco.mjtObj.mjOBJ_BODY, body, vel, 0)
    write("frames.json", {
        "mujoco_version": mujoco.__version__, "steps": 20,
        "tool_x_axis_in_world": rot[:, 0].tolist(),
        "angle_between_tool_x_and_world_x_deg": float(np.degrees(np.arccos(np.clip(rot[0, 0], -1, 1)))),
        "omega_body_rad_s": omega_body.tolist(), "omega_world_rad_s": (rot @ omega_body).tolist(),
        "omega_world_from_mj_objectVelocity_rad_s": vel[:3].tolist(),
        "torque_body_Nm": torque_body.tolist(), "torque_world_Nm": (rot @ torque_body).tolist()})


def _transfer(rows, dest, plan):
    sub = [r for r in rows if r["destination"] == dest and r["plan"] == plan]
    held = [r for r in sub if r["held_final"]]
    return {"held": len(held), "n": len(sub),
            "gain_if_held": float(np.mean([r["cos_gain"] for r in held])) if held else None}


def corrections() -> None:
    import mujoco
    searches = OUT / "searches"
    pairs = [("1 mm packed, free", "20261007-padsT-seed1-256.json", "pads-seed1-256-fd1e-2-cem20261007.json"),
             ("1 mm packed, forced release", "20261007-padsT-release-seed1-256.json",
              "pads-release-seed1-256-fd1e-2-cem20261007.json"),
             ("convex mesh, free", "20261007-mesh-seed1-256.json", "mesh-seed1-256-fd1e-2-cem20261007.json"),
             ("convex mesh, forced release", "20261007-mesh-release-seed1-256.json",
              "mesh-release-seed1-256-fd1e-2-cem20261007.json"),
             ("single sphere, free", "20261007-sphere-seed1-256.json", "sphere-seed1-256-fd1e-2-cem20261007.json"),
             ("single sphere, forced release", "20261007-sphere-release-seed1-256.json",
              "sphere-release-seed1-256-fd1e-2-cem20261007.json")]
    rerun = []
    for label, before, after in pairs:
        b = json.loads((GAIT / before).read_text())
        a = json.loads((searches / after).read_text())
        assert float(a["fd_step"]) == 1e-2 and a["cem_seed"] == 20261007
        pick = lambda d: {"direct": d["candidates"]["direct_gradient"]["cos_gain"],  # noqa: E731
                          "cem": d["candidates"]["cem"]["cos_gain"], "hybrid": d["candidates"]["hybrid"]["cos_gain"],
                          "hybrid_seed": d["candidates"]["hybrid_seed"]["cos_gain"],
                          "held": "/".join("yes" if d["candidates"][k]["held_final"] else "no"
                                           for k in ("direct_gradient", "hybrid")),
                          "wall_s": {k: v["wall_s"] for k, v in d["methods"].items()}}
        rerun.append({"label": label, "before": pick(b), "after": pick(a), "before_file": before, "after_file": after})
    seeds = {}
    for tip, name in (("1 mm packed", "pads"), ("convex mesh", "mesh"), ("single sphere", "sphere")):
        seeds[tip] = []
        for cem in (20261007, 1, 2, 3, 4):
            d = json.loads((searches / f"{name}-seed1-256-fd1e-2-cem{cem}.json").read_text())
            seeds[tip].append({"cem_seed": cem, "cem_gain": d["candidates"]["cem"]["cos_gain"],
                               "cem_held": d["candidates"]["cem"]["held_final"],
                               "hybrid_gain": d["candidates"]["hybrid"]["cos_gain"],
                               "hybrid_held": d["candidates"]["hybrid"]["held_final"],
                               "cem128_gain": d["candidates"]["hybrid_seed"]["cos_gain"],
                               "direct_gain": d["candidates"]["direct_gradient"]["cos_gain"]})
    old = json.loads((GAIT / "20261007-transfer-mj360.json").read_text())["rows"]
    new = json.loads((OUT / "transfer-fd1e-2-mj360.json").read_text())["rows"]
    transfer = {}
    for plan in ("tpu6_padsT_direct_gradient", "tpu6_padsT_hybrid_seed", "tpu6_padsT_hybrid",
                 "tpu6_padsT_release_hybrid", "tpu6_pt_direct_gradient", "tpu6_pt_hybrid"):
        transfer[plan] = {dest: {"before": _transfer(old, dest, plan), "after": _transfer(new, dest, plan)}
                          for dest in ("tpu6_padsT", "tpu6_pt", "sphere_pt")}
    # CPU finite differences of the 2026-10-07 recontact derivative: the 1 mm CEM release plan replayed to 280 ms
    # on 2 mm pads, then the index MCP target moved by +-0.01 rad for 10 steps (MuJoCo 3.6; the 10-07 replay ran 3.3.1)
    plan = json.loads((GAIT / "20261007-padsT-release-seed1-256.json").read_text())["candidates"]["cem"]["x"]
    scene = d1.Scene("tpu6", "padsT", seed=1, pad_spacing_mm=2.0)
    state = scene.state_at(plan, 280)
    index = scene.model.actuator("a_index_mcp").id
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        d1.cpu_metrics(scene.model, state, 10, index, 0.01)
        d1.cpu_metrics(scene.model, state, 10, index, -0.01)
        times.append(time.perf_counter() - t0)
    write("corrections.json", {
        "mujoco_version": mujoco.__version__, "fd_step": 1e-2, "rerun": rerun, "cem_seeds": seeds,
        "transfer": transfer, "recontact_cpu_fd_s": float(np.median(times)),
        "recontact_index_force_at_280ms_N": d1.contact_summary(scene.model, state)["finger_normal_force_N"]["index"]})


COMMANDS = {"interfinger": interfinger, "fdstep": fdstep, "cfd_reach": cfd_reach, "frames": frames,
            "corrections": corrections}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("commands", nargs="+", choices=sorted(COMMANDS))
    for name in parser.parse_args().commands:
        COMMANDS[name]()


if __name__ == "__main__":
    main()
