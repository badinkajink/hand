#!/usr/bin/env python3
"""MuJoCo-Warp GPU scaling of the sphere-pad holding fixture (2026-10-05).

Wrapper over scripts/distributed_contact_gpu.py (Codex, 2026-10-04). It keeps that
script's fixture (distributed_contact_transfer.build), its guide kernel, its mjlab
Simulation options and its timing: one CUDA graph per 10 ms block, timed between
synchronizations, host audit outside the timed region. It adds

* per-world contact and constraint capacities as options (Codex fixed 2048 / 8192);
* a 'legacy' contact mapping: plain positive-format MuJoCo solref (tc, 1) and solimp
  (d0 d0 .001 .5 2) at impratio 100, set once, with no direct-format or
  inverse-weight translation (the hom_chain.MjChainPlant legacy pads,
  d0 = 1 - 1/(tc^2 K_sphere diag));
* a whole-batch audit every block: per-world peak contacts and constraints, total
  contacts against naconmax, finiteness and the 25 mm / 2000 rad/s bound in every world;
* device VRAM per point and the GPU's other processes before and after each point;
* one JSONL row per point, fsynced as it lands; a point that saw contention is repeated once.
"""

import argparse
import gc
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp
from mjlab.sim import MujocoCfg, Simulation, SimulationCfg

import hom_chain as C
from contact_surface.scaling import compensated_solref
from distributed_contact_gpu import guide
from distributed_contact_transfer import build

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/experiments/20261005-gpu_scaling/gpu_scaling.jsonl"
MIB = 1024.0 * 1024.0


def gpu_snapshot():
    me = os.getpid()
    snap = dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"))
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.free,memory.total",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20,
        ).stdout.strip().split(",")
        snap.update(util_pct=float(q[0]), used_mib=float(q[1]), free_mib=float(q[2]),
                    total_mib=float(q[3]))
    except Exception as e:  # noqa: BLE001
        snap["query_error"] = repr(e)
    others, own = [], None
    try:
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20,
        ).stdout.strip().splitlines()
        for line in apps:
            parts = [x.strip() for x in line.split(",")]
            if len(parts) < 3:
                continue
            try:
                mem = float(parts[2])
            except ValueError:
                mem = None
            if int(parts[0]) == me:
                own = mem
            else:
                others.append(dict(pid=int(parts[0]), name=parts[1], used_mib=mem))
    except Exception as e:  # noqa: BLE001
        snap["apps_error"] = repr(e)
    sm = {}
    try:
        pm = subprocess.run(["nvidia-smi", "pmon", "-c", "1", "-s", "u"], capture_output=True,
                            text=True, timeout=20).stdout.splitlines()
        for line in pm:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 4 and parts[1].isdigit() and parts[3].isdigit():
                sm[int(parts[1])] = int(parts[3])
    except Exception as e:  # noqa: BLE001
        snap["pmon_error"] = repr(e)
    for o in others:
        o["sm_pct"] = sm.get(o["pid"])
    snap.update(own_mib=own, own_sm_pct=sm.get(me), others=others)
    return snap


def contended(snap, after=False):
    """Another process computing on the GPU. Device utilization counts only in the
    snapshot taken while this process idles (before a point); after a point it is
    this process's own load, so only the other processes' SM share counts there."""
    busy = [o for o in snap.get("others", []) if (o.get("sm_pct") or 0) >= 5]
    return bool(busy) or (not after and snap.get("util_pct", 0.0) > 20.0)


def apply_legacy(m, spacing):
    qdict, _ = C.postures([0.0])
    dirs = C.contact_dirs(qdict[0.0])
    trial = C.make_trial(0, d_cg=0.0, perturb=False)
    trial["mscale"] = 1.0
    _, info, _ = C.chain_scene(f"mj:spheres:s{spacing}:rs0.75:tr0.03:ir100", trial, dirs,
                               pad_tc=(0.015, 0.9))
    K = float(info["K_sphere"])
    tc = 0.03 / 2.0
    diag = float(m.body_invweight0[m.body("thumb_tip").id, 0] + m.body_invweight0[m.body("tool").id, 0])
    d0 = 1.0 - 1.0 / (tc**2 * K * diag)
    if d0 < 0.05:
        raise ValueError(f"legacy d0 {d0:.4f} below the 0.05 floor")
    m.opt.impratio = 100.0
    imp = [d0, d0, 0.001, 0.5, 2.0]
    for i in range(m.npair):
        m.pair_solref[i] = [tc, 1.0]
        m.pair_solimp[i] = imp
        m.pair_solreffriction[i] = [0.0, 0.0]
    for g in range(m.ngeom):
        if m.geom(g).name.startswith(("thumb_pad", "index_pad")):
            m.geom_solref[g] = [tc, 1.0]
            m.geom_solimp[g] = imp
    return dict(mapping="legacy_positive_solref", K_sphere=K, solref=[tc, 1.0], solimp=imp,
                diag_invweight=diag, impratio=100.0)


def apply_compiled(m, p, meta):
    # Same as distributed_contact_gpu.run: welded-body inverse weights for Warp 3.6.
    for i in range(m.npair):
        lam = sum(m.body_invweight0[m.body_weldid[m.geom_bodyid[g]], 0]
                  for g in (m.pair_geom1[i], m.pair_geom2[i]))
        sr = compensated_solref(p["stiffness"], p["relaxation"], meta["d0"], lam)
        m.pair_solref[i] = sr
        m.pair_solreffriction[i] = [0.0, sr[1] * 10.0 / m.opt.impratio]
    return dict(mapping="compiled_direct_solref", stiffness=float(p["stiffness"]),
                relaxation=float(p["relaxation"]), d0=float(meta["d0"]),
                impratio=float(m.opt.impratio), pair_solref_0=[float(x) for x in m.pair_solref[0]])


def array_bytes(obj, seen=None, depth=0):
    """Bytes held by every Warp array reachable from obj (Data and its nested structs)."""
    seen = set() if seen is None else seen
    if id(obj) in seen or depth > 4:
        return 0
    seen.add(id(obj))
    if isinstance(obj, wp.array):
        return int(obj.capacity or 0)
    fields = getattr(obj, "__dict__", None)
    if not fields:
        return 0
    return sum(array_bytes(v, seen, depth + 1) for v in fields.values()
               if isinstance(v, wp.array) or hasattr(v, "__dict__"))


def run_point(pt):
    m, d, p, meta = build(dict(spacing_mm=pt["spacing_mm"], timestep=pt["timestep"], policy="compiled",
                               torque_peak=pt["torque_peak"], duration=pt["duration"]))
    mapping = apply_legacy(m, pt["spacing_mm"]) if pt["policy"] == "legacy" else apply_compiled(m, p, meta)
    mujoco.mj_forward(m, d)
    dt = m.opt.timestep
    nworld = pt["nworld"]
    cfg = SimulationCfg(
        nconmax=pt["nconmax"],
        njmax=pt["njmax"],
        mujoco=MujocoCfg(timestep=dt, integrator="implicitfast", impratio=m.opt.impratio,
                         cone="elliptic", jacobian="dense", solver="newton", iterations=200,
                         tolerance=pt["solver_tolerance"], ls_iterations=50, gravity=(0.0, 0.0, 0.0)),
    )
    dev = wp.get_device("cuda:0")
    wp.synchronize()
    free_before = dev.free_memory
    pool_before = wp.get_mempool_used_mem_current("cuda:0")
    t0 = time.perf_counter()
    sim = Simulation(nworld, cfg, m, "cuda:0")
    wd, wm = sim.wp_data, sim.wp_model
    q0 = np.tile(d.qpos, (nworld, 1)).astype(np.float32)

    def reset():
        wd.qpos.assign(q0)
        wd.qvel.zero_()
        wd.qacc_warmstart.zero_()
        wd.time.zero_()
        sim.forward()
        wp.synchronize()

    wd.ctrl.assign(np.tile(d.ctrl, (nworld, 1)).astype(np.float32))
    reset()
    block = round(0.01 / dt)
    P = np.asarray(meta["P"])
    tqa = meta["tqa"]
    inp = [wd.qpos, wd.qvel, wd.time, wd.xfrc_applied, meta["tqa"], meta["tva"], meta["tool"],
           wp.vec3(meta["P"]), wp.vec3(meta["u"]), pt["torque_peak"]]
    wp.launch(guide, dim=nworld, inputs=inp)
    wp.synchronize()
    with wp.ScopedCapture() as cap:
        for _ in range(block):
            wp.launch(guide, dim=nworld, inputs=inp)
            mjw.step(wm, wd)
    reset()
    startup = time.perf_counter() - t0
    free_after = dev.free_memory
    pool_live = wp.get_mempool_used_mem_current("cuda:0") - pool_before
    data_bytes = array_bytes(wd)
    snap_alloc = gpu_snapshot()
    naconmax = int(getattr(wd, "naconmax", pt["nconmax"] * nworld))
    njmax = int(getattr(wd, "njmax", pt["njmax"]))
    ticks = []
    peak_world_con, peak_total_con, peak_nefc = 0, 0, 0
    max_disp, max_v = 0.0, 0.0
    status, failure, bad_worlds = "complete", None, 0
    start = time.perf_counter()
    for _ in range(round(pt["duration"] / 0.01)):
        wp.synchronize()
        tick = time.perf_counter()
        wp.capture_launch(cap.graph)
        wp.synchronize()
        ticks.append(time.perf_counter() - tick)
        allq, allv = wd.qpos.numpy(), wd.qvel.numpy()
        nacon = int(wd.nacon.numpy()[0])
        wid = wd.contact.worldid[: min(nacon, naconmax)].numpy()
        per_world = np.bincount(wid, minlength=nworld) if wid.size else np.zeros(1, int)
        peak_world_con = max(peak_world_con, int(per_world.max()))
        peak_total_con = max(peak_total_con, nacon)
        peak_nefc = max(peak_nefc, int(wd.nefc.numpy().max()))
        fin = np.isfinite(allq).all(axis=1) & np.isfinite(allv).all(axis=1)
        disp = np.linalg.norm(allq[:, tqa : tqa + 3] - P, axis=1)
        vabs = np.abs(allv).max(axis=1)
        ok = fin & (disp <= 0.025) & (vabs <= 2000)
        max_disp = max(max_disp, float(np.nanmax(disp)) if fin.any() else float("nan"))
        max_v = max(max_v, float(np.nanmax(vabs)) if fin.any() else float("nan"))
        if not ok.all():
            bad_worlds = int((~ok).sum())
            status = "failed"
            failure = (f"{bad_worlds} of {nworld} worlds nonfinite or beyond 25 mm / 2000 rad/s "
                       f"at t = {float(wd.time.numpy()[0]):.3f} s; nonfinite {int((~fin).sum())}")
            break
    elapsed = sum(ticks)
    simulated = block * len(ticks) * dt
    steps = block * len(ticks)
    row = dict(
        status=status, failure=failure, mapping=mapping,
        startup_s=startup, wall_s=time.perf_counter() - start, physics_s=elapsed,
        simulated_s=simulated, steps=steps,
        ms_per_step_median=float(np.median(ticks) / block * 1000),
        ms_per_step_p90=float(np.percentile(ticks, 90) / block * 1000),
        world_steps_per_s=nworld * steps / elapsed,
        sim_s_per_wall_s=nworld * simulated / elapsed,
        per_world_real_time_factor=simulated / elapsed,
        peak_contacts_per_world=peak_world_con, nconmax_per_world=pt["nconmax"],
        peak_contacts_total=peak_total_con, naconmax=naconmax,
        contact_overflow=bool(peak_total_con > naconmax),
        peak_constraints_per_world=peak_nefc, njmax_per_world=njmax,
        constraint_overflow=bool(peak_nefc > njmax),
        all_worlds_finite=bool(status == "complete"), bad_worlds=bad_worlds,
        peak_displacement_mm=max_disp * 1000, peak_abs_qvel=max_v,
        vram_alloc_mib=(free_before - free_after) / MIB,
        vram_pool_live_mib=pool_live / MIB, vram_data_arrays_mib=data_bytes / MIB,
        vram_process_mib=snap_alloc.get("own_mib"), gpu_free_after_alloc_mib=snap_alloc.get("free_mib"),
    )
    del cap, sim, wd, wm
    gc.collect()
    wp.synchronize()
    return row


def key(pt):
    return (f"{pt['policy']}_s{pt['spacing_mm']}_dt{pt['timestep']}_n{pt['nworld']}"
            f"_c{pt['nconmax']}_j{pt['njmax']}")


def plan(preset):
    base = dict(torque_peak=0.012, duration=0.8, solver_tolerance=1e-8)
    cap1, cap05, codex = (256, 1024), (1024, 2048), (2048, 8192)
    pts = []

    def add(policy, s, dt, ns, cap):
        for n in ns:
            pts.append(dict(base, policy=policy, spacing_mm=s, timestep=dt, nworld=n,
                            nconmax=cap[0], njmax=cap[1]))

    if preset in ("main", "all"):
        add("compiled", 1.0, 0.0005, [128], codex)  # Codex reference: 168k world-steps/s
        add("compiled", 1.0, 0.001, [1024, 4096, 8192], cap1)
        add("compiled", 1.0, 0.0005, [1024, 4096, 8192], cap1)
        add("legacy", 1.0, 0.001, [1024, 4096, 8192], cap1)
        add("legacy", 1.0, 0.0005, [1024, 4096, 8192], cap1)
        add("compiled", 1.0, 0.0005, [1024], codex)  # capacity effect at equal nworld
    if preset in ("fine", "all"):
        add("compiled", 0.5, 0.0005, [1024, 4096], cap05)
        add("compiled", 0.5, 0.001, [1024, 4096], cap05)
        add("legacy", 0.5, 0.001, [1024, 4096], cap05)
    return pts


def consolidate(raw_files, out):
    """One row per point: the last uncontended complete attempt, else the last attempt."""
    rows = []
    for path in raw_files:
        rows += [dict(json.loads(line), raw_file=path.name) for line in path.read_text().splitlines()]
    for r in rows:  # first-pass rows used the after-snapshot's own-load utilization
        r["contention"] = contended(r.get("gpu_before", {})) or contended(r.get("gpu_after", {}), after=True)
    best = {}
    for r in rows:  # later files and later attempts win among equals
        clean = r.get("status") == "complete" and not r.get("contention")
        prev = best.get(r["key"])
        prev_clean = prev is not None and prev.get("status") == "complete" and not prev.get("contention")
        if prev is None or clean or not prev_clean:
            best[r["key"]] = r
    order = [key(pt) for pt in plan("all")]
    with open(out, "w") as fh:
        for k in order:
            if k in best:
                fh.write(json.dumps(best[k]) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    print(f"wrote {sum(k in best for k in order)} points to {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", default="all", choices=["main", "fine", "all"])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--min-free-mib", type=float, default=2048.0)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--consolidate", type=Path, nargs="+",
                    help="raw attempt files to merge into --out, one row per point; no simulation")
    args = ap.parse_args()
    if args.consolidate:
        consolidate(args.consolidate, args.out)
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            r = json.loads(line)
            if r.get("status") in ("complete", "failed", "skipped") and not r.get("contention"):
                done.add(r["key"])
    try:
        wp.set_mempool_release_threshold("cuda:0", 0)
    except Exception:  # noqa: BLE001
        pass
    vram_per_world = {}
    launched = 0
    for pt in plan(args.preset):
        k = key(pt)
        if k in done:
            continue
        if args.limit is not None and launched >= args.limit:
            break
        fam = (pt["policy"], pt["spacing_mm"], pt["nconmax"], pt["njmax"])
        for attempt in range(2):
            time.sleep(2.0)
            before = gpu_snapshot()
            row = dict(key=k, attempt=attempt, config=pt, gpu_before=before,
                       host=os.uname().nodename, warp=wp.__version__, mujoco=mujoco.__version__)
            predicted = vram_per_world.get(fam, 0.0) * pt["nworld"] * 1.15
            if predicted and before.get("free_mib", 1e9) - predicted < args.min_free_mib:
                row.update(status="skipped",
                           failure=f"predicted {predicted:.0f} MiB leaves < {args.min_free_mib:.0f} MiB free")
            else:
                try:
                    row.update(run_point(pt))
                except Exception as e:  # noqa: BLE001
                    row.update(status="error", error=repr(e), traceback=traceback.format_exc())
                    gc.collect()
            after = gpu_snapshot()
            row["gpu_after"] = after
            row["contention"] = contended(before) or contended(after, after=True)
            if row.get("vram_alloc_mib"):
                vram_per_world[fam] = max(vram_per_world.get(fam, 0.0), row["vram_alloc_mib"] / pt["nworld"])
            with open(args.out, "a") as fh:
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            print(k, attempt, row["status"], f"{row.get('world_steps_per_s', 0):.0f} ws/s",
                  f"{row.get('ms_per_step_median', 0):.3f} ms", f"vram {row.get('vram_alloc_mib', 0):.0f} MiB",
                  f"con {row.get('peak_contacts_per_world')}/{pt['nconmax']}",
                  f"nefc {row.get('peak_constraints_per_world')}/{pt['njmax']}",
                  "CONTENDED" if row["contention"] else "", flush=True)
            if not row["contention"] or row["status"] == "skipped":
                break
        launched += 1


if __name__ == "__main__":
    main()
