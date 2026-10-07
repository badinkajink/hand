#!/usr/bin/env python3
"""GPU physics throughput of the hand holding the tool: MuJoCo-Warp against Newton, per fingertip contact model.

Scene: D6's bench scene (reorient_backends.build_scene: palm welded at the plan's pose, the 25 mm tool on the post,
the fitted servo plant kp 4 N m/rad, tau 0.02 s, 1 N m, mu 1.0) held at the plan's grip, gravity on. Solver settings
are the RL trainer's: 2 ms step, implicitfast, elliptic cone, impratio 10, Newton solver 10 iterations / 20 line
search. nworld copies step with the grip targets; after a 0.4 s settle, blocks of 50 steps are captured as one CUDA
graph and replayed.

  mode      mjw       raw MuJoCo-Warp (put_model, make_data)
            nt_mjc    Newton SolverMuJoCo with MuJoCo-Warp's own collision (use_mujoco_contacts=True)
            nt_pt     Newton SolverMuJoCo with Newton's collision pipeline (point contacts)
            nt_hydro  Newton's pipeline with the tips and the tool as hydroelastic SDF shapes (kh 1e7/8.5 mm =
                      1.18e9 N/m^3 on the tips, 100x on the tool, 0.5 mm voxels, +-6 mm narrow band, contact
                      reduction). Tip `mesh`: the printed TPU block as one newton.Mesh per tip built from the OBJ's
                      vertices (Newton's importer would need trimesh) with its SDF cooked by Mesh.build_sdf
MuJoCo-Warp collision uses the SAP_SEGMENTED broadphase (mjlab's choice; put_model defaults to all pairs).
  tip       legacy    the 10.6 x 21.2 x 15 mm box the plans were exported against
            mesh      the TPU block (2.7 mm fillets) as one convex mesh
            pads      the TPU block as 1 mm sphere pads (1059 per tip) that touch only the tool; the block's convex
                      mesh takes the other contacts (as in the RL scenes of make_pad_morphology_run.py)

--pad-d0 newton (pads only): the pads reach their stiffness through solimp d0 = 1 - 1/(tc^2 K (w_tip + w_tool)) with the
inverse weights w of the MJCF compile (reorient_backends.replace_tips). Newton forwards the pads' solref/solimp to its
contacts unchanged, so d0 is recomputed from the inverse weights of Newton's own MuJoCo model (a one-world build first):
d0' = 1 - (1 - d0) (w_tip + w_tool)_mjcf / (w_tip + w_tool)_newton. Every row records both sets of inverse weights.

Newton shapes get margin 0 and a 0.5 mm contact gap (its default gap reports every pair within centimetres and filled
the 64-contact buffer). Measured: us per world-step (wall time of a block / steps / nworld), contacts and constraints per world, GPU memory
in use (nvidia-smi), and the fraction of worlds still holding the tool (centre within 20 mm of its start) after 1 s.

    PY=logs/20261004-contact-transfer/venv/bin/python     # newton 1.7.0.dev0, mujoco / mujoco_warp 3.14
    WARP_CACHE_PATH=$(mktemp -d) $PY scripts/newton_hand_throughput.py --mode nt_pt --tip legacy --nworld 1024
Rows: docs/experiments/20261006-rl_contact/newton_throughput.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "docs/experiments/20261006-rl_contact/newton_throughput.jsonl"
SCENES = ROOT / "logs/20261006-rl_contact/scenes"
HAND = "D6"
PLANT = "kp4_kv0_fr1_fl0_dp0.08"
DT, ITER, LS, IMPRATIO = 0.002, 10, 20, 10.0
BLOCK = 50
KH_TIP = 1e7 / 0.0085       # E / h of the pad model (h = the block's half depth), N/m^3
SDF_VOXEL, SDF_BAND = 0.0005, 0.006   # the contact bed's +-6 mm band: a +-3 mm band let the squeezed tips sink through


def read_obj(path):
    V, F = [], []
    for line in open(path):
        t = line.split()
        if not t:
            continue
        if t[0] == "v":
            V.append([float(x) for x in t[1:4]])
        elif t[0] == "f":
            idx = [int(x.split("/")[0]) - 1 for x in t[1:]]
            for j in range(1, len(idx) - 1):
                F += [idx[0], idx[j], idx[j + 1]]
    return np.array(V, np.float32), np.array(F, np.int32)


def scene_xml(tip):
    """The bench scene MJCF (built with the hom venv, which has scipy for the TPU geometry) and its meta."""
    tag = {"legacy": ("legacy", "pt"), "mesh": ("tpu2.7", "pt"), "pads": ("tpu2.7", "padsT")}[tip]
    meta_p = SCENES / f"{HAND}_{tag[0]}_{tag[1]}.json"
    if not meta_p.exists():
        code = (f"import sys,json; sys.path.insert(0,'{ROOT}/scripts'); import reorient_backends as RB; "
                f"from pathlib import Path; p,m=RB.build_scene('{HAND}','{tag[0]}','{tag[1]}','{PLANT}','bed',"
                f"{IMPRATIO},1.0,out_dir=Path('{SCENES}')); m['scene']=str(p); "
                f"json.dump(m, open('{meta_p}','w'), default=lambda o: o.tolist() if hasattr(o,'tolist') else str(o))")
        subprocess.run([str(ROOT / "logs/20261001-hom_contact/venv/bin/python"), "-c", code], check=True)
    meta = json.loads(meta_p.read_text())
    root = ET.parse(meta["scene"]).getroot()
    opt = root.find("option")
    opt.set("timestep", f"{DT}")
    opt.set("iterations", f"{ITER}")
    opt.set("ls_iterations", f"{LS}")
    opt.set("tolerance", "1e-8")
    opt.set("impratio", f"{IMPRATIO:g}")
    return ET.tostring(root, encoding="unicode"), meta


def gpu_used_mb():
    try:
        return int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True).stdout.split()[0])
    except Exception:
        return None


FINGERS, JOINTS = ("thumb", "index", "middle"), ("yaw", "mcp", "pip")


def grip_targets(meta):
    import csv  # noqa: F401
    plan = json.loads(next((ROOT / "docs/experiments").glob("*/deploy/sv1_u0308_b050_plan.json")).read_text())
    poses = {p["name"]: p["joints"] for p in plan["poses"]}
    return {f"{f}_{j}": float(np.radians(poses["grip"][f][j])) for f in FINGERS for j in JOINTS}


MJW = {"broadphase": "sap", "nconmax": None, "njmax": None, "used": None}


def run_mjw(xml, meta, nworld, nblocks):
    import mujoco
    import mujoco_warp as mjw
    import warp as wp
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    tb = m.body("screwdriver_medium").id
    qa = m.jnt_qposadr[m.body_jntadr[tb]]
    d.qpos[qa:qa + 7] = meta["tool7"]
    grip = grip_targets(meta)
    for n, v in meta["q0"].items():
        d.qpos[m.jnt_qposadr[m.joint(n).id]] = v
    for n, v in grip.items():
        d.ctrl[m.actuator(f"a_{n}").id] = v
    mujoco.mj_forward(m, d)
    wm = mjw.put_model(m)
    if MJW["broadphase"] == "sap":
        wm.opt.broadphase = mjw.BroadphaseType.SAP_SEGMENTED  # rows before 2026-10-06 20:00; mjlab keeps put_model's
    MJW["used"] = int(wm.opt.broadphase)
    nconmax, njmax = (64, 256) if m.ngeom < 100 else (320, 1024)
    nconmax, njmax = MJW["nconmax"] or nconmax, MJW["njmax"] or njmax
    wd = mjw.put_data(m, d, nworld=nworld, nconmax=nconmax, njmax=njmax)

    def block():
        for _ in range(BLOCK):
            mjw.step(wm, wd)
    for _ in range(int(0.4 / DT / BLOCK)):
        block()
    wp.synchronize()
    with wp.ScopedCapture() as cap:
        block()
    t0 = time.perf_counter()
    for _ in range(nblocks):
        wp.capture_launch(cap.graph)
    wp.synchronize()
    el = time.perf_counter() - t0
    xp = wd.xpos.numpy()[:, tb, :]
    held = float(np.mean(np.linalg.norm(xp - np.asarray(meta["tool7"][:3]), axis=1) < 0.02))
    nacon = int(wd.nacon.numpy()[0])
    return dict(us_per_world_step=1e6 * el / (nblocks * BLOCK * nworld), contacts_per_world=nacon / nworld,
                nefc_world_max=int(wd.nefc.numpy().max()), held_frac=held, ngeom=m.ngeom,
                sim_s=round(0.4 + (nblocks + 1) * BLOCK * DT, 3), broadphase=MJW["used"], nconmax=nconmax,
                njmax=njmax)


TOOL = "screwdriver_medium"


def mjcf_invweights(xml):
    """Inverse weights and masses of the tips and the tool in a plain MuJoCo compile of the scene."""
    import mujoco
    m = mujoco.MjModel.from_xml_string(xml)
    names = [f"{f}_tip" for f in FINGERS] + [TOOL]
    return ({n: float(m.body_invweight0[m.body(n).id, 0]) for n in names},
            {n: float(m.body_mass[m.body(n).id]) for n in names})


def solver_invweights(model, solver):
    """The same from Newton's own MuJoCo model (world 0), the one MuJoCo-Warp steps."""
    mjm = solver.mj_model
    labels = [x.split("/")[-1] for x in model.body_label]
    m2n = solver.mjc_body_to_newton.numpy()[0]
    names = [f"{f}_tip" for f in FINGERS] + [TOOL]
    w, mass = {}, {}
    for bi in range(mjm.nbody):
        nb = int(m2n[bi])
        nm = labels[nb] if 0 <= nb < len(labels) else ""
        if nm in names:
            w[nm], mass[nm] = float(mjm.body_invweight0[bi, 0]), float(mjm.body_mass[bi])
    return w, mass


def pad_d0_from(xml, meta, w_mjcf, w_new):
    """Rewrite each finger's pad solimp with d0' = 1 - (1 - d0)(w_tip + w_tool)_mjcf / (w_tip + w_tool)_newton."""
    d0n = {}
    for f in FINGERS:
        d0 = float(meta["pad_d0"][f])
        d0n[f] = 1.0 - (1.0 - d0) * (w_mjcf[f"{f}_tip"] + w_mjcf[TOOL]) / (w_new[f"{f}_tip"] + w_new[TOOL])
        xml = re.sub(rf'(<geom name="{f}_pad\d+"[^>]*?solimp=")[0-9.eE+-]+ [0-9.eE+-]+',
                     lambda m_, d=d0n[f]: f"{m_.group(1)}{d:.6g} {d:.6g}", xml)
    return xml, d0n


def run_newton(xml, meta, nworld, nblocks, mode, tip, buffer_fraction=1.0, buffer_mult_broad=1, pad_d0="mjcf",
               _probe=False):
    import newton
    import warp as wp
    hydro = mode == "nt_hydro"
    extra = {}
    if tip == "pads" and not _probe:
        w_mjcf, m_mjcf = mjcf_invweights(xml)
        w_new, m_new = run_newton(xml, meta, 1, 0, mode, tip, _probe=True)
        extra = dict(invweight0_mjcf=w_mjcf, invweight0_newton=w_new, body_mass_mjcf=m_mjcf, body_mass_newton=m_new,
                     pad_d0_mjcf={f: float(meta["pad_d0"][f]) for f in FINGERS}, pad_d0_mode=pad_d0)
        if pad_d0 == "newton":
            xml, d0n = pad_d0_from(xml, meta, w_mjcf, w_new)
            extra["pad_d0_newton"] = d0n
    if tip in ("pads", "mesh"):
        # Newton's MJCF importer loads mesh files through trimesh, which its environment lacks: drop the block mesh
        # here and, for the mesh tip, add it back below from the OBJ's own vertices
        root = ET.fromstring(xml)
        for b in root.iter("body"):
            if b.get("name", "").endswith("_tip"):
                for g in list(b.findall("geom")):
                    if g.get("type") == "mesh":
                        b.remove(g)
        xml = ET.tostring(root, encoding="unicode")
    xml = re.sub(r'solref="([0-9.eE+-]+)"', r'solref="\1 1"', xml)       # one-value solref -> damping ratio 1
    b0 = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    b0.add_mjcf(xml, ctrl_direct=True, parse_sites=False, parse_visuals=False)
    # Newton's default contact gap reports every shape pair within centimetres (floor against every finger link) and
    # fills the contact buffer; MuJoCo's margin is 0. A 0.5 mm gap gives MuJoCo's 5 contacts in the box-tip grip.
    for i in range(len(b0.shape_gap)):
        b0.shape_gap[i] = 0.0005
        b0.shape_margin[i] = 0.0
    q0 = dict(meta["q0"])
    for i, lab in enumerate(b0.joint_label):
        n = lab.split("/")[-1]
        if n in q0:
            b0.joint_q[b0.joint_q_start[i]] = float(q0[n])
    body_of = {x.split("/")[-1]: k for k, x in enumerate(b0.body_label)}
    if tip == "mesh":
        # the printed TPU block as one mesh per tip, vertices as authored in the tip body frame
        cfg = newton.ModelBuilder.ShapeConfig(margin=0.0, gap=0.0 if hydro else 0.0005, mu=1.0, kh=KH_TIP,
                                              density=1000.0, is_hydroelastic=hydro, mu_torsional=0.0, mu_rolling=0.0)
        for f, sign in (("thumb", 1.0), ("index", -1.0), ("middle", -1.0)):
            V, F = read_obj(meta["meshes"][str(sign)])
            mesh = newton.Mesh(V, F)
            if hydro:
                mesh.build_sdf(target_voxel_size=SDF_VOXEL, narrow_band_range=(-SDF_BAND, SDF_BAND), margin=SDF_BAND)
            k = b0.add_shape_mesh(body_of[f"{f}_tip"], mesh=mesh, cfg=cfg, label=f"{f}_tpu")
            if hydro:
                b0.shape_material_kf[k] = 10.0
    if hydro:
        tool_body = body_of["screwdriver_medium"]
        for i, lab in enumerate(b0.shape_label):
            n = lab.split("/")[-1]
            on_tip = n.endswith("_tipgeom")
            if not (on_tip or b0.shape_body[i] == tool_body):
                continue
            b0.shape_flags[i] |= int(newton.ShapeFlags.HYDROELASTIC)
            b0.shape_sdf_target_voxel_size[i] = SDF_VOXEL
            b0.shape_sdf_narrow_band_range[i] = (-SDF_BAND, SDF_BAND)
            b0.shape_sdf_padding[i] = SDF_BAND
            b0.shape_margin[i] = 0.0
            b0.shape_gap[i] = 0.0
            b0.shape_material_kh[i] = KH_TIP * (1.0 if on_tip else 100.0)
            b0.shape_material_mu[i] = 1.0
            b0.shape_material_kf[i] = 10.0
    scene = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    scene.replicate(b0, nworld)
    model = scene.finalize(device="cuda:0")
    big = model.shape_count // nworld > 100
    nconmax, njmax = (320, 1024) if (big or mode == "nt_hydro") else (64, 256)
    pipe = None
    if mode != "nt_mjc":
        kw = {}
        if mode == "nt_hydro":
            from newton.geometry import HydroelasticSDF
            kw["sdf_hydroelastic_config"] = HydroelasticSDF.Config(reduce_contacts=True, anchor_contact=True,
                                                                   buffer_fraction=buffer_fraction,
                                                                   buffer_mult_broad=int(buffer_mult_broad))
        pipe = newton.CollisionPipeline(model, reduce_contacts=True, rigid_contact_max=nworld * nconmax,
                                        broad_phase="explicit", **kw)
    solver = newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=(mode == "nt_mjc"), disable_sensors=True,
                                         njmax=njmax, nconmax=nconmax, iterations=ITER, ls_iterations=LS,
                                         cone="elliptic", integrator="implicitfast", tolerance=1e-8,
                                         impratio=IMPRATIO)
    if _probe:
        return solver_invweights(model, solver)
    if hasattr(solver, "mjw_model"):
        import mujoco_warp as mjw
        solver.mjw_model.opt.broadphase = mjw.BroadphaseType.SAP_SEGMENTED
    if tip == "pads":
        mm0 = solver.mj_model
        pad_gi = [g for g in range(mm0.ngeom) if "_pad" in (mm0.geom(g).name or "")]
        if pad_gi:
            extra["pad_solimp_solver"] = [float(x) for x in mm0.geom_solimp[pad_gi[0]]]
    s0, s1 = model.state(), model.state()
    ctrl = model.control()
    mm = solver.mj_model
    grip = grip_targets(meta)
    # Newton's MuJoCo model carries no actuator names: map each actuator to its joint
    full = [mm.joint(int(mm.actuator_trnid[a, 0])).name for a in range(mm.nu)]   # body-path names
    names = [next((g for g in grip if f.endswith(g)), None) for f in full]
    if None in names:
        raise ValueError(f"actuated joints without a grip target: {[f for f, n in zip(full, names) if n is None]}")
    tgt = np.array([grip[n] for n in names], np.float32)
    c = ctrl.mujoco.ctrl
    c.assign(np.tile(tgt, nworld).reshape(c.shape))
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    cc = pipe.contacts() if pipe is not None else None
    labels = [x.split("/")[-1] for x in model.body_label]
    tool_ids = np.array([i for i, x in enumerate(labels) if x == "screwdriver_medium"])
    p_start = s0.body_q.numpy()[tool_ids, :3].copy()
    st = {"s0": s0, "s1": s1}

    def block():
        for _ in range(BLOCK):
            st["s0"].clear_forces()
            if pipe is not None:
                pipe.collide(st["s0"], cc)
            solver.step(st["s0"], st["s1"], ctrl, cc, DT)
            st["s0"], st["s1"] = st["s1"], st["s0"]
    for _ in range(int(0.4 / DT / BLOCK)):
        block()
    wp.synchronize()
    hs = getattr(pipe, "hydroelastic_sdf", None) if pipe is not None else None
    if hs is not None:
        hs._host_warning_poll_interval = 10 ** 12
    with wp.ScopedCapture() as cap:
        block()                        # BLOCK is even, so s0/s1 return to the same arrays after each replay
    t0 = time.perf_counter()
    for _ in range(nblocks):
        wp.capture_launch(cap.graph)
    wp.synchronize()
    el = time.perf_counter() - t0
    p_end = st["s0"].body_q.numpy()[tool_ids, :3]
    held = float(np.mean(np.linalg.norm(p_end - p_start, axis=1) < 0.02))
    md = solver.mjw_data
    na = int(md.nacon.numpy()[0])
    out = dict(us_per_world_step=1e6 * el / (nblocks * BLOCK * nworld), contacts_per_world=na / nworld,
               nefc_world_max=int(md.nefc.numpy().max()), held_frac=held, ngeom=int(mm.ngeom),
               sim_s=round(0.4 + (nblocks + 1) * BLOCK * DT, 3), buffer_fraction=buffer_fraction,
               buffer_mult_broad=int(buffer_mult_broad), **extra)
    if hs is not None:
        # stage fullness (newton_scaling.right_size): iso-refinement and face-contact stages at 1.5x their peak,
        # the broad phase by an integer multiplier that restores its count at that fraction
        counts = [int(c.numpy()[0]) for c in hs.iso_buffer_counts]
        caps = [int(hs.max_num_blocks_broad), *[int(x) for x in hs.iso_max_dims]]
        red = getattr(hs, "contact_reduction", None)
        fc = getattr(red, "contact_count", None)
        face = int(fc.numpy()[0]) if fc is not None else 0
        frac_stage = [c / max(1, k) for c, k in zip(counts, caps)]
        fr = max(max(frac_stage[1:]) if len(frac_stage) > 1 else 0.0, face / max(1, int(hs.max_num_face_contacts)))
        rec = float(min(1.0, np.ceil(fr * 1.5 * 1000) / 1000))
        out.update(hydro_counts=counts, hydro_caps=caps, face_contacts=face, face_cap=int(hs.max_num_face_contacts),
                   recommended_buffer_fraction=rec,
                   recommended_buffer_mult_broad=int(np.ceil(frac_stage[0] / max(rec, 1e-6) * 1.02)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["mjw", "nt_mjc", "nt_pt", "nt_hydro"])
    ap.add_argument("--tip", required=True, choices=["legacy", "mesh", "pads"])
    ap.add_argument("--nworld", type=int, nargs="+", default=[1024])
    ap.add_argument("--blocks", type=int, default=20, help="timed graph replays of 50 steps")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--buffer-fraction", type=float, default=1.0, help="hydroelastic stage buffers")
    ap.add_argument("--buffer-mult-broad", type=int, default=1)
    ap.add_argument("--broadphase", choices=["sap", "auto"], default="sap",
                    help="mjw: SAP_SEGMENTED (the earlier rows) or put_model's choice (NXN here, as in mjlab)")
    ap.add_argument("--nconmax", type=int, default=None, help="mjw: contacts per world (the broadphase fills it)")
    ap.add_argument("--njmax", type=int, default=None)
    ap.add_argument("--pad-d0", choices=["mjcf", "newton"], default="mjcf",
                    help="pads: solimp d0 from the MJCF compile's inverse weights or from Newton's model")
    a = ap.parse_args()
    MJW.update(broadphase=a.broadphase, nconmax=a.nconmax, njmax=a.njmax)
    xml, meta = scene_xml(a.tip)
    import mujoco_warp
    import warp
    for nw in a.nworld:
        row = {"mode": a.mode, "tip": a.tip, "nworld": nw, "hand": HAND, "plant": PLANT, "dt": DT, "iterations": ITER,
               "ls_iterations": LS, "impratio": IMPRATIO, "warp": warp.__version__,
               "mujoco_warp": getattr(mujoco_warp, "__version__", "?"), "when": time.strftime("%Y-%m-%d %H:%M"),
               "gpu_before_mb": gpu_used_mb()}
        try:
            r = run_mjw(xml, meta, nw, a.blocks) if a.mode == "mjw" else run_newton(
                xml, meta, nw, a.blocks, a.mode, a.tip, a.buffer_fraction, a.buffer_mult_broad, a.pad_d0)
            row.update(r, status="ok", gpu_used_mb=gpu_used_mb())
            try:
                import newton
                row["newton"] = newton.__version__
            except Exception:
                pass
            print(f"{a.mode:<9} {a.tip:<7} {nw:>5}: {r['us_per_world_step']:.2f} us/world-step, "
                  f"{r['contacts_per_world']:.1f} contacts/world, held {r['held_frac']:.2f}, "
                  f"GPU {row['gpu_used_mb']} MB", flush=True)
        except Exception as e:
            row.update(status="error", error=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-1500:])
            print(f"{a.mode} {a.tip} {nw}: {row['error']}", flush=True)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "a") as fh:
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
    return 0


if __name__ == "__main__":
    sys.exit(main())
