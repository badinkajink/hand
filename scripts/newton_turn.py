#!/usr/bin/env python3
"""The deployed open-loop three-finger turn in Newton with hydroelastic TPU fingertips (2026-10-06).

Same scenes, schedule, placements and scoring as the MuJoCo and Drake rows of reorient_backends.py on the working
plant (docs/experiments/20261006-hom_turn3/plans_mujoco.jsonl, plans_drake.jsonl): palm welded at the plan's pose,
the 25 mm tool on its post, the plan's grip 0.8 s, its 200-row joint trajectory, 1.5 s hold and 4 s more. Newton
1.7 SolverMuJoCo with Newton's collision pipeline; the printed TPU block (2.7 mm fillets) as one newton.Mesh per tip
with an SDF (0.5 mm voxels, +-6 mm band, Mesh.build_sdf), hydroelastic with kh = E/h = 1e7/8.5 mm = 1.18e9 N/m^3, the
tool cylinder hydroelastic at 100x; mu 1.0; servo kp 4 N m/rad, joint damping 0.08, 1 N m; 1 ms step, implicitfast,
elliptic cone, impratio 100, 100 solver / 50 line-search iterations. One world per placement: seed 0 at the plan's
pose, seeds 1-4 jittered 2 mm / 2 deg (reorient_backends.jitter).

    PY=logs/20261004-contact-transfer/venv/bin/python
    WARP_CACHE_PATH=$(mktemp -d) $PY scripts/newton_turn.py --hands D7 --seeds 0 1 2 3 4
Rows: docs/experiments/20261006-hom_turn3/plans_newton.jsonl, one fsynced line per hand x placement.

--mass-correct (rows plans_newton_mc.jsonl): SolverMuJoCo realises each hydroelastic contact's stiffness times the
tip-tool effective mass m_eff = 1 / (invweight0[tip] + invweight0[tool]) (contact_bed_newton.py, models *_mc). Each
tip's kh is multiplied by invweight0[tip] + invweight0[tool], read from solver.mj_model through
solver.mjc_body_to_newton, and the tool's kh is 100x the largest tip kh.

--contact pads (rows docs/experiments/20261006-simulator_agreement/plans_newton_pads.jsonl): the 1 mm sphere pads of
reorient_backends.replace_tips (1059 per tip, solimp d0 from the MJCF compile's inverse weights, which Newton's MuJoCo
model reproduces to 1e-7) in Newton's point-contact pipeline, the tool rigid. The pads collide with the tool alone
(scene model 'padsT'); the block mesh that takes the tips' other contacts in that scene is dropped, as Newton's
importer cannot load mesh files. Newton's importer gives every shape a friction gain kf 1000, which SolverMuJoCo turns
into a friction-row time constant of 4e-5 s for elliptic cones; the pads get the kf that restores their own 10 ms
(match_pad_friction).

Every row records the pads' normal force on the tool at the grip, hold and end marks (F_pads_*_N: the solver's
MuJoCo-Warp contacts between a fingertip body and the tool, elliptic-cone normal row efc_address[c, 0]) and the
fingers carrying more than 1e-6 N (pads_*). held_end counts those fingers, as reorient_backends.score does; rows
before 2026-10-07 counted fingers with any contact in Newton's list (fingers_end, kept).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import reorient_backends as RB  # noqa: E402

OUT = ROOT / "docs/experiments/20261006-hom_turn3/plans_newton.jsonl"
OUT_MC = ROOT / "docs/experiments/20261006-hom_turn3/plans_newton_mc.jsonl"
OUT_PADS = ROOT / "docs/experiments/20261006-simulator_agreement/plans_newton_pads.jsonl"
SCENES = ROOT / "logs/20261006-hom_turn3/scenes"
PLANT = "kp4_kv0_fr1_fl0_dp0.08"
KH_TIP = 1e7 / 0.0085
SDF_VOXEL, SDF_BAND = 0.0005, 0.006
DT, BLOCK = 0.001, 10
TOOL = "screwdriver_medium"


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


def quat_wxyz_to_xyzw(q):
    return np.array([q[1], q[2], q[3], q[0]], float)


NUM = dict(dt=0.001, iterations=100, ls_iterations=50, impratio=100.0, tolerance=1e-10)
OPT = dict(mass_correct=False, contact="hydro")
NCON = {"hydro": (512, 1536), "pads": (640, 3072)}   # MuJoCo-Warp contacts and constraint rows per world


def mass_correct(model, solver):
    """Multiply each tip's hydroelastic kh by 1/m_eff = invweight0[tip] + invweight0[tool] of the solver's own MuJoCo
    model; the tool's kh becomes 100x the largest tip kh. Returns the per-finger inverse weights and kh."""
    mjm = solver.mj_model
    labels = [x.split("/")[-1] for x in model.body_label]
    m2n = solver.mjc_body_to_newton.numpy()[0]          # MuJoCo body -> Newton body, world 0
    w = {}
    for bi in range(mjm.nbody):
        nb = int(m2n[bi])
        nm = labels[nb] if 0 <= nb < len(labels) else ""
        if nm in (TOOL,) + tuple(f"{f}_tip" for f in RB.FINGERS):
            w[nm] = float(mjm.body_invweight0[bi, 0])
            w[nm + "_mass"] = float(mjm.body_mass[bi])
    kh_tip = {f: KH_TIP * (w[f"{f}_tip"] + w[TOOL]) for f in RB.FINGERS}
    kh_tool = 100.0 * max(kh_tip.values())
    shape_body = model.shape_body.numpy()
    flags = model.shape_flags.numpy()
    kh = model.shape_material_kh.numpy()
    for i in range(len(kh)):
        b = int(shape_body[i])
        if b < 0 or not flags[i] & int(__import__("newton").ShapeFlags.HYDROELASTIC):
            continue
        nm = labels[b]
        if nm == TOOL:
            kh[i] = kh_tool
        elif nm.endswith("_tip"):
            kh[i] = kh_tip[nm[:-4]]
    model.shape_material_kh.assign(kh)                 # the hydroelastic pipeline reads this array
    return {"invweight0": {k: v for k, v in w.items() if not k.endswith("_mass")},
            "body_mass": {k[:-5]: v for k, v in w.items() if k.endswith("_mass")},
            "kh_tip": kh_tip, "kh_tool": kh_tool}


def build(hand, seeds):
    import newton
    import warp as wp  # noqa: F401
    from newton.geometry import HydroelasticSDF
    pads = OPT["contact"] == "pads"
    path, meta = RB.build_scene(hand, "tpu2.7", "padsT" if pads else "pt", PLANT, "bed", 100.0, 1.0,
                                out_dir=SCENES)
    root = ET.parse(path).getroot()
    for b in root.iter("body"):
        if b.get("name", "").endswith("_tip"):
            for g in list(b.findall("geom")):
                if g.get("type") == "mesh":
                    b.remove(g)
    xml = re.sub(r'solref="([0-9.eE+-]+)"', r'solref="\1 1"', ET.tostring(root, encoding="unicode"))
    b0 = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    b0.add_mjcf(xml, ctrl_direct=True, parse_sites=False, parse_visuals=False)
    for i in range(len(b0.shape_gap)):
        b0.shape_gap[i] = 0.0005
        b0.shape_margin[i] = 0.0
    for i, lab in enumerate(b0.joint_label):
        n = lab.split("/")[-1]
        if n in meta["q0"]:
            b0.joint_q[b0.joint_q_start[i]] = float(meta["q0"][n])
    body_of = {x.split("/")[-1]: k for k, x in enumerate(b0.body_label)}
    cfg = newton.ModelBuilder.ShapeConfig(margin=0.0, gap=0.0, mu=1.0, kh=KH_TIP, density=1000.0,
                                          is_hydroelastic=True, mu_torsional=0.0, mu_rolling=0.0)
    for f, sign in () if pads else (("thumb", 1.0), ("index", -1.0), ("middle", -1.0)):
        V, F = read_obj(meta["meshes"][str(sign)])
        mesh = newton.Mesh(V, F)
        mesh.build_sdf(target_voxel_size=SDF_VOXEL, narrow_band_range=(-SDF_BAND, SDF_BAND), margin=SDF_BAND)
        k = b0.add_shape_mesh(body_of[f"{f}_tip"], mesh=mesh, cfg=cfg, label=f"{f}_tpu")
        b0.shape_material_kf[k] = 10.0
    for i in range(len(b0.shape_label)):
        if b0.shape_body[i] == body_of[TOOL] and not pads:
            b0.shape_flags[i] |= int(newton.ShapeFlags.HYDROELASTIC)
            b0.shape_sdf_target_voxel_size[i] = SDF_VOXEL
            b0.shape_sdf_narrow_band_range[i] = (-SDF_BAND, SDF_BAND)
            b0.shape_sdf_padding[i] = SDF_BAND
            b0.shape_margin[i] = 0.0
            b0.shape_gap[i] = 0.0
            b0.shape_material_kh[i] = KH_TIP * 100.0
            b0.shape_material_mu[i] = 1.0
            b0.shape_material_kf[i] = 10.0
    scene = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    scene.replicate(b0, len(seeds))
    model = scene.finalize(device="cuda:0")
    nconmax, njmax = NCON[OPT["contact"]]
    kw = {} if pads else {"sdf_hydroelastic_config": HydroelasticSDF.Config(reduce_contacts=True, anchor_contact=True)}
    pipe = newton.CollisionPipeline(model, reduce_contacts=True, rigid_contact_max=len(seeds) * nconmax,
                                    broad_phase="explicit", **kw)
    solver = newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=False, disable_sensors=True, njmax=njmax,
                                         nconmax=nconmax, iterations=NUM["iterations"], ls_iterations=NUM["ls_iterations"],
                                         cone="elliptic", integrator="implicitfast", tolerance=NUM["tolerance"],
                                         impratio=NUM["impratio"])
    # per-world tool placement
    labels_j = [x.split("/")[-1] for x in model.joint_label]
    qs = model.joint_q_start.numpy()
    jq = model.joint_q.numpy()
    tool_j = [k for k, x in enumerate(labels_j) if x.startswith(TOOL)]
    tool7 = np.asarray(meta["tool7"], float)
    for w, (k, seed) in enumerate(zip(tool_j, seeds)):
        dx, dy, dyaw = RB.jitter(seed)
        q = RB._yaw_quat(tool7[3:], dyaw)
        a = qs[k]
        jq[a:a + 3] = tool7[:3] + [dx, dy, 0.0]
        jq[a + 3:a + 7] = quat_wxyz_to_xyzw(q / np.linalg.norm(q))
    model.joint_q.assign(jq)
    meta["mass_correct"] = mass_correct(model, solver) if OPT["mass_correct"] else None
    if pads:
        mjm = solver.mj_model
        gi = [g for g in range(mjm.ngeom) if "_pad" in (mjm.geom(g).name or "")]
        meta["pad_solimp_solver"] = [float(x) for x in mjm.geom_solimp[gi[0]]] if gi else None
        meta["n_pad_geoms"] = len(gi)
        meta["pad_kf"] = match_pad_friction(model, solver, meta)
    return model, pipe, solver, meta


def match_pad_friction(model, solver, meta, tc=0.01):
    """Give each finger's pads the friction gain kf for which SolverMuJoCo's elliptic-cone mapping (friction time
    constant 2 / (kf w ((1 - d0) / impratio + d0)), w = invweight0[tip] + invweight0[tool]) reproduces the pads' own
    solref time constant tc, MuJoCo's default for friction rows. The pads have priority 1, so their kf alone sets the
    pad-tool contact; Newton's importer gives every shape kf 1000 (time constant 4e-5 s)."""
    mjm = solver.mj_model
    labels = [x.split("/")[-1] for x in model.body_label]
    m2n = solver.mjc_body_to_newton.numpy()[0]
    w = {}
    for bi in range(mjm.nbody):
        nb = int(m2n[bi])
        if 0 <= nb < len(labels):
            w[labels[nb]] = float(mjm.body_invweight0[bi, 0])
    kf_f = {}
    for f in RB.FINGERS:
        d0 = float(meta["pad_d0"][f])
        kf_f[f] = 2.0 / (tc * (w[f"{f}_tip"] + w[TOOL]) * ((1.0 - d0) / NUM["impratio"] + d0))
    shape_lab = [x.split("/")[-1] for x in model.shape_label]
    kf = model.shape_material_kf.numpy()
    for i, nm in enumerate(shape_lab):
        for f in RB.FINGERS:
            if nm.startswith(f"{f}_pad"):
                kf[i] = kf_f[f]
    model.shape_material_kf.assign(kf)
    return kf_f


def run(hand, seeds):
    import newton
    import warp as wp
    model, pipe, solver, meta = build(hand, seeds)
    nw = len(seeds)
    s0, s1 = model.state(), model.state()
    ctrl = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    cc = pipe.contacts()
    mm = solver.mj_model
    full = [mm.joint(int(mm.actuator_trnid[a, 0])).name for a in range(mm.nu)]
    names = [next(n for n in (f"{f}_{j}" for f in RB.FINGERS for j in RB.JOINTS) if fn.endswith(n)) for fn in full]
    labels = [x.split("/")[-1] for x in model.body_label]
    tool_ids = np.array([i for i, x in enumerate(labels) if x == TOOL])
    body_finger = np.array([next((f for f in RB.FINGERS if x.startswith(f + "_")), "") for x in labels])
    shape_body = model.shape_body.numpy()
    st = {"s0": s0, "s1": s1}

    def block():
        for _ in range(BLOCK):
            st["s0"].clear_forces()
            pipe.collide(st["s0"], cc)
            solver.step(st["s0"], st["s1"], ctrl, cc, NUM["dt"])
            st["s0"], st["s1"] = st["s1"], st["s0"]

    def set_targets(tgt):
        row = np.array([tgt[n] for n in names], np.float32)
        ctrl.mujoco.ctrl.assign(np.tile(row, nw).reshape(ctrl.mujoco.ctrl.shape))

    def snap():
        bq = st["s0"].body_q.numpy()[tool_ids]
        out = []
        for w in range(nw):
            x, y, z, qx, qy, qz, qw = bq[w]
            cos = 1.0 - 2.0 * (qx * qx + qy * qy)          # R[2,2] of the tool's frame
            out.append((float(x), float(y), float(z), float(cos)))
        return out

    # the solver's MuJoCo-Warp contacts: geom -> finger index (0-2) or the tool (3)
    mjm = solver.mj_model
    m2n = solver.mjc_body_to_newton.numpy()[0]          # MuJoCo body -> Newton body, world 0 (names are paths)
    geom_cls = np.full(mjm.ngeom, -1)
    for gi in range(mjm.ngeom):
        nb = int(m2n[int(mjm.geom_bodyid[gi])])
        bn = labels[nb] if 0 <= nb < len(labels) else ""
        if bn == TOOL:
            geom_cls[gi] = 3
        else:
            geom_cls[gi] = next((k for k, f in enumerate(RB.FINGERS) if bn.startswith(f + "_")), -1)
    md = solver.mjw_data
    njmax_ = md.efc.force.shape[1]

    def pad_forces():
        """(nworld, 3) normal force of each finger on the tool, N."""
        n = int(md.nacon.numpy()[0])
        g = md.contact.geom.numpy()[:n]
        wid = md.contact.worldid.numpy()[:n]
        adr = md.contact.efc_address.numpy()[:n, 0]
        ef = md.efc.force.numpy()
        F = np.zeros((nw, 3))
        c0, c1 = geom_cls[g[:, 0]], geom_cls[g[:, 1]]
        for k0, k1, w, a in zip(c0, c1, wid, adr):
            if a < 0 or a >= njmax_ or not ((k0 == 3) ^ (k1 == 3)):
                continue
            k = k1 if k0 == 3 else k0
            if k >= 0:
                F[w, k] += max(0.0, float(ef[w, a]))
        return F

    def fingers_touching():
        n = int(cc.rigid_contact_count.numpy()[0])
        a = cc.rigid_contact_shape0.numpy()[:n]
        b = cc.rigid_contact_shape1.numpy()[:n]
        touch = [set() for _ in range(nw)]
        for i, j in zip(a, b):
            bi, bj = shape_body[i], shape_body[j]
            for t, o in ((bi, bj), (bj, bi)):
                if t in tool_ids and o >= 0 and body_finger[o]:
                    touch[int(np.where(tool_ids == t)[0][0])].add(body_finger[o])
        return [len(s) for s in touch]

    plan, traj = RB.load_plan(hand)
    segs = RB.schedule(plan, traj)
    set_targets(segs[0][1])
    block()                                            # compile the kernels outside the capture
    hs = getattr(pipe, "hydroelastic_sdf", None)
    if hs is not None:
        hs._host_warning_poll_interval = 10 ** 12
    wp.synchronize()
    with wp.ScopedCapture() as cap:
        block()
    t = BLOCK * NUM["dt"]                              # the compile block ran; the capture does not step
    t_traj_end = [e for e, _, k in segs if k == "traj"][-1]
    marks = {"grip": RB.T_GRIP, "hold": t_traj_end + RB.T_HOLD}
    got, forces = {}, {}
    w0 = time.perf_counter()
    for t_end, tgt, _ in segs:
        set_targets(tgt)
        while t < t_end - 1e-9:
            wp.capture_launch(cap.graph)
            t += BLOCK * NUM["dt"]
            for k, tm in marks.items():
                if k not in got and t >= tm - 1e-9:
                    got[k] = snap()
                    forces[k] = pad_forces()
    wp.synchronize()
    wall = time.perf_counter() - w0
    end = snap()
    forces["end"] = pad_forces()
    nf = fingers_touching()
    rows = []
    for w, seed in enumerate(seeds):
        g, h, e = got["grip"][w], got["hold"][w], end[w]

        def turn(a, b):
            return math.degrees(math.acos(max(-1, min(1, a[3]))) - math.acos(max(-1, min(1, b[3]))))
        dropped = e[2] < g[2] - 0.020
        npad = {k: int(np.sum(forces[k][w] > 1e-6)) for k in ("grip", "hold", "end")}
        Fp = {k: round(float(np.sum(forces[k][w])), 4) for k in ("grip", "hold", "end")}
        rows.append({"hand": hand, "seed": seed, "sim": "newton", "tip": "tpu2.7",
                     "contact": "pads" if OPT["contact"] == "pads" else "hydroelastic",
                     "F_pads_grip_N": Fp["grip"], "F_pads_hold_N": Fp["hold"], "F_pads_end_N": Fp["end"],
                     "F_finger_hold_N": dict(zip(RB.FINGERS, (round(float(x), 4) for x in forces["hold"][w]))),
                     "pads_hold": npad["hold"], "pads_end": npad["end"],
                     "held_hold": bool(h[2] >= g[2] - 0.020 and npad["hold"] >= 2),
                     "dt": NUM["dt"], "iterations": NUM["iterations"], "impratio": NUM["impratio"],
                     "plant": PLANT, "mu": 1.0, "cos_grip": g[3], "cos_hold": h[3], "cos_end": e[3],
                     "turn_hold_deg": turn(g, h), "turn_end_deg": turn(g, e), "z_grip": g[2], "z_end": e[2],
                     "fingers_end": nf[w], "dropped_end": bool(dropped), "held_end": (not dropped) and npad["end"] >= 2,
                     "wall_s_batch": round(wall, 1), "n_worlds": nw, "jitter": list(RB.jitter(seed)),
                     "mass_correct": meta["mass_correct"], "pad_solimp_solver": meta.get("pad_solimp_solver"),
                     "n_pad_geoms": meta.get("n_pad_geoms"), "pad_d0_mjcf": meta.get("pad_d0"),
                     "pad_kf": meta.get("pad_kf"),
                     "script": "scripts/newton_turn.py"})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hands", nargs="+", default=list(RB.HANDS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", type=Path, default=None, help="default plans_newton.jsonl, or plans_newton_mc.jsonl")
    ap.add_argument("--mass-correct", action="store_true", help="divide each tip's kh by the tip-tool m_eff")
    ap.add_argument("--contact", choices=["hydro", "pads"], default="hydro",
                    help="hydro: the TPU block as one hydroelastic mesh per tip; pads: 1 mm sphere pads")
    ap.add_argument("--rl-numerics", action="store_true",
                    help="the RL trainer's 2 ms step, 10/20 iterations, impratio 10, tolerance 1e-8")
    a = ap.parse_args()
    OPT["mass_correct"] = a.mass_correct
    OPT["contact"] = a.contact
    if a.contact == "pads" and a.mass_correct:
        ap.error("--mass-correct applies to the hydroelastic tips; the pads carry the correction in their solimp")
    if a.out is None:
        a.out = OUT_PADS if a.contact == "pads" else OUT_MC if a.mass_correct else OUT
    if a.rl_numerics:
        NUM.update(dt=0.002, iterations=10, ls_iterations=20, impratio=10.0, tolerance=1e-8)
    import newton
    for hand in a.hands:
        try:
            rows = run(hand, a.seeds)
        except Exception as e:
            rows = [{"hand": hand, "sim": "newton", "status": "error", "error": f"{type(e).__name__}: {e}",
                     "tb": traceback.format_exc()[-1500:]}]
        with open(a.out, "a") as fh:
            for r in rows:
                r.setdefault("status", "complete")
                r["newton"] = newton.__version__
                r["when"] = time.strftime("%Y-%m-%d %H:%M")
                fh.write(json.dumps(r) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        for r in rows:
            print(f"{hand} seed {r.get('seed')}: " + (f"turn {r['turn_end_deg']:+.1f} deg, held {r['held_end']}, F_hold {r['F_pads_hold_N']:.2f} N, "
                  f"fingers {r['fingers_end']}, z {1e3 * r['z_grip']:.1f} -> {1e3 * r['z_end']:.1f} mm, "
                  f"{r['wall_s_batch']} s" if r["status"] == "complete" else r["error"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
