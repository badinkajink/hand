#!/usr/bin/env python3
"""Steel (Skild AI's PGS solver in Newton) and Newton's hydroelastic fingertip contact against MuJoCo with the sphere
pads, on the twist of the inner-loop benchmark (2026-11-13; owner: "if steel has hydroelastic contact built in, and it
is that much faster than warp, then i think we do have to compare to them").

Hand, object and plant are the benchmark's (scripts/hom/inner_loop_plant.py): the cad_v2 SR2 hand with the v4b tips,
the 40 mm sphere, the SCS0009 servo plant (0.72 N m/rad, 0.18 N m, damping 0.0144 N m s/rad), 1 ms physics, scene
friction 1.0, the scene cache logs/20261102-hom_servo_plant/scenes, the twenty (layout, grasp) pairs of
docs/experiments/20261112-inner_loop_benchmark/rows/mpc_pairs_twist.json.

Plants (--plant):
    mjc_pads           MuJoCo CPU, the 1 mm sphere pads at 2.5 MPa (the benchmark's model; hom_tasks.Plant)
    mjc_pt             MuJoCo CPU, point contact on the v4b tip's convex mesh
    nt_mjc_hydro       Newton's SolverMuJoCo (MuJoCo-Warp) with the tip meshes and the sphere as hydroelastic shapes
    steel_hydro        SolverSteel, the same hydroelastic contacts as implicit compliant rows (contact_compliance)
    steel_hydro_hard   SolverSteel, hard contact rows at the hydroelastic contact points
    steel_pt           SolverSteel, point contact on the tip mesh (Steel's default persistent friction patches)
    steel_pt_pf        SolverSteel, point contact with point friction (friction_anchor_beta 0)
Hydroelastic pressure: p = kh x depth, kh = E / h on the tips (E 2.5 MPa, h 8.5 mm, the pad's depth semi-axis), 100 kh
on the sphere; SDFs at 0.5 mm voxels and a +-6 mm band (the contact bed's, 2026-10-06).

Commands:
    hom     HOM (hom_tasks.rollout, the CPU controller of hom_gpu.cpu_rollout, variant base) through the twist on a plant,
            per (pair, seed): J (hom_landscape.track_score), held, the metrics, the wall time, and the trajectory
            (object position and orientation, pad forces) per control period in logs/20261113-steel/traj/<plant>/
    speed   world-steps per second from the squeezed grasp at the task's start (t = 2.4 s, the state of the mjc_pads
            run of the first pair), nworld copies, targets held, gravity on, post removed: us per world-step, contacts
            per world, GPU memory, held share and object drift after 1 s
Rows: docs/experiments/20261113-steel_solver/rows/{hom_<plant>,speed}.jsonl, appended and fsynced per row.

    PY=.venv/bin/python                         # MuJoCo 3.6, MuJoCo-Warp 3.6: the mjc_* and mjw_* rows
    PYS=logs/20261113-steel/venv/bin/python     # skild-ai/newton f6fc82a8 (Newton 1.9.0.dev0, Warp 1.19, MuJoCo 3.17)
    $PY scripts/contact/steel_hand_compare.py hom --plant mjc_pads --seeds 0 1
    WARP_CACHE_PATH=$(mktemp -d) $PYS scripts/contact/steel_hand_compare.py hom --plant steel_hydro --seeds 0 1
    WARP_CACHE_PATH=$(mktemp -d) $PYS scripts/contact/steel_hand_compare.py speed --plant steel_hydro --nworld 16 256 1024
"""
from __future__ import annotations

import os

os.environ.setdefault("SCALE_HAND", "cad_v2")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import xml.etree.ElementTree as ET  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path
import inner_loop_plant  # noqa: E402,F401  the benchmark's plant and scene cache, before hom_gpu
import hom_gpu as HG  # noqa: E402
import hom_landscape as L  # noqa: E402
import hom_tasks as T  # noqa: E402
from sr2hand.geometry.rotations import quat_mul, quat_yaw  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/experiments/20261113-steel_solver"
ROWS = OUT / "rows"
LOGS = ROOT / "logs/20261113-steel"
PAIRS = ROOT / "docs/experiments/20261112-inner_loop_benchmark/rows/mpc_pairs_twist.json"
TASK, D_MM, VAR = "twist", 40.0, "base"
FINGERS, JOINTS = HG.FINGERS, HG.JOINTS
PHYS_DT = 0.001
T_TASK = 2.4                                  # s: the task phase of the benchmark starts here
E_TIP, H_TIP = 2.5e6, 0.0085                  # Pa, m: the pads' modulus and the v4b pad's depth semi-axis
KH_TIP = E_TIP / H_TIP                        # N/m^3
KH_OBJ = 100.0 * KH_TIP
SDF_VOXEL, SDF_BAND = 0.0005, 0.006
GAP = 0.0005                                  # m: point-contact gap (Newton's default reports pairs centimetres apart)
TAU_MAX = 0.18                                # N m: the SCS0009's force range in the scene (ctrlrange +-1 rad)
PLANTS = ("mjc_pads", "mjc_pt", "nt_mjc_hydro", "steel_hydro", "steel_hydro_hard", "steel_pt", "steel_pt_pf")


# ------------------------------------------------------------------------------------------------ worlds
def worlds(seeds, n_pairs=None):
    """The benchmark's pairs at the seeds, as inner_loop_mpc.worlds_of builds them: [(pair, key, mxy, setup, q_open,
    seed)] with the reference scenes of hom_gpu.Scene."""
    prs = json.loads(PAIRS.read_text())["pairs"][:n_pairs]
    scene = HG.Scene(T.SHAPE[TASK], D_MM)
    want = {(p["x"], p["y"], p["grasp"]): p for p in prs}
    found = {}
    for key, mxy, setup in HG.layout_list(TASK, "grid2dff_size", 0, D_MM):
        d = key["d"]
        k = (round(float(d[2] - d[0]), 3), round(float(d[3] - d[5]), 3), int(key.get("grasp", 0)))
        if k in want and k not in found:
            _bm, q_open = scene.prepare(mxy, setup)
            found[k] = (key, mxy, setup, q_open)
    missing = [k for k in want if k not in found]
    if missing:
        raise SystemExit(f"pairs not in the grid: {missing}")
    out = []
    for p in prs:
        key, mxy, setup, q_open = found[(p["x"], p["y"], p["grasp"])]
        for sd in seeds:
            out.append((p, key, mxy, setup, q_open, sd))
    return scene, out


def pair_name(p):
    return f"x{p['x']:g}_y{p['y']:g}_g{p['grasp']}"


def append(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def done(path):
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("status") == "ok":
                out.add((r["pair"], r["seed"]))
    return out


def gpu_used_mb():
    try:
        return int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True).stdout.split()[0])
    except Exception:
        return None


def host_rss_mb():
    """This process's peak resident set (MB)."""
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


# ------------------------------------------------------------------------------------------------ Newton plant
def pair_xml(scene, bm, palm):
    """The point-contact scene of the reference layout with this pair's palm pose and finger mounts written into the
    MJCF (hom_gpu.cpu_rollout moves the same bodies in the compiled model); checked against that route."""
    import mujoco
    root = ET.parse(scene.pt_path).getroot()
    pos = {"palm_pose": np.asarray(palm, float), **{f"{f}_mount": np.asarray(bm[f], float) for f in FINGERS}}
    for b in root.iter("body"):
        n = b.get("name")
        if n in pos:
            b.set("pos", " ".join(f"{v:.9g}" for v in pos[n]))
    xml = ET.tostring(root, encoding="unicode")
    m1 = mujoco.MjModel.from_xml_string(xml)
    m2 = mujoco.MjModel.from_xml_path(str(scene.pt_path))
    T.S.set_palm(m2, palm)
    for f in FINGERS:
        m2.body_pos[m2.body(f"{f}_mount").id] = bm[f]
    if not np.allclose(m1.body_pos, m2.body_pos, atol=1e-9):
        raise RuntimeError("the edited MJCF does not reproduce the compiled model's body positions")
    return xml


def object_only(m):
    """MuJoCo's contact masks with every geom off the object meeting the object only (contype 2, conaffinity 0), the
    object's own masks kept: the contact pairs of the Newton plants with --contact-pairs obj."""
    tool = np.flatnonzero(m.geom_bodyid == m.body(T.S.OBJ).id)
    other = np.setdiff1d(np.flatnonzero((m.geom_contype != 0) | (m.geom_conaffinity != 0)), tool)
    m.geom_contype[other], m.geom_conaffinity[other] = 2, 0
    m.geom_conaffinity[tool] |= 2
    return tool


def object_pairs(xml, lab):
    """The pairs of the object's geoms with every other geom that has contact enabled (bodies not welded together), as
    index pairs of the Newton builder's shapes (labels `lab`). Newton's own sizing over all 1,348 shapes asked for 9.3 GB
    at 256 worlds; the benchmark GPU plant's filter (hom_gpu.Batch) still leaves 81,588 pairs per world, most of them
    between the fingers' link spheres (2026-11-13)."""
    import mujoco
    m = mujoco.MjModel.from_xml_string(xml)
    coll = np.flatnonzero((m.geom_contype != 0) | (m.geom_conaffinity != 0))
    if len(coll) != len(lab):
        raise RuntimeError(f"{len(coll)} colliding geoms against {len(lab)} Newton shapes")
    # Newton imports the colliding geoms under their names and names an unnamed one <body>_geom_<k>
    byname = {n: i for i, n in enumerate(lab)}
    index = {}
    for g in coll:
        name = m.geom(g).name
        if not name:
            body = m.body(m.geom_bodyid[g]).name
            cand = sorted(n for n in lab if n.startswith(f"{body}_geom_"))
            k = [int(h) for h in coll if m.geom_bodyid[h] == m.geom_bodyid[g] and not m.geom(h).name].index(int(g))
            name = cand[k] if k < len(cand) else None
        if name not in byname:
            raise RuntimeError(f"geom {g} ({m.geom(g).name!r}) has no Newton shape")
        index[int(g)] = byname[name]
    tool = object_only(m)
    wt = m.body_weldid[m.body(T.S.OBJ).id]
    out = [(index[int(t)], index[int(g)]) for t in tool for g in coll
           if g not in tool and m.body_weldid[m.geom_bodyid[g]] != wt]
    return np.array(out, np.int32).reshape(-1, 2)


def world_builder(newton, xml, plant, pairs="obj"):
    """One world's builder from a pair's MJCF with the plant's contact and servo settings, and its index maps. With pairs
    "obj" the mesh geoms other than the hydroelastic tips collide as their convex hulls, as in MuJoCo; Newton's importer
    keeps them triangle meshes (stage 1), and a 40 mm sphere among them met 17,000 triangles per world."""
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    if plant.startswith("steel"):
        newton.solvers.SolverSteel.register_custom_attributes(b)
    b.add_mjcf(xml, parse_sites=False, parse_visuals=False)
    lab = [x.split("/")[-1] for x in b.shape_label]
    blab = [x.split("/")[-1] for x in b.body_label]
    obj_body = blab.index(T.S.OBJ)
    tips = [i for i, n in enumerate(lab) if n.endswith("_tipgeom")]
    obj_shapes = [i for i in range(len(lab)) if b.shape_body[i] == obj_body]
    if pairs == "obj":
        keep = set(tips + obj_shapes) if "hydro" in plant else set()
        meshes = [i for i in range(len(lab)) if b.shape_type[i] == newton.GeoType.MESH and i not in keep]
        b.approximate_meshes("convex_hull", shape_indices=meshes, raise_on_failure=True)
        if len(b.shape_label) != len(lab):
            raise RuntimeError("the convex hulls changed the shape count")
    for i in range(len(lab)):
        b.shape_margin[i] = 0.0
        b.shape_gap[i] = GAP
        b.shape_material_mu[i] = 1.0
    if "hydro" in plant:
        for i in tips + obj_shapes:
            b.shape_flags[i] |= int(newton.ShapeFlags.HYDROELASTIC)
            b.shape_sdf_target_voxel_size[i] = SDF_VOXEL
            b.shape_sdf_narrow_band_range[i] = (-SDF_BAND, SDF_BAND)
            b.shape_sdf_padding[i] = SDF_BAND
            b.shape_gap[i] = 0.0
            b.shape_material_kh[i] = KH_OBJ if i in obj_shapes else KH_TIP
            if not plant.startswith("steel"):
                # SolverMuJoCo's friction rows: the hydroelastic gain of 2026-10-05/06 (kf 10, a 3.9 ms row; the
                # importer's kf 1000 gives 4e-5 s rows, docs/experiments/20261006-simulator_agreement)
                b.shape_material_kf[i] = 10.0
    # the servos: MuJoCo's position actuators (kp 0.72, force range 0.18 N m) as joint drives clamped at the force range
    # (drive_mode rows bounds the whole drive reaction, as MuJoCo clamps the actuator force)
    jl = [x.split("/")[-1] for x in b.joint_label]
    dof = {}
    for f in FINGERS:
        for j in JOINTS:
            ji = jl.index(f"{f}_{j}")
            d = b.joint_qd_start[ji]
            dof[(f, j)] = (b.joint_q_start[ji], d)
            b.joint_effort_limit[d] = TAU_MAX
            b.joint_target_kd[d] = 0.0
    free_q = b.joint_q_start[jl.index(next(n for n in jl if n.endswith("freejoint")))]
    return b, {"dof": dof, "free_q": free_q, "obj_body": obj_body, "nbody": len(blab), "nshape": len(lab),
               "post": lab.index("post"), "blab": blab, "pairs": object_pairs(xml, lab)}


class NewtonBatch:
    """n worlds of the point-contact scene in one Newton model (one MJCF per world, or one replicated), stepped by
    SolverMuJoCo or SolverSteel. Host copies of the joint state, the targets and the shape transforms are pushed before
    each stretch of steps and read back after it; World views give each world hom_tasks.Plant's interface."""

    def __init__(self, xmls, plant, steel_iters=None, drive_mode="rows", capture=True, hydro_buf=(1.0, 1),
                 pairs="obj"):
        import newton
        import warp as wp
        self.newton, self.wp, self.plant, self.n = newton, wp, plant, len(xmls)
        self.hydro = "hydro" in plant
        scene = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        if plant.startswith("steel"):
            newton.solvers.SolverSteel.register_custom_attributes(scene)
        if all(x == xmls[0] for x in xmls):
            b0, info = world_builder(newton, xmls[0], plant, pairs)
            scene.replicate(b0, self.n)
            local = [info["pairs"]] * self.n
        else:
            local = []
            for x in xmls:
                b0, info = world_builder(newton, x, plant, pairs)
                scene.add_world(b0)
                local.append(info["pairs"])
        self.info = info
        self.dof, self.free_q, self.obj_body = info["dof"], info["free_q"], info["obj_body"]
        self.model = model = scene.finalize(device="cuda:0" if wp.is_cuda_available() else "cpu")
        self.dev = model.device
        n = self.n
        self.nq, self.nqd = model.joint_coord_count // n, model.joint_dof_count // n
        self.nb, self.ns = model.body_count // n, model.shape_count // n
        if self.nb != info["nbody"] or self.ns != info["nshape"] or model.shape_count % n:
            raise RuntimeError("the worlds differ in their bodies or shapes")
        blab = info["blab"]
        sb = model.shape_body.numpy()
        self.shape_world = model.shape_world.numpy()
        self.shape_body = sb
        self.fidx = -np.ones(model.shape_count, int)
        self.tidx = -np.ones(model.shape_count, int)
        self.obj_mask = np.zeros(model.shape_count, bool)
        for sh in range(model.shape_count):
            b = int(sb[sh])
            if b < 0:
                continue
            bn = blab[b % self.nb]
            self.obj_mask[sh] = (b % self.nb) == self.obj_body
            for k, f in enumerate(FINGERS):
                if bn.startswith(f + "_"):
                    self.fidx[sh] = k
                    if bn == f"{f}_tip":
                        self.tidx[sh] = k
        self.post = np.array([w * self.ns + info["post"] for w in range(n)])
        cmax = 1024 if self.hydro else 256
        kw = {}
        # pairs "obj": the object's pairs (object_pairs) through the explicit broad phase, and a triangle-pair budget
        # of 4,096 per world (with the convex hulls of world_builder no pair reaches the triangle path); "sap": sweep
        # and prune over all shapes with Newton's own sizing (stage 1)
        self.pairs_mode = pairs
        if pairs == "obj":
            allp = np.concatenate([lp + w * self.ns for w, lp in enumerate(local)]).astype(np.int32)
            self.npairs_world = len(local[0])
            kw.update(broad_phase="explicit", shape_pairs_filtered=wp.array(allp, dtype=wp.vec2i, device=self.dev),
                      max_triangle_pairs=4096 * n)
        else:
            kw.update(broad_phase="sap")
        if self.hydro:
            from newton.geometry import HydroelasticSDF
            kw["sdf_hydroelastic_config"] = HydroelasticSDF.Config(reduce_contacts=True, anchor_contact=True,
                                                                   buffer_fraction=float(hydro_buf[0]),
                                                                   buffer_mult_broad=int(hydro_buf[1]))
        self.pipe = newton.CollisionPipeline(model, reduce_contacts=True, rigid_contact_max=n * cmax, **kw)
        self.contacts_buf = self.pipe.contacts()
        if plant.startswith("steel"):
            # row capacities of Skild's Panda hydroelastic preset (the default 32 dense rows per world dropped 148 of
            # the grasp's 180 rows and the sphere fell)
            opts = dict(enable_joint_limits=True, drive_mode=drive_mode, enable_restitution=False,
                        articulation_constraint_max=1536, free_body_constraint_max=1024, warn_constraint_overflow=True)
            if plant == "steel_hydro":
                opts.update(contact_compliance=True, friction_anchor_beta=0.0, solve_mode="colored",
                            iterations=steel_iters or 16, articulated_contact_response="propagation")
            elif plant in ("steel_hydro_hard", "steel_pt_pf"):
                opts.update(friction_anchor_beta=0.0, iterations=steel_iters or 8)
            else:
                opts.update(iterations=steel_iters or 8)
            self.solver = newton.solvers.SolverSteel(model, **opts)
        else:
            opts = dict(use_mujoco_contacts=False, disable_sensors=True, njmax=2048, nconmax=1024, iterations=20,
                        ls_iterations=20, cone="elliptic", integrator="implicitfast", tolerance=1e-6, impratio=100.0)
            self.solver = newton.solvers.SolverMuJoCo(model, **opts)
        self.solver_opts = opts
        self.obs = self.solver.observables([self.solver.ObservableKind.CONTACT_F], contacts=self.contacts_buf)
        self.s0, self.s1 = model.state(), model.state()
        self.ctrl = model.control()
        self.capture = capture and wp.get_device(self.dev).is_cuda
        self.graph = None
        self.time = 0.0
        self.q = self.s0.joint_q.numpy().reshape(n, self.nq).copy()
        self.qd = np.zeros((n, self.nqd))
        self.tgt = self.ctrl.joint_target_q.numpy().reshape(n, self.nq).copy()
        self.st = model.shape_transform.numpy().copy()
        self.st0 = self.st.copy()
        self.dirty_q = self.dirty_st = True
        self.bq = self.s0.body_q.numpy().reshape(n, self.nb, 7)
        self.bqd = np.zeros((n, self.nb, 6))
        self.cop = [dict() for _ in range(n)]
        self.Ftot = [{f: 0.0 for f in FINGERS} for _ in range(n)]
        self.snaps = [None] * n
        self.nan_t = [None] * n

    # -------------------------------------------------------------- stepping
    def _steps(self, k):
        for _ in range(k):
            self.s0.clear_forces()
            self.pipe.collide(self.s0, self.contacts_buf)
            self.solver.step(self.s0, self.s1, self.ctrl, self.contacts_buf, PHYS_DT, observables=self.obs)
            self.s0, self.s1 = self.s1, self.s0

    def run(self, k):
        """k physics steps; with capture, blocks of 10 steps (even, so s0 and s1 return to their arrays) replayed as one
        CUDA graph; the first block runs eagerly (Skild's Panda example: a captured first Steel step diverges)."""
        wp = self.wp
        if not self.capture:
            self._steps(k)
        else:
            nb, r = divmod(k, 10)
            if nb and self.graph is None:
                self._steps(10)
                nb -= 1
                if nb:
                    with wp.ScopedCapture(device=self.dev) as cap:      # records without running
                        self._steps(10)
                    self.graph = cap.graph
            for _ in range(nb):
                wp.capture_launch(self.graph)
            self._steps(r)
        self.time += k * PHYS_DT

    def push(self):
        if self.dirty_q:
            self.s0.joint_q.assign(self.q.reshape(-1))
            self.s0.joint_qd.assign(self.qd.reshape(-1))
            self.newton.eval_fk(self.model, self.s0.joint_q, self.s0.joint_qd, self.s0)
            self.dirty_q = False
        if self.dirty_st:
            self.model.shape_transform.assign(self.st)
            self.dirty_st = False
        self.ctrl.joint_target_q.assign(self.tgt.reshape(-1))

    def pull(self):
        n = self.n
        self.q = self.s0.joint_q.numpy().reshape(n, self.nq).copy()
        self.qd = self.s0.joint_qd.numpy().reshape(n, self.nqd).copy()
        self.bq = self.s0.body_q.numpy().reshape(n, self.nb, 7)
        self.bqd = self.s0.body_qd.numpy().reshape(n, self.nb, 6)
        bad = ~np.isfinite(self.q).all(1)
        for w in np.flatnonzero(bad):
            if self.nan_t[w] is None:
                self.nan_t[w] = round(self.time, 3)
        self.read_contacts()

    def step_to(self, t1):
        self.push()
        k = int(round((t1 - self.time) / PHYS_DT))
        if k > 0:
            self.run(k)
        self.pull()
        if self.time >= T_TASK - 1e-9:
            for w in range(self.n):
                if self.snaps[w] is None:
                    self.snaps[w] = self.snapshot(w)

    def read_contacts(self):
        """Per world and finger the centre of pressure and normal force of its tip body's contacts with the object, and
        the force of all its bodies on the object (hom_tasks.Plant.contacts)."""
        c = self.contacts_buf
        nc = int(c.rigid_contact_count.numpy()[0])
        self.cop = [dict() for _ in range(self.n)]
        self.Ftot = [{f: 0.0 for f in FINGERS} for _ in range(self.n)]
        self.ncon = nc
        if nc == 0:
            return
        s0 = c.rigid_contact_shape0.numpy()[:nc]
        s1 = c.rigid_contact_shape1.numpy()[:nc]
        p0 = c.rigid_contact_point0.numpy()[:nc]
        nrm = c.rigid_contact_normal.numpy()[:nc]
        fc = self.obs.contact_f.numpy()[:nc, :3]
        ok = (s0 >= 0) & (s1 >= 0)
        s0c, s1c = np.maximum(s0, 0), np.maximum(s1, 0)
        t0, t1 = self.obj_mask[s0c], self.obj_mask[s1c]
        pair = ok & (t0 != t1)
        if not pair.any():
            return
        other = np.where(t0, s1c, s0c)
        fn = np.abs(np.einsum("ij,ij->i", fc, nrm))
        b0 = self.shape_body[s0c]
        bq = self.s0.body_q.numpy()
        pw = p0.astype(float).copy()
        on = pair & (b0 >= 0)
        tr = bq[b0[on]]
        pw[on] = tr[:, :3] + rot_xyzw(tr[:, 3:7], p0[on])
        world = self.shape_world[other]
        fi, ti = self.fidx[other], self.tidx[other]
        for i in np.flatnonzero(pair & (fn > 0)):
            w = int(world[i])
            if fi[i] >= 0:
                self.Ftot[w][FINGERS[fi[i]]] += float(fn[i])
        for w in np.unique(world[pair]):
            sel_w = pair & (world == w) & (fn > 0)
            for k, f in enumerate(FINGERS):
                sel = sel_w & (ti == k)
                F = float(fn[sel].sum())
                if F > 0:
                    self.cop[int(w)][f] = ((fn[sel, None] * pw[sel]).sum(0) / F, F)

    def snapshot(self, w):
        q, qd, t = self.q[w], self.qd[w], self.tgt[w]
        bq = self.bq[w, self.obj_body]
        qp = {f"{f}_{j}": float(q[self.dof[(f, j)][0]]) for f in FINGERS for j in JOINTS}
        qp["object"] = [*map(float, bq[:3]), float(bq[6]), *map(float, bq[3:6])]      # MuJoCo's wxyz
        return {"t": float(self.time), "qpos": qp,
                "qvel": {f"{f}_{j}": float(qd[self.dof[(f, j)][1]]) for f in FINGERS for j in JOINTS},
                "ctrl": {f"{f}_{j}": float(t[self.dof[(f, j)][0]]) for f in FINGERS for j in JOINTS}}

    def set_state_all(self, snap):
        """Every world from a snapshot (by joint name, MuJoCo conventions), object at rest."""
        for f in FINGERS:
            for j in JOINTS:
                qa, da = self.dof[(f, j)]
                self.q[:, qa] = snap["qpos"][f"{f}_{j}"]
                self.qd[:, da] = snap["qvel"][f"{f}_{j}"]
                self.tgt[:, qa] = snap["ctrl"][f"{f}_{j}"]
        o = snap["qpos"]["object"]
        self.q[:, self.free_q:self.free_q + 3] = o[:3]
        self.q[:, self.free_q + 3:self.free_q + 7] = [o[4], o[5], o[6], o[3]]
        self.qd[:, self.free_q:self.free_q + 6] = 0.0 if self.free_q < self.nqd else self.qd[:, :0]
        self.dirty_q = True


def rot_xyzw(q, v):
    """Vectors v (n x 3) rotated by quaternions q (n x 4, Newton's xyzw)."""
    u, w = q[:, :3], q[:, 3:4]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


class World:
    """hom_tasks.Plant's interface on world w of a NewtonBatch; advance() waits for the other worlds' rollouts."""

    def __init__(self, B, w, setup, sync):
        self.B, self.w, self.setup, self.sync = B, w, setup, sync
        self.hold_s = 0.0
        self.time = 0.0

    def reset(self, q0, seed):
        B, w = self.B, self.w
        rng = np.random.default_rng(seed)
        jit = np.zeros(3) if seed == 0 else np.array([rng.uniform(-1, 1) * 1e-3, rng.uniform(-1, 1) * 1e-3, 0.0])
        yaw = 0.0 if seed == 0 else math.radians(rng.uniform(-2, 2))
        pos = np.array([0.0, 0.0, T.S.Z_OBJ]) + jit
        qw = quat_mul(quat_yaw(yaw), self.setup["quat"])                # wxyz
        B.q[w, B.free_q:B.free_q + 3] = pos
        B.q[w, B.free_q + 3:B.free_q + 7] = [qw[1], qw[2], qw[3], qw[0]]  # Newton: xyzw
        for f in FINGERS:
            for k, j in enumerate(JOINTS):
                B.q[w, B.dof[(f, j)][0]] = q0[f][k]
        B.qd[w] = 0.0
        B.st[B.post[w]] = B.st0[B.post[w]]
        B.dirty_q = B.dirty_st = True
        B.bqd[w] = 0.0
        B.cop[w], B.Ftot[w] = {}, {f: 0.0 for f in FINGERS}
        self.pos0, self.quat0 = pos, np.asarray(qw, float)
        self.set_targets(q0)

    def post_off(self):
        """The post's shape moved 1 m down in its body's frame (hom_tasks.Plant turns its collision off)."""
        self.B.st[self.B.post[self.w], 2] = self.B.st0[self.B.post[self.w], 2] - 1.0
        self.B.dirty_st = True

    def set_targets(self, q_cmd):
        for f in FINGERS:
            for k, j in enumerate(JOINTS):
                self.B.tgt[self.w, self.B.dof[(f, j)][0]] = float(np.clip(q_cmd[f][k], -1.0, 1.0))   # ctrlrange

    def advance(self, t1):
        self.sync.request(self.w, t1)
        self.time = t1

    def state(self):
        B, w = self.B, self.w
        q, qd = B.q[w], B.qd[w]
        o = q[B.free_q:B.free_q + 7]
        bqd = B.bqd[w, B.obj_body]
        return {"t": float(self.time),
                "q": {f: np.array([q[B.dof[(f, j)][0]] for j in JOINTS], float) for f in FINGERS},
                "qd": {f: np.array([qd[B.dof[(f, j)][1]] for j in JOINTS], float) for f in FINGERS},
                "pos": np.array(o[:3], float), "quat": np.array([o[6], o[3], o[4], o[5]], float),
                "w": np.array(bqd[3:6], float), "v": np.array(bqd[:3], float),
                "cop": dict(B.cop[w]), "Ftot": dict(B.Ftot[w])}

    @property
    def snap(self):
        return self.B.snaps[self.w]


class Sync:
    """The worlds' rollouts run in threads; each advance() asks for a time, and the batch steps once every rollout still
    running has asked (a finished or failed rollout leaves the set, so none waits for it)."""

    def __init__(self, n):
        import threading
        self.cv = threading.Condition()
        self.active = set(range(n))
        self.pending = {}
        self.gen = 0

    def request(self, w, t1):
        with self.cv:
            self.pending[w] = t1
            g = self.gen
            self.cv.notify_all()
            while self.gen == g:
                self.cv.wait()

    def finish(self, w):
        with self.cv:
            self.active.discard(w)
            self.pending.pop(w, None)
            self.cv.notify_all()

    def serve(self, B):
        while True:
            with self.cv:
                while self.active and not self.active <= set(self.pending):
                    self.cv.wait()
                if not self.active:
                    return
                ts = [self.pending[w] for w in self.active]
                if max(ts) - min(ts) > 1e-9:
                    raise RuntimeError(f"rollouts out of step: {min(ts)} .. {max(ts)} s")
                B.step_to(ts[0])
                self.pending.clear()
                self.gen += 1
                self.cv.notify_all()


class RecordingPlant(T.Plant):
    """hom_tasks.Plant that keeps the state at the task's start (for the speed runs)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.snap = None

    def advance(self, t1):
        super().advance(t1)
        if self.snap is None and self.d.time >= T_TASK - 1e-9:
            m, d = self.m, self.d
            qp = {f"{f}_{j}": float(d.qpos[m.jnt_qposadr[m.joint(f"{f}_{j}").id]]) for f in FINGERS for j in JOINTS}
            qv = {f"{f}_{j}": float(d.qvel[m.jnt_dofadr[m.joint(f"{f}_{j}").id]]) for f in FINGERS for j in JOINTS}
            qp["object"] = d.qpos[self.qa:self.qa + 7].tolist()
            ct = {f"{f}_{j}": float(d.ctrl[m.actuator(f"a_{f}_{j}").id]) for f in FINGERS for j in JOINTS}
            self.snap = {"t": float(d.time), "qpos": qp, "qvel": qv, "ctrl": ct}


# ------------------------------------------------------------------------------------------------ hom
def hom_one(scene, p, mxy, setup, q_open, sd, plant):
    """One HOM rollout through the twist on a plant (hom_gpu.cpu_rollout's controller and variant)."""
    import mujoco
    bm = scene.body_mounts(mxy)
    m_pt = mujoco.MjModel.from_xml_path(str(scene.pt_path))
    for f in FINGERS:
        m_pt.body_pos[m_pt.body(f"{f}_mount").id] = bm[f]
    T.S.set_palm(m_pt, setup["palm"])
    fs, os_ = T.sr2_specs(setup)
    mirror = HG.H.Mirror(m_pt, fs, os_)
    over, noise = T.VARIANTS[VAR]
    return T.rollout(TASK, setup, plant, mirror, q_open, sd, T.PRM, replace(HG.H.HomParams(dt=T.DT_C), **over), None,
                     noise, T.PLANT_VARIANTS.get(VAR, {}).get("hold_s", 0.0))


def save_run(trd, pn, sd, trace, snap):
    if snap is not None:
        (trd / f"{pn}_s{sd}_snap.json").write_text(json.dumps(snap))
    np.savez_compressed(trd / f"{pn}_s{sd}.npz", t=np.array([r["t"] for r in trace]),
                        pos_mm=np.array([r["pos_mm"] for r in trace]), axis=np.array([r["axis"] for r in trace]),
                        F=np.array([[r["F"][f] for f in FINGERS] for r in trace]),
                        w_dps=np.array([r["w_dps"] for r in trace]))


def cmd_hom(a):
    import threading
    scene, ws = worlds(a.seeds, a.pairs)
    tag = a.plant + (f"_it{a.iters}" if a.iters else "") + (f"_{a.drive_mode}" if a.drive_mode != "rows" else "") \
        + ("_eager" if a.no_capture else "") + ("_sap" if a.contact_pairs == "sap" else "") + a.tag
    path = ROWS / f"hom_{tag}.jsonl"
    skip = done(path)
    trd = LOGS / "traj" / tag
    trd.mkdir(parents=True, exist_ok=True)
    ws = [w for w in ws if (pair_name(w[0]), w[5]) not in skip]

    def base_row(p, sd):
        return {"pair": pair_name(p), "seed": sd, "plant": a.plant, "x": p["x"], "y": p["y"], "grasp": p["grasp"],
                "by": p["by"], "iters": a.iters, "drive_mode": a.drive_mode, "capture": not a.no_capture,
                "when": time.strftime("%Y-%m-%d %H:%M")}

    def report(row):
        append(path, row)
        print(f"{a.plant} {row['pair']} s{row['seed']}: J {row.get('J')} held {row.get('held')} wall "
              f"{row.get('wall_s')} s" + (f"  {row['error']}" if row["status"] != "ok" else ""), flush=True)

    if a.plant.startswith("mjc_"):
        for p, key, mxy, setup, q_open, sd in ws:
            row = base_row(p, sd)
            try:
                plant = RecordingPlant(scene.pads_path if a.plant == "mjc_pads" else scene.pt_path, setup)
                bm = scene.body_mounts(mxy)
                for f in FINGERS:
                    plant.m.body_pos[plant.m.body(f"{f}_mount").id] = bm[f]
                res, trace, _ = hom_one(scene, p, mxy, setup, q_open, sd, plant)
                row.update(status="ok", J=round(L.track_score(TASK, res), 4), **res)
                save_run(trd, row["pair"], sd, trace, plant.snap)
            except Exception as e:
                row.update(status="error", error=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-2000:])
            report(row)
        return 0
    # Newton plants: the worlds in batches of --chunk, one rollout thread per world
    for c0 in range(0, len(ws), a.chunk):
        chunk = ws[c0:c0 + a.chunk]
        t0 = time.time()
        xmls = [pair_xml(scene, scene.body_mounts(mxy), setup["palm"]) for _p, _k, mxy, setup, _q, _s in chunk]
        B = NewtonBatch(xmls, a.plant, steel_iters=a.iters, drive_mode=a.drive_mode, capture=not a.no_capture,
                        pairs=a.contact_pairs)
        build_s = time.time() - t0
        sync = Sync(len(chunk))
        out = [None] * len(chunk)

        def job(w, item):
            p, key, mxy, setup, q_open, sd = item
            try:
                out[w] = hom_one(scene, p, mxy, setup, q_open, sd, World(B, w, setup, sync))
            except Exception as e:
                out[w] = e, traceback.format_exc()[-2000:]
            finally:
                sync.finish(w)
        th = [threading.Thread(target=job, args=(w, it), daemon=True) for w, it in enumerate(chunk)]
        t1 = time.time()
        for t in th:
            t.start()
        sync.serve(B)
        for t in th:
            t.join()
        wall = time.time() - t1
        for w, (p, key, mxy, setup, q_open, sd) in enumerate(chunk):
            row = base_row(p, sd)
            row.update(chunk=len(chunk), build_s=round(build_s, 1), batch_wall_s=round(wall, 1), nan_t=B.nan_t[w],
                       pairs=B.pairs_mode)
            r = out[w]
            if isinstance(r, tuple) and len(r) == 2 and isinstance(r[0], Exception):
                row.update(status="error", error=f"{type(r[0]).__name__}: {r[0]}", tb=r[1])
            else:
                res, trace, _ = r
                row.update(status="ok", J=round(L.track_score(TASK, res), 4), **res)
                save_run(trd, row["pair"], sd, trace, B.snaps[w])
            report(row)
        del B
    return 0


# ------------------------------------------------------------------------------------------------ speed
def first_snap(plant="mjc_pads"):
    """The state at the task's start of the plant's own HOM run, seed 0, on the first pair that has one, and the pair's
    index: a rigid point contact started from the pads' squeezed grasp (the tip meshes 0.36-0.62 mm inside the sphere)
    throws the sphere out in Steel."""
    for i, p in enumerate(json.loads(PAIRS.read_text())["pairs"]):
        f = LOGS / "traj" / plant / f"{pair_name(p)}_s0_snap.json"
        if f.exists():
            return i, p, json.loads(f.read_text())
    raise SystemExit(f"no state at 2.4 s under {LOGS / 'traj' / plant}: run `hom --plant {plant} --seeds 0` first")


def speed_newton(a, nw, scene, setup, mxy, snap):
    xml = pair_xml(scene, scene.body_mounts(mxy), setup["palm"])
    t0 = time.time()
    # the hydroelastic stage buffers at their default size take 13.8 GB at 128 worlds (2026-10-06); above 64 worlds the
    # 2026-10-06 sizing (newton_hand_throughput.py: fraction 0.013, broad phase x40) with a margin. At (0.02, 40) the
    # broad phase held 1,382 pairs per world and the grasp asked for about 1,390 (stage 2, 256 and 1024 worlds)
    buf = (1.0, 1) if nw <= 64 else tuple(a.hydro_buf) if a.hydro_buf else (0.02, 60)
    B = NewtonBatch([xml] * nw, a.plant, steel_iters=a.iters, drive_mode=a.drive_mode, hydro_buf=buf, pairs=a.contact_pairs)
    build_s = time.time() - t0
    B.set_state_all(snap)
    B.st[B.post, 2] = B.st0[B.post, 2] - 1.0
    B.dirty_st = True
    o0 = B.q[:, B.free_q:B.free_q + 3].copy()
    B.step_to(0.2)
    B.wp.synchronize()
    nstep = int(round(a.sim_s / PHYS_DT))
    t1 = time.perf_counter()
    B.run(nstep)
    B.wp.synchronize()
    el = time.perf_counter() - t1
    B.pull()
    drift = np.linalg.norm(B.q[:, B.free_q:B.free_q + 3] - o0, axis=1) * 1e3
    return dict(us_per_world_step=1e6 * el / (nstep * nw), contacts_per_world=B.ncon / nw,
                held_frac=float(np.mean(np.nan_to_num(drift, nan=1e9) < 20.0)),
                drift_mm_median=float(np.nanmedian(drift)), drift_mm_max=float(np.nanmax(drift)),
                nan_worlds=int(sum(t is not None for t in B.nan_t)), build_s=round(build_s, 1), hydro_buf=list(buf),
                pairs=B.pairs_mode, pairs_per_world=getattr(B, "npairs_world", None), host_rss_mb=host_rss_mb(),
                solver_opts={k: str(v) for k, v in B.solver_opts.items()})


def speed_mjw(a, nw, scene, setup, mxy, snap):
    """MuJoCo-Warp with the benchmark's settings (hom_gpu.Batch: 20 iterations, 10 line-search steps, tolerance 1e-6,
    impratio 100; the pads and the tip's fine rigid spheres meet the object only)."""
    import re
    import mujoco
    import mujoco_warp as mjw
    import warp as wp
    path = scene.pads_path if a.plant == "mjw_pads" else scene.pt_path
    m = mujoco.MjModel.from_xml_path(str(path))
    T.S.set_palm(m, setup["palm"])
    bm = scene.body_mounts(mxy)
    for f in FINGERS:
        m.body_pos[m.body(f"{f}_mount").id] = bm[f]
    fine = re.compile(r"_(dip_link|cassette|cap)_s\d+$")
    ob = m.body(T.S.OBJ).id
    pads = np.array([g for g in range(m.ngeom) if "_pad" in (m.geom(g).name or "")
                     or (fine.search(m.geom(g).name or "") and m.geom_size[g][0] < 0.002)], int)
    tool = np.flatnonzero(m.geom_bodyid == ob)
    if a.contact_pairs == "obj":
        object_only(m)
    elif len(pads):
        m.geom_contype[pads], m.geom_conaffinity[pads] = 2, 0
        m.geom_conaffinity[tool] = m.geom_conaffinity[tool] | 2
    m.opt.tolerance, m.opt.iterations, m.opt.ls_iterations = 1e-6, 20, 10
    post = m.geom("post").id
    m.geom_contype[post] = m.geom_conaffinity[post] = 0
    d = mujoco.MjData(m)
    for f in FINGERS:
        for j in JOINTS:
            jid = m.joint(f"{f}_{j}").id
            d.qpos[m.jnt_qposadr[jid]] = snap["qpos"][f"{f}_{j}"]
            d.ctrl[m.actuator(f"a_{f}_{j}").id] = snap["ctrl"][f"{f}_{j}"]
    qa = m.jnt_qposadr[m.body_jntadr[ob]]
    d.qpos[qa:qa + 7] = snap["qpos"]["object"]
    mujoco.mj_forward(m, d)
    wm = mjw.put_model(m)
    wd = mjw.put_data(m, d, nworld=nw, nconmax=HG.NCONMAX, njmax=HG.NJMAX)

    def block():
        for _ in range(10):
            mjw.step(wm, wd)
    for _ in range(20):
        block()
    wp.synchronize()
    o0 = wd.xpos.numpy()[:, ob, :].copy()
    with wp.ScopedCapture() as cap:
        block()
    nb = int(a.sim_s / PHYS_DT / 10)
    t0 = time.perf_counter()
    for _ in range(nb):
        wp.capture_launch(cap.graph)
    wp.synchronize()
    el = time.perf_counter() - t0
    drift = np.linalg.norm(wd.xpos.numpy()[:, ob, :] - o0, axis=1) * 1e3
    return dict(us_per_world_step=1e6 * el / (nb * 10 * nw), contacts_per_world=int(wd.nacon.numpy()[0]) / nw,
                held_frac=float(np.mean(drift < 20.0)), drift_mm_median=float(np.median(drift)),
                drift_mm_max=float(drift.max()), nefc_world_max=int(wd.nefc.numpy().max()), pairs=a.contact_pairs,
                host_rss_mb=host_rss_mb())


def cmd_speed(a):
    src = {"mjw_pads": "mjc_pads", "mjw_pt": "mjc_pt"}.get(a.plant, a.plant)
    src += (f"_it{a.iters}" if a.iters and not a.plant.startswith("mjw_") else "") + \
        (f"_{a.drive_mode}" if a.drive_mode != "rows" and not a.plant.startswith("mjw_") else "")
    i, p, snap = first_snap(a.snap_from or src)
    scene, ws = worlds([0], i + 1)
    _p, _key, mxy, setup, _q_open, _sd = ws[i]
    import warp
    for nw in a.nworld:
        row = {"plant": a.plant, "nworld": nw, "pair": pair_name(p), "iters": a.iters, "drive_mode": a.drive_mode,
               "sim_s": a.sim_s, "warp": warp.__version__, "when": time.strftime("%Y-%m-%d %H:%M"),
               "gpu_before_mb": gpu_used_mb()}
        try:
            import newton
            row["newton"] = newton.__version__
        except Exception:
            pass
        try:
            r = speed_mjw(a, nw, scene, setup, mxy, snap) if a.plant.startswith("mjw_") else \
                speed_newton(a, nw, scene, setup, mxy, snap)
            row.update(r, status="ok", gpu_used_mb=gpu_used_mb())
            print(f"{a.plant} {nw:>5}: {r['us_per_world_step']:.2f} us/world-step, {r['contacts_per_world']:.1f} "
                  f"contacts/world, held {r['held_frac']:.2f}, drift {r['drift_mm_median']:.2f} mm, "
                  f"GPU {row['gpu_used_mb']} MB", flush=True)
        except Exception as e:
            row.update(status="error", error=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-2000:])
            print(f"{a.plant} {nw}: {row['error']}", flush=True)
        append(ROWS / f"speed{a.tag}.jsonl", row)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hom")
    h.add_argument("--plant", required=True, choices=PLANTS)
    h.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    h.add_argument("--pairs", type=int, default=None, help="the first n pairs only")
    h.add_argument("--no-capture", action="store_true", help="Newton plants: eager steps, no CUDA graphs")
    h.add_argument("--chunk", type=int, default=40, help="Newton plants: worlds per batch")
    s = sub.add_parser("speed")
    s.add_argument("--plant", required=True, choices=PLANTS[2:] + ("mjw_pads", "mjw_pt"))
    s.add_argument("--nworld", type=int, nargs="+", default=[16, 256, 1024])
    s.add_argument("--sim-s", type=float, default=0.5)
    s.add_argument("--snap-from", default=None, help="the hom run (rows tag) whose state at 2.4 s starts the worlds")
    s.add_argument("--hydro-buf", type=float, nargs=2, default=None, metavar=("FRACTION", "MULT_BROAD"),
                   help="hydroelastic buffers above 64 worlds (default 0.02 60)")
    for q in (h, s):
        q.add_argument("--iters", type=int, default=None, help="Steel: PGS iterations (default per plant)")
        q.add_argument("--drive-mode", choices=["rows", "augmented"], default="rows")
        q.add_argument("--tag", default="")
        q.add_argument("--contact-pairs", choices=["obj", "sap"], default="obj",
                       help="obj: the object's contact pairs only (Newton: explicit list; MuJoCo-Warp: masks); sap: Newton's "
                            "sweep and prune over all shapes, MuJoCo-Warp with the benchmark's filter")
    a = ap.parse_args()
    return {"hom": cmd_hom, "speed": cmd_speed}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
