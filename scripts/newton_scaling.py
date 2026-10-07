#!/usr/bin/env python3
"""Newton hydroelastic GPU scaling on the SR2 holding fixture (2026-10-05).

The fixture is the hold of scripts/newton_pinch_probe.py (attempt tag 'soft_fallback'):
Codex's SR2 thumb-index pinch as built by distributed_contact_newton.setup, hydroelastic
tip spheres with kh_pad = E/R = 9.48e8 N/m^3, tool kh 100x, 0.5 mm SDF voxels, mu 1, kf 10,
the two-value soft fallback solref "0.5 1" written into the MJCF before add_mjcf, impratio
100, elliptic cone, Newton solver at 200 iterations, 1 ms step. The tool is held 0.5 s by
the pinch under the 20 N/m centring guide and 2e-5 N m s rotational damping of the
MuJoCo-Warp pad fixture, with no applied torque (the probe's hold phase).

What this script adds to the probe:

* the single-world builder is replicated into nworld worlds (ModelBuilder.replicate);
* per-world capacities (Newton rigid contacts, MuJoCo-Warp nconmax and njmax) and the
  hydroelastic buffer fraction come from a calibration hold at generous capacities, with
  25 % headroom on contacts and constraints and 50 % on the hydroelastic stages; every
  10 ms block audits all of them, and the reduction hashtable's insert failures;
* one CUDA graph per 10-step block, timed between synchronizations, audits outside the
  timed region. HydroelasticSDF.launch copies ht_insert_failures to the host every 120
  launches; a capture that contains that copy fails with CUDA error 906 ("operation would
  make the legacy stream depend on a capturing blocking stream", wp_memcpy_d2h). The poll
  interval is raised before capture and the counter is read in the audit instead;
  `--mode diag906` reproduces the failure by placing the poll inside the capture;
* per-world fidelity at the end of the hold from the solver's own contact forces: normal
  force per pad, tool slip relative to the fingertip pair, torsion capacity
  mu * sum(r_i f_i) about the pressure-centre axis and about the fixed pinch point P.

Models and fixtures (2026-10-07). `--model hydro` is the 10-05 hydroelastic tip, kh = E/h; `hydro_mc` divides out
the tip-tool effective mass as in contact_bed_newton.py and newton_turn.py: each tip's kh is multiplied by
invweight0[tip] + invweight0[tool] of the solver's own MuJoCo model, the tool's is 100x the largest tip's; `pads1` is
the MuJoCo-Warp pad fixture of scripts/gpu_scaling_mjwarp.py imported into Newton: 1 mm sphere pads (205 per tip,
r 0.75 mm) with the legacy mapping (solref (0.015, 1), solimp d0 from the MJCF inverse weights, impratio 100),
d0 rescaled to the inverse weights of Newton's MuJoCo model and the friction gain kf matched to the 15 ms solref
(contact_bed_newton.NewtonRig._match_friction), shape gap 0.5 mm, margin 0. `--fixture hold` is the 10-05 0.5 s hold
without torque; `--fixture twist` is the MuJoCo-Warp pad fixture's 0.8 s run with the two opposing 12 mN m
raised-cosine torque pulses about the pinch axis at 0.3-0.6 s (distributed_contact_gpu.guide). Rows of the new models
carry the fixture in their key.

Run with logs/20261004-contact-transfer/venv/bin/python and WARP_CACHE_PATH=$(mktemp -d).
Rows: docs/experiments/20261005-gpu_scaling/newton_scaling.jsonl, one fsynced line each.
"""

import argparse
import gc
import json
import os
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import warp as wp

import newton
from newton.geometry import HydroelasticSDF

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from distributed_contact_transfer import build  # noqa: E402

OUT = ROOT / "docs/experiments/20261005-gpu_scaling/newton_scaling.jsonl"
WORK = ROOT / "logs/20261005-newton_scaling"
MIB = 1024.0 * 1024.0
R_PAD = 0.01055
KH_PAD = 1e7 / R_PAD
TOOL_RATIO = 100.0
FALLBACK_SOLREF = "0.5 1"
IGNORED_APPS = ("gnome-remote-desktop-daemon",)  # resident display service, 0 % SM
GENEROUS = dict(rigid=4096, nconmax=4096, njmax=12288, buffer_fraction=1.0)
BLOCK = 10  # steps per CUDA graph (even, so the s0/s1 ping-pong returns to its start)
POLL = 120  # HydroelasticSDF._host_warning_poll_interval in newton 1.7.0.dev0


def git_rev():
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def gpu_snapshot():
    """Device memory and utilization, and every compute process other than this one."""
    me = os.getpid()
    snap = dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"))
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.free,memory.total,clocks.sm",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20,
        ).stdout.strip().split(",")
        snap.update(util_pct=float(q[0]), used_mib=float(q[1]), free_mib=float(q[2]),
                    total_mib=float(q[3]), sm_clock_mhz=float(q[4]))
    except Exception as e:  # noqa: BLE001
        snap["query_error"] = repr(e)
    others, ignored, own = [], [], None
    try:
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20,
        ).stdout.strip().splitlines()
        for line in apps:
            parts = [x.strip() for x in line.split(",")]
            if len(parts) < 3 or not parts[0].isdigit():
                continue
            try:
                mem = float(parts[2])
            except ValueError:
                mem = None
            entry = dict(pid=int(parts[0]), name=parts[1], used_mib=mem)
            if entry["pid"] == me:
                own = mem
            elif any(s in parts[1] for s in IGNORED_APPS):
                ignored.append(entry)
            else:
                others.append(entry)
    except Exception as e:  # noqa: BLE001
        snap["apps_error"] = repr(e)
    snap.update(own_mib=own, others=others, ignored=ignored)
    return snap


def contended(snap, after=False):
    """Another compute process on the GPU; before a point also device utilization > 20 %."""
    return bool(snap.get("others")) or (not after and snap.get("util_pct", 0.0) > 20.0)


def wait_clear(max_wait_s, log=print):
    """Block until no other compute process runs and the GPU idles; returns (snapshot, waited_s)."""
    t0 = time.time()
    while True:
        snap = gpu_snapshot()
        if not contended(snap):
            time.sleep(3.0)
            snap = gpu_snapshot()  # two clean samples 3 s apart
            if not contended(snap):
                return snap, time.time() - t0
        if time.time() - t0 > max_wait_s:
            return snap, time.time() - t0
        log(f"GPU busy ({[o['name'] for o in snap.get('others', [])]}, util {snap.get('util_pct')} %); waiting",
            flush=True)
        time.sleep(15.0)


@wp.kernel
def hold_guide(bodyq: wp.array[wp.transform], bodyv: wp.array[wp.spatial_vector],
               bodyf: wp.array[wp.spatial_vector], tools: wp.array[wp.int32], center: wp.vec3):
    # distributed_contact_newton.drive with peak 0, one thread per world.
    b = tools[wp.tid()]
    position = wp.transform_get_translation(bodyq[b])
    force = 20.0 * (center - position) - 0.05 * wp.spatial_top(bodyv[b])
    torque = -0.00002 * wp.spatial_bottom(bodyv[b])
    bodyf[b] = wp.spatial_vector(force, torque)


@wp.kernel
def twist_guide(bodyq: wp.array[wp.transform], bodyv: wp.array[wp.spatial_vector],
                bodyf: wp.array[wp.spatial_vector], tools: wp.array[wp.int32], center: wp.vec3,
                axis: wp.vec3, peak: float, times: wp.array[float]):
    # distributed_contact_gpu.guide: centring guide plus two opposing raised-cosine torque pulses about the axis.
    b = tools[wp.tid()]
    t = times[0]
    tau = float(0.0)
    if t >= 0.3 and t < 0.45:
        tau = peak * wp.pow(wp.sin(wp.pi * (t - 0.3) / 0.15), 2.0)
    elif t >= 0.45 and t < 0.6:
        tau = -peak * wp.pow(wp.sin(wp.pi * (t - 0.45) / 0.15), 2.0)
    position = wp.transform_get_translation(bodyq[b])
    force = 20.0 * (center - position) - 0.05 * wp.spatial_top(bodyv[b])
    torque = tau * axis - 0.00002 * wp.spatial_bottom(bodyv[b])
    bodyf[b] = wp.spatial_vector(force, torque)


@wp.kernel
def advance_time(times: wp.array[float], dt: float):
    times[0] = times[0] + dt


FIXTURES = {"hold": dict(peak=0.0, duration=0.5), "twist": dict(peak=0.012, duration=0.8)}
MODELS = ("hydro", "hydro_mc", "pads1")
PAD_TC = 0.015            # the legacy pads' solref time constant (gpu_scaling_mjwarp.apply_legacy)


def apply_legacy(m, spacing):
    """gpu_scaling_mjwarp.apply_legacy (that module imports mjlab, which the Newton venv lacks): plain positive-format
    solref (0.015, 1) and solimp d0 = 1 - 1/(tc^2 K_sphere diag) on every pad, impratio 100."""
    import hom_chain as C
    qdict, _ = C.postures([0.0])
    dirs = C.contact_dirs(qdict[0.0])
    trial = C.make_trial(0, d_cg=0.0, perturb=False)
    trial["mscale"] = 1.0
    _, info, _ = C.chain_scene(f"mj:spheres:s{spacing}:rs0.75:tr0.03:ir100", trial, dirs, pad_tc=(0.015, 0.9))
    K = float(info["K_sphere"])
    tc = PAD_TC
    diag = float(m.body_invweight0[m.body("thumb_tip").id, 0] + m.body_invweight0[m.body("tool").id, 0])
    d0 = 1.0 - 1.0 / (tc**2 * K * diag)
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


def pad_world_builder(dt, directory):
    """The MuJoCo-Warp pad fixture (gpu_scaling_mjwarp.run_point, policy 'legacy', 1 mm) as a Newton builder: the MJCF
    saved right after build(), the pads' legacy solref and solimp written into it, and d0 rescaled to the tip + tool
    inverse weight of the MuJoCo model SolverMuJoCo builds (probed on one world), d0' = 1 - (1 - d0) w_mjcf / w_solver.
    Returns the builder, meta and the mapping."""
    m, d, p, meta = build(dict(spacing_mm=1.0, timestep=dt, policy="compiled", torque_peak=0.012, duration=0.8))
    directory.mkdir(parents=True, exist_ok=True)
    mujoco.mj_saveLastXML(str(directory / "pads_source.xml"), m)    # before apply_legacy compiles other models
    mapping = apply_legacy(m, 1.0)
    tc, d0 = mapping["solref"][0], mapping["solimp"][0]
    tree = ET.parse(directory / "pads_source.xml").getroot()
    contact = tree.find("contact")
    if contact is not None:      # explicit pairs only repeat the pads' solref; the bitmasks select the same pairs
        tree.remove(contact)

    def xml_with(d0x):
        for g in tree.iter("geom"):
            if g.get("name", "").startswith(("thumb_pad", "index_pad")):
                g.set("solref", f"{tc:g} 1")
                g.set("solimp", f"{d0x:.6g} {d0x:.6g} 0.001 0.5 2")
        return ET.tostring(tree, encoding="unicode")

    def builder(xml):
        b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        b.add_mjcf(xml, ctrl_direct=True, parse_sites=False, parse_visuals=False)
        for i in range(len(b.shape_gap)):
            b.shape_gap[i] = 0.0005
            b.shape_margin[i] = 0.0
        for f, qq in zip(("thumb", "index", "middle"), np.array(meta["qtouch"]).reshape(3, 3)):
            for name, q in zip((f + "_yaw", f + "_mcp", f + "_pip"), qq):
                j = next(i for i, x in enumerate(b.joint_label) if x.split("/")[-1] == name)
                b.joint_q[b.joint_q_start[j]] = float(q)
        return b

    pm = builder(xml_with(d0)).finalize(device="cuda:0")
    ps = newton.solvers.SolverMuJoCo(pm, use_mujoco_contacts=False, disable_sensors=True, njmax=64, nconmax=16)
    labels = [x.split("/")[-1] for x in pm.body_label]
    m2n = ps.mjc_body_to_newton.numpy()[0]
    w = {labels[int(m2n[bi])]: float(ps.mj_model.body_invweight0[bi, 0]) for bi in range(ps.mj_model.nbody)
         if 0 <= int(m2n[bi]) < len(labels)}
    w_new = w["thumb_tip"] + w["tool"]
    d0_new = 1.0 - (1.0 - d0) * mapping["diag_invweight"] / w_new
    del ps, pm
    xml = xml_with(d0_new)
    (directory / "hand_pads.xml").write_text(xml)
    mapping.update(d0_solver=d0_new, w_solver=w_new, n_pad_geoms_xml=xml.count('name="thumb_pad') + xml.count('name="index_pad'))
    return builder(xml), meta, mapping


def world_builder(dt, directory):
    """Single-world builder exactly as distributed_contact_newton.setup builds it, with the
    probe's soft fallback solref written into the MJCF before add_mjcf."""
    m, d, p, meta = build(dict(policy="compiled", spacing_mm=2.0, timestep=dt))
    directory.mkdir(parents=True, exist_ok=True)
    mujoco.mj_saveLastXML(str(directory / "source.xml"), m)
    tree = ET.parse(directory / "source.xml").getroot()
    for parent in tree.iter():
        for el in list(parent):
            if el.tag == "geom" and el.get("name", "").startswith(("thumb_pad", "index_pad")):
                parent.remove(el)
    contact = tree.find("contact")
    if contact is not None:
        tree.remove(contact)
    for geom in tree.iter("geom"):
        if geom.get("name", "") in ("thumb_tipsphere", "index_tipsphere"):
            geom.set("contype", "1")
            geom.set("conaffinity", "2")
            geom.set("solimp", ".9 .9 .001 .5 2")
            geom.set("friction", "1 0 0")
    xml = ET.tostring(tree, encoding="unicode")
    n_fix = xml.count('solref="0.006"')
    xml = xml.replace('solref="0.006"', f'solref="{FALLBACK_SOLREF}"')
    (directory / "hand_sphere.xml").write_text(xml)
    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    b.add_mjcf(xml, ctrl_direct=True, parse_sites=False, parse_visuals=False)
    for f, qq in zip(("thumb", "index", "middle"), np.array(meta["qtouch"]).reshape(3, 3)):
        for name, q in zip((f + "_yaw", f + "_mcp", f + "_pip"), qq):
            j = next(i for i, x in enumerate(b.joint_label) if x.split("/")[-1] == name)
            b.joint_q[b.joint_q_start[j]] = float(q)
    for i, label in enumerate(b.shape_label):
        name = label.split("/")[-1]
        if name not in ("thumb_tipsphere", "index_tipsphere", "tool"):
            continue
        b.shape_flags[i] |= int(newton.ShapeFlags.HYDROELASTIC)
        b.shape_sdf_target_voxel_size[i] = 0.0005
        b.shape_sdf_narrow_band_range[i] = (-0.003, 0.003)
        b.shape_sdf_padding[i] = 0.003
        b.shape_margin[i] = 0.0
        b.shape_gap[i] = 0.0
        b.shape_material_kh[i] = KH_PAD * (TOOL_RATIO if name == "tool" else 1.0)
        b.shape_material_mu[i] = 1.0
        b.shape_material_kf[i] = 10.0
        b.shape_material_mu_torsional[i] = 0.0
        b.shape_material_mu_rolling[i] = 0.0
    return b, meta, n_fix


class Sim:
    """nworld replicated holding fixtures with one shared collision pipeline and solver."""

    def __init__(self, nworld, reduce, caps, dt=0.001, model="hydro", fixture="hold"):
        self.nworld, self.reduce, self.caps, self.dt = nworld, reduce, dict(caps), dt
        self.kind, self.fixture = model, fixture
        self.pads = model == "pads1"
        self.pad_info = None
        if self.pads:
            b, self.meta, mapping = pad_world_builder(dt, WORK / f"model_pads_dt{dt}")
            self.n_fix = 0
            self.pad_info = dict(mapping)
        else:
            b, self.meta, self.n_fix = world_builder(dt, WORK / f"model_dt{dt}")
        scene = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        scene.replicate(b, nworld)
        self.model = model = scene.finalize(device="cuda:0")
        kw = {}
        if not self.pads:
            kw["sdf_hydroelastic_config"] = HydroelasticSDF.Config(
                reduce_contacts=reduce, anchor_contact=True, moment_matching=False,
                buffer_fraction=caps["buffer_fraction"], buffer_mult_broad=int(caps.get("buffer_mult_broad", 1)))
        self.pipe = newton.CollisionPipeline(model, reduce_contacts=reduce, rigid_contact_max=nworld * caps["rigid"],
                                             broad_phase="explicit", **kw)
        self.solver = newton.solvers.SolverMuJoCo(
            model, use_mujoco_contacts=False, disable_sensors=True, njmax=caps["njmax"], nconmax=caps["nconmax"],
            iterations=200, ls_iterations=50, cone="elliptic", jacobian="dense", integrator="implicitfast",
            tolerance=1e-8, impratio=100.0)
        self.inv_w = self._inv_weights()
        if self.kind == "hydro_mc":
            self.mass_correction = self._mass_correct()
        if self.pads:
            self.pad_info.update(self._pad_contact())
        self.times = wp.zeros(1, dtype=float, device="cuda:0")
        self.s0, self.s1 = model.state(), model.state()
        self.ctrl = model.control()
        c = self.ctrl.mujoco.ctrl
        self.ctrl.mujoco.ctrl.assign(np.tile(np.array(self.meta["targets"], np.float32), nworld).reshape(c.shape))
        newton.eval_fk(model, model.joint_q, model.joint_qd, self.s0)
        self.cc = self.pipe.contacts()
        labels = [x.split("/")[-1] for x in model.body_label]
        bw = model.body_world.numpy()
        self.nb = model.body_count // nworld
        if not (np.all(np.diff(bw) >= 0) and model.body_count == self.nb * nworld):
            raise ValueError("bodies are not laid out world by world")
        ids = {n: np.array([i for i, x in enumerate(labels) if x == n]) for n in ("tool", "thumb_tip", "index_tip")}
        for n, v in ids.items():
            if len(v) != nworld or not np.all(bw[v] == np.arange(nworld)):
                raise ValueError(f"body {n}: {len(v)} matches for {nworld} worlds")
        self.ids = ids
        self.tools = wp.array(ids["tool"].astype(np.int32), dtype=wp.int32, device="cuda:0")
        self.P = np.array(self.meta["P"])
        self.u = np.array(self.meta["u"])
        self.a = np.array(self.meta["a"])
        mm = self.solver.mj_model
        names = [mm.geom(g).name.split("/")[-1] for g in range(mm.ngeom)]
        self.geom_pad = np.full(mm.ngeom, -1)
        for g, n in enumerate(names):
            if n.startswith(("thumb_tipsphere", "thumb_pad")):
                self.geom_pad[g] = 0
            elif n.startswith(("index_tipsphere", "index_pad")):
                self.geom_pad[g] = 1
        self.hs = getattr(self.pipe, "hydroelastic_sdf", None)
        self.init_pose = self.poses()
        self.steps = 0

    # ------------------------------------------------------------------ model set-up
    def _inv_weights(self):
        """invweight0 of the tips and the tool in the solver's own MuJoCo model (world 0)."""
        mjm = self.solver.mj_model
        labels = [x.split("/")[-1] for x in self.model.body_label]
        m2n = self.solver.mjc_body_to_newton.numpy()[0]
        w = {}
        for bi in range(mjm.nbody):
            nb = int(m2n[bi])
            nm = labels[nb] if 0 <= nb < len(labels) else ""
            if nm in ("thumb_tip", "index_tip", "tool"):
                w[nm] = float(mjm.body_invweight0[bi, 0])
        return w

    def _mass_correct(self):
        """kh of each tip x (invweight0[tip] + invweight0[tool]); the tool 100x the largest tip (newton_turn.mass_correct)."""
        w = self.inv_w
        kh_tip = {f: KH_PAD * (w[f + "_tip"] + w["tool"]) for f in ("thumb", "index")}
        kh_tool = TOOL_RATIO * max(kh_tip.values())
        lab = [x.split("/")[-1] for x in self.model.shape_label]
        kh = self.model.shape_material_kh.numpy()
        for i, n in enumerate(lab):
            if n == "tool":
                kh[i] = kh_tool
            elif n in ("thumb_tipsphere", "index_tipsphere"):
                kh[i] = kh_tip[n.split("_")[0]]
        self.model.shape_material_kh.assign(kh)
        return dict(invweight0=w, kh_tip=kh_tip, kh_tool=kh_tool)

    def _pad_contact(self):
        """Check the pads' d0 in the solver model against the rescaled value; set the friction gain kf on pads and tool
        so that the friction rows keep the pads' 15 ms solref (contact_bed_newton.NewtonRig._match_friction)."""
        d0, w_new = self.pad_info["d0_solver"], self.inv_w["thumb_tip"] + self.inv_w["tool"]
        mjm = self.solver.mj_model
        gi = [g for g in range(mjm.ngeom) if (mjm.geom(g).name or "").split("/")[-1].startswith(("thumb_pad", "index_pad"))]
        out = dict(n_pad_geoms_solver=len(gi), d0_solver_model=float(mjm.geom_solimp[gi[0], 0]) if gi else None,
                   w_check=w_new)
        kf = 2.0 / (PAD_TC * w_new * ((1.0 - d0) / 100.0 + d0))
        lab = [x.split("/")[-1] for x in self.model.shape_label]
        kfa = self.model.shape_material_kf.numpy()
        for i, n in enumerate(lab):
            if n.startswith(("thumb_pad", "index_pad", "tool")):
                kfa[i] = kf
        self.model.shape_material_kf.assign(kfa)
        out["kf"] = kf
        return out

    # ------------------------------------------------------------------ stepping
    def step_block(self):
        fx = FIXTURES[self.fixture]
        for _ in range(BLOCK):
            self.s0.clear_forces()
            if self.fixture == "hold":
                wp.launch(hold_guide, dim=self.nworld,
                          inputs=[self.s0.body_q, self.s0.body_qd, self.s0.body_f, self.tools, wp.vec3(self.P)])
            else:
                wp.launch(twist_guide, dim=self.nworld,
                          inputs=[self.s0.body_q, self.s0.body_qd, self.s0.body_f, self.tools, wp.vec3(self.P),
                                  wp.vec3(self.u), fx["peak"], self.times])
            self.pipe.collide(self.s0, self.cc)
            self.solver.step(self.s0, self.s1, self.ctrl, self.cc, self.dt)
            self.s0, self.s1 = self.s1, self.s0
            wp.launch(advance_time, dim=1, inputs=[self.times, self.dt])

    def capture(self, poll_in_capture=False):
        """Record one block as a CUDA graph. The hydroelastic host poll is moved out of the
        captured launches unless poll_in_capture places it inside on purpose."""
        if self.hs is not None:
            if poll_in_capture:
                self.hs._host_warning_poll_interval = POLL
                self.hs._launch_counter = POLL - BLOCK // 2  # the poll falls on launch BLOCK/2
            else:
                self.hs._host_warning_poll_interval = 10**12
        with wp.ScopedCapture() as cap:
            self.step_block()
        return cap.graph

    # ------------------------------------------------------------------ audit
    def poses(self):
        bq = self.s0.body_q.numpy()
        return {n: bq[v] for n, v in self.ids.items()}

    def audit(self):
        bq = self.s0.body_q.numpy().reshape(self.nworld, self.nb, 7)
        bv = self.s0.body_qd.numpy().reshape(self.nworld, self.nb, 6)
        fin = np.isfinite(bq).all(axis=(1, 2)) & np.isfinite(bv).all(axis=(1, 2))
        tool = self.s0.body_q.numpy()[self.ids["tool"], :3]
        disp = np.linalg.norm(tool - self.P, axis=1)
        vmax = np.abs(bv).max(axis=(1, 2))
        ok = fin & (disp <= 0.025) & (vmax <= 2000.0)
        out = dict(ok=ok, fin=fin, disp=disp, vmax=vmax)
        n_rigid = int(self.cc.rigid_contact_count.numpy()[0])
        out["rigid_total"] = n_rigid
        if n_rigid:
            s0 = self.cc.rigid_contact_shape0.numpy()[: min(n_rigid, self.cc.rigid_contact_max)]
            sw = self.model.shape_world.numpy()[s0]
            out["rigid_per_world_max"] = int(np.bincount(sw[sw >= 0], minlength=self.nworld).max())
        else:
            out["rigid_per_world_max"] = 0
        md = self.solver.mjw_data
        na = int(md.nacon.numpy()[0])
        out["mj_total"] = na
        wid = md.contact.worldid.numpy()[: min(na, md.naconmax)]
        out["mj_per_world_max"] = int(np.bincount(wid, minlength=self.nworld).max()) if wid.size else 0
        out["nefc_max"] = int(md.nefc.numpy().max())
        out["niter_max"] = int(md.solver_niter.numpy().max())
        if self.hs is not None:
            hs = self.hs
            counts = [int(c.numpy()[0]) for c in hs.iso_buffer_counts]
            caps = [hs.max_num_blocks_broad, *hs.iso_max_dims]
            out["hydro_counts"] = counts
            out["hydro_caps"] = [int(x) for x in caps]
            red = getattr(hs, "contact_reduction", None)
            fc = getattr(red, "contact_count", None)
            out["face_contacts"] = int(fc.numpy()[0]) if fc is not None else None
            out["face_cap"] = int(hs.max_num_face_contacts)
            ht = getattr(getattr(red, "reducer", None), "ht_insert_failures", None)
            out["ht_insert_failures"] = int(ht.numpy()[0]) if ht is not None else None
        return out

    def fidelity(self):
        """Per-world end-of-hold metrics from the MuJoCo-Warp solve of the last step."""
        md = self.solver.mjw_data
        nw = self.nworld
        na = min(int(md.nacon.numpy()[0]), md.naconmax)
        wid = md.contact.worldid.numpy()[:na]
        adr = md.contact.efc_address.numpy()[:na, 0]
        geom = md.contact.geom.numpy()[:na]
        pos = md.contact.pos.numpy()[:na].astype(float)
        nrm = md.contact.frame.numpy()[:na, 0, :].astype(float)
        mu = md.contact.friction.numpy()[:na, 0].astype(float)
        force = md.efc.force.numpy()
        act = adr >= 0
        fn = np.zeros(na)
        fn[act] = force[wid[act], adr[act]]
        pad = np.maximum(self.geom_pad[geom[:, 0]], self.geom_pad[geom[:, 1]])
        fpad = np.zeros((nw, 2))
        for k in (0, 1):
            sel = act & (pad == k)
            np.add.at(fpad[:, k], wid[sel], fn[sel])
        ncon = np.bincount(wid[act & (fn > 0)], minlength=nw)
        u = self.u

        def capacity(center):
            arm = np.cross(u, pos - center)
            vt = arm - np.sum(arm * nrm, axis=1)[:, None] * nrm
            out = np.zeros(nw)
            np.add.at(out, wid[act], (mu * fn * np.linalg.norm(vt, axis=1))[act])
            return out

        cap_P = capacity(self.P)
        ftot = np.zeros(nw)
        np.add.at(ftot, wid[act], fn[act])
        cen = np.zeros((nw, 3))
        np.add.at(cen, wid[act], (fn[:, None] * pos)[act])
        cen = cen / np.maximum(ftot, 1e-12)[:, None]
        cap_c = capacity(cen[wid])
        now = self.poses()
        ini = self.init_pose

        def rel(pp):
            s = pp["tool"][:, :3] - 0.5 * (pp["thumb_tip"][:, :3] + pp["index_tip"][:, :3])
            return s - (s @ u)[:, None] * u

        ds = rel(now) - rel(ini)
        slip = np.linalg.norm(ds, axis=1)
        slip_axial = ds @ self.a
        disp = np.linalg.norm(now["tool"][:, :3] - self.P, axis=1)
        rot = twist_deg_xyzw(ini["tool"][:, 3:7], now["tool"][:, 3:7], u)
        return dict(fn_thumb_N=fpad[:, 0], fn_index_N=fpad[:, 1], torsion_capacity_Nm=cap_c,
                    torsion_capacity_about_P_Nm=cap_P, slip_mm=slip * 1e3, slip_axial_mm=slip_axial * 1e3,
                    tool_disp_mm=disp * 1e3, tool_twist_deg=rot, contacts=ncon.astype(float))

    def free(self):
        for k in ("cc", "pipe", "solver", "s0", "s1", "ctrl", "model", "tools", "hs", "times"):
            if hasattr(self, k):
                delattr(self, k)


def twist_deg_xyzw(q0, q1, u):
    inv = np.concatenate([-q0[:, :3], q0[:, 3:4]], axis=1)
    av, aw, bv, bw = q1[:, :3], q1[:, 3:4], inv[:, :3], inv[:, 3:4]
    v = aw * bv + bw * av + np.cross(av, bv)
    w = (aw * bw)[:, 0] - np.sum(av * bv, axis=1)
    return np.degrees(2.0 * np.arctan2(v @ u, w))


def summarize(f, ok):
    """min / median / max over the worlds that stayed finite and bounded, plus world 0."""
    out = {}
    for k, v in f.items():
        g = v[ok] if ok.any() else v
        out[k] = dict(w0=float(v[0]), min=float(np.min(g)), median=float(np.median(g)), max=float(np.max(g)))
    return out


def run_point(nworld, reduce, caps, duration=None, use_graph=True, poll_in_capture=False, model="hydro", fixture="hold"):
    duration = FIXTURES[fixture]["duration"] if duration is None else duration
    dev = wp.get_device("cuda:0")
    wp.synchronize()
    free_before = dev.free_memory
    t0 = time.perf_counter()
    sim = Sim(nworld, reduce, caps, model=model, fixture=fixture)
    nblocks = round(duration / (BLOCK * sim.dt))
    # first block eager: compiles kernels and starts the hold; it is not timed
    sim.step_block()
    wp.synchronize()
    graph, graph_error = None, None
    if use_graph:
        try:
            graph = sim.capture(poll_in_capture=poll_in_capture)
        except Exception as e:  # noqa: BLE001
            graph_error = repr(e)[:400]
            graph = None
    wp.synchronize()
    startup = time.perf_counter() - t0
    free_after = dev.free_memory
    snap_alloc = gpu_snapshot()
    ticks, peaks = [], dict(rigid_per_world_max=0, mj_per_world_max=0, nefc_max=0, niter_max=0, rigid_total=0,
                            mj_total=0, face_contacts=0, ht_insert_failures=0)
    hydro_peak = None
    first_bad = None
    a = sim.audit()
    for k in peaks:
        peaks[k] = max(peaks[k], a.get(k) or 0)
    hydro_peak = list(a.get("hydro_counts", []))
    for blk in range(1, nblocks):
        wp.synchronize()
        tick = time.perf_counter()
        if graph is not None:
            wp.capture_launch(graph)
        else:
            sim.step_block()
        wp.synchronize()
        ticks.append(time.perf_counter() - tick)
        a = sim.audit()
        for k in peaks:
            peaks[k] = max(peaks[k], a.get(k) or 0)
        if "hydro_counts" in a:
            hydro_peak = [max(x, y) for x, y in zip(hydro_peak, a["hydro_counts"])]
        if first_bad is None and not a["ok"].all():
            first_bad = dict(time_s=(blk + 1) * BLOCK * sim.dt, bad_worlds=int((~a["ok"]).sum()),
                             nonfinite=int((~a["fin"]).sum()))
        if not a["fin"].any():
            break
    ok = a["ok"]
    fid = sim.fidelity()
    elapsed = sum(ticks)
    steps = BLOCK * len(ticks)
    simulated = steps * sim.dt
    hydro = None
    if "hydro_caps" in a:
        hydro = dict(stage_peak=hydro_peak, stage_cap=a["hydro_caps"],
                     stage_peak_frac=[p / c for p, c in zip(hydro_peak, a["hydro_caps"])],
                     face_contacts_peak=peaks["face_contacts"], face_cap=a["face_cap"],
                     ht_insert_failures=peaks["ht_insert_failures"],
                     overflow=bool(any(p > c for p, c in zip(hydro_peak, a["hydro_caps"]))
                                   or peaks["face_contacts"] > a["face_cap"]))
    row = dict(
        status="complete" if ok.all() else ("failed" if not ok.any() else "partial"),
        failure=first_bad,
        graph=graph is not None, graph_error=graph_error,
        startup_s=startup, physics_s=elapsed, simulated_s_timed=simulated, steps_timed=steps,
        hold_s=nblocks * BLOCK * sim.dt,
        ms_per_step_median=float(np.median(ticks) / BLOCK * 1000) if ticks else None,
        ms_per_step_p90=float(np.percentile(ticks, 90) / BLOCK * 1000) if ticks else None,
        us_per_step_median=float(np.median(ticks) / BLOCK * 1e6) if ticks else None,
        world_steps_per_s=nworld * steps / elapsed if elapsed else None,
        sim_s_per_wall_s=nworld * simulated / elapsed if elapsed else None,
        per_world_real_time_factor=simulated / elapsed if elapsed else None,
        finite_worlds=int(a["fin"].sum()), bounded_worlds=int(ok.sum()),
        peak_displacement_mm=float(np.nanmax(a["disp"]) * 1e3) if a["fin"].any() else None,
        caps=dict(sim.caps, rigid_contact_max_total=nworld * sim.caps["rigid"],
                  naconmax=int(sim.solver.mjw_data.naconmax), njmax=int(sim.solver.mjw_data.njmax)),
        peaks=peaks,
        contact_overflow=bool(peaks["rigid_total"] > nworld * sim.caps["rigid"]
                              or peaks["mj_total"] > int(sim.solver.mjw_data.naconmax)),
        constraint_overflow=bool(peaks["nefc_max"] > int(sim.solver.mjw_data.njmax)),
        hydro=hydro,
        vram_alloc_mib=(free_before - free_after) / MIB,
        vram_process_mib=snap_alloc.get("own_mib"), gpu_free_after_alloc_mib=snap_alloc.get("free_mib"),
        fidelity=summarize(fid, ok),
        fallback_solref_replacements=sim.n_fix,
        mass_correction=getattr(sim, "mass_correction", None), pad_contact=sim.pad_info,
        host_maxrss_mib=__import__("resource").getrusage(__import__("resource").RUSAGE_SELF).ru_maxrss / 1024.0,
    )
    sim.free()
    del sim, graph
    gc.collect()
    wp.synchronize()
    return row


def right_size(cal):
    """Capacities from a calibration row (run at buffer_fraction 1): 25 % headroom on contacts
    and constraints, rounded up to 32. The hydroelastic iso-refinement and face-contact stages
    get a buffer fraction 1.5x their fullest stage; the broad phase, whose candidate count is
    every tile of the traversal SDF (it can never exceed that), gets an integer multiplier
    that restores at least its calibrated count at that fraction."""
    p = cal["peaks"]

    def up(x, h=1.25, q=32):
        return int(np.ceil(x * h / q) * q)

    rigid = up(max(p["rigid_per_world_max"], p["mj_per_world_max"]))
    frac, mult_broad = 1.0, 1
    h = cal.get("hydro")
    if h:
        fr = max(max(h["stage_peak_frac"][1:]), h["face_contacts_peak"] / h["face_cap"])
        frac = float(min(1.0, np.ceil(fr * 1.5 * 1000) / 1000))
        need = h["stage_peak_frac"][0]  # broad count / broad capacity at fraction 1
        mult_broad = int(np.ceil(need / frac * 1.02))
    return dict(rigid=rigid, nconmax=rigid, njmax=up(p["nefc_max"]), buffer_fraction=frac,
                buffer_mult_broad=mult_broad)


def model_name(reduce, model="hydro"):
    if model == "pads1":
        return "newton_pads1"
    base = "newton_hydro_mc" if model == "hydro_mc" else "newton_hydro"
    return base if reduce else base + "_unreduced" if model == "hydro" else "newton_hydro_unreduced_mc"


def key(kind, reduce, nworld, repeat, graph=True, model="hydro", fixture="hold"):
    fx = "" if fixture == "hold" else f"_{fixture}"
    return f"{kind}_{model_name(reduce, model)}{fx}_n{nworld}_r{repeat}{'' if graph else '_nograph'}"


def write(row, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as fh:
        fh.write(json.dumps(row, default=float) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def load(out):
    if not out.exists():
        return []
    return [json.loads(x) for x in out.read_text().splitlines() if x.strip()]


def base_row(kind, reduce, nworld, repeat, graph, caps, model="hydro", fixture="hold"):
    if model == "pads1":
        spec = "newton:spheres:s1:rs0.75:legacy_solref[0.015 1]:ir100:gap0.5:kfmatch"
    else:
        spec = (f"newton:hydro:kh{KH_PAD:.4g}:tool{TOOL_RATIO:g}x:vox0.5mm:kf10:ir100:"
                f"{'reduce' if reduce else 'noreduce'}:fallback_solref[{FALLBACK_SOLREF}]"
                f"{':kh_over_meff_solver' if model == 'hydro_mc' else ''}")
    fx_text = ("0.5 s hold, 20 N/m centring guide, no torque" if fixture == "hold" else
               "0.8 s, 20 N/m centring guide, two opposing 12 mN m raised-cosine torque pulses at 0.3-0.6 s "
               "(the MuJoCo-Warp pad fixture of gpu_scaling_mjwarp.py)")
    return dict(
        task="gpu_scaling", kind=kind, key=key(kind, reduce, nworld, repeat, graph, model, fixture),
        model=model_name(reduce, model), fixture_name=fixture, rig_spec=spec,
        fixture=f"SR2 thumb-index pinch, distributed_contact_transfer.build; {fx_text}; ctrl targets for 2 N pad load",
        dt_ms=1.0, nworld=nworld, repeat=repeat, reduce_contacts=reduce, requested_caps=caps,
        script="scripts/newton_scaling.py", git_rev=git_rev(), host=os.uname().nodename,
        versions=dict(newton=newton.__version__, warp=wp.__version__, mujoco=mujoco.__version__,
                      mujoco_warp=__import__("mujoco_warp").__version__ if hasattr(__import__("mujoco_warp"), "__version__") else None),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["calibrate", "sweep", "diag906"], default="sweep")
    ap.add_argument("--reduce", type=int, choices=[0, 1], nargs="+", default=[1, 0])
    ap.add_argument("--nworld", type=int, nargs="+", default=[1, 64, 256, 1024, 4096, 8192])
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--poll-in-capture", type=int, default=1)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--min-free-mib", type=float, default=2048.0)
    ap.add_argument("--max-wait-s", type=float, default=1800.0)
    ap.add_argument("--model", choices=MODELS, default="hydro")
    ap.add_argument("--fixture", choices=list(FIXTURES), default="hold")
    args = ap.parse_args()
    MK = dict(model=args.model, fixture=args.fixture)
    wp.init()
    try:
        wp.set_mempool_release_threshold("cuda:0", 0)
    except Exception:  # noqa: BLE001
        pass
    rows = load(args.out)

    if args.mode == "diag906":
        reduce = args.reduce[0]
        caps = GENEROUS
        snap, waited = wait_clear(args.max_wait_s)
        row = base_row("diag906", reduce, 1, int(args.poll_in_capture), True, caps, **MK)
        row.update(gpu_before=snap, waited_s=waited, poll_in_capture=bool(args.poll_in_capture))
        try:
            row.update(run_point(1, bool(reduce), caps, duration=0.1, poll_in_capture=bool(args.poll_in_capture), **MK))
        except Exception as e:  # noqa: BLE001
            row.update(status="error", error=repr(e)[:600], traceback=traceback.format_exc()[-2000:])
        row["gpu_after"] = gpu_snapshot()
        write(row, args.out)
        print(row["key"], row.get("status"), "graph", row.get("graph"), row.get("graph_error"), flush=True)
        return

    for reduce in args.reduce:
        reduce = bool(reduce)
        cal = [r for r in rows if r.get("kind") == "calibration" and r["reduce_contacts"] == reduce
               and r.get("model") == model_name(reduce, args.model) and r.get("fixture_name", "hold") == args.fixture
               and r.get("status") == "complete" and not r.get("contention")]
        if args.mode == "calibrate" or not cal:
            snap, waited = wait_clear(args.max_wait_s)
            row = base_row("calibration", reduce, 1, 0, True, GENEROUS, **MK)
            row.update(gpu_before=snap, waited_s=waited)
            try:
                row.update(run_point(1, reduce, GENEROUS, **MK))
            except Exception as e:  # noqa: BLE001
                row.update(status="error", error=repr(e)[:600], traceback=traceback.format_exc()[-2000:])
            row["gpu_after"] = gpu_snapshot()
            row["contention"] = contended(row["gpu_before"]) or contended(row["gpu_after"], after=True)
            if row.get("status") == "complete":
                row["right_sized_caps"] = right_size(row)
            write(row, args.out)
            rows.append(row)
            print(row["key"], row.get("status"), row.get("peaks"), row.get("right_sized_caps"),
                  row.get("hydro", {}) and row["hydro"].get("stage_peak_frac"), row.get("error"), flush=True)
            if args.mode == "calibrate":
                continue
            cal = [row] if row.get("status") == "complete" else []
        if not cal:
            print("no calibration for reduce", reduce, flush=True)
            continue
        caps = cal[-1]["right_sized_caps"]
        # VRAM per world from the largest batch run so far (smaller batches carry the fixed allocation)
        prev = [r for r in rows if r.get("kind") == "timing" and r.get("model") == model_name(reduce, args.model)
                and r.get("fixture_name", "hold") == args.fixture and r.get("nworld", 0) >= 64 and r.get("vram_alloc_mib")]
        big = max(prev, key=lambda r: r["nworld"], default=None)
        vram_per_world = big["vram_alloc_mib"] / big["nworld"] if big else 0.0
        for nworld in args.nworld:
            for rep in range(args.repeats):
                k = key("timing", reduce, nworld, rep, not args.no_graph, **MK)
                done = [r for r in rows if r.get("key") == k and not r.get("contention")
                        and r.get("status") in ("complete", "partial", "failed")]
                if done:
                    continue
                for attempt in range(2):
                    snap, waited = wait_clear(args.max_wait_s)
                    row = base_row("timing", reduce, nworld, rep, not args.no_graph, caps, **MK)
                    row.update(attempt=attempt, gpu_before=snap, waited_s=waited)
                    predicted = vram_per_world * nworld * 1.15
                    if predicted and snap.get("free_mib", 1e9) - predicted < args.min_free_mib:
                        row.update(status="skipped", failure=f"predicted {predicted:.0f} MiB would leave less "
                                   f"than {args.min_free_mib:.0f} MiB free (free {snap.get('free_mib')} MiB)")
                    else:
                        try:
                            row.update(run_point(nworld, reduce, caps, use_graph=not args.no_graph, **MK))
                        except Exception as e:  # noqa: BLE001
                            row.update(status="error", error=repr(e)[:600], traceback=traceback.format_exc()[-2000:])
                            gc.collect()
                    row["gpu_after"] = gpu_snapshot()
                    row["contention"] = contended(row["gpu_before"]) or contended(row["gpu_after"], after=True)
                    if row.get("vram_alloc_mib") and nworld >= 64:
                        vram_per_world = row["vram_alloc_mib"] / nworld
                    write(row, args.out)
                    rows.append(row)
                    f = row.get("fidelity") or {}
                    print(k, attempt, row.get("status"), f"{row.get('world_steps_per_s') or 0:.0f} ws/s",
                          f"{row.get('ms_per_step_median') or 0:.3f} ms/step", f"vram {row.get('vram_alloc_mib') or 0:.0f} MiB",
                          f"graph {row.get('graph')}", f"finite {row.get('finite_worlds')}/{nworld}",
                          "fn", f.get("fn_thumb_N", {}).get("median"), f.get("fn_index_N", {}).get("median"),
                          "cap", f.get("torsion_capacity_Nm", {}).get("median"),
                          "slip", f.get("slip_mm", {}).get("median"),
                          "CONTENDED" if row["contention"] else "", row.get("error", ""), flush=True)
                    if not row["contention"] or row.get("status") == "skipped":
                        break


if __name__ == "__main__":
    main()
