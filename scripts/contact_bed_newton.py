#!/usr/bin/env python3
"""Contact-model comparison bed: Newton hydroelastic rows for the static pinch, T1 pull, T2 twist,
T3 roll (when scripts/contact_bed_roll.py exists), T4 shake and T5 brake.

The rig is the two-pad pinch of `scripts/hom_contact_rig.py` (MJCF from `mj_xml`, the point3 variant:
one sphere geom per pad, R 10.55 mm, 20 g, on force-controlled x rails with 5 N s/m damping; the
screwdriver cylinder r 12.5 mm, 100 mm, 24.5 g, axis along world y; mu 1). The MJCF is imported into
Newton with every solref rewritten to the two-value soft fallback "0.5 1" before `add_mjcf`
(Newton's importer stores a one-value solref with damping ratio 0, which made earlier runs blow up;
see scripts/newton_pinch_probe.py). Pads and tool are then switched to hydroelastic SDF shapes:
kh_pad = E/R = 1e7/0.01055 = 9.48e8 N/m^3, kh_tool = 100 kh_pad, 0.5 mm voxels, +-6 mm narrow band (the native mapping lets a pad sink 2-4 mm at 3-6 N),
zero margin and gap, mu 1, friction gain kf 10, no torsional or rolling coefficient. SolverMuJoCo
(MuJoCo-Warp) with Newton's contacts, native stiffness mapping, elliptic cone, impratio 100,
implicitfast, 200 iterations. Contact solimp is ".9 .9 .001 .5 2" on pads and tool, the fingertip
value of the SR2 probe rig. Models: `newton_hydro` (contact reduction on, anchor contact) and
`newton_hydro_unreduced`.

The task logic is imported where it exists: T1 is `contact_bed_pull.run_case` on a Newton rig object
with the MjRig/DrakeRig interface, T5 is `hom_contact_rig.exp_brake`, the T2 kinetic torque is
`hom_contact_rig.exp_torsion`, T2 uses `contact_bed_twist.py` when present. One fsynced JSON line per
case; cases already in the output are skipped.

    PY=logs/20261004-contact-transfer/venv/bin/python    # newton 1.7.0.dev0, warp 1.17, mujoco 3.14
    WARP_CACHE_PATH=$(mktemp -d) MUJOCO_GL=egl $PY scripts/contact_bed_newton.py static
    WARP_CACHE_PATH=$(mktemp -d) MUJOCO_GL=egl $PY scripts/contact_bed_newton.py pull --models newton_hydro --dt 1 5
    logs/20261001-hom_contact/venv/bin/python scripts/contact_bed_newton.py drake-static   # Drake reference rows
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import hom_contact_rig as H  # noqa: E402

OUTDIR = ROOT / "docs/experiments/20261005-contact_bed"
FILMDIR = OUTDIR / "media"
SCRATCH = ROOT / "logs/20261005-contact-bed-newton"
KH_PAD = 1e7 / H.R_PAD
SCRIPT = "scripts/contact_bed_newton.py"
MODELS = {
    "newton_hydro": dict(reduce=True),
    "newton_hydro_unreduced": dict(reduce=False),
}
BASE = dict(kh=KH_PAD, tool_ratio=100.0, voxel=0.5e-3, band=0.006, impratio=100.0, kf=10.0,
            solimp="0.9 0.9 0.001 0.5 2", fallback="0.5 1", iterations=200, ls_iterations=50,
            tolerance=1e-8)


def git_rev():
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:
        return None


def rig_spec(model):
    c = dict(BASE, **MODELS[model])
    return (f"newton:hydro:kh{c['kh']:.3g}:tool{c['tool_ratio']:g}:vox{c['voxel'] * 1e3:g}:ir{c['impratio']:g}"
            f":kf{c['kf']:g}:{'reduced' if c['reduce'] else 'unreduced'}:solimp0.9:fallback0.5x1")


def done(path, key):
    have = set()
    if Path(path).exists():
        for line in open(path):
            try:
                have.add(key(json.loads(line)))
            except Exception:
                pass
    return have


def finish_row(row, task, model, dt_ms, status="complete", film=None):
    row.update(task=task, model=model, rig_spec=rig_spec(model) if model in MODELS else row.get("rig_spec"),
               dt_ms=dt_ms, status=row.get("status", status), film=film, script=SCRIPT, git_rev=git_rev(),
               when=time.strftime("%Y-%m-%d %H:%M"))
    return row


# ------------------------------------------------------------------------------------ Newton rig

class NewtonRig:
    """The hom_contact_rig two-pad pinch in Newton, with the MjRig / DrakeRig interface
    (set_pad_force, set_tool_wrench, step, contacts, tool_state, pad_x, t)."""

    sim = "newton"

    def __init__(self, model="newton_hydro", dt=1e-3, d_cg=0.0, gravity=False, gravity_vec=None, recorder=None,
                 xml_hook=None):
        import newton
        import warp as wp
        from newton.geometry import HydroelasticSDF

        self.wp, self.newton = wp, newton
        c = dict(BASE, **MODELS[model])
        self.cfg, self.model_name, self.dt = c, model, dt
        sp = H.parse_spec("mj:point3:ir100")
        dt_old = H.DT
        H.DT = dt
        xml, _ = H.mj_xml(sp, d_cg, gravity)
        H.DT = dt_old
        if xml_hook is not None:
            xml = xml_hook(xml)
        xml = re.sub(r'solref="[^"]*"', f'solref="{c["fallback"]}"', xml)
        xml = re.sub(r'solimp="[^"]*"', f'solimp="{c["solimp"]}"', xml)
        self.xml = xml
        g = gravity_vec if gravity_vec is not None else ((0.0, 0.0, -H.G) if gravity else (0.0, 0.0, 0.0))
        b = newton.ModelBuilder(gravity=tuple(float(x) for x in g))
        b.add_mjcf(xml, ctrl_direct=True, parse_sites=False, parse_visuals=False)
        self.hydro_shapes = {}
        for i, label in enumerate(b.shape_label):
            name = label.split("/")[-1]
            if name not in ("padL", "padR", "tool"):
                continue
            b.shape_flags[i] |= int(newton.ShapeFlags.HYDROELASTIC)
            b.shape_sdf_target_voxel_size[i] = c["voxel"]
            b.shape_sdf_narrow_band_range[i] = (-c["band"], c["band"])
            b.shape_sdf_padding[i] = c["band"]
            b.shape_margin[i] = 0.0
            b.shape_gap[i] = 0.0
            b.shape_material_kh[i] = c["kh"] * (c["tool_ratio"] if name == "tool" else 1.0)
            b.shape_material_mu[i] = H.MU
            b.shape_material_kf[i] = c["kf"]
            b.shape_material_mu_torsional[i] = 0.0
            b.shape_material_mu_rolling[i] = 0.0
            self.hydro_shapes[i] = name
        self.model = b.finalize(device="cuda:0")
        hc = HydroelasticSDF.Config(reduce_contacts=c["reduce"], anchor_contact=True)
        self.pipe = newton.CollisionPipeline(self.model, reduce_contacts=c["reduce"], rigid_contact_max=8192,
                                             broad_phase="explicit", sdf_hydroelastic_config=hc)
        self.solver = newton.solvers.SolverMuJoCo(
            self.model, use_mujoco_contacts=False, disable_sensors=True, njmax=16384, nconmax=8192,
            iterations=c["iterations"], ls_iterations=c["ls_iterations"], cone="elliptic", jacobian="dense",
            integrator="implicitfast", tolerance=c["tolerance"], impratio=c["impratio"])
        self.s0, self.s1 = self.model.state(), self.model.state()
        self.ctrl = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.s0)
        self.cc = self.pipe.contacts()
        labels = [x.split("/")[-1] for x in self.model.body_label]
        self.tool = labels.index("tool")
        self.pads = {s: labels.index("pad" + s) for s in "LR"}
        self.shape_side = {i: n[-1] for i, n in self.hydro_shapes.items() if n.startswith("pad")}
        mjm = self.solver.mj_model
        self.geom_side = {}
        for gi in range(mjm.ngeom):
            nm = re.sub(r"_\d+$", "", (mjm.geom(gi).name or "").split("/")[-1])
            if nm in ("padL", "padR"):
                self.geom_side[gi] = nm[-1]
        self.kh_eff = c["kh"] * c["kh"] * c["tool_ratio"] / (c["kh"] + c["kh"] * c["tool_ratio"])
        self.N = 0.0
        self.f_tool, self.tau_tool = np.zeros(3), np.zeros(3)
        self._t = 0.0
        self.theta_prev, self.theta_unwrap = None, 0.0
        self.recorder = recorder
        self.ctrl.mujoco.ctrl.assign(np.zeros(2, dtype=np.float32))
        self._nonfinite = False

        @wp.kernel
        def _wrench(body_f: wp.array(dtype=wp.spatial_vector), bi: int, f: wp.vec3, tau: wp.vec3):
            body_f[bi] = wp.spatial_vector(f, tau)

        self._wrench_kernel = _wrench

    @property
    def t(self):
        return self._t

    def set_pad_force(self, N):
        self.N = float(N)
        self.ctrl.mujoco.ctrl.assign(np.array([N, -N], dtype=np.float32))

    def set_tool_wrench(self, f, tau):
        self.f_tool, self.tau_tool = np.asarray(f, float), np.asarray(tau, float)

    def step(self, T):
        wp = self.wp
        n = max(1, int(round(T / self.dt)))
        f = wp.vec3(*[float(x) for x in self.f_tool])
        tau = wp.vec3(*[float(x) for x in self.tau_tool])
        for _ in range(n):
            self.s0.clear_forces()
            wp.launch(self._wrench_kernel, dim=1, inputs=[self.s0.body_f, self.tool, f, tau])
            self.pipe.collide(self.s0, self.cc)
            self.solver.step(self.s0, self.s1, self.ctrl, self.cc, self.dt)
            self.s0, self.s1 = self.s1, self.s0
            self._t += self.dt
            if self.recorder is not None:
                self.recorder.maybe(self)
        wp.synchronize()

    # ---- readout
    def mj_contacts(self):
        """Contacts the MuJoCo-Warp solve used: position, normal, normal force, pad side."""
        d = self.solver.mjw_data
        n = int(d.nacon.numpy()[0])
        if n == 0:
            return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0), []
        pos = d.contact.pos.numpy()[:n]
        frame = d.contact.frame.numpy()[:n]
        geom = d.contact.geom.numpy()[:n]
        addr = d.contact.efc_address.numpy()[:n, 0]
        force = d.efc.force.numpy()
        force = force[0] if force.ndim == 2 else force
        fn = np.where(addr >= 0, force[np.clip(addr, 0, None)], 0.0)
        side = [self.geom_side.get(int(g0)) or self.geom_side.get(int(g1)) for g0, g1 in geom]
        return pos, frame[:, 0, :], fn, side

    def contacts(self):
        pos, nrm, fn, side = self.mj_contacts()
        out = {s: {"N": 0.0, "cop": np.zeros(3), "n": 0} for s in "LR"}
        for s in "LR":
            m = np.array([x == s for x in side], bool)
            if m.any():
                F = float(fn[m].sum())
                out[s]["N"] = F
                out[s]["n"] = int((fn[m] > 1e-6).sum())
                if F > 1e-9:
                    out[s]["cop"] = (fn[m][:, None] * pos[m]).sum(0) / F
        return out

    def newton_contacts(self):
        """Newton's own hydroelastic contact set: stiffness k_i (area x kh_eff for a face), signed
        distance, world point, side."""
        wp, newton = self.wp, self.newton
        cc = self.cc
        n = int(cc.rigid_contact_count.numpy()[0])
        if n == 0:
            return dict(n=0)
        dist = wp.zeros(cc.rigid_contact_max, dtype=float)
        p0 = wp.zeros(cc.rigid_contact_max, dtype=wp.vec3)
        p1 = wp.zeros(cc.rigid_contact_max, dtype=wp.vec3)
        newton.eval_rigid_contact_kinematics(self.model, self.s0, cc, out_distance=dist, out_point0_world=p0,
                                             out_point1_world=p1)
        s0 = cc.rigid_contact_shape0.numpy()[:n]
        s1 = cc.rigid_contact_shape1.numpy()[:n]
        side = [self.shape_side.get(int(a)) or self.shape_side.get(int(b)) for a, b in zip(s0, s1)]
        return dict(n=n, k=cc.rigid_contact_stiffness.numpy()[:n].astype(float), d=dist.numpy()[:n].astype(float),
                    p=0.5 * (p0.numpy()[:n] + p1.numpy()[:n]), side=side)

    def tool_state(self):
        q = self.s0.body_q.numpy()[self.tool].astype(float)
        v = self.s0.body_qd.numpy()[self.tool].astype(float)
        if not (np.isfinite(q).all() and np.isfinite(v).all()):
            self._nonfinite = True
        x, y, z, w = q[3:7]
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        a = R[:, 2]
        theta = math.atan2(a[2], a[1])
        if self.theta_prev is not None:
            dth = (theta - self.theta_prev + math.pi) % (2 * math.pi) - math.pi
            self.theta_unwrap += dth
        else:
            self.theta_unwrap = theta
        self.theta_prev = theta
        return {"theta": self.theta_unwrap, "omega_x": float(v[3]), "omega": v[3:6].copy(), "vel": v[0:3].copy(),
                "pos": q[:3].copy(), "quat_xyzw": q[3:7].copy(), "axis": a.copy(), "R": R}

    def pad_x(self):
        q = self.s0.body_q.numpy()
        return {s: float(q[b, 0]) for s, b in self.pads.items()}


# ------------------------------------------------------------------------------------- films

class Recorder:
    """Snapshots of the Newton state every `every` seconds of simulated time, for later drawing."""

    def __init__(self, every, info=None, max_frames=600):
        self.every, self.next, self.frames, self.info, self.max_frames = every, 0.0, [], info, max_frames

    def maybe(self, rig):
        if rig.t + 1e-9 < self.next or len(self.frames) >= self.max_frames:
            return
        self.next = rig.t + self.every
        st = rig.tool_state()
        pos, nrm, fn, side = rig.mj_contacts()
        keep = fn > 1e-7
        self.frames.append(dict(t=rig.t, tool_pos=st["pos"], tool_quat=st["quat_xyzw"], pad_x=rig.pad_x(),
                                cpos=pos[keep], cfn=fn[keep], cside=[x for x, k in zip(side, keep) if k], N=rig.N,
                                text=self.info(rig, st) if self.info else "", every=self.every))


LABEL = {"newton_hydro": "Newton hydroelastic, reduced contacts",
         "newton_hydro_unreduced": "Newton hydroelastic, all faces"}


def render_film(frames, path, model, d_cg=0.0, view=None, poster=None, fscale=None):
    """Draw recorded Newton states through the bed's film model (contact_bed_common.BedFilm: 480x360,
    pads translucent, close-up inset of the -x pad). Newton's contact points are drawn as spheres
    coloured by the solver's normal force per contact, 0 to `fscale` (98th percentile over the film)."""
    import mujoco
    import contact_bed_common as B
    B.SHORT.update(LABEL)
    allf = np.concatenate([f["cfn"] for f in frames if len(f["cfn"])]) if any(len(f["cfn"]) for f in frames) else np.ones(1)
    fscale = fscale or float(np.percentile(allf, 98))
    film = B.BedFilm("mj:point3:ir100", d_cg, 480, 360, model=model, view=view or ((0.0, 0.0, 0.0), 0.13, 18.0, -14.0))
    film.point = film.spheres = False
    m, d = film.m, film.d
    qa = m.jnt_qposadr[m.body_jntadr[film.tool]]
    imgs = []

    def deco(scn, fr, closeup):
        for p, f, sd in zip(fr["cpos"], fr["cfn"], fr["cside"]):
            if closeup and sd != "L":
                continue
            B._add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.00035 if closeup else 0.0005] * 3, p, np.eye(3),
                        B.heat(f / max(fscale, 1e-12)))

    for fr in frames:
        d.qpos[:] = 0
        d.qpos[qa:qa + 3] = fr["tool_pos"]
        x, y, z, w = fr["tool_quat"]
        d.qpos[qa + 3:qa + 7] = [w, x, y, z]
        for sd in "LR":
            d.qpos[m.joint("rail" + sd).qposadr[0]] = fr["pad_x"][sd] - (-H.X0 if sd == "L" else H.X0)
        mujoco.mj_forward(m, d)
        m.geom_rgba[:] = film.rgba0
        for g in film.pad_env:
            m.geom_rgba[g, 3] = 0.18
        film.r.update_scene(d, film.cam)
        deco(film.r.scene, fr, False)
        img = film.r.render().copy()
        for g in film.tool_g:
            m.geom_rgba[g, 3] = 0.30
        for g in film.pad_env:
            m.geom_rgba[g, 3] = 0.10
        for g in film.right_g:
            m.geom_rgba[g, 3] = 0.0
        film.ri.update_scene(d, film.cami)
        deco(film.ri.scene, fr, True)
        ins = np.array(H.annotate(film.ri.render().copy(), f"-x pad, 0-{fscale * 1e3:.3g} mN/contact", size=11))
        ins[[0, -1], :, :] = 90
        ins[:, [0, -1], :] = 90
        y0, x0 = 30, img.shape[1] - film.inset - 6
        img[y0:y0 + film.inset, x0:x0 + film.inset] = ins
        m.geom_rgba[:] = film.rgba0
        img = H.annotate(img, f"{model}: {LABEL[model]}   {fr['every'] * 25:g}x", size=14)
        img = H.annotate_bottom(img, f"t {fr['t']:5.3f} s   " + fr["text"], size=13)
        imgs.append(np.asarray(img))
    film.close()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    if poster:
        Image.fromarray(imgs[min(len(imgs) - 1, int(len(imgs) * poster))][:, :, :3]).save(path.with_suffix(".jpg"), quality=88)
    if str(path).endswith(".png"):
        Image.fromarray(imgs[len(imgs) // 2][:, :, :3]).save(path)
        return path
    h, w = imgs[0].shape[:2]
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "25",
           "-i", "-", "-c:v", "libx264", "-crf", "27", "-pix_fmt", "yuv420p", "-preset", "medium",
           "-movflags", "+faststart", str(path)]
    pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for im in imgs:
        pr.stdin.write(np.ascontiguousarray(im[:, :, :3]).tobytes())
    pr.stdin.close()
    pr.wait()
    return path


def save_frames(frames, name):
    import pickle
    SCRATCH.mkdir(parents=True, exist_ok=True)
    with open(SCRATCH / f"{name}.pkl", "wb") as fh:
        pickle.dump(frames, fh)


# ------------------------------------------------------------------------------------ static

def static_newton(model, N, dt_ms, T=1.0):
    dt = dt_ms * 1e-3
    t0w = time.time()
    rig = NewtonRig(model, dt, gravity=False)
    rig.set_pad_force(N)
    rig.set_tool_wrench(np.zeros(3), np.zeros(3))
    snaps = []
    W = []
    status = "complete"
    for k in range(int(round(T / 0.1))):
        w0 = time.perf_counter()
        rig.step(0.1)
        W.append((time.perf_counter() - w0) / max(1, int(round(0.1 / dt))))
        st = rig.tool_state()
        px = rig.pad_x()
        c = rig.contacts()
        snaps.append(dict(t=round(rig.t, 4), pen_L_mm=(H.X0 + px["L"]) * 1e3, pen_R_mm=(H.X0 - px["R"]) * 1e3,
                          N_L=c["L"]["N"], N_R=c["R"]["N"], tool_disp_mm=float(np.linalg.norm(st["pos"])) * 1e3))
        if rig._nonfinite or abs(snaps[-1]["pen_L_mm"]) > 6:
            status = "failed"
            break
    row = dict(task="static", N=N, settle_s=T, trace=snaps)
    if status == "complete":
        pos, nrm, fn, side = rig.mj_contacts()
        nc = rig.newton_contacts()
        for s in "LR":
            m = np.array([x == s for x in side], bool)
            rho = np.hypot(pos[m, 1], pos[m, 2])
            F = float(fn[m].sum())
            row[f"N_{s}"] = F
            row[f"n_solver_{s}"] = int((fn[m] > 1e-7).sum())
            row[f"rbar_solver_{s}_mm"] = float((fn[m] * rho).sum() / max(F, 1e-12)) * 1e3
            row[f"patch_rmax_solver_{s}_mm"] = float(rho[fn[m] > 1e-7].max()) * 1e3 if (fn[m] > 1e-7).any() else 0.0
            mm = np.array([x == s for x in nc["side"]], bool) if nc["n"] else np.zeros(0, bool)
            if mm.any():
                k, dd, p = nc["k"][mm], nc["d"][mm], nc["p"][mm]
                pen = np.clip(-dd, 0, None)
                fl = k * pen
                rh = np.hypot(p[:, 1], p[:, 2])
                row[f"n_newton_{s}"] = int(mm.sum())
                row[f"area_eq_{s}_mm2"] = float(k.sum() / rig.kh_eff) * 1e6
                row[f"F_law_{s}"] = float(fl.sum())
                row[f"rbar_law_{s}_mm"] = float((fl * rh).sum() / max(fl.sum(), 1e-12)) * 1e3
                row[f"overlap_max_{s}_mm"] = float(pen.max()) * 1e3
        row["pen_mm"] = 0.5 * (snaps[-1]["pen_L_mm"] + snaps[-1]["pen_R_mm"])
        row["pen_change_last_0p1s_mm"] = snaps[-1]["pen_L_mm"] - snaps[-2]["pen_L_mm"] if len(snaps) > 1 else None
        row["pen_at_0p4s_mm"] = next((s["pen_L_mm"] for s in snaps if s["t"] >= 0.4 - 1e-9), None)
        row["rbar_solver_mm"] = 0.5 * (row["rbar_solver_L_mm"] + row["rbar_solver_R_mm"])
        row["N_mean"] = 0.5 * (row["N_L"] + row["N_R"])
        row["solver_over_law"] = (row["N_L"] + row["N_R"]) / max(row.get("F_law_L", 0) + row.get("F_law_R", 0), 1e-12)
        row["rbar_law_analytic_mm"] = 0.996 * N ** 0.25
        st_, why = pinch_check({"N_L": row["N_L"], "N_R": row["N_R"], "n_L": row["n_solver_L"],
                                "n_R": row["n_solver_R"]}, N)
        if why:
            status, row["failure"] = st_, why
        dl, rb, _ = H.winkler_law(N, 1e7)
        row["winkler_E1e7_delta_mm"], row["winkler_E1e7_rbar_mm"] = dl * 1e3, rb * 1e3
    row["status"] = status
    row["us_per_step_median"] = float(np.median(W) * 1e6)
    row["wall_s"] = time.time() - t0w
    return finish_row(row, "static", model, dt_ms)


def drake_static(N, dt_ms, T=1.0):
    """Drake hydro reference (run under the rig venv): pad approach, normal force, contact area
    and pressure-weighted arm about the pinch axis per pad."""
    t0w = time.time()
    H.DT = dt_ms * 1e-3
    spec = "drake:hydro:E1e7:r1:rt0.01"
    sp = H.parse_spec(spec)
    sp["dt"] = dt_ms * 1e-3
    rig = H.DrakeRig(sp, 0.0, False)
    rig.set_pad_force(N)
    rig.set_tool_wrench(np.zeros(3), np.zeros(3))
    rig.step(T)
    cr = rig.plant.get_contact_results_output_port().Eval(rig.pc)
    row = dict(task="static", model="drake_hydro", rig_spec=spec, N=N, settle_s=T)
    px = rig.pad_x()
    row["pen_mm"] = 0.5 * ((H.X0 + px["L"]) + (H.X0 - px["R"])) * 1e3
    for i in range(cr.num_hydroelastic_contacts()):
        info = cr.hydroelastic_contact_info(i)
        srf = info.contact_surface()
        s = rig.geom_side.get(srf.id_M()) or rig.geom_side.get(srf.id_N())
        tri = srf.is_triangle()
        mesh = srf.tri_mesh_W() if tri else srf.poly_mesh_W()
        field = srf.tri_e_MN() if tri else srf.poly_e_MN()
        nf = mesh.num_elements()
        A = np.array([mesh.area(f) for f in range(nf)])
        C = np.array([mesh.element_centroid(f) for f in range(nf)])
        P = np.array([field.EvaluateCartesian(f, C[f]) for f in range(nf)])
        rho = np.hypot(C[:, 1], C[:, 2])
        F = np.array(info.F_Ac_W().translational())
        row[f"N_{s}"] = float(abs(F[0]))
        row[f"N_pressure_{s}"] = float((P * A).sum())
        row[f"area_{s}_mm2"] = float(A.sum()) * 1e6
        row[f"rbar_pressure_{s}_mm"] = float((P * A * rho).sum() / (P * A).sum()) * 1e3
        row[f"patch_rmax_{s}_mm"] = float(rho.max()) * 1e3
        row[f"faces_{s}"] = int(nf)
    row["N_mean"] = 0.5 * (row.get("N_L", 0) + row.get("N_R", 0))
    row["rbar_pressure_mm"] = 0.5 * (row.get("rbar_pressure_L_mm", 0) + row.get("rbar_pressure_R_mm", 0))
    row["area_mm2"] = 0.5 * (row.get("area_L_mm2", 0) + row.get("area_R_mm2", 0))
    row["rbar_law_analytic_mm"] = 0.996 * N ** 0.25
    row["wall_s"] = time.time() - t0w
    row.update(dt_ms=dt_ms, status="complete", film=None, script=SCRIPT, git_rev=git_rev(),
               when=time.strftime("%Y-%m-%d %H:%M"))
    return row


# ------------------------------------------------------------------------------------- T1 pull

def run_pull(model, N, dt_ms, film=None):
    import contact_bed_pull as P
    rec = {}

    def factory(spec, dt):
        if film and "rec" not in rec:
            rec["rec"] = Recorder(0.02, info=lambda r, st: f"N {r.N:4.2f} N   pull {float(np.linalg.norm(r.f_tool)):5.3f} N   "
                                                         f"u {float((st['pos'][1])) * 1e3:6.3f} mm")
            return NewtonRig(model, dt, gravity=False, recorder=rec["rec"])
        return NewtonRig(model, dt, gravity=False)

    P.MODELS[model] = rig_spec(model)
    P.new_rig = factory
    row = P.run_case(model, N, dt_ms)
    row.pop("exp", None)
    row["exp"] = "pull_slip"
    fpath = None
    if film and "rec" in rec:
        fr = rec["rec"].frames
        save_frames(fr, f"pull_{model}_N{N:g}_dt{dt_ms:g}")
        fpath = render_film(fr, film, model)
    status = "complete" if row.get("slipped") else "failed"
    status, why = pinch_check(row.get("settle", {}), N, row.get("hold_disp_mm"), row.get("mu_eff"), status)
    if why:
        row["failure"] = why
    return finish_row(row, "pull", model, dt_ms, status, film=str(Path(fpath).relative_to(OUTDIR)) if fpath else None)


def pinch_check(settle, N, hold_disp_mm=None, mu_eff=None, status="complete"):
    """`ejected` when the tool left the pinch (a pad without contact, a hold displacement over 5 mm or
    mu_eff under 0.5); `failed` when a pad's normal force at the end of the settle is more than 30 %
    off N (the pads oscillate). Returns (status, reason)."""
    hd = hold_disp_mm if hold_disp_mm is not None and math.isfinite(hold_disp_mm) else 0.0
    if settle.get("n_L", 1) == 0 or settle.get("n_R", 1) == 0 or abs(hd) > 5 or (mu_eff is not None and mu_eff < 0.5):
        return "ejected", (f"tool left the pinch: hold displacement {hd:.1f} mm, mu_eff {mu_eff}, "
                           f"settle n_L {settle.get('n_L')} n_R {settle.get('n_R')}")
    if abs(settle.get("N_L", N) - N) > 0.3 * N or abs(settle.get("N_R", N) - N) > 0.3 * N:
        return "failed", (f"pads oscillate at the end of the settle: N_L {settle['N_L']:.2f} N, "
                          f"N_R {settle['N_R']:.2f} N for N = {N:g} N")
    return status, None


# ------------------------------------------------------------------------------------ T2 twist

def run_twist(model, N, dt_ms, film=None):
    """contact_bed_twist.run_case on Newton rigs (ramp, hold, then exp_torsion's driven 1 rad/s spin)."""
    import contact_bed_common as B
    import contact_bed_twist as TW
    B.MODELS[model] = (rig_spec(model), rig_spec(model))
    rec = {"n": 0}

    def info(r, st):
        th0 = r.__dict__.setdefault("theta0", st["theta"])
        return f"N {r.N:4.2f} N   tau {r.tau_tool[0] * 1e3:6.3f} mN m   theta {math.degrees(st['theta'] - th0):+7.2f} deg"

    def new_rig(spec, dt, d_cg=0.0, gravity=False):
        rec["n"] += 1
        recorder = None
        if film and rec["n"] == 1:
            recorder = rec["ramp"] = Recorder(0.02, info=info)
        return NewtonRig(model, dt, d_cg=d_cg, gravity=gravity, recorder=recorder)

    def make_rig(spec, d_cg=0.0, gravity=True, kinematic=False):
        recorder = rec["spin"] = Recorder(0.04, info=info) if film else None
        return NewtonRig(model, H.DT, d_cg=d_cg, gravity=gravity, recorder=recorder)

    B.new_rig = new_rig
    H.make_rig = make_rig
    row = TW.run_case(model, N, dt_ms, film=False)
    row["logic"] = "contact_bed_twist.run_case"
    fpath = None
    if film:
        fr = rec["ramp"].frames + (rec["spin"].frames if rec.get("spin") else [])
        save_frames(fr, f"twist_{model}_N{N:g}_dt{dt_ms:g}")
        fpath = render_film(fr, film, model, poster=0.5)
        row["film_segments"] = [len(rec["ramp"].frames), len(fr) - len(rec["ramp"].frames)]
    return finish_row(row, "twist", model, dt_ms, row.get("status", "complete"),
                      film=str(Path(fpath).relative_to(OUTDIR)) if fpath else None)


# ------------------------------------------------------------------------------------ T4 shake

def run_shake(model, N, a_g, dt_ms, film=None):
    """contact_bed_shake.run_case on a Newton rig: settle 0.4 s at N with the weight compensated,
    0.5 s with the weight -m g along y (the tool axis) as a body force, then 2 s of m a_pk sin(w t)
    along y at 5 Hz."""
    import contact_bed_common as CB
    import contact_bed_shake as SH
    CB.MODELS[model] = (rig_spec(model), rig_spec(model))
    rec = Recorder(0.01, info=lambda r, st: f"N {r.N:4.2f} N  a {a_g:g} g  F_y {r.f_tool[1] * 1e3:+6.1f} mN  "
                                                f"y {st['pos'][1] * 1e3:+7.3f} mm") if film else None

    def new_rig(spec, dt, d_cg=0.0, gravity=False):
        return NewtonRig(model, dt, d_cg=d_cg, gravity=gravity, recorder=rec)

    CB.new_rig = new_rig
    row = SH.run_case(model, N, dt_ms, a_g, film=None)
    row["logic"] = "contact_bed_shake.run_case"
    fpath = None
    if film:
        save_frames(rec.frames, f"shake_{model}_N{N:g}_a{a_g:g}_dt{dt_ms:g}")
        fpath = render_film(rec.frames, film, model, poster=0.6)
        row["film_playback"] = 0.25
    return finish_row(row, "shake", model, dt_ms, row.get("status", "complete"),
                      film=str(Path(fpath).relative_to(OUTDIR)) if fpath else None)


# ------------------------------------------------------------------------------------ T5 brake

def run_brake(model, dt_ms, film=None):
    """contact_bed_brake.run_case (hom_contact_rig.exp_brake with per-step angles) on a Newton rig:
    gravity on, CG offset 15 mm, pinch 6 N held 0.5 s with the tool horizontal, lowered
    geometrically to 0.2 N over 4 s, held 1 s."""
    import contact_bed_brake as BR
    import contact_bed_common as B
    B.MODELS[model] = (rig_spec(model), rig_spec(model))
    rec = {}

    def info(r, st):
        return f"N {r.N:5.3f} N   phi {-math.degrees(st['theta']):6.1f} deg"

    def make_rig(spec, d_cg=0.0, gravity=True, kinematic=False):
        rec["r"] = Recorder(0.04, info=info) if film else None
        return NewtonRig(model, H.DT, d_cg=d_cg, gravity=gravity, recorder=rec["r"])

    H.make_rig = make_rig
    row = BR.run_case(model, dt_ms, film=False)
    row["logic"] = "contact_bed_brake.run_case"
    fpath = None
    if film and rec.get("r"):
        save_frames(rec["r"].frames, f"brake_{model}_dt{dt_ms:g}")
        fpath = render_film(rec["r"].frames, film, model, d_cg=BR.D_CG, poster=0.5, view=BR.VIEW)
    return finish_row(row, "brake", model, dt_ms, row.get("status", "complete"),
                      film=str(Path(fpath).relative_to(OUTDIR)) if fpath else None)


# ------------------------------------------------------------------------------------- T3 roll

class NewtonRollRig(NewtonRig):
    """The roll rig of contact_bed_roll.py in Newton: the +x pad gets the z slide `liftR` (armature
    M_LIFT) after its x rail; the slide is driven by a generalized force (Control.joint_f)."""

    def __init__(self, model, dt, recorder=None):
        import contact_bed_roll as RL
        super().__init__(model, dt, recorder=recorder, xml_hook=RL.add_lift)
        m = self.model
        jc = m.joint_child.numpy()
        j = int(np.where(jc == self.pads["R"])[0][0])
        self.lift_qi = int(m.joint_q_start.numpy()[j]) + 1
        self.lift_qd = int(m.joint_qd_start.numpy()[j]) + 1
        self.lift_mass = RL.M_LIFT + H.M_PAD
        self._jf = np.zeros(m.joint_dof_count, dtype=np.float32)

    def set_lift_force(self, u):
        self._jf[self.lift_qd] = u
        self.ctrl.joint_f.assign(self._jf)

    def lift_state(self):
        return float(self.s0.joint_q.numpy()[self.lift_qi]), float(self.s0.joint_qd.numpy()[self.lift_qd])

    def tool_kin(self):
        st = self.tool_state()
        return st["pos"], st["R"], st["vel"], st["omega"]

    def pad_kin(self):
        q = self.s0.body_q.numpy()
        v = self.s0.body_qd.numpy()
        return {s: (q[b, :3].astype(float), v[b, :3].astype(float)) for s, b in self.pads.items()}

    def contact_patches(self, detail=False):
        d = self.solver.mjw_data
        out = {s: {"F": np.zeros(3), "cop": None, "n": 0, "pts": []} for s in "LR"}
        n = int(d.nacon.numpy()[0])
        if n == 0:
            return out
        pos = d.contact.pos.numpy()[:n].astype(float)
        frame = d.contact.frame.numpy()[:n].astype(float)
        geom = d.contact.geom.numpy()[:n]
        addr = d.contact.efc_address.numpy()[:n, 0]
        force = d.efc.force.numpy()
        force = (force[0] if force.ndim == 2 else force).astype(float)
        acc = {s: np.zeros(3) for s in "LR"}
        wsum = {s: 0.0 for s in "LR"}
        for i in range(n):
            g0, g1 = int(geom[i][0]), int(geom[i][1])
            s = self.geom_side.get(g0) or self.geom_side.get(g1)
            if s is None or addr[i] < 0:
                continue
            f3 = force[addr[i]:addr[i] + 3]
            fr = frame[i].reshape(3, 3)
            f = fr.T @ f3
            if g1 in self.geom_side:
                f = -f
            out[s]["F"] += f
            fn = float(f3[0])
            if fn > 1e-9:
                acc[s] += fn * pos[i]
                wsum[s] += fn
                out[s]["n"] += 1
                if detail:
                    out[s]["pts"].append((pos[i].copy(), fr[0].copy(), fn, 0.0005, "f"))
        for s in "LR":
            if wsum[s] > 0:
                out[s]["cop"] = acc[s] / wsum[s]
        return out


def run_roll(model, N, dt_ms, v_mm_s, film=None):
    """contact_bed_roll.run_roll + summarize on the Newton roll rig."""
    import contact_bed_roll as RL
    dt, V, T = dt_ms * 1e-3, v_mm_s * 1e-3, RL.SPEEDS[v_mm_s]
    t0w = time.time()
    row = {"task": "roll", "N": N, "v_mm_s": v_mm_s, "T_s": T, "travel_mm": v_mm_s * T, "ramp_ms": RL.T_RAMP * 1e3,
           "T_settle_s": RL.T_SETTLE, "T_hold_s": RL.T_HOLD, "mu": H.MU, "gravity": False,
           "lift_mass_kg": RL.M_LIFT + H.M_PAD, "lift_wn_rad_s": RL.LIFT_WN, "logic": "contact_bed_roll.run_roll"}
    snaps = []
    rig = NewtonRollRig(model, dt)
    met, (cols, tr), W = RL.run_roll(rig, N, dt, V, T, film_every=(RL.FILM_SPEED / RL.FILM_FPS) if film else None,
                                     on_frame=snaps.append if film else None)
    row.update(RL.summarize(met, cols, tr, N, dt, V, T))
    row["us_per_step_median"] = float(np.median(W) * 1e6) if len(W) else None
    stride = max(1, int(round(0.010 / dt)))
    row["trace_cols"] = cols
    row["trace"] = [[round(float(x), 6) for x in r] for r in tr[stride - 1::stride]]
    fpath = None
    if film and snaps:
        allf = [p[2] for sn in snaps for p in sn["pts"]]
        fref = float(np.percentile(allf, 98)) if allf else 1.0
        for sn in snaps:
            sn["pts"] = [(p, nn, v / fref * RL.F_SCALE, r, k) for p, nn, v, r, k in sn["pts"]]
        fr = RL.RollFilm()
        top = f"{model}: {LABEL[model]}"
        sub = f"N {N:g} N per pad, pad speed {v_mm_s:g} mm/s, dt {dt_ms:g} ms, contacts 0-{fref * 1e3:.3g} mN"
        frames = [fr.frame(sn, top, sub, RL.film_bottom(sn)) for sn in snaps]
        fpath = FILMDIR / f"roll_{model}.mp4"
        RL.encode(fpath, frames)
        from PIL import Image
        Image.fromarray(frames[len(frames) // 2]).save(fpath.with_suffix(".jpg"), quality=88)
    row["wall_s"] = time.time() - t0w
    return finish_row(row, "roll", model, dt_ms, row.get("status", "complete"),
                      film=str(Path(fpath).relative_to(OUTDIR)) if fpath else None)


# ------------------------------------------------------------------------------------ compare

def rows_of(path):
    out = []
    if Path(path).exists():
        for line in open(path):
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def compare(dt_ms=1.0):
    """Newton against Drake hydroelastic at the same N and step; flags differences over 20 %."""
    def pct(a, b):
        return None if (a is None or b is None or b == 0) else 100.0 * (a - b) / abs(b)

    lines = []

    def emit(task, metric, N, newton, drake, model):
        d = pct(newton, drake)
        flag = "  >20%" if d is not None and abs(d) > 20 else ""
        lines.append(f"{task:6s} {model:24s} N {str(N):9s} {metric:24s} newton {newton!s:>10.10s}  drake {drake!s:>10.10s}"
                     f"  diff {'' if d is None else f'{d:+.0f} %'}{flag}")

    st = rows_of(OUTDIR / "static_newton.jsonl")
    dk = {(r["N"], r["dt_ms"]): r for r in st if r["model"] == "drake_hydro"}
    for r in st:
        if r["model"] == "drake_hydro" or r["dt_ms"] != dt_ms or r["status"] != "complete":
            continue
        d = dk.get((r["N"], r["dt_ms"]))
        if d:
            emit("static", "pen_mm", r["N"], round(r["pen_mm"], 3), round(d["pen_mm"], 3), r["model"])
            emit("static", "area_mm2", r["N"], round(0.5 * (r["area_eq_L_mm2"] + r["area_eq_R_mm2"]), 2),
                 round(d["area_mm2"], 2), r["model"])
            emit("static", "rbar_mm", r["N"], round(r["rbar_solver_mm"], 3), round(d["rbar_pressure_mm"], 3), r["model"])
    pdk = {(r["N"], r["dt_ms"]): r for r in rows_of(OUTDIR / "pull_slip.jsonl") if r.get("chain_spec") == "drake:hydro:rt0.01"}
    for r in rows_of(OUTDIR / "pull_slip_newton.jsonl"):
        d = pdk.get((r["N"], r["dt_ms"]))
        if r["dt_ms"] != dt_ms or not d or r["status"] != "complete":
            continue
        for k in ("mu_eff", "u_pre_mm", "creep_mm_s", "v_slip_mean50_mm_s", "mu_slide"):
            emit("pull", k, r["N"], round(r[k], 4), round(d[k], 4), r["model"])
    tdk = {(r["N"], r["dt_ms"]): r for r in rows_of(OUTDIR / "twist_slip.jsonl") if r.get("model") == "drake_hydro"}
    for r in rows_of(OUTDIR / "twist_slip_newton.jsonl"):
        d = tdk.get((r["N"], r["dt_ms"]))
        if r["dt_ms"] != dt_ms or r.get("status") != "complete":
            continue
        for k in ("tau_onset_Nm", "rbar_onset_mm", "rot_pre_deg", "creep_deg_s", "rbar_kin_mm", "rbar_ratio_3_05_onset",
                  "rbar_ratio_3_05_kin"):
            if k in r:
                emit("twist", k, r["N"], round(r[k], 5), round(d[k], 5) if d and k in d else None, r["model"])
    sdk = {(r["N"], r["a_pk_g"], r["dt_ms"]): r for r in rows_of(OUTDIR / "shake.jsonl") if r.get("model") == "drake_hydro"}
    for r in rows_of(OUTDIR / "shake_newton.jsonl"):
        d = sdk.get((r["N"], r.get("a_pk_g"), r["dt_ms"]))
        if r["dt_ms"] != dt_ms:
            continue
        for k in ("drift_per_cycle_mm", "peak_rel_disp_mm", "drop"):
            emit("shake", k, (r["N"], r.get("a_pk_g")), r.get(k) if not isinstance(r.get(k), float) else round(r[k], 4),
                 (d.get(k) if not isinstance(d.get(k), float) else round(d[k], 4)) if d else None, r["model"])
    bdk = {r["dt_ms"]: r for r in rows_of(OUTDIR / "brake.jsonl") if r.get("model") == "drake_hydro"}
    for r in rows_of(OUTDIR / "brake_newton.jsonl"):
        d = bdk.get(r["dt_ms"])
        if r["dt_ms"] != dt_ms:
            continue
        for k in ("phi_end_deg", "t_80_s", "N_at_10_N", "N_at_80_N", "peak_rate_deg_s", "pinched_end"):
            v = r.get(k)
            emit("brake", k, "6->0.2", round(v, 3) if isinstance(v, float) else v,
                 (round(d[k], 3) if isinstance(d.get(k), float) else d.get(k)) if d else None, r["model"])
    print("\n".join(lines))


# ------------------------------------------------------------------------------------- CLI

def append(path, row):
    H.append_row(path, row)
    slim = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()
            if not isinstance(v, (list, dict)) or k in ("settle",)}
    print(json.dumps(slim, default=H._json_default)[:1500], flush=True)


def guarded(fn, task, model, N, dt, **kw):
    """A case that raises becomes a failed row, not a lost batch."""
    import traceback
    args = (model, dt) if N is None else (model, N, dt)
    if task in ("shake", "roll"):
        args = (model, N[0], dt, N[1]) if task == "roll" else (model, N[0], N[1], dt)
    try:
        return fn(*args, **kw)
    except Exception as e:
        row = dict(task=task, N=N[0] if isinstance(N, tuple) else N, status="failed", error=repr(e)[:500],
                   traceback=traceback.format_exc()[-2000:])
        if isinstance(N, tuple):
            row["v_mm_s" if task == "roll" else "a_pk_g"] = N[1]
        return finish_row(row, task, model, dt, "failed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", choices=["static", "drake-static", "pull", "twist", "shake", "brake", "roll", "compare"])
    ap.add_argument("--models", nargs="+", default=["newton_hydro"])
    ap.add_argument("--N", nargs="+", type=float)
    ap.add_argument("--dt", nargs="+", type=float, default=[1.0, 5.0])
    ap.add_argument("--a", nargs="+", type=float, help="T4 peak accelerations, g")
    ap.add_argument("--v", nargs="+", type=float, help="T3 pad speeds, mm/s")
    ap.add_argument("--film", action="store_true", help="film the protocol case (N 1, dt 1 ms; T4 N 0.5, 2 g)")
    a = ap.parse_args()
    if a.task == "compare":
        for dt in a.dt:
            compare(dt)
        return
    if a.task == "drake-static":
        out = OUTDIR / "static_newton.jsonl"
        have = done(out, lambda r: (r["model"], r["N"], r["dt_ms"]))
        for dt in a.dt:
            for N in a.N or [0.5, 1.0, 3.0]:
                if ("drake_hydro", N, dt) not in have:
                    append(out, drake_static(N, dt))
        return
    import warp as wp
    wp.config.log_level = wp.LOG_WARNING if hasattr(wp, "LOG_WARNING") else None
    wp.init()
    if a.task == "static":
        out = OUTDIR / "static_newton.jsonl"
        have = done(out, lambda r: (r["model"], r["N"], r["dt_ms"]))
        for model in a.models:
            for dt in a.dt:
                for N in a.N or [0.5, 1.0, 3.0]:
                    if (model, N, dt) not in have:
                        append(out, static_newton(model, N, dt))
    elif a.task == "pull":
        out = OUTDIR / "pull_slip_newton.jsonl"
        have = done(out, lambda r: (r["model"], r["N"], r["dt_ms"]))
        for model in a.models:
            for dt in a.dt:
                for N in a.N or [0.5, 1.0, 3.0]:
                    if (model, N, dt) in have:
                        continue
                    film = FILMDIR / f"pull_{model}.mp4" if (a.film and N == 1.0 and dt == 1.0) else None
                    append(out, guarded(run_pull, "pull", model, N, dt, film=film))
    elif a.task == "twist":
        out = OUTDIR / "twist_slip_newton.jsonl"
        have = done(out, lambda r: (r["model"], r["N"], r["dt_ms"]))
        import contact_bed_twist as TW
        for model in a.models:
            for dt in a.dt:
                for N in a.N or [0.5, 1.0, 3.0]:
                    if (model, N, dt) in have:
                        continue
                    film = FILMDIR / f"twist_{model}.mp4" if (a.film and N == 1.0 and dt == 1.0) else None
                    r = guarded(run_twist, "twist", model, N, dt, film=film)
                    TW.add_ratio(r, out)
                    append(out, r)
    elif a.task == "shake":
        out = OUTDIR / "shake_newton.jsonl"
        have = done(out, lambda r: (r["model"], r["N"], r["a_pk_g"], r["dt_ms"]))
        for model in a.models:
            for dt in a.dt:
                for N in a.N or [0.2, 0.5]:
                    for ag in a.a or [0.5, 1.0, 2.0, 4.0]:
                        if (model, N, ag, dt) in have:
                            continue
                        film = FILMDIR / f"shake_{model}.mp4" if (a.film and N == 0.5 and ag == 2.0 and dt == 1.0) else None
                        r = guarded(run_shake, "shake", model, (N, ag), dt, film=film)
                        r.setdefault("a_pk_g", ag)
                        append(out, r)
    elif a.task == "roll":
        out = OUTDIR / "roll_newton.jsonl"
        have = done(out, lambda r: (r["model"], r["N"], r["dt_ms"], r["v_mm_s"]))
        for model in a.models:
            for dt in a.dt:
                for N in a.N or [0.5, 1.0, 3.0]:
                    for v in a.v or [10.0, 50.0]:
                        if (model, N, dt, v) in have:
                            continue
                        film = (a.film and N == 1.0 and dt == 1.0 and v == 10.0)
                        r = guarded(run_roll, "roll", model, (N, v), dt, film=film)
                        r.setdefault("v_mm_s", v)
                        append(out, r)
    elif a.task == "brake":
        out = OUTDIR / "brake_newton.jsonl"
        have = done(out, lambda r: (r["model"], r["dt_ms"]))
        for model in a.models:
            for dt in a.dt:
                if (model, dt) in have:
                    continue
                film = FILMDIR / f"brake_{model}.mp4" if (a.film and dt == 1.0) else None
                r = guarded(run_brake, "brake", model, None, dt, film=film)
                append(out, r)


if __name__ == "__main__":
    main()
