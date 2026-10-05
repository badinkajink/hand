#!/usr/bin/env python3
"""Shared pieces of the contact-model comparison bed (docs/experiments/20261005-contact_bed/PROTOCOL.md).

Model table (protocol names -> rig spec and hom_chain spec), row bookkeeping (skip-existing reader,
fsynced append, git revision), a per-step wall-time probe for rigs built inside `hom_contact_rig`
experiments, and the bed's film renderer: `BedFilm` subclasses `hom_contact_rig.FilmRig`, draws the
pads translucent and the contacts coloured by force or pressure, adds a close-up inset of the -x pad's
contact, and labels model, simulated time and playback speed. Drake states are copied into the MuJoCo
film model (FilmRig does this); Drake hydroelastic faces are drawn from the solver's contact surface.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import hom_contact_rig as H  # noqa: E402

BED = ROOT / "docs/experiments/20261005-contact_bed"
MEDIA = BED / "media"

# protocol model name -> (rig spec for hom_contact_rig.make_rig, hom_chain.make_plant spec)
MODELS = {
    "mj_point3": ("mj:point3:ir100", "mj:point3"),
    "mj_point4s": ("mj:point4s:fit0.000996044x0.2498:ir100", "mj:point4s"),
    "mj_pads1": ("mj:spheres:s1:rs0.75:ir100:tr0.02", "mj:spheres:s1:rs0.75:tr0.02"),
    "mj_pads2": ("mj:spheres:s2:rs0.75:ir100:tr0.02", "mj:spheres:s2:rs0.75:tr0.02"),
    "mj_pads05": ("mj:spheres:s0.5:rs0.75:ir100:tr0.03", "mj:spheres:s0.5:rs0.75:tr0.03"),
    "drake_hydro": ("drake:hydro:E1e7:r1:rt0.01", "drake:hydro:rt0.01"),
}
MAIN = ["mj_point3", "mj_point4s", "mj_pads1", "drake_hydro"]
MAIN_DT = [1.0, 5.0]
CONSISTENCY = [("mj_pads1", 0.5), ("mj_pads1", 2.0), ("mj_pads2", 1.0), ("mj_pads05", 1.0)]
FILM_ORDER = ["mj_point3", "mj_point4s", "drake_hydro", "mj_pads2", "mj_pads1", "mj_pads05"]
SHORT = {"mj_point3": "MuJoCo point contact, condim 3", "mj_point4s": "MuJoCo condim 4, mu_t rescheduled",
         "mj_pads1": "MuJoCo 1 mm sphere pads", "mj_pads2": "MuJoCo 2 mm sphere pads",
         "mj_pads05": "MuJoCo 0.5 mm sphere pads", "drake_hydro": "Drake hydroelastic, SAP"}

SCALE_POINT_N, SCALE_PRESSURE = 4.0, 3e5      # colour full scale: point-contact force; pressure (Pa)


def grid(models=None, dts=None):
    """(model, dt_ms) cases: the main models at 1 and 5 ms plus the consistency set."""
    out = [(m, dt) for m in MAIN for dt in MAIN_DT] + list(CONSISTENCY)
    if models:
        out = [c for c in out if c[0] in models]
    if dts:
        out = [c for c in out if c[1] in dts]
    return out


def git_rev():
    try:
        rev = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, timeout=10).stdout.strip()
        return rev or None
    except Exception:
        return None


def done(path, key=("model", "N", "dt_ms")):
    have = set()
    if Path(path).exists():
        for line in open(path):
            try:
                r = json.loads(line)
                have.add(tuple(r[k] for k in key))
            except Exception:
                pass
    return have


def read_rows(path):
    rows = []
    if Path(path).exists():
        for line in open(path):
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def new_rig(spec, dt, d_cg=0.0, gravity=False):
    H.DT = dt                    # the rig reads its module DT for the MJCF timestep and its step count
    return H.make_rig(spec, d_cg=d_cg, gravity=gravity)


class StepProbe:
    """Patch `hom_contact_rig.make_rig` so every rig an experiment builds times its single-step calls and
    optionally calls `on_step(rig, k)` after each of them (k counts single steps). Multi-step calls
    (the experiments' settle phases) are not timed and do not call back."""

    def __init__(self, on_step=None):
        self.on_step, self.W, self.k, self.rig = on_step, [], 0, None
        self.settle_state, self.settle_t = None, None

    def __enter__(self):
        self._orig = H.make_rig

        def make(*a, **kw):
            rig = self._orig(*a, **kw)
            self.rig = rig
            orig_step = rig.step

            def step(T):
                n = max(1, int(round(T / H.DT)))
                w0 = time.perf_counter()
                orig_step(T)
                w = time.perf_counter() - w0
                if n == 1:
                    self.W.append(w)
                    self.k += 1
                    if self.on_step is not None:
                        self.on_step(rig, self.k)
                else:
                    self.settle_state, self.settle_t = rig.tool_state(), rig.t
            rig.step = step
            return rig
        H.make_rig = make
        return self

    def __exit__(self, *exc):
        H.make_rig = self._orig
        return False


# ------------------------------------------------------------------------------------------ films

def heat(x):
    """Dark violet -> red -> orange -> pale yellow for x in [0, 1] (the colour map of hom_chain's films)."""
    x = min(max(float(x), 0.0), 1.0)
    stops = [(0.0, (0.22, 0.06, 0.38)), (0.35, (0.70, 0.13, 0.32)), (0.7, (0.96, 0.47, 0.10)), (1.0, (1.0, 0.94, 0.45))]
    for (x0, c0), (x1, c1) in zip(stops, stops[1:]):
        if x <= x1:
            f = (x - x0) / (x1 - x0)
            return [c0[i] + f * (c1[i] - c0[i]) for i in range(3)] + [1.0]
    return list(stops[-1][1]) + [1.0]


def _add_geom(scn, kind, size, pos, mat, rgba):
    import mujoco
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, kind, np.asarray(size, float), np.asarray(pos, float), np.asarray(mat, float).reshape(-1),
                        np.asarray(rgba, np.float32))
    scn.ngeom += 1


class BedFilm(H.FilmRig):
    """480x360 view along the pinch axis from the -x side with a close-up inset of the -x pad's contact.

    Pads translucent; contacts coloured on one scale per kind: MuJoCo sphere pads by each sphere's force over
    its share of pad area and Drake hydroelastic faces by pressure (0-0.3 MPa), MuJoCo point contacts by
    force (0-4 N) with the normal force drawn as an arrow. `view` = (lookat, distance, azimuth, elevation)."""

    def __init__(self, spec, d_cg, w=480, h=360, model="", view=None, inset=150, inset_side="right", speed=None):
        import mujoco
        super().__init__(spec, d_cg, w, h)
        self.spec, self.sp_live = spec, H.parse_spec(spec)
        self.model_name = model or spec
        m = self.m
        lookat, dist, az, el = view or ((0.0, 0.0, 0.0), 0.15, 18.0, -14.0)
        self.cam.lookat[:] = lookat
        self.cam.distance, self.cam.azimuth, self.cam.elevation = dist, az, el
        self.inset, self.inset_side, self.speed = inset, inset_side, speed
        self.ri = mujoco.Renderer(m, inset, inset) if inset else None
        self.cami = mujoco.MjvCamera()
        self.cami.lookat[:] = [-H.R_TOOL, 0.0, 0.0]
        self.cami.distance, self.cami.azimuth, self.cami.elevation = 0.027, 0.0, -15.0
        self.rgba0 = m.geom_rgba.copy()
        self.tool_g = [g for g in range(m.ngeom) if m.geom_bodyid[g] == self.tool]
        self.pad_env = [g for g in range(m.ngeom) if (m.geom(g).name or "") in ("padL", "padR", "padL_vis", "padR_vis")]
        self.pad_sph = {m.geom(g).name: g for g in range(m.ngeom) if "_s" in (m.geom(g).name or "")}
        self.spheres = self.sp_live["sim"] == "mj" and self.sp_live["model"] == "spheres"
        self.point = self.sp_live["sim"] == "mj" and not self.spheres
        self.right_g = [g for g in range(m.ngeom) if (m.geom(g).name or "").startswith("padR")]
        self.A_s = None

    def close(self):
        for r in (getattr(self, "r", None), getattr(self, "ri", None)):
            if r is not None:
                r.close()
        self.r = self.ri = None

    def _sync(self, rig):
        import mujoco
        m, d = self.m, self.d
        d.qpos[:] = 0
        if rig.sim == "mujoco":
            for name in ("railL", "railR"):
                d.qpos[m.joint(name).qposadr[0]] = rig.d.qpos[rig.m.joint(name).qposadr[0]]
            qa = m.jnt_qposadr[m.body_jntadr[self.tool]]
            qb = rig.m.jnt_qposadr[rig.m.body_jntadr[rig.tool]]
            d.qpos[qa:qa + 7] = rig.d.qpos[qb:qb + 7]
        else:
            X = rig.plant.EvalBodyPoseInWorld(rig.pc, rig.toolb)
            qa = m.jnt_qposadr[m.body_jntadr[self.tool]]
            d.qpos[qa:qa + 3] = X.translation()
            d.qpos[qa + 3:qa + 7] = X.rotation().ToQuaternion().wxyz()
            px = rig.pad_x()
            for s in "LR":
                d.qpos[m.joint("rail" + s).qposadr[0]] = px[s] - (-H.X0 if s == "L" else H.X0)
        mujoco.mj_forward(m, d)

    def _mj_contacts(self, rig):
        """[(side, pos, normal into the pad, force N, live geom name)] of the live MuJoCo rig."""
        out, f6 = [], np.zeros(6)
        for i in range(rig.d.ncon):
            c = rig.d.contact[i]
            g0, g1 = int(c.geom[0]), int(c.geom[1])
            s = rig.geom_side.get(g0) or rig.geom_side.get(g1)
            if s is None:
                continue
            rig.mj.mj_contactForce(rig.m, rig.d, i, f6)
            g = g0 if g0 in rig.geom_side else g1
            n = np.array(c.frame[:3])
            if g == g1:
                n = -n
            out.append((s, np.array(c.pos), n, float(f6[0]), rig.m.geom(g).name))
        return out

    def _drake_faces(self, rig):
        """[(side, centroid, normal, area, pressure)] of the Drake hydroelastic contact surfaces."""
        out = []
        cr = rig.plant.get_contact_results_output_port().Eval(rig.pc)
        for i in range(cr.num_hydroelastic_contacts()):
            srf = cr.hydroelastic_contact_info(i).contact_surface()
            s = rig.geom_side.get(srf.id_M()) or rig.geom_side.get(srf.id_N())
            if s is None:
                continue
            tri = srf.is_triangle()
            mesh = srf.tri_mesh_W() if tri else srf.poly_mesh_W()
            field = srf.tri_e_MN() if tri else srf.poly_e_MN()
            for f in range(mesh.num_elements()):
                c = np.array(mesh.element_centroid(f))
                out.append((s, c, np.array(mesh.face_normal(f)), float(mesh.area(f)), float(field.EvaluateCartesian(f, c))))
        return out

    def _decorate(self, scn, rig, cons, faces, closeup):
        import mujoco
        if closeup:                                  # the inset shows the -x pad only
            cons = [c for c in cons if c[0] == "L"]
            faces = [f for f in faces if f[0] == "L"]
        if self.point:
            for s, pos, n, fN, _ in cons:
                _add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.0008] * 3 if closeup else [0.0011] * 3, pos,
                          np.eye(3), heat(fN / SCALE_POINT_N))
                if fN > 1e-4 and scn.ngeom < scn.maxgeom:
                    g = scn.geoms[scn.ngeom]
                    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.eye(3).reshape(-1),
                                        np.asarray((0.1, 0.1, 0.1, 1), np.float32))
                    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.0004 if closeup else 0.0007, pos,
                                         pos - n * 0.003 * fN)
                    scn.ngeom += 1
        elif faces:
            for s, c, nv, a, p in faces:
                r = math.sqrt(max(a, 1e-14) / math.pi) * 1.15
                _add_geom(scn, mujoco.mjtGeom.mjGEOM_CYLINDER, [r, r, 0.00005], c + 0.00005 * nv,
                          H._rot_z_to(nv), heat(p / SCALE_PRESSURE))

    def frame(self, rig, text, bottom=None, speed=None):
        speed = self.speed if speed is None else speed
        self._sync(rig)
        m = self.m
        cons = self._mj_contacts(rig) if rig.sim == "mujoco" else []
        faces = self._drake_faces(rig) if (rig.sim == "drake" and self.sp_live["model"] == "hydro") else []
        if self.spheres and self.A_s is None:
            self.A_s = rig.info.get("area_per_sphere_mm2", 1.0) * 1e-6
        m.geom_rgba[:] = self.rgba0
        for g in self.pad_env:
            m.geom_rgba[g, 3] = 0.18
        force = {}
        if self.spheres:
            for s, pos, n, fN, name in cons:
                force[name] = force.get(name, 0.0) + fN
            for name, g in self.pad_sph.items():
                m.geom_rgba[g] = heat(force[name] / self.A_s / SCALE_PRESSURE) if force.get(name, 0.0) > 1e-5 \
                    else [0.80, 0.80, 0.82, 0.55]
        self.r.update_scene(self.d, self.cam)
        self._decorate(self.r.scene, rig, cons, faces, False)
        img = self.r.render().copy()
        if self.ri is not None:
            for g in self.tool_g:
                m.geom_rgba[g, 3] = 0.30
            for g in self.pad_env:
                m.geom_rgba[g, 3] = 0.10
            for g in self.right_g:
                m.geom_rgba[g, 3] = 0.0
            self.ri.update_scene(self.d, self.cami)
            self._decorate(self.ri.scene, rig, cons, faces, True)
            ins = self.ri.render().copy()
            if self.point:
                cap = f"-x pad, 0-{SCALE_POINT_N:g} N"
            elif self.spheres or faces:
                cap = f"-x pad, 0-{SCALE_PRESSURE / 1e6:g} MPa"
            else:
                cap = "-x pad"
            ins = np.array(H.annotate(ins, cap, size=11))
            ins[[0, -1], :, :] = 90
            ins[:, [0, -1], :] = 90
            # inset corner: "right"/"left" at the top under the title, "bottom-right" above the bottom label
            y0 = 30 if self.inset_side in ("right", "left") else img.shape[0] - 24 - self.inset - 6
            x0 = 6 if self.inset_side == "left" else img.shape[1] - self.inset - 6
            img[y0:y0 + self.inset, x0:x0 + self.inset] = ins
        m.geom_rgba[:] = self.rgba0
        top = f"{self.model_name}: {SHORT.get(self.model_name, H.label_of(self.spec))}"
        if speed is not None:
            top += f"   {speed:g}x"
        img = H.annotate(img, top, size=14)
        if bottom:
            img = H.annotate_bottom(img, bottom, size=13)
        return img


def write_h264(frames, path, fps=25, crf=27):
    import imageio.v2 as imageio
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    w = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=None, pixelformat="yuv420p",
                           macro_block_size=8, ffmpeg_params=["-crf", str(crf), "-preset", "medium"],
                           ffmpeg_log_level="error")
    for f in frames:
        w.append_data(np.asarray(f, np.uint8))
    w.close()


def read_frames(path):
    import imageio.v2 as imageio
    r = imageio.get_reader(str(path))
    out = [np.asarray(f) for f in r]
    r.close()
    return out


def tile(clip_paths, out_mp4, out_jpg, cols=3, poster_at=0.5, fps=25):
    """Tile per-model clips (equal frame size) into a grid; shorter clips hold their last frame."""
    from PIL import Image
    clips = [read_frames(p) for p in clip_paths]
    n = max(len(c) for c in clips)
    h, w = clips[0][0].shape[:2]
    rows = int(math.ceil(len(clips) / cols))
    frames = []
    for k in range(n):
        canvas = np.full((rows * h, cols * w, 3), 255, np.uint8)
        for i, c in enumerate(clips):
            f = c[min(k, len(c) - 1)][:, :, :3]
            r_, c_ = divmod(i, cols)
            canvas[r_ * h:(r_ + 1) * h, c_ * w:(c_ + 1) * w] = f
        frames.append(canvas)
    write_h264(frames, out_mp4, fps=fps)
    Image.fromarray(frames[int(round(poster_at * (n - 1)))]).save(out_jpg, quality=88)
    return n
