#!/usr/bin/env python3
"""The two-pad pinch rig of the contact bed with the SR2 TPU fingertip block in place of the 10.55 mm sphere.

Every bed task (T1 pull, T2 twist, T4 shake, T5 brake; docs/experiments/20261005-contact_bed/PROTOCOL.md)
builds its rig through `hom_contact_rig.make_rig(spec)`. This module wraps that function: a spec carrying a
`tpu<r>` field (fillet radius in mm, e.g. `tpu6`, `tpu2.7`, `tpu0` for the sharp box) builds the same rig with
`fingertip_geometry`'s block on each rail, palmar face toward the tool, PIP axis along the tool axis (the
bench grasp's orientation). The bed scripts run unchanged.

    mj:spheres:s1:rs0.75:ir100:tr0.02:tpu6     1 mm sphere pads on the block (stiffness E/h A_s, h 8.5 mm)
    mj:mesh:ir100:tpu6                          the block as one convex mesh, MuJoCo point contact
    mj:meshmc:ir100:tpu6                        the same with multiccd (up to 4 contacts per pair)
    drake:hydro:E1e7:r1:rt0.01:tpu6             Drake compliant convex block against the rigid tool, SAP

    PY=logs/20261001-hom_contact/venv/bin/python
    $PY scripts/tpu_tip_rig.py static --N 0.5 1 3                  # patch area, extent and arm per model
    $PY scripts/tpu_tip_rig.py bed --tasks T1 T2 T5 --dt 1 5       # bed tasks on the TPU models
"""
from __future__ import annotations

import argparse
import contextlib
import math
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fingertip_geometry as G  # noqa: E402
import hom_contact_rig as H  # noqa: E402

OUT_DIR = ROOT / "docs/experiments/20261006-fingertip_backends"
MESH_DIR = ROOT / "assets/mjcf/experimental/20261006-tpu_tip"
SCRIPT = "scripts/tpu_tip_rig.py"
SIGN = {"L": +1.0, "R": -1.0}          # rail L sits at -x and faces +x


def parse(spec: str) -> dict:
    """hom_contact_rig.parse_spec plus the tpu<r> field and the mesh models."""
    parts = spec.split(":")
    r = None
    keep = []
    for p in parts:
        if p.startswith("tpu"):
            r = float(p[3:]) * 1e-3
        else:
            keep.append(p)
    model = keep[1]
    base = ":".join(keep if model not in ("mesh", "meshmc") else [keep[0], "point3"] + keep[2:])
    sp = H.parse_spec(base)
    sp["spec"] = spec
    sp["model"] = model
    sp["tpu_r"] = r
    return sp


def _strip_pad_geoms(root):
    for side in "LR":
        body = root.find(f".//body[@name='pad{side}']")
        for g in list(body.findall("geom")):
            body.remove(g)
    return root


def _contact_attr(sp, tc=None, d0=None):
    if sp["model"] == "spheres":
        return {"condim": "3", "friction": f"{H.MU:g} 0 0", "solref": f"{tc:.6g} 1",
                "solimp": f"{d0:.6g} {d0:.6g} 0.001 0.5 2"}
    return {"condim": "3", "friction": f"{H.MU:g} 0 0", "solref": "0.006 1", "solimp": "0.97 0.995 0.0005"}


def tpu_xml(sp, d_cg, gravity, kinematic, pad_stiff=None):
    """The rig MJCF with TPU blocks on the rails. Returns (xml, info)."""
    xml0, _ = H.mj_xml(dict(sp, model="point3"), d_cg, gravity, kinematic)
    root = _strip_pad_geoms(ET.fromstring(xml0))
    r = sp["tpu_r"]
    paths = G.mesh_paths(MESH_DIR, r)
    asset = root.find("asset")
    for sgn, p in paths.items():
        ET.SubElement(asset, "mesh", name=f"tpu_{'pos' if sgn > 0 else 'neg'}", file=str(p))
    info = {"tpu_r_mm": r * 1e3}
    if sp["model"] in ("mesh", "meshmc"):
        if sp["model"] == "meshmc":
            opt = root.find("option")
            fl = opt.find("flag")
            if fl is None:
                fl = ET.SubElement(opt, "flag")
            fl.set("multiccd", "enable")
        for side in "LR":
            body = root.find(f".//body[@name='pad{side}']")
            ET.SubElement(body, "geom", name=f"pad{side}", type="mesh",
                          mesh=f"tpu_{'pos' if SIGN[side] > 0 else 'neg'}", contype="1", conaffinity="0",
                          rgba="0.85 0.55 0.35 1", **_contact_attr(sp))
        tool = root.find(".//geom[@name='tool']")
        for k, v in _contact_attr(sp).items():
            tool.set(k, v)
        return ET.tostring(root, encoding="unicode"), info
    # sphere pads on the block
    s, rs = sp["s"], sp["rs"]
    tc, d0 = pad_stiff if pad_stiff is not None else (0.01, 0.9)
    c_attr = _contact_attr(sp, tc, d0)
    for side in "LR":
        body = root.find(f".//body[@name='pad{side}']")
        ET.SubElement(body, "geom", name=f"pad{side}_vis", type="mesh",
                      mesh=f"tpu_{'pos' if SIGN[side] > 0 else 'neg'}", contype="0", conaffinity="0",
                      group="1", rgba="0.85 0.55 0.35 0.35")
        C, Nrm, A, K = G.pad_spheres(r, s, rs, SIGN[side], sp["E"])
        for i, p in enumerate(C):
            ET.SubElement(body, "geom", name=f"pad{side}_s{i}", type="sphere", size=f"{rs}",
                          pos=f"{p[0]:.7f} {p[1]:.7f} {p[2]:.7f}", contype="1", conaffinity="0",
                          rgba="0.75 0.4 0.25 1", **c_attr)
        info.update(n_spheres=len(C), area_per_sphere_mm2=float(np.median(A)) * 1e6, K_sphere=float(np.median(K)),
                    padded_area_mm2=float(A.sum()) * 1e6)
    tool = root.find(".//geom[@name='tool']")
    for k, v in c_attr.items():
        tool.set(k, v)
    return ET.tostring(root, encoding="unicode"), info


class TpuMjRig(H.MjRig):
    sim = "mujoco"

    def __init__(self, sp: dict, d_cg: float, gravity: bool, kinematic: bool = False):
        import mujoco
        self.mj, self.sp, self.d_cg = mujoco, sp, d_cg
        xml, self.info = tpu_xml(sp, d_cg, gravity, kinematic)
        if sp["model"] == "spheres":
            m0 = mujoco.MjModel.from_xml_string(xml)
            diag = m0.body_invweight0[m0.body("padL").id, 0] + m0.body_invweight0[m0.body("tool").id, 0]
            K_s = self.info["K_sphere"]
            tc = sp.get("tr", 0.02) / 2.0
            d0 = 1.0 - 1.0 / (tc ** 2 * K_s * diag)
            if d0 < 0.05:
                raise ValueError(f"relaxation {sp.get('tr')} s is below this pad's floor")
            xml, self.info = tpu_xml(sp, d_cg, gravity, kinematic, pad_stiff=(tc, d0))
            self.info.update(solref_timeconst=tc, solimp_d0=d0, relaxation_s=2 * tc, diagApprox=float(diag))
        self.xml = xml
        self.m = mujoco.MjModel.from_xml_string(xml)
        self.d = mujoco.MjData(self.m)
        self.tool = self.m.body("tool").id
        self.pads = {s: self.m.body("pad" + s).id for s in "LR"}
        self.tool_geom = self.m.geom("tool").id
        self.geom_side = {}
        for gi in range(self.m.ngeom):
            b = self.m.geom_bodyid[gi]
            for s, bid in self.pads.items():
                if b == bid and self.m.geom_contype[gi]:
                    self.geom_side[gi] = s
        self.pad_geom = {}
        self.lastN = {"L": 0.0, "R": 0.0}
        self.f6 = np.zeros(6)
        mujoco.mj_forward(self.m, self.d)
        self.theta_prev, self.theta_unwrap = None, 0.0


def _patch_stats(P, F, a_s_mm2=None):
    if len(F) == 0:
        return {"n": 0}
    cop = (F[:, None] * P).sum(0) / F.sum()
    rr = np.linalg.norm((P - cop)[:, 1:], axis=1)
    return {"n": int(len(F)), "N": float(F.sum()), "arm_mm": float((F * rr).sum() / F.sum() * 1e3),
            "extent_y_mm": float(np.ptp(P[:, 1]) * 1e3), "extent_z_mm": float(np.ptp(P[:, 2]) * 1e3),
            "area_mm2": float(len(F) * a_s_mm2) if a_s_mm2 else None, "cop_mm": (cop * 1e3).tolist()}


class TpuDrakeRig(H.DrakeRig):
    sim = "drake"

    def __init__(self, sp: dict, d_cg: float, gravity: bool, kinematic: bool = False):
        from pydrake.all import (
            AddCompliantHydroelasticProperties, AddContactMaterial, AddMultibodyPlant,
            AddRigidHydroelasticProperties, CoulombFriction, Convex, Cylinder, DiagramBuilder,
            FixedOffsetFrame, MultibodyPlantConfig, PrismaticJoint, ProximityProperties,
            RigidTransform, RotationMatrix, Simulator, SpatialInertia,
        )
        self.sp, self.d_cg = sp, d_cg
        hydro = sp["model"] == "hydro"
        paths = G.mesh_paths(MESH_DIR, sp["tpu_r"])
        b = DiagramBuilder()
        cfg = MultibodyPlantConfig(time_step=sp.get("dt", H.DT), discrete_contact_approximation="sap",
                                   contact_model="hydroelastic_with_fallback" if hydro else "point")
        plant, sg = AddMultibodyPlant(cfg, b)
        if not gravity:
            plant.mutable_gravity_field().set_gravity_vector([0.0, 0.0, 0.0])
        self.pad_bodies, self.joints = {}, {}
        for side, sgn in (("L", -1.0), ("R", 1.0)):
            body = plant.AddRigidBody("pad" + side, SpatialInertia.SolidSphereWithMass(H.M_PAD, H.R_PAD))
            pp = ProximityProperties()
            if hydro:
                AddCompliantHydroelasticProperties(sp["res"], sp["E"], pp)
            AddContactMaterial(dissipation=sp.get("hc", 10.0), point_stiffness=sp.get("kp", 1e4),
                               friction=CoulombFriction(H.MU, H.MU), properties=pp)
            if "rt" in sp:
                pp.AddProperty("material", "relaxation_time", sp["rt"])
            plant.RegisterCollisionGeometry(body, RigidTransform(), Convex(str(paths[SIGN[side]])), "pad" + side, pp)
            self.pad_bodies[side] = body
            if not kinematic:
                fr = plant.AddFrame(FixedOffsetFrame("slot" + side, plant.world_frame(),
                                                     RigidTransform([sgn * H.X0, 0.0, 0.0])))
                j = plant.AddJoint(PrismaticJoint("rail" + side, fr, body.body_frame(), [1.0, 0.0, 0.0],
                                                  damping=H.RAIL_DAMP))
                plant.AddJointActuator("f" + side, j)
                self.joints[side] = j
        tool = plant.AddRigidBody("tool", SpatialInertia.SolidCylinderWithMass(
            H.M_TOOL, H.R_TOOL, 2 * H.HL_TOOL, [0.0, 0.0, 1.0]))
        tp = ProximityProperties()
        if hydro:
            AddRigidHydroelasticProperties(sp.get("res_tool", 0.0005), tp)
        AddContactMaterial(dissipation=sp.get("hc", 10.0), point_stiffness=sp.get("kp", 1e4),
                           friction=CoulombFriction(H.MU, H.MU), properties=tp)
        if "rt" in sp:
            tp.AddProperty("material", "relaxation_time", sp["rt"])
        plant.RegisterCollisionGeometry(tool, RigidTransform(), Cylinder(H.R_TOOL, 2 * H.HL_TOOL), "tool", tp)
        plant.Finalize()
        self.plant, self.sg, self.toolb = plant, sg, tool
        self.diagram = b.Build()
        self.simulator = Simulator(self.diagram)
        self.ctx = self.simulator.get_mutable_context()
        self.pc = plant.GetMyMutableContextFromRoot(self.ctx)
        self.sgc = sg.GetMyMutableContextFromRoot(self.ctx)
        X_WT = RigidTransform(RotationMatrix.MakeXRotation(-math.pi / 2), [0.0, d_cg, 0.0])
        plant.SetFreeBodyPose(self.pc, tool, X_WT)
        self.geom_side = {}
        for s, body in self.pad_bodies.items():
            for gid in plant.GetCollisionGeometriesForBody(body):
                self.geom_side[gid] = s
        self.kinematic = kinematic
        if kinematic:
            for s, sgn in (("L", -1.0), ("R", 1.0)):
                plant.SetFreeBodyPose(self.pc, self.pad_bodies[s], RigidTransform([sgn * (H.X0 + 0.01), 0, 0]))
        else:
            self._fix_inputs(0.0, np.zeros(3), np.zeros(3))
            self.simulator.Initialize()
        self.N = 0.0
        self.f_tool, self.tau_tool = np.zeros(3), np.zeros(3)
        self.theta_prev, self.theta_unwrap = None, 0.0
        self.info = {"tpu_r_mm": sp["tpu_r"] * 1e3}


def patch_of(rig):
    """Per side of any bed rig: loaded contacts, their force-weighted centre, the force-weighted arm about
    the pinch normal (world x) through that centre, and the extent along y (tool axis) and z."""
    out = {s: {"n": 0} for s in "LR"}
    if rig.sim == "mujoco":
        f6 = np.zeros(6)
        a_s = (getattr(rig, "info", {}) or {}).get("area_per_sphere_mm2")
        for s in "LR":
            P, F = [], []
            for i in range(rig.d.ncon):
                c = rig.d.contact[i]
                if rig.geom_side.get(int(c.geom[0])) != s and rig.geom_side.get(int(c.geom[1])) != s:
                    continue
                rig.mj.mj_contactForce(rig.m, rig.d, i, f6)
                if f6[0] > 1e-6:
                    P.append(np.array(c.pos)), F.append(float(f6[0]))
            out[s] = _patch_stats(np.array(P), np.array(F), a_s)
        return out
    cr = rig.plant.get_contact_results_output_port().Eval(rig.pc)
    for i in range(cr.num_hydroelastic_contacts()):
        srf = cr.hydroelastic_contact_info(i).contact_surface()
        s = rig.geom_side.get(srf.id_M()) or rig.geom_side.get(srf.id_N())
        if s is None:
            continue
        tri = srf.is_triangle()
        mesh = srf.tri_mesh_W() if tri else srf.poly_mesh_W()
        fld = srf.tri_e_MN() if tri else srf.poly_e_MN()
        nf = mesh.num_elements()
        C = np.array([mesh.element_centroid(k) for k in range(nf)])
        A = np.array([mesh.area(k) for k in range(nf)])
        Pr = np.array([fld.EvaluateCartesian(k, C[k]) for k in range(nf)])
        st = _patch_stats(C, Pr * A)
        st.update(area_mm2=float(A.sum() * 1e6), n=nf)
        out[s] = st
    return out


def make_rig_tpu(orig):
    def make(spec: str, d_cg: float = 0.0, gravity: bool = True, kinematic: bool = False):
        if ":tpu" not in spec:
            return orig(spec, d_cg, gravity, kinematic)
        sp = parse(spec)
        return (TpuDrakeRig if sp["sim"] == "drake" else TpuMjRig)(sp, d_cg, gravity, kinematic)
    return make


@contextlib.contextmanager
def tpu_rigs():
    orig = H.make_rig
    H.make_rig = make_rig_tpu(orig)
    try:
        yield
    finally:
        H.make_rig = orig


# model name -> (rig spec, chain spec): the bed's table, extended
def tpu_models(radii=(6.0, 2.7, 0.0)):
    out = {}
    for r in radii:
        t = f"tpu{r:g}"
        n = f"{r:g}".replace(".", "p")
        out[f"mj_tpu{n}_pads1"] = (f"mj:spheres:s1:rs0.75:ir100:tr0.02:{t}", f"mj:spheres:s1:rs0.75:tr0.02:{t}")
        out[f"mj_tpu{n}_pads1_ir1e4"] = (f"mj:spheres:s1:rs0.75:ir10000:tr0.02:{t}",
                                         f"mj:spheres:s1:rs0.75:tr0.02:ir10000:{t}")
        out[f"mj_tpu{n}_mesh"] = (f"mj:mesh:ir100:{t}", f"mj:mesh:{t}")
        out[f"mj_tpu{n}_meshmc"] = (f"mj:meshmc:ir100:{t}", f"mj:meshmc:{t}")
        out[f"drake_tpu{n}"] = (f"drake:hydro:E1e7:r1:rt0.01:{t}", f"drake:hydro:rt0.01:{t}")
    return out


def static(models, Ns, dt_ms, out):
    import contact_bed_common as CB
    rows = []
    with tpu_rigs():
        for name in models:
            spec = CB.MODELS[name][0]
            for N in Ns:
                t0 = time.time()
                try:
                    H.DT = dt_ms * 1e-3
                    rig = H.make_rig(spec, d_cg=0.0, gravity=False)
                    rig.set_pad_force(N)
                    rig.set_tool_wrench(np.zeros(3), np.zeros(3))
                    W = []
                    for _ in range(int(round(0.6 / H.DT))):
                        w0 = time.perf_counter()
                        rig.step(H.DT)
                        W.append(time.perf_counter() - w0)
                    pt = patch_of(rig)
                    row = {"task": "static", "model": name, "rig_spec": spec, "N": N, "dt_ms": dt_ms,
                           "L": pt["L"], "R": pt["R"], "info": getattr(rig, "info", {}),
                           "us_per_step_median": float(np.median(W) * 1e6), "status": "complete"}
                except Exception as e:                      # a failed case is a row
                    row = {"task": "static", "model": name, "rig_spec": spec, "N": N, "dt_ms": dt_ms,
                           "status": "failed", "error": repr(e), "traceback": traceback.format_exc()[-1500:]}
                row.update(script=SCRIPT, git_rev=CB.git_rev(), wall_s=time.time() - t0,
                           when=time.strftime("%Y-%m-%d %H:%M"))
                H.append_row(out, row)
                L = row.get("L", {})
                print(f"{name:22s} N={N:<4} n={L.get('n')} area={L.get('area_mm2')} arm={L.get('arm_mm')} "
                      f"ext y={L.get('extent_y_mm')} z={L.get('extent_z_mm')} us={row.get('us_per_step_median')} "
                      f"{row['status']}", flush=True)
                rows.append(row)
    return rows


def bed(models, tasks, Ns, dts, out_dir):
    """Run the bed tasks on the TPU models; rows go to <out_dir>/tip_<task>.jsonl."""
    import contact_bed_brake as BR
    import contact_bed_common as CB
    import contact_bed_pull as P
    import contact_bed_shake as SH
    import contact_bed_twist as TW
    for k, v in tpu_models().items():
        CB.MODELS.setdefault(k, v)
        P.MODELS.setdefault(v[1], v[0])
    with tpu_rigs():
        for task in tasks:
            out = Path(out_dir) / f"tip_{task}.jsonl"
            have = CB.done(out, key=("model", "N", "dt_ms"))
            for name in models:
                for dt in dts:
                    for N in (Ns if task in ("T1", "T2") else ([0.5] if task == "T4" else [None])):
                        if (name, N, dt) in have:
                            continue
                        t0 = time.time()
                        try:
                            if task == "T1":
                                r = P.run_case(CB.MODELS[name][1], N, dt)
                                r["model"] = name
                            elif task == "T2":
                                r = TW.run_case(name, N, dt, film=False)
                            elif task == "T4":
                                r = SH.run_case(name, N, dt, 2.0, film=None)
                            else:
                                r = BR.run_case(name, dt, film=False)
                                r["N"] = None
                        except Exception as e:
                            r = {"model": name, "N": N, "dt_ms": dt, "status": "failed", "error": repr(e),
                                 "traceback": traceback.format_exc()[-1500:]}
                        r.update(task=task, model=name, dt_ms=dt, script=SCRIPT, git_rev=CB.git_rev(),
                                 wall_s_tip=time.time() - t0, when=time.strftime("%Y-%m-%d %H:%M"))
                        r.setdefault("N", N)
                        H.append_row(out, r)
                        keys = {"T1": ("mu_eff", "creep_mm_s", "v_slip_mean50_mm_s"),
                                "T2": ("tau_onset_Nm", "rbar_onset_mm", "creep_deg_s", "rbar_kin_mm"),
                                "T4": ("drift_per_cycle_mm",),
                                "T5": ("phi_end_deg", "N_at_45_N", "N_at_80_N", "slip_end_mm")}[task]
                        print(task, name, N, dt, r.get("status"),
                              {k: round(r[k], 5) for k in keys + ("us_per_step_median",)
                               if isinstance(r.get(k), (int, float))}, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s1 = sub.add_parser("static")
    s1.add_argument("--models", nargs="+")
    s1.add_argument("--N", nargs="+", type=float, default=[0.5, 1.0, 3.0])
    s1.add_argument("--dt", type=float, default=1.0)
    s1.add_argument("--out", type=Path, default=OUT_DIR / "tip_static.jsonl")
    s2 = sub.add_parser("bed")
    s2.add_argument("--models", nargs="+")
    s2.add_argument("--tasks", nargs="+", default=["T1", "T2", "T5"])
    s2.add_argument("--N", nargs="+", type=float, default=[0.5, 1.0, 3.0])
    s2.add_argument("--dt", nargs="+", type=float, default=[1.0, 5.0])
    s2.add_argument("--out-dir", type=Path, default=OUT_DIR)
    a = ap.parse_args()
    import contact_bed_common as CB
    for k, v in tpu_models().items():
        CB.MODELS.setdefault(k, v)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if a.cmd == "static":
        models = a.models or ["mj_pads1", "drake_hydro"] + [k for k in tpu_models() if not k.endswith("_ir1e4")]
        static(models, a.N, a.dt, a.out)
    else:
        models = a.models or list(tpu_models())
        bed(models, a.tasks, a.N, a.dt, a.out_dir)


if __name__ == "__main__":
    main()
