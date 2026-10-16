#!/usr/bin/env python3
"""Ledger step 17 of the curved-tip study (docs/handoff/20261015-curved_tips.md): the CAD hand's 10 mm cylinder cell
(cylinder:10:0.75 of hand_object_scale_sim's diagonal family) with the as-built block, turned by the HOM controller
under three plants with the same candidates, seeds and controller code (hand_object_scale_sim.turn_rollout):

  v1      MuJoCo, builder-v1 pads (one stiffness per finger, foundation depth 8.5 mm), as on 2026-10-14
  v2      MuJoCo, builder-v2 pads (per-pad stiffness E a_i / h_i, h_i the depth below the tip's volume centroid),
          the repository default since 2026-10-15
  drake   Drake 1.57 (CellDrake below): each fingertip a compliant hydroelastic convex hull of the tip mesh (E 10 MPa,
          relaxation 0.01 s, Hunt-Crossley dissipation 10 s/m), the cylinder, floor, palm and post rigid hydroelastic,
          the CAD links' collision spheres point contact (1e4 N/m); PD servos with the servo plant's gains and effort
          limit; SAP at 1 ms. reorient_backends.DrakeBench's recipe for the real_v1 bench, rewritten for the CAD hand
          (its links are sphere sets, its palm pose is set per candidate, and the post's collisions are filtered out
          when turn_rollout switches the post off after the squeeze)

The 2026-10-14 stage score selects per cell the candidate with the best held turn on seed 0 and scores it by its mean
held turn on seeds 1-2. Rows: docs/experiments/20261015-curved_tips/cell10/turn.jsonl, one per (arm, candidate, seed),
with turn_rollout's result and its 50 ms trace (turn angle and finger normal forces).

    PY=logs/20261001-hom_contact/venv/bin/python
    ~/.claude/bin/resguard.sh run --mem 2G --cpu 100 -- taskset -c 12 $PY scripts/curved_tips_cell_drake.py \\
        --arms drake --cands 0 1 2 --seeds 0 1 2 3 4 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

os.environ["SCALE_HAND"] = "cad_v1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import curved_tips_common as C  # noqa: E402
import curved_tips_rig  # noqa: E402,F401  (this study's scene folders on tip_codesign_rig)
import hand_object_scale_sim as S  # noqa: E402
import hom_turn3 as HT  # noqa: E402
import reorient_backends as RB  # noqa: E402
import tip_codesign_rig as R  # noqa: E402

OUT = C.D / "cell10" / "turn.jsonl"
SCRIPT = "scripts/curved_tips_cell_drake.py"
BUILDERS = {"v1": ("fixed", "median"), "v2": ("local", "each")}


class _PostSwitch:
    """The part of an MjModel that turn_rollout touches to switch the support post off: geom('post').id and the
    post's contype and conaffinity. CellDrake reads it before each advance."""

    class _G:
        id = 0

    def __init__(self):
        self.geom_contype = np.array([1])
        self.geom_conaffinity = np.array([1])

    def geom(self, name):
        return self._G()


class CellDrake:
    """A scale cell's point-contact scene in Drake, with HT.DrakePlant's interface (reset, set_targets, advance,
    state) and the post switch of turn_rollout."""

    def __init__(self, scene: Path, meta: dict, palm, rt: float = 0.01, E: float = RB.E_TPU, tip_files=None):
        from pydrake.all import (AddCompliantHydroelasticProperties, AddContactMaterial, AddMultibodyPlant,
                                 AddRigidHydroelasticProperties, CollisionFilterDeclaration, Convex, CoulombFriction,
                                 DiagramBuilder, GeometrySet, MultibodyPlantConfig, Parser, PdControllerGains,
                                 ProximityProperties, RigidTransform, RoleAssign, Simulator)
        self.meta = meta
        self.m = _PostSwitch()
        self.CollisionFilterDeclaration, self.GeometrySet = CollisionFilterDeclaration, GeometrySet
        root = ET.parse(scene).getroot()
        root.find(".//body[@name='palm_pose']").set("pos", " ".join(f"{float(v):.7f}" for v in palm))
        mesh_file = {m.get("name"): m.get("file") for m in root.find("asset").findall("mesh")}
        tipmesh = {}
        for f in RB.FINGERS:
            tipb = root.find(f".//body[@name='{f}_tip']")
            for g in list(tipb.findall("geom")):
                if g.get("type") == "mesh" and g.get("contype") != "0":
                    tipmesh[f] = (tip_files or mesh_file)[g.get("mesh")]
                tipb.remove(g)
        for parent in root.iter():
            for child in list(parent):
                if child.tag == "geom" and (child.get("class") == "cad_visual" or child.get("type") == "mesh"):
                    parent.remove(child)
                elif child.tag in {"site", "light", "texture", "position", "motor", "camera"}:
                    parent.remove(child)
        for tag in ("visual", "keyframe", "actuator", "option", "statistic", "size", "contact"):
            for el in root.findall(tag):
                root.remove(el)
        asset = root.find("asset")
        for el in list(asset):
            if el.tag in ("mesh", "texture"):
                asset.remove(el)
        for el in root.iter():
            if el.tag == "material":
                for attr in ("texture", "texuniform", "texrepeat", "reflectance"):
                    el.attrib.pop(attr, None)
            if el.tag in ("geom", "default", "joint", "body"):
                for attr in ("solref", "solimp", "condim", "frictionloss", "contype", "conaffinity", "group",
                             "priority", "gravcomp", "solmix", "margin", "gap"):
                    el.attrib.pop(attr, None)
        b = DiagramBuilder()
        cfg = MultibodyPlantConfig(time_step=RB.DT, discrete_contact_approximation="sap",
                                   contact_model="hydroelastic_with_fallback")
        plant, sg = AddMultibodyPlant(cfg, b)
        self.instance, = Parser(plant, sg).AddModelsFromString(ET.tostring(root, encoding="unicode"), "xml")
        P = RB.plant_args(meta["plant"])
        for n in HT.NAMES:
            a = plant.AddJointActuator("a_" + n, plant.GetJointByName(n), P["forcerange"])
            a.set_controller_gains(PdControllerGains(p=P["kp"], d=P["kv"]))
            a.set_default_rotor_inertia(0.001)
            a.set_default_gear_ratio(1.0)
        sid = plant.get_source_id()

        def props(kind):
            pp = ProximityProperties()
            if kind == "compliant":
                AddCompliantHydroelasticProperties(0.001, E, pp)
            elif kind == "rigid":
                AddRigidHydroelasticProperties(0.0005, pp)
            AddContactMaterial(dissipation=10.0, point_stiffness=1e4, friction=CoulombFriction(meta["mu"], meta["mu"]),
                               properties=pp)
            pp.AddProperty("material", "relaxation_time", rt)
            return pp
        self.finger_gid = {}
        for f in RB.FINGERS:
            gid = plant.RegisterCollisionGeometry(plant.GetBodyByName(f"{f}_tip"), RigidTransform(), Convex(tipmesh[f]),
                                                  f"{f}_tipgeom", props("compliant"))
            self.finger_gid[gid] = f
        plant.Finalize()
        bodies = [plant.get_body(i) for i in plant.GetBodyIndices(self.instance)]
        self.finger_bodies = {f: [bb for bb in bodies if bb.name().startswith(f + "_")] for f in RB.FINGERS}
        for f, bs in self.finger_bodies.items():
            for bb in bs:
                for gid in plant.GetCollisionGeometriesForBody(bb):
                    if gid not in self.finger_gid:
                        sg.AssignRole(sid, gid, props("point"), RoleAssign.kReplace)
                        self.finger_gid[gid] = f
        self.tool = plant.GetBodyByName(RB.OBJ)
        self.tool_gids = set(plant.GetCollisionGeometriesForBody(self.tool))
        self.post_gids = list(plant.GetCollisionGeometriesForBody(plant.GetBodyByName("post")))
        rigid = list(self.tool_gids) + self.post_gids + list(plant.GetCollisionGeometriesForBody(plant.world_body()))
        rigid += list(plant.GetCollisionGeometriesForBody(plant.GetBodyByName("palm_pose")))
        for gid in rigid:
            sg.AssignRole(sid, gid, props("rigid"), RoleAssign.kReplace)
        cfm = sg.collision_filter_manager()
        for e in ET.parse(scene).getroot().iter("exclude"):
            g1 = plant.GetCollisionGeometriesForBody(plant.GetBodyByName(e.get("body1")))
            g2 = plant.GetCollisionGeometriesForBody(plant.GetBodyByName(e.get("body2")))
            if g1 and g2:
                cfm.Apply(CollisionFilterDeclaration().ExcludeBetween(GeometrySet(g1), GeometrySet(g2)))
        for f, bs in self.finger_bodies.items():
            gs = [gid for bb in bs for gid in plant.GetCollisionGeometriesForBody(bb)]
            cfm.Apply(CollisionFilterDeclaration().ExcludeWithin(GeometrySet(gs)))
        self.plant, self.sg = plant, sg
        self.diagram = b.Build()
        self.sim = Simulator(self.diagram)
        self.ctx = self.sim.get_mutable_context()
        self.pc = plant.GetMyMutableContextFromRoot(self.ctx)
        self.sgc = sg.GetMyMutableContextFromRoot(self.ctx)
        self.jidx = [plant.GetJointByName(n) for n in HT.NAMES]
        self.act_order = [plant.get_joint_actuator(i).joint().name() for i in plant.GetJointActuatorIndices(self.instance)]
        self.post_filter = None
        self.n_geoms = {"finger": len(self.finger_gid), "tool": len(self.tool_gids), "post": len(self.post_gids),
                        "rigid": len(rigid)}

    def reset(self, seed):
        from pydrake.all import Quaternion, RigidTransform
        p = self.plant
        if self.post_filter is not None:
            self.sg.collision_filter_manager(self.sgc).RemoveDeclaration(self.post_filter)
            self.post_filter = None
        p.SetVelocities(self.pc, np.zeros(p.num_velocities()))
        for j, n in zip(self.jidx, HT.NAMES):
            j.set_angle(self.pc, float(self.meta["q0"][n]))
        tool7 = np.asarray(self.meta["tool7"], float)
        dx, dy, dyaw = RB.jitter(seed)
        q = RB._yaw_quat(tool7[3:], dyaw)
        p.SetFreeBodyPose(self.pc, self.tool, RigidTransform(Quaternion(q / np.linalg.norm(q)), tool7[:3] + [dx, dy, 0.0]))
        self.ctx.SetTime(0.0)
        self.sim.Initialize()

    def set_targets(self, q_cmd):
        tgt = dict(zip(HT.NAMES, (float(v) for v in q_cmd)))
        pos = np.array([tgt[n] for n in self.act_order])
        self.plant.get_desired_state_input_port(self.instance).FixValue(self.pc, np.r_[pos, np.zeros(len(pos))])
        self.plant.get_actuation_input_port().FixValue(self.pc, np.zeros(len(pos)))

    def advance(self, t1):
        if self.post_filter is None and not self.m.geom_contype[0]:
            decl = self.CollisionFilterDeclaration().ExcludeBetween(self.GeometrySet(self.post_gids),
                                                                     self.GeometrySet(list(self.tool_gids)))
            self.post_filter = self.sg.collision_filter_manager(self.sgc).ApplyTransient(decl)
        self.sim.AdvanceTo(t1)

    def state(self):
        p, pc = self.plant, self.pc
        q = np.array([j.get_angle(pc) for j in self.jidx])
        qd = np.array([j.get_angular_rate(pc) for j in self.jidx])
        X = p.EvalBodyPoseInWorld(pc, self.tool)
        F = {f: 0.0 for f in RB.FINGERS}
        n = {f: 0 for f in RB.FINGERS}
        cr = p.get_contact_results_output_port().Eval(pc)
        for i in range(cr.num_hydroelastic_contacts()):
            info = cr.hydroelastic_contact_info(i)
            srf = info.contact_surface()
            ids = (srf.id_M(), srf.id_N())
            if not (set(ids) & self.tool_gids):
                continue
            f = self.finger_gid.get(ids[0]) or self.finger_gid.get(ids[1])
            if f is None:
                continue
            Fv = np.array(info.F_Ac_W().translational())
            nrm = np.asarray(srf.face_normal(0)) if srf.num_faces() else np.zeros(3)
            fn = float(abs(Fv @ nrm)) if np.linalg.norm(nrm) > 0 else float(np.linalg.norm(Fv))
            if fn > 1e-6:
                F[f] += fn
                n[f] += 1
        for i in range(cr.num_point_pair_contacts()):
            info = cr.point_pair_contact_info(i)
            pp = info.point_pair()
            if not ({pp.id_A, pp.id_B} & self.tool_gids):
                continue
            f = self.finger_gid.get(pp.id_A) or self.finger_gid.get(pp.id_B)
            if f is None:
                continue
            fn = float(abs(np.array(info.contact_force()) @ np.array(pp.nhat_BA_W)))
            if fn > 1e-6:
                F[f] += fn
                n[f] += 1
        return {"t": float(self.ctx.get_time()), "q": q, "qd": qd, "p": np.array(X.translation()),
                "quat": np.array(X.rotation().ToQuaternion().wxyz()), "R": X.rotation().matrix(), "F": F, "n": n}


def scenes_for(arm, tip, mu):
    """Install the arm's pad builder on tip_codesign_rig and point the scale study's scene cache at this study's
    folder for that builder (the screen's own folders: <tip>_<tag>_mu<mu>)."""
    h, K = BUILDERS["v1" if arm == "v1" else "v2"]
    ns = argparse.Namespace(E=RB.E_TPU, pf=None, tr=RB.PAD_TR, tag=None, h=h, K=K)
    R.install(ns, force_tip=tip)
    S.SCENES = R.SCN.parent / "scale" / f"{tip}_{R.tag_of(ns)}_mu{mu:g}"
    S.MU, S.PLANT = mu, C.SERVO
    return R.tag_of(ns)


def done(path):
    have = set()
    if path.exists():
        for line in open(path):
            try:
                r = json.loads(line)
                if r.get("status") == "ok":
                    have.add((r["arm"], r["tip"], r["cell"], int(r["cand"]), int(r["seed"])))
            except Exception:
                pass
    return have


def append(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(row, default=float) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def main():
    import mujoco
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", default=["v1", "v2", "drake"], choices=["v1", "v2", "drake"])
    ap.add_argument("--tip", default=C.ASBUILT)
    ap.add_argument("--cell", default="cylinder:10:0.75")
    ap.add_argument("--mu", type=float, default=0.7)
    ap.add_argument("--cands", nargs="*", type=int, default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--set", nargs="*", default=["kp_model=0.72", "post_off=1"])
    ap.add_argument("--mesh-eps", type=float, default=1e-6,
                    help="Drake's tip hulls with each terrace outline simplified to this deviation (m); Drake rejects "
                         "the near-zero facets of collinear outline points (2026-10-14 ledger step 30)")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    shape, dd, t = a.cell.split(":")
    dd, t = float(dd), float(t)
    prm = dict(HT.DEFAULTS)
    for kv in a.set:
        k, v = kv.split("=")
        prm[k] = float(v)
    kp = RB.plant_args(C.SERVO)["kp"]
    have = done(a.out)
    for arm in a.arms:
        tag = scenes_for(arm, a.tip, a.mu)
        cell = next(c for c, sh, d_ in S.cell_list("diag") if sh == shape and d_ == dd and c.get("t") == t)
        cands = S.candidates(cell, shape, dd)
        sc = S.cell_scenes(cell, shape, dd, models=("pads", "pt"))
        m_pt = mujoco.MjModel.from_xml_path(str(sc["pt"][0]))
        for ci in (a.cands if a.cands is not None else range(len(cands))):
            todo = [s for s in a.seeds if (arm, a.tip, a.cell, ci, s) not in have]
            if not todo:
                continue
            cand = cands[ci]
            gg = S.grasp_geometry(cell, shape, dd, cand["h"], cand["spread"])
            gp = S.grip_poses(m_pt, gg, kp)
            mdl = "pt" if arm == "drake" else "pads"
            path, meta0 = sc[mdl]
            meta = S.candidate_meta(meta0, gg, gp)
            t0 = time.time()
            if arm == "drake":
                files = None
                if a.mesh_eps:
                    import tip_codesign_geometry as TG
                    sh = TG.shape(a.tip)
                    mp = sh.mesh_paths(C.LOGS / "cell10" / "meshes" / sh.key, eps=a.mesh_eps)
                    files = {"tpu_pos": str(mp[1.0]), "tpu_neg": str(mp[-1.0])}
                plant = CellDrake(Path(path), meta, gg["palm"], tip_files=files)
            else:
                plant = HT.MjPlant(path, meta)
                S.set_palm(plant.m, gg["palm"])
            build_s = time.time() - t0
            mirror = S.Mirror(Path(sc["pt"][0]))
            S.set_palm(mirror.m, gg["palm"])
            for s in todo:
                row = {"arm": arm, "tip": a.tip, "cell": a.cell, "cand": ci, "seed": s, "mu": a.mu, "plant": C.SERVO,
                       "tip_tag": tag if arm != "drake" else f"drake_E1e+07_rt0.01_dp{(a.mesh_eps or 0) * 1e6:g}", "scene": str(Path(path).relative_to(ROOT)),
                       "h_mm": round(cand["h"] * 1000, 2), "spread_mm": round(cand["spread"] * 1000, 2),
                       "touch_gap_mm": gp["touch_gap_mm"], **{f"prm_{k}": v for k, v in prm.items()},
                       "build_s": round(build_s, 2), "script": SCRIPT, "git_rev": RB.git_rev(),
                       "when": time.strftime("%Y-%m-%d %H:%M")}
                if arm == "drake":
                    row["drake_geoms"] = plant.n_geoms
                try:
                    res = S.turn_rollout(plant, mirror, meta, s, prm)
                    row.update(res, status="ok")
                except Exception as e:
                    row.update(status="error", error=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-1500:])
                append(a.out, row)
                print(f"{arm:5s} c{ci} s{s}: " + (f"held {row['held']} turn {row['turn_end_deg']:+.1f} frozen {row['frozen_deg']} "
                                                  f"fingers {row['fingers_loaded_end']} F {row['F_end']} {row['wall_s']} s"
                                                  if row["status"] == "ok" else row["error"]), flush=True)
            del plant


if __name__ == "__main__":
    main()
