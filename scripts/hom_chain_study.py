#!/usr/bin/env python3
"""Driver for docs/experiments/20261002-hom_chain: Exp 1 of arXiv 2609.25619 on D8, the nominal chain filmed
in six contact models, a perturbed MuJoCo batch (seeds 1-10), and a close-up of the sphere-packed pad.
Resumable: rows already in the jsonl files are skipped.

    PY=logs/20261001-hom_contact/venv/bin/python
    MUJOCO_GL=egl $PY scripts/hom_chain_study.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import hom_chain as K  # noqa: E402
import hom_contact_rig as H  # noqa: E402
import hom_control as C  # noqa: E402
import hom_hand_brake as B  # noqa: E402

OUT = ROOT / "docs/experiments/20261002-hom_chain"
MEDIA = OUT / "media"
NOMINAL = {"p4s_closed": ("mj:point4s", "closed"), "p4s_open": ("mj:point4s", "open"),
           "s1_closed": ("mj:spheres:s1:rs0.75:tr0.02", "closed"), "mp3_closed": ("mj:point3", "closed"),
           "dhy_closed": ("drake:hydro:rt0.01", "closed"), "dpt_closed": ("drake:point", "closed")}
BATCH = {"p4s": "mj:point4s", "s1": "mj:spheres:s1:rs0.75:tr0.02"}
SEEDS = range(1, 11)


def done(path, key):
    if not path.exists():
        return set()
    return {key(json.loads(l)) for l in open(path) if l.strip()}


def main():
    MEDIA.mkdir(parents=True, exist_ok=True)
    e1 = OUT / "exp1.jsonl"
    have = done(e1, lambda r: r["plant"])
    for plant in ("cal", "stiff"):
        if plant not in have:
            r = C.exp1(plant, 500.0, film=str(MEDIA / f"20261002-exp1_{plant}.mp4"))
            H.append_row(e1, r)
            print("exp1", plant, round(r["rmse_ang_dps"], 1), round(r["rmse_lin_mmps"], 1), flush=True)
    ch = OUT / "chain.jsonl"
    have = done(ch, lambda r: (r["spec"], r["brake"], r["trial"]["seed"]))
    for key, (spec, brake) in NOMINAL.items():
        if (spec, brake, 0) in have:
            continue
        r = K.run_chain(spec, 0, brake, film=str(MEDIA / f"20261002-chain_{key}.mp4"))
        r["key"] = key
        H.append_row(ch, r)
        print("nominal", key, r["phi_end"], r["chain_ok"], round(r["wall_s"], 1), flush=True)
    for bk, spec in BATCH.items():
        for brake in ("open", "closed"):
            for seed in SEEDS:
                if (spec, brake, seed) in have:
                    continue
                r = K.run_chain(spec, seed, brake)
                r["key"] = f"{bk}_{brake}"
                r.pop("trace", None)
                H.append_row(ch, r)
                print("batch", bk, brake, seed, r["phi_end"], r["chain_ok"], flush=True)
    tile_films()
    closeup()


def tile_films():
    import imageio.v2 as imageio
    from PIL import Image
    order = ["dhy_closed", "s1_closed", "p4s_closed", "dpt_closed", "mp3_closed", "p4s_open"]
    fr = {k: [f for f in imageio.get_reader(str(MEDIA / f"20261002-chain_{k}.mp4"))] for k in order}
    n = max(len(v) for v in fr.values())
    get = lambda k, i: fr[k][min(i, len(fr[k]) - 1)]  # noqa: E731
    grid = [np.concatenate([np.concatenate([get(k, i) for k in order[:3]], 1),
                            np.concatenate([get(k, i) for k in order[3:]], 1)], 0) for i in range(n)]
    H.write_mp4(grid, MEDIA / "20261002-chain_six_models.mp4", fps=25)
    Image.fromarray(grid[min(n - 1, 130)]).save(MEDIA / "20261002-chain_six_models_poster.png")


def closeup():
    """Thumb pad at 1, 0.5 and 2 mm sphere spacing pressed 0.4 mm into the tool, contacts drawn."""
    import mujoco
    from PIL import Image
    path, _ = K.postures([0.0, -0.0004])
    q = path[-0.0004]
    dirs = K.contact_dirs(path[0.0])
    trial = K.make_trial(0)
    imgs = []
    for s_mm in (2, 1, 0.5):
        xml, info, lay = K.chain_scene(f"mj:spheres:s{s_mm}:rs0.75:tr0.02", trial, dirs)
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        P, u, a = K.pinch_frame()
        s_hat = lay["c"] - trial["d_cg"] * lay["axis"]
        d.qpos[[m.jnt_qposadr[m.joint(n).id] for f in B.FINGERS for n in B.JOINTS[f]]] = q
        d.qpos[[m.jnt_qposadr[m.joint(n).id] for n in ("palm_x", "palm_y", "palm_z", "palm_yaw")]] = np.r_[s_hat - P, 0]
        mujoco.mj_forward(m, d)
        tool = m.body("tool").id
        ncon = sum(1 for i in range(d.ncon) if tool in (m.geom_bodyid[d.contact[i].geom[0]], m.geom_bodyid[d.contact[i].geom[1]])
                   and m.body(m.geom_bodyid[d.contact[i].geom[0]]).name.startswith("thumb"))
        m.geom_rgba[m.geom("tool").id] = [0.55, 0.6, 0.68, 0.25]
        r = mujoco.Renderer(m, 480, 480)
        opt = mujoco.MjvOption()
        opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        m.vis.scale.contactwidth, m.vis.scale.contactheight = 0.08, 0.02
        cam = mujoco.MjvCamera()
        tip = d.xpos[m.body("thumb_tip").id]
        cam.lookat[:] = tip + 0.006 * u
        cam.distance, cam.azimuth, cam.elevation = 0.07, math.degrees(math.atan2(-u[1], -u[0])) + 35, -20
        r.update_scene(d, cam, opt)
        img = H.annotate(r.render().copy(), f"{s_mm:g} mm spacing: {info['n_spheres']} spheres, {ncon} touching", size=14)
        imgs.append(img)
    Image.fromarray(np.concatenate(imgs, 1)).save(MEDIA / "20261002-sphere_pad_closeup.png")
    print("closeup done", flush=True)


if __name__ == "__main__" and len(sys.argv) == 1:
    main()


PERF = {"mp3": "mj:point3", "p4s": "mj:point4s", "s2": "mj:spheres:s2:rs0.75:tr0.02",
        "s1": "mj:spheres:s1:rs0.75:tr0.02", "s05": "mj:spheres:s0.5:rs0.75:tr0.02",
        "dpt": "drake:point", "dhy": "drake:hydro:rt0.01"}


def perf():
    """Nominal chain with the closed-loop brake, no film, one core: wall time per simulated second."""
    import time
    p = OUT / "perf.jsonl"
    have = done(p, lambda r: r["spec"])
    for key, spec in PERF.items():
        if spec in have:
            continue
        try:
            r = K.run_chain(spec, 0, "closed")
        except ValueError as e:                     # pad stiffness/relaxation outside MuJoCo's soft-contact range
            H.append_row(p, {"key": key, "spec": spec, "error": str(e)})
            print("perf", key, "skipped:", e, flush=True)
            continue
        H.append_row(p, {"key": key, "spec": spec, "wall_s": r["wall_s"], "sim_s": r["sim_s"], "phi_end": r["phi_end"],
                         "chain_ok": r["chain_ok"], "n_spheres": (r.get("info") or {}).get("n_spheres"),
                         "when": time.strftime("%Y-%m-%d %H:%M")})
        print("perf", key, round(r["wall_s"], 2), round(r["sim_s"], 2), r["phi_end"], flush=True)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "perf":
    perf()


FILMS = {   # key: (spec, brake, seed); nominal films of every contact model, then perturbed trials
    "dhy_closed": ("drake:hydro:rt0.01", "closed", 0),
    "s05_closed": ("mj:spheres:s0.5:rs0.75:tr0.03", "closed", 0),
    "s1_closed": ("mj:spheres:s1:rs0.75:tr0.02", "closed", 0),
    "s2_closed": ("mj:spheres:s2:rs0.75:tr0.02", "closed", 0),
    "p4s_closed": ("mj:point4s", "closed", 0),
    "p4s_open": ("mj:point4s", "open", 0),
    "mp3_closed": ("mj:point3", "closed", 0),
    "dpt_closed": ("drake:point", "closed", 0),
    "p4s_closed_s1": ("mj:point4s", "closed", 1),
    "p4s_closed_s4": ("mj:point4s", "closed", 4),
    "p4s_closed_s3": ("mj:point4s", "closed", 3),
    "p4s_open_s3": ("mj:point4s", "open", 3),
    "p4s_closed_s9": ("mj:point4s", "closed", 9),
    "p4s_open_s9": ("mj:point4s", "open", 9),
}


def films():
    """Every FILMS entry rendered as wide shot + thumb-pad close-up (rows in films.jsonl), then two grids:
    the six-model wide shots and the five contact close-ups side by side."""
    import imageio.v2 as imageio
    from PIL import Image
    MEDIA.mkdir(parents=True, exist_ok=True)
    p = OUT / "films.jsonl"
    have = done(p, lambda r: r["key"])
    for key, (spec, brake, seed) in FILMS.items():
        film = MEDIA / f"20261002-chain_{key}.mp4"
        if key in have and film.exists():
            continue
        r = K.run_chain(spec, seed, brake, film=str(film))
        r["key"] = key
        H.append_row(p, r)
        print("film", key, r["phi_end"], r["chain_ok"], round(r["wall_s"], 1), flush=True)
    rd = lambda k: [f for f in imageio.get_reader(str(MEDIA / f"20261002-chain_{k}.mp4"))]  # noqa: E731
    fr = {k: rd(k) for k in ("dhy_closed", "s1_closed", "p4s_closed", "dpt_closed", "mp3_closed", "p4s_open",
                             "s2_closed", "s05_closed")}
    n = max(len(v) for v in fr.values())
    get = lambda k, i: fr[k][min(i, len(fr[k]) - 1)]  # noqa: E731
    order = ["dhy_closed", "s1_closed", "p4s_closed", "dpt_closed", "mp3_closed", "p4s_open"]
    grid = [np.concatenate([np.concatenate([get(k, i)[:, :480] for k in order[:3]], 1),
                            np.concatenate([get(k, i)[:, :480] for k in order[3:]], 1)], 0) for i in range(n)]
    H.write_mp4(grid, MEDIA / "20261002-chain_six_models.mp4", fps=25)
    Image.fromarray(grid[min(n - 1, 130)]).save(MEDIA / "20261002-chain_six_models_poster.png")
    close = ["dhy_closed", "s05_closed", "s1_closed", "s2_closed", "p4s_closed"]
    strip = [np.concatenate([get(k, i)[:, 480:] for k in close], 1) for i in range(n)]
    H.write_mp4(strip, MEDIA / "20261002-contact_closeups.mp4", fps=25)
    Image.fromarray(strip[min(n - 1, 60)]).save(MEDIA / "20261002-contact_closeups_poster.png")
    print("grids done", flush=True)


BENCH = [("mj:point3", 5), ("mj:point4s", 5), ("mj:spheres:s2:rs0.75:tr0.02", 5), ("mj:spheres:s1:rs0.75:tr0.02", 5),
         ("mj:spheres:s0.5:rs0.75:tr0.03", 3), ("drake:point", 3), ("drake:hydro:rt0.01:hr2", 3),
         ("drake:hydro:rt0.01", 3), ("drake:hydro:rt0.01:hr0.5", 2)]


def bench():
    """Nominal chain, closed-loop brake, no film, one core: physics time per stage (inside the simulator's step),
    controller time (mirror, frames, least squares, brake), contacts per stage. Repeated runs per model."""
    import time
    p = OUT / "bench.jsonl"
    have = done(p, lambda r: (r["spec"], r["rep"]))
    for spec, reps in BENCH:
        for rep in range(reps):
            if (spec, rep) in have:
                continue
            r = K.run_chain(spec, 0, "closed")
            ix = {c: i for i, c in enumerate(r["trace_cols"])}
            ncon = {}
            for row in r["trace"]:
                ncon.setdefault(row[ix["phase"]], []).append(row[ix["n_thumb"]] + row[ix["n_index"]])
            H.append_row(p, {"spec": spec, "rep": rep, "wall_s": r["wall_s"], "sim_s": r["sim_s"], "perf": r["perf"],
                             "ncon": {k: float(np.mean(v)) for k, v in ncon.items()}, "phi_end": r["phi_end"],
                             "chain_ok": r["chain_ok"], "n_spheres": (r.get("info") or {}).get("n_spheres"),
                             "when": time.strftime("%Y-%m-%d %H:%M")})
            print("bench", spec, rep, round(r["wall_s"], 2), round(sum(r["perf"]["phys"].values()), 2),
                  round(r["perf"]["ctrl"], 2), flush=True)


CHECK = ["mj:point3", "mj:point4s", "mj:spheres:s2:rs0.75:tr0.02", "mj:spheres:s1:rs0.75:tr0.02",
         "mj:spheres:s0.5:rs0.75:tr0.03", "drake:point", "drake:hydro:rt0.01:hr2", "drake:hydro:rt0.01"]


class _Stop(Exception):
    pass


def _hold_times():
    """Middle of the hold phase (tool pinched at 3 N) in each model's filmed nominal run."""
    out = {}
    for line in open(OUT / "films.jsonl"):
        r = json.loads(line)
        if r.get("brake") != "closed" or r["trial"]["seed"] != 0:
            continue
        ix = {c: i for i, c in enumerate(r["trace_cols"])}
        ts = [row[ix["t"]] for row in r["trace"] if row[ix["phase"]] == "hold"]
        if ts:
            out[r["spec"]] = 0.5 * (ts[0] + ts[-1])
    return out


def bench_check():
    """Independent check of the cost table. The nominal chain runs to the middle of its hold phase; the controller
    then stops and the bare physics is timed from that state, servo targets frozen, three windows of 1 s simulated.
    MuJoCo: mj_step(m, d, nstep=1000) inside C on a copy of the state, and the plant's own Python step loop (which
    adds the condim-4 rescheduling); Drake: Simulator.AdvanceTo. Then a one-contact scene in each simulator: one
    fingertip on a vertical slide resting on the fixed tool cylinder, nothing else."""
    import time
    import mujoco
    p = OUT / "bench_check.jsonl"
    hold = _hold_times()
    have = done(p, lambda r: r.get("spec") or r.get("model"))
    for spec in CHECK:
        if spec in have:
            continue
        sp = B.parse_spec(spec)
        res = {"spec": spec, "t_snap": hold.get(spec) or hold["drake:hydro:rt0.01"], "when": time.strftime("%Y-%m-%d %H:%M")}
        cls = K.MjChainPlant if sp["sim"] == "mj" else K.DrakeChainPlant
        orig = cls.step

        def step(self, T, _orig=orig, _res=res):
            _orig(self, T)
            if self.t < _res["t_snap"]:
                return
            if self.sim == "mujoco":
                m, d = self.m, self.d
                c_us, ncon, nefc, nit = [], [], [], []
                for _ in range(3):
                    d2 = mujoco.MjData(m)
                    mujoco.mj_copyData(d2, m, d)
                    t0 = time.perf_counter()
                    mujoco.mj_step(m, d2, nstep=1000)
                    c_us.append((time.perf_counter() - t0) * 1e3)
                d2 = mujoco.MjData(m)
                mujoco.mj_copyData(d2, m, d)
                for _ in range(300):
                    mujoco.mj_step(m, d2)
                    ncon.append(d2.ncon)
                    nefc.append(d2.nefc)
                    nit.append(int(d2.solver_niter[0]))
                py = []
                for _ in range(3):
                    t0 = time.perf_counter()
                    _orig(self, 1.0)
                    py.append((time.perf_counter() - t0) * 1e3)
                _res.update(c_us_per_step=c_us, plant_us_per_step=py, ncon=float(np.mean(ncon)), nefc=float(np.mean(nefc)),
                            solver_iter=float(np.mean(nit)), nv=int(m.nv), ngeom=int(m.ngeom),
                            held=bool(self.contacts()["thumb"]["n"] > 0 and self.contacts()["index"]["n"] > 0))
            else:
                py = []
                for _ in range(3):
                    t0 = time.perf_counter()
                    self.simulator.AdvanceTo(self.t + 1.0)
                    py.append((time.perf_counter() - t0) * 1e3)
                cc = self.contacts()
                _res.update(plant_us_per_step=py, contacts=cc["thumb"]["n"] + cc["index"]["n"],
                            held=bool(cc["thumb"]["n"] > 0 and cc["index"]["n"] > 0),
                            nv=int(self.plant.num_velocities()))
            raise _Stop

        cls.step = step
        try:
            K.run_chain(spec, 0, "closed")
        except _Stop:
            pass
        finally:
            cls.step = orig
        H.append_row(p, res)
        print("check", spec, {k: v for k, v in res.items() if k not in ("spec", "when")}, flush=True)
    for model in ("mj:point3", "mj:point4", "mj:spheres1", "drake:point", "drake:hydro"):
        if model in have:
            continue
        r = one_contact(model)
        r["when"] = time.strftime("%Y-%m-%d %H:%M")
        H.append_row(p, r)
        print("one contact", r, flush=True)


def one_contact(model, T=1.0):
    """One fingertip sphere (r 10.55 mm, 20 g) on a vertical slide, resting by its weight on the fixed tool
    cylinder (r 12.5 mm, axis horizontal); same solver settings as the chain. Wall time per 1 ms step."""
    import time
    r_t, r_c = B.R_TIP, H.R_TOOL
    z0 = r_c + r_t - 1e-4
    if model.startswith("mj"):
        import mujoco
        cond = {"mj:point3": 'condim="3" friction="1 0 0"', "mj:point4": 'condim="4" friction="1 0.001 0"',
                "mj:spheres1": 'condim="3" friction="1 0 0" contype="0" conaffinity="0"'}[model]
        pads = ""
        if model == "mj:spheres1":
            s, rs, cap = 0.001, 0.00075, math.radians(45.0)
            area = 2 * math.pi * r_t ** 2 * (1 - math.cos(cap))
            n = int(round(area / s ** 2))
            K_s = 1e7 / r_t * area / n * ((r_t - rs) / r_t) ** 2
            tc = 0.01
            diag = 1.0 / 0.02                         # the tip's inverse mass; the cylinder is fixed
            d0 = 1.0 - 1.0 / (tc ** 2 * K_s * diag)
            for i, dv in enumerate(H.fib_cap(n, cap)):
                q = (r_t - rs) * np.array([dv[0], dv[1], -dv[2]])
                pads += (f'<geom type="sphere" size="{rs}" pos="{q[0]:.7f} {q[1]:.7f} {q[2]:.7f}" condim="3" friction="1 0 0" '
                         f'priority="1" solref="{tc} 1" solimp="{d0:.5f} {d0:.5f} 0.001 0.5 2" mass="0"/>')
        xml = f"""<mujoco><option timestep="0.001" cone="elliptic" impratio="100" solver="Newton" iterations="200" tolerance="1e-10"/>
<worldbody><geom type="cylinder" size="{r_c} 0.05" euler="90 0 0" friction="0.3 0 0" solref="0.006 1" solimp="0.97 0.995 0.0005"/>
<body pos="0 0 {z0}"><joint type="slide" axis="0 0 1"/>
<geom type="sphere" size="{r_t}" mass="0.02" priority="1" solref="0.006 1" solimp="0.97 0.995 0.0005" {cond}/>{pads}</body>
</worldbody></mujoco>"""
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        mujoco.mj_step(m, d, nstep=200)
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            mujoco.mj_step(m, d, nstep=int(T * 1000))
            ts.append((time.perf_counter() - t0) * 1e3 / T)
        return {"model": model, "c_us_per_step": ts, "ncon": int(d.ncon), "nefc": int(d.nefc)}
    from pydrake.all import (AddCompliantHydroelasticProperties, AddContactMaterial, AddMultibodyPlant,
                             AddRigidHydroelasticProperties, CoulombFriction, Cylinder, DiagramBuilder,
                             MultibodyPlantConfig, PrismaticJoint, ProximityProperties, RigidTransform,
                             RotationMatrix, Simulator, SpatialInertia, Sphere)
    hydro = model == "drake:hydro"
    b = DiagramBuilder()
    plant, sg = AddMultibodyPlant(MultibodyPlantConfig(time_step=1e-3, discrete_contact_approximation="sap",
                                                       contact_model="hydroelastic_with_fallback" if hydro else "point"), b)
    tip = plant.AddRigidBody("tip", SpatialInertia.SolidSphereWithMass(0.02, r_t))
    plant.AddJoint(PrismaticJoint("z", plant.world_frame(), tip.body_frame(), [0, 0, 1]))
    pc_ = ProximityProperties()
    if hydro:
        AddCompliantHydroelasticProperties(1e-3, 1e7, pc_)
    AddContactMaterial(dissipation=10.0, point_stiffness=1e4, friction=CoulombFriction(1.0, 1.0), properties=pc_)
    pc_.AddProperty("material", "relaxation_time", 0.01)
    plant.RegisterCollisionGeometry(tip, RigidTransform(), Sphere(r_t), "tip", pc_)
    pr = ProximityProperties()
    if hydro:
        AddRigidHydroelasticProperties(0.0005, pr)
    AddContactMaterial(dissipation=10.0, point_stiffness=1e4, friction=CoulombFriction(1.0, 1.0), properties=pr)
    pr.AddProperty("material", "relaxation_time", 0.01)
    plant.RegisterCollisionGeometry(plant.world_body(), RigidTransform(RotationMatrix.MakeXRotation(math.pi / 2), [0, 0, 0]),
                                    Cylinder(r_c, 0.1), "tool", pr)
    plant.Finalize()
    sim = Simulator(b.Build())
    ctx = sim.get_mutable_context()
    plant.GetJointByName("z").set_translation(plant.GetMyMutableContextFromRoot(ctx), z0)
    sim.Initialize()
    sim.AdvanceTo(0.2)
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        sim.AdvanceTo(ctx.get_time() + T)
        ts.append((time.perf_counter() - t0) * 1e3 / T)
    cr = plant.get_contact_results_output_port().Eval(plant.GetMyContextFromRoot(ctx))
    nf = sum(cr.hydroelastic_contact_info(i).contact_surface().num_faces() for i in range(cr.num_hydroelastic_contacts()))
    return {"model": model, "plant_us_per_step": ts, "faces": nf, "point_pairs": cr.num_point_pair_contacts()}


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] in ("films", "bench", "check"):
    {"films": films, "bench": bench, "check": bench_check}[sys.argv[1]]()
