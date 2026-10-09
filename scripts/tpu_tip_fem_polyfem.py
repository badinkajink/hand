#!/usr/bin/env python3
"""PolyFEM (IPC contact) checks for the TPU tip job, step 18: the force PolyFEM exports against first-principles
references, before PolyFEM is used for large deformation of the printed tip.

Force read-out: the sum of the `contact_forces` field over the deformable body's collision vertices in
<out>/step_<n>_surf_contact.vtu (the collision mesh is the finite-element surface), checked against the reaction at the
Dirichlet face (minus the sum of `elastic_forces` there). The volume file step_<n>.vtu samples the solution on a refined
visualization mesh, so a nodal force summed over it is not a force.

Cases (block 10 x 10 x 5 mm, bottom fixed, E = 1 MPa, nu = 0.3, quasistatic):
  plate   frictionless rigid plate pressed delta into the top; reference: linear finite elements of the same tet mesh
          with the top displaced by the block's own top displacement (scikit-fem), and E A delta / H;
  sphere  rigid sphere R 10 mm pressed delta; reference: Hertz, F = 4/3 E/(1 - nu^2) R^1/2 d^3/2.

    PY=~/miniconda3/envs/femtip/bin/python
    $PY scripts/tpu_tip_fem_polyfem.py plate --h-min 0.5 --h-max 1.0
    $PY scripts/tpu_tip_fem_polyfem.py sphere --h-min 0.05 --h-max 1.2 --sub 6 --dhat 5e-4
Rows: docs/experiments/20261009-tpu_tip_fem/polyfem_runs.jsonl; run directories logs/20261009-tpu_tip_fem/polyfem/<tag>/.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BIN = Path.home() / "Programs/polyfem/build/PolyFEM_bin"
RG = Path.home() / ".claude/bin/resguard.sh"
RUNS = ROOT / "logs/20261009-tpu_tip_fem/polyfem"
ROWS = ROOT / "docs/experiments/20261009-tpu_tip_fem/polyfem_runs.jsonl"
L, W, H = 10.0, 10.0, 5.0
GAP = 0.005


def block_mesh(path, h_min, h_max, r_fine=0.6, r_grow=6.0):
    """Tet mesh of the block, graded from h_min within r_fine of the top centre to h_max beyond r_grow."""
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("block")
    gmsh.model.occ.addBox(0, 0, 0, L, W, H)
    gmsh.model.occ.synchronize()
    p = gmsh.model.occ.addPoint(L / 2, W / 2, H)
    gmsh.model.occ.synchronize()
    f = gmsh.model.mesh.field
    d = f.add("Distance")
    f.setNumbers(d, "PointsList", [p])
    t = f.add("Threshold")
    f.setNumber(t, "InField", d)
    f.setNumber(t, "SizeMin", h_min)
    f.setNumber(t, "SizeMax", h_max)
    f.setNumber(t, "DistMin", r_fine)
    f.setNumber(t, "DistMax", r_grow)
    f.setAsBackgroundMesh(t)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.model.addPhysicalGroup(3, [1], 1)
    gmsh.model.mesh.generate(3)
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.write(str(path))
    n_tet = len(gmsh.model.mesh.getElementsByType(4)[0])
    gmsh.finalize()
    return n_tet


def plate_obj(path, z0, half=12.0):
    with open(path, "w") as fh:
        fh.write(f"v {L / 2 - half} {W / 2 - half} {z0}\nv {L / 2 + half} {W / 2 - half} {z0}\n"
                 f"v {L / 2 + half} {W / 2 + half} {z0}\nv {L / 2 - half} {W / 2 + half} {z0}\n")
        fh.write("f 1 3 2\nf 1 4 3\n")        # normals point down, towards the block


def sphere_obj(path, R, z_bottom, sub):
    import trimesh
    m = trimesh.creation.icosphere(subdivisions=sub, radius=R)
    m.apply_translation([L / 2, W / 2, z_bottom + R])
    m.export(path)
    edge = float(np.mean(m.edges_unique_length))
    return len(m.vertices), edge


def config(mesh, obst, delta, dhat, mu, material, E, nu, steps, solver):
    return {
        "geometry": [{"mesh": mesh, "volume_selection": 1,
                      "surface_selection": [{"id": 1, "axis": "-z", "position": 1e-3}]},
                     {"mesh": obst, "is_obstacle": True, "surface_selection": 2}],
        "contact": {"enabled": True, "dhat": dhat, "friction_coefficient": mu, "epsv": 5e-4},
        "time": {"tend": 1.0, "time_steps": steps, "quasistatic": True},
        "space": {"discr_order": 1},
        "boundary_conditions": {"dirichlet_boundary": [{"id": 1, "value": [0, 0, 0]}],
                                "obstacle_displacements": [{"id": 2, "value": [0, 0, f"-{GAP + delta}*t"]}]},
        "materials": {"type": material, "E": E, "nu": nu, "rho": 1e-3},
        "solver": {"linear": {"solver": solver}},
        "output": {"json": "stats.json",
                   "paraview": {"file_name": "out.vtu", "skip_frame": steps,
                                "options": {"contact_forces": True, "friction_forces": True, "forces": True}}},
    }


def read_forces(out_dir, step):
    """Contact force on the block and the Dirichlet reaction, from the collision-surface file of `step`."""
    import meshio
    m = meshio.read(Path(out_dir) / f"step_{step}_surf_contact.vtu")
    P, d = m.points, m.point_data
    blk = P[:, 2] <= H + 1e-9
    bot = blk & (P[:, 2] < 1e-6)
    top = blk & (P[:, 2] > H - 1e-6)
    u = d["displacement"]
    return dict(F_contact=float(-d["contact_forces"][blk, 2].sum()), F_reaction=float(-d["elastic_forces"][bot, 2].sum()),
                u_top_mean=float(-u[top, 2].mean()), u_top_min=float(-u[top, 2].max()), u_top_max=float(-u[top, 2].min()),
                obstacle_dz=float(-u[~blk, 2].mean()) if (~blk).any() else None, n_block_surface=int(blk.sum()))


def linear_reference(mesh_path, delta_top, E, nu):
    """Linear elastic force of the same tet mesh with the bottom fixed and the top displaced by delta_top along -z
    (free in x and y: a frictionless rigid plate)."""
    import meshio
    import skfem
    from skfem.models.elasticity import lame_parameters, linear_elasticity
    m = meshio.read(mesh_path)
    tets = m.cells_dict["tetra"]
    used = np.unique(tets)
    remap = -np.ones(len(m.points), int)
    remap[used] = np.arange(len(used))
    mesh = skfem.MeshTet(m.points[used].T.copy(), remap[tets].T.copy())
    e = skfem.ElementVector(skfem.ElementTetP1())
    basis = skfem.Basis(mesh, e)
    K = skfem.asm(linear_elasticity(*lame_parameters(E, nu)), basis)
    X = mesh.p
    bot = np.where(X[2] < 1e-9)[0]
    top = np.where(X[2] > H - 1e-9)[0]
    nd = basis.N
    u = np.zeros(nd)
    D = np.concatenate([3 * bot, 3 * bot + 1, 3 * bot + 2, 3 * top + 2])
    u[3 * top + 2] = -delta_top
    I = np.setdiff1d(np.arange(nd), D)
    u = skfem.solve(*skfem.condense(K, np.zeros(nd), x=u, I=I))
    R = K @ u
    return float(-R[3 * top + 2].sum())


def bonded_layer_factor(chi, nu):
    """Force of a rigid sphere on an elastic layer bonded to a rigid base over the Hertz force, chi = sqrt(R d) / h
    (Dimitriadis et al., Biophys. J. 82, 2002, bonded sample, coefficients alpha_0 and beta_0 for Poisson ratio nu)."""
    a0 = -(1.2876 - 1.4678 * nu + 1.3442 * nu ** 2) / (1 - nu)
    b0 = (0.6387 - 1.0277 * nu + 1.5164 * nu ** 2) / (1 - nu)
    pi = math.pi
    return (1 - 2 * a0 / pi * chi + 4 * a0 ** 2 / pi ** 2 * chi ** 2 - 8 / pi ** 3 * (a0 ** 3 + 4 * pi ** 2 * b0 / 15) * chi ** 3
            + 16 * a0 / pi ** 4 * (a0 ** 3 + 3 * pi ** 2 * b0 / 5) * chi ** 4)


def run(tag, cfg, mesh_src, obst_src, threads=4, mem="6G"):
    d = RUNS / tag
    d.mkdir(parents=True, exist_ok=True)
    for src in (mesh_src, obst_src):
        dst = d / Path(src).name
        if Path(src).resolve() != dst.resolve():
            dst.write_bytes(Path(src).read_bytes())
    (d / "run.json").write_text(json.dumps(cfg, indent=1))
    t0 = time.time()
    cmd = [str(RG), "run", "--mem", mem, "--cpu", str(100 * (threads + 1)), "--", "/usr/bin/time", "-v", str(BIN),
           "-j", str(d / "run.json"), "-o", str(d / "out"), "--max_threads", str(threads), "--log_level", "info"]
    with open(d / "run.log", "w") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=d).returncode
    wall = time.time() - t0
    log = (d / "run.log").read_text()
    rss = next((float(l.split(":")[1]) / 1e6 for l in log.splitlines() if "Maximum resident set size" in l), None)
    return rc, wall, rss, d / "out"


def write_row(row):
    ROWS.parent.mkdir(parents=True, exist_ok=True)
    with open(ROWS, "a") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def main():
    global L, W, H
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case", choices=["plate", "sphere"])
    ap.add_argument("--h-min", type=float, default=0.3)
    ap.add_argument("--h-max", type=float, default=1.2)
    ap.add_argument("--delta", type=float, default=None, help="mm; plate 0.05, sphere 0.1")
    ap.add_argument("--dhat", type=float, default=2e-3)
    ap.add_argument("--mu", type=float, default=0.0)
    ap.add_argument("--material", default="NeoHookean")
    ap.add_argument("--E", type=float, default=1.0)
    ap.add_argument("--nu", type=float, default=0.3)
    ap.add_argument("--R", type=float, default=10.0)
    ap.add_argument("--sub", type=int, default=5, help="icosphere subdivisions of the sphere")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--solver", default="Eigen::PardisoLDLT")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--size", type=float, nargs=3, default=[L, W, H], help="block L W H, mm")
    ap.add_argument("--tag")
    a = ap.parse_args()
    L, W, H = a.size
    delta = a.delta if a.delta is not None else (0.05 if a.case == "plate" else 0.1)
    tag = a.tag or f"{a.case}_h{a.h_min:g}_d{a.dhat:g}_{a.material}" + (f"_sub{a.sub}" if a.case == "sphere" else "")
    d = RUNS / tag
    d.mkdir(parents=True, exist_ok=True)
    n_tet = block_mesh(d / "block.msh", a.h_min, a.h_max, r_fine=0.6 if a.case == "sphere" else 20.0)
    info = {}
    if a.case == "plate":
        plate_obj(d / "obstacle.obj", H + GAP)
    else:
        nv, edge = sphere_obj(d / "obstacle.obj", a.R, H + GAP, a.sub)
        info = dict(sphere_vertices=nv, sphere_edge_mm=edge)
    cfg = config("block.msh", "obstacle.obj", delta, a.dhat, a.mu, a.material, a.E, a.nu, a.steps, a.solver)
    rc, wall, rss, out = run(tag, cfg, d / "block.msh", d / "obstacle.obj", threads=a.threads)
    row = dict(case=a.case, tag=tag, rc=rc, n_tet=n_tet, size_mm=[L, W, H], h_min=a.h_min, h_max=a.h_max, delta_mm=delta, dhat_mm=a.dhat,
               mu=a.mu, material=a.material, E=a.E, nu=a.nu, steps=a.steps, solver=a.solver, wall_s=wall, rss_GB=rss,
               script="scripts/tpu_tip_fem_polyfem.py", **info)
    if rc == 0:
        f = read_forces(out, a.steps)
        row.update(f)
        if a.case == "plate":
            row["F_linear_fe"] = linear_reference(d / "block.msh", f["u_top_mean"], a.E, a.nu)
            row["F_EAdH"] = a.E * L * W * f["u_top_mean"] / H
        else:
            dd = f["u_top_max"]                 # approach: the deepest point of the block's top surface
            row["approach_mm"] = dd
            row["F_hertz"] = 4 / 3 * a.E / (1 - a.nu ** 2) * math.sqrt(a.R) * dd ** 1.5
            row["F_bonded_layer"] = row["F_hertz"] * bonded_layer_factor(math.sqrt(a.R * dd) / H, a.nu)
    write_row(row)
    print(json.dumps(row, indent=1))


if __name__ == "__main__":
    main()
