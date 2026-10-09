#!/usr/bin/env python3
"""One-minute smoke tests of the finite-element tools in the femtip conda env (step 1 of the TPU tip job).

  dolfinx   a compressible Neo-Hookean block (E 1 MPa, nu 0.3, quarter of 20 x 20 x 10 mm, graded gmsh tets, bottom
            clamped) indented by a rigid sphere R 10 mm through a penalty on the gap, frictionless, Newton per load
            step; force and contact radius against Hertz.
  sfepy     linear elasticity of a cube in uniaxial compression from its Python interface; modulus recovered.
  skfem     the same cube in scikit-fem; modulus recovered.
  ipctk     the barrier potential and friction dissipative potential between a triangle and a point.

    ~/miniconda3/envs/femtip/bin/python scripts/tpu_tip_fem_smoke.py dolfinx|sfepy|skfem|ipctk
"""
from __future__ import annotations

import math
import sys
import time

import numpy as np


def _graded_quarter_box(L=10.0, H=10.0, h_min=0.08, h_max=1.5, r_fine=1.2, r_far=8.0):
    """gmsh tetrahedral mesh of the quarter block [0, L]^2 x [0, H], refined around (0, 0, H)."""
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("q")
    gmsh.model.occ.addBox(0, 0, 0, L, L, H)
    gmsh.model.occ.synchronize()
    p = gmsh.model.occ.addPoint(0, 0, H)
    gmsh.model.occ.synchronize()
    f = gmsh.model.mesh.field
    d = f.add("Distance")
    f.setNumbers(d, "PointsList", [p])
    t = f.add("Threshold")
    f.setNumber(t, "InField", d)
    f.setNumber(t, "SizeMin", h_min)
    f.setNumber(t, "SizeMax", h_max)
    f.setNumber(t, "DistMin", r_fine)
    f.setNumber(t, "DistMax", r_far)
    f.setAsBackgroundMesh(t)
    for k in ("MeshSizeExtendFromBoundary", "MeshSizeFromPoints", "MeshSizeFromCurvature"):
        gmsh.option.setNumber("Mesh." + k, 0)
    vol = gmsh.model.getEntities(3)
    gmsh.model.addPhysicalGroup(3, [v[1] for v in vol], 1)
    gmsh.model.mesh.generate(3)
    return gmsh


def smoke_dolfinx(depths=(0.01, 0.02, 0.04, 0.06, 0.08, 0.1), E=1.0, nu=0.3, R=10.0, H=10.0):
    """Quarter of a 20 x 20 x 10 mm Neo-Hookean block (bottom clamped, symmetry planes x = 0 and y = 0) indented by a
    rigid sphere R 10 mm on its top centre, penalty on the gap, frictionless; contact radius at 0.1 mm is 1 mm."""
    from mpi4py import MPI
    import ufl
    import dolfinx
    from dolfinx import fem, mesh, default_scalar_type
    from dolfinx.io import gmsh as gmshio
    from dolfinx.fem.petsc import NonlinearProblem
    t0 = time.time()
    g = _graded_quarter_box(H=H)
    md = gmshio.model_to_mesh(g.model, MPI.COMM_WORLD, 0, gdim=3)
    dom = md.mesh if hasattr(md, "mesh") else md[0]
    g.finalize()
    V = fem.functionspace(dom, ("Lagrange", 1, (3,)))
    mu, lam = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))
    u = fem.Function(V)
    v = ufl.TestFunction(V)
    I = ufl.Identity(3)
    F = I + ufl.grad(u)
    J = ufl.det(F)
    psi = mu / 2 * (ufl.tr(F.T * F) - 3) - mu * ufl.ln(J) + lam / 2 * ufl.ln(J) ** 2
    x = ufl.SpatialCoordinate(dom)
    delta = fem.Constant(dom, default_scalar_type(0.0))
    r2 = (x[0] + u[0]) ** 2 + (x[1] + u[1]) ** 2
    gap = (H + R - delta - ufl.sqrt(ufl.max_value(R ** 2 - r2, 1e-8))) - (x[2] + u[2])
    kpen = fem.Constant(dom, default_scalar_type(E * 50 / 0.08))          # overlap stiffness, N/mm^3
    top = mesh.locate_entities_boundary(dom, 2, lambda X: np.isclose(X[2], H))
    ds = ufl.Measure("ds", domain=dom, subdomain_data=mesh.meshtags(dom, 2, top, np.full(len(top), 1, np.int32)))
    Pi = psi * ufl.dx + 0.5 * kpen * ufl.max_value(-gap, 0) ** 2 * ds(1)
    Res = ufl.derivative(Pi, u, v)
    bcs = []
    for ax, val in ((2, 0.0), (0, 0.0), (1, 0.0)):
        facets = mesh.locate_entities_boundary(dom, 2, lambda X, ax=ax, val=val: np.isclose(X[ax], val))
        if ax == 2:
            bcs.append(fem.dirichletbc(np.zeros(3, dtype=default_scalar_type), fem.locate_dofs_topological(V, 2, facets), V))
        else:
            Vs = V.sub(ax)
            bcs.append(fem.dirichletbc(default_scalar_type(0.0), fem.locate_dofs_topological(Vs, 2, facets), Vs))
    problem = NonlinearProblem(Res, u, bcs=bcs, petsc_options_prefix="smoke_",
                               petsc_options={"snes_type": "newtonls", "snes_linesearch_type": "bt",
                                              "snes_rtol": 1e-9, "snes_atol": 1e-11, "snes_max_it": 40,
                                              "ksp_type": "preonly", "pc_type": "lu",
                                              "pc_factor_mat_solver_type": "mumps"})
    Fz = fem.form(kpen * ufl.max_value(-gap, 0) * ds(1))
    Ac = fem.form(ufl.conditional(ufl.lt(gap, 0), 1.0, 0.0) * ds(1))
    Es = E / (1 - nu ** 2)
    rows = []
    for dl in depths:
        delta.value = dl
        problem.solve()
        conv, its = problem.solver.getConvergedReason(), problem.solver.getIterationNumber()
        force = 4 * fem.assemble_scalar(Fz)
        area = 4 * fem.assemble_scalar(Ac)
        hertz = 4 / 3 * Es * math.sqrt(R) * dl ** 1.5
        a_h = math.sqrt(R * dl)
        rows.append(dict(depth=dl, force=force, hertz=hertz, a=math.sqrt(area / math.pi), a_hertz=a_h, its=its, reason=conv))
        print(f"  depth {dl:.3f} mm: force {force * 1e3:7.3f} mN, Hertz {hertz * 1e3:7.3f} mN (ratio {force / hertz:.3f}); "
              f"contact radius {math.sqrt(area / math.pi):.3f} mm, Hertz {a_h:.3f}; Newton its {its}, reason {conv}")
    n_t = dom.topology.index_map(3).size_global
    print(f"  dolfinx {dolfinx.__version__}: {n_t} tets, {V.dofmap.index_map.size_global * 3} dof, {time.time() - t0:.1f} s")
    return rows


def smoke_sfepy():
    import sfepy
    from sfepy.discrete.fem import Mesh, FEDomain, Field
    from sfepy.discrete import FieldVariable, Material, Integral, Equation, Equations, Problem
    from sfepy.discrete.conditions import Conditions, EssentialBC
    from sfepy.terms import Term
    from sfepy.mechanics.matcoefs import stiffness_from_youngpoisson
    from sfepy.mesh.mesh_generators import gen_block_mesh
    from sfepy.solvers.ls import ScipyDirect
    from sfepy.solvers.nls import Newton
    t0 = time.time()
    m = gen_block_mesh([1, 1, 1], [9, 9, 9], [0.5, 0.5, 0.5], name="cube", verbose=False)
    dom = FEDomain("d", m)
    om = dom.create_region("Omega", "all")
    bot = dom.create_region("Bot", "vertices in z < 0.001", "facet")
    top = dom.create_region("Top", "vertices in z > 0.999", "facet")
    xm = dom.create_region("X0", "vertices in x < 0.001", "facet")
    ym = dom.create_region("Y0", "vertices in y < 0.001", "facet")
    field = Field.from_args("u", np.float64, "vector", om, approx_order=1)
    u = FieldVariable("u", "unknown", field)
    v = FieldVariable("v", "test", field, primary_var_name="u")
    E, nu = 30.0, 0.45
    mat = Material("m", D=stiffness_from_youngpoisson(3, E, nu))
    integral = Integral("i", order=2)
    t1 = Term.new("dw_lin_elastic(m.D, v, u)", integral, om, m=mat, v=v, u=u)
    eqs = Equations([Equation("eq", t1)])
    strain = 0.01
    bcs = Conditions([EssentialBC("b", bot, {"u.2": 0.0}), EssentialBC("t", top, {"u.2": -strain}),
                      EssentialBC("x", xm, {"u.0": 0.0}), EssentialBC("y", ym, {"u.1": 0.0})])
    pb = Problem("cube", equations=eqs)
    pb.set_bcs(ebcs=bcs)
    pb.set_solver(Newton({}, lin_solver=ScipyDirect({})))
    state = pb.solve(save_results=False)
    # mean axial stress over the cube (unit volume, unit top area) = force on the top face
    sig = pb.evaluate("ev_cauchy_stress.2.Omega(m.D, u)", mode="eval", m=mat, u=u)
    Fz = -float(np.asarray(sig).ravel()[2])
    print(f"  sfepy {sfepy.__version__}: uniaxial compression modulus {Fz / strain:.3f} MPa (E {E}), {time.time() - t0:.1f} s")
    return Fz / strain


def smoke_skfem():
    import skfem
    from skfem import MeshHex, ElementVector, ElementHex1, Basis, asm, condense, solve
    from skfem.models.elasticity import linear_elasticity, lame_parameters
    t0 = time.time()
    m = MeshHex().refined(3)
    e = ElementVector(ElementHex1())
    ib = Basis(m, e)
    E, nu = 30.0, 0.45
    K = asm(linear_elasticity(*lame_parameters(E, nu)), ib)
    dofs = ib.get_dofs
    D = np.concatenate([ib.get_dofs(lambda x: x[2] == 0).nodal["u^3"], ib.get_dofs(lambda x: x[2] == 1).nodal["u^3"],
                        ib.get_dofs(lambda x: x[0] == 0).nodal["u^1"], ib.get_dofs(lambda x: x[1] == 0).nodal["u^2"]])
    x = np.zeros(K.shape[0])
    top = ib.get_dofs(lambda x: x[2] == 1).nodal["u^3"]
    x[top] = -0.01
    u = solve(*condense(K, x=x, D=D))
    Fz = -(K @ u)[top].sum()
    print(f"  scikit-fem {skfem.__version__}: uniaxial compression modulus {Fz / 0.01:.3f} MPa (E {E}), {time.time() - t0:.1f} s")
    return Fz / 0.01


def smoke_ipctk():
    import ipctk
    V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.2, 0.2, 1e-3]], float)
    F = np.array([[0, 1, 2]])
    E = ipctk.edges(F)
    cm = ipctk.CollisionMesh(V, E, F)
    dhat = 1e-2
    col = ipctk.NormalCollisions()
    col.build(cm, V, dhat)
    B = ipctk.BarrierPotential(dhat, 1.0)
    print(f"  ipctk {ipctk.__version__ if hasattr(ipctk, '__version__') else ''}: {len(col)} collision(s), barrier "
          f"{B(col, cm, V):.4e}, min distance {col.compute_minimum_distance(cm, V):.3e}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in (("dolfinx", smoke_dolfinx), ("sfepy", smoke_sfepy), ("skfem", smoke_skfem), ("ipctk", smoke_ipctk)):
        if which in (name, "all"):
            print(name)
            fn()
