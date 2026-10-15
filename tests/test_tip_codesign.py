"""Checks of the fingertip co-design geometry and dispatcher (scripts/tip_codesign_geometry.py, tip_codesign_rig.py,
tip_codesign_bed.py).

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_tip_codesign.py -q
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import tip_codesign_geometry as TG  # noqa: E402


def test_as_built_matches_cad_outer_surface():
    """The as-built member (r_side = r_end = 3.0 mm) against the CAD insert: radial distance of CAD surface samples
    on the outer palmar half (normal n_x > -0.5, 1 mm or more from the mount face) to the shape, p99 under 0.05 mm.
    The insert is a shell over the mount (an inner slot from the mount face), so volume is not compared."""
    trimesh = pytest.importorskip("trimesh")
    cad = trimesh.load(ROOT / "assets/cad_v1/fingertip.stl")
    b, fi = trimesh.sample.sample_surface(cad, 40000, seed=3)
    bn = cad.face_normals[fi]
    sh = TG.shape("tcd_rs3_re3")
    P = np.column_stack([b[:, 0] * 1e-3 + (TG.REACH - 0.0085), b[:, 1] * 1e-3, TG.Z_MOUNT - b[:, 2] * 1e-3])
    D = P - TG.C0
    r = np.linalg.norm(D, axis=1)
    S = sh._bisect(D / r[:, None], sh.inside_design)
    d = (r - np.linalg.norm(S - TG.C0, axis=1)) * 1e3
    outer = (d > -0.2) & (bn[:, 0] > -0.5) & (b[:, 2] > 1.0)
    assert outer.sum() > 20000
    assert np.percentile(np.abs(d[outer]), 99) < 0.05


def test_volume_centroid_matches_grid_integral():
    sh = TG.shape("tcd_rs3_re3")
    cV, vol = sh.volume_centroid
    ax = [(lo + (np.arange(n) + 0.5) * (hi - lo) / n, (hi - lo) / n)
          for lo, hi, n in ((TG.X_BACK, TG.X_FACE, 113), (-TG.WIDTH / 2, TG.WIDTH / 2, 99), (TG.Z_END, TG.Z_MOUNT, 143))]
    X, Y, Z = np.meshgrid(ax[0][0], ax[1][0], ax[2][0], indexing="ij")
    P = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    ins = sh.inside_design(P)
    c_grid = P[ins].mean(0)
    assert abs(ins.sum() * ax[0][1] * ax[1][1] * ax[2][1] - vol) / vol < 0.005
    assert np.abs(c_grid - cV).max() < 1e-5


def test_mesh_paths_are_keyed(tmp_path):
    a = TG.shape("tcd_rs3_re3").mesh_paths(tmp_path / "a")
    b = TG.shape("tcd_rs1_re3").mesh_paths(tmp_path / "a")
    assert not set(map(str, a.values())) & set(map(str, b.values()))


def test_pads_cover_the_padded_region():
    sh = TG.shape("tcd_rs3_re3")
    C, N, A, K = sh.pads(full=True)
    P, Nd, Ad = sh.dense
    m = (Nd[:, 0] > -0.5) & (Nd[:, 2] < 0.5)
    assert abs(A.sum() - Ad[m].sum()) / Ad[m].sum() < 1e-6
    assert 0.4e-6 < A.min() and A.max() < 2.5e-6
    assert np.all(sh.inside(C, tol=1e-9))


def test_tpu_route_reproduces_u2_d7_seed0(tmp_path):
    """The dispatcher leaves tpu keys to reorient_backends: D7 seed 0, tpu2.7 pads, servo plant, mu 1.0 ends at the
    U2 row's turn and held outcome."""
    pytest.importorskip("mujoco")
    import argparse
    import tip_codesign_rig as R
    out = tmp_path / "rows.jsonl"
    R.SCN = tmp_path / "scenes"
    a = argparse.Namespace(cmd="replay", hands=["D7"], seeds=[0], E=1e7, pf=None, tr=0.02, tag=None, h="fixed",
                           K="median", out=out, sim="mujoco", tips=["tpu2.7"], models=["pads"], plants=[R.SERVO],
                           numerics=["bed"], ir=[100.0], mu=["1.0"], dt=None, spin=None, jitter="j2", traces=False)
    R.cmd_replay(a)
    row = json.loads(out.read_text().splitlines()[-1])
    assert row["held_end"] is True
    assert abs(row["turn_end_deg"] - 44.82255084959616) < 1e-6


def test_per_pad_stiffness_is_realised():
    """On the bed with per-pad K (tcd ...:Keach), every pad touching the tool after a 1 N settle carries
    f = K_i x penetration within 1 %."""
    pytest.importorskip("mujoco")
    import hom_contact_rig as H
    import tip_codesign_bed as B
    import tpu_tip_rig as T
    name, v = B.tcd_model("tcd_rs3_re3", "Keach")
    B.register({name: v})
    with T.tpu_rigs():
        H.DT = 1e-3
        rig = H.make_rig(v[0], d_cg=0.0, gravity=False)
    rig.set_pad_force(1.0)
    rig.set_tool_wrench(np.zeros(3), np.zeros(3))
    rig.step(1.0)
    sh = TG.shape("tcd_rs3_re3")
    _, _, _, K = sh.pads(1e7, 0.001, 0.00075, 1.0, K="each", full=True)
    f6 = np.zeros(6)
    ratios = []
    for i in range(rig.d.ncon):
        c = rig.d.contact[i]
        g = [int(c.geom[0]), int(c.geom[1])]
        nm = [rig.m.geom(k).name for k in g]
        pad = [n for n in nm if n.startswith("padL_s")]
        if not pad or c.dist >= -1e-7:
            continue
        rig.mj.mj_contactForce(rig.m, rig.d, i, f6)
        ratios.append(f6[0] / (-c.dist * K[int(pad[0][6:])]))
    assert len(ratios) > 10
    assert np.abs(np.array(ratios) - 1).max() < 0.01
