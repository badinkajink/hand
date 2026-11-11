#!/usr/bin/env python3
"""Numbers, figures and film of the sphere-pad implementation page (scripts/contact/sphere_pad_code_page.py), 2026-11-11.

Owner, 2026-11-11: "an accompaniment to the sphere packed contact model, similar to the hom/hom-implementation docpages,
that just details specifically how we implement the dense sphere packing for an arbitrary surface, with links out to the
collapsed code".

Subcommands, each through resguard (the surface step bisects 300,000 rays; the press and the film load a mesh):
  steps    the v4b tip (tip_codesign_geometry key tcd_ell7.4x8.5x11x22) through TipShape.pads with its intermediates,
           checked against the builder's own output; Figures 1 to 3
  scene    the pads of one scene of the talk's landscapes against the builder: positions, solimp d0, stiffness
  press    a flat plate pressed on one tip in MuJoCo, at rest, against the pads' springs (overview Eq. 10) and the
           continuous law; Figure 4
  film     the press as a film, on Mesa's software renderer (off the GPU while a MuJoCo-Warp queue runs)

    ~/.claude/bin/resguard.sh run --mem 3G --cpu 300 -- .venv/bin/python scripts/contact/sphere_pad_code_data.py steps
Rows: docs/experiments/20261111-sphere_pad_implementation/rows/20261111-{steps,scene,press}.json, 20261111-pads.npz;
media: the same folder's media/.
"""
from __future__ import annotations

import inspect
import json
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path

D = ROOT / "docs/experiments/20261111-sphere_pad_implementation"
ROWS, MEDIA = D / "rows", D / "media"
KEY = "tcd_ell7.4x8.5x11x22"            # the v4b tip of the cad_v2 hand (build_cad_v2_finger.py)
SCENE = ROOT / ("logs/20261102-hom_servo_plant/scenes/cyl025.0/"
                "x092.5_y102.5_pads_bv2__kp0.72_kv0_fr0.18_fl0_dp0.0144.xml")   # a scene of the talk's landscapes
SIGN = {"thumb": 1.0, "index": -1.0, "middle": -1.0}
FORCES = (0.25, 0.5, 1.0, 2.0, 4.0)     # N, the flat presses
M_PLATE = 0.025                         # kg, the HOM objects' mass
SOFT_GL = {"MUJOCO_GL": "egl", "__EGL_VENDOR_LIBRARY_FILENAMES": "/usr/share/glvnd/egl_vendor.d/50_mesa.json",
           "EGL_PLATFORM": "surfaceless", "LP_NUM_THREADS": "3"}
PAD_RGBA = (0.75, 0.4, 0.25, 1.0)       # the scenes' pad colour


def q(x, p):
    return float(np.percentile(x, p))


def save(name, obj):
    ROWS.mkdir(parents=True, exist_ok=True)
    p = ROWS / f"20261111-{name}.json"
    p.write_text(json.dumps(obj, indent=1) + "\n")
    print("wrote", p.relative_to(ROOT), flush=True)


# ---------------------------------------------------------------------------------------------------- steps
def build_steps():
    """The v4b tip through TipShape.pads, keeping what the method throws away; the end result must equal pads()."""
    import reorient_backends as RB
    import tip_codesign_geometry as TG
    from scipy.spatial import cKDTree
    src = inspect.getsource(TG.TipShape.hull_fine.func)
    assert "fibonacci(300_000)" in src, "hull_fine's ray count changed: update N_RAYS"
    n_rays = 300_000
    sh = TG.shape(KEY)
    s, rs, E = RB.PAD_S, RB.PAD_RS, RB.E_TPU
    V, F = sh.hull_fine
    P0, N0, A0 = sh.dense
    m = (N0[:, 0] > -0.5) & (N0[:, 2] < 0.5)             # TipShape.pads' mask
    P, N, A = P0[m], N0[m], A0[m]
    with sh._origin():
        inv, _ = TG._voxel_pick(P, A, s)
        n_vox = inv.max() + 1
        Aw0 = np.bincount(inv, A, n_vox)
        Pv = np.column_stack([np.bincount(inv, A * P[:, k], n_vox) for k in range(3)]) / Aw0[:, None]
        keep = Aw0 >= 0.3 * s * s
        idx = np.flatnonzero(keep)
        _, near = cKDTree(Pv[idx]).query(P)
        own = np.full(n_vox, -1)
        own[idx] = np.arange(len(idx))
        moved = own[inv] != near                          # samples whose pad is not their voxel's
        Aw = np.bincount(near, A, len(idx))
        Pm = np.column_stack([np.bincount(near, A * P[:, k], len(idx)) for k in range(3)]) / Aw[:, None]
        Nm = np.column_stack([np.bincount(near, A * N[:, k], len(idx)) for k in range(3)])
        Nm /= np.linalg.norm(Nm, axis=1, keepdims=True)
        D_ = Pm - sh.c0
        D_ /= np.linalg.norm(D_, axis=1, keepdims=True)
        Ps = sh._bisect(D_, sh.inside_design)             # the means back onto the surface
        C0_ = Ps - rs * Nm
        C = sh._clamp(C0_, rs)
    cV = sh.centroid()
    h = np.maximum(np.einsum("ij,ij->i", Nm, Ps - cV), 1e-3)
    K = E / h * Aw
    Cr, Nr, Ar, Kr = sh.pads(E, s, rs, 1.0, h="local", K="each", full=True)
    assert np.allclose(C, Cr, atol=1e-12) and np.allclose(Nm, Nr) and np.allclose(Aw, Ar) and np.allclose(K, Kr), \
        "the replica of TipShape.pads differs from the method"
    # the same through reorient_backends.pad_set, which replace_tips calls
    Cb, Nb, Ab, Kb = RB.pad_set(KEY, s, rs, 1.0, E, "v2")
    assert np.allclose(Cb, C, atol=1e-12) and np.allclose(Kb, K)
    return dict(sh=sh, V=V, F=F, P0=P0, N0=N0, A0=A0, P=P, N=N, A=A, inv=inv, n_vox=n_vox, Aw0=Aw0, keep=keep,
                near=near, moved=moved, Pv=Pv, Pm=Pm, Ps=Ps, Nm=Nm, Aw=Aw, C0=C0_, C=C, cV=cV, h=h, K=K, s=s, rs=rs,
                E=E, n_rays=n_rays)


def cmd_steps():
    from scipy.spatial import cKDTree
    S = build_steps()
    sh, C, Nm, Aw, K, h, Ps, rs = S["sh"], S["C"], S["Nm"], S["Aw"], S["K"], S["h"], S["Ps"], S["rs"]
    nn = cKDTree(Ps).query(Ps, k=2)[0][:, 1]
    # sphere tops against the surface: the outermost point C + rs n, radially against the exact surface
    top = C + rs * Nm
    with sh._origin():
        Dt = top - sh.c0
        r_top = np.linalg.norm(Dt, axis=1)
        r_srf = np.linalg.norm(sh._bisect(Dt / r_top[:, None], sh.inside_design) - sh.c0, axis=1)
    dev = r_top - r_srf
    # scallops: how far the exact surface lies outside the union of the spheres, on the dome (n_x > 0.5)
    dome = S["N0"][:, 0] > 0.5
    Pd = S["P0"][dome]
    dist, _ = cKDTree(C).query(Pd, k=8)
    gap = dist.min(axis=1) - rs
    dome_pad = Nm[:, 0] > 0.5
    clamp = np.linalg.norm(S["C"] - S["C0"], axis=1)
    proj = np.linalg.norm(S["Pm"] - S["Ps"], axis=1)
    sliver_area = S["Aw0"][~S["keep"]]
    nnc = cKDTree(C).query(C, k=2)[0][:, 1]
    big = clamp > 1e-4
    out = {
        "key": KEY, "s_mm": S["s"] * 1e3, "rs_mm": rs * 1e3, "E_MPa": S["E"] / 1e6,
        "semi_axes_mm": {"across": sh.ay * 1e3, "depth": sh.ax * 1e3, "distal": sh.az * 1e3, "proximal": sh.azp * 1e3},
        "n_rays": S["n_rays"], "hull_vertices": int(len(S["V"])), "hull_faces": int(len(S["F"])),
        "dense_n": int(len(S["P0"])), "dense_area_mm2": float(S["A0"].sum() * 1e6),
        "piece_area_mm2_median": float(np.median(S["A0"]) * 1e6),
        "masked_n": int(len(S["P"])), "masked_area_mm2": float(S["A"].sum() * 1e6),
        "voxels": int(S["n_vox"]), "slivers": int((~S["keep"]).sum()),
        "sliver_area_mm2_max": float(sliver_area.max() * 1e6) if len(sliver_area) else 0.0,
        "pads": int(len(C)), "samples_moved_frac": float(S["moved"].mean()),
        "proj_um_median": float(np.median(proj) * 1e6), "proj_um_max": float(proj.max() * 1e6),
        "clamp_moved": int((clamp > 1e-9).sum()), "clamp_um_max": float(clamp.max() * 1e6),
        "clamp_um_median": float(np.median(clamp) * 1e6), "clamp_over_10um": int((clamp > 1e-5).sum()),
        "clamp_over_100um": int(big.sum()),
        "clamp_over_100um_where": {"x_mm": [float(S["C"][big, 0].min() * 1e3), float(S["C"][big, 0].max() * 1e3)],
                                   "z_mm": [float(S["C"][big, 2].min() * 1e3), float(S["C"][big, 2].max() * 1e3)]}
        if big.any() else None,
        "clamp_over_100um_at_back_or_mount": int((big & ((np.abs(S["C"][:, 0] - (sh.reach - 0.017 + rs)) < 1e-9)
                                                          | (np.abs(S["C"][:, 2] - (sh.top_z - rs)) < 1e-9)
                                                          | (np.abs(S["C"][:, 2] - (-0.01055 + rs)) < 1e-9))).sum()),
        "nn_centres_mm": {"median": float(np.median(nnc) * 1e3), "mean": float(nnc.mean() * 1e3)},
        "nn_mean_mm": float(nn.mean() * 1e3),
        "nn_mm": {"median": float(np.median(nn) * 1e3), "p5": q(nn, 5) * 1e3, "p95": q(nn, 95) * 1e3,
                  "min": float(nn.min() * 1e3), "max": float(nn.max() * 1e3)},
        "area_mm2": {"median": float(np.median(Aw) * 1e6), "p5": q(Aw, 5) * 1e6, "p95": q(Aw, 95) * 1e6,
                     "min": float(Aw.min() * 1e6), "max": float(Aw.max() * 1e6), "sum": float(Aw.sum() * 1e6)},
        "h_mm": {"median": float(np.median(h) * 1e3), "min": float(h.min() * 1e3), "max": float(h.max() * 1e3),
                 "apex": float(h[np.argmax(Ps[:, 0])] * 1e3)},
        "cV_mm": [float(v * 1e3) for v in S["cV"]],
        "K_Npm": {"median": float(np.median(K)), "min": float(K.min()), "max": float(K.max()),
                  "sum_N_per_mm": float(K.sum() * 1e-3), "dome_sum_N_per_mm": float(K[dome_pad].sum() * 1e-3)},
        "dome_pads": int(dome_pad.sum()),
        "top_dev_um": {"dome_absmax": float(np.abs(dev[dome_pad]).max() * 1e6), "absmax": float(np.abs(dev).max() * 1e6),
                       "dome_p99": q(np.abs(dev[dome_pad]), 99) * 1e6},
        "scallop_mm": {"median": float(np.median(gap) * 1e3), "p95": q(gap, 95) * 1e3, "max": float(gap.max() * 1e3),
                       "min": float(gap.min() * 1e3)},
        "x_top_mm": float((C[:, 0] + rs).max() * 1e3), "x_apex_mm": float(sh.reach * 1e3),
    }
    fig_section(S)
    out["y_row_mm"] = S["y_row"] * 1e3
    save("steps", out)
    np.savez_compressed(ROWS / "20261111-pads.npz", C=C, N=Nm, A=Aw, K=K, h=h, P=Ps, cV=S["cV"])
    print(json.dumps(out, indent=1))
    fig_hist(S, nn)
    fig_render(S)


# ---------------------------------------------------------------------------------------------------- figures
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 13,
                         "axes.titlesize": 14, "axes.labelsize": 14, "legend.fontsize": 12, "xtick.labelsize": 12,
                         "ytick.labelsize": 12, "mathtext.fontset": "dejavuserif"})
    return plt


def _colour_graph(lab, n, k=7):
    """Greedy colouring of the labels of a raster so that neighbouring cells of different labels differ."""
    adj = [set() for _ in range(n)]
    for a, b in ((lab[:, 1:], lab[:, :-1]), (lab[1:, :], lab[:-1, :])):
        m = (a != b) & (a >= 0) & (b >= 0)
        for u, v in zip(a[m].ravel(), b[m].ravel()):
            adj[u].add(v)
            adj[v].add(u)
    col = np.full(n, -1)
    for u in np.argsort([-len(x) for x in adj]):
        used = {col[v] for v in adj[u]}
        col[u] = next(c for c in range(k + 20) if c not in used)
    return col


def fig_section(S):
    """Figure 1: (a) the dome seen from the palm, each pad's share of the dense samples in its own colour, with the
    1 mm voxel grid and the surface's height contours at the voxel layers' boundaries; (b) the section y = 0 at the
    distal end with the exact outline and each sphere's own section in that plane."""
    plt = _mpl()
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Circle
    from scipy.spatial import cKDTree
    sh, s, rs = S["sh"], S["s"], S["rs"]
    P, N, near = S["P"], S["N"], S["near"]
    vis = N[:, 0] > 0.0
    tree = cKDTree(P[vis][:, 1:])
    ys = np.arange(-7.6, 7.6, 0.02) * 1e-3
    zs = np.arange(-10.8, 11.1, 0.02) * 1e-3
    Y, Zg = np.meshgrid(ys, zs)
    dd, ii = tree.query(np.column_stack([Y.ravel(), Zg.ravel()]))
    lab = np.where(dd < 6e-5, near[vis][ii], -1).reshape(Y.shape)
    X = np.where(dd < 6e-5, P[vis][ii, 0], np.nan).reshape(Y.shape)
    col = _colour_graph(lab, len(S["C"]))
    pal = ["#8fb8de", "#f2b880", "#a8d5a2", "#e8a0a8", "#c3b1e1", "#f6e27f", "#9fd8d8", "#d9c3a5", "#b5c99a"]
    img = np.where(lab >= 0, col[np.maximum(lab, 0)], len(pal))
    cmap = ListedColormap(pal[:col.max() + 1] + ["#ffffff"] * (len(pal) - col.max()))
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 5.9), gridspec_kw={"width_ratios": [1.0, 1.0]})
    ax = axs[0]
    ax.imshow(img, origin="lower", extent=(ys[0] * 1e3, ys[-1] * 1e3, zs[0] * 1e3, zs[-1] * 1e3), cmap=cmap,
              vmin=0, vmax=len(pal), interpolation="nearest")
    cs = ax.contour(Y * 1e3, Zg * 1e3, X * 1e3, levels=np.arange(5, 11), colors="k", linewidths=0.9,
                    linestyles="dashed")
    ax.clabel(cs, fmt="%d", fontsize=11, inline_spacing=1)
    Ps, Nm = S["Ps"], S["Nm"]
    pv = Nm[:, 0] > 0.0
    ax.plot(Ps[pv, 1] * 1e3, Ps[pv, 2] * 1e3, "k.", ms=2.2)
    ax.set_xlim(-7.6, 7.6)
    ax.set_ylim(-10.8, 11.1)
    ax.set_aspect("equal")
    ax.set_xlabel("y, across the finger (mm)")
    ax.set_ylabel("z, along the finger (mm)")
    ax.set_title("(a) Pads of the dome, palmar view")
    ax = axs[1]
    C = S["C"]
    row = (C[:, 1] > 0) & (C[:, 1] < s) & (C[:, 2] < -0.004) & (C[:, 0] > 0.0035)
    y_row = float(np.median(C[row, 1]))                  # the plane through one row of centres
    S["y_row"] = y_row
    xs = np.linspace(0.0035, 0.0112, 900)
    zz = np.linspace(-0.0112, -0.0040, 840)
    Xs, Zs = np.meshgrid(xs, zz)
    ins = sh.inside_design(np.column_stack([Xs.ravel(), np.full(Xs.size, y_row), Zs.ravel()])).reshape(Xs.shape)
    for g in range(4, 12):
        ax.axvline(g, color="0.8", lw=0.6, zorder=0)
    for g in range(-11, -3):
        ax.axhline(g, color="0.8", lw=0.6, zorder=0)
    ax.contourf(Xs * 1e3, Zs * 1e3, ins.astype(float), levels=[0.5, 1.5], colors=["#f3e6d8"], zorder=0.5)
    ax.contour(Xs * 1e3, Zs * 1e3, ins.astype(float), levels=[0.5], colors="k", linewidths=1.3, zorder=2)
    sec = np.abs(C[:, 1] - y_row) < rs
    for c in C[sec]:
        rr = np.sqrt(rs ** 2 - (c[1] - y_row) ** 2)
        ax.add_patch(Circle((c[0] * 1e3, c[2] * 1e3), rr * 1e3, fc="#b5643f", ec="#5a2a14", lw=0.8, alpha=0.85,
                            zorder=3))
    ax.set_xlim(3.5, 11.2)
    ax.set_ylim(-11.2, -4.0)
    ax.set_aspect("equal")
    ax.set_xlabel("x, along the palmar normal (mm)")
    ax.set_title(f"(b) Section y = {y_row * 1e3:.2f} mm")
    fig.tight_layout()
    out = MEDIA / "20261111-voxel_pads.png"
    MEDIA.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print("wrote", out.relative_to(ROOT), flush=True)


def fig_hist(S, nn):
    """Figure 3: the distributions of nearest-neighbour spacing, area share, depth and stiffness over the 1,018 pads."""
    plt = _mpl()
    fig, axs = plt.subplots(2, 2, figsize=(8.6, 5.6))
    data = [(nn * 1e3, "Spacing of surface points (mm)"), (S["Aw"] * 1e6, "Area share $a_i$ (mm$^2$)"),
            (S["h"] * 1e3, "Depth $h_i$ (mm)"), (S["K"], "Stiffness $K_i$ at 2.5 MPa (N/m)")]
    dome = S["Nm"][:, 0] > 0.5
    for ax, (x, lab) in zip(axs.ravel(), data):
        bins = np.linspace(x.min(), x.max(), 41)
        ax.hist(x[~dome], bins=bins, color="0.65", label="sides and end")
        ax.hist(x[dome], bins=bins, color="#1b6ca8", alpha=0.85, label="dome, $n_x$ > 0.5")
        ax.set_xlabel(lab)
        ax.set_ylabel("spheres")
    axs[0, 0].legend(frameon=False)
    fig.tight_layout()
    out = MEDIA / "20261111-pad_distributions.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print("wrote", out.relative_to(ROOT), flush=True)


def tip_xml(C, rgba, solimp=None, tc=None, plate=False):
    """A fixed tip body with the pads (and optionally the plate on a slide joint along x)."""
    pads = []
    for i, c in enumerate(C):
        a = (f'<geom name="pad{i}" type="sphere" size="{RS_M}" pos="{c[0]:.7f} {c[1]:.7f} {c[2]:.7f}" condim="3" '
             f'priority="1" friction="1 0 0" mass="0" rgba="{" ".join(f"{v:.3f}" for v in rgba[i])}"')
        if solimp is not None:
            a += f' solref="{tc:.6g} 1" solimp="{solimp[i]:.6g} {solimp[i]:.6g} 0.001 0.5 2"'
        pads.append(a + "/>")
    plate_xml = ""
    if plate:
        plate_xml = f"""<body name="plate" pos="{X_TOP + 0.00105:.6f} 0 0">
      <joint name="slide" type="slide" axis="1 0 0"/>
      <geom name="plate" type="box" size="0.001 0.016 0.016" mass="{M_PLATE}" friction="1 0 0" material="glass"
            rgba="0.55 0.68 0.85 0.28"/>
    </body>"""
    return f"""<mujoco model="v4b_pads">
  <compiler angle="radian"/>
  <option timestep="0.001" gravity="0 0 0" integrator="implicitfast" cone="elliptic" solver="Newton"
          iterations="100" ls_iterations="50" tolerance="1e-10" impratio="100"/>
  <visual><global offwidth="1600" offheight="1200"/><quality shadowsize="4096"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.55 0.55 0.55" specular="0.1 0.1 0.1"/></visual>
  <asset><texture name="sky" type="skybox" builtin="flat" rgb1="1 1 1" rgb2="1 1 1" width="16" height="16"/>
    <material name="glass" specular="0" shininess="0" reflectance="0"/></asset>
  <worldbody>
    <light pos="0.08 0.05 0.10" dir="-0.6 -0.35 -0.7" diffuse="0.5 0.5 0.5"/>
    <body name="tip">
      {chr(10).join(pads)}
    </body>
    {plate_xml}
  </worldbody>
</mujoco>"""


RS_M = 0.00075
X_TOP = None


def fig_render(S):
    """Figure 2: the 1,018 pads of one v4b tip coloured by stiffness, from the palm and from the side (MuJoCo)."""
    import matplotlib
    import mujoco
    plt = _mpl()
    K = S["K"]
    cmap = matplotlib.colormaps["viridis"]
    lo, hi = float(K.min()), float(K.max())
    rgba = [tuple(cmap((k - lo) / (hi - lo))[:3]) + (1.0,) for k in K]
    m = mujoco.MjModel.from_xml_string(tip_xml(S["C"], rgba))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    imgs = []
    with mujoco.Renderer(m, 900, 760) as r:
        for az, el, dist in ((180.0, 0.0, 0.040), (125.0, -12.0, 0.042)):
            cam = mujoco.MjvCamera()
            cam.lookat[:] = (0.0015, 0.0, 0.0)
            cam.distance, cam.azimuth, cam.elevation = dist, az, el
            r.update_scene(d, camera=cam)
            imgs.append(r.render().copy())
    fig = plt.figure(figsize=(8.6, 4.4))
    ax1 = fig.add_axes([0.0, 0.05, 0.43, 0.88])
    ax2 = fig.add_axes([0.43, 0.05, 0.43, 0.88])
    for ax, im, t in ((ax1, imgs[0], "Palmar view"), (ax2, imgs[1], "Side view")):
        ax.imshow(im)
        ax.set_title(t)
        ax.axis("off")
    cax = fig.add_axes([0.885, 0.12, 0.025, 0.72])
    sm = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(lo, hi), cmap=cmap)
    fig.colorbar(sm, cax=cax, label="$K_i$ (N/m)")
    out = MEDIA / "20261111-pad_stiffness_render.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print("wrote", out.relative_to(ROOT), flush=True)


# ---------------------------------------------------------------------------------------------------- scene
def cmd_scene():
    """The talk scene's pads against the builder: positions (mirrored per finger), d0, and the stiffness that d0 gives
    with the compiled inverse weights."""
    import mujoco
    import reorient_backends as RB
    Z = np.load(ROWS / "20261111-pads.npz")
    C, K = Z["C"], Z["K"]
    meta = json.loads(SCENE.with_suffix(".json").read_text())
    root = ET.parse(SCENE).getroot()
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    tool = RB.OBJ
    out = {"scene": str(SCENE.relative_to(ROOT)), "meta": {k: meta[k] for k in ("pad_E_Pa", "pad_tr_s", "tip_key",
                                                                                  "n_pads", "pad_builder", "impratio")}}
    for f, sg in SIGN.items():
        g = sorted((e for e in root.iter("geom") if e.get("name", "").startswith(f"{f}_pad")),
                   key=lambda e: int(e.get("name")[len(f) + 4:]))
        pos = np.array([[float(v) for v in e.get("pos").split()] for e in g])
        d0 = np.array([float(e.get("solimp").split()[0]) for e in g])
        tc = float(g[0].get("solref").split()[0])
        diag = m.body_invweight0[m.body(f"{f}_tip").id, 0] + m.body_invweight0[m.body(tool).id, 0]
        Kd = 1.0 / ((1.0 - d0) * tc ** 2 * diag)
        flip = np.array([sg, 1.0, 1.0])
        out[f] = {"n": len(g), "pos_err_um": float(np.abs(pos - C * flip).max() * 1e6), "tc_s": tc,
                  "diag": float(diag), "inv_tip": float(m.body_invweight0[m.body(f"{f}_tip").id, 0]),
                  "inv_tool": float(m.body_invweight0[m.body(tool).id, 0]),
                  "d0_min": float(d0.min()), "d0_median": float(np.median(d0)), "d0_max": float(d0.max()),
                  "K_rel_err_max": float(np.abs(Kd / K - 1).max())}
    save("scene", out)
    print(json.dumps(out, indent=1))


# ---------------------------------------------------------------------------------------------------- press
def press_model(C, K, rgba=None):
    """The fixed tip and the plate; d0 per pad from this model's inverse weights, as replace_tips sets it."""
    import mujoco
    import reorient_backends as RB
    global X_TOP
    X_TOP = float((C[:, 0] + RS_M).max())
    rgba = rgba or [PAD_RGBA] * len(C)
    tc = RB.PAD_TR / 2.0
    m0 = mujoco.MjModel.from_xml_string(tip_xml(C, rgba, plate=True))
    diag = m0.body_invweight0[m0.body("tip").id, 0] + m0.body_invweight0[m0.body("plate").id, 0]
    d0 = 1.0 - 1.0 / (tc ** 2 * K * diag)
    assert d0.min() > 0.05
    m = mujoco.MjModel.from_xml_string(tip_xml(C, rgba, solimp=d0, tc=tc, plate=True))
    return m, {"tc": tc, "diag": float(diag), "d0_min": float(d0.min()), "d0_max": float(d0.max())}


def contacts(m, d, npad):
    """(pad index, penetration m, normal force N) of every pad contact."""
    import mujoco
    f6 = np.zeros(6)
    rows = []
    gp = m.geom("plate").id
    for i in range(d.ncon):
        c = d.contact[i]
        g = c.geom2 if c.geom1 == gp else c.geom1
        if gp not in (c.geom1, c.geom2) or g >= npad:
            continue
        mujoco.mj_contactForce(m, d, i, f6)
        rows.append((int(g), float(-c.dist), float(f6[0])))
    return rows


def cmd_press():
    import mujoco
    import tip_codesign_geometry as TG
    Z = np.load(ROWS / "20261111-pads.npz")
    C, K = Z["C"], Z["K"]
    st = json.loads((ROWS / "20261111-steps.json").read_text())
    m, info = press_model(C, K)
    d = mujoco.MjData(m)
    bid = m.body("plate").id
    x_apex = st["x_apex_mm"] * 1e-3
    res = []
    for F in FORCES:
        mujoco.mj_resetData(m, d)
        d.xfrc_applied[bid, 0] = -F
        for _ in range(2000):
            mujoco.mj_step(m, d)
        rows = contacts(m, d, len(C))
        x_face = d.xpos[bid, 0] - 0.001
        idx = np.array([r[0] for r in rows])
        pen = np.array([r[1] for r in rows])
        fn = np.array([r[2] for r in rows])
        res.append({"F": F, "delta_um": float((x_apex - x_face) * 1e6), "sum_fn": float(fn.sum()),
                    "n_touch": int((fn > 0).sum()), "qvel": float(abs(d.qvel[0])),
                    "ratio_med": float(np.median(fn / (K[idx] * pen))),
                    "ratio_maxdev": float(np.abs(fn / (K[idx] * pen) - 1).max()),
                    "idx": idx.tolist(), "pen_um": (pen * 1e6).round(4).tolist(), "fn_mN": (fn * 1e3).round(5).tolist()})
        print(F, {k: v for k, v in res[-1].items() if k not in ("idx", "pen_um", "fn_mN")}, flush=True)
    # the same d0 against a plate four times as heavy: the inverse weight in d0 is the 25 g plate's, so each sphere is
    # stiffer by the ratio of the two inverse weights
    m.body_mass[bid] = 4 * M_PLATE
    mujoco.mj_setConst(m, d)
    mujoco.mj_resetData(m, d)
    d.xfrc_applied[bid, 0] = -1.0
    for _ in range(3000):
        mujoco.mj_step(m, d)
    rows = contacts(m, d, len(C))
    idx = np.array([r[0] for r in rows])
    pen = np.array([r[1] for r in rows])
    fn = np.array([r[2] for r in rows])
    heavy = {"m_plate": 4 * M_PLATE, "F": 1.0, "inv_plate": float(m.body_invweight0[bid, 0]),
             "ratio_med": float(np.median(fn / (K[idx] * pen))), "n_touch": int((fn > 0).sum()),
             "delta_um": float((x_apex - (d.xpos[bid, 0] - 0.001)) * 1e6)}
    print("heavy plate", heavy, flush=True)
    # the pads' springs and the continuous law along the approach
    sh = TG.shape(KEY)
    P0, N0, A0 = sh.dense
    face = N0[:, 0] > 0.0                               # the surface that faces the plate
    P, A = P0[face], A0[face]
    hs = np.maximum(np.einsum("ij,ij->i", N0[face], P - Z["cV"]), 1e-3)
    N = N0[face][:, 0]                                  # n_x: the projected area is a n_x
    E = st["E_MPa"] * 1e6
    deltas = np.linspace(0, 0.65e-3, 651)
    top = C[:, 0] + RS_M
    F_pad = [float((K * np.maximum(0, top - (x_apex - dl))).sum()) for dl in deltas]
    F_law = [float((E / hs * A * N * np.maximum(0, P[:, 0] - (x_apex - dl))).sum()) for dl in deltas]
    F_law_s = [float((E / hs * A * np.maximum(0, P[:, 0] - (x_apex - dl))).sum()) for dl in deltas]
    comp = []
    for r in res:
        dl = r["delta_um"] * 1e-6
        fp = float((K * np.maximum(0, top - (x_apex - dl))).sum())
        fl = float((E / hs * A * N * np.maximum(0, P[:, 0] - (x_apex - dl))).sum())
        comp.append({"F": r["F"], "delta_um": r["delta_um"], "F_pads_eq10": fp, "F_law": fl,
                     "delta_law_um": float(np.interp(r["F"], F_law, deltas) * 1e6)})
    out = {"model": info, "x_top_mm": X_TOP * 1e3, "m_plate": M_PLATE, "presses": res, "comp": comp, "heavy": heavy,
           "curve": {"delta_um": (deltas * 1e6).round(3).tolist(), "F_pads": F_pad, "F_law": F_law,
                     "F_law_surface_area": F_law_s}}
    save("press", out)
    fig_press(out, K)


def fig_press(out, K):
    """Figure 4: force against approach (law, the pads' springs, MuJoCo at rest) and each touching sphere's force against
    its spring K_i times its penetration."""
    plt = _mpl()
    cv = out["curve"]
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.9))
    ax = axs[0]
    ax.plot(cv["delta_um"], cv["F_law"], color="k", lw=1.6, label="continuous law")
    ax.plot(cv["delta_um"], cv["F_pads"], color="#1b6ca8", lw=1.6, label="pads, Eq. (10)")
    ax.plot([r["delta_um"] for r in out["presses"]], [r["F"] for r in out["presses"]], "o", color="#d1495b", ms=6,
            label="MuJoCo at rest")
    ax.set_xlabel("Approach (µm)")
    ax.set_ylabel("Plate force (N)")
    ax.set_xlim(0, 650)
    ax.set_ylim(0, 4.6)
    ax.legend(frameon=False, loc="upper left")
    ax = axs[1]
    cols = plt.cm.plasma(np.linspace(0.1, 0.85, len(out["presses"])))
    for r, c in zip(out["presses"], cols):
        idx = np.array(r["idx"])
        pen = np.array(r["pen_um"]) * 1e-6
        ax.loglog(K[idx] * pen * 1e3, np.array(r["fn_mN"]), "o", ms=3.5, color=c, label=f"{r['F']:g} N")
    lim = [0.5, 300.0]
    ax.plot(lim, lim, color="k", lw=0.8)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("$K_i$ times penetration (mN)")
    ax.set_ylabel("Sphere force (mN)")
    ax.legend(frameon=False, title="plate force", loc="upper left")
    fig.tight_layout()
    p = MEDIA / "20261111-flat_press.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    print("wrote", p.relative_to(ROOT), flush=True)


# ---------------------------------------------------------------------------------------------------- film
def cmd_film():
    """Film 1: the plate pressed on the pads from 0 to 4 N in 3 s and held for 1 s. Left, the tip seen through the plate,
    each touching sphere coloured by its force; right, the plate force against the approach, with the pads' springs
    and the continuous law of Figure 4."""
    import imageio
    import matplotlib
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    plt = _mpl()
    plt.rcParams.update({"font.size": 18, "axes.titlesize": 18, "axes.labelsize": 19, "legend.fontsize": 17,
                         "xtick.labelsize": 17, "ytick.labelsize": 17})    # >= 13 px when the film is shown 860 px wide
    Z = np.load(ROWS / "20261111-pads.npz")
    C, K = Z["C"], Z["K"]
    m, _ = press_model(C, K)
    m.geom_rgba[m.geom("plate").id] = (0.85, 0.88, 0.92, 0.18)
    d = mujoco.MjData(m)
    d.qpos[0] = -0.00005                                # the plate's face starts on the outermost sphere, at rest
    mujoco.mj_forward(m, d)
    bid = m.body("plate").id
    npad = len(C)
    pr = json.loads((ROWS / "20261111-press.json").read_text())
    x_apex = json.loads((ROWS / "20261111-steps.json").read_text())["x_apex_mm"] * 1e-3
    f_hi = max(max(r["fn_mN"]) for r in pr["presses"])          # mN, the colour scale's top: the largest at 4 N
    cmap = matplotlib.colormaps["inferno"]
    lo_c = 0.12
    T_RAMP, T_HOLD, FPS, F_MAX = 3.0, 1.0, 25, 4.0
    W = 640
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", 24)
    cv = pr["curve"]
    fig = plt.figure(figsize=(6.4, 6.4), dpi=100)
    ax = fig.add_axes([0.16, 0.13, 0.58, 0.77])
    ax.plot(cv["delta_um"], cv["F_law"], color="k", lw=1.6, label="continuous law")
    ax.plot(cv["delta_um"], cv["F_pads"], color="#1b6ca8", lw=1.6, label="pads, Eq. (10)")
    trail, = ax.plot([], [], color="#d1495b", lw=2.4, label="MuJoCo")
    dot, = ax.plot([], [], "o", color="#d1495b", ms=9)
    ax.set_xlim(0, 650)
    ax.set_ylim(0, 4.6)
    ax.set_xlabel("Approach (\u00b5m)")
    ax.set_ylabel("Plate force (N)")
    ax.legend(frameon=False, loc="upper left")
    title = ax.set_title(" ")
    cax = fig.add_axes([0.80, 0.13, 0.035, 0.77])
    sm = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0, f_hi),
                                      cmap=matplotlib.colors.ListedColormap(cmap(np.linspace(lo_c, 1, 256))))
    fig.colorbar(sm, cax=cax, label="Sphere force (mN)")
    base = np.array(PAD_RGBA)
    steps_per = int(round(1.0 / FPS / m.opt.timestep))
    n_frames = int((T_RAMP + T_HOLD) * FPS) + 1
    tr_x, tr_y, frames = [], [], []
    with mujoco.Renderer(m, W, W) as r:
        for k in range(n_frames):
            if k:
                for _ in range(steps_per):
                    d.xfrc_applied[bid, 0] = -F_MAX * min(d.time / T_RAMP, 1.0)
                    mujoco.mj_step(m, d)
            F = F_MAX * min(d.time / T_RAMP, 1.0)
            rows = contacts(m, d, npad)
            m.geom_rgba[:npad] = base
            touch = 0
            for g, _, fn in rows:
                if fn > 0:
                    touch += 1
                    m.geom_rgba[g] = (*cmap(lo_c + (1 - lo_c) * min(fn * 1e3 / f_hi, 1.0))[:3], 1.0)
            cam = mujoco.MjvCamera()
            cam.lookat[:] = (0.004, 0.0, 0.0005)
            cam.distance, cam.azimuth, cam.elevation = 0.024, 180.0, 0.0
            r.update_scene(d, camera=cam)
            left = r.render().copy()
            dl = (x_apex - (d.xpos[bid, 0] - 0.001)) * 1e6
            tr_x.append(max(dl, 0.0))
            tr_y.append(F)
            trail.set_data(tr_x, tr_y)
            dot.set_data([tr_x[-1]], [F])
            title.set_text(f"F = {F:4.2f} N, {touch} spheres touch")
            fig.canvas.draw()
            right = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
            bar = Image.new("RGB", (2 * W, 48), "white")
            dr = ImageDraw.Draw(bar)
            dr.text((12, 10), "The v4b fingertip through the plate", fill="black", font=font)
            dr.text((W + 12, 10), "Plate force against approach", fill="black", font=font)
            out = Image.new("RGB", (2 * W, W + 48), "white")
            out.paste(bar, (0, 0))
            out.paste(Image.fromarray(left), (0, 48))
            out.paste(Image.fromarray(right), (W, 48))
            frames.append(np.asarray(out))
    plt.close(fig)
    p = MEDIA / "20261111-flat_press.mp4"
    imageio.mimwrite(str(p), frames, fps=FPS, codec="libx264", quality=7, macro_block_size=8)
    Image.fromarray(frames[-1]).save(MEDIA / "20261111-flat_press_poster.jpg", quality=88)
    print("wrote", p.relative_to(ROOT), f"{p.stat().st_size / 1e6:.2f} MB", flush=True)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "steps"
    if cmd in ("steps", "film"):
        for k, v in SOFT_GL.items():
            os.environ.setdefault(k, v)
    {"steps": cmd_steps, "scene": cmd_scene, "press": cmd_press, "film": cmd_film}[cmd]()


if __name__ == "__main__":
    main()
