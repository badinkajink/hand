#!/usr/bin/env python3
"""A 3-D film of the GCF twist and HOM's least squares on one SR2 finger (owner, 2026-11-10: "videos that really explain
well the physical, spatial intuition of the GCF twist and hom control"; the SR2 hand "is really bulky and im afraid that
using it will be visually cluttered"). The paper's Exp. A (hom_tasks Schedule.expa): the thumb moves round the cylinder
without contact while each of the six components of its relative twist is commanded + then - (40 deg/s or 20 mm/s),
the others at zero. The index and the middle are hidden.

Each frame: the thumb, the cylinder, the thumb's GCF at its witness point (x1 the normal, red; y1 green; z1 blue) and the
gap to the cylinder's witness point; the commanded relative twist (purple) and the twist Eq. (2) plans (orange), linear
parts as arrows and angular parts as arrows along their axes; on the right, the six components and HOM's law. The
controller's per-period info reaches the film through the plant's set_hom hook of hom_tasks.rollout, which a position
plant otherwise lacks; the plant's physics is unchanged.

    HOS_PLANT=servo HOS_SCENES=logs/20261102-hom_servo_plant/scenes HOM_ROWS_TAG=_sv25 MUJOCO_GL=egl \\
        .venv/bin/python scripts/hom/hom_gcf_film.py [test]
"""
from __future__ import annotations

import io
import math
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path
import hom_tasks as T  # noqa: E402
from hom_films import arrow  # noqa: E402
from sr2hand.control import hom as H  # noqa: E402

MEDIA = ROOT / "docs/experiments/20261029-hom_control_review/media"
EQ = ROOT / "docs/experiments/20261110-codesign_formulation/media/20261110-eq_inner5.png"
MOVER = T.EXPA_MOVER
CMD, PLAN = (0.49, 0.23, 0.93, 0.95), (0.92, 0.35, 0.05, 0.95)
AX = [(0.86, 0.15, 0.15, 1.0), (0.10, 0.60, 0.20, 1.0), (0.15, 0.39, 0.92, 1.0)]
NAMES = {"spin": "spin, about x1", "roll_y": "roll, about y1", "roll_z": "roll, about z1", "sep": "separation, along x1",
         "slide_y": "slide, along y1", "slide_z": "slide, along z1"}
plt.rcParams.update({"font.family": "Lato", "font.size": 16, "mathtext.fontset": "cm"})


class GcfFilm(T.Film):
    def __init__(self, plant, setup, w=1120, h=840):
        plant.m.vis.global_.offwidth = max(plant.m.vis.global_.offwidth, w)
        plant.m.vis.global_.offheight = max(plant.m.vis.global_.offheight, h)
        super().__init__(plant, setup, w=w, h=h)
        import mujoco
        m = plant.m
        for g in range(m.ngeom):                    # hide the index and the middle: a geom group the scene skips
            b = m.body(m.geom_bodyid[g]).name or ""
            if b.startswith(("index", "middle")):
                m.geom_group[g] = 5
            elif m.geom_bodyid[g] == plant.tb:          # the cylinder see-through, so the contact frame shows
                m.geom_rgba[g, 3] = 0.35
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[5] = 0
        self.r.close()                              # 1,018 pad spheres per tip: room for the overlays
        self.r = mujoco.Renderer(m, h, w, max_geom=30000)
        self.eq = np.asarray(Image.open(EQ).convert("RGB"))
        self.frames_ci = []

    def frame(self, plant, title, line):
        ci = getattr(plant, "_ci", None)
        if ci is not None:                          # the camera follows the contact frame
            p = np.asarray(ci[MOVER]["gcf"].p1, float)
            self.look = p if getattr(self, "look", None) is None else 0.85 * self.look + 0.15 * p
            self.cam.lookat[:] = self.look
        self.r.update_scene(plant.d, self.cam, scene_option=self.opt)
        if ci is not None:
            c = ci[MOVER]
            g = c["gcf"]
            p1, p0, R1 = np.asarray(g.p1), np.asarray(g.p0), np.asarray(g.R1)
            for k in range(3):
                arrow(self.r.scene, p1, p1 + 0.024 * R1[:, k], AX[k], width=0.0014)
            arrow(self.r.scene, p1, p0, (0.4, 0.4, 0.4, 0.8), width=0.0005, kind="capsule")
            for v, rgba in ((np.asarray(c["v_ref"], float), CMD), (np.asarray(c["v_plan"], float), PLAN)):
                lin = R1 @ v[3:]                    # m/s in the world
                ang = R1 @ v[:3]                    # rad/s in the world
                if np.linalg.norm(lin) > 2e-4:
                    arrow(self.r.scene, p1, p1 + 1.3 * lin, rgba, width=0.0022)
                if np.linalg.norm(ang) > math.radians(1.0):
                    arrow(self.r.scene, p1, p1 + 0.04 * ang, rgba[:3] + (0.6,), width=0.0034)
        img = self.r.render().copy()
        return compose(img, ci, title, line, self.eq)


def compose(img, ci, title, line, eq):
    fig = plt.figure(figsize=(16, 9), dpi=120)
    ax = fig.add_axes([0.0, 0.0, 0.62, 0.88])
    ax.imshow(img)
    ax.axis("off")
    tok = line.split()
    comp = tok[0] if tok and tok[0] in NAMES else None
    sign = tok[1] if comp and len(tok) > 1 and tok[1] in ("+", "-") else ""
    fig.text(0.02, 0.93, "HOM on one finger: the six components of the relative twist", fontsize=26, weight="bold",
             color="#1f2937")
    fig.text(0.02, 0.89, (f"now: {NAMES[comp]}, commanded {sign}" if comp else "now: rest")
             + ", every other component at zero", fontsize=17, color="#4b5563")
    e = fig.add_axes([0.63, 0.74, 0.36, 0.12])
    e.imshow(eq)
    e.axis("off")
    fig.text(0.635, 0.70, "least squares every 10 ms: joint speeds $u$ that best make $v_{ref}$ happen", fontsize=13,
             color="#4b5563")
    if ci is not None:
        c = ci[MOVER]
        vr, vp = np.asarray(c["v_ref"], float), np.asarray(c["v_plan"], float)
        for j, k in enumerate(H.COMPONENTS):
            f, unit, lim = (180 / math.pi, "deg/s", 50.0) if j < 3 else (1000.0, "mm/s", 25.0)
            b = fig.add_axes([0.73, 0.60 - 0.085 * j, 0.25, 0.05])
            b.barh([0], [vp[j] * f], color="#ea580c", height=0.7)
            b.axvline(vr[j] * f, color="#7c3aed", lw=3, ls="--")
            b.axvline(0, color="#1f2937", lw=1)
            b.set_xlim(-lim, lim)
            b.set_yticks([])
            b.tick_params(labelsize=10)
            for sp in ("top", "right", "left"):
                b.spines[sp].set_visible(False)
            hl = k == comp
            fig.text(0.635, 0.615 - 0.085 * j, k.replace("_", " "), fontsize=15, weight="bold" if hl else "normal",
                     color="#7c3aed" if hl else "#1f2937")
            fig.text(0.635, 0.592 - 0.085 * j, f"{vp[j] * f:+.1f} {unit}", fontsize=12, color="#ea580c")
    lg = fig.add_axes([0.635, 0.04, 0.35, 0.05])
    lg.axis("off")
    lg.plot([0.0, 0.07], [0.5, 0.5], color="#7c3aed", lw=3, ls="--")
    lg.text(0.09, 0.5, "commanded $v_{ref}$", va="center", fontsize=14, color="#7c3aed")
    lg.plot([0.52, 0.59], [0.5, 0.5], color="#ea580c", lw=4)
    lg.text(0.61, 0.5, "least-squares plan", va="center", fontsize=14, color="#ea580c")
    lg.set_xlim(0, 1)
    lg.set_ylim(0, 1)
    fig.text(0.635, 0.885, "GCF axes: x1 red (normal), y1 green, z1 blue\nthick arrows: angular parts; shown at 1/2 speed",
             fontsize=12, color="#6b7280", va="top")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"))


def main():
    import mujoco
    import hom_contact_rig as HR
    import os
    os.environ.setdefault("HOM_FILM_VIEW", "70,14,0.10")
    test = len(sys.argv) > 1 and sys.argv[1] == "test"
    p_pads, p_pt, setup = T.build("expa", T.SIZES["expa"][0])
    m_pt = mujoco.MjModel.from_xml_path(str(p_pt))
    T.S.set_palm(m_pt, setup["palm"])
    fs, os_ = T.sr2_specs(setup)
    mirror = H.Mirror(m_pt, fs, os_)
    q_open, _ = T.open_pose(setup, mujoco.MjModel.from_xml_path(str(p_pt)))
    plant = T.Plant(p_pads, setup)
    plant.set_hom = lambda ci: setattr(plant, "_ci", ci)          # stores the period's info; no physics
    film = GcfFilm(plant, setup)
    over, noise = T.VARIANTS["base"]
    res, rows, frames = T.rollout("expa", setup, plant, mirror, q_open, 0, T.PRM, replace(H.HomParams(dt=T.DT_C), **over),
                                  film, noise)
    film.r.close()
    if test:
        for k in (20, 45, 70):
            Image.fromarray(frames[min(k, len(frames) - 1)]).resize((1280, 720)).save(
                f"/tmp/claude-1001/-home-humanoid-Programs-hand/8955e04e-1546-4892-9d3b-3e25a5d6bf60/scratchpad/gcf_{k}.jpg",
                quality=85)
        print(len(frames), "frames", flush=True)
        return 0
    slow = [f for f in frames for _ in range(2)]                     # 1/2 speed
    out = MEDIA / "20261110-gcf_expa_thumb.mp4"
    HR.write_mp4(slow + [slow[-1]] * 30, out, fps=25)
    Image.fromarray(frames[len(frames) // 3]).save(out.with_suffix(".jpg"), quality=88)
    print(out.relative_to(ROOT), len(slow), "frames;", {k: res["elements"][k] for k in list(res.get("elements", {}))[:6]},
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
