#!/usr/bin/env python3
"""Native-MuJoCo candidates for presliding compliance on the contact bed (2026-10-07), run through the bed's own task
scripts by patching hom_contact_rig.make_rig for their specs.

(a) Compliant skin, spec suffix `:skin`: each pad's spheres sit on a child body `skin<L|R>` joined to the fingertip by
    two slides tangent to the pad (y, z) and a hinge about its normal (x), with joint springs and dampers. Stiffness is
    the Cattaneo-Mindlin initial stiffness of the Hertz contact at the reference load N_ref = 1 N
    (scripts/contact_reference_laws.py, TPU stated as E 10 MPa, nu 0.45): k_t = 8 G a / (2 - nu) = 13.8 kN/m, the shear
    stiffness G A / h of a layer of area pi a^2 and thickness pi a (2 - nu) / 8 = 0.47 mm, and k_theta = 16 G a^3 / 3 =
    8.55 mN m/rad; damping critical on the skin's inertia. Skin 2 g with 18 g of armature. The sphere contacts keep
    the pad calibration, with the inverse weight of the skin body in place of the pad's. Friction is made stiff with
    impratio 1000 so that the skin, not the friction rows, carries the presliding motion.

    PY=logs/20261001-hom_contact/venv/bin/python
    $PY scripts/contact_bed_candidates.py t8 --models mj_pads1_skin mj_pads1_ir1000
    $PY scripts/contact_bed_candidates.py t2 --models mj_pads1_skin
"""
from __future__ import annotations

import argparse
import math
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contact_bed_common as B  # noqa: E402
import contact_reference_laws as L  # noqa: E402

H = B.H
OUT = B.ROOT / "docs/experiments/20261007-native_compliance"
N_REF = 1.0
_mat = L.mat()
_a = L.hertz(N_REF)["a"]
# Skin 2 g with 18 g of armature on its slides (5e-7 kg m^2 on the hinge): the sphere contacts' inverse weight follows the
# skin body, and MuJoCo softens a contact's friction rows in proportion to it; with the bare 2 g skin the tool slid 49 um
# over the spheres during a 2 N/s ramp and returned when the ramp stopped (20 g: 5 um, 200 g: 1 um).
# Damping: critical (zeta 1) at the skin's own inertia. With c = k t_r at the pads' t_r = 20 ms (8x critical) the tool led
# the skin by +5 um while the force rose at 2 N/s and trailed it by 3 um while it fell (17 um at 8 N/s), which reversed
# the T8 loop; at t_r 2 ms the offset was under 1 um.
SKIN = dict(kt=8 * _mat["G"] * _a / (2 - _mat["nu"]), kth=16 * _mat["G"] * _a ** 3 / 3, zeta=1.0, m=0.002, I=5e-8, arm=0.018,
            arm_t=5e-7)


def skin_damping(sk):
    """Slide and hinge damping at damping ratio zeta on the skin's inertia (mass or moment plus armature)."""
    return (2 * sk["zeta"] * math.sqrt(sk["kt"] * (sk["m"] + sk["arm"])),
            2 * sk["zeta"] * math.sqrt(sk["kth"] * (sk["I"] + sk["arm_t"])))
# (e) Rolling bristles (owner, 2026-10-07): each pad sphere on its own body with a ball joint at its centre, resisted by a
# rotational spring k_theta = k_t rs^2, so that the contact point rolls elastically by rs * theta before that sphere's
# Coulomb contact slips (a brush with one bristle per sphere). k_t = G A_s / h, with h = 2.7 mm chosen so that the ~11
# spheres in contact per pad at 1 N give Mindlin's initial stiffness 13.8 kN/m; damping k_theta x 1 ms (implicit, so the
# 0.1 ms bristle period does not limit the step); sphere bodies 1 mg each, taken off the pad's 20 g. `:bristle<deg>` gives
# joints to the spheres within <deg> of the pad's pole only (the rest stay on the pad).
BRISTLE = dict(h=2.7e-3, tau=1e-3, m=1e-6)
CANDIDATES = {
    "mj_pads1_ir1000": "mj:spheres:s1:rs0.75:ir1000:tr0.02",
    "mj_pads1_skin": "mj:spheres:s1:rs0.75:ir1000:tr0.02:skin",
    "mj_pads1_bristle": "mj:spheres:s1:rs0.75:ir1000:tr0.02:bristle",
    "mj_pads1_bristle20": "mj:spheres:s1:rs0.75:ir1000:tr0.02:bristle20",
    "mj_pads1_bristle20a": "mj:spheres:s1:rs0.75:ir1000:tr0.02:bristle20a",
    "mj_pads2_bristle20a": "mj:spheres:s2:rs0.75:ir1000:tr0.02:bristle20a",
}
# As specified (1 mg spheres, no armature) the bristles go unstable under tangential load at 1 and 2 ms; armature worth
# 0.2 g at the contact still does; 2 g at the contact (1.1e-9 kg m^2 per ball-joint dof) runs (suffix `a`).
BRISTLE_ARM = 2e-3 * 0.75e-3 ** 2


def add_skin(xml, sk):
    """Move each pad's sphere geoms onto a child body `skin<side>` with two tangential slides and a normal hinge."""
    ct, cth = skin_damping(sk)
    for s in "LR":
        geoms = re.findall(rf'<geom name="pad{s}_s\d+"[^>]*/>', xml)
        for g in geoms:
            xml = xml.replace(g, "", 1)
        body = (f'<body name="skin{s}">\n'
                f'        <joint name="skin{s}_y" type="slide" axis="0 1 0" stiffness="{sk["kt"]:.6g}" damping="{ct:.6g}" armature="{sk["arm"]:.6g}"/>\n'
                f'        <joint name="skin{s}_z" type="slide" axis="0 0 1" stiffness="{sk["kt"]:.6g}" damping="{ct:.6g}" armature="{sk["arm"]:.6g}"/>\n'
                f'        <joint name="skin{s}_t" type="hinge" axis="1 0 0" stiffness="{sk["kth"]:.6g}" damping="{cth:.6g}" armature="{sk["arm_t"]:.6g}"/>\n'
                f'        <inertial pos="0 0 0" mass="{sk["m"]}" diaginertia="{sk["I"]} {sk["I"]} {sk["I"]}"/>\n        '
                + "\n        ".join(geoms) + "\n      </body>")
        vis = re.search(rf'<geom name="pad{s}_vis"[^>]*/>', xml).group(0)
        xml = xml.replace(vis, vis + "\n      " + body, 1)
    return xml


class SkinRig(H.MjRig):
    """hom_contact_rig.MjRig with the pad spheres on a compliant skin body (candidate a)."""

    def __init__(self, sp, d_cg, gravity, kinematic=False, skin=None):
        import mujoco
        sk = dict(SKIN, **(skin or {}))
        self.mj, self.sp, self.d_cg, self.skin = mujoco, sp, d_cg, sk
        self.x0 = H.tool_touch(sp)
        xml, info = H.mj_xml(sp, d_cg, gravity, kinematic)
        m0 = mujoco.MjModel.from_xml_string(add_skin(xml, sk))
        diag = float(m0.body_invweight0[m0.body("skinL").id, 0] + m0.body_invweight0[m0.body("tool").id, 0])
        tc = sp["tr"] / 2.0
        d0 = 1.0 - 1.0 / (tc ** 2 * info["K_sphere"] * diag)
        if d0 < 0.05:
            raise ValueError(f"skin pad d0 {d0:.3f} below 0.05")
        xml, self.info = H.mj_xml(sp, d_cg, gravity, kinematic, pad_stiff=(tc, d0))
        self.xml = add_skin(xml, sk)
        self.info.update(solref_timeconst=tc, solimp_d0=d0, relaxation_s=2 * tc, diagApprox=diag, skin=sk)
        self.m = mujoco.MjModel.from_xml_string(self.xml)
        self.d = mujoco.MjData(self.m)
        self.tool = self.m.body("tool").id
        self.pads = {s: self.m.body("pad" + s).id for s in "LR"}
        self.skins = {s: self.m.body("skin" + s).id for s in "LR"}
        self.tool_geom = self.m.geom("tool").id
        self.geom_side = {gi: s for gi in range(self.m.ngeom) for s in "LR"
                          if self.m.geom_bodyid[gi] in (self.pads[s], self.skins[s]) and self.m.geom_contype[gi]}
        self.pad_geom = {}
        self.lastN = {"L": 0.0, "R": 0.0}
        self.f6 = np.zeros(6)
        mujoco.mj_forward(self.m, self.d)
        self.theta_prev, self.theta_unwrap = None, 0.0

    def skin_state(self):
        return {s: [float(self.d.qpos[self.m.joint(f"skin{s}_{j}").qposadr[0]]) for j in "yzt"] for s in "LR"}


def add_bristles(xml, sp, info, cap_deg=None, br=None):
    """Put each pad sphere (within cap_deg of the pole, or all) on its own body with a spring-loaded ball joint."""
    br = dict(BRISTLE, **(br or {}))
    G = _mat["G"]
    A_s = info["area_per_sphere_mm2"] * 1e-6
    rs = sp["rs"]
    kt = G * A_s / br["h"]
    kth = kt * rs ** 2
    I = 0.4 * br["m"] * rs ** 2
    n_b = {"L": 0, "R": 0}
    for side, face in (("L", 1.0), ("R", -1.0)):
        for g in re.findall(rf'<geom name="pad{side}_s(\d+)"[^>]*/>', xml):
            pass
        for m in list(re.finditer(rf'<geom name="pad{side}_s(\d+)" type="sphere" size="([^"]+)" pos="([^"]+)"([^>]*)/>', xml)):
            pos = np.array([float(v) for v in m.group(3).split()])
            ang = math.degrees(math.acos(np.clip(face * pos[0] / np.linalg.norm(pos), -1, 1)))
            if cap_deg is not None and ang > cap_deg:
                continue
            i = m.group(1)
            arm = f' armature="{br["arm"]:.6g}"' if br.get("arm") else ""
            body = (f'<body name="pad{side}_b{i}" pos="{m.group(3)}"><joint name="pad{side}_j{i}" type="ball" '
                    f'stiffness="{kth:.6g}" damping="{kth * br["tau"]:.6g}"{arm}/><inertial pos="0 0 0" mass="{br["m"]:.6g}" '
                    f'diaginertia="{I:.6g} {I:.6g} {I:.6g}"/><geom name="pad{side}_s{i}" type="sphere" size="{m.group(2)}"{m.group(4)}/></body>')
            xml = xml.replace(m.group(0), body, 1)
            n_b[side] += 1
    for side in "LR":                                   # keep each fingertip at 20 g
        xml = xml.replace(f'<body name="pad{side}" pos=', f'<body name="pad{side}" pos=', 1)
    m_pad = H.M_PAD - n_b["L"] * br["m"]
    xml = re.sub(rf'mass="{H.M_PAD}"', f'mass="{m_pad:.6g}"', xml)
    return xml, dict(kt_sphere=kt, kth_sphere=kth, n_bristles=n_b, h=br["h"], tau=br["tau"], m_sphere=br["m"])


class BristleRig(H.MjRig):
    """hom_contact_rig.MjRig with spring-loaded ball joints under the pad spheres (candidate e)."""

    def __init__(self, sp, d_cg, gravity, kinematic=False, cap_deg=None, arm=None):
        import mujoco
        self.mj, self.sp, self.d_cg = mujoco, sp, d_cg
        self.x0 = H.tool_touch(sp)
        xml, info = H.mj_xml(sp, d_cg, gravity, kinematic)
        br = {"arm": arm} if arm else None
        x1, _ = add_bristles(xml, sp, info, cap_deg, br)
        m0 = mujoco.MjModel.from_xml_string(x1)
        bid = m0.body("padL_b0").id if cap_deg is None or "padL_b0" in x1 else m0.body("padL").id
        diag = float(m0.body_invweight0[bid, 0] + m0.body_invweight0[m0.body("tool").id, 0])
        tc = sp["tr"] / 2.0
        d0 = 1.0 - 1.0 / (tc ** 2 * info["K_sphere"] * diag)
        xml, self.info = H.mj_xml(sp, d_cg, gravity, kinematic, pad_stiff=(tc, d0))
        self.xml, binfo = add_bristles(xml, sp, self.info, cap_deg, br)
        self.info.update(solref_timeconst=tc, solimp_d0=d0, relaxation_s=2 * tc, diagApprox=diag, bristle=binfo)
        self.m = mujoco.MjModel.from_xml_string(self.xml)
        self.d = mujoco.MjData(self.m)
        self.tool = self.m.body("tool").id
        self.pads = {s: self.m.body("pad" + s).id for s in "LR"}
        self.tool_geom = self.m.geom("tool").id
        root = {b: b for b in range(self.m.nbody)}
        self.geom_side = {}
        for gi in range(self.m.ngeom):
            b = int(self.m.geom_bodyid[gi])
            for s in "LR":
                if (b == self.pads[s] or self.m.body_parentid[b] == self.pads[s]) and self.m.geom_contype[gi]:
                    self.geom_side[gi] = s
        self.pad_geom = {}
        self.lastN = {"L": 0.0, "R": 0.0}
        self.f6 = np.zeros(6)
        mujoco.mj_forward(self.m, self.d)
        self.theta_prev, self.theta_unwrap = None, 0.0


_orig_make_rig = H.make_rig


def make_rig(spec, d_cg=0.0, gravity=True, kinematic=False):
    if spec.endswith(":skin"):
        return SkinRig(H.parse_spec(spec[:-5]), d_cg, gravity, kinematic)
    mb = re.search(r":bristle(\d*)(a?)$", spec)
    if mb:
        return BristleRig(H.parse_spec(spec[:mb.start()]), d_cg, gravity, kinematic,
                          cap_deg=float(mb.group(1)) if mb.group(1) else None, arm=BRISTLE_ARM if mb.group(2) else None)
    return _orig_make_rig(spec, d_cg, gravity, kinematic)


def install():
    """Route candidate specs through make_rig and register the candidates with every bed script's model table."""
    H.make_rig = make_rig
    for k, spec in CANDIDATES.items():
        B.MODELS[k] = (spec, spec)


def mjw_cost(model, nworlds=(1, 1024, 4096), N=1.0, blocks=20, block=10):
    """World-steps per second of a bed model in MuJoCo-Warp from a pinched state (settled 0.4 s at N on the CPU), 1 ms step,
    10-step CUDA graphs, put_model's broadphase, 512 contacts and 2048 constraint rows per world."""
    import mujoco_warp as mjw
    import warp as wp
    rig = B.new_rig(B.MODELS[model][0] if model in B.MODELS else model, 1e-3)
    rig.set_pad_force(N)
    rig.set_tool_wrench(np.zeros(3), np.zeros(3))
    rig.step(0.4)
    m, d = rig.m, rig.d
    rows = []
    for nw in nworlds:
        wp.synchronize()
        dev = wp.get_device("cuda:0")
        free0 = dev.free_memory
        wm = mjw.put_model(m)
        wd = mjw.make_data(m, nworld=nw, nconmax=512, njmax=2048)
        wd.qpos.assign(np.tile(d.qpos, (nw, 1)).astype(np.float32))
        wd.qvel.assign(np.tile(d.qvel, (nw, 1)).astype(np.float32))
        wd.ctrl.assign(np.tile(d.ctrl, (nw, 1)).astype(np.float32))
        mjw.forward(wm, wd)
        wp.synchronize()
        with wp.ScopedCapture() as cap:
            for _ in range(block):
                mjw.step(wm, wd)
        ticks = []
        for _ in range(blocks):
            wp.synchronize()
            t0 = time.perf_counter()
            wp.capture_launch(cap.graph)
            wp.synchronize()
            ticks.append(time.perf_counter() - t0)
        q = wd.qpos.numpy()
        tq = m.jnt_qposadr[m.body_jntadr[rig.tool]]
        held = float((np.linalg.norm(q[:, tq:tq + 3] - d.qpos[tq:tq + 3], axis=1) < 1e-3).mean())
        per = float(np.median(ticks)) / block
        rows.append(dict(task="mjw_cost", model=model, nworld=nw, N=N, dt_ms=1.0, ms_per_step=per * 1e3,
                         world_steps_per_s=nw / per, us_per_world_step=per / nw * 1e6, held_fraction=held,
                         nacon_per_world=float(wd.nacon.numpy()[0]) / nw, vram_mib=(free0 - dev.free_memory) / 2 ** 20,
                         graph_steps=block, script="scripts/contact_bed_candidates.py", git_rev=B.git_rev(),
                         when=time.strftime("%Y-%m-%d %H:%M")))
        print(model, nw, {k: round(v, 3) for k, v in rows[-1].items() if isinstance(v, float)}, flush=True)
        del wd, wm
    return rows


def main():
    import contact_bed_compliance as CC
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", choices=["t1", "t2", "t5", "t8", "t9", "cost"])
    ap.add_argument("--models", nargs="+", default=list(CANDIDATES))
    ap.add_argument("--N", nargs="+", type=float)
    ap.add_argument("--dt", nargs="+", type=float, default=[1.0])
    a = ap.parse_args()
    install()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.task == "cost":
        import warp as wp
        wp.init()
        for model in a.models:
            for r in mjw_cost(model):
                H.append_row(OUT / "mjw_cost.jsonl", r)
        return
    if a.task in ("t8", "t9"):
        out = OUT / ("t8_cycle.jsonl" if a.task == "t8" else "t9_sweep.jsonl")
        have = B.done(out)
        for model in a.models:
            for dt in a.dt:
                for N in a.N or ([0.5, 1.0, 3.0] if a.task == "t8" else [0.25, 0.5, 1.0, 2.0, 4.0]):
                    if (model, N, dt) in have:
                        continue
                    try:
                        r = CC.run_t8(model, N, dt) if a.task == "t8" else CC.run_t9(model, N, dt)
                    except Exception as e:
                        r = {"task": a.task, "model": model, "N": N, "dt_ms": dt, "status": "failed", "error": repr(e)[:500],
                             "traceback": traceback.format_exc()[-1500:]}
                    H.append_row(out, r)
                    print(model, N, dt, r.get("status"), {k: round(r[k], 4) for k in r if isinstance(r[k], float) and
                                                          k.endswith(("_um", "_uJ", "_mm", "overlap", "median"))}, flush=True)
    elif a.task == "t2":
        import contact_bed_twist as TW
        out = OUT / "twist_slip.jsonl"
        have = B.done(out)
        for model in a.models:
            for dt in a.dt:
                for N in a.N or [0.5, 1.0, 3.0]:
                    if (model, N, dt) in have:
                        continue
                    try:
                        r = TW.run_case(model, N, dt, film=False)
                    except Exception as e:
                        r = {"task": "twist", "model": model, "N": N, "dt_ms": dt, "status": "failed", "error": repr(e)[:500],
                             "traceback": traceback.format_exc()[-1500:]}
                    TW.add_ratio(r, out)
                    H.append_row(out, r)
                    print(model, N, dt, r.get("status"), {k: round(r[k], 4) for k in ("rbar_onset_mm", "rbar_slide_mm", "rot_pre_deg",
                                                                                     "creep_deg_s", "rbar_ratio_3_05_onset",
                                                                                     "us_per_step_median") if isinstance(r.get(k), float)},
                          r.get("error", ""), flush=True)
    elif a.task == "t1":
        import contact_bed_pull as PL
        out = OUT / "pull_slip.jsonl"
        have = B.done(out, key=("chain_spec", "N", "dt_ms"))
        for model in a.models:
            PL.MODELS[model] = B.MODELS[model][0]
            for dt in a.dt:
                for N in a.N or [0.5, 1.0, 3.0]:
                    if (model, N, dt) in have:
                        continue
                    try:
                        r = PL.run_case(model, N, dt)
                        r["model"] = model
                    except Exception as e:
                        r = {"task": "pull", "chain_spec": model, "model": model, "N": N, "dt_ms": dt, "status": "failed",
                             "error": repr(e)[:500], "traceback": traceback.format_exc()[-1500:]}
                    H.append_row(out, r)
                    print(model, N, dt, {k: round(r[k], 4) for k in ("mu_eff", "u_pre_mm", "creep_mm_s", "mu_slide", "us_per_step_median")
                                         if isinstance(r.get(k), float)}, r.get("error", ""), flush=True)
    elif a.task == "t5":
        import contact_bed_brake as BR
        out = OUT / "brake.jsonl"
        have = B.done(out, key=("model", "dt_ms"))
        for model in a.models:
            for dt in a.dt:
                if (model, dt) in have:
                    continue
                try:
                    r = BR.run_case(model, dt, film=False)
                except Exception as e:
                    r = {"task": "brake", "model": model, "dt_ms": dt, "status": "failed", "error": repr(e)[:500],
                         "traceback": traceback.format_exc()[-1500:]}
                H.append_row(out, r)
                print(model, dt, r.get("status"), {k: round(r[k], 3) for k in ("phi_max_deg", "phi_end_deg", "N_at_80_N", "t_80_s",
                                                                              "us_per_step_median") if isinstance(r.get(k), float)},
                      r.get("error", ""), flush=True)


if __name__ == "__main__":
    main()
