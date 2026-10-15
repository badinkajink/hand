#!/usr/bin/env python3
"""Contact-bed cases of the fingertip co-design study (docs/notes/20261014-tip_codesign_plan.md, steps 3 and 5).

The two-pad pinch rig of the contact bed (docs/experiments/20261005-contact_bed/PROTOCOL.md: two force-controlled rails
squeeze the 12.5 mm screwdriver, mu 1, 1 ms) with tpu_tip_rig's rounded blocks, wrapped without editing the bed
scripts: pad_usability_bed supplies the rigs' missing x0, the block meshes go to logs/20261014-tip_codesign/meshes
(written through a temporary file), and contact_bed_twist.C_LAW is set per case so that the law's onset falls at 1.0 s
of the T2 ramp, then restored.

Cases (one fsynced row each, resumable by key):
  static N          patch statistics after 0.8 s at pinch N per pad (tpu_tip_rig.static's case); approach = rail
                    travel from first touch, mean and sd over the last 0.2 s
  T2 N              contact_bed_twist.run_case: twist onset, hold creep and the kinetic arm at 1 rad/s
  spin N omega      the tool driven about the pinch axis at omega for 1.2/omega s after a 1 s settle; the arm is the
                    mean drive torque over the turn 0.4-1.2 rad (the contact line at 23-69 deg) over 2 mu N. In MuJoCo
                    every sample also splits each pad's normal force into the terms of the elliptic cone's sliding
                    zone (MuJoCo engine_core_constraint.c): f_n = D/(1 + mu_r^2) (-jar_n + mu ||jar_t||) with
                    jar = J qacc - aref, i.e. spring K I |r|, normal damping -B v_n, normal acceleration -a_n and the
                    slip term mu ||a_t + B v_t||; in the stick zone f_n = -D jar_n.
  T5                contact_bed_brake.run_case: the braked swing, N at 10/45/80 deg

Spec fields added to tpu_tip_rig's (stripped before hom_contact_rig.parse_spec):
  zt<z>      pad damping ratio z at unchanged stiffness: solref (tc / z, z) in place of (tc, 1)
  cpyr       pyramidal friction cone
  plate      the tool is a 25 mm thick plate (box, faces 30 x 30 mm) in place of the cylinder: the pinched patch is
             the palmar flat face and does not sweep as the tool spins
  pf<t>      pad-tool contacts as explicit <pair> elements with solreffriction (t, 1): the friction rows' damping
             B = 2 / (dmax t) in place of the normal rows' 2 / (dmax tc)

    PY=logs/20261001-hom_contact/venv/bin/python
    $PY scripts/tip_codesign_bed.py --models mj_tpu2p7_pads1 --cases spin --N 1 --omega 0.1 1 3 --out <rows.jsonl>
"""
from __future__ import annotations

import argparse
import math
import os
import re
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import pad_usability_backends as PUB  # noqa: E402,F401  (atomic mesh writes)
import pad_usability_bed  # noqa: E402,F401  (x0 on tpu_tip_rig's rigs)
import contact_bed_brake as BR  # noqa: E402
import contact_bed_common as CB  # noqa: E402
import contact_bed_pull as P  # noqa: E402
import contact_bed_twist as TW  # noqa: E402
import hom_contact_rig as H  # noqa: E402
import tpu_tip_rig as T  # noqa: E402

SCRIPT = "scripts/tip_codesign_bed.py"
OUT = ROOT / "docs/experiments/20261014-tip_codesign/deterioration"
T.MESH_DIR = ROOT / "logs/20261014-tip_codesign/meshes"
XFIELDS = ("zt", "cpyr", "plate", "pf")
PLATE_HALF = 0.015
try:
    import mujoco as _mj
    ELLIPTIC = int(_mj.mjtConstraint.mjCNSTR_CONTACT_ELLIPTIC)
    CONE = int(_mj.mjtConstraintState.mjCNSTRSTATE_CONE)
except Exception:                                   # interpreters without MuJoCo import this module for its specs
    ELLIPTIC, CONE = 7, 4


# ---------------------------------------------------------------------------------- spec fields

_parse_spec = H.parse_spec


def parse_spec(spec: str) -> dict:
    """hom_contact_rig.parse_spec with the fields of XFIELDS taken out into sp['x']."""
    parts, x = spec.split(":"), {}
    keep = []
    for p in parts:
        if p.startswith("zt"):
            x["zt"] = float(p[2:])
        elif p == "cpyr":
            x["cpyr"] = True
        elif p == "plate":
            x["plate"] = True
        elif p.startswith("pf"):
            x["pf"] = float(p[2:])
        elif p.startswith("tcd="):
            x["tcd"] = p[4:]
        elif p in ("hloc", "Keach"):
            x[p] = True
        else:
            keep.append(p)
    sp = _parse_spec(":".join(keep))
    sp["spec"], sp["x"] = spec, x
    return sp


H.parse_spec = parse_spec


def xopt(xml: str, sp: dict, pad_stiff) -> str:
    """Apply sp['x'] to a rig MJCF whose pads carry solref (tc, 1) and solimp d0 (the second, calibrated pass)."""
    x = sp.get("x") or {}
    if not x or pad_stiff is None:
        return xml
    tc, d0 = pad_stiff
    if "cpyr" in x:
        xml = xml.replace('cone="elliptic"', 'cone="pyramidal"')
    if "Keach" in x and _PAD_K.get("K") is not None:
        diag = _PAD_K["diag"]
        root = ET.fromstring(xml)
        for side in "LR":
            Ks = _PAD_K["K"][side]
            for i, Ki in enumerate(Ks):
                g = root.find(f".//geom[@name='pad{side}_s{i}']")
                d0i = 1.0 - 1.0 / (tc ** 2 * Ki * diag)
                if d0i < 0.05:
                    raise ValueError(f"pad {side}{i}: d0 {d0i:.3f} below 0.05")
                g.set("solimp", f"{d0i:.6g} {d0i:.6g} 0.001 0.5 2")
                g.set("priority", "1")              # the pad's own solimp, not its mix with the tool's
        xml = ET.tostring(root, encoding="unicode")
    if "zt" in x:
        z = x["zt"]
        xml = xml.replace(f'solref="{tc:.6g} 1"', f'solref="{tc / z:.6g} {z:g}"')
    if "plate" in x or "pf" in x:
        root = ET.fromstring(xml)
        tool = root.find(".//geom[@name='tool']")
        if "plate" in x:
            tool.set("type", "box")
            tool.set("size", f"{H.R_TOOL:g} {PLATE_HALF:g} {PLATE_HALF:g}")
        if "pf" in x:
            con = root.find("contact")
            if con is None:
                con = ET.SubElement(root, "contact")
            for g in root.iter("geom"):
                n = g.get("name", "")
                if re.fullmatch(r"pad[LR]_s\d+", n):
                    g.set("contype", "2")               # no dynamic pair with the tool (conaffinity 1); the rig
                    g.set("conaffinity", "0")           # finds pad geoms by a nonzero contype
                    ET.SubElement(con, "pair", geom1=n, geom2="tool", condim="3",
                                  friction=f"{H.MU:g} {H.MU:g} 0.005 0.0001 0.0001", solref=g.get("solref"),
                                  solreffriction=f"{x['pf']:g} 1", solimp=g.get("solimp"))
        xml = ET.tostring(root, encoding="unicode")
    return xml


_tpu_xml = T.tpu_xml


def tpu_xml(sp, d_cg, gravity, kinematic, pad_stiff=None):
    xml, info = _tpu_xml(sp, d_cg, gravity, kinematic, pad_stiff)
    return xopt(xml, sp, pad_stiff), info


T.tpu_xml = tpu_xml
_mj_xml = H.mj_xml
_PAD_K = {}


def _tcd_init(cls):
    """Wrap a tpu_tip_rig rig's constructor: for a tcd=<key> spec, fingertip_geometry's pad sampler and mesh writer
    are replaced by the shape's while the rig is built."""
    init = cls.__init__

    def __init__(self, sp, d_cg, gravity, kinematic=False):
        x = sp.get("x") or {}
        if "tcd" not in x:
            return init(self, sp, d_cg, gravity, kinematic)
        import fingertip_geometry as G
        import tip_codesign_geometry as TG
        sh = TG.shape(x["tcd"])
        saved = G.pad_spheres, G.mesh_paths
        K_by_side = {}

        def pad_spheres(r, s_, rs, sign=1.0, E=1e7):
            C, Nn, A, K = sh.pads(E, s_, rs, sign, h="local" if "hloc" in x else "fixed", K="each", full=True)
            K_by_side["L" if sign > 0 else "R"] = K
            return C, Nn, A, (K if "Keach" in x else np.full(len(K), float(np.median(K))))
        G.pad_spheres = pad_spheres
        G.mesh_paths = lambda out_dir, r: sh.mesh_paths(Path(out_dir) / sh.key)
        _PAD_K.clear()
        if "Keach" in x and cls is T.TpuMjRig:
            import mujoco
            xml0, _ = _tpu_xml(dict(sp, x={}), d_cg, gravity, kinematic, None)
            m0 = mujoco.MjModel.from_xml_string(xml0)
            _PAD_K.update(K=K_by_side, diag=m0.body_invweight0[m0.body("padL").id, 0]
                          + m0.body_invweight0[m0.body("tool").id, 0])
        try:
            init(self, sp, d_cg, gravity, kinematic)
            self.info = dict(getattr(self, "info", {}) or {}, tcd=sh.key)
        finally:
            G.pad_spheres, G.mesh_paths = saved
            _PAD_K.clear()
    cls.__init__ = __init__


_tcd_init(T.TpuMjRig)
_tcd_init(T.TpuDrakeRig)


def mj_xml(sp, d_cg, gravity, kinematic=False, pad_stiff=None, **kw):
    xml, info = _mj_xml(sp, d_cg, gravity, kinematic, pad_stiff, **kw)
    return (xopt(xml, sp, pad_stiff) if sp.get("model") == "spheres" else xml), info


H.mj_xml = mj_xml


def register(models: dict):
    """Model name -> (rig spec, chain spec) into the bed's tables, as tpu_tip_rig.bed does."""
    for k, v in models.items():
        CB.MODELS[k] = v
        P.MODELS[v[1]] = v[0]


def tcd_model(key: str, extra: str = "", drake: bool = False):
    """(name, (rig spec, chain spec)) of a tip_codesign_geometry shape on the bed (its nominal tpu field r_side)."""
    import tip_codesign_geometry as TG
    r_mm = TG.shape(key).a * 1e3
    ex = (":" + extra) if extra else ""
    t = f"tpu{r_mm:g}:tcd={key}"
    if drake:
        return f"drake_{key}", (f"drake:hydro:E1e7:r1:rt0.01:{t}", f"drake:hydro:rt0.01:{t}")
    return (f"mj_{key}_pads1" + extra.replace(":", "_").replace(".", "p"),
            (f"mj:spheres:s1:rs0.75:ir100:tr0.02{ex}:{t}", f"mj:spheres:s1:rs0.75:tr0.02{ex}:{t}"))


def tpu_model(r_mm: float, extra: str = "", name: str | None = None, drake: bool = False):
    """(name, (rig spec, chain spec)) of tpu_tip_rig's 1 mm pads (or its Drake block) at fillet r, with extra fields."""
    t = f"tpu{r_mm:g}"
    n = f"{r_mm:g}".replace(".", "p")
    if drake:
        return name or f"drake_tpu{n}", (f"drake:hydro:E1e7:r1:rt0.01:{t}", f"drake:hydro:rt0.01:{t}")
    ex = (":" + extra) if extra else ""
    return (name or f"mj_tpu{n}_pads1" + extra.replace(":", "_").replace(".", "p"),
            (f"mj:spheres:s1:rs0.75:ir100:tr0.02{ex}:{t}", f"mj:spheres:s1:rs0.75:tr0.02{ex}:{t}"))


# ---------------------------------------------------------------------------------- cases

def case_static(name, N, dt_ms):
    spec = CB.MODELS[name][0]
    with T.tpu_rigs():
        H.DT = dt_ms * 1e-3
        rig = H.make_rig(spec, d_cg=0.0, gravity=False)
        rig.set_pad_force(N)
        rig.set_tool_wrench(np.zeros(3), np.zeros(3))
        W, X = [], []
        n = int(round(0.8 / H.DT))
        for k in range(n):
            w0 = time.perf_counter()
            rig.step(H.DT)
            W.append(time.perf_counter() - w0)
            if k >= n - int(round(0.2 / H.DT)):
                X.append(H.tool_touch(rig.sp) - abs(rig.pad_x()["L"]))
        pt = T.patch_of(rig)
    X = np.array(X) * 1e6
    return {"L": pt["L"], "R": pt["R"], "info": getattr(rig, "info", {}),
            "approach_um": float(X.mean()), "approach_sd_um": float(X.std()), "T_static_s": 0.8,
            "us_per_step_median": float(np.median(W) * 1e6)}


def case_T2(name, N, dt_ms, rbar_law_m):
    """contact_bed_twist.run_case with C_LAW = rbar_law / N^(1/4), so the ramp reaches the law's onset torque at
    1.0 s and the 3 s cap at three times it."""
    saved = TW.C_LAW
    TW.C_LAW = rbar_law_m / N ** 0.25
    try:
        with T.tpu_rigs():
            r = TW.run_case(name, N, dt_ms, film=False)
    finally:
        TW.C_LAW = saved
    r["rbar_law_used_mm"] = rbar_law_m * 1e3
    return r


def case_T5(name, dt_ms):
    with T.tpu_rigs():
        r = BR.run_case(name, dt_ms, film=False)
    r["N"] = None
    return r


def _pad_bins():
    return [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 7), (7, 12)]


def decompose(rig, side="L"):
    """Per loaded contact of one pad: rho (mm) from the tool centre's axis line about x, f_n, and the terms of f_n
    (spring, normal damping, normal acceleration, slip) from the efc arrays of the last step."""
    m, d, mj = rig.m, rig.d, rig.mj
    if d.nefc == 0:
        return np.zeros((0, 7))
    a1 = np.zeros(d.nefc)
    mj.mj_mulJacVec(m, d, a1, d.qacc)
    jar = a1 - d.efc_aref
    KBIP = np.asarray(d.efc_KBIP).reshape(-1, 4)
    com = rig.tool_state()["pos"]
    out = []
    for i in range(d.ncon):
        c = d.contact[i]
        if rig.geom_side.get(int(c.geom[0])) != side and rig.geom_side.get(int(c.geom[1])) != side:
            continue
        a = int(c.efc_address)
        if a < 0 or d.efc_force[a] <= 1e-9:
            continue
        rho = math.hypot(c.pos[1] - com[1], c.pos[2] - com[2]) * 1e3
        K, B, I = KBIP[a, 0], KBIP[a, 1], KBIP[a, 2]
        D = d.efc_D[a]
        fn = float(d.efc_force[a])
        if int(d.efc_type[a]) == ELLIPTIC:
            dim = int(c.dim)
            mu_r = float(c.mu)
            fr = float(c.friction[0])
            cone = int(d.efc_state[a]) == CONE
            sl = fr * float(np.linalg.norm(jar[a + 1:a + dim])) if cone else 0.0
            g = D / (1 + mu_r ** 2) if cone else D
            spring = g * K * I * (-d.efc_pos[a])
            damp = g * (-B * d.efc_vel[a])
            acc = g * (-a1[a])
            slip = g * sl
            out.append([rho, fn, spring, damp, acc, slip, float(d.efc_state[a])])
    return np.array(out) if out else np.zeros((0, 7))


def case_spin(name, N, omega, dt_ms, decomp=True):
    spec = CB.MODELS[name][0]
    with T.tpu_rigs():
        H.DT = dt_ms * 1e-3
        rig = H.make_rig(spec, d_cg=0.0, gravity=False)
    H.settle(rig, N, T=1.0, weight_comp=False)
    st = rig.tool_state()
    th0, t0 = st["theta"], rig.t
    kp, kd = 0.5, 0.005
    T_end = 1.2 / omega
    n = int(round(T_end / H.DT))
    k_s = max(1, int(round(0.01 / (omega * H.DT))))       # a sample every 0.01 rad of turn
    taus, trace, W, parts, drift = [], [], [], [], 0.0
    p0 = st["pos"].copy()
    for k in range(n):
        st = rig.tool_state()
        tt = rig.t - t0
        tau = kp * (th0 + omega * tt - st["theta"]) + kd * (omega - st["omega_x"])
        rig.set_tool_wrench(np.zeros(3), np.array([tau, 0.0, 0.0]))
        w0 = time.perf_counter()
        rig.step(H.DT)
        W.append(time.perf_counter() - w0)
        th = st["theta"] - th0
        if th >= 0.4:
            taus.append(tau)
            if k % k_s == 0:
                c = rig.contacts()
                trace.append([round(th, 4), tau, c["L"]["N"], c["L"]["n"]])
                drift = max(drift, float(np.linalg.norm(st["pos"] - p0)))
                if decomp and rig.sim == "mujoco":
                    parts.append(decompose(rig, "L"))
    c = rig.contacts()
    row = {"model": name, "rig_spec": spec, "N": N, "omega": omega, "dt_ms": dt_ms, "T_spin_s": T_end,
           "tau_mean_Nm": float(np.mean(taus)), "tau_sd_Nm": float(np.std(taus)),
           "rbar_spin_mm": float(np.mean(taus)) / (2 * H.MU * N) * 1e3, "N_L": c["L"]["N"], "N_R": c["R"]["N"],
           "n_L": c["L"]["n"], "n_R": c["R"]["n"], "drift_max_mm": drift * 1e3, "info": getattr(rig, "info", {}),
           "us_per_step_median": float(np.median(W) * 1e6), "trace_cols": ["theta_rad", "tau_Nm", "N_L", "n_L"],
           "trace": trace[::max(1, len(trace) // 40)]}
    if parts:
        A = np.concatenate([p for p in parts if len(p)])
        K_s = (row["info"] or {}).get("K_sphere")
        fn, rho = A[:, 1], A[:, 0]
        row["n_samples"] = len(parts)
        row["loaded_per_side"] = float(len(A) / len(parts))
        row["rho_fw_mm"] = float((fn * rho).sum() / fn.sum())
        row["share"] = {k: float(A[:, j].sum() / fn.sum()) for j, k in ((2, "spring"), (3, "damp"), (4, "acc"),
                                                                         (5, "slip"))}
        row["cone_frac"] = float((A[:, 6] == CONE).mean()) if len(A) else None
        nsl = fn - A[:, 5]
        row["rho_fw_noslip_mm"] = float((nsl * rho).sum() / max(nsl.sum(), 1e-12))
        row["rho_spring_w_mm"] = float((A[:, 2] * rho).sum() / max(A[:, 2].sum(), 1e-12))
        bins = []
        for lo, hi in _pad_bins():
            mk = (rho >= lo) & (rho < hi)
            if mk.any():
                bins.append({"rho_mm": [lo, hi], "n": int(mk.sum() / len(parts)), "f_N": float(fn[mk].sum() / len(parts)),
                             "spring_N": float(A[mk, 2].sum() / len(parts)), "slip_N": float(A[mk, 5].sum() / len(parts)),
                             "damp_N": float(A[mk, 3].sum() / len(parts)), "acc_N": float(A[mk, 4].sum() / len(parts))})
        row["bins"] = bins
        row["K_sphere"] = K_s
    return row


# ---------------------------------------------------------------------------------- driver

def key_of(r):
    return (r["case"], r["model"], r.get("N"), r.get("omega"), r["dt_ms"])


def run(models, cases, Ns, omegas, dts, out, laws=None, decomp=True):
    """Every (case, model, N, omega, dt) not yet in `out`. laws: {(model, N): onset arm of the law, m} for T2."""
    have = {key_of(r) for r in CB.read_rows(out) if "case" in r}
    for case in cases:
        for name in models:
            for dt in dts:
                for N in (Ns if case in ("static", "T2", "spin") else [None]):
                    for om in (omegas if case == "spin" else [None]):
                        k = (case, name, N, om, dt)
                        if k in have:
                            continue
                        t0 = time.time()
                        try:
                            if case == "static":
                                r = case_static(name, N, dt)
                            elif case == "T2":
                                r = case_T2(name, N, dt, laws[(name, N)])
                            elif case == "spin":
                                r = case_spin(name, N, om, dt, decomp)
                            else:
                                r = case_T5(name, dt)
                            r.setdefault("status", "complete")
                        except Exception as e:                  # a failed case is a row
                            r = {"status": "failed", "error": repr(e), "traceback": traceback.format_exc()[-1500:]}
                        r.update(case=case, model=name, rig_spec=CB.MODELS[name][0], N=N, omega=om, dt_ms=dt,
                                 script=SCRIPT, git_rev=CB.git_rev(), wall_s=time.time() - t0,
                                 when=time.strftime("%Y-%m-%d %H:%M"))
                        PUB.append_row(out, r)
                        have.add(k)
                        keys = ("approach_um", "rbar_onset_mm", "rbar_kin_mm", "rbar_spin_mm", "rho_fw_mm",
                                "N_at_45_N", "phi_end_deg", "us_per_step_median")
                        print(case, name, N, om, dt, r.get("status"),
                              {q: round(r[q], 4) for q in keys if isinstance(r.get(q), (int, float))},
                              r.get("share"), r.get("error", ""), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True, help="registered names, or r<mm>[:fields] / drake<mm>")
    ap.add_argument("--cases", nargs="+", default=["static", "spin"])
    ap.add_argument("--N", nargs="+", type=float, default=[1.0])
    ap.add_argument("--omega", nargs="+", type=float, default=[1.0])
    ap.add_argument("--dt", nargs="+", type=float, default=[1.0])
    ap.add_argument("--law-arm", nargs="*", default=[], help="N:mm pairs, the law's onset arm used to scale the T2 ramp")
    ap.add_argument("--no-decomp", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    names = []
    for s in a.models:
        if s in CB.MODELS:
            names.append(s)
            continue
        if s.startswith("tcd_") or s.startswith("drake_tcd_"):
            key, _, extra = s.removeprefix("drake_").partition(":")
            n, v = tcd_model(key, extra, drake=s.startswith("drake_"))
        elif s.startswith("drake"):
            n, v = tpu_model(float(s[5:]), drake=True)
        else:
            r_mm, _, extra = s[1:].partition(":")
            n, v = tpu_model(float(r_mm), extra)
        register({n: v})
        names.append(n)
    law = {float(k): float(v) * 1e-3 for k, v in (x.split(":") for x in a.law_arm)}
    laws = {(n, N): law[N] for n in names for N in law}
    run(names, a.cases, a.N, a.omega, a.dt, a.out, laws=laws, decomp=not a.no_decomp)


if __name__ == "__main__":
    main()
