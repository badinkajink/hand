#!/usr/bin/env python3
"""Contact-model comparison bed, task 6: creep of MuJoCo's soft friction under noslip and impratio.

The MuJoCo models of the bed (mj_point3, mj_point4s, mj_pads1) run the T1 pull and hold
(contact_bed_pull.run_case), the T2 twist and torque hold (contact_bed_twist.run_case) and the T4 shake at
a sub-threshold acceleration (contact_bed_shake.run_case) with noslip_iterations 0 / 10 crossed with
impratio 100 / 1000. The variant is applied by wrapping hom_contact_rig.make_rig, through which every rig
of those scripts is built; the scripts themselves are unchanged. Rows keep every metric of the base task.

Creep law of the soft friction. For a contact row MuJoCo solves J qacc + R f = aref with
aref = -b v - k imp r, b = 2 / (dmax tc), and R = (1 - imp) / imp * Lambda for the normal row
(Lambda = efc_diagApprox); with an elliptic cone the friction rows carry R_t = R_n / impratio and no
position term. At steady creep the relative tangential acceleration is zero, so R_t f_t = -b v on each
friction row: the contact behaves as a viscous damper of coefficient b / R_t. n contacts sharing a
tangential load F_t creep at

    v = R_t F_t / (n b) = (1 - d0) Lambda dmax tc F_t / (2 d0 impratio n)

with d0 the impedance (constant for the sphere pads, d(r) at the penetration for the point contacts) and
tc the solref time constant after MuJoCo's clamp tc >= 2 dt. `predict` evaluates it three ways from the
constraint state at the T1 hold: the formula with the mean d0, Lambda and b of the loaded contacts; the
per-contact sum v = F_t / sum(b_i / R_t,i); and a cone-aware version in which contacts whose share would
exceed mu f_n are capped at mu f_n. Noslip re-solves the friction rows with R = 0, so it predicts no creep.

    PY=logs/20261001-hom_contact/venv/bin/python
    $PY scripts/contact_bed_creep.py --parts T1 --models mj_pads1 --N 1 --dt 1     # one case, 4 variants
    $PY scripts/contact_bed_creep.py                       # T6 grid, then the formula against pull_slip.jsonl
    $PY scripts/contact_bed_creep.py --parts T1 T2 T4 T5 --models mj_pads1 mj_pads1_tr05 mj_pads1_tr10 \
        --variants 0:100 1:100 3:100 10:100 0:300 0:1000 10:1000    # the 2026-10-06 pad grid
"""
from __future__ import annotations

import argparse
import contextlib
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import contact_bed_brake as BR  # noqa: E402
import contact_bed_common as CB  # noqa: E402
import contact_bed_pull as P  # noqa: E402
import contact_bed_shake as SH  # noqa: E402
import contact_bed_twist as TW  # noqa: E402

H = CB.H
OUT = CB.BED / "creep.jsonl"
MODELS6 = ["mj_point3", "mj_point4s", "mj_pads1"]
# 1 mm pads at longer relaxation times. At the calibrated sphere stiffness K the creep law reads
# v = F_t / (impratio n K tr), so tr 0.05 and 0.1 s should creep 2.5x and 5x slower than tr 0.02 s.
EXTRA_MODELS = {
    "mj_pads1_tr05": ("mj:spheres:s1:rs0.75:ir100:tr0.05", "mj:spheres:s1:rs0.75:tr0.05"),
    "mj_pads1_tr10": ("mj:spheres:s1:rs0.75:ir100:tr0.1", "mj:spheres:s1:rs0.75:tr0.1"),
}
for _k, (_rig, _chain) in EXTRA_MODELS.items():
    CB.MODELS.setdefault(_k, (_rig, _chain))
    P.MODELS.setdefault(_chain, _rig)
VARIANTS = [(0, 100.0), (10, 100.0), (0, 1000.0), (10, 1000.0)]   # (noslip_iterations, impratio)
T1_N = [0.5, 1.0, 3.0]
T2_N = [0.5, 1.0, 3.0]
T4_CASES = [(0.5, 2.0), (0.2, 0.5)]        # (N, a_pk in g): load ratios m (g + a_pk) / 2 mu N of 0.72 and 0.90
DTS = [1.0, 5.0]
SCRIPT = "scripts/contact_bed_creep.py"


@contextlib.contextmanager
def variant(ns, ir):
    """Every rig built inside the block gets noslip_iterations `ns` and impratio `ir`."""
    orig = H.make_rig

    def make(*a, **kw):
        rig = orig(*a, **kw)
        if rig.sim == "mujoco":
            rig.m.opt.noslip_iterations = int(ns)
            rig.m.opt.impratio = float(ir)
            rig.mj.mj_forward(rig.m, rig.d)
        return rig
    H.make_rig = make
    try:
        yield
    finally:
        H.make_rig = orig


def efc_snapshot(rig):
    """Constraint state of the pad-tool contacts: per contact the normal force, the friction force vector,
    imp, b, k, R_n, R_t, Lambda, solref tc, solimp dmax and the contact position."""
    import mujoco
    m, d = rig.m, rig.d
    refsafe = not (m.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_REFSAFE))
    out = {k: [] for k in ("side", "fn", "ft", "imp", "B", "K", "Rn", "Rt", "Lam", "tc", "dmax", "pos", "state")}
    for i in range(d.ncon):
        c = d.contact[i]
        s = rig.geom_side.get(int(c.geom[0])) or rig.geom_side.get(int(c.geom[1]))
        adr = int(c.efc_address)
        if s is None or adr < 0:
            continue
        out["side"].append(s)
        out["fn"].append(float(d.efc_force[adr]))
        out["ft"].append(np.array(d.efc_force[adr + 1:adr + 3]))
        out["imp"].append(float(d.efc_KBIP[adr, 2]))
        out["B"].append(float(d.efc_KBIP[adr, 1]))
        out["K"].append(float(d.efc_KBIP[adr, 0]))
        out["Rn"].append(float(d.efc_R[adr]))
        out["Rt"].append(float(d.efc_R[adr + 1]))
        out["Lam"].append(float(d.efc_diagApprox[adr]))
        out["tc"].append(float(c.solref[0]))
        out["dmax"].append(float(c.solimp[1]))
        out["pos"].append(np.array(c.pos))
        out["state"].append(int(d.efc_state[adr]))
    for k in out:
        if k != "side":
            out[k] = np.array(out[k])
    out["dt"] = float(m.opt.timestep)
    out["impratio"] = float(m.opt.impratio)
    out["noslip"] = int(m.opt.noslip_iterations)
    out["refsafe"] = bool(refsafe)
    return out


def predict(snap, F_t, mu=H.MU):
    """Creep speed (m/s) of a tangential load F_t shared by the loaded contacts of `snap`."""
    load = snap["fn"] > 1e-9
    n = int(load.sum())
    if n == 0:
        return {"pred_n_contacts": 0}
    imp, Lam, Rt, B = snap["imp"][load], snap["Lam"][load], snap["Rt"][load], snap["B"][load]
    tc = snap["tc"][load]
    dmax = snap["dmax"][load]
    tc_eff = np.maximum(tc, 2 * snap["dt"]) if snap["refsafe"] else tc
    d0, L0, dm0, tce0 = float(imp.mean()), float(Lam.mean()), float(dmax.mean()), float(tc_eff.mean())
    ir = snap["impratio"]
    Rn_f = (1 - d0) / d0 * L0
    Rt_f = Rn_f / ir
    b_f = 2.0 / (dm0 * tce0)
    v_formula = Rt_f * F_t / (n * b_f)
    c = B / Rt                                  # per-contact viscous coefficient of the friction rows, N s/m
    v_rows = F_t / c.sum()
    cap = mu * snap["fn"][load]
    if cap.sum() <= F_t:
        v_cone = math.inf
    else:
        lo, hi = 0.0, max(v_rows, 1e-12)
        while (np.minimum(c * hi, cap)).sum() < F_t:
            hi *= 2
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if (np.minimum(c * mid, cap)).sum() < F_t:
                lo = mid
            else:
                hi = mid
        v_cone = 0.5 * (lo + hi)
    ft_mag = np.linalg.norm(snap["ft"][load], axis=1)
    return {"pred_n_contacts": n, "pred_d0": d0, "pred_dmax": dm0, "pred_tc_s": float(tc.mean()),
            "pred_tc_eff_s": tce0, "pred_Lambda": L0, "pred_impratio": ir, "pred_R_n": Rn_f, "pred_R_t": Rt_f,
            "pred_R_t_model": float(Rt.mean()), "pred_b": b_f, "pred_b_model": float(B.mean()),
            "pred_F_t": F_t, "pred_v_formula_mm_s": v_formula * 1e3, "pred_v_rows_mm_s": v_rows * 1e3,
            "pred_v_cone_mm_s": v_cone * 1e3 if math.isfinite(v_cone) else None,
            "pred_n_saturated": int((c * v_cone >= cap * (1 - 1e-9)).sum()) if math.isfinite(v_cone) else n,
            "pred_util_max": float((ft_mag / np.maximum(cap, 1e-12)).max()),
            "pred_fn_sum": float(snap["fn"][load].sum()),
            "pred_states": {str(s): int((snap["state"][load] == s).sum()) for s in set(snap["state"][load].tolist())}}


def t1_hold_state(model, N, dt_ms, F_h, t_snap=0.6, t_after=0.4):
    """The T1 hold of contact_bed_pull (settle, ramp at 2 N/s to F_h, hold), stopped t_snap into the hold for
    a constraint snapshot, then held t_after more for a creep slope of its own."""
    spec = CB.MODELS[model][0]
    dt = dt_ms * 1e-3
    rig = P.new_rig(spec, dt)
    p0, a0, _ = P.settle(rig, N)
    t_r = F_h / P.RATE
    T, F, U, V, W, _ = P.drive(rig, dt, p0, a0, lambda tt: min(P.RATE * tt, F_h), lambda *a: False, t_r + t_snap)
    snap = efc_snapshot(rig)
    t0 = T[-1]
    T2, F2, U2, V2, W2, _ = P.drive(rig, dt, p0, a0, lambda tt: F_h, lambda *a: False, t_after)
    slope = float(np.polyfit(T2, U2, 1)[0]) if len(T2) > 3 else float("nan")
    return snap, slope


def _base(part, model, N, dt_ms, ns, ir, a_pk_g=None):
    return {"task": "T6", "part": part, "model": model, "rig_spec": CB.MODELS[model][0], "N": N, "dt_ms": dt_ms,
            "a_pk_g": a_pk_g, "noslip_iterations": ns, "impratio": ir,
            "case_id": f"{part}|{model}|{N}|{dt_ms}|{a_pk_g}|{ns}|{ir}"}


def run_t1(model, N, dt_ms, ns, ir):
    row = _base("T1", model, N, dt_ms, ns, ir)
    with variant(ns, ir):
        r = P.run_case(CB.MODELS[model][1], N, dt_ms)
        r.pop("exp", None)
        row.update({k: v for k, v in r.items() if k not in ("spec",)})
        if r.get("slipped"):
            snap, slope = t1_hold_state(model, N, dt_ms, r["F_hold"])
            row.update(predict(snap, r["F_hold"]))
            row["creep_check_mm_s"] = slope * 1e3
    row["status"] = "complete"
    return row


def run_t2(model, N, dt_ms, ns, ir):
    row = _base("T2", model, N, dt_ms, ns, ir)
    with variant(ns, ir):
        r = TW.run_case(model, N, dt_ms, film=False)
    r.pop("task", None)
    row.update(r)
    row["base_task"] = "twist"
    return row


def run_t4(model, N, a_g, dt_ms, ns, ir):
    row = _base("T4", model, N, dt_ms, ns, ir, a_g)
    with variant(ns, ir):
        r = SH.run_case(model, N, dt_ms, a_g, film=None)
    r.pop("task", None)
    row.update(r)
    row["base_task"] = "T4"
    return row


def run_t5(model, dt_ms, ns, ir):
    row = _base("T5", model, None, dt_ms, ns, ir)
    with variant(ns, ir):
        r = BR.run_case(model, dt_ms, film=False)
    r.pop("task", None)
    row.update(r)
    row["base_task"] = "brake"
    return row


def formula_vs_pull(out):
    """The creep law against every MuJoCo row of the T1 table (pull_slip.jsonl), at its own hold force."""
    have = CB.done(out, key=("case_id",))
    rev = {v[1]: k for k, v in CB.MODELS.items()}
    for r in CB.read_rows(CB.BED / "pull_slip.jsonl"):
        model = rev.get(r.get("chain_spec"))
        if model not in MODELS6 or not r.get("slipped"):
            continue
        cid = f"T1_formula|{model}|{r['N']}|{r['dt_ms']}|pull_slip"
        if (cid,) in have:
            continue
        t0 = time.time()
        snap, slope = t1_hold_state(model, r["N"], r["dt_ms"], r["F_hold"])
        row = {"task": "T6", "part": "T1_formula", "model": model, "rig_spec": CB.MODELS[model][0], "N": r["N"],
               "dt_ms": r["dt_ms"], "noslip_iterations": 0, "impratio": snap["impratio"], "case_id": cid,
               "source": "pull_slip.jsonl", "F_hold": r["F_hold"], "creep_mm_s": r["creep_mm_s"],
               "creep_check_mm_s": slope * 1e3}
        row.update(predict(snap, r["F_hold"]))
        row["formula_over_measured"] = row["pred_v_formula_mm_s"] / r["creep_mm_s"]
        row["rows_over_measured"] = row["pred_v_rows_mm_s"] / r["creep_mm_s"]
        if row.get("pred_v_cone_mm_s"):
            row["cone_over_measured"] = row["pred_v_cone_mm_s"] / r["creep_mm_s"]
        row.update(status="complete", film=None, script=SCRIPT, git_rev=CB.git_rev(), wall_s=time.time() - t0,
                   when=time.strftime("%Y-%m-%d %H:%M"))
        H.append_row(out, row)
        print("formula", model, r["N"], r["dt_ms"], "measured %.5f formula %.5f rows %.5f cone %s n %d d0 %.4f" % (
            r["creep_mm_s"], row["pred_v_formula_mm_s"], row["pred_v_rows_mm_s"], row.get("pred_v_cone_mm_s"),
            row["pred_n_contacts"], row["pred_d0"]), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parts", nargs="+", default=["T1", "T2", "T4", "formula"])
    ap.add_argument("--models", nargs="+", default=MODELS6)
    ap.add_argument("--N", nargs="+", type=float)
    ap.add_argument("--dt", nargs="+", type=float, default=DTS)
    ap.add_argument("--variants", nargs="+", help="ns:ir pairs, e.g. 0:100 10:1000")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    variants = [(int(v.split(":")[0]), float(v.split(":")[1])) for v in a.variants] if a.variants else VARIANTS
    have = CB.done(a.out, key=("case_id",))
    jobs = []
    for part in a.parts:
        for model in a.models:
            for dt in a.dt:
                for ns, ir in variants:
                    if part == "T1":
                        jobs += [("T1", model, N, dt, ns, ir, None) for N in (a.N or T1_N)]
                    elif part == "T2":
                        jobs += [("T2", model, N, dt, ns, ir, None) for N in (a.N or T2_N)]
                    elif part == "T4":
                        jobs += [("T4", model, N, dt, ns, ir, ag) for N, ag in T4_CASES if not a.N or N in a.N]
                    elif part == "T5":
                        jobs.append(("T5", model, None, dt, ns, ir, None))
    for part, model, N, dt, ns, ir, ag in jobs:
        cid = f"{part}|{model}|{N}|{dt}|{ag}|{ns}|{ir}"
        if (cid,) in have:
            continue
        t0 = time.time()
        try:
            if part == "T1":
                r = run_t1(model, N, dt, ns, ir)
            elif part == "T2":
                r = run_t2(model, N, dt, ns, ir)
            elif part == "T5":
                r = run_t5(model, dt, ns, ir)
            else:
                r = run_t4(model, N, ag, dt, ns, ir)
        except Exception as e:                           # a failed case is a row, not a lost run
            r = _base(part, model, N, dt, ns, ir, ag)
            r.update(status="failed", error=repr(e), traceback=traceback.format_exc()[-2000:])
        r.update(film=None, script=SCRIPT, git_rev=CB.git_rev(), wall_s_t6=time.time() - t0,
                 when=time.strftime("%Y-%m-%d %H:%M"))
        H.append_row(a.out, r)
        keys = {"T1": ("mu_eff", "u_pre_mm", "creep_mm_s", "pred_v_rows_mm_s", "v_slip_mean50_mm_s", "mu_slide"),
                "T2": ("tau_onset_Nm", "rbar_onset_mm", "rot_pre_deg", "creep_deg_s", "rbar_kin_mm"),
                "T4": ("creep_g_mm_s", "drift_per_cycle_mm", "pp_last_cycle_mm", "N_sum_max_over_2N"),
                "T5": ("phi_end_deg", "overshoot_deg", "peak_rate_deg_s", "N_at_45_N", "N_at_80_N", "slip_end_mm")}[part]
        print(part, model, N, dt, ag, ns, ir, r.get("status"),
              {k: round(r[k], 6) for k in keys + ("us_per_step_median",) if isinstance(r.get(k), (int, float))},
              flush=True)
    if "formula" in a.parts:
        formula_vs_pull(a.out)


if __name__ == "__main__":
    main()
