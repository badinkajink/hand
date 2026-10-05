#!/usr/bin/env python3
"""Contact-model comparison bed, task 7: largest stable step of the sphere-pad pinch against d0 and tc.

The mj_pads1 rig (1 mm sphere pads, scripts/hom_contact_rig.py) holds the tool at N = 1 N per pad with
gravity on (MuJoCo default, -z, across the tool axis). Every contact geom gets solref (tc, 1) and solimp
(d0, d0, 0.001, 0.5, 2) set directly, over d0 in {0.2 ... 0.95} and tc in {5, 10, 20} ms. Each step dt in
{0.25 ... 15} ms runs a 0.4 s settle with the weight compensated and a 1 s hold without; the case holds when
the state stays finite (no MuJoCo bad-qacc/qvel/qpos warning), the tool centre stays within 5 mm of its
settled position and both pads keep a contact. MuJoCo's clamp tc >= 2 dt (refsafe) is on by default; every
combination also runs with refsafe disabled, which keeps tc as given and tests the bound itself.

Prediction. With constant impedance d0 a contact row obeys a = d aref + (1 - d) a0 with
aref = -b v - k r, b = 2 / (d0 tc), k = 1 / (d0^2 tc^2). n rows acting on one body in the same direction
share A = Lambda 1 1^T, so their collective impedance is d' = n d0 / (n d0 + 1 - d0) and the collective
damping rate is d' b = 2 d' / (d0 tc). MuJoCo evaluates aref with the velocity at the start of the step and
updates the velocity before the position (semi-implicit Euler), so the mode is stable when
4 - 2 dt d' b - dt^2 d' k d0 > 0, and the damping alone needs d' b dt < 2, i.e.
dt < d0 tc / d' = tc (d0 + (1 - d0) / n): d0 tc for many spheres, tc / n for small d0, tc for one point.
Each row carries these bounds for n = the spheres in contact on one pad and on both, and a numerical
version that replaces d' by the largest eigenvalue of A_act (A + R)^-1 over the loaded normal rows (A from
efc_J and M at the settled state; A_act uses M + dt * damping, as implicitfast does for the rails).

    PY=logs/20261001-hom_contact/venv/bin/python
    $PY scripts/contact_bed_stability.py --d0 0.5 --tc 10                  # one combination, all steps
    $PY scripts/contact_bed_stability.py                                   # pads map + point-contact checks
"""
from __future__ import annotations

import argparse
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import contact_bed_common as CB  # noqa: E402

H = CB.H
OUT = CB.BED / "stability.jsonl"
D0S = [0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95]
TCS_MS = [5.0, 10.0, 20.0]
DTS_MS = [0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0]
POINT = [(0.2, 10.0), (0.5, 10.0), (0.9, 10.0)]     # (d0, tc ms) for the point contact, n = 1 per pad
N_PINCH = 1.0
T_SETTLE, T_HOLD = 0.4, 1.0
EJECT_M = 0.005
SCRIPT = "scripts/contact_bed_stability.py"


def build(model, dt_ms, tc, d0, refsafe):
    import mujoco
    H.DT = dt_ms * 1e-3
    rig = H.make_rig(CB.MODELS[model][0], d_cg=0.0, gravity=True)
    m = rig.m
    for g in range(m.ngeom):
        if m.geom_contype[g] or m.geom_conaffinity[g]:
            m.geom_solref[g] = [tc, 1.0]
            m.geom_solimp[g] = [d0, d0, 0.001, 0.5, 2.0]
    if not refsafe:
        m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_REFSAFE)
    mujoco.mj_forward(m, rig.d)
    return rig


def bad_state(rig):
    import mujoco
    W = mujoco.mjtWarning
    if any(rig.d.warning[int(w)].number > 0 for w in (W.mjWARN_BADQACC, W.mjWARN_BADQVEL, W.mjWARN_BADQPOS)):
        return True
    return not (np.all(np.isfinite(rig.d.qpos)) and np.all(np.isfinite(rig.d.qvel)))


def loaded_rows(rig):
    """efc row indices (normal, friction) of the pad-tool contacts that carry normal force, and per-pad counts."""
    d = rig.d
    nrm, fr, cnt = [], [], {"L": 0, "R": 0}
    for i in range(d.ncon):
        c = d.contact[i]
        s = rig.geom_side.get(int(c.geom[0])) or rig.geom_side.get(int(c.geom[1]))
        adr = int(c.efc_address)
        if s is None or adr < 0 or d.efc_force[adr] <= 1e-9:
            continue
        nrm.append(adr)
        fr += [adr + 1, adr + 2]
        cnt[s] += 1
    return nrm, fr, cnt


def lam_max(rig, rows, dt):
    """Largest eigenvalue of A_act (A + R)^-1 over `rows`: A = J M^-1 J^T, A_act = J (M + dt D)^-1 J^T."""
    import mujoco
    m, d = rig.m, rig.d
    if not rows:
        return None, None
    nv = m.nv
    J = np.array(d.efc_J).reshape(d.nefc, nv)[rows]
    Minv = np.zeros((nv, nv))
    mujoco.mj_solveM(m, d, Minv, np.eye(nv))
    M = np.zeros((nv, nv))
    mujoco.mj_fullM(m, M, d.qM)
    Mi_act = np.linalg.inv(M + dt * np.diag(m.dof_damping))
    A = J @ Minv @ J.T
    A_act = J @ Mi_act @ J.T
    R = np.diag(np.array(d.efc_R)[rows])
    lam = float(np.max(np.linalg.eigvals(np.linalg.solve(A + R, A)).real))
    lam_act = float(np.max(np.linalg.eigvals(A_act @ np.linalg.inv(A + R)).real))
    return lam, lam_act


def hold(model, dt_ms, tc, d0, refsafe):
    """Settle 0.4 s with the weight compensated, hold 1 s. Returns the run's metrics."""
    dt = dt_ms * 1e-3
    rig = build(model, dt_ms, tc, d0, refsafe)
    rig.set_pad_force(N_PINCH)
    comp = np.array([0.0, 0.0, H.M_TOOL * H.G])
    out = {"n_steps": 0}
    W = []
    n_set = max(1, int(round(T_SETTLE / dt)))
    n_hold = max(1, int(round(T_HOLD / dt)))
    status, t_fail, why = "complete", None, None
    p_set = None
    nL, nR, vel, NS = [], [], [], []
    dp_max = 0.0
    for k in range(n_set + n_hold):
        rig.set_tool_wrench(comp if k < n_set else np.zeros(3), np.zeros(3))
        w0 = time.perf_counter()
        rig.step(dt)
        W.append(time.perf_counter() - w0)
        t = (k + 1) * dt
        if bad_state(rig):
            status, t_fail, why = "failed", t, "non-finite state or MuJoCo bad-state warning"
            break
        st = rig.tool_state()
        if k == n_set - 1:
            p_set = st["pos"].copy()
            nrm, fr, cnt = loaded_rows(rig)
            out["n_L_settle"], out["n_R_settle"] = cnt["L"], cnt["R"]
            lam, lam_act = lam_max(rig, nrm, dt)
            lam_t, lam_t_act = lam_max(rig, fr, dt)
            out.update(lam_n=lam, lam_n_act=lam_act, lam_t=lam_t, lam_t_act=lam_t_act)
            c = rig.contacts()
            out["N_settle"] = [c["L"]["N"], c["R"]["N"]]
        if k >= n_set:
            dp = float(np.linalg.norm(st["pos"] - p_set))
            dp_max = max(dp_max, dp)
            qa = rig.m.jnt_dofadr[rig.m.body_jntadr[rig.tool]]
            vel.append(float(np.linalg.norm(rig.d.qvel[qa:qa + 3])))
            c = rig.contacts()
            nL.append(c["L"]["n"]), nR.append(c["R"]["n"]), NS.append(c["L"]["N"] + c["R"]["N"])
            if dp > EJECT_M:
                status, t_fail, why = "ejected", t, "tool centre moved more than 5 mm"
                break
        out["n_steps"] = k + 1
    out["us_per_step_median"] = float(np.median(W) * 1e6)
    if status == "complete":
        if nL[-1] == 0 or nR[-1] == 0:
            status, why = "ejected", "a pad lost contact"
    last = slice(-max(1, int(round(0.5 / dt))), None)
    if vel:
        out.update(dp_max_mm=dp_max * 1e3, v_max_mm_s=max(vel) * 1e3,
                   v_rms_last05_mm_s=float(np.sqrt(np.mean(np.square(vel[last])))) * 1e3,
                   n_L_mean=float(np.mean(nL)), n_R_mean=float(np.mean(nR)),
                   n_L_end=int(nL[-1]), n_R_end=int(nR[-1]),
                   N_sum_mean_last05=float(np.mean(NS[last])), N_sum_sd_last05=float(np.std(NS[last])),
                   N_sum_max=float(np.max(NS)), N_sum_min=float(np.min(NS)))
    out.update(status=status, held=status == "complete", t_fail=t_fail, why=why)
    return out


def bounds(d0, tc, dt, n, refsafe):
    """Stability predictions at step dt for n rows on one body; tc in s."""
    tce = max(tc, 2 * dt) if refsafe else tc
    dp = n * d0 / (n * d0 + 1 - d0) if n > 0 else 0.0
    b = 2.0 / (d0 * tce)
    kimp = 1.0 / (d0 * tce ** 2)
    return {"tc_eff_ms": tce * 1e3, "d_coll": dp, "rate_x_dt": dp * b * dt,
            "damping_ok": bool(dp * b * dt < 2.0), "full_ok": bool(4 - 2 * dt * dp * b - dt * dt * dp * kimp > 0),
            "dt_max_damping_ms": (d0 * tce / dp * 1e3) if dp > 0 else None}


def dt_root(lam, d0, tce):
    """Largest dt with 4 - 2 dt lam b - dt^2 lam k imp > 0 for fixed tc_eff (s)."""
    b, kimp = 2.0 / (d0 * tce), 1.0 / (d0 * tce ** 2)
    a2, a1 = lam * kimp, 2 * lam * b
    return (-a1 + math.sqrt(a1 * a1 + 16 * a2)) / (2 * a2)


def run_combo(model, d0, tc_ms, refsafe, dts, out):
    have = CB.done(out, key=("case_id",))
    tc = tc_ms * 1e-3
    rows = []
    for dt_ms in dts:
        cid = f"{model}|{d0}|{tc_ms}|{dt_ms}|{int(refsafe)}"
        if (cid,) in have:
            rows += [r for r in CB.read_rows(out) if r.get("case_id") == cid]
            continue
        t0 = time.time()
        try:
            m = hold(model, dt_ms, tc, d0, refsafe)
        except Exception as e:
            m = {"status": "failed", "held": False, "why": repr(e), "traceback": traceback.format_exc()[-1500:]}
        dt = dt_ms * 1e-3
        row = {"task": "T7", "kind": "run", "model": model, "rig_spec": CB.MODELS[model][0], "N": N_PINCH,
               "dt_ms": dt_ms, "d0": d0, "tc_ms": tc_ms, "refsafe": refsafe, "case_id": cid,
               "solref": [tc, 1.0], "solimp": [d0, d0, 0.001, 0.5, 2.0], "gravity": "-z (across the tool axis)",
               "T_settle_s": T_SETTLE, "T_hold_s": T_HOLD}
        row.update(m)
        nL = m.get("n_L_settle") or 0
        nR = m.get("n_R_settle") or 0
        row["pred_simple_ok"] = bool(dt < d0 * (max(tc, 2 * dt) if refsafe else tc))
        row["pred_pad"] = bounds(d0, tc, dt, nL, refsafe)
        row["pred_both"] = bounds(d0, tc, dt, nL + nR, refsafe)
        tce = max(tc, 2 * dt) if refsafe else tc
        for key in ("lam_n_act", "lam_t_act"):
            lam = m.get(key)
            if lam:
                b = 2.0 / (d0 * tce)
                row[f"pred_{key}_damping_ok"] = bool(lam * b * dt < 2.0)
                row[f"pred_{key}_dt_max_damping_ms"] = d0 * tce / lam * 1e3
                if key == "lam_n_act":
                    row["pred_lam_n_act_full_ok"] = bool(4 - 2 * dt * lam * b - dt * dt * lam / (d0 * tce ** 2) > 0)
                    row["pred_lam_n_act_dt_max_full_ms"] = dt_root(lam, d0, tce) * 1e3
        row.update(film=None, script=SCRIPT, git_rev=CB.git_rev(), wall_s=time.time() - t0,
                   when=time.strftime("%Y-%m-%d %H:%M"))
        H.append_row(out, row)
        rows.append(row)
        print(model, d0, tc_ms, dt_ms, int(refsafe), row["status"], "n", nL, nR,
              "lam %.3f/%.3f" % (m.get("lam_n_act") or -1, m.get("lam_t_act") or -1),
              "pred simple %s pad %s" % (row["pred_simple_ok"], row["pred_pad"]["damping_ok"]),
              "us %.1f" % m.get("us_per_step_median", -1), flush=True)
    return rows


def summary(model, d0, tc_ms, refsafe, rows, out):
    """One row per (model, d0, tc, refsafe): largest step that held and the predicted bounds."""
    cid = f"summary|{model}|{d0}|{tc_ms}|{int(refsafe)}"
    if (cid,) in CB.done(out, key=("case_id",)):
        return
    rows = sorted(rows, key=lambda r: r["dt_ms"])
    held = [r["dt_ms"] for r in rows if r.get("held")]
    dt_max = max(held) if held else None
    first_fail = next((r["dt_ms"] for r in rows if not r.get("held")), None)
    ref = next((r for r in rows if r.get("n_L_settle")), {})
    tc = tc_ms * 1e-3
    nL, nR = ref.get("n_L_settle", 0), ref.get("n_R_settle", 0)
    s = {"task": "T7", "kind": "summary", "model": model, "rig_spec": CB.MODELS[model][0], "N": N_PINCH,
         "d0": d0, "tc_ms": tc_ms, "refsafe": refsafe, "case_id": cid, "dt_ms": None,
         "dt_tested_ms": [r["dt_ms"] for r in rows], "held": [bool(r.get("held")) for r in rows],
         "status_by_dt": {str(r["dt_ms"]): r["status"] for r in rows},
         "dt_max_held_ms": dt_max, "dt_first_fail_ms": first_fail,
         "monotone": bool(first_fail is None or dt_max is None or dt_max < first_fail),
         "n_L_ref": nL, "n_R_ref": nR, "lam_n_ref": ref.get("lam_n_act"), "lam_t_ref": ref.get("lam_t_act"),
         "pred_dt_max_simple_ms": d0 * tc_ms,
         "pred_dt_max_pad_ms": tc_ms * (d0 + (1 - d0) / nL) if nL else None,
         "pred_dt_max_both_ms": tc_ms * (d0 + (1 - d0) / (nL + nR)) if nL + nR else None,
         "pred_dt_max_lam_damping_ms": d0 * tc_ms / ref["lam_n_act"] if ref.get("lam_n_act") else None,
         "pred_dt_max_lam_full_ms": dt_root(ref["lam_n_act"], d0, tc) * 1e3 if ref.get("lam_n_act") else None,
         "pred_dt_max_friction_ms": d0 * tc_ms / ref["lam_t_act"] if ref.get("lam_t_act") else None,
         "pred_from_dt_list": {k: max([r["dt_ms"] for r in rows if f(r)], default=None) for k, f in (
             ("simple", lambda r: r["pred_simple_ok"]), ("pad_damping", lambda r: r["pred_pad"]["damping_ok"]),
             ("pad_full", lambda r: r["pred_pad"]["full_ok"]), ("both_damping", lambda r: r["pred_both"]["damping_ok"]),
             ("lam_full", lambda r: r.get("pred_lam_n_act_full_ok", False)))},
         "us_per_step_median": None, "status": "complete", "film": None, "script": SCRIPT, "git_rev": CB.git_rev(),
         "when": time.strftime("%Y-%m-%d %H:%M")}
    H.append_row(out, s)
    print("summary", model, d0, tc_ms, int(refsafe), "held up to", dt_max, "first fail", first_fail,
          "pred simple %.2f pad %s lam_full %s" % (s["pred_dt_max_simple_ms"], s["pred_dt_max_pad_ms"],
                                                   s["pred_dt_max_lam_full_ms"]), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--d0", nargs="+", type=float, default=D0S)
    ap.add_argument("--tc", nargs="+", type=float, default=TCS_MS, help="ms")
    ap.add_argument("--dt", nargs="+", type=float, default=DTS_MS, help="ms")
    ap.add_argument("--refsafe", nargs="+", type=int, default=[1, 0])
    ap.add_argument("--no-point", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    combos = [("mj_pads1", d0, tc, bool(rs)) for rs in a.refsafe for tc in a.tc for d0 in a.d0]
    if not a.no_point:
        combos += [("mj_point3", d0, tc, bool(rs)) for rs in a.refsafe for d0, tc in POINT]
    for model, d0, tc, rs in combos:
        rows = run_combo(model, d0, tc, rs, a.dt, a.out)
        if len(rows) == len(a.dt):
            summary(model, d0, tc, rs, rows, a.out)


if __name__ == "__main__":
    main()
