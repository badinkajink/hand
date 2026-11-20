"""The fingertip boards in the arm presses of 2026-11-19 evening (lab/arm_press.py): the arm pushes one pad into a plate
or the sphere's top while the finger's servos hold, the HEX-E reads the force along the press.

Per press (each rep of a run): the board's change d from its rest (15 components, uT; tip_contacts.Tip.fields, the
second before the press's start, the hand at the plan's end), in 0.1 s bins, against the HEX-E's force along the press
(F_p, N: the change from the same second, turned into B by R_BP R_PH, minus along the press direction; positive pushing).
Least squares d = g F_p + c over the bins of the loading and unloading (F_p over 0.05 N): g is the board's field per
newton (uT/N, 15), its fit's R^2, and the cosine of g with the tip's calibration holds (the R 5 sphere's near-normal
presses of 1-4 N, any place on the pad: 20261105-viewer_inputs/correction): the best match, its place and the
calibration's |dB|/N there (median of the 15 best). Beside it the trials' estimator (tip_contacts, GP and physical
model, with and without the own-pose term) at the bins' times: the slope of its f_n against F_p. And the joints' change
and the pressing finger's servo loads at the peak, for the motor-current hypothesis (a servo's current changing the
board's field).

    /usr/bin/python3 sim/20261119-arm_press_boards.py [RUN.jsonl ...] [--out data/s3/20261119-arm_press_boards.json]
        default: every logs/20261119-sr2_s3_press/*_arm_press.jsonl with a press that reached its target
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[2]
sys.path[:0] = [str(ROOT / "src/sr2hand/driver/reskin5x/host"), str(ROOT / "src"), str(ROOT / "src/sr2hand/driver/manta/host")]
import tip_contacts as TC  # noqa: E402
from manta_hand import plan as MP  # noqa: E402

LOGS = ROOT / "logs/20261119-sr2_s3_press"
MUXES = sorted((ROOT / "logs/20261119-three_finger_mux").glob("20261119-*-mux.bin"))
CORR = ROOT / "docs/experiments/20261022-ur5_tip_calibration/data/20261105-viewer_inputs/correction"
CAL = {1: CORR / "20261028-spherev3id1_twists", 2: CORR / "20261029-spherev3id2", 3: CORR / "20261030-spherev3id3"}
BIN = 0.1


def mux_for(t):
    import datetime as dt
    starts = [dt.datetime.strptime(f.name[:15], "%Y%m%d-%H%M%S").timestamp() for f in MUXES]
    k = max(i for i, s0 in enumerate(starts) if s0 <= t)
    return MUXES[k]


def servo33(sim):
    col = {"aa": 0, "fe1": 1, "fe2": 2}
    J = np.zeros((3, 3))
    for k, f in enumerate(TC.FINGERS):
        for j in ("yaw", "mcp", "pip"):
            J[k, col[MP.SIM_JOINT_TO_SERVO[j]]] = sim[f][j] * MP.JOINT_SIGN[(f, j)]
    return J


def cal_table(tip):
    z = np.load(CAL[tip] / "holds.npz", allow_pickle=True)
    ok = np.isfinite(z["d_board"]).all(1) & np.isfinite(z["p"]).all(1) & np.isfinite(z["f"]).all(1)
    f, n, p, B, k = z["f"][ok], z["n"][ok], z["p"][ok], z["d_board"][ok], z["kind"][ok].astype(str)
    fn = np.abs(np.sum(f * n, 1))
    ft = np.linalg.norm(f - np.sum(f * n, 1)[:, None] * n, axis=1)
    m = (ft < 0.15 * fn) & (fn >= 1.0) & (fn <= 4.0) & (k == "press")
    return p[m], B[m] / fn[m][:, None]


def fit_line(F, D):
    A = np.c_[F, np.ones(len(F))]
    sol, *_ = np.linalg.lstsq(A, D, rcond=None)
    pred = A @ sol
    ss_res = float(np.sum((D - pred) ** 2))
    ss_tot = float(np.sum((D - D.mean(0)) ** 2)) + 1e-12
    return sol[0], sol[1], 1.0 - ss_res / ss_tot


def analyse(run, cal):
    lines = [json.loads(x) for x in open(run) if x.strip()]
    head = lines[0]
    reps = [x for x in lines[1:] if "rep" in x]
    raws = {json.loads(x)["rep"]: json.loads(x) for x in open(str(run).replace(".jsonl", "_raw.jsonl")) if x.strip()}
    f = head["finger"]
    tip_id = TC.TIP_ID[f]
    Cp, CBn = cal[tip_id]
    d_B = np.asarray(head["press_dir_B"], float)
    R_BH = np.asarray(head["T_BP_end"], float)[:3, :3] @ np.asarray(head["R_PH"], float)
    out = []
    for rep in reps:
        raw = raws.get(rep["rep"])
        if raw is None:
            continue
        t0, t1 = rep["t_start"], rep["t_end"]
        th = np.asarray(raw["hexe_t"], float)
        Fh = np.asarray(raw["hexe"], float)[:, :3]
        F0 = np.asarray(rep["F0"], float)
        Fp = -((Fh - F0) @ R_BH.T) @ d_B
        drift = 0.0
        if rep.get("F_after") is not None:              # the HEX-E's drift (z: about 0.4 N a minute after an arm move),
            drift = float(-((np.asarray(rep["F_after"], float) - F0) @ R_BH.T) @ d_B)   # taken off linearly in time
            Fp = Fp - drift * np.clip((th - t0) / max(t1 - t0, 1e-6), 0.0, 1.0)
        st = raw["station"]
        ts = np.array([x[0] for x in st])
        J = np.array([[x[1][k] for k in ("yaw", "mcp", "pip")] for x in st], float)
        Ld = np.array([[x[2][k] if x[2][k] is not None else np.nan for k in ("yaw", "mcp", "pip")] for x in st], float)
        others = head["joints"]
        feed = TC.MuxLog(mux_for(t0), t0 - 2.0 - TC.KEEP_S, t1 + 1.0)

        def joints(t):
            q = {g: dict(others[g]) for g in TC.FINGERS}
            q[f] = {j: float(np.interp(t, ts, J[:, i])) for i, j in enumerate(("yaw", "mcp", "pip"))}
            return servo33(q)

        est = {}
        for name, (gp, own) in {"gp_own_pose": (True, True), "gp": (True, False), "physical": (False, False)}.items():
            tc = TC.TipContacts(feed, np.zeros(6), gp=gp, own_pose=own)
            tip = tc.tips[f]
            tip.threshold, tip.min_force = -1e9, -1e9
            tc.rest(t0 - 0.1, 0.9, refit=True, joints_servo=joints(t0 - 0.5))
            est[name] = tc
        tip = est["gp"].tips[f]
        bins = np.arange(t0, t1, BIN)
        rows = []
        for b in bins:
            m = (th >= b) & (th < b + BIN)
            if m.sum() < 3:
                continue
            tt, rr = feed.window(f, b + BIN, BIN)
            ok = TC.valid(rr)
            if ok.sum() < 3:
                continue
            d = tip.fields(rr[ok], rr[ok]).mean(0) - tip.base
            row = dict(t=float(b - t0), F=float(Fp[m].mean()), d=d)
            row["q"] = [float(np.interp(b + BIN / 2, ts, J[:, i])) for i in range(3)]
            row["load"] = [float(np.interp(b + BIN / 2, ts, np.nan_to_num(Ld[:, i]))) for i in range(3)]
            rows.append(row)
        F = np.array([r["F"] for r in rows])
        D = np.array([r["d"] for r in rows])
        on = F > 0.05
        if on.sum() < 10:
            out.append(dict(rep=rep["rep"], error="too few loaded bins"))
            continue
        g, c, r2 = fit_line(F[on], D[on])
        cos = CBn @ g / (np.linalg.norm(CBn, axis=1) * np.linalg.norm(g) + 1e-12)
        top = np.argsort(-cos)[:15]
        cal_mag = float(np.median(np.linalg.norm(CBn[top], axis=1)))
        # the estimators at every 3rd loaded bin
        ests = {}
        pk = int(np.argmax(F))
        for name, tc in est.items():
            xs, ys, ws, ps = [], [], [], []
            for i in np.flatnonzero(on)[::3]:
                tb = t0 + rows[i]["t"] + BIN
                e = tc.tips[f].estimate(*feed.window(f, tb, TC.KEEP_S), TC.TipContacts._q(joints(tb)).get(f))
                if e.get("contact"):
                    xs.append(F[i])
                    ys.append(e["f_n"])
                    ws.append(e.get("w"))
                    ps.append(e["p"])
            if len(xs) >= 5:
                A = np.c_[xs, np.ones(len(xs))]
                (sl, off), *_ = np.linalg.lstsq(A, np.array(ys), rcond=None)
                ests[name] = dict(slope=round(float(sl), 3), offset_N=round(float(off), 3), n=len(xs),
                                  w_med=None if ws[0] is None else round(float(np.median(ws)), 3),
                                  p_med=np.round(np.median(ps, 0), 2).tolist())
        rec = dict(rep=rep["rep"], indenter=head["indenter"], finger=f, tip=tip_id, peak_N=round(float(F.max()), 3),
                   hexe_drift_N=round(drift, 3),
                   n_bins=int(on.sum()), field_per_N_uT=round(float(np.linalg.norm(g)), 2),
                   offset_uT=round(float(np.linalg.norm(c)), 2), r2=round(r2, 4), g=np.round(g, 3).tolist(),
                   best_cos=round(float(cos[top[0]]), 4), best_p=np.round(Cp[top[0]], 2).tolist(),
                   top15_cos=round(float(np.median(cos[top])), 4), top15_p=np.round(np.median(Cp[top], 0), 2).tolist(),
                   cal_field_per_N_uT=round(cal_mag, 2), ratio=round(float(np.linalg.norm(g)) / cal_mag, 3),
                   joints_change_at_peak=np.round(np.array(rows[pk]["q"]) - np.array(rows[0]["q"]), 2).tolist(),
                   loads_at_peak=np.round(rows[pk]["load"], 0).tolist(), estimators=ests)
        out.append(rec)
        print(f"{Path(run).name} rep {rec['rep']} {f} ({rec['indenter']}): peak {rec['peak_N']} N; field per N "
              f"{rec['field_per_N_uT']} uT/N (R^2 {rec['r2']}); best calibration match cos {rec['best_cos']} at "
              f"{rec['best_p']} (top 15 {rec['top15_cos']} at {rec['top15_p']}), calibration there "
              f"{rec['cal_field_per_N_uT']} uT/N -> ratio {rec['ratio']}; joints moved {rec['joints_change_at_peak']}, "
              f"loads {rec['loads_at_peak']}; estimators " + "; ".join(f"{k} {v['slope']} (+{v['offset_N']}, w "
                                                                       f"{v['w_med']})" for k, v in ests.items()),
              flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--out", default=str(STUDY / "data/s3/20261119-arm_press_boards.json"))
    a = ap.parse_args()
    runs = a.runs or [str(p) for p in sorted(LOGS.glob("20261119-*_arm_press.jsonl"))]
    cal = {t: cal_table(t) for t in CAL}
    res = []
    for r in runs:
        lines = [json.loads(x) for x in open(r) if x.strip()]
        if not any(x.get("stops", [None, None])[1] == "target" for x in lines[1:] if "rep" in x):
            continue
        res.append({"run": Path(r).name, "header": {k: v for k, v in lines[0].items() if k not in ("R_PH", "T_BP_end")},
                    "presses": analyse(r, cal)})
    Path(a.out).write_text(json.dumps({"what": __doc__.split("\n\n")[0].replace("\n", " "), "runs": res},
                                      separators=(",", ":")) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
