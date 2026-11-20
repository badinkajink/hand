"""How smooth, steady and fast the fingertip boards' contact estimate is on the SR2 hand (the owner, 2026-11-19 18:4x:
"what matters more is pose sensing smoothness, stability, throughput").

The arm presses of 2026-11-19 (lab/arm_press.py) held each pad against the plate for 2 s at 2.1-2.5 N with the arm
stopped and the servos holding, and the second before each press had the pad in free air. For each press, from the
splitter's log (logs/20261119-three_finger_mux, 160.6 records a second with all three boards) through the trials'
estimator (tip_contacts: thermal correction, the rest from the second before the press, the pooled board GP, without
the own-pose term since the joints moved under 1.5 deg):
  rate      the pressing board's records per second and its longest gap in the press
  noise     the field's per-component standard deviation in the free-air second, per record and over 50 ms means
            (the estimator's window), uT
  jitter    the place's standard deviation (along z and across y, mm) and the force's (N) in the hold, per record and
            per 50 ms mean, after a straight-line fit in time (the drift, mm/s, reported apart)
  settle    the place's path while the force rises: the force above which the place stays within 0.5 mm of the hold's
Tells what a pose filter must smooth and how fast it can run.

    /usr/bin/python3 sim/20261119-board_estimate_stability.py [--out data/s3/20261119-board_estimate_stability.json]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[2]
sys.path[:0] = [str(ROOT / "src/sr2hand/driver/reskin5x/host"), str(ROOT / "src")]
import tip_contacts as TC  # noqa: E402

LOGS = ROOT / "logs/20261119-sr2_s3_press"
MUXES = sorted((ROOT / "logs/20261119-three_finger_mux").glob("20261119-*-mux.bin"))
RUNS = {"20261119-170343_index_arm_press": "index, PIP 0", "20261119-170846_middle_arm_press": "middle, PIP 0",
        "20261119-171311_thumb_arm_press": "thumb, PIP 0", "20261119-171923_index_arm_press": "index, PIP 10",
        "20261119-172508_index_arm_press": "index, PIP 20"}
WIN = TC.WIN_S


def mux_for(t):
    import datetime as dt
    starts = [dt.datetime.strptime(f.name[:15], "%Y%m%d-%H%M%S").timestamp() for f in MUXES]
    return MUXES[max(i for i, s0 in enumerate(starts) if s0 <= t)]


def detrended_sd(t, x):
    """(sd about a straight line in time, the line's slope) per column."""
    A = np.c_[t - t.mean(), np.ones(len(t))]
    sol, *_ = np.linalg.lstsq(A, x, rcond=None)
    return np.std(x - A @ sol, axis=0), sol[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(STUDY / "data/s3/20261119-board_estimate_stability.json"))
    a = ap.parse_args()
    gp = TC.BoardGP()
    rows = []
    for stem, label in RUNS.items():
        L = [json.loads(x) for x in (LOGS / f"{stem}.jsonl").read_text().splitlines() if x.strip()]
        h = L[0]
        f = h["finger"]
        raws = {json.loads(x)["rep"]: json.loads(x) for x in (LOGS / f"{stem}_raw.jsonl").read_text().splitlines()
                if x.strip()}
        R_BH = np.asarray(h["T_BP_end"], float)[:3, :3] @ np.asarray(h["R_PH"], float)
        d_B = np.asarray(h["press_dir_B"], float)
        for rep in [x for x in L[1:] if "rep" in x and x.get("t_hold")]:
            t0, th0, t1 = rep["t_start"], rep["t_hold"], rep["t_end"]
            feed = TC.MuxLog(mux_for(t0), t0 - 2.0 - TC.KEEP_S, t1 + 1.0)
            tc = TC.TipContacts(feed, np.zeros(6), gp=True, own_pose=False)
            tip = tc.tips[f]
            tc.rest(t0 - 0.1, 0.9, refit=True)
            raw = raws[rep["rep"]]
            th = np.asarray(raw["hexe_t"], float)
            Fp = -((np.asarray(raw["hexe"], float)[:, :3] - np.asarray(rep["F0"], float)) @ R_BH.T) @ d_B
            # every record of the press, its field change, and the GP's estimate
            T, Rr = feed.window(f, t1 + 0.5, t1 + 0.5 - (t0 - 1.5))
            ok = TC.valid(Rr)
            T, Rr = T[ok], Rr[ok]
            _, rT = feed.window(f, t0, TC.KEEP_S)
            D = tip.fields(Rr, rT) - tip.base
            dt = np.diff(T)
            free = (T >= t0 - 1.1) & (T < t0 - 0.1)
            hold = (T >= th0 + 0.2) & (T <= th0 + 2.0)
            # 50 ms means on a 50 ms grid
            edges = np.arange(T[0], T[-1], WIN)
            k = np.digitize(T, edges)
            M = np.array([D[k == i].mean(0) for i in range(1, len(edges)) if (k == i).sum() >= 3])
            Mt = np.array([T[k == i].mean() for i in range(1, len(edges)) if (k == i).sum() >= 3])
            est = np.array([np.r_[gp.predict(d)[0], gp.predict(d)[1]] for d in D[hold]])
            estm = np.array([np.r_[gp.predict(d)[0], gp.predict(d)[1]] for d in M[(Mt >= th0 + 0.2) & (Mt <= th0 + 2.0)]])
            fn = lambda E: -np.sum(E[:, 3:] * np.array([1.0, 0.0, 0.0]), 1)   # noqa: E731 (x out of the face)
            sd_rec, slope_rec = detrended_sd(T[hold], est[:, [1, 2]])
            mt = Mt[(Mt >= th0 + 0.2) & (Mt <= th0 + 2.0)]
            sd_m, slope_m = detrended_sd(mt, estm[:, [1, 2]])
            fsd_rec = float(np.std(np.linalg.norm(est[:, 3:], axis=1)))
            fsd_m = float(np.std(np.linalg.norm(estm[:, 3:], axis=1)))
            # settle: the place along z of 50 ms means while the HEX-E's force rises to the hold
            load = (Mt >= t0) & (Mt <= th0)
            Fm = np.interp(Mt[load], th, Fp)
            zl = np.array([gp.predict(d)[0][2] for d in M[load]])
            zh = float(np.median(estm[:, 2]))
            off = np.abs(zl - zh) > 0.5
            settle = float(Fm[off].max()) if off.any() else float(Fm.min()) if len(Fm) else None
            row = dict(run=stem, group=label, rep=rep["rep"], finger=f,
                       rate_hz=round(float(1.0 / np.median(dt)), 1), longest_gap_ms=round(float(dt.max() * 1e3), 1),
                       n_hold_records=int(hold.sum()),
                       noise_free_uT_record=round(float(np.median(np.std(D[free], axis=0))), 3),
                       noise_free_uT_50ms=round(float(np.median(np.std(M[(Mt >= t0 - 1.1) & (Mt < t0 - 0.1)],
                                                                       axis=0))), 3),
                       hold_force_hexe_N=round(float(np.interp(th0 + 1.0, th, Fp)), 3),
                       hold_place_z_mm=round(zh, 2),
                       jitter_y_z_mm_record=np.round(sd_rec, 3).tolist(), jitter_y_z_mm_50ms=np.round(sd_m, 3).tolist(),
                       drift_y_z_mm_per_s=np.round(slope_m, 3).tolist(),
                       force_sd_N_record=round(fsd_rec, 3), force_sd_N_50ms=round(fsd_m, 3),
                       settle_force_N=None if settle is None else round(settle, 2),
                       loading_path=[[round(float(x), 2), round(float(y), 2)] for x, y in zip(Fm[::2], zl[::2])])
            rows.append(row)
            print(f"{label:14s} rep {rep['rep']}: {row['rate_hz']} Hz (gap {row['longest_gap_ms']} ms); free-air noise "
                  f"{row['noise_free_uT_record']} uT/record, {row['noise_free_uT_50ms']} uT/50 ms; hold z {zh:.1f} mm at "
                  f"{row['hold_force_hexe_N']} N: jitter y,z {row['jitter_y_z_mm_record']} mm/record, "
                  f"{row['jitter_y_z_mm_50ms']} mm/50 ms, drift {row['drift_y_z_mm_per_s']} mm/s; force sd "
                  f"{row['force_sd_N_50ms']} N/50 ms; place within 0.5 mm above {row['settle_force_N']} N", flush=True)
    Path(a.out).write_text(json.dumps({"what": __doc__.split("\n\n")[0].replace("\n", " "),
                                       "script": str(Path(__file__).resolve().relative_to(ROOT)), "rows": rows}))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
