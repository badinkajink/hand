"""The fingertip boards' own-pose GP refitted on the lab's free-air sweep of 2026-11-17 (W2 of the identification queue;
owner, 18:0x: "a free-air sweep of the fingertip boards at random poses ... for the own-pose GP at the trial layout").

The sweep (lab, 20:10-20:53, plan logs/20261117-sr2_boards_sweep/20261117-plan_random_s3.json): 300 random poses of all
nine joints, 3 s each, and a zero pose of 8 s before every eleventh, at the roll layout (85, 170) at 7.5 V, after the
owner's re-zeroing of 16:1x. Its events are in data/sweep/, its telemetry in logs/20261117-sr2_boards_sweep/s3/ and the
boards' records in the splitter's log logs/20261117-three_finger_mux/20261117-180517-mux.bin (gitignored).

The rows are built by scripts/fingertip/sr2_crosstalk_hand.py, as for the sweep of 2026-11-04/05: each dwell's mean
reading per board (thermally corrected) less the zero pose's, interpolated in time between the zero dwells (steps,
dwell_means, baselines), and the dwells held still (own_models: no joint moving over 2 deg, the board's spread under
25 uT). own_models also gives the five-fold error of each board's GP on its own finger's three angles. The rows go to
data/sweep/20261117-own_pose_rows_s3.npz in that script's format (q, <finger>_rows, <finger>_Y), and
tip_contacts.fit_own_pose fits them to a new file beside the trials' default (tip_contacts.OWN_POSE is left as it is).

The trials' GP (tip_contacts.OWN_POSE, fitted on 2026-11-04/05 at 5 V before the re-zeroing) is tested on the new rows
as the estimator uses it: the predicted change between the zero pose and the pose, P(q) - P(0). The same test is
repeated with a joint offset per finger fitted to the new rows, to see how much of the old GP's error a shift of the
joint zeros explains.

    ~/.claude/bin/resguard.sh run --mem 3G --cpu 300 -- env NUMPY_MADVISE_HUGEPAGE=0 OPENBLAS_NUM_THREADS=2 \
        .venv/bin/python docs/experiments/20261117-sr2_system_identification/sim/20261117-own_pose_sweep_fit.py

The sweep of 2026-11-19 (lab, 15:12-15:55, at the trials' layout (70, 110), plan data/sweep/
20261119-plan_random_x70y110.json) runs with --sweep 20261119: the same steps on its own files, and both earlier GPs
(the trials' and the 11-17 sweep's) tested on its rows.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "scripts/fingertip"), str(ROOT / "src/sr2hand/driver/reskin5x/host")]
import _paths  # noqa: E402,F401
import sr2_crosstalk_hand as XH  # noqa: E402
import tip_contacts as TC  # noqa: E402

EXP = HERE.parent
SWEEPS = {
    "20261117": dict(run=ROOT / "logs/20261117-sr2_boards_sweep/s3",
                     mux=ROOT / "logs/20261117-three_finger_mux/20261117-180517-mux.bin",
                     rows=EXP / "data/sweep/20261117-own_pose_rows_s3.npz",
                     out=EXP / "data/sweep/20261117-own_pose_sweep_fit.json",
                     gp_new=TC.OWN_POSE.parent / "20261117-own_pose_gp_sweep_x85y170_7v5.npz",
                     layout="roll (85, 170)",
                     source="lab's boards sweep, 2026-11-17 20:10-20:53 (327 steps: 300 random poses, 27 zero poses)"),
    "20261119": dict(run=ROOT / "logs/20261119-sr2_boards_sweep/s1",
                     mux=ROOT / "logs/20261119-three_finger_mux/20261119-142714-mux.bin",
                     rows=EXP / "data/sweep/20261119-own_pose_rows_x70y110.npz",
                     out=EXP / "data/sweep/20261119-own_pose_sweep_fit.json",
                     gp_new=TC.OWN_POSE.parent / "20261119-own_pose_gp_sweep_x70y110_7v5.npz",
                     layout="trials (70, 110)",
                     source="lab's boards sweep, 2026-11-19 15:12-15:55 (327 steps), after the afternoon's presses")}
GP_11_17 = SWEEPS["20261117"]["gp_new"]
OLD_JSON = ROOT / "docs/experiments/20261104-sr2_hand_crosstalk/data/20261105-sr2_crosstalk_hand_20261104_20261105.json"
F = XH.F


def pct(v, p):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return round(float(np.percentile(v, p)), 2) if len(v) else None


def errs(e, upn):
    return dict(median_uT=pct(e, 50), p95_uT=pct(e, 95), median_N=round(pct(e, 50) / upn, 3),
                p95_N=round(pct(e, 95) / upn, 3))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--no-offset", action="store_true", help="skip the joint-offset test of the old GP")
    ap.add_argument("--sweep", default="20261117", choices=list(SWEEPS))
    a = ap.parse_args()
    sw = SWEEPS[a.sweep]
    RUN, MUX, ROWS, OUT, GP_NEW = sw["run"], sw["mux"], sw["rows"], sw["out"], sw["gp_new"]
    t0 = time.time()
    old = json.loads(OLD_JSON.read_text())
    upn = old["uT_per_N"]                                  # each tip's change per newton on the UR5 (11-05's values)
    cfg = dict(dir=RUN.parent, runs=[RUN.name], logs=[MUX], index_lost=None)
    S = XH.steps(cfg, a.sweep)
    XH.dwell_means(cfg, S)
    print(f"{len(S)} steps read in {time.time() - t0:.0f} s", flush=True)
    names = XH.layout_name(S)
    Z, noise = XH.baselines(S)
    own, pred = XH.own_models(S, upn, names)
    print(f"own models: {time.time() - t0:.0f} s", flush=True)
    mounts = {f: [round(float(np.median([s["mounts"][f][c] for s in S])), 2) for c in (0, 1)] for f in F}
    np.savez_compressed(ROWS, **{f"{f}_{k}": v for f, pr in pred.items() for k, v in
                                 (("rows", np.array(pr["rows"])), ("Y", pr["Y"]),
                                  ("P_own_angles", pr["P"]["own_angles"]))},
                        q=np.array([s["q"] for s in S]), q_cmd=np.array([s["q_cmd"] for s in S]),
                        t=np.array([0.5 * (s["t0"] + s["t1"]) for s in S]),
                        mounts=np.array([[s["mounts"][f][c] for f in F for c in (0, 1)] for s in S]),
                        family=np.array([s["family"] for s in S]), step=np.array([s["step"] for s in S]))
    info = TC.fit_own_pose(out=GP_NEW, rows=ROWS)
    print(f"fitted -> {GP_NEW}: {info}", flush=True)
    gp_old, gp_new = TC.OwnPose(TC.OWN_POSE), TC.OwnPose(GP_NEW)
    gp17 = TC.OwnPose(GP_11_17) if a.sweep != "20261117" and GP_11_17.exists() else None
    z = np.load(ROWS)
    q = z["q"]
    res = {}
    for k, f in enumerate(F):
        if f"{f}_rows" not in z.files:
            continue
        X, Y = q[z[f"{f}_rows"]][:, 3 * k:3 * k + 3], z[f"{f}_Y"]
        P0 = gp_old.predict(f, np.zeros(3))
        Po = np.array([gp_old.predict(f, x) for x in X])
        r = dict(rows=int(len(X)), change=errs(np.linalg.norm(Y, axis=1), upn[f]),
                 five_folds_new=own[f]["own_angles"]["five_folds"],
                 five_folds_11_04_05=old["own"][f]["own_angles"]["five_folds"],
                 left_out_roll_85_170_11_05=old["own"][f]["own_angles"].get("left_out hom_roll_85_170"),
                 old_gp_on_new=errs(np.linalg.norm(Y - (Po - P0), axis=1), upn[f]),
                 old_gp_on_new_raw=errs(np.linalg.norm(Y - Po, axis=1), upn[f]),
                 old_gp_at_zero_uT=round(float(np.linalg.norm(P0)), 2),
                 new_gp_in_sample=errs(np.linalg.norm(Y - np.array([gp_new.predict(f, x) for x in X]), axis=1),
                                       upn[f]),
                 gp_11_17_on_new=(errs(np.linalg.norm(Y - (np.array([gp17.predict(f, x) for x in X])
                                                           - gp17.predict(f, np.zeros(3))), axis=1), upn[f])
                                  if gp17 is not None else None),
                 joint_ranges_deg=np.round(np.c_[X.min(0), X.max(0)], 1).tolist(),
                 left_out_moving=own[f]["left_out_moving"], gp=info.get(f))
        if not a.no_offset:
            from scipy.optimize import least_squares

            def resid(d, X=X, Y=Y, f=f):
                p0 = gp_old.predict(f, d)
                return (Y - (np.array([gp_old.predict(f, x + d) for x in X]) - p0)).ravel()

            fit = least_squares(resid, np.zeros(3), loss="soft_l1", f_scale=10.0)
            e = np.linalg.norm(fit.fun.reshape(-1, 15), axis=1)
            r["old_gp_with_joint_offset"] = dict(offset_deg=np.round(fit.x, 2).tolist(), **errs(e, upn[f]))
        res[f] = r
        print(f"{f}: {r['rows']} rows, change median {r['change']['median_uT']} uT; five folds new "
              f"{r['five_folds_new']['median_uT']} (11-04/05 {r['five_folds_11_04_05']['median_uT']}); old GP on new "
              f"{r['old_gp_on_new']['median_uT']} (raw {r['old_gp_on_new_raw']['median_uT']})"
              + (f"; 11-17 sweep GP {r['gp_11_17_on_new']['median_uT']}" if r.get("gp_11_17_on_new") else "")
              + (f"; with offset {r['old_gp_with_joint_offset']['offset_deg']} deg "
                 f"{r['old_gp_with_joint_offset']['median_uT']}" if "old_gp_with_joint_offset" in r else ""), flush=True)
    out = dict(source=sw["source"], layout=sw["layout"], mounts_P_mm=mounts, run=str(RUN.relative_to(ROOT)),
               mux=str(MUX.relative_to(ROOT)), rows=str(ROWS.relative_to(ROOT)), gp_new=str(GP_NEW.relative_to(ROOT)),
               gp_trials=str(TC.OWN_POSE.relative_to(ROOT)), script=str(Path(__file__).relative_to(ROOT)),
               uT_per_N=upn, zero_noise=noise, steps=len(S),
               terms=dict(change="|d|, a board's 15 components less the zero pose's (uT)",
                          five_folds="the GP on the own finger's yaw, MCP and PIP fitted on four fifths of the rows and "
                                     "tested on the fifth, |predicted - measured| (uT, and N by the tip's uT per N)",
                          old_gp_on_new="the trials' GP (11-04/05) on the new rows, P(q) - P(0) against d",
                          old_gp_with_joint_offset="the same with one offset per joint added to the measured angles, "
                                                   "fitted to the new rows (soft L1)"),
               boards=res, seconds=round(time.time() - t0, 1))
    OUT.write_text(json.dumps(out, indent=1))
    print("->", OUT)


if __name__ == "__main__":
    main()
