"""The fingertip pads against the tracked sphere during the grasp of the night's g0 twists (W2 of the identification
queue, 2026-11-18): does the hand's model put its pads on the sphere where the hand's pads are?

For each of the 20 scored trials of data/20261118-night_trials_replay.json, every server row in two windows sets the
replay plant's finger joints to the hand's measured joints (hom_bench.sim_from_servo) and the sphere's centre to the
tracked one (P is the world frame shifted to the palm); the plant's kinematics give each finger's 1,018 pad spheres
(0.75 mm). A finger's gap is the least distance from a pad sphere's surface to the sphere's surface (mm; negative: the
model's pads are inside the sphere), its median over the window's rows. The windows:
  close    0.60-0.80 s: HOM's close, targets a few degrees past the sphere; the fingers stop where they meet it
  squeeze  1.30-1.58 s: the squeeze before the lift at 1.6 s, the index and middle yaw targets 10-15 deg past the sphere
On the hand the pads touch the sphere in both, compressed by about a millimetre at 1-3 N. Each window also gives the
boards' normal forces (the server's 'contacts', N; 0 when a board reports none), the joints and targets (simulator
degrees), the gap with the S1(b) joint zeros (data/s1/20261118-s1b_tag_fit.json, model 'both'), and the gap's change
for +5 deg at each joint.

    HOS_PLANT=kp0.55_kv0_fr0.18_fl0_dp0.011 SCALE_HAND=cad_v2 OMP_NUM_THREADS=1 \\
        .venv/bin/python sim/20261118-grasp_pad_gap.py [--out data/20261118-grasp_pad_gap.json]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[2]
sys.path.insert(0, str(ROOT / "logs/20261117-g13_open_pose"))
import night_replay as NR  # noqa: E402

FINGERS = ("thumb", "index", "middle")
JOINTS = ("yaw", "mcp", "pip")
WINDOWS = {"close": (0.60, 0.80), "squeeze": (1.30, 1.58)}


def main():
    import mujoco
    import hom_trial_tracking as TT
    from sr2hand.control import hom_bench as HB
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(STUDY / "data/20261118-grasp_pad_gap.json"))
    a = ap.parse_args()
    NR.RPL.RG.gear_class = NR.RPL.RC2.clutch_class
    NR.RPL.RC2.STIFF, NR.RPL.RC2.BREAKAWAY = True, 0.0
    NR.RPL.RG.RF.RL.rows_file()
    NR.RPL.RG.init(str(NR.RPL.RG.RF.SCENE))
    plant, setup = NR.make_plant("pads", "7v5")
    m, d = plant.m, mujoco.MjData(plant.m)
    palm = np.asarray(setup["palm"], float)
    pads = {f: np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(f"{f}_tip").id
                         and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE]) for f in FINGERS}
    r_pad = float(m.geom_size[pads["index"][0]][0])
    R = float(max(m.geom_size[g][0] for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE))
    qadr = {(f, j): m.jnt_qposadr[m.joint(f"{f}_{j}").id] for f in FINGERS for j in JOINTS}
    fit = json.loads((STUDY / "data/s1/20261118-s1b_tag_fit.json").read_text())["sets"]
    corr = {f: fit[f]["models"]["both"]["params"] for f in FINGERS}

    def gaps(q, c, dq=None):
        d.qpos[:] = m.qpos0
        for (f, j), adr in qadr.items():
            d.qpos[adr] = q[f][JOINTS.index(j)] + (math.radians(dq[f].get(j, 0.0)) if dq else 0.0)
        mujoco.mj_kinematics(m, d)
        return {f: 1e3 * (float(np.min(np.linalg.norm(d.geom_xpos[pads[f]] - c, axis=1))) - R - r_pad)
                for f in FINGERS}

    def window(rows, lo, hi):
        win = [r for r in rows if lo <= float(r["t"]) <= hi and r.get("obj_pos_mm") is not None]
        acc = {k: {f: [] for f in FINGERS} for k in ("built", "zeros", "force_N")}
        sens = {f: {j: [] for j in JOINTS} for f in FINGERS}
        qs, ts = [], []
        for r in win:
            q = HB.sim_from_servo(r["joints"])
            c = palm + np.asarray(r["obj_pos_mm"], float) / 1e3
            qs.append([math.degrees(q[f][i]) for f in FINGERS for i in range(3)])
            ts.append(list(r["targets_sim_deg"]))
            g0 = gaps(q, c)
            g1 = gaps(q, c, dq=corr)
            for f in FINGERS:
                acc["built"][f].append(g0[f])
                acc["zeros"][f].append(g1[f])
                acc["force_N"][f].append(float((r.get("contacts") or {}).get(f, [None, 0.0])[1]))
            for j in JOINTS:
                gj = gaps(q, c, dq={f: {j: 5.0} for f in FINGERS})
                for f in FINGERS:
                    sens[f][j].append(gj[f] - g0[f])
        names = [f"{f}.{j}" for f in FINGERS for j in JOINTS]
        return {"rows": len(win), **{k: {f: round(float(np.median(v[f])), 2) for f in FINGERS} for k, v in acc.items()},
                "joints_deg": dict(zip(names, np.median(qs, axis=0).round(1).tolist())),
                "targets_deg": dict(zip(names, np.median(ts, axis=0).round(1).tolist())),
                "per_5deg": {f: {j: round(float(np.median(sens[f][j])), 2) for j in JOINTS} for f in FINGERS}}

    rep = json.loads((STUDY / "data/20261118-night_trials_replay.json").read_text())
    runs = ROOT / "docs/experiments/20261115-hom_bench_trials/runs"
    res = []
    for t in rep["trials"]:
        fn = next(runs.glob(f"{t['stamp']}_twist_d40_trial*.jsonl"))
        head, rows, _ = TT.load_server(fn)
        row = {"stamp": t["stamp"], "tips": t["tips"], "file": str(fn.relative_to(ROOT)),
               **{w: window(rows, *lim) for w, lim in WINDOWS.items()}}
        res.append(row)
        print(f"{t['stamp']} tips {t['tips']:3s}: " + " | ".join(
            f"{w} gap " + ", ".join(f"{row[w]['built'][f]:+5.1f}" for f in FINGERS) + " F "
            + ", ".join(f"{row[w]['force_N'][f]:.2f}" for f in FINGERS) + " yaw "
            + ", ".join(f"{row[w]['joints_deg'][f + '.yaw']:+5.1f}/{row[w]['targets_deg'][f + '.yaw']:+5.1f}"
                        for f in ("index", "middle")) for w in WINDOWS), flush=True)

    def mmm(vals):
        return [round(float(np.median(vals)), 2), round(float(np.min(vals)), 2), round(float(np.max(vals)), 2)]
    summ = {w: {k: {f: mmm([r[w][k][f] for r in res]) for f in FINGERS} for k in ("built", "zeros", "force_N")}
            for w in WINDOWS}
    for w in WINDOWS:
        summ[w]["joints_deg"] = {k: round(float(np.median([r[w]["joints_deg"][k] for r in res])), 1)
                                 for k in res[0][w]["joints_deg"]}
        summ[w]["targets_deg"] = {k: round(float(np.median([r[w]["targets_deg"][k] for r in res])), 1)
                                  for k in res[0][w]["targets_deg"]}
        summ[w]["per_5deg"] = {f: {j: round(float(np.median([r[w]["per_5deg"][f][j] for r in res])), 2)
                                   for j in JOINTS} for f in FINGERS}
    summ["yaw_advance_close_to_squeeze_deg"] = {
        f: mmm([r["squeeze"]["joints_deg"][f + ".yaw"] - r["close"]["joints_deg"][f + ".yaw"] for r in res])
        for f in FINGERS}
    print(json.dumps(summ, indent=1))
    out = {"what": __doc__.split("\n\n")[0].replace("\n", " "), "script": str(Path(__file__).relative_to(ROOT)),
           "windows_s": WINDOWS, "sphere_radius_mm": R * 1e3, "pad_radius_mm": r_pad * 1e3,
           "s1b_zeros": {f: {j: corr[f][j] for j in JOINTS} for f in FINGERS},
           "summary_median_min_max": summ, "trials": res}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
