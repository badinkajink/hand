"""How well the own-pose GPs predict the boards in the per-layout free-air sweeps of 2026-11-19 night
(docs/experiments/20261115-hom_bench_trials/lab/pose_sweep.py; the owner, relayed 18:1x: "a short free-air pose sweep
at each layout ... just to see if it actually matters").

Each sweep holds 12 poses for 3 s at one layout, nothing touching. Per finger and pose: the board's mean field over the
hold (the trials' thermal correction, tip_contacts.Tip.fields) less its mean over the same sweep's zero-pose hold, against
each own-pose GP's prediction for the same change (its value at the pose's measured joints less at the zero hold's):
  deployed  tip_contacts.OWN_POSE, the 11-04/05 fit over many layouts, which the trials load
  x70y110   the refit of 11-19 afternoon on a sweep at (70, 110) (20261119-own_pose_gp_sweep_x70y110_7v5.npz)
The residual's size (uT, 15 components) per pose; its median per finger and sweep says whether a layout needs its own
term (the five-fold floor on 11-19 was 4.4-5.7 uT).

The thumb's DIP tag (pose_sweep.py since 21:00 on 11-19; the owner, relayed: "maybe tracking the thumb fingertip tag
would help us understand slop as well?"): per pose its centre and normal against the hand model at the measured joints,
from both cameras where both saw it (one camera's depth scatters 2.5 mm along its view, both cameras' 0.3 mm).

    /usr/bin/python3 sim/20261119-pose_sweep_own_pose.py [SWEEP.jsonl ...] [--out data/s3/20261119-pose_sweep_own_pose.json]
"""
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[2]
sys.path[:0] = [str(ROOT / "src/sr2hand/driver/reskin5x/host"), str(ROOT / "src")]
import tip_contacts as TC  # noqa: E402

SWEEPS = ROOT / "logs/20261119-pose_sweeps"
MUXES = sorted((ROOT / "logs/20261119-three_finger_mux").glob("20261119-*-mux.bin"))
GPS = {"deployed": TC.OWN_POSE,
       "x70y110": ROOT / "docs/experiments/20261115-hom_bench_trials/data/20261117-tip_contacts/"
                         "20261119-own_pose_gp_sweep_x70y110_7v5.npz"}


def mux_for(t):
    starts = [dt.datetime.strptime(f.name[:15], "%Y%m%d-%H%M%S").timestamp() for f in MUXES]
    return MUXES[max(i for i, s0 in enumerate(starts) if s0 <= t)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweeps", nargs="*")
    ap.add_argument("--out", default=str(STUDY / "data/s3/20261119-pose_sweep_own_pose.json"))
    a = ap.parse_args()
    files = [Path(s) for s in a.sweeps] or sorted(SWEEPS.glob("*_sweep.jsonl"))
    gps = {k: TC.OwnPose(v) for k, v in GPS.items()}
    res = []
    for fpath in files:
        L = [json.loads(x) for x in fpath.read_text().splitlines() if x.strip()]
        head, poses = L[0], [r for r in L[1:] if r.get("t_hold")]
        if not poses:
            continue
        t0, t1 = poses[0]["t_hold"] - 5.0, poses[-1]["t_end"] + 1.0
        feed = TC.MuxLog(mux_for(t0), t0 - TC.KEEP_S, t1)
        tc = TC.TipContacts(feed, np.zeros(6), gp=False, own_pose=False)
        zero = next((r for r in poses if r["pose"] == "zero"), None)
        if zero is None:
            continue
        rec = dict(sweep=fpath.name, design=head.get("design"), mounts=head.get("mounts"), fingers={})
        for f in TC.FINGERS:
            tip = tc.tips[f]

            def mean_field(r):
                tt, rr = feed.window(f, r["t_end"], r["t_end"] - r["t_hold"] - 0.3)
                ok = TC.valid(rr)
                _, rT = feed.window(f, r["t_end"], TC.KEEP_S)
                return tip.fields(rr[ok], rT).mean(0)

            b0 = mean_field(zero)
            q0 = [zero["measured"][f][k] for k in ("yaw", "mcp", "pip")]
            rows = []
            for r in poses:
                if r["pose"] == "zero" or not r.get("measured"):
                    continue
                q = [r["measured"][f][k] for k in ("yaw", "mcp", "pip")]
                d = mean_field(r) - b0
                row = dict(pose=r["pose"], q=np.round(q, 2).tolist(), change_uT=round(float(np.linalg.norm(d)), 2))
                for name, g in gps.items():
                    pred = g.predict(f, q) - g.predict(f, q0)
                    row[f"resid_{name}_uT"] = round(float(np.linalg.norm(d - pred)), 2)
                rows.append(row)
            moved = [r for r in rows if max(abs(x - y) for x, y in zip(r["q"], q0)) > 3.0]
            rec["fingers"][f] = dict(poses=rows, n_moved=len(moved),
                                     change_median_uT=round(float(np.median([r["change_uT"] for r in moved])), 2)
                                     if moved else None,
                                     **{f"resid_{k}_median_uT": round(float(np.median([r[f'resid_{k}_uT'] for r in moved])), 2)
                                        if moved else None for k in gps})
            print(f"{fpath.name[:15]} {head.get('design', '')[:32]:32s} {f:6s}: {len(moved):2d} poses moved, change "
                  f"{rec['fingers'][f]['change_median_uT']} uT; residual deployed "
                  f"{rec['fingers'][f]['resid_deployed_median_uT']}, x70y110 {rec['fingers'][f]['resid_x70y110_median_uT']} uT",
                  flush=True)
        tt = [(r["pose"], r.get("thumb_tag") or {}) for r in poses]
        two = [(n, t) for n, t in tt if "d_mm_2cam" in t]
        one = [(n, t) for n, t in tt if "d_mm" in t]
        rec["thumb_tag"] = {"poses": [{"pose": n, **t} for n, t in tt if t],
                            "seen_poses": len(one), "two_camera_poses": len(two),
                            "d_mm_2cam_median": np.round(np.median([t["d_mm_2cam"] for _, t in two], axis=0), 2).tolist()
                            if two else None,
                            "normal_deg_2cam_median": round(float(np.median([t["normal_deg_2cam"] for _, t in two])), 2)
                            if two else None,
                            "normal_deg_median": round(float(np.median([t["normal_deg"] for _, t in one])), 2)
                            if one else None}
        if one:
            print(f"   thumb tag: seen at {len(one)} poses ({len(two)} by both cameras); normal "
                  f"{rec['thumb_tag']['normal_deg_median']} deg from the model, both cameras "
                  f"{rec['thumb_tag']['d_mm_2cam_median']} mm, {rec['thumb_tag']['normal_deg_2cam_median']} deg", flush=True)
        res.append(rec)
    Path(a.out).write_text(json.dumps({"what": __doc__.split("\n\n")[0].replace("\n", " "),
                                       "script": str(Path(__file__).resolve().relative_to(ROOT)), "sweeps": res}, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
