"""Whether the middle board's high readings in the tips-on twists of 2026-11-17 at (70, 110) came from the sphere or from
the thumb (the twist ranking's check of 11-19 night: on the CAD collision spheres, the measured joints put the thumb's
tip 5-11 mm into the middle or index finger in four of them, and the middle read 5.3 and 4.0 N in two).

Per trial, the corvus record's periods with a middle contact, split at --split N: the median force, the place on the pad
(y across, z along from the mount face; apex z 10.86), the contact point's distance from the tracked sphere's surface
(|p - c| - r; positive outside), and its distance from the thumb's nearest part (hom_setup.finger_pts: tip pad, tip end,
PIP joint, surfaces) at the measured joints (the record's servo degrees through the driver's signs).

    /usr/bin/python3 sim/20261119-thumb_middle_contact.py [--out data/s3/20261119-thumb_middle_contact.json]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[2]
LAB = ROOT / "docs/experiments/20261115-hom_bench_trials/lab"
sys.path[:0] = [str(LAB), str(ROOT / "src")]
import hom_setup as HS  # noqa: E402

TRIALS = ROOT / "logs/20261116-hom_trials"
PLAN = LAB.parent / "plans/20261117-film_x70_y110_twist_gr_mcpstop_plan.json"
# the tips-on twists of 11-17 at (70, 110): corvus stamp (the server's record a few seconds later) and session folder
RUNS = [("182434", "20261117-1815_7v5_trials", "182443"), ("194718", "20261117-night_repeats", "194727"),
        ("194913", "20261117-night_repeats", None), ("195729", "20261117-night_repeats", None),
        ("195922", "20261117-night_repeats", "195931"), ("200757", "20261117-night_repeats", "200806")]


def sim_joints(P, j):
    v = np.asarray(j, float).reshape(3, 3)
    return {f: {jn: float(v[k][("aa", "fe1", "fe2").index(P.SIM_JOINT_TO_SERVO[jn])]) / P.JOINT_SIGN[(f, jn)]
                for jn in ("yaw", "mcp", "pip")} for k, f in enumerate(("thumb", "index", "middle"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=float, default=3.0)
    ap.add_argument("--out", default=str(STUDY / "data/s3/20261119-thumb_middle_contact.json"))
    a = ap.parse_args()
    P = HS._driver_plan()
    plan = HS.load_plan(str(PLAN))
    r = 1e3 * float(plan["meta"]["hom"]["setup"]["r_m"])
    out = []
    for stamp, folder, server in RUNS:
        f = TRIALS / folder / f"20261117-{stamp}_twist_corvus.jsonl"
        st = [x for x in (json.loads(y) for y in f.read_text().splitlines() if y.strip()) if "seq" in x]
        rows = {"high": [], "low": []}
        for x in st:
            m = (x.get("tips") or {}).get("middle")
            if not m or not x.get("joints"):
                continue
            p = np.asarray(m["p_P"], float)
            o = x.get("obj") or {}
            c = np.asarray(o.get("p_PO") or [np.nan] * 3, float)
            surf = float(np.linalg.norm(p - c) - r) if np.all(np.isfinite(c)) else np.nan
            q = sim_joints(P, x["joints"])
            d_th = min(float(np.linalg.norm(p - pt)) - rad for pt, rad in HS.finger_pts(plan, "thumb", q["thumb"]))
            rows["high" if m["f_n"] > a.split else "low"].append((x["t"], m["f_n"], surf, d_th, m["p_pad"][1], m["p_pad"][2]))
        rec = {"corvus": stamp, "server": server, "periods": len(st)}
        for k, v in rows.items():
            if not v:
                rec[k] = None
                continue
            A = np.array(v, float)
            fin = np.isfinite(A[:, 2])
            rec[k] = {"periods": len(A), "t_s": [round(A[0, 0], 2), round(A[-1, 0], 2)],
                      "force_N": round(float(np.median(A[:, 1])), 2),
                      "to_sphere_surface_mm": round(float(np.median(A[fin, 2])), 1) if fin.any() else None,
                      "tracked_share": round(float(fin.mean()), 2),
                      "to_thumb_mm": round(float(np.median(A[:, 3])), 1),
                      "place_y_mm": round(float(np.median(A[:, 4])), 1), "place_z_mm": round(float(np.median(A[:, 5])), 1)}
        out.append(rec)
        print(stamp, json.dumps({k: rec[k] for k in ("high", "low")}), flush=True)
    Path(a.out).write_text(json.dumps({"what": __doc__.split("\n\n")[0].replace("\n", " "),
                                       "script": str(Path(__file__).resolve().relative_to(ROOT)), "split_N": a.split,
                                       "trials": out}, indent=1) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
