"""The fingertip contacts in the HOM trials of 2026-11-19 night: what the boards say during real manipulation (the
owner, 18:4x: pose sensing first, its smoothness and stability).

Each trial's corvus record (logs/20261116-hom_trials/<stamp>_<task>_corvus.jsonl, hom_trial.py since 11-17) holds, every
20 ms period, each finger's contact from the trials' estimator (tip_contacts, in its own process: the point in P through
the hand model at the measured joints, the place on the pad, the normal, the force, the GP's weight w), or none, and
the tracked object's pose in P. Per trial and finger:
  contact    the share of periods with a contact (0.25 N to start, 0.15 N to stay)
  place      the place on the pad (z along it, y across; pad frame of the calibrations), median and quartiles
  jitter     the change of the place from one estimate to the next (estimates whose time stamps differ), median and
             90th percentile
  force      the normal force, median and 90th percentile
  surface    the contact point's distance from the tracked object's surface (sphere: |p - c| - r; cylinder: the
             distance from its axis less r), median and quartiles: negative inside the object. The pad's indentation
             (about 0.5-1.5 mm at 1-2.5 N) puts an estimate on the undeformed pad inside a rigid object; more than that
             is the hand model's error at the measured joints, or the estimate's.

    /usr/bin/python3 sim/20261119-trial_contacts.py [RECORD ...] [--out data/s3/20261119-trial_contacts.json]
        default: every logs/20261116-hom_trials/20261119-*_corvus.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[2]
sys.path[:0] = [str(ROOT / "src")]
from sr2hand.geometry.rotations import quat_to_mat  # noqa: E402

TRIALS = ROOT / "logs/20261116-hom_trials"
FINGERS = ("thumb", "index", "middle")


def plan_shape(head):
    p = head.get("plan")
    if not p:
        return None
    pp = Path(p) if Path(p).is_absolute() else ROOT / "docs/experiments/20261115-hom_bench_trials/lab" / p
    if not pp.exists():
        pp = ROOT / "docs/experiments/20261115-hom_bench_trials/plans" / Path(p).name
    s = json.loads(pp.read_text())["meta"]["hom"]["setup"]
    return 1e3 * float(s["r_m"]), 1e3 * float(s.get("hl_m", 0.0)), pp.name


def surface_dist(p, obj, r, hl):
    c = np.asarray(obj["p_PO"], float)
    if hl == 0.0:
        return float(np.linalg.norm(p - c) - r)
    ax = quat_to_mat(np.asarray(obj["q_PO"], float))[:, 2]
    v = p - c
    return float(np.linalg.norm(v - (v @ ax) * ax) - r)


def q(v, ps=(25, 50, 75)):
    return [round(float(x), 2) for x in np.percentile(v, ps)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("records", nargs="*")
    ap.add_argument("--out", default=str(STUDY / "data/s3/20261119-trial_contacts.json"))
    a = ap.parse_args()
    files = [Path(x) for x in a.records] or sorted(TRIALS.glob("20261119-*_corvus.jsonl"))
    out = []
    for fp in files:
        L = [json.loads(x) for x in fp.read_text().splitlines() if x.strip()]
        head = L[0]
        st = [x for x in L if "seq" in x]
        shp = plan_shape(head)
        if shp is None or not st:
            continue
        r, hl, plan = shp
        rec = dict(record=fp.name, plan=plan, task=head.get("task"), object=head.get("object"), periods=len(st),
                   stop=next((x.get("stop") for x in L if x.get("stop")), None), fingers={})
        for f in FINGERS:
            rows, last_t = [], None
            for x in st:
                t = (x.get("tips") or {}).get(f)
                o = x.get("obj") or {}
                if t is None or not o.get("p_PO") or not o.get("q_PO"):
                    continue
                fresh = last_t is None or t.get("t") != last_t
                last_t = t.get("t")
                rows.append((t["p_pad"][2], t["p_pad"][1], t["f_n"],
                             surface_dist(np.asarray(t["p_P"], float), o, r, hl), t.get("w") or np.nan, fresh))
            if not rows:
                rec["fingers"][f] = dict(contact_share=0.0)
                continue
            A = np.array([x[:5] for x in rows], float)
            fr = np.array([x[5] for x in rows], bool)
            Z = A[fr]
            dz = np.abs(np.diff(Z[:, 0])) if len(Z) > 2 else np.array([np.nan])
            dy = np.abs(np.diff(Z[:, 1])) if len(Z) > 2 else np.array([np.nan])
            rec["fingers"][f] = dict(contact_share=round(len(rows) / len(st), 3), estimates=int(fr.sum()),
                                     place_z_mm=q(A[:, 0]), place_y_mm=q(A[:, 1]),
                                     jitter_z_mm=[round(float(np.nanmedian(dz)), 3), round(float(np.nanpercentile(dz, 90)), 3)],
                                     jitter_y_mm=[round(float(np.nanmedian(dy)), 3), round(float(np.nanpercentile(dy, 90)), 3)],
                                     force_N=[round(float(np.median(A[:, 2])), 2), round(float(np.percentile(A[:, 2], 90)), 2)],
                                     surface_mm=q(A[:, 3]), w_median=round(float(np.nanmedian(A[:, 4])), 2))
            g = rec["fingers"][f]
            print(f"{fp.name[:15]} {plan[:34]:34s} {f:6s}: contact {g['contact_share']:.2f}, z {g['place_z_mm']}, "
                  f"jitter z {g['jitter_z_mm']} mm, force {g['force_N']} N, surface {g['surface_mm']} mm", flush=True)
        out.append(rec)
    Path(a.out).write_text(json.dumps({"what": __doc__.split("\n\n")[0].replace("\n", " "),
                                       "script": str(Path(__file__).resolve().relative_to(ROOT)), "trials": out}, indent=1))
    print("wrote", a.out, len(out), "trials")


if __name__ == "__main__":
    main()
