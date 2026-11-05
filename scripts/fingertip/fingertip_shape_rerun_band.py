#!/usr/bin/env python3
"""Worst-case score over a band of scene frictions for the stage-3 tips of the fingertip-shape rerun (ledger step 12):
the band option of scripts/hom/hom_grasp_refine.py (hand-20261104-hom_worst_case_score) with each tip in place of the v4b pad.

The band band_k21mu is the STS3032 (kp2.1_kv0_fr0.515_fl0_dp0.042) with pads at 2.5 MPa at scene friction 0.4, 0.7 and 1.0.
Its picks (per layout of its 30-layout sample: S1 the pose toy's grasp, S2 the grasp refined, S3 grasp, forces and modes
refined, each chosen by its lowest three-seed mean over the band) were found with the v4b tip; here they run unchanged with
another tip, so a tip's score is the score the v4b tip's designs keep on it.

  scenes    per tip and friction, the HOM tasks' reference scenes (hom_gpu.Scene of sphere:40 and cylinder:25) built with
            hand_object_scale_sim.TIP set to the tip, into logs/<run>/hom_scenes/<tip>_mu<mu>/ (each band plant's
            HOS_SCENES); every scene's meta is checked for the tip, the friction, the plant and 2.5 MPa
  validate  hom_grasp_refine's band validation of the band's picks at seeds 20-24 on the three plants, the plants' scene
            caches replaced by the tip's and S.TIP set in the parent (open poses on the tip's mesh); rows in
            docs/experiments/<run>/band/<tip>/val/<plant>/<task>_d<d>.jsonl
  report    per tip and task, each pick's worst-case score (the lowest of its three five-seed means) against the v4b tip's
            on the same pick (its rows of 11-04, rows/band_k21mu/val/) -> band/summary.json

    SCALE_HAND=cad_v2 .venv/bin/python scripts/fingertip/fingertip_shape_rerun_band.py scenes --tip <tip>
    SCALE_HAND=cad_v2 .venv/bin/python scripts/fingertip/fingertip_shape_rerun_band.py validate --tip <tip> --task twist
    python3 scripts/fingertip/fingertip_shape_rerun_band.py report
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("SCALE_HAND", "cad_v2")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path
import fingertip_shape_rerun_common as C  # noqa: E402

BAND = "band_k21mu"
STS = "kp2.1_kv0_fr0.515_fl0_dp0.042"
MU = {"k21mu0p4": 0.4, "k21mu0p7": 0.7, "k21": 1.0}
OUT = C.D / "band"
TASKS = ("twist", "roll", "pinpush", "press")


def cache(tip, plant):
    return C.LOGS / "hom_scenes" / f"{tip}_mu{MU[plant]:g}"


def tag(tip, plant):
    return f"_fsr_{C.short(tip).replace(' ', '').replace('/', '-').replace('(', '').replace(')', '').replace(',', '')}_{plant}"


def plants_for(tip):
    """hom_grasp_refine.PLANTS entries of the band with the tip's scene caches and row tags."""
    return {p: (STS, str(cache(tip, p).relative_to(ROOT)), tag(tip, p)) for p in MU}


def cmd_scenes(a):
    """One fresh process per friction (hand_object_scale_sim reads HOS_* at import)."""
    if a.plant:
        os.environ.update(HOS_PLANT=STS, HOS_SCENES=str(cache(a.tip, a.plant).relative_to(ROOT)),
                          HOM_ROWS_TAG=tag(a.tip, a.plant))
        import hand_object_scale_sim as S
        S.TIP, S.MU = a.tip, MU[a.plant]
        import hom_gpu as HG
        for shape, d in (("sphere", 40.0), ("cylinder", 25.0)):
            sc = HG.Scene(shape, d)
            m = json.loads(Path(sc.pads_path).with_suffix(".json").read_text())
            print(f"{a.tip} {a.plant} {shape} {d:g}: {Path(sc.pads_path).relative_to(ROOT)} tip {m.get('tip_key')} "
                  f"mu {m.get('mu')} plant {m.get('plant')} E {m.get('pad_E_Pa')}", flush=True)
        bad = []
        for p in sorted(cache(a.tip, a.plant).glob("*/*.json")):
            m = json.loads(p.read_text())
            if "pad_E_Pa" in m and (m.get("tip_key") != a.tip or abs(m.get("mu", -1) - MU[a.plant]) > 1e-9
                                    or m.get("plant") != STS or abs(m["pad_E_Pa"] - C.E_PA) > 1.0):
                bad.append(p)
        if bad:
            raise SystemExit(f"{len(bad)} scenes of another tip, friction, plant or modulus, e.g. {bad[0]}")
        return
    for p in MU:
        subprocess.run([sys.executable, __file__, "scenes", "--tip", a.tip, "--plant", p], check=True)


def cmd_validate(a):
    import hom_grasp_refine as HR
    HR.PLANTS.update(plants_for(a.tip))
    band = HR.BANDS[BAND]
    HR.setenv(band[-1])
    import hand_object_scale_sim as S
    S.TIP = a.tip
    import fp_sr2_toy as FP
    import hom_ea as E
    import hom_gpu as HG
    task = a.task
    FP.set_sizes(**({"sph_d_mm": HR.TASKS[task]} if HR.POSE[task] == "tripod" else {"cyl_d_mm": HR.TASKS[task]}))
    lays, items = HR.val_items(task, BAND, BAND)           # the band's layouts and picks, from the 11-04 rows
    paths = {p: OUT / a.tip / "val" / p / f"{task}_d{HR.TASKS[task]:g}.jsonl" for p in band}
    for q in paths.values():
        q.parent.mkdir(parents=True, exist_ok=True)
    scene = HG.Scene("sphere" if HR.POSE[task] == "tripod" else "cylinder", HR.TASKS[task])
    n = HR.evaluate_band(task, list(band), paths, items, HR.VAL_SEEDS, scene, HG, E)
    print(f"{a.tip} {task}: {len(items)} picks on {', '.join(band)}, {n} rollouts run", flush=True)


def worst(task, rows_by_plant):
    """{(call, pick): lowest over the plants of the five-seed mean score (hom_landscape.rollout_score)}."""
    import numpy as np
    import hom_landscape as L
    per = {}
    for p, rows in rows_by_plant.items():
        acc = {}
        for r in rows:
            acc.setdefault((r["call"], r["pick"]), []).append(float(L.rollout_score(task, r)))
        for k, v in acc.items():
            per.setdefault(k, {})[p] = float(np.mean(v))
    return {k: min(v.values()) for k, v in per.items() if len(v) == len(MU)}


def cmd_report(a):
    import numpy as np
    import hom_grasp_refine as HR
    old = HR.rows_dir(BAND) / "val"
    out = {"band": BAND, "plants": list(MU), "tips": {}}
    for tip_dir in sorted(p for p in OUT.glob("*") if (p / "val").is_dir()):
        tip = tip_dir.name
        T = {}
        for task in TASKS:
            f = f"{task}_d{HR.TASKS[task]:g}.jsonl"
            new = worst(task, {p: HR.read_rows(tip_dir / "val" / p / f) for p in MU})
            ref = worst(task, {p: HR.read_rows(old / p / f) for p in MU})
            T[task] = {}
            for pick in ("S1", "S2", "S3"):
                ks = [k for k in new if k[1] == pick and k in ref]
                if not ks:
                    continue
                d = [new[k] - ref[k] for k in ks]
                T[task][pick] = {"n": len(ks), "worst_median": float(np.median([new[k] for k in ks])),
                                 "v4b_worst_median": float(np.median([ref[k] for k in ks])),
                                 "diff_median": float(np.median(d)), "diff_q25": float(np.percentile(d, 25)),
                                 "diff_q75": float(np.percentile(d, 75))}
        out["tips"][tip] = T
        print(tip, {t: {p: round(v["worst_median"], 3) for p, v in x.items()} for t, x in T.items()}, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(out, indent=1) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scenes")
    s.add_argument("--tip", required=True)
    s.add_argument("--plant", default=None, choices=list(MU))
    v = sub.add_parser("validate")
    v.add_argument("--tip", required=True)
    v.add_argument("--task", required=True, choices=TASKS)
    sub.add_parser("report")
    a = ap.parse_args()
    {"scenes": cmd_scenes, "validate": cmd_validate, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
