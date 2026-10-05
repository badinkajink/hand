#!/usr/bin/env python3
"""Contact-model comparison bed, task 1 (T1 pull) on the consistency set of the protocol.

Runs contact_bed_pull.run_case, unchanged, for mj_pads1 at 0.5 and 2 ms and for the 2 mm and 0.5 mm sphere
pads (mj_pads2, mj_pads05) at 1 ms, N = 0.5, 1, 3. The two pad resolutions are added to contact_bed_pull's
MODELS table here rather than in that script. Rows go to pull_slip.jsonl with the bed's protocol fields
(task, model, rig_spec, status, film, script, git_rev) added; cases already in the file are skipped.

    PY=logs/20261001-hom_contact/venv/bin/python
    $PY scripts/contact_bed_pull_consistency.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contact_bed_common as B  # noqa: E402
import contact_bed_pull as P  # noqa: E402

for _name, (_rig, _chain) in B.MODELS.items():
    P.MODELS.setdefault(_chain, _rig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--N", nargs="+", type=float, default=[0.5, 1.0, 3.0])
    ap.add_argument("--out", default=str(P.OUT))
    a = ap.parse_args()
    have = P.done(a.out)
    rev = B.git_rev()
    for model, dt in B.CONSISTENCY:
        rig_spec, chain = B.MODELS[model]
        assert P.MODELS[chain] == rig_spec, (chain, P.MODELS[chain], rig_spec)
        for N in a.N:
            if (chain, N, dt) in have:
                continue
            r = P.run_case(chain, N, dt)
            r.update(task="pull", model=model, rig_spec=rig_spec, status="complete" if r.get("slipped") else "failed",
                     film=None, script="scripts/contact_bed_pull_consistency.py (contact_bed_pull.run_case)",
                     git_rev=rev)
            P.H.append_row(a.out, r)
            keys = ("mu_eff", "u_pre_mm", "creep_mm_s", "v_slip_mean50_mm_s", "mu_slide", "us_per_step_median", "wall_s")
            print(model, N, dt, {k: round(r[k], 4) for k in keys if k in r}, flush=True)


if __name__ == "__main__":
    main()
