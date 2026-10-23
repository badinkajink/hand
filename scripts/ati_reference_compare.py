#!/usr/bin/env python3
"""Which reference calibrates the ATI: the weighed mass, or the HEX-E with the ATI coaxial or turned 90 deg (owner,
2026-10-23: "the ati will be stuck onto the F/T sensor with the Z-axis aligned. however, we now know that the Z-axis is
the noisiest axis ... maybe we have to design an alternate mount that has it rotated?").

On the pairs of a simulated session (scripts/ati_calibration_session.py --sim, then scripts/ati_calibration_analysis.py),
the ATI's gage differences are fitted against three references: the mass's own wrench; the HEX-E's reading of it with
the ATI coaxial; and the HEX-E's reading with the ATI turned 90 deg about its y axis, so that the HEX-E's z reads the
ATI's x. The HEX-E carries what the UR5 runs of 2026-10-22 measured: force gains 1.135, 1.085 and 0.78 of their mean
(first2's gravity fit, R8 of the tip calibration's queue), torque gains off by 3 % (unknown), and per-sample noise of
0.035, 0.035 and 0.21 N and 2, 2 and 1 N mm (Fz from the lab at 100 Hz with filter 3, the rest from its data sheet),
averaged over each rest's 300 samples. Each fit is scored against the simulated sensor's truth on 2000 wrenches of the
pad's range (scripts/ati_calibration_analysis.py score_truth).

    .venv/bin/python scripts/ati_reference_compare.py <simulated session folder> [--out json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ati_calibration_analysis as AN  # noqa: E402
import ati_calibration_fit as FIT  # noqa: E402

GAIN = np.array([1.135, 1.085, 0.78, 1.0, 1.0, 1.0])
NOISE = np.array([0.035, 0.035, 0.21, 0.002, 0.002, 0.001]) / np.sqrt(300.0)
P_TURN = np.array([[0, 0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0], [-1, 0, 0, 0, 0, 0],      # the HEX-E's axes in the turned
                   [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0], [0, 0, 0, -1, 0, 0]], float)   # ATI's: 90 deg about y


def run(folder, seeds=20):
    folder = Path(folder)
    z = np.load(folder / "analysis/pairs.npz")
    G, W = z["G"], z["W"]
    truth = json.loads((folder / "sim_truth.json").read_text())
    out = {}
    for name, P in (("mass", None), ("hexe_coaxial", np.eye(6)), ("hexe_turned", P_TURN)):
        meds = []
        for sd in range(seeds if P is not None else 1):
            rng = np.random.default_rng(sd)
            if P is None:
                ref = W
            else:
                g = GAIN * np.r_[1.0, 1.0, 1.0, 1.0 + rng.normal(0, 0.03, 3)]
                hx = (W @ P.T) * g + rng.normal(0.0, 1.0, W.shape) * NOISE      # what the HEX-E reads, its own axes
                ref = hx @ P                                                    # back in the ATI's axes
            C = FIT.fit_full(G, ref)
            meds.append(AN.score_truth(C, truth, n=1000, seed=sd)["median"])
        out[name] = dict(median=np.median(meds, axis=0).tolist(), seeds=len(meds))
        m = out[name]["median"]
        print(f"{name:13s}: median error F {m[0]:.3f} {m[1]:.3f} {m[2]:.3f} N, T {m[3] * 1e3:.1f} {m[4] * 1e3:.1f} "
              f"{m[5] * 1e3:.1f} N mm")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folder")
    ap.add_argument("--out", default=str(ROOT / "docs/experiments/20261023-ati_ft_calibration/20261023-ati_reference_compare.json"))
    a = ap.parse_args()
    res = run(a.folder)
    res["folder"] = str(Path(a.folder))
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
