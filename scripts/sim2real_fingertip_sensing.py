#!/usr/bin/env python3
"""Five-chip magnet readout of the 85A magnet blocks on the CAD finger (docs/handoff/20261016-sim2real_fingertip_readiness.md,
step 8).

The readout model of 2026-10-13/16 (scripts/tpu_tip_magnet_sensing.py, scripts/mlx90393_model.py): point dipoles (N52,
remanence 1.45 T, every north pole toward the palmar face), the ReSkin 5X board's five MLX90393 (U1 under the contact point,
U2-U5 7 mm from it), the 85A finite elements' magnet motions under 1 N patch loads at 15 face positions
(scripts/sim2real_fingertip_fe.py), one 141 Hz sample at GAIN_SEL 7's datasheet noise, and the Cramer-Rao bound of the
normal force with the contact position and a uniform background field unknown.

The chips' sensing plane, behind the tip's back face:
  touch   0.42 mm: the chips against the back face, the earlier pages' assumption; on the CAD finger this needs a DIP
          link without its 3.0 mm dorsal plate behind the tip
  plate   3.42 mm: the board on the dorsal face of the DIP link's 3.0 mm plate, chips toward the plate (the scenes' board)
  plate_flip 4.08 mm: the same board turned over, FR-4 on the plate and chips facing out

Per FE row and plane: the static field at each chip and the highest GAIN_SEL whose range covers it, the force bound at the
contact point and over the 15 positions, the position bounds, the magnet travel per newton and its range over a solid
modulus of 6-24 MPa (travel scales as 12 MPa / E_s, the force bound as E_s / 12 MPa), and the field change at each chip
when one magnet is mounted the wrong way round (the assembly's polarity check).

    .venv/bin/python scripts/sim2real_fingertip_sensing.py
Out: docs/experiments/20261016-sim2real_fingertip_readiness/sensing/sensing.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import mlx90393_model as MLX  # noqa: E402
import tpu_tip_magnet_sensing as MS  # noqa: E402

D = ROOT / "docs/experiments/20261016-sim2real_fingertip_readiness"
FE_ROWS = D / "fe/fem.jsonl"
OUT = D / "sensing/sensing.json"
PLATE_MM = 3.0                         # the DIP link's dorsal plate behind the tip (finger frame x 6.45-9.45 mm)
PLANES = {"touch": MLX.SPOT_MM, "plate": PLATE_MM + MLX.SPOT_MM,
          "plate_flip": PLATE_MM + MLX.BOARD_T_MM + MLX.PKG_H_MM - MLX.SPOT_MM}
E_S = (6.0, 9.0, 12.0, 18.0, 24.0)


def gain_for(bmax_mT):
    """Highest GAIN_SEL (RES 0) whose range covers |B|; the ranges scale with the LSB from 4.8 mT at GAIN_SEL 7."""
    for g in range(7, -1, -1):
        rng = MS.B_RANGE_MT * MLX.SENS[g][0][0] / MLX.SENS[7][0][0]
        if rng >= bmax_mT:
            return g, rng
    return None, None


def polarity(row, gap):
    """Field change at each chip (uT, |dB| per chip) when one magnet is reversed, per magnet."""
    ch = MS.chips(row, "board", gap)
    base = MS.B0(row, ch)
    out = []
    for k in range(len(row["magnets"])):
        r2 = dict(row)
        M = MS.mag_state(row)
        B = np.zeros_like(base)
        for i, c in enumerate(ch):
            for j, (pm, m) in enumerate(M):
                B[i] += MS.dipole(c, pm, -m if j == k else m)
        out.append([round(float(v), 1) for v in np.linalg.norm(B - base, axis=1) * 1e6])
    return out


def main():
    rows = [json.loads(l) for l in open(FE_ROWS)]
    res = []
    for r in rows:
        travel = r["per_force"]["1"].get("magnets", [])
        tr = [float(np.linalg.norm(t)) for t, _ in travel]
        for plane, gap in PLANES.items():
            a = MS.analyse(r, "board", gap)
            ch = MS.chips(r, "board", gap)
            B = MS.B0(r, ch) * 1e3                                    # mT, (5, 3)
            bmax = float(np.abs(B).max())
            g, rng = gain_for(bmax)
            pts = a["points"]
            both = [p["sigF_n_unk_bg"] for p in pts]
            c = a["centre"]
            res.append(dict(
                layout=r["layout"], recipe=r["recipe"], magnets=len(r["magnets"]), plane=plane, gap_mm=round(gap, 3),
                E_star_MPa=round(r["E_best_MPa"], 2), approach_1N_um=round(r["per_force"]["1"]["delta_um"], 1),
                B_chips_mT=np.round(B, 3).tolist(), B_max_component_mT=round(bmax, 3),
                B_U1_mT=round(float(np.linalg.norm(B[0])), 3), gain_sel=g, gain_range_mT=None if rng is None else round(rng, 2),
                headroom=None if rng is None else round(rng / bmax, 2),
                signal_U1_uT_per_N=round(float(np.linalg.norm(MS.signal(r, r["patches"][len(r["patches"]) // 2]["normal"],
                                                                            ch[:1]))), 2),
                centre_force_N=round(c["sigF_n_unk_bg"], 4), centre_force_known_N=round(c["sigF_n"], 4),
                centre_sig_y_mm=round(c["sig_y_mm_bg"], 2), centre_sig_z_mm=round(c["sig_z_mm_bg"], 2),
                median_force_N=round(float(np.median(both)), 4), range_force_N=[round(min(both), 4), round(max(both), 4)],
                magnet_travel_1N_um=[round(x, 2) for x in tr],
                travel_by_Es_um_per_N={f"{e:g}": round(float(np.mean(tr)) * 12.0 / e, 2) for e in E_S},
                centre_force_by_Es_N={f"{e:g}": round(c["sigF_n_unk_bg"] * e / 12.0, 4) for e in E_S},
                polarity_flip_dB_uT=polarity(r, gap)))
            print(f"{r['layout']:18s} {r['recipe']} {plane:10s} gap {gap:.2f}: B max {bmax:6.2f} mT GAIN_SEL {g}, centre "
                  f"{c['sigF_n_unk_bg']:.3f} N, median {np.median(both):.3f} N, travel {np.mean(tr):.1f} um/N", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(
        rows=res, planes_mm={k: round(v, 3) for k, v in PLANES.items()},
        setting="GAIN_SEL 7, RES 0, OSR 0, DIG_FILT 5 noise (one 141 Hz sample); below GAIN_SEL 7 the datasheet gives no "
                "noise, so a bound at a lower gain is a lower bound",
        terms=dict(centre_force_N="Cramer-Rao bound of the normal force at the contact point from one sample of the five chips, "
                                  "contact position and a uniform background field unknown",
                   headroom="the GAIN_SEL's range over the largest field component at any chip"),
        script="scripts/sim2real_fingertip_sensing.py", when=time.strftime("%Y-%m-%d %H:%M")), indent=1))


if __name__ == "__main__":
    main()
