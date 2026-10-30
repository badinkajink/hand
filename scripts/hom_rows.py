"""Rows of the HOM task runs (scripts/hom_tasks.py, scripts/hom_gpu.py) as the page builders read them, without the
simulation stack.

rescale_metrics: until the evening of 2026-10-29, hom_tasks.metrics integrated the object's angular velocity and the
centre's reference path with a fixed 0.01 s step, the 100 Hz controller's period. A run at another period (every 50 Hz
variant: rate50, sense, the fingertip-sensing variants, the filter sweep's dt 0.02) therefore reported its twist angles
and step angles times 0.01 / dt, half at 50 Hz; the step gains, velocity errors and the pinch tasks' scores were right.
"""
from __future__ import annotations

DT_C = 0.01
INTEGRATED = ("ramp_max_deg", "ramp_three_pads_max_deg", "ramp_end_deg")


def rescale_metrics(row):
    """A row scored before 2026-10-29 evening, when metrics integrated the angular velocity and the centre's reference
    path with a fixed 0.01 s step: a run at another controller period (the 50 Hz variants: rate50, sense, the fingertip
    variants, the filter sweep's dt 0.02) reported its twist angles and step angles times 0.01 / dt. This scales them
    back in place (the step gains were right) and drops the roll's centre error and reference travel, which cannot be
    rescaled (the twist's reference path is zero, so its centre error was right). Rows scored since carry metrics_dt and pass unchanged."""
    if "metrics_dt" in row or row.get("status", "ok") != "ok":
        return row
    dt = float((row.get("hom") or {}).get("dt", row.get("dt", DT_C)))
    row["metrics_dt"] = dt
    if abs(dt - 0.01) < 1e-9:
        return row
    f = dt / 0.01
    for k in INTEGRATED:
        if row.get(k) is not None:
            row[k] = round(row[k] * f, 2)
    for s_ in row.get("steps") or []:
        for k in ("angle_deg", "ref_deg"):
            if s_.get(k) is not None:
                s_[k] = round(s_[k] * f, 2)
    if row.get("centre_travel_ref_mm"):                  # a reference path (the roll): its error cannot be rescaled
        row["centre_err_max_mm"] = row["centre_travel_ref_mm"] = None
    return row
