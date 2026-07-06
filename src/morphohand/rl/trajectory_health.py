"""Trajectory-health characterization: flag DEGENERATE grasp/reorient policies.

Motivation (2026-07-02): our headline metrics (tip_lost, object-held, mean reward)
MASKED real defects visible only on video — a 2-finger grasp with a LATE third finger,
high-frequency jitter, de-centering slides, idle-finger degenerate pinches, over-clamp.
This module turns a logged rollout into an explicit PASS/WARN/FAIL scorecard for each of
those failure modes, so any eval or training run can flag a bad policy automatically
instead of trusting aggregate rewards.

Pure function on numpy arrays (no sim / torch deps) so it is trivially reusable and
testable. `characterize_trajectory(...)` -> Scorecard; `format_scorecard(sc)` -> str.

The canonical trajectory dict (all per-step, length T, in POLICY steps):
  finger_found : (T, 3) float/bool  per-finger tip contact found  [thumb, index, middle]
  finger_force : (T, 3) float       per-finger tip contact normal force (N)
  obj_z        : (T,)   float       object center height (m)
  obj_cos      : (T,)   float       held-vertical cos (body +Z . world +Z), 1=vertical
  obj_xy       : (T, 2) float       object center xy (m), for lateral drift
  obj_angvel   : (T,)   float       object angular-speed magnitude (rad/s), for jitter
Phase boundaries (policy steps):
  grasp_end    : int  end of the grasp/lift window (first-contact is judged here)
  hold_start   : int  start of the steady hold/reorient window (balance/force/drift here)
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

FINGERS = ("thumb", "index", "middle")

# ---- thresholds (documented; tuned to the screwdriver_medium morphohand task) --------
FLOOR_Z = 0.05             # object-center z below this = on/near the floor (a drop)
LATE_SPREAD_STEPS = 12     # spread in per-finger first-contact step > this = late finger
LATE_ABS_STEPS = 30        # a finger whose FIRST contact is later than this (others early)
IDLE_TOUCH_FRAC = 0.5      # a finger touching < this fraction of the hold window = idle
IDLE_FORCE_N = 1.0         # ... or carrying < this mean force = idle (degenerate pinch)
CONTACT_COUNT_MIN = 2.6    # mean tip-contact count in hold < this = not a firm tripod
ANG_JERK_WARN = 20.0       # object angular jerk (mean |Δangvel|/step, 1/s^2)
ANG_JERK_FAIL = 40.0
DRIFT_NET_CM = 3.0         # net lateral drift of the object center over the hold (cm)
SLIDE_RATIO = 3.0          # path-length / net-displacement > this = sliding around
OVERCLAMP_N = 5.0          # mean fingertip force > this = over-clamp (grip reward sat's ~3N)


@dataclass
class Check:
    name: str
    status: str            # "PASS" | "WARN" | "FAIL"
    value: float
    detail: str = ""


@dataclass
class Scorecard:
    checks: list[Check] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    @property
    def verdict(self) -> str:
        s = {c.status for c in self.checks}
        return "FAIL" if "FAIL" in s else ("WARN" if "WARN" in s else "PASS")

    @property
    def failed(self) -> list[str]:
        return [c.name for c in self.checks if c.status == "FAIL"]

    def as_dict(self) -> dict:
        return {"verdict": self.verdict,
                "checks": {c.name: {"status": c.status, "value": round(float(c.value), 4),
                                    "detail": c.detail} for c in self.checks},
                "metrics": {k: (round(float(v), 4) if np.isscalar(v) else v)
                            for k, v in self.metrics.items()}}


def _first_contact_steps(finger_found: np.ndarray, window: slice) -> list[int]:
    """First step (within window) each finger makes contact; -1 if never."""
    out = []
    ff = finger_found[window] > 0.5
    for f in range(ff.shape[1]):
        idx = np.argmax(ff[:, f]) if ff[:, f].any() else -1
        out.append(int(idx) if idx != -1 or ff[0, f] else (int(idx) if ff[:, f].any() else -1))
    # argmax returns 0 when none True; guard: only valid if that step is actually contact
    return [int(np.argmax(ff[:, f])) if ff[:, f].any() else -1 for f in range(ff.shape[1])]


def characterize_trajectory(finger_found, finger_force, obj_z, obj_cos, obj_xy,
                            obj_angvel, *, grasp_end: int, hold_start: int) -> Scorecard:
    finger_found = np.asarray(finger_found, float)
    finger_force = np.asarray(finger_force, float)
    obj_z = np.asarray(obj_z, float)
    obj_cos = np.asarray(obj_cos, float)
    obj_xy = np.asarray(obj_xy, float)
    obj_angvel = np.asarray(obj_angvel, float)
    T = len(obj_z)
    grasp_end = min(grasp_end, T)
    hold_start = min(hold_start, T - 1)
    sc = Scorecard()

    # ---- 1. LATE FINGER (the delayed-finger bug) --------------------------------------
    fc = _first_contact_steps(finger_found, slice(0, grasp_end))
    contacted = [s for s in fc if s >= 0]
    n_never = sum(1 for s in fc if s < 0)
    spread = (max(contacted) - min(contacted)) if len(contacted) >= 2 else 0
    late_detail = ", ".join(f"{FINGERS[f]}@{fc[f]}" for f in range(3))
    if n_never > 0:
        sc.checks.append(Check("late_finger", "FAIL", spread,
                               f"{n_never} finger(s) NEVER contact in grasp ({late_detail})"))
    elif spread > LATE_SPREAD_STEPS or max(contacted) > LATE_ABS_STEPS:
        sc.checks.append(Check("late_finger", "FAIL", spread,
                               f"first-contact spread {spread} steps ({late_detail})"))
    else:
        sc.checks.append(Check("late_finger", "PASS", spread, f"({late_detail})"))

    # ---- 2. IDLE FINGER / 2-finger grasp (degenerate pinch) ---------------------------
    hold = slice(hold_start, T)
    touch_frac = (finger_found[hold] > 0.5).mean(axis=0)
    force_mean = finger_force[hold].mean(axis=0)
    contact_count = (finger_found[hold] > 0.5).sum(axis=1).mean()
    idle = [FINGERS[f] for f in range(3)
            if touch_frac[f] < IDLE_TOUCH_FRAC or force_mean[f] < IDLE_FORCE_N]
    bal_detail = " ".join(f"{FINGERS[f]}={force_mean[f]:.1f}N/{touch_frac[f]:.2f}" for f in range(3))
    if idle:
        sc.checks.append(Check("idle_finger", "FAIL", float(min(force_mean)),
                               f"idle: {','.join(idle)} | {bal_detail}"))
    elif contact_count < CONTACT_COUNT_MIN:
        sc.checks.append(Check("idle_finger", "WARN", float(contact_count),
                               f"mean tip-contacts {contact_count:.2f}<3 | {bal_detail}"))
    else:
        sc.checks.append(Check("idle_finger", "PASS", float(contact_count), bal_detail))

    # ---- 3. DROP ---------------------------------------------------------------------
    min_z_hold = float(obj_z[hold].min()) if T > hold_start else float(obj_z.min())
    sc.checks.append(Check("drop", "FAIL" if min_z_hold < FLOOR_Z else "PASS", min_z_hold,
                           f"min hold-phase obj-z {min_z_hold:.3f} m (floor<{FLOOR_Z})"))

    # ---- 4. JITTER (object angular jerk) ---------------------------------------------
    ang_jerk = float(np.abs(np.diff(obj_angvel[hold])).mean()) * 50.0 if T - hold_start > 2 else 0.0
    jstatus = "FAIL" if ang_jerk > ANG_JERK_FAIL else ("WARN" if ang_jerk > ANG_JERK_WARN else "PASS")
    sc.checks.append(Check("jitter", jstatus, ang_jerk, f"ang-jerk {ang_jerk:.1f} 1/s^2"))

    # ---- 5. DE-CENTERING / SLIDE -----------------------------------------------------
    ref = obj_xy[hold_start] if hold_start < T else obj_xy[0]
    net_cm = float(np.linalg.norm(obj_xy[-1] - ref)) * 100.0
    path_cm = float(np.linalg.norm(np.diff(obj_xy[hold_start:], axis=0), axis=1).sum()) * 100.0
    slide_ratio = path_cm / max(net_cm, 0.1)
    if net_cm > DRIFT_NET_CM:
        sc.checks.append(Check("de_centering", "FAIL", net_cm,
                               f"net lateral drift {net_cm:.1f} cm (path {path_cm:.1f})"))
    elif slide_ratio > SLIDE_RATIO and path_cm > 2.0:
        sc.checks.append(Check("de_centering", "WARN", slide_ratio,
                               f"sliding: path {path_cm:.1f}cm >> net {net_cm:.1f}cm"))
    else:
        sc.checks.append(Check("de_centering", "PASS", net_cm,
                               f"net {net_cm:.1f}cm path {path_cm:.1f}cm"))

    # ---- 6. OVER-CLAMP (grip force) --------------------------------------------------
    tip_force = float(finger_force[hold].sum(axis=1).mean() / 3.0)
    sc.checks.append(Check("over_clamp", "WARN" if tip_force > OVERCLAMP_N else "PASS", tip_force,
                           f"mean fingertip {tip_force:.1f} N (reward saturates ~3N)"))

    sc.metrics.update(
        first_contact=fc, contact_spread=spread, touch_frac=touch_frac.round(2).tolist(),
        force_mean=force_mean.round(1).tolist(), contact_count=round(float(contact_count), 2),
        min_z_hold=round(min_z_hold, 4), ang_jerk=round(ang_jerk, 1),
        net_drift_cm=round(net_cm, 1), slide_ratio=round(slide_ratio, 1),
        tip_force=round(tip_force, 1), held_cos_tail=round(float(obj_cos[hold].mean()), 3),
        peak_cos=round(float(obj_cos.max()), 3),
    )
    return sc


_ICON = {"PASS": "✓", "WARN": "▲", "FAIL": "✗"}


def format_scorecard(sc: Scorecard, title: str = "") -> str:
    lines = [f"┌── policy health: {title}  =>  VERDICT: {sc.verdict}"]
    for c in sc.checks:
        lines.append(f"│ {_ICON[c.status]} {c.status:4s} {c.name:13s} {c.detail}")
    lines.append("└" + "─" * 40)
    return "\n".join(lines)
