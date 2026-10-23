#!/usr/bin/env python3
"""The board's zero over a calibration session of the fingertip on the UR5: how it drifts, what a press leaves behind,
and what subtracting it buys the viewer's estimator (owner, 2026-10-23: "track drift over time and see if we can learn
to subtract whatever accumulates, plus the existing estimator").

The rests: every action's rest before and after it (0.5 s out of contact at the action's rest pose), as the board's mean
reading after the chips' temperature correction (Session.b of scripts/ur5_tip_calibration_analysis.py). Their changes
split into three parts:

  field      Earth's field turning with the tool: the lab's fit to first2's gravity stations
             (lab/20261022-earth_field_fit.json, B_k = b0_k + A_k R_BE^T w), subtracted at every rest and hold
  drift      a slow change in time (and with the chips' temperature, after their gain correction)
  steps      what a press leaves: its rest after less its rest before. Whether a step stays (the next action's rest
             before matches it) or decays (the next rest goes back) is read from the rest that follows.

Two learned models of the zero are fitted on the session's earlier actions in time order and tested on the later ones:
the zero against time alone (the drift since the first rest, as eFlesh subtracts it), and the zero as the first rest plus
the steps predicted from each press's features (its top force, its place on the pad and its time in contact) and a drift
in time. They answer whether what accumulates can be predicted rather than measured at rest.

The holds and their ground truth come from the same analysis (Session.holds; the four presses that carried load off the
pad are left out there), and the estimator is the viewer's (estimator()). Each hold's board change is taken against these
zeros, each with and without the field term's change from the zero's orientation to the hold's:

  first      the session's first rest, never re-zeroed
  learned    the first rest plus the learned steps and drift (trained on the earlier actions)
  before     the action's own rest before it: a re-zero at rest just before contact, as the viewer's auto-zero gives
  both       the rests before and after interpolated to the hold (offline, the analysis's own)

    /usr/bin/python3 scripts/ur5_tip_calibration_drift.py [--session DIR] [--gravity-from DIR] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src/morphohand/driver/reskin5x/host"))
import ur5_tip_calibration_analysis as A  # noqa: E402

LAB = ROOT / "docs/experiments/20261022-ur5_tip_calibration/lab"
SESSION = LAB / "20261022-210852-sphere2"
FIRST2 = LAB / "20261022-195420-first2"
EARTH = LAB / "20261022-earth_field_fit.json"
OUT = ROOT / "docs/experiments/20261022-ur5_tip_calibration/data/20261023-board_drift"
TRAIN = 0.5                 # the share of the actions, in time order, that the learned zeros are fitted on


class Field:
    """Earth's field in each chip's axes for a flange rotation R_BE (15,), from the lab's fit."""

    def __init__(self, path=EARTH):
        d = json.loads(Path(path).read_text())
        self.w = np.array(d["w_base_uT"], float)
        self.A = np.array(d["A_k"], float)

    def __call__(self, R_BE):
        return np.concatenate([self.A[k] @ np.asarray(R_BE).T @ self.w for k in range(5)])


def rests(ses, field):
    """Every rest of the session in time order: the action, before or after, its time, the reading less the field term
    (15,), the field term itself, the chips' temperature, and the action's top force and time in contact."""
    out = []
    for e in ses.d["ev"]:
        rec = e.get("rec")
        if e.get("event") != "end" or not isinstance(rec, dict):
            continue
        a = ses.acts.get(e["id"], {})
        top = rec["levels"][-1]["N"] if rec.get("levels") else a.get("stop_N")      # a touch-off stops at 0.5 N
        t_in = (rec["t_rest1"][0] - rec["t_move0"]) if rec.get("t_rest1") and rec.get("t_move0") else None
        for which in ("t_rest0", "t_rest1"):
            w = rec.get(which)
            if not w:
                continue
            m = (ses.d["t_b"] >= w[0]) & (ses.d["t_b"] <= w[1])
            T_BE = ses.pose(*w)
            if m.sum() < 20 or T_BE is None:
                continue
            f = field(T_BE[:3, :3])
            out.append(dict(id=e["id"], status=e.get("status"), stage=a.get("stage", ""), kind=a.get("kind", ""),
                            region=(a.get("target") or {}).get("region", ""),
                            target=np.array((a.get("target") or {}).get("p", [np.nan] * 3), float),
                            which="before" if which == "t_rest0" else "after", t=0.5 * (w[0] + w[1]),
                            raw=ses.b[m].mean(axis=0), field=f, z=ses.b[m].mean(axis=0) - f,
                            T=ses.T_chip[m].mean(axis=0), top_N=top, t_in=t_in))
    out.sort(key=lambda r: r["t"])
    return out


def steps(rs):
    """Each action with both rests: its step (after less before, field removed), and what the next rest before keeps of
    it (carry = next before less this after: 0 if the step stays, -step if it goes back)."""
    out = []
    for i, r in enumerate(rs):
        if r["which"] != "before":
            continue
        after = next((x for x in rs[i + 1:i + 3] if x["id"] == r["id"] and x["which"] == "after"), None)
        if after is None:
            continue
        nxt = next((x for x in rs[i + 1:] if x["which"] == "before" and x["id"] != r["id"]), None)
        s = after["z"] - r["z"]
        row = dict(id=r["id"], kind=r["kind"], stage=r["stage"], region=r["region"], target=r["target"], status=r["status"],
                   t0=r["t"], t1=after["t"], step=s, top_N=r["top_N"], t_in=r["t_in"], dT=after["T"] - r["T"])
        if nxt is not None:
            row.update(carry=nxt["z"] - after["z"], gap_s=nxt["t"] - after["t"])
        out.append(row)
    return out


def features(st):
    """A press's features for its step: its top force, its time in contact, its target on the pad (P frame, mm) and
    its region, with a constant."""
    reg = ["face", "edge", "flank"]
    p = np.nan_to_num(st["target"])
    return np.r_[1.0, st["top_N"] or 0.0, (st["t_in"] or 0.0) / 60.0, p / 10.0,
                 [1.0 if st["region"] == r else 0.0 for r in reg]]


def ridge(X, Y, lam=1.0):
    """Least squares with a ridge on all but the constant: Y (n, 15) on X (n, k)."""
    P = lam * np.eye(X.shape[1])
    P[0, 0] = 0.0
    return np.linalg.solve(X.T @ X + P, X.T @ Y)


def fit_decay(st, t_split):
    """The share of a step that the next rest still holds against the time to it, as a + (1 - a) exp(-gap / tau),
    weighted by the step's size, on the steps over 3 uT ended before t_split. Returns (a, tau_s)."""
    rows = [s for s in st if "carry" in s and s["t1"] <= t_split and np.linalg.norm(s["step"]) > 3.0]
    g = np.array([s["gap_s"] for s in rows])
    k = np.array([1.0 + (s["carry"] @ s["step"]) / (s["step"] @ s["step"]) for s in rows])
    w = np.array([np.linalg.norm(s["step"]) for s in rows])
    best = None
    for tau in np.r_[np.arange(2.0, 60.0, 1.0), np.arange(60.0, 1200.0, 10.0)]:
        e = np.exp(-g / tau)
        a = float(np.clip(np.sum(w * (1 - e) * (k - e)) / np.sum(w * (1 - e) ** 2), 0.0, 1.0))
        r = float(np.sum(w * (k - (a + (1 - a) * e)) ** 2))
        if best is None or r < best[0]:
            best = (r, a, float(tau))
    return best[1], best[2]


def kept(dt, decay):
    a, tau = decay
    return a + (1.0 - a) * math.exp(-max(dt, 0.0) / tau)


class LearnedZero:
    """The zero at time t without re-zeroing: the first rest plus a drift in time and the steps of the actions ended
    before t, each step predicted from its press's features and decaying as fit_decay found. Fitted on the rests and steps
    before t_split; time_only keeps the drift in time alone (the zero against time since the first rest)."""

    def __init__(self, rs, st, t_split, decay, time_only=False):
        self.t0 = rs[0]["t"]
        self.z0 = rs[0]["z"]
        self.decay = decay
        tr = [s for s in st if s["t1"] <= t_split and s["top_N"] is not None]
        self.B = ridge(np.array([features(s) for s in tr]), np.array([s["step"] for s in tr])) if tr and not time_only else None
        self.st = st
        # the drift in time: fitted on the rests before t_split, after the steps so far are taken out
        rr = [r for r in rs if r["t"] <= t_split]
        tt = np.array([(r["t"] - self.t0) / 60.0 for r in rr])
        Z = np.array([r["z"] - self.z0 - self.steps_before(r["t"]) for r in rr])
        self.v = np.linalg.lstsq(np.c_[tt], Z, rcond=None)[0][0] if len(rr) > 2 else np.zeros(15)

    def steps_before(self, t):
        if self.B is None:
            return np.zeros(15)
        done = [s for s in self.st if s["t1"] <= t and s["top_N"] is not None]
        return sum((features(s) @ self.B * kept(t - s["t1"], self.decay) for s in done), np.zeros(15))

    def __call__(self, t):
        return self.z0 + self.v * (t - self.t0) / 60.0 + self.steps_before(t)


def recovery(st, h, decay):
    """How far the earlier actions' steps move the zero from the hold's rest before to the hold, as they keep decaying
    (15,): what a zero that re-zeroes at rest and knows the decay adds."""
    out = np.zeros(15)
    for s in st:
        if s["t1"] < h["t_rest0"] - 0.5:
            out += s["step"] * (kept(h["t"] - s["t1"], decay) - kept(h["t_rest0"] - s["t1"], decay))
    return out


def alignment(ses, st):
    """For each press with a step over 3 uT: the cosine between its step and its top hold's change from the rest
    before (both 15-vectors, at the same pose), and the step's share along that change. Random directions in 15
    dimensions give a median |cosine| of 0.19."""
    by = {s["id"]: s for s in st}
    out = []
    for aid, e in ses.ends.items():
        rec = e.get("rec") or {}
        if e.get("status") != "ok" or not rec.get("levels") or aid not in by or np.linalg.norm(by[aid]["step"]) < 3.0:
            continue
        t0, t1 = rec["levels"][-1]["t"]
        m = (ses.d["t_b"] >= t0) & (ses.d["t_b"] <= t1)
        m0 = (ses.d["t_b"] >= rec["t_rest0"][0]) & (ses.d["t_b"] <= rec["t_rest0"][1])
        if m.sum() < 5 or m0.sum() < 5:
            continue
        d = ses.b[m].mean(axis=0) - ses.b[m0].mean(axis=0)
        s = by[aid]["step"]
        if np.linalg.norm(d) < 10.0:
            continue
        out.append(dict(id=aid, top_N=rec["levels"][-1]["N"], change_uT=float(np.linalg.norm(d)),
                        step_uT=float(np.linalg.norm(s)), cos=float(s @ d / np.linalg.norm(s) / np.linalg.norm(d)),
                        share=float(s @ d / (d @ d))))
    return out


def hold_rows(ses, field):
    """The analysis's holds with the readings and orientations each zero needs."""
    holds = ses.holds()
    for h in holds:
        rec = ses.ends[h["id"]]["rec"]
        rest, _ = ses.rest_board(rec, h["t"])
        h["b_hold"] = h["d_board"] + rest
        R0 = ses.pose(*rec["t_rest0"])[:3, :3]
        R1 = ses.pose(*rec["t_rest1"])[:3, :3] if rec.get("t_rest1") else R0
        h["f_hold"] = field(np.asarray(h["R_BE"]).reshape(3, 3))
        h["f_rest0"], h["f_rest1"] = field(R0), field(R1)
        m0 = (ses.d["t_b"] >= rec["t_rest0"][0]) & (ses.d["t_b"] <= rec["t_rest0"][1])
        h["b_rest0"] = ses.b[m0].mean(axis=0)
        if rec.get("t_rest1"):
            m1 = (ses.d["t_b"] >= rec["t_rest1"][0]) & (ses.d["t_b"] <= rec["t_rest1"][1])
            h["b_rest1"] = ses.b[m1].mean(axis=0)
        else:
            h["b_rest1"] = h["b_rest0"]
        h["t_rest0"], h["t_rest1"] = float(np.mean(rec["t_rest0"])), float(np.mean(rec.get("t_rest1") or rec["t_rest0"]))
    return holds


def zeros(h, first, learned, st, decay):
    """The board's change at hold h against each zero (15,), keyed by name."""
    a = (h["t"] - h["t_rest0"]) / max(h["t_rest1"] - h["t_rest0"], 1e-6)
    z_before = h["b_rest0"]
    z_both = h["b_rest0"] + a * (h["b_rest1"] - h["b_rest0"])
    f_both = h["f_rest0"] + a * (h["f_rest1"] - h["f_rest0"])
    out = {
        "first": h["b_hold"] - (first["z"] + first["field"]),
        "first+field": h["b_hold"] - h["f_hold"] - first["z"],
        "time+field": h["b_hold"] - h["f_hold"] - h["time_zero"],
        "learned+field": h["b_hold"] - h["f_hold"] - learned(h["t"]),
        "before": h["b_hold"] - z_before,
        "before+field": h["b_hold"] - z_before - (h["f_hold"] - h["f_rest0"]),
        "decay+field": h["b_hold"] - z_before - (h["f_hold"] - h["f_rest0"]) - recovery(st, h, decay),
        "both": h["b_hold"] - z_both,
        "both+field": h["b_hold"] - z_both - (h["f_hold"] - f_both),
    }
    return out


def score(holds, key, est):
    """The estimator on each hold's change against zero key: the tangential distance from the true contact point (mm),
    the normal force's error (N) and its relative error (above 0.4 N), and whether it found a contact."""
    for h in holds:
        est.set_noise_scale(1.0 / math.sqrt(max(h["n_rec"], 1)))
        e = est.estimate(h["dz"][key])
        h.setdefault("res", {})[key] = e


def stats(holds, key, sel):
    hh = [h for h in holds if sel(h)]
    loc, fe, rel, found = [], [], [], 0
    for h in hh:
        e = h["res"][key]
        if not e.get("contact"):
            fe.append(abs(h["f_n"]))
            continue
        found += 1
        loc.append(float(np.linalg.norm(A.tangential(np.array(e["p"]) - h["p"], h["n"]))))
        fe.append(abs(e["f_n"] - h["f_n"]))
        if h["f_n"] > 0.4:
            rel.append(abs(e["f_n"] / h["f_n"] - 1.0))
    q = lambda v, p: float(np.percentile(v, p)) if v else None          # noqa: E731
    return dict(n=len(hh), found=found, loc_med=q(loc, 50), loc_p90=q(loc, 90), f_med=q(fe, 50), f_p90=q(fe, 90),
                rel_med=q(rel, 50))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--session", default=str(SESSION))
    ap.add_argument("--gravity-from", default=str(FIRST2))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    A.GRAVITY_FROM = a.gravity_from
    ses = A.Session(a.session)
    field = Field()
    rs = rests(ses, field)
    st = steps(rs)
    t_all = np.array([r["t"] for r in rs])
    t_split = float(np.quantile(t_all, TRAIN))
    decay = fit_decay(st, t_split)
    learned = LearnedZero(rs, st, t_split, decay)
    time_only = LearnedZero(rs, st, t_split, decay, time_only=True)

    # the rests: the field's share, the drift in time, the steps and what stays of them
    Z = np.array([r["z"] - rs[0]["z"] for r in rs])
    Rw = np.array([r["raw"] - rs[0]["raw"] for r in rs])
    t_min = (t_all - t_all[0]) / 60.0
    S = np.array([s["step"] for s in st])
    has = [s for s in st if "carry" in s]
    keep = np.array([1.0 + (s["carry"] @ s["step"]) / max(s["step"] @ s["step"], 1e-9) for s in has])
    test_r = [r for r in rs if r["t"] > t_split]
    miss = {name: np.array([np.linalg.norm(r["z"] - m(r["t"])) for r in test_r])
            for name, m in (("learned", learned), ("time", time_only))}
    miss["first"] = np.array([np.linalg.norm(r["z"] - rs[0]["z"]) for r in test_r])
    # the steps' learnability: their prediction on the later actions against their size
    test_s = [s for s in st if s["t0"] > t_split and s["top_N"] is not None]
    pred_err = np.array([np.linalg.norm(s["step"] - features(s) @ learned.B) for s in test_s]) if learned.B is not None else None
    size = np.array([np.linalg.norm(s["step"]) for s in test_s])

    # the holds against each zero, through the viewer's estimator
    holds = hold_rows(ses, field)
    for h in holds:
        h["time_zero"] = time_only(h["t"])
        h["dz"] = zeros(h, rs[0], learned, st, decay)
    cal, est = A.estimator(ses, holds)
    keys = list(holds[0]["dz"])
    for k in keys:
        score(holds, k, est)
    later = lambda h: h["t"] > t_split                                       # noqa: E731   the learned zeros' test half
    groups = {
        "presses at 0.25 and 0.5 N": lambda h: h["kind"] == "press" and (h["level_N"] or 0) < 0.9 and later(h),
        "presses at 1 N": lambda h: h["kind"] == "press" and 0.9 <= (h["level_N"] or 0) < 1.5 and later(h),
        "presses at 2 N": lambda h: h["kind"] == "press" and 1.5 <= (h["level_N"] or 0) < 2.5 and later(h),
        "presses at 3 to 5 N": lambda h: h["kind"] == "press" and (h["level_N"] or 0) >= 2.5 and later(h),
        "rolls and twists": lambda h: h["kind"] in ("roll", "twist") and later(h),
        "presses from 1 N": lambda h: h["kind"] == "press" and (h["level_N"] or 0) >= 0.9 and later(h),
    }
    table = {g: {k: stats(holds, k, sel) for k in keys} for g, sel in groups.items()}
    # the estimator's normal force against the HEX-E's: the median signed error and ratio, by group and zero
    bias = {g: {k: dict(err=float(np.median([(h["res"][k].get("f_n") if h["res"][k].get("contact") else 0.0) - h["f_n"]
                                             for h in holds if sel(h)])),
                        ratio=float(np.median([(h["res"][k].get("f_n") if h["res"][k].get("contact") else 0.0) / h["f_n"]
                                               for h in holds if sel(h) and h["f_n"] > 0.2])))
                for k in keys} for g, sel in groups.items()}
    al = alignment(ses, st)
    # the location error as a vector (P frame, tangential) from 2 N up on the second half: its mean (the bias) and the
    # scatter about it, by zero; a zero whose offset cancels part of the estimator's bias shows a smaller mean, not a
    # smaller scatter
    hb = [h for h in holds if h["kind"] == "press" and (h["level_N"] or 0) >= 1.9 and later(h)]
    loc_bias = {}
    for k in keys:
        E = np.array([A.tangential(np.array(h["res"][k]["p"]) - h["p"], h["n"]) for h in hb if h["res"][k].get("contact")])
        loc_bias[k] = dict(n=len(E), mean=E.mean(axis=0).tolist(), mean_norm=float(np.linalg.norm(E.mean(axis=0))),
                           median=float(np.median(np.linalg.norm(E, axis=1))),
                           scatter=float(np.median(np.linalg.norm(E - E.mean(axis=0), axis=1))))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    R = dict(
        session=str(Path(a.session).relative_to(ROOT)), n_rests=len(rs), minutes=float(t_min[-1]), t_split_min=float((t_split - t_all[0]) / 60.0),
        raw_span_uT=float(np.ptp(Rw, axis=0).max()), field_span_uT=float(np.ptp(Z, axis=0).max()),
        field_share=float(1.0 - np.sum((Z - Z.mean(0)) ** 2) / np.sum((Rw - Rw.mean(0)) ** 2)),
        n_steps=len(st), step_med_uT=float(np.median(np.linalg.norm(S, axis=1))),
        step_p90_uT=float(np.percentile(np.linalg.norm(S, axis=1), 90)),
        keep_med=float(np.median(keep)), keep_iqr=[float(np.percentile(keep, 25)), float(np.percentile(keep, 75))],
        gap_med_s=float(np.median([s["gap_s"] for s in has])),
        accum_end_uT=float(np.linalg.norm(Z[-1])), steps_sum_uT=float(np.linalg.norm(S.sum(axis=0))),
        steps_rms_walk_uT=float(math.sqrt(np.sum(np.linalg.norm(S, axis=1) ** 2))),
        decay_a=decay[0], decay_tau_s=decay[1], drift_uT_min=float(np.linalg.norm(learned.v)), drift_time_only_uT_min=float(np.linalg.norm(time_only.v)),
        miss_med_uT={k: float(np.median(v)) for k, v in miss.items()},
        step_pred_med_uT=float(np.median(pred_err)) if pred_err is not None else None,
        step_size_med_uT=float(np.median(size)) if len(size) else None,
        n_holds=len(holds), table=table, bias=bias, align=al, loc_bias=loc_bias,
        align_cos_med=float(np.median([x["cos"] for x in al])), align_pos=float(np.mean([x["cos"] > 0 for x in al])),
        align_share_med=float(np.median([x["share"] for x in al])),
        decay_corr_med_uT=float(np.median([np.linalg.norm(recovery(st, h, decay)) for h in holds])),
        decay_corr_p90_uT=float(np.percentile([np.linalg.norm(recovery(st, h, decay)) for h in holds], 90)),
        long_gaps=int(sum(1 for s in st if s.get("gap_s", 0) > 60 and np.linalg.norm(s["step"]) > 3)),
        series=dict(t_min=t_min.tolist(), z_norm=np.linalg.norm(Z, axis=1).tolist(), raw_norm=np.linalg.norm(Rw, axis=1).tolist(),
                    which=[r["which"] for r in rs], kind=[r["kind"] for r in rs],
                    learned=[float(np.linalg.norm(learned(r["t"]) - rs[0]["z"])) for r in rs],
                    time_only=[float(np.linalg.norm(time_only(r["t"]) - rs[0]["z"])) for r in rs]),
        step_rows=[dict(id=s["id"], kind=s["kind"], region=s["region"], t_min=(s["t0"] - t_all[0]) / 60.0,
                        step_uT=float(np.linalg.norm(s["step"])), top_N=s["top_N"], t_in_s=s["t_in"],
                        keep=(1.0 + (s["carry"] @ s["step"]) / max(s["step"] @ s["step"], 1e-9)) if "carry" in s else None,
                        gap_s=s.get("gap_s")) for s in st],
        hold_rows=[dict(id=h["id"], kind=h["kind"], level_N=h["level_N"], turn_deg=h["turn_deg"], t_min=(h["t"] - t_all[0]) / 60.0,
                        field_change_uT=float(np.linalg.norm(h["f_hold"] - h["f_rest0"])),
                        decay_uT=float(np.linalg.norm(recovery(st, h, decay))),
                        **{f"loc_{k}": (float(np.linalg.norm(A.tangential(np.array(h["res"][k]["p"]) - h["p"], h["n"])))
                                        if h["res"][k].get("contact") else None) for k in keys},
                        **{f"fn_{k}": (h["res"][k].get("f_n") if h["res"][k].get("contact") else 0.0) for k in keys},
                        f_n=h["f_n"]) for h in holds],
    )
    (out / "drift.json").write_text(json.dumps(R, indent=1, default=float))
    print(json.dumps({k: v for k, v in R.items() if k not in ("series", "step_rows", "hold_rows", "table")}, indent=1))
    for g, row in table.items():
        print(f"== {g}")
        for k, s in row.items():
            print(f"  {k:14s} n {s['n']:3d} found {s['found']:3d}  loc {s['loc_med'] or float('nan'):.2f} / "
                  f"{s['loc_p90'] or float('nan'):.2f} mm  force {s['f_med'] or float('nan'):.3f} / {s['f_p90'] or float('nan'):.3f} N"
                  f"  rel {s['rel_med'] if s['rel_med'] is not None else float('nan'):.2f}")
    print(f"-> {out / 'drift.json'}")


if __name__ == "__main__":
    main()
