#!/usr/bin/env python3
"""HOM's commanded targets on the hand, replayed offline with and without the torque budget (hand-4a, 2026-11-20 11:5x;
the lab's 20261120-command_jitter.py found the commands reversing in 37-55 % of frames since the budget of 11-16 evening;
owner, 11:4x: "this jitter is really HOM commanded? so interesting").

Each HOM server record (runs/<stamp>_twist_d40_trial<n>.jsonl) is fed period by period, in process, to a HomBench built
as hom_server._bench built it (the header's plan, plant, hand, dt, budget and HOM options): every period's time, the
joints the server passed (bus, or its fallbacks), the last tracked pose and its capture time, and the fingertip contacts
when the trial gave them to HOM. Three command series per record:
    recorded   the targets the server sent (targets_sim_deg)
    replay     the replay with the record's budget; equal to the recorded targets when the replay is faithful
    raw        the replay without the budget: HOM's own targets for the same measurements, before any cut
    smooth_j   the replay with the budget's offsets taken from the joints low-passed at 0.1 s (hom_bench.HomBench
               budget_joint_tau, added 2026-11-20 12:1x)
    smooth_jt  the same with HOM's targets also low-passed at 0.04 s before the budget (target_tau)
The replay is open loop: the measured joints are the hand's response to the recorded commands, so "raw" is HOM's answer
to the hand's measurements, not the trial HOM would have run without the budget.

Metrics over the periods from the end of the lift (the phases after open, close, squeeze and lift: wz +, wz -,
rest, twist ramp, hold; t >= 2.0 s) to the record's end, per series (sim degrees):
    rev %    the share of periods where a joint's command reverses its direction of travel (steps over 0.1 deg), over
             the nine joints (the lab's definition)
    hf       RMS of the command about its 7-period moving average, mean over the nine joints, deg
    step     RMS of the command's change per period over the nine joints, deg
and per record the budget's scale (the share of periods below 1, its mean, and the RMS of its change per period).

    .venv/bin/python docs/experiments/20261115-hom_bench_trials/analysis/20261120-command_jitter_replay.py [STAMP ...]
Writes 20261120-command_jitter_replay.json next to this script and prints the table.
"""
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
ROOT = BENCH.parents[2]
RUNS = BENCH / "runs"
OUT = HERE / "20261120-command_jitter_replay.json"
sys.path.insert(0, str(ROOT / "src"))


def hf(X, w=7):
    k = np.ones(w) / w
    sm = np.column_stack([np.convolve(X[:, i], k, mode="same") for i in range(X.shape[1])])
    return float(np.mean(np.sqrt(np.mean((X - sm)[w:-w] ** 2, axis=0))))


def rev(X, eps=0.1):
    d = np.diff(X, axis=0)
    s = np.sign(np.where(np.abs(d) > eps, d, 0))
    a, b = s[1:], s[:-1]
    m = (a != 0) & (b != 0)
    return float(np.mean(a[m] != b[m])) if m.any() else 0.0


def metrics(X):
    X = np.asarray(X, float)
    if len(X) < 20:
        return None
    return {"rev_pct": round(100 * rev(X), 1), "hf_deg": round(hf(X), 3),
            "step_deg": round(float(np.sqrt(np.mean(np.diff(X, axis=0) ** 2))), 3)}


def load(p):
    lines = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    head = next(x for x in lines if x.get("kind") == "header")
    rows = [x for x in lines if "kind" not in x and "seq" in x]
    return head, rows


PRE = ("open", "close", "squeeze", "lift", None)
_BENCH = {}


def bench_for(head, budget, **smooth):
    """A HomBench as hom_server._bench built it for this header, with the given budget (None: none)."""
    from sr2hand.control import hom_bench as HB
    key = (head["plan"], head["task"], float(head["d_mm"]), float(head["dt"]), budget, head.get("tips"),
           head.get("contact_force_min_N"), head.get("contact_force_off_N"), head.get("filter_step_s"),
           bool(head.get("filter_exact")), head.get("eq1_frames", "measured"), tuple(sorted(smooth.items())))
    if key not in _BENCH:
        if any(k[0] != key[0] for k in _BENCH):          # one plan's controllers at a time (about 0.1 GB each)
            _BENCH.clear()
        b = HB.HomBench.from_plan(ROOT / head["plan"], head["task"], float(head["d_mm"]), float(head["dt"]),
                                  torque_budget=budget, pose_tau=None, **smooth)
        over = {}
        if head.get("contact_force_min_N") is not None:
            over["contact_force_min"] = float(head["contact_force_min_N"])
        if head.get("contact_force_off_N") is not None:
            over["contact_force_off"] = float(head["contact_force_off_N"])
        if head.get("filter_step_s") or head.get("filter_exact"):
            over["filter"] = replace(b.prm.filter, step_s=head.get("filter_step_s") or None,
                                     exact=bool(head.get("filter_exact")))
        if head.get("eq1_frames", "measured") != "measured":
            over["eq1_frames"] = head["eq1_frames"]
        if over:
            b.prm = replace(b.prm, **over)
        _BENCH[key] = b
    b = _BENCH[key]
    b.reset()
    return b


def replay(head, rows, budget, **smooth):
    """[(t, phase, targets in sim degrees (9,), budget scale)] of the record's periods replayed with the budget."""
    from sr2hand.control import hom_bench as HB
    from sr2hand.control import hom_server as HS
    from sr2hand.control import servo_limits as SL
    b = bench_for(head, budget, **smooth)
    out = []
    for r in rows:
        if r.get("obj_pos_mm") is None or r.get("error"):
            out.append(None)
            continue
        cap = r.get("t_capture_ms")
        age = (r["t_ms"] - cap) / 1000.0 if cap is not None and math.isfinite(cap) else None
        try:
            tg, info = b.step(float(r["t"]), r["joints"], np.asarray(r["obj_pos_mm"], float),
                              np.asarray(r["obj_quat"], float),
                              r.get("contacts") if head.get("tips", "off") != "off" else None, frame_age_s=age,
                              frame=cap if cap is not None and math.isfinite(cap) else None)
        except Exception as e:                  # noqa: BLE001
            out.append({"error": repr(e)[:120]})
            continue
        SL.clip_servo(tg)
        out.append((float(r["t"]), info["phase"], np.asarray(HS._sim_deg(HB, tg), float), info.get("budget_scale")))
    return out


def one(p):
    head, rows = load(p)
    if head.get("task") != "twist" or len(rows) < 60:
        return None
    rec_t = np.array([r["targets_sim_deg"] for r in rows], float)
    phase = [r.get("phase") for r in rows]
    task = [i for i, ph in enumerate(phase) if ph not in PRE]
    if len(task) < 30:
        return {"record": p.name, "note": "fewer than 30 task periods"}
    i0 = task[0]
    tb = head.get("torque_budget")
    rep = replay(head, rows, tb)
    raw = replay(head, rows, None)
    sj = replay(head, rows, tb, budget_joint_tau=0.1) if tb else None
    sjt = replay(head, rows, tb, budget_joint_tau=0.1, target_tau=0.04) if tb else None
    ok = [i for i in range(i0, len(rows)) if isinstance(rep[i], tuple) and isinstance(raw[i], tuple)]
    if len(ok) < 30:
        return {"record": p.name, "note": "replay failed", "errors": [x for x in rep if isinstance(x, dict)][:3]}
    R = rec_t[ok]
    P = np.array([rep[i][2] for i in ok])
    W = np.array([raw[i][2] for i in ok])
    sc = np.array([rows[i].get("budget_scale") if rows[i].get("budget_scale") is not None else 1.0 for i in ok], float)
    j = np.array([[float(v) for f in ("thumb", "index", "middle") for v in rows[i]["joints"][f].values()] for i in ok])
    return {"record": p.name, "plan": Path(head["plan"]).name, "budget_nm": tb, "periods_task": len(ok),
            "replay_max_abs_diff_deg": round(float(np.max(np.abs(P - R))), 4),
            "recorded": metrics(R), "replay": metrics(P), "raw": metrics(W),
            "raw_minus_recorded_rms_deg": round(float(np.sqrt(np.mean((W - R) ** 2))), 3),
            "smooth_j": metrics(np.array([sj[i][2] for i in ok])) if sj and all(isinstance(sj[i], tuple) for i in ok)
            else None,
            "smooth_jt": metrics(np.array([sjt[i][2] for i in ok])) if sjt and all(isinstance(sjt[i], tuple) for i in ok)
            else None,
            "budget_below_1_pct": round(100 * float(np.mean(sc < 0.999)), 1),
            "budget_scale_mean": round(float(np.mean(sc)), 3),
            "budget_scale_step_rms": round(float(np.sqrt(np.mean(np.diff(sc) ** 2))), 4),
            "measured_servo": metrics(j)}


def main():
    names = sys.argv[1:]
    files = sorted(RUNS.glob("*_twist_*trial*.jsonl"))
    if names:
        files = [f for f in files if any(f.name.startswith(n) for n in names)]
    from sr2hand.control import hom_server as HS
    h0 = load(files[0])[0]
    HS.configure(h0.get("plant", "measured"), h0.get("hand", "cad_v2"))
    res = []
    for f in files:
        t0 = time.time()
        try:
            r = one(f)
        except Exception as e:                  # noqa: BLE001
            r = {"record": f.name, "note": f"failed: {e!r}"[:200]}
        if r is None:
            continue
        res.append(r)
        if "recorded" in r:
            print(f"{r['record'][:26]:26s} budget {r['budget_nm']}  rev% rec {r['recorded']['rev_pct']:5.1f} "
                  f"replay {r['replay']['rev_pct']:5.1f} raw {r['raw']['rev_pct']:5.1f}  hf rec "
                  f"{r['recorded']['hf_deg']:5.2f} raw {r['raw']['hf_deg']:5.2f}  smooth rev% "
                  f"{(r.get('smooth_j') or {}).get('rev_pct')} / {(r.get('smooth_jt') or {}).get('rev_pct')}  |replay-rec| "
                  f"{r['replay_max_abs_diff_deg']:.3f}  scale<1 {r['budget_below_1_pct']:5.1f}%  "
                  f"({time.time() - t0:.0f} s)", flush=True)
        else:
            print(r, flush=True)
    OUT.write_text(json.dumps({"what": __doc__.split("\n\n")[0].replace("\n", " "),
                               "generated": time.strftime("%F %H:%M"), "records": res}, indent=1) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
