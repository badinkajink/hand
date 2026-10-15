#!/usr/bin/env python3
"""Headline numbers of the fingertip co-design page recomputed from the rows (ledger step 24), with code separate from
tip_codesign_analysis.py, and compared with the summaries the page reads. Called by scripts/tip_codesign_verify.py
(numbers); each check is {name, recomputed, summary, ok}.

    .venv/bin/python scripts/tip_codesign_verify.py numbers
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "docs/experiments/20261014-tip_codesign"
SERVO = "kp0.72_kv0_fr0.18_fl0_dp0.0144"
CELLS = [("cylinder", 25.0, 0.125), ("cylinder", 16.0, 0.625), ("cylinder", 10.0, 0.75), ("cylinder", 16.0, -0.375)]
HANDS = [f"D{i}" for i in range(1, 9)]


def rows(p):
    p = Path(p)
    return [json.loads(l) for l in open(p) if l.strip()] if p.exists() else []


def ht(r):
    return float(r["turn_end_deg"]) if r.get("held") else 0.0


def check(out, name, got, ref, tol):
    ok = got is not None and ref is not None and abs(got - ref) <= tol
    out.append({"name": name, "recomputed": got, "summary": ref, "tol": tol, "ok": bool(ok)})


def score(turn_rows, hom_rows, sel_seed, score_seeds):
    """The stage score: per cell the grasp with the best held turn on sel_seed (ties to the higher index), its mean held
    turn on score_seeds; the HOM mean held turn; the mean of the parts over 90 deg."""
    parts = []
    for c in CELLS:
        by = defaultdict(dict)
        for r in turn_rows:
            if r.get("status") == "ok" and (r["shape"], float(r["d_mm"]), r.get("t")) == c:
                by[int(r["cand"])][int(r["seed"])] = ht(r)
        sel = [(v[sel_seed], k) for k, v in by.items() if sel_seed in v]
        if not sel:
            continue
        k = max(sel)[1]
        vals = [by[k][s] for s in score_seeds if s in by[k]]
        if vals:
            parts.append(np.mean(vals) / 90.0)
    hom = [ht(r) for r in hom_rows if r.get("status") == "ok"]
    if hom:
        parts.append(np.mean(hom) / 90.0)
    return float(np.mean(parts)) if parts else None


def deterioration(out):
    S = json.loads((D / "deterioration/summary.json").read_text())["members"]
    law = {r["tip"]: r for r in rows(D / "deterioration/law.jsonl") if r["field"] == "drake"}
    t2 = {(r["model"], r["N"], r["dt_ms"]): r for r in rows(D / "deterioration/T2.jsonl") if r.get("status") == "complete"}
    sp = {(r["model"], r["N"], r["omega"], r["dt_ms"]): r for r in rows(D / "deterioration/spin.jsonl") if r.get("status") == "complete"}
    t5 = {(r["model"], r["dt_ms"]): r for r in rows(D / "deterioration/T5.jsonl") if r.get("status") == "complete"}
    for r in ("0", "2.7", "7.4"):
        n = r.replace(".", "p")
        L = law[f"tpu{r}"]
        m = t2.get((f"mj_tpu{n}_pads1", 1.0, 1.0))
        check(out, f"onset/law pads r {r}", m and m["rbar_onset_mm"] / L["table"]["1|0"]["arm_mm"],
              S[r]["base"]["onset_over_law"], 0.001)
        m = sp.get((f"mj_tpu{n}_pads1", 1.0, 1.0, 1.0))
        check(out, f"spin/law pads 1 rad/s r {r}", m and m["rbar_spin_mm"] / L["spin_arm_mm"]["1"],
              S[r]["base"]["spin_over_law"]["1"], 1e-6)
        m = sp.get((f"drake_tpu{n}", 1.0, 1.0, 1.0))
        check(out, f"spin/law Drake 1 rad/s r {r}", m and m["rbar_spin_mm"] / L["spin_arm_mm"]["1"],
              S[r]["drake"]["spin_over_law"]["1"], 1e-6)
        a, b = t5.get((f"mj_tpu{n}_pads1", 1.0)), t5.get((f"drake_tpu{n}", 0.5))
        check(out, f"brake N45 pads/Drake r {r}", a and b and a["N_at_45_N"] / b["N_at_45_N"], S[r]["base"]["N45_over_drake"], 1e-6)
    for w in ("0.03", "1", "3"):
        m = sp.get(("mj_tpu2p7_pads1", 1.0, float(w), 1.0))
        check(out, f"spin arm pads r 2.7 at {w} rad/s (mm)", m and m["rbar_spin_mm"], S["2.7"]["base"]["spin_mm"][w], 1e-9)


def stage1(out):
    S = json.loads((D / "screen/summary_stage1.json").read_text())["candidates"]
    hom = rows(D / "screen/hom.jsonl")
    got = {}
    for k in S:
        hk = [r for r in hom if r.get("tip") == k and str(r.get("mu")) == "0.7"]
        got[k] = score(rows(D / f"screen/scale/{k}/turn.jsonl"), hk, 0, (1, 2))
        check(out, f"J1 {k}", got[k], S[k]["J1"], 1e-9)
    top = sorted(got, key=lambda k: -(got[k] or -1))[:8]
    out.append({"name": "J1 top 8", "recomputed": top, "summary": sorted(S, key=lambda k: -(S[k]["J1"] or -1))[:8],
                "ok": top == sorted(S, key=lambda k: -(S[k]["J1"] or -1))[:8]})


def stage2(out):
    S = json.loads((D / "screen/summary_stage2.json").read_text())
    C = S["candidates"]
    hom = [r for r in rows(D / "screen/stage2/hom.jsonl") if r.get("status") == "ok"]
    rep = [r for r in rows(D / "screen/stage2/replay.jsonl") if r.get("status") == "complete"]
    conds = (("1.0", SERVO), ("0.7", SERVO), ("0.5", SERVO), ("0.7", "kp0.5_kv0_fr0.18_fl0_dp0.0144"),
             ("0.7", "kp0.72_kv0_fr0.12_fl0_dp0.0144"))
    for k, v in C.items():
        Js = []
        for mu, pl in conds:
            tr = rows(D / (f"screen/scale/{k}/turn.jsonl" if (mu, pl) == ("0.7", SERVO) else
                           f"screen/stage2/scale/{k}_mu{mu}_{pl}/turn.jsonl"))
            hk = [r for r in hom if r.get("tip") == k and str(r.get("mu")) == mu and r.get("plant") == pl]
            Js.append(score(tr, hk, 0, (1, 2)))
        check(out, f"J2 {k}", min(Js) if None not in Js else None, v["J2"], 1e-9)
        mj = {(r["hand"], int(r["seed"])): r for r in rep if r["sim"] == "mujoco" and r["tip"] == k}
        dr = {(r["hand"], int(r["seed"])): r for r in rep if r["sim"] == "drake" and r["tip"] == k}
        n = sum(1 for x in mj if x in dr and bool(mj[x]["held_end"]) == bool(dr[x]["held_end"])
                and abs(mj[x]["turn_end_deg"] - dr[x]["turn_end_deg"]) <= 3.0)
        check(out, f"Drake agreement {k}", n, v["agree_drake_3deg"], 0)


def stage3(out, n_boot=10000, seed=7):
    S = json.loads((D / "final/summary_stage3.json").read_text())
    C, fin = S["candidates"], S["finalists"]
    conds = [("0.7", SERVO), ("0.5", SERVO), ("0.7", "kp0.5_kv0_fr0.18_fl0_dp0.0144"), ("0.7", "kp0.72_kv0_fr0.12_fl0_dp0.0144")]
    hom = [r for r in rows(D / "final/hom.jsonl") if r.get("status") == "ok"]
    arr = {}
    for k in fin:
        for mu, pl in conds:
            sel_rows = rows(D / (f"screen/scale/{k}/turn.jsonl" if (mu, pl) == ("0.7", SERVO) else
                                 f"screen/stage2/scale/{k}_mu{mu}_{pl}/turn.jsonl"))
            new = [r for r in rows(D / f"final/scale/{k}_mu{mu}_{pl}/turn.jsonl") if r.get("status") == "ok"]
            Cm = np.full((4, 3), np.nan)
            for i, c in enumerate(CELLS):
                cand = [(ht(r), int(r["cand"])) for r in sel_rows if r.get("status") == "ok" and int(r["seed"]) == 0
                        and (r["shape"], float(r["d_mm"]), r.get("t")) == c]
                if not cand:
                    continue
                best = max(cand)[1]
                for r in new:
                    if (r["shape"], float(r["d_mm"]), r.get("t")) == c and int(r["cand"]) == best and int(r["seed"]) in (3, 4, 5):
                        Cm[i, int(r["seed"]) - 3] = ht(r)
            Hm = np.full((8, 5), np.nan)
            for r in hom:
                if r.get("tip") == k and str(r.get("mu")) == mu and r.get("plant") == pl and int(r["seed"]) in range(5, 10):
                    Hm[HANDS.index(r["hand"]), int(r["seed"]) - 5] = ht(r)
            arr[(k, mu, pl)] = (Cm, Hm)

    def j3(k, ic=None, ih=None):
        vals = []
        for mu, pl in conds:
            Cm, Hm = arr[(k, mu, pl)]
            if ic is not None:
                Cm = np.take_along_axis(Cm, ic, axis=1)
                Hm = np.take_along_axis(Hm, ih, axis=1)
            parts = [np.nanmean(Cm[i]) / 90 for i in range(4) if np.isfinite(Cm[i]).any()] + [np.nanmean(Hm) / 90]
            vals.append(np.mean(parts))
        return min(vals)
    for k in fin:
        check(out, f"J3 {k}", j3(k), C[k]["J3"], 1e-9)
    rng = np.random.default_rng(seed)
    base = "tcd_rs3_re3_p0.2"
    diffs = defaultdict(list)
    for _ in range(n_boot):
        ic = rng.integers(0, 3, size=(4, 3))
        ih = rng.integers(0, 5, size=(8, 5))
        b = j3(base, ic, ih)
        for k in fin:
            if k != base:
                diffs[k].append(j3(k, ic, ih) - b)
    for k, d in diffs.items():
        lo, hi = np.percentile(d, [5, 95])
        check(out, f"dJ3 90% low {k}", float(lo), C[k]["dJ3_ci90"][0], 0.012)
        check(out, f"dJ3 90% high {k}", float(hi), C[k]["dJ3_ci90"][1], 0.012)
        got = bool(lo > 0) and bool(C[k].get("hold_ok", True))
        out.append({"name": f"beats as-built {k}", "recomputed": got, "summary": C[k]["beats_base"],
                    "ok": got == bool(C[k]["beats_base"])})


def run():
    out = []
    for f in (deterioration, stage1, stage2, stage3):
        try:
            f(out)
        except Exception as e:  # a missing summary is a failed check, not a crash
            out.append({"name": f.__name__, "error": f"{type(e).__name__}: {e}", "ok": False})
    for c in out:
        if not c["ok"]:
            print("FAIL", c)
    print(f"numbers: {sum(c['ok'] for c in out)}/{len(out)} checks pass")
    return {"checks": out, "ok": all(c["ok"] for c in out)}


if __name__ == "__main__":
    run()
