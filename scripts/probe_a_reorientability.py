"""Zero-training-cost test of a cheap A-selector (2026-07-19).

The probe suite's closing finding: the dominant evaluator-noise term is the Policy-A DRAW
(which delivered grip you get), and the A trajectory-health GATE cannot select the good-for-
reorient draws — it is gate-invisible and even ANTI-ORDERS them (health-FAIL As produced
G02_00's best B draws 0.635/0.681). reorientation.md §P2 avar Finding 2 states it outright:
"Best-of-N by the gate buys collapse insurance only; it cannot select for downstream
reorientability."

HYPOTHESIS: a delivered grip is reorientable iff a COMPETENT reorienter can actually roll it.
So run the proven reorienter b33 ZERO-SHOT (no training) as Policy B on each design's own A
delivery, through the exact continuous-handoff eval, and read its held-cos. If that probe score
rank-correlates with the design's ACTUAL trained-imit-B held-cos, we have a ~1-min/A selector
that sees the variance the health gate can't — turning "average many full A->B draws" (days)
into "draw a few A's, probe each, keep the reorientable one, train ONE imit-B".

This uses ONLY on-disk assets (P4 global12x2 + confirm kept-A checkpoints, their CEM dirs, and
their known trained-imit-B held_cos_tail) => zero GPU TRAINING, just deterministic rollouts.

Within-design replicates are the clean test (same geometry, vary the A draw):
  G02_00 {0.504,0.635,0.107,0.681}  G02_05 {-0.499,0.887,-0.079,0.532}  + 7 designs x n=2
  cf_m05 x3  cf_l13 x3.  b33 is m05-biased => trust WITHIN-design ranking; pooled corr is 2nd.

Run (detached):
  nohup setsid env MUJOCO_GL=egl uv run --extra rl --extra gpu \
    python scripts/probe_a_reorientability.py > logs/probe_reorientability.run.log 2>&1 &
Resumable: re-run skips A's whose probe row is already in the JSON.
"""
from __future__ import annotations
import json, re, subprocess, sys, time
from pathlib import Path

from morphohand.studies import runlib
from morphohand.studies.runlib import ROOT

B33 = ROOT / "results/rl/b33_20260702-1353-policyB_m05_reorient_ik10/tensorboard/model_270.pt"
SRC_JSONS = ["docs/experiments/MORPH_PIPELINE_global12x2.json",
             "docs/experiments/MORPH_PIPELINE_confirm.json"]
OUT_JSON = ROOT / "docs/experiments/PROBE_A_REORIENTABILITY.json"
PROBE_VID = ROOT / "results/probe_reorient"          # scratch mp4s + health jsons (gitignored)


def load_pairs() -> list[dict]:
    pairs = []
    for jp in SRC_JSONS:
        p = ROOT / jp
        if not p.exists():
            continue
        for r in json.loads(p.read_text()):
            A = r.get("A") or {}
            run, ck = A.get("run"), A.get("best_ckpt")
            cos = (((r.get("handoff") or {}).get("health") or {}).get("metrics", {})
                   .get("held_cos_tail"))
            if not run or not ck or cos is None:
                continue
            a_ckpt = ROOT / f"results/rl/{run}/tensorboard/{ck}"
            cem = ROOT / f"results/phase1/morph_sweep/{r['id']}_cem"
            if not a_ckpt.exists() or not cem.is_dir():
                continue
            # design = the geometry, collapsing replica (_rN) / confirm-seed (_sN) suffixes so
            # same-geometry draws group together (G02_00_r0..r3 -> G02_00; cf_m05_s0..2 -> cf_m05).
            design = re.sub(r"_[rs]\d+$", "", r["id"])
            pairs.append({"id": r["id"], "design": design,
                          "trained_cos": float(cos), "a_ckpt": str(a_ckpt),
                          "cem_dir": str(cem), "a_run": run})
    return pairs


def probe_one(pair: dict, env, cache_env) -> dict:
    PROBE_VID.mkdir(parents=True, exist_ok=True)
    out = PROBE_VID / f"probe_{pair['id']}.mp4"
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts/rl_demo_handoff_continuous.py"),
         "--policy-a", pair["a_ckpt"], "--policy-b", str(B33),
         "--morphology-run", pair["cem_dir"], "--lift-delta", "0.10",
         "--open-finger-from-keyframe", "--output", str(out)],
        check=False, capture_output=True, text=True, env=cache_env, timeout=1800)
    hj = out.with_suffix(".health.json")
    probe_cos, min_z_post, verdict, jerk = None, None, None, None
    if hj.exists():
        try:
            h = json.loads(hj.read_text())
            m = h.get("metrics", {})
            probe_cos = m.get("held_cos_tail")
            min_z_post = m.get("min_z_hold")
            jerk = m.get("ang_jerk")
            verdict = h.get("verdict")
        except Exception:
            pass
    # honest post-handoff min-z from stdout (more faithful than scorecard min_z_hold)
    for ln in (r.stdout or "").splitlines():
        if "honest hold metric" in ln:
            for t in ln.split(":")[-1].split():
                try:
                    min_z_post = float(t); break
                except ValueError:
                    continue
    return {"probe_cos": probe_cos, "probe_min_z_post": min_z_post,
            "probe_verdict": verdict, "probe_jerk": jerk, "rc": r.returncode}


def analyze(rows: list[dict]) -> str:
    import numpy as np
    from collections import defaultdict
    have = [r for r in rows if r.get("probe_cos") is not None]
    lines = [f"\n===== PROBE ANALYSIS ({len(have)}/{len(rows)} evaluable) =====\n"]

    def spearman(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
            return float("nan")
        rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
        return float(np.corrcoef(rx, ry)[0, 1])

    # (1) WITHIN-design: does the probe pick the best A draw? (the clean, unconfounded test)
    by = defaultdict(list)
    for r in have:
        by[r["design"]].append(r)
    lines.append("-- WITHIN-DESIGN (same geometry, vary A draw) --")
    rhos, best_hits, best_n = [], 0, 0
    for d, rs in sorted(by.items()):
        if len(rs) < 2:
            continue
        pc = [r["probe_cos"] for r in rs]; tc = [r["trained_cos"] for r in rs]
        rho = spearman(pc, tc)
        # did the probe's argmax match the trained argmax?
        hit = int(np.argmax(pc) == np.argmax(tc))
        best_hits += hit; best_n += 1
        if not (rho != rho):  # not nan
            rhos.append(rho)
        pairs_str = " ".join(f"{r['id'].split('_r')[-1]}:p{r['probe_cos']:+.2f}/t{r['trained_cos']:+.2f}"
                             for r in sorted(rs, key=lambda z: z["id"]))
        lines.append(f"  {d:8} n={len(rs)} rho={rho:+.2f} best-A-hit={'Y' if hit else 'N'} | {pairs_str}")
    if rhos:
        lines.append(f"  >> mean within-design Spearman = {np.mean(rhos):+.3f} (n_designs={len(rhos)})")
    if best_n:
        lines.append(f"  >> probe picked the best-A in {best_hits}/{best_n} designs "
                     f"(chance ~ sum 1/n_i)")

    # (2) POOLED (confounded by b33's m05-bias across geometries; secondary)
    pc = [r["probe_cos"] for r in have]; tc = [r["trained_cos"] for r in have]
    lines.append(f"\n-- POOLED (n={len(have)}, b33-bias confounded) --")
    lines.append(f"  Spearman(probe_cos, trained_cos) = {spearman(pc, tc):+.3f}")
    lines.append(f"  Pearson  = {float(np.corrcoef(pc, tc)[0,1]):+.3f}")

    # (3) does the probe at least separate reorienters (trained>=0.5) from static (<0.2)?
    hi = [r["probe_cos"] for r in have if r["trained_cos"] >= 0.5]
    lo = [r["probe_cos"] for r in have if r["trained_cos"] < 0.2]
    if hi and lo:
        lines.append(f"\n-- SEPARATION: trained>=0.5 (n={len(hi)}) probe {np.mean(hi):+.2f}"
                     f" vs trained<0.2 (n={len(lo)}) probe {np.mean(lo):+.2f}  "
                     f"(gap {np.mean(hi)-np.mean(lo):+.2f})")
    return "\n".join(lines)


def main():
    env = runlib.base_env()
    cache_env = runlib.warp_cache_env(env)     # one cache for the whole SEQUENTIAL run (no race)
    pairs = load_pairs()
    print(f"[probe] {len(pairs)} (A-ckpt + cem + trained-cos) pairs on disk", flush=True)
    store = runlib.RecordStore(OUT_JSON, key_field="id")
    for i, pair in enumerate(pairs):
        prev = store.get(pair["id"])
        if prev is not None and prev.get("probe_cos") is not None:
            print(f"[skip] {pair['id']} (probe_cos {prev['probe_cos']})", flush=True); continue
        t0 = time.time()
        print(f"[{i+1}/{len(pairs)}] probe {pair['id']} (trained_cos {pair['trained_cos']:+.3f}) ...",
              flush=True)
        res = probe_one(pair, env, cache_env)
        rec = {**pair, **res, "secs": round(time.time() - t0)}
        store.put(rec)
        print(f"    -> probe_cos {res['probe_cos']} min_z {res['probe_min_z_post']} "
              f"verdict {res['probe_verdict']} ({rec['secs']}s)", flush=True)
    rows = [r for r in store.values() if r]
    report = analyze(rows)
    print(report)
    (ROOT / "docs/experiments/PROBE_A_REORIENTABILITY.txt").write_text(
        f"# b33 zero-shot A-reorientability probe  {time.strftime('%Y-%m-%d %H:%M')}\n" + report + "\n")
    runlib.Sentinel(ROOT / "logs/PROBE_A_REORIENTABILITY.DONE").write(f"{len(rows)} rows")
    print("[probe] DONE", flush=True)


if __name__ == "__main__":
    main()
