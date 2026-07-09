"""Compliance-robustness sweep — how sensitive are TRAINED policies to contact stiffness? (2026-07-08)

The sim2real retrains showed single runs are seed-noisy. This measures robustness cleanly with NO
training: take fixed trained policies and evaluate each across a RANGE of contact compliances
(geom solimp), recording held-cos / min-z (drop) / fingertip force at each. Produces a
compliance-response curve per policy — the honest "how much contact hardening does this policy
tolerate" measure. Eval-only, deterministic, ~1-2 min per (policy, compliance) point.

Run: MUJOCO_GL=egl uv run --extra rl --extra gpu python scripts/compliance_robustness_sweep.py
"""
from __future__ import annotations
import re, shutil, subprocess, sys, time

from morphohand.studies import runlib
from morphohand.studies.runlib import ROOT, RESULTS_RL as RL, final_ckpt, latest_run

SOFT_REF = ROOT / "results/phase1/landscape/m05_ik_cem"     # m05 CEM (best_rollout for the obs ref)
OUT = ROOT / "results/phase1/sim2real/compliance"
JSON = ROOT / "docs/experiments/COMPLIANCE_ROBUSTNESS.json"
TXT = ROOT / "docs/experiments/COMPLIANCE_ROBUSTNESS.txt"


# fixed trained policies to probe (all m05-design). (name, A_ckpt, B_ckpt)
def policies():
    a10 = RL / "a10_20260702-1256-policyA_m05_ik10/tensorboard/model_609.pt"
    b33 = RL / "b33_20260702-1353-policyB_m05_reorient_ik10/tensorboard/model_270.pt"
    a_hard = final_ckpt(latest_run("*policyA_m05_hardcontact"))
    b_hard = final_ckpt(latest_run("*policyB_m05_hard_imit_k1"))
    out = [("soft_b33", a10, b33), ("soft_imitB", a10, final_ckpt(latest_run("*vstudy_m05_imit_k0")))]
    if a_hard and b_hard:
        out.append(("hard_retrained", a_hard, b_hard))
    # compliance-DR retrains (scripts/compliance_dr_pipeline.py) — the success test is a
    # FLAT high held-cos curve here, vs the fragile single-stiffness policies above.
    a_cdr = final_ckpt(latest_run("*policyA_m05_cdr"))
    if a_cdr:
        for k in (0, 1):
            b_cdr = final_ckpt(latest_run(f"*policyB_m05_cdr_imit_k{k}"))
            if b_cdr:
                out.append((f"cdr_imitB_k{k}", a_cdr, b_cdr))
    return [(n, a, b) for n, a, b in out if a and b and a.exists() and b.exists()]


# compliance levels: geom solimp (dmin, dmax). dmax↑ = harder (less penetration). soft is the m05 default.
COMPLIANCES = [
    (0.97, 0.995, "soft"),      # the training default (softest)
    (0.98, 0.997, "c997"),
    (0.983, 0.998, "c998"),
    (0.985, 0.999, "c999"),     # the "mild-hard" step
    (0.99, 0.9995, "c9995"),    # harder
]


def make_scene_dir(dmin, dmax, label):
    d = OUT / f"m05_{label}"
    d.mkdir(parents=True, exist_ok=True)
    for f in ("best_rollout.npz", "summary.json"):
        shutil.copy(SOFT_REF / f, d / f)
    src = (SOFT_REF / "frozen_scene.xml").read_text()
    src = re.sub(r'solimp="[^"]*"', f'solimp="{dmin} {dmax} 0.0004"', src)
    (d / "frozen_scene.xml").write_text(src)
    return d


def evaluate(name, a_ck, b_ck, scene_dir, env):
    e = runlib.warp_cache_env(env)
    out_mp4 = OUT / f"{name}_{scene_dir.name}.mp4"
    r = subprocess.run([sys.executable, str(ROOT / "scripts/rl_demo_handoff_continuous.py"),
                        "--policy-a", str(a_ck), "--policy-b", str(b_ck),
                        "--morphology-run", str(scene_dir), "--lift-delta", "0.10",
                        "--open-finger-from-keyframe", "--output", str(out_mp4)],
                       check=False, capture_output=True, text=True, env=e, timeout=900)
    s = r.stdout or ""
    def grab(pat, i=1):
        m = re.search(pat, s)
        return float(m.group(i)) if m else None
    return dict(
        held_cos=grab(r"held-vertical cos POST-HANDOFF \(last 50 steps mean\): ([-\d.]+)"),
        peak_cos=grab(r"\(peak ([-\d.]+)\)"),
        min_z=grab(r"POST-HANDOFF \(honest hold metric\): ([-\d.]+) m"),
        force=grab(r"fingertip mean\s+([-\d.]+)N"),
        verdict=(re.search(r"VERDICT: ([A-Z]+)", s) or [None, None])[1] if "policy health" in s else None,
    )


def main():
    env = runlib.base_env()
    OUT.mkdir(parents=True, exist_ok=True)
    store = runlib.RecordStore(JSON, key_field="key")
    pols = policies()
    print("policies:", [p[0] for p in pols])
    report = runlib.TxtReport(
        TXT, f"# compliance-robustness sweep {time.strftime('%Y-%m-%d %H:%M')}\n"
             f"# fixed trained policies x solimp range; eval-only. held-cos / min-z / force.\n")
    for dmin, dmax, label in COMPLIANCES:
        scene = make_scene_dir(dmin, dmax, label)
        for name, a_ck, b_ck in pols:
            key = f"{name}@{label}"
            if key in store:
                print(f"[skip] {key}"); continue
            t0 = time.time()
            try:
                m = evaluate(name, a_ck, b_ck, scene, env)
            except Exception as ex:
                m = {"error": f"{type(ex).__name__}: {str(ex)[:120]}"}
            rec = {"key": key, "policy": name, "label": label, "dmax": dmax, **m, "secs": round(time.time() - t0)}
            store.put(rec)
            report.line(f"{key:26} dmax {dmax:<7} cos {str(rec.get('held_cos')):>7} peak {str(rec.get('peak_cos')):>7} "
                        f"min_z {str(rec.get('min_z')):>7} force {str(rec.get('force')):>6} {rec.get('verdict') or rec.get('error','')}")
    runlib.Sentinel(ROOT / "logs/COMPLIANCE_ROBUSTNESS.DONE").write()
    print(f"[compliance-sweep] COMPLETE -> {TXT}")


if __name__ == "__main__":
    main()
