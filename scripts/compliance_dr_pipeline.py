"""Compliance-DR pipeline — train a stiffness-ROBUST m05 policy and prove it. (2026-07-08)

The compliance-robustness sweep showed single-stiffness training overfits the contact model
(soft_b33 reorients at solimp dmax 0.995 and 0.998 but drops at 0.997 / fails at 0.999).
Fix (docs/rl/compliance_dr_plan.md): per-env contact-compliance DR — every reset samples one
softness per env and lerps every geom's solimp (dmin, dmax) jointly across soft..hard
(env_cfg `compliance_dr`, mjlab_terms.randomize_geom_solimp; verified per-world by probe).

Stages (sequential — single 16 GB GPU):
  1. Policy A on m05 from scratch WITH compliance DR (the morph-pipeline A recipe + --compliance-dr).
  2. n imitation-B seeds off that A via the live-A reset, same DR (the variance-study `imit`
     recipe — the base that degraded most gracefully vs stiffness).
  3. Re-run scripts/compliance_robustness_sweep.py — it picks up *policyA_m05_cdr /
     *policyB_m05_cdr_imit_k* automatically. SUCCESS = flat, high held-cos across dmax
     0.995..0.999 vs the fragile single-stiffness curves.

Resumable (per-stage JSON checkpoint), DONE sentinel, logs under logs/compliance_dr/.
Run (detached):
  nohup setsid uv run --extra rl --extra gpu python scripts/compliance_dr_pipeline.py \
      > logs/compliance_dr/pipeline.run.log 2>&1 </dev/null & disown
"""
from __future__ import annotations
import argparse, subprocess, sys, time
from pathlib import Path

from morphohand.studies import runlib
from morphohand.studies.runlib import ROOT, final_ckpt, latest_run, log

M05 = ROOT / "results/phase1/landscape/m05_ik_cem"
IMIT_REF = ROOT / "results/reorient_ref/m05_a10b33_fingertip_obj.npz"
LOGDIR = ROOT / "logs/compliance_dr"


def train_A_cdr(env) -> tuple[Path | None, bool]:
    """m05 A from scratch (morph-pipeline recipe) + compliance DR."""
    tag = "policyA_m05_cdr"
    logf = LOGDIR / "A_m05_cdr.trainer.log"
    e = dict(env)
    e.update(MORPH_RUN=str(M05), WARMSTART="none", LIFT_DELTA_A="0.10",
             EXTRA_ARGS="--open-finger-from-keyframe --lift-phase-start-step 60 --compliance-dr",
             TAG=tag, LOG=str(logf), NUM_ENVS="2048", TOTAL_TS="30000000")
    subprocess.run(["bash", str(ROOT / "scripts/train_A_on_morph.sh")],
                   check=False, capture_output=True, text=True, env=e, timeout=12000)
    run = latest_run(f"*{tag}")
    aborted = Path(str(logf) + ".COLLAPSED").exists()
    return final_ckpt(run), aborted


def train_B_cdr(a_ck: Path, k: int, env) -> tuple[Path | None, bool]:
    """Imitation-B off the DR-trained A, live-A reset, same DR (variance-study imit recipe)."""
    tag = f"policyB_m05_cdr_imit_k{k}"
    logf = LOGDIR / f"B_m05_cdr_imit_k{k}.trainer.log"
    extra = ("--open-finger-from-keyframe"
             f" --imitation-ref-npz {IMIT_REF} --imitation-weight 60 --imitation-alpha 300"
             " --imitation-curriculum-iters 150 --imitation-weight-final 0"
             " --compliance-dr")
    e = dict(env)
    e.update(MORPH="results/phase1/landscape/m05_ik_cem",
             A_CKPT=str(a_ck), B_CKPT=str(a_ck), LIFT_DELTA="0.10",
             EXTRA_ARGS=extra,
             ONSET_STEP="40", BLEND="8", LIFT_TERM_START="58", REORIENT_START="58",
             TAG=tag, LOG=str(logf), NUM_ENVS="3072", TOTAL_TS="20000000")
    subprocess.run(["bash", str(ROOT / "scripts/train_handoff_liveA_reset.sh")],
                   check=False, capture_output=True, text=True, env=e, timeout=10000)
    run = latest_run(f"*{tag}")
    aborted = Path(str(logf) + ".COLLAPSED").exists()
    return final_ckpt(run), aborted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-b-seeds", type=int, default=2)
    args = ap.parse_args()
    LOGDIR.mkdir(parents=True, exist_ok=True)
    env = runlib.base_env()
    sentinel = runlib.Sentinel(LOGDIR / "COMPLIANCE_DR_PIPELINE.DONE")
    sentinel.clear()
    state = runlib.JsonState(LOGDIR / "COMPLIANCE_DR_PIPELINE.json")

    # ---- stage 1: DR-trained A -----------------------------------------
    if not state.get("a_ckpt"):
        log("stage 1: train A on m05 from scratch WITH compliance DR (~2h) ...")
        a_ck, aborted = train_A_cdr(env)
        if a_ck is None:
            log("FATAL: no A checkpoint produced; see logs/compliance_dr/A_m05_cdr.trainer.log")
            sys.exit(1)
        state.update(a_ckpt=str(a_ck), a_aborted=aborted)
        state.save()
        log(f"stage 1 done: {a_ck} (watchdog-aborted={aborted})")
    a_ck = Path(state.get("a_ckpt"))

    # ---- stage 2: n imitation-B seeds, same DR --------------------------
    for k in range(args.n_b_seeds):
        key = f"b_ckpt_k{k}"
        if state.get(key):
            continue
        log(f"stage 2: train imitation-B seed k{k} off DR-A (~40min) ...")
        b_ck, aborted = train_B_cdr(a_ck, k, env)
        state.update(**{key: str(b_ck) if b_ck else None, f"b_aborted_k{k}": aborted})
        state.save()
        log(f"stage 2 k{k} done: {b_ck} (watchdog-aborted={aborted})")

    # ---- stage 3: compliance-robustness sweep (the success test) --------
    log("stage 3: compliance-robustness sweep over the DR policies ...")
    r = subprocess.run([sys.executable, str(ROOT / "scripts/compliance_robustness_sweep.py")],
                       check=False, capture_output=True, text=True, env=env, timeout=14400)
    state.update(sweep_rc=r.returncode)
    state.save()
    log(f"stage 3 done (rc={r.returncode}); curves in docs/experiments/COMPLIANCE_ROBUSTNESS.txt")

    sentinel.write()
    log("PIPELINE COMPLETE")


if __name__ == "__main__":
    main()
