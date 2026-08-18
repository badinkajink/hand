#!/usr/bin/env python3
"""Harden the inline reorienter against the measured sim2real cliffs, then re-measure it.

    nohup setsid uv run --extra rl --extra gpu python scripts/harden_b33_queue.py \
        > logs/harden/queue.run.log 2>&1 </dev/null & disown

WHAT IS BEING FIXED. `docs/experiments/SIM2REAL_ROBUSTNESS.txt` walked the shipped a10->b33
handoff along one physical axis at a time and found three cliffs, all of them the same statement
-- anything that reduces purchase at the contact kills it, one-sidedly:

    pad friction x0.5 (mu 2.4 -> 1.2)   hold 1.00 -> 0.09    the sharpest
    contact stiffness dmax 0.995->0.997 reorient 0.91 -> 0.09
    shaft diameter x0.85               hold 1.00 -> 0.03

The first two get domain randomisation here. THE THIRD DOES NOT, deliberately: randomising the
shaft diameter means writing geom_size per world, which moves collision bounds rather than a
material constant, and it is the least urgent of the three for a sim2real attempt with one known
tool. It is left as a characterised limitation rather than a rushed event term.

WHY FINETUNE b33 AND NOT RETRAIN. Compliance DR has been tried on this hand once and it failed:
`compliance_dr_pipeline.py` trained a fresh DR Policy A and fresh imitation-Bs, and every
resulting pair holds but does not reorient (reorient rate 0.00 at the trained stiffness --
COMPLIANCE_RATE.txt). That pipeline changed the grip AND discarded the reorient skill at the same
time. The untried variant is to keep the skill: warmstart b33's actor AND critic (actor-only
warmstart wrecks finetunes, gotcha #8) and turn the DR on underneath it, so the policy adapts a
working reorient instead of rediscovering one.

THREE ARMS, so a result can be attributed. Compliance-only reproduces the earlier attempt's axis
under the new finetune protocol; friction-only tests the sharpest cliff on its own; both tests
whether they compose or fight. The eval at the end walks BOTH axes for all four policies
(baseline b33 included) -- success is a FLAT high curve, not a high point, and the baseline row
is what makes flatness legible.

FLAG PARITY is asserted, not hoped for: each run's config.yaml is diffed against b33's own
config the moment it is written, and the queue aborts that arm on any unexplained difference.
The recipe pins term_tip_lost_steps=3 while b33 actually trained at 10, which is exactly the
kind of silent drift that has killed four experiments here.

Resumable: finished arms are skipped on relaunch. Writes docs/experiments/HARDEN_B33.{json,txt}.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from morphohand.studies import runlib
from morphohand.studies.runlib import ROOT, RESULTS_RL as RL, final_ckpt, latest_run, log

sys.path.insert(0, str(ROOT / "scripts"))
from sim2real_robustness_sweep import rate_point, scene_dir  # noqa: E402

MORPH = ROOT / "results/phase1/landscape/m05_ik_cem"
A_CKPT = RL / "a10_20260702-1256-policyA_m05_ik10/tensorboard/model_609.pt"
B_CKPT = RL / "b33_20260702-1353-policyB_m05_reorient_ik10/tensorboard/model_270.pt"
REFERENCE = RL / "b33_20260702-1353-policyB_m05_reorient_ik10"
LOGDIR = ROOT / "logs/harden"
JSON_P = ROOT / "docs/experiments/HARDEN_B33.json"
TXT_P = ROOT / "docs/experiments/HARDEN_B33.txt"

TOTAL_TS = int(os.environ.get("TOTAL_TS", "40000000"))
NUM_ENVS = os.environ.get("NUM_ENVS", "3072")

# (arm tag, extra trainer flags). The DR fields are the intended divergence from b33 and are
# listed as --allow below, so parity still catches everything else.
ARMS = [
    ("b33dr_compliance", "--compliance-dr"),
    ("b33dr_friction", "--friction-dr"),
    ("b33dr_both", "--compliance-dr --friction-dr"),
]
# Allowed divergences from b33's config, each one checked rather than waved through (the dry run
# against a smoke arm is what surfaced this list):
#   the DR fields          - the point of the experiment;
#   total_timesteps        - a 40M finetune vs b33's 20M from-warmstart run;
#   init_actor_checkpoint  - b33 warmstarted off a10, these warmstart off b33;
#   min_tips_in_contact    - field added after b33; its default 3 IS b33's behaviour;
#   imitation_alpha        - inert while imitation_weight is 0, which it is;
#   thumb_brace_*          - inert while thumb_brace_weight is 0, which it is;
#   viewer_*               - render size, cosmetic.
# Anything NOT on this list kills the arm in the first minute rather than at dawn.
ALLOW = ["env.compliance_dr", "env.compliance_dr_dmin", "env.compliance_dr_dmax",
         "env.friction_dr", "env.friction_dr_scale", "ppo.total_timesteps",
         "ppo.init_actor_checkpoint", "env.min_tips_in_contact", "env.imitation_alpha",
         "env.thumb_brace_align_thresh", "env.thumb_brace_max_force",
         "env.viewer_azimuth", "env.viewer_elevation", "env.viewer_height", "env.viewer_width"]


def gpu_idle(threshold_mb: int = 2000, timeout_s: int = 600) -> None:
    """Wait for GPU memory to fall back before the next arm (CLAUDE.md: ~1 GB after a Warp run)."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                                  "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=30).stdout.strip()
            if int(out.splitlines()[0]) < threshold_mb:
                return
        except Exception:
            return
        time.sleep(15)


def train_arm(tag: str, extra: str) -> tuple[Path | None, str]:
    """One DR finetune of b33 off the live-A reset. Returns (final ckpt, status)."""
    logf = LOGDIR / f"{tag}.trainer.log"
    env = dict(os.environ)
    env.update(
        MORPH="results/phase1/landscape/m05_ik_cem",
        A_CKPT=str(A_CKPT), B_CKPT=str(B_CKPT),
        LIFT_DELTA="0.1", ONSET_STEP="40", BLEND="0",
        # b33's own config, NOT the launcher/recipe defaults: it trained with terminations and
        # the reorient reward engaging at step 58 and a 10-step tip-loss grace (the recipe says
        # 3). Left implicit, these three silently define a different experiment.
        LIFT_TERM_START="58", REORIENT_START="58", TIP_LOST_STEPS="10",
        NUM_ENVS=NUM_ENVS, TOTAL_TS=str(TOTAL_TS),
        RESID_SCALE="0.5", EASING="ease_out_quad", CONTACT_GATE="1",
        EXTRA_ARGS=f"--open-finger-from-keyframe {extra}",
        TAG=tag, LOG=str(logf),
    )
    proc = subprocess.Popen(["bash", str(ROOT / "scripts/train_handoff_liveA_reset.sh")],
                            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Parity gate: check the config the moment the trainer writes it, while the run is cheap.
    time.sleep(20)
    run = latest_run(f"*{tag}")
    if run is not None:
        chk = subprocess.run(
            [sys.executable, str(ROOT / "scripts/assert_config_parity.py"),
             "--run", str(run), "--reference", str(REFERENCE)]
            + [f"--allow={k}" for k in ALLOW],
            capture_output=True, text=True)
        if chk.returncode == 1:
            proc.kill()
            log(f"[{tag}] PARITY FAIL — killed:\n{chk.stdout}")
            return None, "PARITY_FAIL"
        log(f"[{tag}] parity ok")

    proc.wait()
    run = latest_run(f"*{tag}")
    if Path(str(logf) + ".COLLAPSED").exists():
        return final_ckpt(run), "COLLAPSED"
    return final_ckpt(run), "ok"


# ---------------------------------------------------------------- the robustness re-measurement
# Both cliff axes, for every policy. A hardened policy is one whose curve is FLAT and high; a
# single good point is what the fragile baseline already has.
EVAL_POINTS = [
    ("trained", None),
    ("dmax0.997", lambda s: s.set_solimp(0.98, 0.997)),
    ("dmax0.998", lambda s: s.set_solimp(0.983, 0.998)),
    ("dmax0.999", lambda s: s.set_solimp(0.985, 0.999)),
    ("mu_x0.5", lambda s: s.scale_friction(0.5)),
    ("mu_x0.7", lambda s: s.scale_friction(0.7)),
]


def main() -> None:
    LOGDIR.mkdir(parents=True, exist_ok=True)
    state = runlib.JsonState(LOGDIR / "HARDEN_B33_STATE.json")
    store = runlib.RecordStore(JSON_P, key_field="key")
    report = runlib.TxtReport(
        TXT_P, f"# b33 DR-hardening {time.strftime('%Y-%m-%d %H:%M')}  "
               f"finetune of b33 (actor+critic) under contact DR, {TOTAL_TS/1e6:.0f}M ts/arm\n"
               f"# eval: continuous a10->B handoff, n=32 x 300 steps. SUCCESS = FLAT high curve, "
               f"not a high point.\n")

    policies = {"b33_baseline": B_CKPT}
    for tag, extra in ARMS:
        key = f"train:{tag}"
        if state.get(key):
            ck = Path(state.get(key))
            if ck.exists():
                policies[tag] = ck
                log(f"[skip] {tag} already trained")
                continue
        gpu_idle()
        log(f"training {tag} ({extra}) ...")
        t0 = time.time()
        ck, status = train_arm(tag, extra)
        log(f"[{tag}] {status} in {(time.time() - t0) / 60:.0f} min -> {ck}")
        if ck is not None and status != "PARITY_FAIL":
            policies[tag] = ck
            state.update(**{key: str(ck)})
            state.save()
        else:
            report.line(f"# {tag} produced no checkpoint ({status})")

    work = ROOT / "logs" / "_harden_tmp"
    work.mkdir(parents=True, exist_ok=True)
    for name, b_ck in policies.items():
        for label, mutate in EVAL_POINTS:
            key = f"{name}@{label}"
            if key in store:
                continue
            gpu_idle()
            try:
                scene = scene_dir(f"harden_{label}", mutate)
                m = rate_point(scene, work, {}, b_ckpt=b_ck)
            except Exception as ex:
                m = {"error": f"{type(ex).__name__}: {str(ex)[:160]}"}
            rec = {"key": key, "policy": name, "level": label, **m}
            store.put(rec)
            report.line(f"{key:34} hold {str(rec.get('hold_rate')):>6} "
                        f"reorient {str(rec.get('reorient_rate')):>6} "
                        f"cos|held {str(rec.get('cos_p50_held')):>7} {rec.get('error', '')}")
            print(f"[{key}] {rec}")
    runlib.Sentinel(LOGDIR / "HARDEN_B33.DONE").write()
    log(f"COMPLETE -> {TXT_P}")


if __name__ == "__main__":
    main()
