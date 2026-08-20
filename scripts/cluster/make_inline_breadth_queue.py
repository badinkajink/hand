#!/usr/bin/env python3
"""Emit packed-queue manifests for the inline-hand questions that breadth can answer.

Three studies, each blocked until now for the same reason: they need many training
draws, and one GPU could only ever produce one. Print a TSV to stdout for
`deltaai_queue.py new`.

  scripts/cluster/make_inline_breadth_queue.py b33_seeds       | ... new b33_seeds
  scripts/cluster/make_inline_breadth_queue.py friction_cliff  | ... new friction_cliff
  scripts/cluster/make_inline_breadth_queue.py design_band     | ... new design_band

FLAG PARITY WARNING. Every command below carries b33's OWN config, not the recipe
defaults: lift-phase/reorient start at step 58 (recipe says 40/45), tip-loss grace
10 (recipe says 3), and --open-finger-from-keyframe. harden_b33_queue.py documents
why — left implicit, those three silently define a different experiment, and at
n=16 that is 13 GPU-hours spent measuring the wrong thing. Do not "simplify" them
out. Reference for `assert_config_parity.py` is the b33 run itself.
"""
from __future__ import annotations

import argparse
import sys

MORPH_M05 = "results/phase1/landscape/m05_ik_cem"
A_CKPT = "results/rl/a10_20260702-1256-policyA_m05_ik10/tensorboard/model_609.pt"
B10 = "results/rl/b10_20260604-1642-policyB_holdonlyws_repro/tensorboard/model_541.pt"
B33 = "results/rl/b33_20260702-1353-policyB_m05_reorient_ik10/tensorboard/model_270.pt"

STEPS = 20_000_000
RATE = 6950            # measured GH200 steps/s
EST_H = round(STEPS / RATE / 3600 + 0.07, 2)   # + startup


def train(tag: str, seed: int, morph: str = MORPH_M05, *,
          warmstart: str | None = B10, scene: str | None = None,
          steps: int = STEPS) -> str:
    """One live-A reorient run, in b33's configuration."""
    a = [
        "python scripts/rl_train_cube.py",
        "--recipe b_liveA",
        f"--morphology-run {morph}",
        f"--num-envs 3072 --total-timesteps {steps}",
        f"--live-a-checkpoint {A_CKPT} --live-a-onset 40 --live-a-blend-steps 0",
        "--lift-target-z-above-init 0.1 --lift-delta-z 0.1",
        "--finger-residual-scale 0.5 --finger-close-easing ease_out_quad",
        # b33's own timing, NOT the recipe's — see the module docstring.
        "--lift-phase-start-step 58 --reorient-start-step 58 --term-tip-lost-steps 10",
        "--target-axis-weight 100.0 --target-axis-progress-weight 300.0",
        "--open-finger-from-keyframe",
        f"--seed {seed} --tag {tag}",
        "--no-wandb --no-record-videos",
        f"--watchdog-collapse-z 0.030 --watchdog-from-iter 50 "
        f"--watchdog-sentinel logs/{tag}.COLLAPSED",
    ]
    if warmstart:
        a.append(f"--init-actor-checkpoint {warmstart}")
    if scene:
        a.append(f"--frozen-scene-xml {scene}")
    return " ".join(a)


def emit(rows: list[tuple[float, str, str]]) -> None:
    for est, label, cmd in rows:
        print(f"{est}\t{label}\t{cmd}")
    total = sum(r[0] for r in rows)
    print(f"# {len(rows)} tasks, {total:.1f} GPU-hours", file=sys.stderr)


def study_b33_seeds(n: int) -> list[tuple[float, str, str]]:
    """How good is b33, really?

    Every inline sim2real conclusion — the robustness cliffs, the fingertip
    ranking, the hardware spec — rests on ONE policy draw evaluated at n=32
    rollouts. The n=32 rollout spread is known (~±0.1). The TRAINING-draw spread
    is not, and elsewhere in this program it is 0.3-0.5, which is larger than
    every difference those studies rank. So: rerun b33's own lineage (b10
    warmstart, 20M ts, m05) across n seeds and get the distribution b33 was drawn
    from. If b33 sits at the top of a wide band, the hardware spec was written
    off a lucky policy and should be re-derived from a median one.
    """
    return [(EST_H, f"b33seed_s{s}", train(f"b33seed_s{s}", s)) for s in range(1, n + 1)]


def study_friction_cliff(n_scratch: int, n_warm: int, n_ctrl: int) -> list[tuple[float, str, str]]:
    """Is the μ×0.5 cliff a strategy limit or a wall?

    Stated as explicitly unresolved: a static probe finds ~5 N of axial capacity
    against a 0.24 N tool at μ×0.5, so HOLDING is not the binding constraint —
    the policy simply never found a low-friction strategy. The one attempt was a
    DR finetune off b33, which starts inside the high-friction basin: the wrong
    initialisation for a hard-exploration problem, and n=1 besides.

    Three arms, and the controls are the point. FROM SCRATCH at μ×0.5 asks
    whether any seed finds the strategy at all. WARMSTART at μ×0.5 asks whether
    b33's prior helps or traps. The μ×1.0 scratch control says whether a failure
    is about friction or just about from-scratch reorient training being hard —
    without it a null result is uninterpretable.
    """
    mu05 = "logs/scenes/m05_mu05.xml"
    mk = (f"python scripts/mutate_scene.py --scene {MORPH_M05}/frozen_scene.xml "
          f"--out {mu05} --friction 0.5")
    rows = []
    for s in range(1, n_scratch + 1):
        rows.append((EST_H, f"mu05_scratch_s{s}",
                     f"{mk} && " + train(f"mu05_scratch_s{s}", s, warmstart=None, scene=mu05)))
    for s in range(1, n_warm + 1):
        rows.append((EST_H, f"mu05_warm_s{s}",
                     f"{mk} && " + train(f"mu05_warm_s{s}", s, warmstart=B33, scene=mu05)))
    for s in range(1, n_ctrl + 1):
        rows.append((EST_H, f"mu10_scratch_s{s}",
                     train(f"mu10_scratch_s{s}", s, warmstart=None)))
    return rows


def study_design_band(designs: list[str], seeds: int) -> list[tuple[float, str, str]]:
    """Does the HAND's geometry widen the contact band that fingertips and DR could not?

    Two independent negatives now say the rolling reorient is a narrow-band
    contact behaviour: fingertip shape leaves the hold-per-turn ratio flat across
    all eight shapes, and contact DR relocates the policy instead of broadening
    it. Both moved the CONTACT. Nobody has moved the HAND — and the morphology
    sweep scored designs on nominal reorient only, never on how much friction or
    stiffness they tolerate.

    If band width turns out to be design-dependent, robustness becomes a
    co-design objective and the sim2real spec loosens. If it is flat across
    designs too, that is the third independent negative and the hardware-side
    strategy is settled rather than merely recommended.
    """
    rows = []
    for d in designs:
        name = d.rstrip("/").split("/")[-1]
        for s in range(1, seeds + 1):
            tag = f"band_{name}_s{s}"
            rows.append((EST_H, tag, train(tag, s, morph=d)))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="study", required=True)

    p = sub.add_parser("b33_seeds"); p.add_argument("-n", type=int, default=16)
    p = sub.add_parser("friction_cliff")
    p.add_argument("--scratch", type=int, default=8)
    p.add_argument("--warm", type=int, default=8)
    p.add_argument("--control", type=int, default=4)
    p = sub.add_parser("design_band")
    p.add_argument("--designs", nargs="+", default=[
        MORPH_M05,
        "results/phase1/morph_sweep/H06_04_r1_cem",
        "results/phase1/landscape/20260819-m05_shortprox",
    ])
    p.add_argument("--seeds", type=int, default=4)

    a = ap.parse_args()
    if a.study == "b33_seeds":
        emit(study_b33_seeds(a.n))
    elif a.study == "friction_cliff":
        emit(study_friction_cliff(a.scratch, a.warm, a.control))
    else:
        emit(study_design_band(a.designs, a.seeds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
