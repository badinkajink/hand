"""Fail a run EARLY if its config diverges from a reference run in a way nobody intended.

This program has now lost four separate experiments to launcher flag drift, always the same
shape: a launcher omits a flag, the trainer supplies a default, the run trains to completion,
and the resulting numbers impersonate a real finding. The documented instances --

  * `--num-envs` / `--total-timesteps` omitted -> 200M-timestep default, two 5-design queues
    dead in a day;
  * `--open-finger-from-keyframe` omitted -> wrong open pose, the object drops (gotcha #5);
  * `--lift-delta-z` omitted -> the whole r5 compact queue trained at 0.05 against r2/r4's
    0.14, and the one design that finished evaluates at align_rate 0.0%;
  * a recipe key that existed on disk in August and is absent from the committed recipe today,
    so a rerun of the "same" recipe silently changes an env field.

-- share a root cause that no amount of comments fixes: the config that actually ran is only
visible AFTER the run, in `config.yaml`, and by then the GPU time is spent. The trainer writes
that file within seconds of launch and long before it starts stepping, so it can be checked
while the run is still cheap to kill.

So: launch, wait for `config.yaml`, diff it against a reference run known to have worked,
abort on any unexplained difference. Differences that ARE intended get listed explicitly with
`--allow`, which makes the intended delta part of the launcher and reviewable in the diff.

  python scripts/assert_config_parity.py --run <new_run_dir> --reference <good_run_dir> \
      --allow env.axial_slip_weight --allow ppo.init_noise_std

Exit 0 = parity (modulo --allow), 1 = divergence (prints the table), 2 = config not found.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

# Keys that legitimately differ between any two runs and say nothing about the experiment.
IGNORE_SUBSTRINGS = ("scene", "dir", "tag", "run_name", "wandb", "seed", "device", "path")


def flatten(d, prefix=""):
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, dict):
            out.update(flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="run dir being launched")
    ap.add_argument("--reference", type=Path, required=True, help="run dir known to have worked")
    ap.add_argument("--allow", action="append", default=[],
                    help="config key permitted to differ (repeatable). State every intended delta.")
    ap.add_argument("--wait", type=float, default=180.0,
                    help="seconds to wait for the run's config.yaml to appear")
    args = ap.parse_args()

    new_cfg = args.run / "config.yaml"
    deadline = time.time() + args.wait
    while not new_cfg.exists() and time.time() < deadline:
        time.sleep(2.0)
    if not new_cfg.exists():
        print(f"[parity] FATAL: {new_cfg} never appeared within {args.wait:.0f}s", file=sys.stderr)
        return 2
    ref_cfg = args.reference / "config.yaml"
    if not ref_cfg.exists():
        print(f"[parity] FATAL: reference {ref_cfg} missing", file=sys.stderr)
        return 2

    # The trainer writes config.yaml in one go, but a truncated read is still possible if we
    # catch it mid-write; a short retry is cheaper than a spurious abort of a good run.
    for _ in range(5):
        try:
            new = flatten(yaml.safe_load(new_cfg.read_text()))
            ref = flatten(yaml.safe_load(ref_cfg.read_text()))
            break
        except yaml.YAMLError:
            time.sleep(1.0)
    else:
        print(f"[parity] FATAL: could not parse {new_cfg}", file=sys.stderr)
        return 2

    allow = set(args.allow)
    diffs, inert = [], []
    for k in sorted(set(new) | set(ref)):
        if k in allow or any(s in k.lower() for s in IGNORE_SUBSTRINGS):
            continue
        a, b = ref.get(k, "<absent>"), new.get(k, "<absent>")
        if a == b:
            continue
        # A field ADDED since the reference ran, sitting at a default that disables it, is not a
        # divergence -- "the feature did not exist" and "the feature is off" are the same run.
        # Without this, every new config key permanently breaks parity against older references
        # and the check trains people to pass --allow reflexively, which defeats it.
        if a == "<absent>" and b in (0, 0.0, False, None, ""):
            inert.append(k)
            continue
        diffs.append((k, a, b))

    if not diffs:
        note = f", {len(inert)} new field(s) at disabling defaults" if inert else ""
        print(f"[parity] OK — {args.run.name} matches {args.reference.name} "
              f"({len(allow)} allowed delta(s){note})")
        return 0

    w = max(len(k) for k, _, _ in diffs)
    print(f"[parity] DIVERGENCE — {args.run.name} vs reference {args.reference.name}:")
    print(f"  {'key':{w}s}  {'reference':>20s}   {'this run':>20s}")
    for k, a, b in diffs:
        print(f"  {k:{w}s}  {str(a):>20s}   {str(b):>20s}")
    print("[parity] If a difference is intended, pass it as --allow <key> in the launcher.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
