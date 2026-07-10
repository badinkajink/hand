"""Trainer-side object-height collapse watchdog (CODEBASE_AUDIT.md step 5).

Replaces the bash grep-the-log loops in train_A_on_morph.sh /
train_handoff_liveA_reset.sh (gotcha #10): a policy that stops lifting mid-run
("collapse") used to burn hours of GPU before anyone noticed. Living in the
trainer, the guard now covers EVERY launcher, with no 30 s polling and no
process-group kill.

Mechanism: wraps the rsl_rl `Logger.log` method. Each PPO iteration, before the
logger aggregates-and-clears `ep_extras`, the wrapper computes the same mean the
console line prints for the watched metric (default
`Metrics/lift_height/object_height` — object height above init, episode mean).
Once `it >= guard_from_iter`, a value below `collapse_z` writes the sentinel
file and raises `TrainingCollapseError`, which the trainer turns into a clean
abort (checkpoints saved at earlier save-intervals remain for salvage —
runlib.best_a_ckpt picks the best pre-collapse one).

The sentinel path keeps the bash-era contract: launchers pass
`--watchdog-sentinel "${LOG}.COLLAPSED"`, so pipelines that check
`<trainer log>.COLLAPSED` keep working unchanged.
"""
from __future__ import annotations

from pathlib import Path
from time import strftime

import torch


class TrainingCollapseError(RuntimeError):
    """Raised by the watchdog when the watched metric collapses."""


def mean_ep_metric(ep_extras: list[dict], key: str) -> float | None:
    """Mean of `key` across the logger's buffered episode extras — the same
    aggregation rsl_rl's Logger.log prints. None if the key never appeared."""
    vals = []
    for ep_info in ep_extras:
        if key not in ep_info:
            continue
        v = ep_info[key]
        if not isinstance(v, torch.Tensor):
            v = torch.tensor([v], dtype=torch.float32)
        vals.append(v.reshape(-1).float().cpu())
    if not vals:
        return None
    return float(torch.cat(vals).mean())


def attach_collapse_watchdog(runner, *, collapse_z: float, guard_from_iter: int,
                             sentinel: Path | None = None,
                             metric_key: str = "Metrics/lift_height/object_height") -> None:
    """Arm the collapse guard on a runner (call once, before runner.learn)."""
    logger = runner.logger
    orig_log = logger.log

    def guarded_log(*args, **kwargs):
        it = kwargs.get("it", args[0] if args else None)
        value = mean_ep_metric(logger.ep_extras, metric_key)  # before orig clears the buffer
        orig_log(*args, **kwargs)
        if it is None or value is None or it < int(guard_from_iter):
            return
        if value < float(collapse_z):
            if sentinel is not None:
                sentinel.parent.mkdir(parents=True, exist_ok=True)
                sentinel.write_text(
                    f"{strftime('%Y-%m-%d %H:%M:%S')} {metric_key}={value:.4f} "
                    f"< {collapse_z} at iter {it}\n")
            raise TrainingCollapseError(
                f"{metric_key}={value:.4f} < {collapse_z} at iter {it} — policy stopped lifting")

    logger.log = guarded_log
    print(f"[watchdog] armed: {metric_key} < {collapse_z} from iter {guard_from_iter} "
          f"aborts{f' (sentinel {sentinel})' if sentinel else ''}")
