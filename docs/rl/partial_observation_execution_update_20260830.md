# Partial-observation reorientation: execution update

Date: 2026-08-30

## Outcome

The scheduled skunkworks experiment did **not** run. The one-shot timer fired at the intended time,
but Codex exited before initializing the resumed conversation. Consequently, this attempt produced no
training results and provides no empirical evidence for or against the partial-observation/RMA
approach.

The research and implementation proposal remains in
`docs/rl/partial_observation_transfer.md`; none of its experiment phases were executed by the
scheduled job.

## Timeline and evidence

| Event | Local time (MDT) | Result |
|---|---|---|
| Timer created | 2026-08-29 21:32:28 | Accepted by user systemd |
| Scheduled trigger | 2026-08-29 22:02:28 | Timer activated the service |
| Codex process exited | 2026-08-29 22:02:57 | Exit status 1 after 643 ms CPU time |

The service log reports:

```text
failed to initialize thread persistence: thread-store conflict:
thread 01a05097-3da2-71e0-92f9-6c75152f1177 already has an active writer
```

The failure occurred in `codex exec resume`, before a model turn, goal, repository inspection,
implementation, or training process began. `get_goal` subsequently reported no active or completed
goal.

Verification commands:

```bash
systemctl --user status morphohand-rma-skunkworks-20260829.service --no-pager
journalctl --user -u morphohand-rma-skunkworks-20260829.service --no-pager
find docs results -type f -newermt '2026-08-29 22:00:00'
git status --short
```

The artifact search found no new files in `docs/` or `results/` from the scheduled execution window.
The working tree changes visible on 2026-08-30 were already present before the timer was created and
must not be attributed to this run.

## Deliverables status

| Requested deliverable | Status |
|---|---|
| Asymmetric deploy-observation environment | Not started |
| Recurrent actor baseline | Not started |
| Observability probes | Not started |
| Anchor/oracle/q-only/q+load comparisons | Not started |
| Multi-seed training | Not started |
| Held-out quantitative metrics | Not started |
| Rendered success/failure videos | Not produced |
| Empirical viability conclusion | Not available |

## Root cause

The fallback scheduling design attempted to resume the same thread from a second local Codex process:

```text
systemd timer -> codex exec resume <desktop-thread-id> -> /goal ...
```

The desktop application retained the active writer for that thread. Codex's thread store correctly
rejected the second writer to prevent concurrent corruption. The timer itself worked; same-thread CLI
resumption was the invalid part of the design.

## Corrected execution design

Use one of these mechanisms:

1. **Native in-chat scheduled task.** This is the appropriate way to return to a chat while preserving
   its context, but the scheduled-task creation capability must be exposed to the session.
2. **Detached local Codex run.** If using a system timer, launch a new `codex exec` session rather than
   resuming the desktop thread. Give it a durable prompt pointing to
   `docs/rl/partial_observation_transfer.md`, run it in this repository, and require it to write all
   code, metrics, videos, exact commands, and its final report to versioned/result paths. The desktop
   conversation can inspect those artifacts afterward.
3. **Interactive durable goal.** Start `/goal` in this conversation while it owns the thread and allow
   it to continue until the explicit experiment stopping conditions are met.

For either executable path, retain the previous safety and integrity constraints:

- simulation only; never operate real hardware;
- preserve unrelated dirty/concurrent changes;
- record baseline and negative results, not just successful rollouts;
- verify video files and report exact checkpoints/configurations;
- distinguish completed multi-seed evidence from smoke tests or partial runs.

## Current scientific conclusion

There is no experimental update. The only supported conclusion remains the pre-experiment assessment:
the fixed-axis, anchor-residual task is a plausible candidate for recurrent asymmetric control and
RMA-like latent supervision, but its observability and sim-to-real viability remain untested.

