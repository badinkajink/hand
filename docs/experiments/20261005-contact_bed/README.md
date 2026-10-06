# Contact-model comparison bed: pull-to-slip runbook

Task 1 of the bed: a pinched tool pulled along its own axis until it slips. Four contact models on
the two-pad pinch rig of the 10-01 study (`scripts/hom_contact_rig.py`), gravity off, mu 1, the
real_v1 fingertip sphere (r 10.55 mm) and screwdriver (r 12.5 mm, 24.5 g).

| chain spec (`hom_chain.make_plant`) | rig spec used here |
|---|---|
| `mj:point3` | `mj:point3:ir100` |
| `mj:point4s` | `mj:point4s:fit0.000996044x0.2498:ir100` |
| `mj:spheres:s1:rs0.75:tr0.02` | `mj:spheres:s1:rs0.75:ir100:tr0.02` (cap 35 deg, the rig default) |
| `drake:hydro:rt0.01` | `drake:hydro:E1e7:r1:rt0.01` |

The rig specs are the ones `docs/experiments/20261001-hom_contact_patch/torsion.jsonl` was measured
with, so the torsion numbers of that study apply to the same models.

## Re-run

    PY=logs/20261001-hom_contact/venv/bin/python        # MuJoCo 3.6 + Drake 1.57
    ~/.claude/bin/resguard.sh status
    ~/.claude/bin/resguard.sh run --mem 1G --cpu 100 -- env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        $PY scripts/contact_bed_pull.py --out docs/experiments/20261005-contact_bed/pull_slip.jsonl

Set the thread limits inside the wrapped command, not before `resguard.sh`: the gate reads `nproc`,
which honours `OMP_NUM_THREADS`, and with 1 it refuses every launch.

The script skips (model, N, dt) cases already in the output and appends one fsynced JSON line per
case. One case: `--models mj:point3 --N 1 --dt 1`. The whole grid (4 models x N 0.5, 1, 3 N x
dt 1, 5 ms) ran single-core in 29 s at 412 MB peak RSS on 2026-10-05; Drake takes 27 s of it.

## What a row holds

Per case: settle normal forces, `F_onset` and `mu_eff = F_onset / (2N)`, `u_pre_mm` (axial
displacement before onset), `v_at_half_onset_mm_s` and `v_at_090_onset_mm_s` (pre-slip creep speed
during the ramp), `creep_mm_s` and `hold_disp_mm` (1 s hold at 50 % of the onset force),
`v_slip_mean50_mm_s` (mean slip speed over the first 50 ms after onset; rigid Coulomb with equal
static and kinetic friction gives `v_slip_mean50_coulomb_mm_s` = 34 mm/s), `v_slip_cv50` (its
per-step coefficient of variation: 0.89 for smooth Coulomb slip, higher values flag stick-slip),
`mu_slide` ((F - m a) / 2N, a from a quadratic fit of the displacement over 5-60 ms after onset), `us_per_step_median` (wall time of the rig's step call), plus 5 ms ramp and 10 ms hold
traces.

Onset definition: before onset the tool creeps at v = c F (c fitted between 30 and 80 % of the force
at which the speed first passes 10 mm/s). Onset is the last step on that law; every later step up to
detection has v > 2 c F + 0.2 mm/s. Force resolution is the ramp step, 2 mN at 1 ms and 10 mN at 5 ms. At 5 ms the MuJoCo point contacts'
solref time constant (6 ms) is clamped by MuJoCo to 2 dt = 10 ms.

## Status at the end of 2026-10-05 (session limit)

The bed was run by six parallel agents that all stopped at the account's session limit. Rows on disk:
`pull_slip.jsonl` (36, incl. the pad consistency set), `twist_slip.jsonl` (36), `shake.jsonl` (68, written by the
first version of `contact_bed_shake.py`; the agent had moved them aside to re-run with two extra Coulomb fields, and
they were restored unchanged), `brake.jsonl` (13), and Newton's `pull_slip_newton.jsonl`, `twist_slip_newton.jsonl`,
`static_newton.jsonl`; GPU rows in `../20261005-gpu_scaling/newton_scaling.jsonl` and `cpu_gpu_consistency.jsonl`.

Task 6 (creep) ran on 2026-10-06: `creep.jsonl` holds the 1 mm pads at relaxation 0.02, 0.05 and 0.1 s crossed with
impratio 100-10 000 and noslip 1-10 iterations on tasks 1, 2, 4 and 5 (`--variants 0:100 1:100 3:100 10:100 0:300 0:1000
10:1000`, then `0:3000 0:10000` on tasks 1, 2 and 5). Results and the printed-fingertip rig:
`docs/experiments/20261006-fingertip_backends/20261006-fingertip_contact_backends.html`.

Not run (scripts written, each resumes from its JSONL):

    PY=logs/20261001-hom_contact/venv/bin/python
    ~/.claude/bin/resguard.sh run --mem 2G --cpu 100 -- env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY scripts/contact_bed_roll.py
    ~/.claude/bin/resguard.sh run --mem 2G --cpu 100 -- env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY scripts/contact_bed_creep.py
    ~/.claude/bin/resguard.sh run --mem 2G --cpu 100 -- env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY scripts/contact_bed_stability.py
    logs/20261004-contact-transfer/venv/bin/python scripts/newton_scaling.py      # 4096-8192 worlds, unreduced 4096+
    logs/20261004-contact-transfer/venv/bin/python scripts/contact_bed_newton.py  # Newton shake, brake, roll

Pages: `python3 scripts/contact_bed_page.py` and `python3 scripts/contact_overview_page.py`, then republish to the
URLs in `artifact_url.txt` beside each page. Open finding: Newton carries 2.3-2.9x the hydroelastic law's torque at
spin onset (task 2); vary its friction gain (`kf10` in the spec) and the 100x tool stiffness first.
