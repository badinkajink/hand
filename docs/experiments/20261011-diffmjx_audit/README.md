# DiffMJX audit runbook

The result page is `20261011-diffmjx_gradient_probe_audit.html`; its builder is `scripts/diffmjx_audit_page.py`
(`.venv/bin/python scripts/diffmjx_audit_page.py`). Every JSON file here is an input of the builder. The audited
studies are `docs/experiments/20261007-diffmjx/` and `docs/experiments/20261007-contact-gait/`; their scripts share
`scripts/real_v1_d1_continuation.py` since 2026-10-11 (tests: `tests/test_real_v1_d1_continuation.py`).

Check `~/.claude/bin/resguard.sh status` before each batch. Every CPU run below is one core and under 1 GB; the fork
runs use the isolated environment of `docs/experiments/20261007-diffmjx/README.md` on the CPU (up to 3.2 GB).

## CPU checks

```bash
uv run --no-sync python scripts/diffmjx_audit_probes.py interfinger fdstep cfd_reach frames
```

## Corrected 400 ms searches

Each line of the batch is `tip model forced cem_seed name`; `forced` 1 adds `--forced-release`:

```bash
~/.claude/bin/resguard.sh run --mem 1G --cpu 100 -- uv run --no-sync python scripts/real_v1_contact_gait_search.py \
  --tip tpu6 --contact-model padsT --seed 1 --budget 256 --fd-step 1e-2 --cem-seed 20261007 \
  --out docs/experiments/20261011-diffmjx_audit/searches/pads-seed1-256-fd1e-2-cem20261007.json
```

The batch covers `tpu6 padsT`, `tpu6 pt` and `sphere pt`, free and forced release with CEM seed 20261007, and the
free searches with CEM seeds 1-4. `searches/mesh-seed1-256-fdscipy.json` is the mesh search with `--fd-step scipy`,
which equals `20261007-mesh-seed1-256.json`. The eight-jitter replay and the summary:

```bash
S=docs/experiments/20261011-diffmjx_audit/searches
uv run --no-sync python scripts/real_v1_contact_gait_transfer.py \
  --plans $S/pads-seed1-256-fd1e-2-cem20261007.json $S/mesh-seed1-256-fd1e-2-cem20261007.json \
  $S/sphere-seed1-256-fd1e-2-cem20261007.json $S/pads-release-seed1-256-fd1e-2-cem20261007.json \
  $S/mesh-release-seed1-256-fd1e-2-cem20261007.json $S/sphere-release-seed1-256-fd1e-2-cem20261007.json \
  --out docs/experiments/20261011-diffmjx_audit/transfer-fd1e-2-mj360.json
uv run --no-sync python scripts/diffmjx_audit_probes.py corrections
```

## Regression checks of the shared module

```bash
R=docs/experiments/20261011-diffmjx_audit/regression
G=docs/experiments/20261007-contact-gait
F=logs/20261007-diffmjx/venv/bin/python
uv run --no-sync python scripts/real_v1_contact_gait_search.py --tip tpu6 --contact-model pt --seed 1 --budget 256 --fd-step scipy --out $R/mesh-seed1-256.json
uv run --no-sync python scripts/real_v1_contact_gait_search.py --tip tpu6 --contact-model padsT --seed 1 --budget 256 --fd-step scipy --out $R/padsT-seed1-256.json
uv run --no-sync python scripts/real_v1_contact_gait_transfer.py --plans $G/20261007-padsT-seed1-256.json $G/20261007-mesh-seed1-256.json --seeds 1 2 --out $R/transfer-mj360-seeds12.json
uv run --no-sync python scripts/real_v1_diffmjx_pad_cpu.py $R/pad_mask_cpu.json
$F scripts/real_v1_diffmjx_local_search.py --gradient-actions --out $R/pad_local_search_20_mj331.json
JAX_PLATFORM_NAME=cpu $F scripts/real_v1_diffmjx_official_gate.py --steps 1 --tip tpu6 --contact-model padsT --pad-spacing-mm 2 --one-direction-gradient --out $R/official_padsT2_cylinder_jvp_1.json
JAX_PLATFORM_NAME=cpu $F scripts/real_v1_diffmjx_pad_jvp.py --steps 1 --pad-spacing-mm 2 --all-directions --out $R/padjvp_padsT2_1_all.json
JAX_PLATFORM_NAME=cpu JAX_ENABLE_X64=1 uv run --no-sync python scripts/real_v1_diffmjx_gate.py --steps 20 --no-post --tip sphere --contact-model pt --out $R/native_sphere_20_cpu.json
WARP_CACHE_PATH=$(mktemp -d) uv run --no-sync python scripts/real_v1_contact_gait_warp.py --plans $G/20261007-padsT-seed1-256.json --tip tpu6 --contact-model padsT --seeds 1 2 --methods hold cem --out $R/warp-padsT-free-seeds12.json
uv run --no-sync python scripts/diffmjx_audit_probes.py module_check
```

`pytest` needs `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` on this workstation: the ROS Humble plugins on the Python path fail
to import otherwise.
