# GPU MJX Usage Guide

Date: 2026-04-13

This guide documents a practical way to run MJX on GPU from the project `uv` environment.

## Quick start with scripts

From repo root:

```bash
scripts/setup_gpu_mjx_env.sh --recreate
scripts/run_phase1_gpu_sweep.sh --y-values "0.0250 0.0300 0.0350" --iterations 12
```

## Why this lane

In this repo, we now run GPU MJX directly through `uv run` after ensuring CUDA-enabled JAX is installed in `.venv`.

## Compatibility snapshot

Project lane (in `.venv`, tested in this workspace):

- `mujoco>=3.6.0,<3.7`
- `mujoco-mjx>=3.6.0,<3.7`

Validated runtime:

- `jax==0.4.38`
- backend: `gpu`
- devices: `CudaDevice(id=0)`
- CUDA plugin/runtime packages pulled by `jax[cuda12]`

## 1) Install CUDA-enabled JAX in project venv

From repo root:

```bash
uv pip install --python .venv/bin/python "jax[cuda12]==0.4.38"
```

## 2) Validate GPU and MJX import

```bash
uv run python - <<'PY'
import jax
import mujoco
import mujoco.mjx as mjx
print("jax", jax.__version__)
print("backend", jax.default_backend())
print("devices", jax.devices())
print("mujoco", mujoco.__version__)
print("mjx module", mjx.__name__)
PY
```

Expected:

- backend is `gpu`
- at least one `CudaDevice(...)`
- `mujoco.mjx` imports successfully

## 3) Run Phase 1 with GPU lane

```bash
MUJOCO_GL=egl JAX_PLATFORM_NAME=gpu uv run python scripts/phase1_optimize_grasp.py \
  --scene-xml assets/mjcf/generated/scene_prism_x0.0200_y0.0300_z0.0200.xml \
  --optimizer mjx-autodiff \
  --iterations 12 \
  --seed 0 \
  --output-dir results/phase1/run_gpu_mjx \
  --tag y0d0300
```

Sweep example:

```bash
for y in 0.0250 0.0300 0.0350; do
  tag="y$(echo "$y" | tr '.' 'd')"
  MUJOCO_GL=egl JAX_PLATFORM_NAME=gpu uv run python scripts/phase1_optimize_grasp.py \
    --scene-xml "assets/mjcf/generated/scene_prism_x0.0200_y${y}_z0.0200.xml" \
    --optimizer mjx-autodiff \
    --iterations 12 \
    --seed 0 \
    --output-dir results/phase1/run_gpu_mjx_sweep \
    --tag "$tag"
done
```

## 4) Common failure modes

### CPU fallback warning

Symptom:

- JAX warns CUDA-enabled `jaxlib` is not installed and falls back to CPU.

Fix:

- Reinstall `jax[cuda12]` into the GPU env interpreter.
- Re-run validation snippet above.

### `mujoco.mjx` import error

Symptom:

- `ModuleNotFoundError: No module named 'mujoco.mjx'`

Fix:

- Install both `mujoco==3.6.0` and `mujoco-mjx==3.6.0` in the same env.

### Plugin attribute mismatch errors

Symptom:

- Errors from `jax_cuda12_plugin`/`jaxlib` symbol mismatches.

Cause:

- Mixed versions from partial upgrades/downgrades in one env.

Fix:

- Reinstall `jax[cuda12]` into `.venv` and re-run validation.

## 5) Operational recommendation

- Keep development and GPU MJX experiments on `uv run` once CUDA-enabled JAX is installed.
- Record lane and package versions in each experiment report for reproducibility.
