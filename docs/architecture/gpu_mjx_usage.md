# GPU MJX Usage Guide

Date: 2026-04-13

This guide documents a practical way to run MJX on GPU without destabilizing the default project environment.

## Quick start with scripts

From repo root:

```bash
scripts/setup_gpu_mjx_env.sh --recreate
scripts/run_phase1_gpu_sweep.sh --y-values "0.0250 0.0300 0.0350" --iterations 12
```

## Why separate lanes

In this repo, the default `uv run` lane is pinned for reproducibility across Python versions and backend extras.
A CUDA JAX stack can work, but mixing it into the same environment often causes resolver and plugin mismatches.

Use two lanes:

- Stable lane (default): reproducible CPU-JAX via project pins.
- GPU lane (separate venv): MJX performance experiments.

## Compatibility snapshot

Stable lane (in `pyproject.toml`):

- `mujoco>=3.6.0,<3.7`
- `mujoco-mjx>=3.6.0,<3.7`
- `jax>=0.4.30,<0.5`
- `jaxlib>=0.4.30,<0.5`

GPU lane (separate env, tested in this workspace):

- `mujoco==3.6.0`
- `mujoco-mjx==3.6.0`
- `jax==0.9.2`
- `jaxlib==0.9.2`
- CUDA plugin/runtime packages pulled by `jax[cuda12]`

## 1) Create isolated GPU env

From repo root:

```bash
uv venv --python 3.12 .venv-gpu
```

Install GPU lane packages into that interpreter:

```bash
uv pip install --python .venv-gpu/bin/python \
  "mujoco==3.6.0" "mujoco-mjx==3.6.0" "jax[cuda12]==0.9.2"

uv pip install --python .venv-gpu/bin/python --no-deps -e .

uv pip install --python .venv-gpu/bin/python \
  "matplotlib>=3.8" "imageio>=2.34" "pillow>=10.0" "tyro>=0.8.5" "pyyaml>=6.0.2"
```

Optional backend packages:

```bash
uv pip install --python .venv-gpu/bin/python -e external/mujoco_warp -e external/comfree_warp
```

## 2) Validate GPU and MJX import

```bash
.venv-gpu/bin/python - <<'PY'
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

Use direct interpreter execution from the GPU env (do not use `uv run` for this lane):

```bash
MUJOCO_GL=egl .venv-gpu/bin/python scripts/phase1_optimize_grasp.py \
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
  MUJOCO_GL=egl .venv-gpu/bin/python scripts/phase1_optimize_grasp.py \
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

- Delete and recreate `.venv-gpu` from scratch.
- Reinstall exactly as in this guide.

## 5) Operational recommendation

- Keep default development on `uv run` (stable lane).
- Run GPU MJX experiments only with `.venv-gpu/bin/python`.
- Record lane and package versions in each experiment report for reproducibility.
