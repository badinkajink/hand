#!/usr/bin/env bash
# One-time environment build on the DeltaAI LOGIN node (runbook step 3).
#
# Differences from the ACT/lerobot recipe in docs/nsf_access_runbook.pdf:
#   - we do NOT layer a venv on the site torch module. uv owns the whole env,
#     because mjlab/mujoco-warp pin versions no site module will match.
#   - the arch trap is the same one that runbook hit, one layer down: every wheel
#     must be the aarch64 build. All of ours have one (checked 2026-08-20):
#     mujoco 3.6.0 manylinux_2_28_aarch64, warp-lang 1.12.x manylinux_2_34_aarch64,
#     torch +cu128 manylinux_2_28_aarch64, mjlab/mujoco-mjx pure python.
#   - the login node has outbound internet and NO GPU; compute nodes are the
#     reverse. So everything that downloads happens here, and everything that
#     touches CUDA (Warp kernel compilation included) happens in deltaai_smoke.sh
#     under srun. Do not try to prime the Warp cache here — wp.init() has no device.
#
#   cd ~/hand && bash scripts/cluster/deltaai_env_setup.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
echo "== arch: $(uname -m)   (must be aarch64)"
[ "$(uname -m)" = "aarch64" ] || echo "!! not aarch64 — are you on the login node?"
echo "== slurm accounts:"; accounts 2>/dev/null || echo "   ('accounts' not found — run it after login to get your --account)"

module purge
module load cuda || echo "!! 'module load cuda' failed; check 'module avail cuda'"

if ! command -v uv >/dev/null 2>&1; then
  echo "== installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

# CUDA 12.8 wheels want a >=570 driver. If the compute node's nvidia-smi reports
# older, flip the pytorch index in pyproject.toml from cu128 to cu126 — the same
# torch versions ship aarch64 cu126 wheels (checked 2026-08-20) — and re-sync.
echo "== uv sync --extra gpu --extra rl   (first run pulls ~4 GB of aarch64 wheels)"
uv sync --extra gpu --extra rl

echo "== installed wheel arch check"
uv run python - <<'PY'
import platform, importlib
print("python", platform.python_version(), platform.machine())
for m in ("torch", "warp", "mujoco", "mujoco_warp", "mjlab"):
    try:
        mod = importlib.import_module(m)
        print(f"  {m:12s} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"  {m:12s} IMPORT FAILED: {type(e).__name__}: {e}")
PY

cat <<'EOF'

Environment built. Nothing has touched a GPU yet — that is the next step, and it
must happen on a compute node:

  sbatch scripts/cluster/deltaai_smoke.slurm       # primes the Warp cache + times a real run

Read its log before you queue anything in bulk. It prints the steps/s that every
GPU-hour estimate in docs/notes/20260820-deltaai_bulk_training_runbook.md depends on.
EOF
