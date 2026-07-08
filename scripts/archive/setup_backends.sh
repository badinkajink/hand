#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTERNAL_DIR="${ROOT_DIR}/external"

mkdir -p "${EXTERNAL_DIR}"

clone_if_missing() {
  local repo_url="$1"
  local target_dir="$2"
  if [[ ! -d "${target_dir}/.git" ]]; then
    git clone "${repo_url}" "${target_dir}"
  fi
}

clone_if_missing "https://github.com/google-deepmind/mujoco_warp.git" "${EXTERNAL_DIR}/mujoco_warp"
clone_if_missing "https://github.com/asu-iris/comfree_warp.git" "${EXTERNAL_DIR}/comfree_warp"

echo "Backends cloned to ${EXTERNAL_DIR}."
echo "Install with:"
echo "  uv pip install -e external/mujoco_warp"
echo "  uv pip install -e external/comfree_warp"
