#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q
