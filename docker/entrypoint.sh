#!/usr/bin/env bash
set -euo pipefail

# Detect GPU presence inside the container
if command -v nvidia-smi >/dev/null 2>&1 || [ -d "/proc/driver/nvidia" ]; then
  INSTALL_GROUP="gpu"
else
  INSTALL_GROUP="cpu"
fi

echo "Detected environment: ${INSTALL_GROUP}"
echo "Installing dependencies with Poetry…"
poetry install --with "${INSTALL_GROUP}"

echo "Running: ${RUN_SCRIPT}"
# Forward any extra args to your script: `docker run ... -- --arg value`
exec poetry run python3 "${RUN_SCRIPT}" "$@"