#!/usr/bin/env bash
set -euo pipefail

cd /app

fast_exec_shell_if_requested() {
  case "${1:-}" in
    bash|/bin/bash)
      shift || true
      exec bash "$@"
      ;;
    sh|/bin/sh)
      shift || true
      exec sh "$@"
      ;;
  esac
}

# Always honor explicit shell request BEFORE any dependency work
fast_exec_shell_if_requested "${1:-}" "$@"

# --- Optional Poetry sync (disabled by default) ---
# Set POETRY_AUTO_SYNC=1 to enable install/sync at container start.
detect_install_group() {
  if command -v nvidia-smi >/dev/null 2>&1 || [ -d "/proc/driver/nvidia" ]; then
    echo "gpu"
  else
    echo "cpu"
  fi
}

maybe_sync_poetry() {
  if [[ "${POETRY_AUTO_SYNC:-0}" != "1" ]]; then
    echo "Skipping Poetry sync (POETRY_AUTO_SYNC!=1)."
    return 0
  fi

  if ! command -v poetry >/dev/null 2>&1; then
    echo "Poetry not found in PATH. Skipping sync."
    return 0
  fi

  local group
  group="$(detect_install_group)"
  echo "Poetry sync enabled. Detected environment: ${group}"
  if [ -x "/app/.venv/bin/python" ]; then
    poetry install --with "${group}" --sync --no-interaction --no-ansi
  else
    poetry install --with "${group}" --no-interaction --no-ansi
  fi
}

# If a Python script/module is requested, we may want deps; otherwise we'll skip
# Decide what to run
SCRIPT_TO_RUN=""
RUN_MODE="script"  # or "module"

if [[ $# -gt 0 ]]; then
  if [[ "$1" == *.py && -f "$1" ]]; then
    SCRIPT_TO_RUN="$1"
    shift
  else
    # Treat as module name (e.g., package.module)
    RUN_MODE="module"
    SCRIPT_TO_RUN="$1"
    shift
  fi
elif [[ -n "${RUN_SCRIPT:-}" ]]; then
  SCRIPT_TO_RUN="$RUN_SCRIPT"
fi

if [[ -z "$SCRIPT_TO_RUN" ]]; then
  # Interactive menu (lightweight find, depth <= 2 to avoid slow scans on huge mounts)
  echo "No script provided. Select one:"
  mapfile -t candidates < <( \
    { \
      find . -maxdepth 2 -type f \( -name "training.py" -o -name "inference.py" -o -name "train.py" -o -name "sim.py" -o -name "main.py" \); \
      find . -maxdepth 2 -type f -name "*.py"; \
    } | sed 's|^\./||' | awk '!seen[$0]++' \
  )
  if [ "${#candidates[@]}" -eq 0 ]; then
    echo "No Python scripts found under /app. Opening a shell."
    exec bash
  fi
  select choice in "${candidates[@]}" "bash (open shell)"; do
    if [[ "$REPLY" == "$(( ${#candidates[@]} + 1 ))" ]]; then
      exec bash
    fi
    if [[ -n "${choice:-}" && -f "$choice" ]]; then
      SCRIPT_TO_RUN="$choice"
      RUN_MODE="script"
      break
    fi
    echo "Invalid selection. Try again."
  done
fi

# Only now (when we’re about to run Python) consider syncing dependencies
maybe_sync_poetry

if [[ "$RUN_MODE" == "module" && "$SCRIPT_TO_RUN" != *.py ]]; then
  echo "Running module: $SCRIPT_TO_RUN $*"
  exec poetry run python3 -m "$SCRIPT_TO_RUN" "$@"
else
  echo "Running: $SCRIPT_TO_RUN $*"
  exec poetry run python3 "$SCRIPT_TO_RUN" "$@"
fi
