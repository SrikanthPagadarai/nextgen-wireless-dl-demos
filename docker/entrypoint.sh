#!/usr/bin/env bash
set -euo pipefail

cd /app

# --- Helper: install deps for CPU/GPU depending on availability ---
detect_install_group() {
  if command -v nvidia-smi >/dev/null 2>&1 || [ -d "/proc/driver/nvidia" ]; then
    echo "gpu"
  else
    echo "cpu"
  fi
}

ensure_poetry_env() {
  local group="$1"
  echo "Detected environment: ${group}"
  echo "Installing/syncing dependencies with Poetry…"
  if [ -x "/app/.venv/bin/python" ]; then
    poetry install --with "${group}" --sync --no-interaction --no-ansi
  else
    poetry install --with "${group}" --no-interaction --no-ansi
  fi
}

# --- Helper: interactive menu to pick a script ---
pick_script_interactively() {
  echo "No script specified. Scanning for likely entry scripts…"
  # Prefer common entry-point names; fall back to all *.py (depth <= 3)
  mapfile -t candidates < <( \
    { \
      find . -maxdepth 3 -type f \( -name "train.py" -o -name "inference.py" -o -name "sim.py" -o -name "main.py" \); \
      find . -maxdepth 3 -type f -name "*.py"; \
    } | sed 's|^\./||' | awk '!seen[$0]++' \
  )

  if [ "${#candidates[@]}" -eq 0 ]; then
    echo "No Python scripts found under /app. Dropping into a shell."
    exec bash
  fi

  echo
  echo "Select a script to run (enter number):"
  select choice in "${candidates[@]}" "bash (open shell)"; do
    if [[ "$REPLY" == "$(( ${#candidates[@]} + 1 ))" ]]; then
      exec bash
    fi
    if [[ -n "${choice:-}" && -f "$choice" ]]; then
      echo "$choice"
      return 0
    fi
    echo "Invalid selection. Try again."
  done
}

# --- Main ---
GROUP="$(detect_install_group)"
ensure_poetry_env "$GROUP"

# USAGE MODES:
# 1) docker run <img> bash                   -> open shell
# 2) docker run <img> path/to/script.py ...  -> run that script with args
# 3) docker run <img>                        -> interactive menu to pick a script
# 4) docker run -e RUN_SCRIPT=... <img> ...  -> use RUN_SCRIPT unless an explicit script is given as $1

if [[ "${1:-}" == "bash" ]]; then
  shift || true
  exec bash "$@"
fi

SCRIPT_TO_RUN=""
if [[ $# -gt 0 ]]; then
  # First arg provided: treat it as script path if it's a .py file; otherwise pass to python -m
  if [[ "$1" == *.py && -f "$1" ]]; then
    SCRIPT_TO_RUN="$1"
    shift
  else
    # If it's a module like "package.module", run with -m
    MODULE_CAND="$1"
    shift
    if [[ -n "$MODULE_CAND" ]]; then
      echo "Running module: $MODULE_CAND"
      exec poetry run python3 -m "$MODULE_CAND" "$@"
    fi
  fi
elif [[ -n "${RUN_SCRIPT:-}" ]]; then
  SCRIPT_TO_RUN="$RUN_SCRIPT"
fi

if [[ -z "$SCRIPT_TO_RUN" ]]; then
  # Prompt user to pick one (requires -it)
  SCRIPT_TO_RUN="$(pick_script_interactively)"
fi

echo "Running: ${SCRIPT_TO_RUN} $*"
exec poetry run python3 "${SCRIPT_TO_RUN}" "$@"
