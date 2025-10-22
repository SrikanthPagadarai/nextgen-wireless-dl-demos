FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    POETRY_HOME=/root/.local \
    PATH=/root/.local/bin:$PATH \
    PIP_CACHE_DIR=/root/.cache/pip \
    POETRY_CACHE_DIR=/root/.cache/pypoetry

# Persist caches even if the container exits
VOLUME ["/root/.cache/pip", "/root/.cache/pypoetry"]

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip \
    build-essential curl git ca-certificates bash findutils \
 && rm -rf /var/lib/apt/lists/*

# Install Poetry (via official install script)
RUN curl -sSL https://install.python-poetry.org | python3 -

WORKDIR /app

# Copy only dependency files first (for better Docker layer caching)
COPY pyproject.toml poetry.lock* /app/

# Create a virtualenv in /app/.venv so it stays with the project
ENV POETRY_VIRTUALENVS_IN_PROJECT=true

# Install base deps (no groups yet; runtime entrypoint will sync specific CPU/GPU group)
RUN poetry install --no-interaction --no-ansi --no-root || true

# Copy the rest of your project
COPY . /app

# Optional default (can be overridden at runtime, or chosen interactively)
ENV RUN_SCRIPT=""

# Copy entrypoint that allows choosing which script to run
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Default entrypoint
ENTRYPOINT ["/entrypoint.sh"]
