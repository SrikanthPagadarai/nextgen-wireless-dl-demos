FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    POETRY_HOME=/root/.local \
    PATH=/root/.local/bin:$PATH

# Cache dirs (so wheels/metadata are reused between runs)
ENV PIP_CACHE_DIR=/root/.cache/pip \
    POETRY_CACHE_DIR=/root/.cache/pypoetry

# Persist caches even if the container exits
VOLUME ["/root/.cache/pip", "/root/.cache/pypoetry"]

# System deps
 RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip build-essential curl git ca-certificates \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - \
 && poetry --version

WORKDIR /app

# Copy just dependency files first to leverage Docker layer cache if they don't change
COPY pyproject.toml poetry.lock* ./

# Ensure venv is created inside project folder (/app/.venv)
RUN poetry config virtualenvs.in-project true

# Copy the rest of your project
COPY . /app

# Runtime configuration:
# You can override RUN_SCRIPT at `docker run` time if needed
ENV RUN_SCRIPT="mimo_ofdm_over_cdl/mimo_ofdm_over_cdl.py"

# Add entrypoint that picks CPU/GPU env and runs your script
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# If using NVIDIA Container Toolkit, the runtime will inject GPU devices.
# No EXPOSE needed unless you serve a web app.
ENTRYPOINT ["/entrypoint.sh"]
