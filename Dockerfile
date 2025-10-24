# syntax=docker/dockerfile:1.7

# Build-time switches
# Default: GPU image on CUDA 12.2 + cuDNN 8 (matches TF 2.15.1 wheel reporting CUDA 12.2)
ARG BASE_IMAGE=nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04
ARG ENABLE_GPU=1                  # 1 = GPU image (CUDA present), 0 = CPU-only image
ARG ENABLE_CUDA_CHECK=1           # 1 = verify TF wheel's CUDA matches image CUDA (GPU builds)
ARG TF_PACKAGE=tensorflow         # "tensorflow" (GPU) or "tensorflow-cpu" (CPU builds)
ARG TF_VERSION=2.15.1             # Pin TF exactly to avoid surprise upgrades

FROM ${BASE_IMAGE} AS runtime

# Common environment
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}" \
    TF_CPP_MIN_LOG_LEVEL=2

# Harmless on CPU; useful on GPU when XLA JIT compiles kernels
ENV XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda

WORKDIR /app

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip \
    build-essential curl git ca-certificates vim \
 && rm -rf /var/lib/apt/lists/*

# Python base + Poetry
RUN python3 -m venv "${VIRTUAL_ENV}" \
 && python -m pip install --upgrade pip wheel setuptools \
 && python -m pip install "poetry==1.8.3"

# App deps via Poetry (no TF here)
COPY pyproject.toml poetry.lock* /app/
ARG POETRY_WITH=""
RUN poetry config virtualenvs.create false \
 && (poetry lock --no-interaction --no-update || poetry lock --no-interaction) \
 && poetry install --no-interaction --no-root --only main ${POETRY_WITH:+--with ${POETRY_WITH}}

# Install TensorFlow last so it "wins"
ARG TF_PACKAGE
ARG TF_VERSION
RUN python -m pip install --upgrade --force-reinstall "${TF_PACKAGE}==${TF_VERSION}"

# (GPU only) Minimal CUDA toolchain for XLA JIT
# Pulls in nvvm/libdevice/ptxas so TF XLA can compile device code.
ARG ENABLE_GPU
RUN if [ "${ENABLE_GPU}" = "1" ]; then \
      set -e; \
      CUDA_MM="$(python - <<'PY'\nimport json,sys\nprint('.'.join(json.load(open('/usr/local/cuda/version.json'))['cuda']['version'].split('.')[:2]))\nPY\n)"; \
      apt-get update; \
      if [ "${CUDA_MM}" = "12.2" ]; then \
        apt-get install -y --no-install-recommends cuda-nvcc-12-2; \
      elif [ "${CUDA_MM}" = "11.8" ]; then \
        apt-get install -y --no-install-recommends cuda-nvcc-11-8; \
      else \
        # Fallback to meta-package for other CUDA minors
        apt-get install -y --no-install-recommends cuda-nvcc; \
      fi; \
      rm -rf /var/lib/apt/lists/*; \
    fi

# (GPU only) Build-time sanity check: TF wheel CUDA == image CUDA
ARG ENABLE_CUDA_CHECK
RUN if [ "${ENABLE_GPU}" = "1" ] && [ "${ENABLE_CUDA_CHECK}" = "1" ]; then \
      python - <<'PY'\nimport json,sys,tensorflow as tf\nimg=json.load(open('/usr/local/cuda/version.json'))['cuda']['version']; img_mm='.'.join(img.split('.')[:2])\nb=tf.sysconfig.get_build_info(); tf_cuda=str(b.get('cuda_version','')).strip()\nprint(f\"[CHECK] Image CUDA: {img_mm} | TF wheel CUDA: {tf_cuda}\")\nif not tf_cuda.startswith(img_mm):\n    print('[ERROR] CUDA mismatch between image and TensorFlow wheel.'); sys.exit(1)\nPY\n; \
    fi

# App code & entrypoint
COPY . /app
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["python3", "mimo_ofdm_over_cdl/training.py"]
