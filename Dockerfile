# ==============================================================================
# TrustNet Stage 10 Production Dockerfile
#
# CUDA Runtime Base: nvidia/cuda:12.6.0-runtime-ubuntu22.04
# (Aligned with active PyTorch CUDA 12.6 wheel: torch 2.14.1+cu126)
#
# Architectural Safeguards:
# 1. Non-root user execution ('appuser', UID 1000) for principle of least privilege.
# 2. Zero baked weights: Models and indices are volume-mounted or downloaded to HF cache.
# 3. Exactly one worker (--workers 1): Prevents CUDA memory collisions and multiple process OOMs.
# 4. Integrated HEALTHCHECK utilizing /v1/health endpoint.
# ==============================================================================

ARG BASE_IMAGE=nvidia/cuda:12.6.0-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

# Prevent interactive debconf prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies (Python 3, pip, curl for healthcheck, and libGL/libgomp for OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create application working directory
WORKDIR /app

# Create unprivileged application user
RUN groupadd -g 1000 appuser && \
    useradd -u 1000 -g appuser -m -s /bin/bash appuser

# Copy dependency requirements first to leverage Docker layer caching
COPY pyproject.toml /app/

# Install Python packages
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python3 -m pip install --no-cache-dir \
        fastapi \
        uvicorn \
        sqlalchemy \
        python-multipart \
        pydantic-settings \
        pytest-asyncio \
        httpx \
        psutil \
        pyyaml \
        pillow \
        torch \
        transformers \
        sentence-transformers \
        open-clip-torch \
        easyocr \
        faiss-cpu \
        rank-bm25 \
        scikit-learn \
        scipy \
        numpy

# Copy application source code and configurations
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY scripts/ /app/scripts/

# Install TrustNet in editable / package mode
RUN python3 -m pip install --no-cache-dir -e .

# Create outputs directory and set file permissions for appuser
RUN mkdir -p /app/outputs/stage10 && \
    mkdir -p /home/appuser/.cache/huggingface && \
    chown -R appuser:appuser /app /home/appuser

# Switch to unprivileged non-root user
USER appuser

# Expose HTTP API port
EXPOSE 8000

# Docker Healthcheck targeting the /v1/health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
    CMD curl -f http://localhost:8000/v1/health || exit 1

# Command entrypoint: Exactly one Uvicorn worker process to prevent GPU memory duplication
CMD ["uvicorn", "src.trustnet.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
