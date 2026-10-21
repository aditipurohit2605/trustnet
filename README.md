# TrustNet

TrustNet is a modular Python framework designed for developing trustworthy, multimodal artificial intelligence systems by integrating text, vision, and information retrieval pipelines with multimodal fusion, probability calibration, selective abstention, multi-factor reasoning, and production-grade API serving.

---

## Stage 10: Backend Verification API

Stage 10 provides an asynchronous, high-throughput verification REST API built on FastAPI, SQLite in Write-Ahead Logging (WAL) mode, and a dedicated single GPU worker queue.

### Architecture Highlights
- **Single GPU Worker**: Jobs are queued in a bounded FIFO queue and processed sequentially by a single background worker to prevent CUDA VRAM thrashing and process termination.
- **Constant-Time Authentication**: Validates API keys via `secrets.compare_digest` from environment variables (`TRUSTNET_API_KEYS`).
- **Resilient Fallback & Graceful Degradation**: If individual components (retrieval, OCR, reasoning) experience failures, the job does not fail; instead, issues are reported in `degraded` and `warnings` fields of the `ReportEnvelope`.
- **Privacy By Design**: Request bodies, message text, user images, and EXIF summaries are strictly never logged. Uploaded image binaries are stored in temporary memory/disk and immediately unlinked upon completion.
- **Memory Management**: Core models load once on startup; heavy secondary models (MarianMT translator, BLIP captioner) load lazily with automated idle-unload timers.
- **SSRF Hardening**: Message URLs are never fetched over the network; claims are verified strictly against authoritative, pre-indexed knowledge corpora.

---

### Host Prerequisites

1. **Local Python Execution**:
   - Python 3.10 to 3.13
   - SQLite 3.35+
   - PyTorch 2.14+ (CUDA 12.6 or CPU)
   - ~4 GB host RAM (CPU mode) or ~8 GB dedicated VRAM (GPU mode)

2. **Docker / Container Execution**:
   - Docker Engine 24.0+ and Docker Compose v2.20+
   - For GPU execution: **NVIDIA Container Toolkit** (`nvidia-ctk`) installed and configured on the host.

---

### Running Locally

1. **Install dependencies**:
   ```bash
   pip install -e .
   pip install fastapi uvicorn sqlalchemy python-multipart pydantic-settings pytest-asyncio httpx psutil
   ```

2. **Set environment variables**:
   ```bash
   # Windows PowerShell
   $env:ENVIRONMENT="development"
   $env:TRUSTNET_API_KEYS="dev-key-12345"

   # Linux / macOS
   export ENVIRONMENT=development
   export TRUSTNET_API_KEYS=dev-key-12345
   ```

3. **Start the API server**:
   ```bash
   uvicorn src.trustnet.api.app:app --host 0.0.0.0 --port 8000 --workers 1
   ```
   *(Note: Always run with `--workers 1` to ensure single-worker GPU inference.)*

4. **Export OpenAPI Specification**:
   ```bash
   python scripts/export_openapi.py
   ```

5. **Profile Model Memory**:
   ```bash
   python scripts/measure_memory.py
   ```

6. **Execute Concurrent Load Tests**:
   ```bash
   python scripts/loadtest.py --url http://127.0.0.1:8000 --api-key dev-key-12345
   ```

---

### Running in Docker

1. **Copy configuration**:
   ```bash
   cp .env.example .env
   ```

2. **Launch with GPU profile (NVIDIA GPUs)**:
   ```bash
   docker compose --profile gpu up --build
   ```

3. **Launch with CPU profile (Laptops / Non-NVIDIA hosts)**:
   ```bash
   docker compose --profile cpu up --build
   ```

4. **Check container health**:
   ```bash
   curl -f http://localhost:8000/v1/health
   ```

---

## API Endpoints Reference

| Method | Path | Description | Auth Required |
| :--- | :--- | :--- | :---: |
| `POST` | `/v1/verify` | Submit multipart claim verification job (text and/or image) | Optional in dev / Required in prod |
| `GET` | `/v1/jobs/{job_id}` | Poll asynchronous job status, stage progress, and envelope | Optional in dev / Required in prod |
| `GET` | `/v1/reports/{report_id}` | Retrieve complete verification report envelope | Optional in dev / Required in prod |
| `DELETE` | `/v1/reports/{report_id}` | Permanently purge stored report record | Optional in dev / Required in prod |
| `POST` | `/v1/feedback` | Ingest human feedback (enforcing consent_to_store) | Optional in dev / Required in prod |
| `GET` | `/v1/health` | Healthcheck and readiness probe | No |
| `GET` | `/v1/memory` | Real-time GPU VRAM and host RAM memory telemetry | Optional in dev / Required in prod |

---

## Stage 2: Text-Only Baseline Classifier

The text baseline fine-tunes sequence classification transformers (such as `microsoft/deberta-v3-base`) to verify claims against a three-class label space: `verified`, `contradicted`, and `insufficient_evidence`.

### Training
```bash
python -m trustnet.text.train --config configs/stage2_baseline.yaml --input_mode claim_only --output_dir outputs/stage2/claim_only
```

### Testing
Run automated unit and integration tests:
```bash
pytest -v
```