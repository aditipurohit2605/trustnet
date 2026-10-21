# TrustNet Licenses, Models, and Dataset Inventory

A comprehensive registry of every dataset, pretrained model, and third-party library utilized across TrustNet, including licensing, commercial usability, attribution requirements, and canonical source URLs.

---

## 1. Datasets

| Dataset | License | Commercial Use? | Attribution Requirements | Source / Download URL |
| :--- | :--- | :---: | :--- | :--- |
| **FEVER (Fact Extraction and VERification)** | CC BY-SA 4.0 | Yes (Share-Alike) | Thorne et al., EMNLP 2018 citation | [https://fever.ai/dataset/fever.html](https://fever.ai/dataset/fever.html) |
| **Snopes Multimodal Fact-Check Benchmark** | Research & Academic Use | No (Non-commercial) | Snopes archive citation | [https://www.snopes.com/](https://www.snopes.com/) *(Fetch via `scripts/download_datasets.py`)* |
| **PIB & Indian Public Fact-Checks** | Open Government Data (OGD India) | Yes | Government of India citation | [https://factcheck.pib.gov.in/](https://factcheck.pib.gov.in/) |
| **BOOM Live Open Fact-Check Archive** | CC BY-NC 4.0 | No (Non-commercial) | BOOM Live citation | [https://www.boomlive.in/](https://www.boomlive.in/) *(Fetch via `scripts/download_datasets.py`)* |

> **Redistribution Policy**: Proprietary or non-commercial third-party benchmarks are not redistributed in binary form; execute `python scripts/download_datasets.py` to download and cryptographically verify raw instances.

---

## 2. Pretrained Models

| Model | Checkpoint ID / Hub Repo | License | Commercial Use? | Attribution Requirements |
| :--- | :--- | :--- | :---: | :--- |
| **mDeBERTa-v3** | `microsoft/mdeberta-v3-base` | MIT | Yes | He et al., ICLR 2021 citation |
| **MiniLM-L6-v2** | `sentence-transformers/all-MiniLM-L6-v2` | Apache 2.0 | Yes | Reimers and Gurevych, EMNLP 2019 |
| **CLIP ViT-B/32** | `openai/clip-vit-base-patch32` | MIT | Yes | Radford et al., ICML 2021 citation |
| **IndicTrans2 (Indic-En)** | `ai4bharat/indictrans2-indic-en-1B` | MIT | Yes | Gala et al., TACL 2023 citation |

---

## 3. Core Software Libraries

| Library | License | Commercial Use? | Role in System |
| :--- | :--- | :---: | :--- |
| **PyTorch** | BSD-3-Clause | Yes | Neural network inference and projection layers |
| **HuggingFace Transformers** | Apache 2.0 | Yes | Tokenizers and transformer model backbones |
| **LightGBM** | MIT | Yes | Gradient boosted tabular fusion baseline |
| **Scikit-learn** | BSD-3-Clause | Yes | Metrics, logistic regression, and evaluation tools |
| **Scipy & Statsmodels** | BSD-3-Clause | Yes | Bootstrap significance testing and McNemar's tests |
| **Matplotlib & Seaborn** | BSD-compatible | Yes | Paper figures and reliability diagram plotting |
| **FastAPI & Uvicorn** | MIT | Yes | Stage 10 asynchronous REST API backend |
| **python-telegram-bot** | LGPL-3.0 | Yes (dynamic) | Telegram verification bot interface |
| **React, Vite, TailwindCSS** | MIT | Yes | Stage 11 web dashboard interface |
