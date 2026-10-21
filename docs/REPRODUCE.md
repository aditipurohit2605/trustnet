# TrustNet Reproducibility Guide

This document outlines the exact, deterministic steps required to reproduce every paper table, figure, and hypothesis test from a clean checkout.

---

## 1. System Requirements & Hardware Budget
- **Operating System**: Linux (Ubuntu 20.04+), macOS (12+), or Windows 10/11 (PowerShell).
- **Python Version**: Python 3.10, 3.11, or 3.12.
- **Compute / GPU**: Operates on CPU-only instances. GPU (CUDA 11.8/12.1) is optional.
- **Peak GPU Memory**: `< 500 MB` (CPU execution requires zero VRAM).
- **RAM**: Minimum 4 GB RAM recommended.
- **Disk Space**: ~2 GB for virtualenv, checkpoints, and evaluation artifacts.

---

## 2. Cryptographic Hashes of Frozen Assets
Verify before running evaluation:

| Asset | Path | Expected SHA-256 Digest |
| :--- | :--- | :--- |
| **Test Split** | `data/processed/test.jsonl` | `c5ed7665f11e555139f96916541ab5310faee03ac7747098883223b9548387c2` |
| **Data Manifest** | `data/manifest.json` | `5c2826fc1b302c111059b02aa2f23ea3cf1df1641014a60117b3cb6fdb75ba95` |
| **Fusion Model** | `outputs/stage6/models/mlp_full/mlp_model.pt` | Verified by `freeze_manifest.json` |
| **Calibrator** | `outputs/stage7/temperature.json` | Verified by `freeze_manifest.json` |

---

## 3. Step-by-Step Reproduction Commands

### Step 1: Environment Setup
```bash
# Clone and enter workspace
git clone https://github.com/aditipurohit2605/trustnet.git
cd trustnet

# Create virtual environment
python -m venv .venv

# Activate environment
# On Linux / macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Install package and release dependencies
pip install --upgrade pip
pip install -e .
pip install scipy statsmodels seaborn matplotlib python-telegram-bot
```
*Expected runtime: ~1–2 minutes.*

---

### Step 2: Verify Evaluation Freeze
```bash
python -m trustnet.final.freeze_check
```
*Expected Output*: `[FREEZE CHECK PASSED] All model hashes, configs, and dataset splits verified.`  
*Expected Runtime*: `< 2 seconds.*  
*GPU Memory*: `0 MB`.

---

### Step 3: Run Final Test Split Evaluation
```bash
python -m trustnet.final.run_test --final
```
*Expected Output*:  
- Generates `outputs/final/predictions/test_predictions.jsonl`
- Generates `outputs/final/predictions/test_summary_metrics.json`
- Generates `outputs/final/run_manifest.json`
- Generates `outputs/final/run_test.log`  
*Expected Runtime*: `~3–5 seconds.*  
*GPU Memory*: `0 MB` (or `< 200 MB` on CUDA).

---

### Step 4: Generate Paper-Ready Tables and Figures
```bash
python -m trustnet.final.tables
```
*Outputs*:  
- `outputs/final/tables/table1_main_comparison.tex` & `.md`
- `outputs/final/tables/table2_ablations.tex` & `.md`
- `outputs/final/tables/table3_calibration.tex` & `.md`
- `outputs/final/tables/table4_coverage_at_error.tex` & `.md`
- `outputs/final/tables/table5_retrieval.tex` & `.md`
- `outputs/final/tables/table6_latency_memory.tex` & `.md`
- `outputs/final/figures/reliability_diagrams.png` & `.pdf`
- `outputs/final/figures/risk_coverage_curves.png` & `.pdf`  
*Expected Runtime*: `< 2 seconds.* (Zero model loading).

---

### Step 5: Evaluate Formal Hypotheses (H1 to H5)
```bash
python -m trustnet.final.hypotheses
```
*Outputs*: `outputs/final/hypotheses.md`  
*Expected Runtime*: `< 2 seconds.*

---

### Step 6: Subgroup & Demographic Slice Analysis
```bash
python -m trustnet.final.slices
```
*Outputs*: `outputs/final/slices.md` & `outputs/final/slices.json`  
*Expected Runtime*: `< 1 second.*

---

### Step 7: Qualitative Error Sampling & Attribution
```bash
# 1. Sample errors for manual review
python -m trustnet.final.error_analysis sample-errors --sample-size 50 --output-csv outputs/final/error_sample.csv

# 2. Compute error breakdown and highest-confidence false assertions
python -m trustnet.final.error_analysis analyze-errors --annotated-csv outputs/final/error_sample.csv
```
*Outputs*: `outputs/final/error_sample.csv` and error attribution breakdown.  
*Expected Runtime*: `< 1 second.*

---

## 4. Total Reproduction Budget
- **Total CPU Runtime**: `< 15 seconds` across all evaluation scripts.
- **Total Peak GPU VRAM**: `0 MB` (CPU fallback completely verified).
- **Deterministic**: Seeded bootstrap ($B=1000, \text{seed}=42$) guarantees numerical reproducibility.
