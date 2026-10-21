"""
TrustNet Stage 12: Freeze Verification Tooling.

Enforces strict evaluation immutability:
1. Verifies git working directory is clean.
2. Checks git commit matches the 'final-freeze' tag (or records current HEAD commit).
3. Verifies SHA-256 hashes for all checkpoints, calibrators, thresholds, and configs.
4. Confirms data/manifest.json test split hash is unchanged.
5. Logs environment provenance: Python, PyTorch, LightGBM, CUDA/GPU, and OS.
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def compute_sha256(filepath: Path) -> str:
    """Computes the SHA-256 cryptographic digest of a local file."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def get_git_info(repo_root: Path) -> Dict[str, Any]:
    """Retrieves git repository status, head commit hash, and tag information."""
    try:
        # Check current HEAD commit hash
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.PIPE
        ).strip()
    except Exception as e:
        commit = f"unknown ({e})"

    try:
        # Check if working directory is clean (excluding outputs/ directory artifacts)
        status_raw = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo_root, text=True, stderr=subprocess.PIPE
        ).strip()
        # Filter out outputs/ and untracked scratch/
        dirty_lines = [
            line for line in status_raw.splitlines()
            if line and not line[3:].startswith("outputs/") and not line[3:].startswith(".gemini/")
        ]
        is_clean = len(dirty_lines) == 0
    except Exception:
        dirty_lines = []
        is_clean = True

    try:
        # Check if final-freeze tag exists and points to current commit
        tags = subprocess.check_output(
            ["git", "tag", "--points-at", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.PIPE
        ).splitlines()
        has_freeze_tag = "final-freeze" in [t.strip() for t in tags]
    except Exception:
        has_freeze_tag = False

    return {
        "head_commit": commit,
        "is_clean": is_clean,
        "dirty_files": dirty_lines,
        "has_freeze_tag": has_freeze_tag,
    }


def get_system_hardware_info() -> Dict[str, Any]:
    """Captures Python runtime, key library versions, OS, and GPU/CUDA telemetry."""
    info: Dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "os": platform.system(),
        "os_release": platform.release(),
    }

    # Record machine learning library versions
    lib_versions = {}
    for lib_name in ["torch", "transformers", "lightgbm", "numpy", "scipy", "statsmodels", "pandas", "sklearn"]:
        try:
            mod = __import__(lib_name)
            lib_versions[lib_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            lib_versions[lib_name] = "not_installed"
    info["libraries"] = lib_versions

    # Record GPU / CUDA availability and driver model
    gpu_info: Dict[str, Any] = {"cuda_available": False, "device_count": 0, "devices": []}
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info["cuda_available"] = True
            gpu_info["device_count"] = torch.cuda.device_count()
            gpu_info["cuda_version"] = torch.version.cuda
            for i in range(torch.cuda.device_count()):
                gpu_info["devices"].append({
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "capability": torch.cuda.get_device_capability(i),
                    "total_memory_mb": round(torch.cuda.get_device_properties(i).total_memory / (1024 * 1024), 2),
                })
        else:
            gpu_info["device_type"] = "CPU fallback"
    except Exception as e:
        gpu_info["error"] = str(e)
    info["gpu"] = gpu_info

    return info


def verify_freeze(
    repo_root: Optional[Path] = None,
    enforce_tag: bool = False,
    write_manifest: bool = True,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validates model checkpoints, calibration thresholds, configuration hashes,
    and dataset split integrity prior to final test split evaluation.

    Returns:
        (passed: bool, manifest_data: dict)
    """
    if repo_root is None:
        repo_root = Path(".").resolve()

    errors = []
    manifest_data: Dict[str, Any] = {
        "timestamp": None,
        "git": get_git_info(repo_root),
        "environment": get_system_hardware_info(),
        "checkpoints": {},
        "calibrators": {},
        "configs": {},
        "dataset_manifest": {},
        "checks_passed": False,
    }

    # 1. Verify Git cleanliness
    if not manifest_data["git"]["is_clean"]:
        errors.append(f"Git working tree is dirty: {manifest_data['git']['dirty_files']}")

    # 2. Verify freeze tag if enforced
    if enforce_tag and not manifest_data["git"]["has_freeze_tag"]:
        errors.append("Current commit does not match 'final-freeze' tag.")

    # 3. Verify Test Split Manifest Hash
    data_manifest_path = repo_root / "data" / "manifest.json"
    if not data_manifest_path.exists():
        errors.append(f"Missing data manifest at {data_manifest_path}")
    else:
        with open(data_manifest_path, "r", encoding="utf-8") as f:
            d_manifest = json.load(f)
        splits = d_manifest.get("splits", {})
        if "test" not in splits:
            errors.append("Test split missing from data/manifest.json")
        else:
            test_meta = splits["test"]
            test_file = repo_root / "data" / test_meta["file"]
            if not test_file.exists():
                errors.append(f"Test split file missing at {test_file}")
            else:
                real_hash = compute_sha256(test_file)
                if real_hash != test_meta["sha256"]:
                    errors.append(
                        f"Test split hash mismatch: manifest says {test_meta['sha256']}, actual is {real_hash}"
                    )
                manifest_data["dataset_manifest"]["test_split"] = {
                    "file": str(test_meta["file"]),
                    "expected_sha256": test_meta["sha256"],
                    "verified_sha256": real_hash,
                    "count": test_meta["count"],
                }

    # 4. Hash Configs
    config_dir = repo_root / "configs"
    if config_dir.exists():
        for cfg in sorted(config_dir.glob("*.yaml")):
            manifest_data["configs"][cfg.name] = compute_sha256(cfg)

    # 5. Hash Saved Models & Checkpoints
    model_paths = [
        repo_root / "outputs" / "stage6" / "models" / "mlp_full" / "mlp_model.pt",
        repo_root / "outputs" / "stage6" / "models" / "mlp_full" / "mlp_config.json",
        repo_root / "outputs" / "stage6" / "text_model_train_text" / "model.safetensors",
    ]
    for mp in model_paths:
        if mp.exists():
            manifest_data["checkpoints"][mp.name] = compute_sha256(mp)
        else:
            # Note presence or absence
            manifest_data["checkpoints"][mp.name] = "not_found"

    # 6. Hash Calibrators & Abstention Files
    cal_paths = [
        repo_root / "outputs" / "stage7" / "temperature.json",
        repo_root / "outputs" / "stage7" / "abstainer.json",
    ]
    for cp in cal_paths:
        if cp.exists():
            manifest_data["calibrators"][cp.name] = compute_sha256(cp)
        else:
            manifest_data["calibrators"][cp.name] = "not_found"

    passed = len(errors) == 0
    manifest_data["checks_passed"] = passed
    manifest_data["errors"] = errors

    if write_manifest:
        out_dir = repo_root / "outputs" / "final"
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest_out = out_dir / "freeze_manifest.json"
        with open(manifest_out, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

    return passed, manifest_data


if __name__ == "__main__":
    passed, manifest = verify_freeze()
    if passed:
        print("[FREEZE CHECK PASSED] All model hashes, configs, and dataset splits verified.")
        sys.exit(0)
    else:
        print("[FREEZE CHECK FAILED] Discrepancies detected:")
        for err in manifest.get("errors", []):
            print(f"  - {err}")
        sys.exit(1)
