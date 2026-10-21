"""
Model Memory and Latency Profiling Script for TrustNet Stage 10.

Loads each core and lazy model sequentially, measuring:
- Elapsed load time (seconds)
- Dedicated GPU VRAM allocated delta and peak VRAM (MB) via torch.cuda memory stats
- Host RAM RSS delta (MB) via psutil
- Device placement (CUDA vs CPU fallback)

Prints the results as a formatted Markdown table suitable for documentation and reporting.
"""

from pathlib import Path
import sys

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from trustnet.api.registry import ModelRegistry


def measure_all_models() -> str:
    """Instantiates registry, loads core and lazy models, and outputs a Markdown summary table."""
    registry = ModelRegistry()

    print("[TrustNet] Profiling core model loading latency and memory deltas...")
    registry.load_core_models()

    print("[TrustNet] Profiling lazy model loading (translator, captioner)...")
    try:
        registry.get("translator")
    except Exception as e:
        print(f"[Warning] Translator lazy load skipped/failed: {e}")

    try:
        registry.get("captioner")
    except Exception as e:
        print(f"[Warning] Captioner lazy load skipped/failed: {e}")

    report = registry.memory_report()
    breakdown = report.get("per_model_breakdown", {})

    lines = []
    lines.append("# TrustNet Model Memory & Load Latency Profile")
    lines.append("")
    lines.append(f"- **Active Device**: `{report.get('device', 'cpu')}`")
    lines.append(f"- **CUDA Hardware Available**: `{report.get('cuda_available', False)}`")
    lines.append(f"- **Total Host Process RSS**: `{report.get('total_cpu_rss_mb', 0.0)} MB`")
    if report.get("cuda_available"):
        lines.append(f"- **Total Current GPU VRAM Allocated**: `{report.get('gpu_allocated_mb', 0.0)} MB`")
        lines.append(f"- **Total Current GPU VRAM Reserved**: `{report.get('gpu_reserved_mb', 0.0)} MB`")
        lines.append(f"- **Peak GPU VRAM Allocated**: `{report.get('gpu_peak_mb', 0.0)} MB`")
    lines.append("")
    lines.append("| Model / Component | Device | Load Time (s) | Peak GPU (MB) | GPU Delta (MB) | CPU Delta (MB) | Status |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    for model_name, stats in breakdown.items():
        dev = stats.get("device", "cpu")
        load_t = f"{stats.get('load_time_seconds', 0.0):.3f}"
        peak_gpu = f"{stats.get('gpu_peak_mb', 0.0):.1f}"
        gpu_delta = f"{stats.get('gpu_allocated_delta_mb', 0.0):.1f}"
        cpu_delta = f"{stats.get('cpu_rss_delta_mb', 0.0):.1f}"
        st = "Loaded" if stats.get("loaded", False) else "Fallback / Missing"

        lines.append(f"| `{model_name}` | {dev} | {load_t} | {peak_gpu} | {gpu_delta} | {cpu_delta} | {st} |")

    lines.append("")
    table_text = "\n".join(lines)
    print("\n" + table_text)
    return table_text


if __name__ == "__main__":
    measure_all_models()
