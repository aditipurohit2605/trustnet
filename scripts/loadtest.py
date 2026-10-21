"""
Concurrent Load Testing Script for TrustNet Verification API.

Submits asynchronous verification jobs at concurrency levels [1, 2, 5, 10] using httpx.
Collects and computes:
- p50 and p95 end-to-end latency (seconds)
- Throughput (completed jobs per second)
- Average queue wait time (seconds)
- Rate of HTTP 429 backpressure rejections
- Rate of failed jobs
- Peak GPU memory consumption from /v1/memory

Outputs findings as a comparative Markdown summary table.
"""

import argparse
import asyncio
from dataclasses import dataclass
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np

# Sample test claims representing realistic user forwarded assertions
SAMPLE_CLAIMS = [
    "The northern municipal dam reservoir water levels are within historical normal bounds and safe.",
    "World Health Organization has not banned Covaxin nationwide.",
    "Controlled clinical studies showed no cognitive enhancement from synthetic mega-dose vitamins.",
    "Weather department forecasts moderate rain across coastal districts tomorrow morning.",
    "Offshore floating solar farm was inaugurated to power municipal water treatment plants.",
]


@dataclass
class JobTestResult:
    submitted: bool
    status_code: int
    job_id: Optional[str]
    is_429: bool
    is_failed: bool
    is_done: bool
    queue_wait_time: float
    end_to_end_latency: float


async def run_single_job(
    client: httpx.AsyncClient,
    base_url: str,
    text: str,
    api_key: Optional[str],
    poll_interval: float = 0.5,
    max_poll_time: float = 60.0,
) -> JobTestResult:
    """Submits a single verification job and polls /v1/jobs/{job_id} until completion."""
    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key

    start_submit = time.perf_counter()
    try:
        resp = await client.post(
            f"{base_url}/v1/verify",
            data={"text": text},
            headers=headers,
            timeout=10.0,
        )
    except Exception as e:
        return JobTestResult(
            submitted=False,
            status_code=0,
            job_id=None,
            is_429=False,
            is_failed=True,
            is_done=False,
            queue_wait_time=0.0,
            end_to_end_latency=time.perf_counter() - start_submit,
        )

    if resp.status_code == 429:
        return JobTestResult(
            submitted=False,
            status_code=429,
            job_id=None,
            is_429=True,
            is_failed=False,
            is_done=False,
            queue_wait_time=0.0,
            end_to_end_latency=time.perf_counter() - start_submit,
        )

    if resp.status_code != 202:
        return JobTestResult(
            submitted=False,
            status_code=resp.status_code,
            job_id=None,
            is_429=False,
            is_failed=True,
            is_done=False,
            queue_wait_time=0.0,
            end_to_end_latency=time.perf_counter() - start_submit,
        )

    data = resp.json()
    job_id = data.get("job_id")
    submit_done = time.perf_counter()

    first_running_time: Optional[float] = None
    poll_start = time.perf_counter()

    while (time.perf_counter() - poll_start) < max_poll_time:
        await asyncio.sleep(poll_interval)
        try:
            status_resp = await client.get(
                f"{base_url}/v1/jobs/{job_id}",
                headers=headers,
                timeout=5.0,
            )
            if status_resp.status_code != 200:
                continue

            status_data = status_resp.json()
            curr_status = status_data.get("status")

            if curr_status == "running" and first_running_time is None:
                first_running_time = time.perf_counter()

            if curr_status == "done":
                total_duration = time.perf_counter() - start_submit
                wait_t = (first_running_time - submit_done) if first_running_time else 0.0
                return JobTestResult(
                    submitted=True,
                    status_code=202,
                    job_id=job_id,
                    is_429=False,
                    is_failed=False,
                    is_done=True,
                    queue_wait_time=max(0.0, wait_t),
                    end_to_end_latency=total_duration,
                )
            elif curr_status == "failed":
                total_duration = time.perf_counter() - start_submit
                return JobTestResult(
                    submitted=True,
                    status_code=202,
                    job_id=job_id,
                    is_429=False,
                    is_failed=True,
                    is_done=False,
                    queue_wait_time=0.0,
                    end_to_end_latency=total_duration,
                )
        except Exception:
            pass

    # Polling timed out
    return JobTestResult(
        submitted=True,
        status_code=202,
        job_id=job_id,
        is_429=False,
        is_failed=True,
        is_done=False,
        queue_wait_time=0.0,
        end_to_end_latency=time.perf_counter() - start_submit,
    )


async def benchmark_concurrency(
    base_url: str,
    concurrency: int,
    total_jobs: int = 10,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Runs a concurrency level benchmark against the API and reports performance metrics."""
    semaphore = asyncio.Semaphore(concurrency)
    results: List[JobTestResult] = []

    async with httpx.AsyncClient() as client:
        # Check initial memory
        peak_gpu = 0.0
        try:
            mem_resp = await client.get(
                f"{base_url}/v1/memory",
                headers={"X-API-Key": api_key} if api_key else {},
                timeout=5.0,
            )
            if mem_resp.status_code == 200:
                peak_gpu = mem_resp.json().get("gpu_peak_mb", 0.0)
        except Exception:
            pass

        async def worker(idx: int):
            text = SAMPLE_CLAIMS[idx % len(SAMPLE_CLAIMS)]
            async with semaphore:
                res = await run_single_job(
                    client=client,
                    base_url=base_url,
                    text=text,
                    api_key=api_key,
                )
                results.append(res)

        bench_start = time.perf_counter()
        tasks = [asyncio.create_task(worker(i)) for i in range(total_jobs)]
        await asyncio.gather(*tasks)
        total_time = time.perf_counter() - bench_start

        # Check post-run peak memory
        try:
            mem_resp = await client.get(
                f"{base_url}/v1/memory",
                headers={"X-API-Key": api_key} if api_key else {},
                timeout=5.0,
            )
            if mem_resp.status_code == 200:
                peak_gpu = max(peak_gpu, mem_resp.json().get("gpu_peak_mb", 0.0))
        except Exception:
            pass

    # Metrics calculation
    num_429 = sum(1 for r in results if r.is_429)
    num_failed = sum(1 for r in results if r.is_failed)
    done_results = [r for r in results if r.is_done]
    num_done = len(done_results)

    e2e_latencies = [r.end_to_end_latency for r in done_results]
    wait_times = [r.queue_wait_time for r in done_results]

    p50_e2e = float(np.percentile(e2e_latencies, 50)) if e2e_latencies else 0.0
    p95_e2e = float(np.percentile(e2e_latencies, 95)) if e2e_latencies else 0.0
    avg_wait = float(np.mean(wait_times)) if wait_times else 0.0
    throughput = round(num_done / total_time, 2) if total_time > 0 else 0.0

    return {
        "concurrency": concurrency,
        "total_submitted": total_jobs,
        "completed": num_done,
        "rate_429_pct": round((num_429 / total_jobs) * 100, 1),
        "rate_failed_pct": round((num_failed / total_jobs) * 100, 1),
        "p50_latency_s": round(p50_e2e, 2),
        "p95_latency_s": round(p95_e2e, 2),
        "avg_queue_wait_s": round(avg_wait, 2),
        "throughput_jobs_sec": throughput,
        "peak_gpu_mb": peak_gpu,
    }


async def main():
    parser = argparse.ArgumentParser(description="TrustNet API Load Tester")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000", help="Base API URL")
    parser.add_argument("--api-key", type=str, default=None, help="API Key for authentication")
    parser.add_argument("--jobs-per-level", type=int, default=5, help="Number of jobs to submit per level")
    args = parser.parse_args()

    levels = [1, 2, 5, 10]
    print(f"[TrustNet] Launching concurrent load benchmark against {args.url}...")
    all_metrics = []

    for lvl in levels:
        print(f"[LoadTest] Running Concurrency Level = {lvl} (Jobs = {args.jobs_per_level})...")
        m = await benchmark_concurrency(
            base_url=args.url,
            concurrency=lvl,
            total_jobs=args.jobs_per_level,
            api_key=args.api_key,
        )
        all_metrics.append(m)

    # Print results table
    print("\n# TrustNet API Concurrency Benchmark Report\n")
    print("| Concurrency | Jobs | Done | 429 Rate | Fail Rate | p50 Latency (s) | p95 Latency (s) | Avg Queue Wait (s) | Throughput (jobs/s) | Peak GPU (MB) |")
    print("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for m in all_metrics:
        print(
            f"| {m['concurrency']} | {m['total_submitted']} | {m['completed']} | "
            f"{m['rate_429_pct']}% | {m['rate_failed_pct']}% | {m['p50_latency_s']}s | "
            f"{m['p95_latency_s']}s | {m['avg_queue_wait_s']}s | {m['throughput_jobs_sec']} | {m['peak_gpu_mb']} |"
        )


if __name__ == "__main__":
    asyncio.run(main())
