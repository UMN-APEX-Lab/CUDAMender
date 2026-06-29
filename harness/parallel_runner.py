"""
Parallel task runner for CUDA-Debugger evaluation.

Distributes tasks across multiple GPU workers using ProcessPoolExecutor.
Each worker gets exclusive GPU access via CUDA_VISIBLE_DEVICES.

Architecture:
    Main Process
      ├── Pre-flight: filter tasks, check GPUs
      ├── Task Queue: [task_1, task_2, ..., task_N]
      ├── Worker Pool (n_workers)
      │   ├── Worker 0 (GPU 0): task_1 → task_4 → ...
      │   ├── Worker 1 (GPU 1): task_2 → task_5 → ...
      │   └── Worker 2 (GPU 2): task_3 → task_6 → ...
      ├── Result Collector: aggregate manifests
      └── Exporter: write summary.json

Constraints:
  - Each worker needs exclusive GPU for nvcc + test execution
  - API rate limits shared across workers (handled by API provider)
  - Separate workdir per worker to avoid disk contention
  - Thread-safe progress reporting
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

logger = logging.getLogger("cuda_debugger_exp")


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------


def detect_available_gpus() -> list[int]:
    """
    Detect available NVIDIA GPUs and return their IDs.

    Respects CUDA_VISIBLE_DEVICES if set.
    Returns list of GPU IDs (e.g., [0, 2, 5]).
    """
    # If CUDA_VISIBLE_DEVICES is set, use those
    env_gpus = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env_gpus:
        try:
            return [int(g.strip()) for g in env_gpus.split(",") if g.strip()]
        except ValueError:
            pass

    # Query nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return [int(line.strip()) for line in result.stdout.strip().splitlines() if line.strip()]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    logger.warning("Could not detect GPUs, assuming GPU 0")
    return [0]


def get_gpu_info() -> list[dict]:
    """Get detailed GPU info for logging."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,compute_cap",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return []
        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory": parts[2],
                    "compute_cap": parts[3],
                })
        return gpus
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Worker function (runs in subprocess)
# ---------------------------------------------------------------------------


def _worker_run_task(
    task_stem: str,
    dataset_dir: str,
    output_dir: str,
    config: dict,
    gpu_id: int,
    worker_workdir: str,
) -> tuple[str, list[dict], str]:
    """
    Run a single task in a worker process with exclusive GPU.

    This function runs in a subprocess via ProcessPoolExecutor.
    Returns (task_stem, records, stop_reason).
    """
    # Set GPU for this worker
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Override workdir for this worker
    worker_config = dict(config)
    worker_config["build"] = {**config.get("build", {}), "workdir": worker_workdir}

    # Import here to avoid pickling issues
    import sys
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    from .llm_client import create_client
    from .run_experiment import run_task

    client = create_client(worker_config)

    try:
        records, stop_reason = run_task(
            task_stem,
            Path(dataset_dir),
            Path(output_dir),
            worker_config,
            client,
        )
        return task_stem, records, stop_reason
    except Exception as e:
        import traceback
        logger.error(f"Worker error on {task_stem}: {traceback.format_exc()}")
        return task_stem, [], f"error: {e}"


# ---------------------------------------------------------------------------
# Progress tracker (thread-safe)
# ---------------------------------------------------------------------------


class ProgressTracker:
    """Thread-safe progress tracker for parallel execution."""

    def __init__(self, total: int):
        self.total = total
        self.completed = 0
        self.passed = 0
        self.failed = 0
        self.errors = 0
        self._lock = threading.Lock()
        self._start_time = time.time()

    def update(self, task_stem: str, records: list[dict], stop_reason: str):
        with self._lock:
            self.completed += 1
            if records and any(r["status"] == "passed" for r in records):
                self.passed += 1
            elif records:
                self.failed += 1
            else:
                self.errors += 1

            elapsed = time.time() - self._start_time
            rate = self.completed / elapsed if elapsed > 0 else 0
            remaining = (self.total - self.completed) / rate if rate > 0 else 0

            logger.info(
                f"  Progress: {self.completed}/{self.total} "
                f"({self.passed} passed, {self.failed} failed, {self.errors} errors) "
                f"[{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining] "
                f"Last: {task_stem} → {stop_reason}"
            )

    def summary(self) -> dict:
        with self._lock:
            return {
                "total": self.total,
                "completed": self.completed,
                "passed": self.passed,
                "failed": self.failed,
                "errors": self.errors,
                "elapsed_seconds": time.time() - self._start_time,
            }


# ---------------------------------------------------------------------------
# Parallel runner
# ---------------------------------------------------------------------------


def run_parallel(
    task_stems: list[str],
    dataset_dir: Path,
    output_dir: Path,
    config: dict,
    n_workers: int = 2,
    gpu_ids: list[int] | None = None,
) -> list[dict]:
    """
    Run tasks in parallel across multiple GPU workers.

    Args:
        task_stems: List of task stems to run
        dataset_dir: Path to dataset directory
        output_dir: Path to output directory
        config: Experiment config dict
        n_workers: Number of parallel workers
        gpu_ids: Specific GPU IDs to use (auto-detected if None)

    Returns:
        List of all manifest records
    """
    if gpu_ids is None:
        gpu_ids = detect_available_gpus()

    n_workers = min(n_workers, len(gpu_ids), len(task_stems))
    # Each worker slot needs an isolated workdir even when multiple slots share a GPU.
    # Use slot index, not gpu_id, so duplicate gpu entries (e.g., --gpus 6,7,6,7) are safe.
    if n_workers <= 0:
        logger.error("No workers available")
        return []

    logger.info(f"Parallel execution: {len(task_stems)} tasks, {n_workers} workers, GPUs: {gpu_ids[:n_workers]}")

    # Log GPU info
    for info in get_gpu_info():
        if info["index"] in gpu_ids[:n_workers]:
            logger.info(f"  GPU {info['index']}: {info['name']} ({info['memory']}, compute {info['compute_cap']})")

    tracker = ProgressTracker(len(task_stems))
    all_records = []
    base_workdir = config.get("build", {}).get("workdir", "/tmp/cuda_debugger_workspace")

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {}
        for i, task_stem in enumerate(task_stems):
            slot = i % n_workers
            gpu_id = gpu_ids[slot]
            worker_workdir = f"{base_workdir}_worker_slot{slot}"

            future = pool.submit(
                _worker_run_task,
                task_stem,
                str(dataset_dir),
                str(output_dir),
                config,
                gpu_id,
                worker_workdir,
            )
            futures[future] = task_stem

        for future in as_completed(futures):
            task_stem = futures[future]
            try:
                stem, records, stop_reason = future.result()
                all_records.extend(records)
                tracker.update(stem, records, stop_reason)
            except Exception as e:
                logger.error(f"Task {task_stem} raised exception: {e}")
                tracker.update(task_stem, [], f"exception: {e}")

    # Log final summary
    summary = tracker.summary()
    logger.info(
        f"\nParallel execution complete: "
        f"{summary['passed']}/{summary['total']} passed, "
        f"{summary['failed']} failed, {summary['errors']} errors "
        f"in {summary['elapsed_seconds']:.0f}s"
    )

    return all_records


def run_parallel_two_phase(
    task_stems: list[str],
    dataset_dir: Path,
    output_dir: Path,
    config: dict,
    n_workers: int = 2,
    gpu_ids: list[int] | None = None,
) -> list[dict]:
    """
    Two-phase parallel execution:
    Phase 1: pass@1 scan (all tasks, 1 iteration, parallel)
    Phase 2: debug@k (failed tasks only, parallel)
    """
    max_iters = config["experiment"]["max_iterations"]

    # Phase 1: pass@1 scan
    logger.info(f"\n{'='*60}")
    logger.info(f"PHASE 1: pass@1 scan ({len(task_stems)} tasks, {n_workers} workers)")
    logger.info(f"{'='*60}")

    phase1_config = dict(config)
    phase1_config["experiment"] = {**config["experiment"], "max_iterations": 1}

    phase1_records = run_parallel(
        task_stems, dataset_dir, output_dir, phase1_config, n_workers, gpu_ids
    )

    # Find failed tasks
    task_results = {}
    for r in phase1_records:
        task_stem = r["sample_id"].rsplit("__", 1)[0]
        if r["status"] == "passed":
            task_results[task_stem] = "passed"
        elif task_stem not in task_results:
            task_results[task_stem] = "failed"

    failed_stems = [s for s in task_stems if task_results.get(s) != "passed"]
    pass1 = len(task_stems) - len(failed_stems)
    logger.info(f"\nPhase 1 complete: {pass1}/{len(task_stems)} passed at iter 1")

    all_records = list(phase1_records)

    # Phase 2: debug@k on failures
    if failed_stems and max_iters > 1:
        logger.info(f"\n{'='*60}")
        logger.info(f"PHASE 2: debug@{max_iters} ({len(failed_stems)} failed tasks, {n_workers} workers)")
        logger.info(f"{'='*60}")

        phase2_records = run_parallel(
            failed_stems, dataset_dir, output_dir, config, n_workers, gpu_ids
        )
        all_records.extend(phase2_records)

    return all_records
