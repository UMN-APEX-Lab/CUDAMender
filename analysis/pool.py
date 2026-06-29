"""Per-task pass status extraction and pooled metric computation."""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def extract_per_task_status(records: list[dict],
                            valid_task_ids: Optional[set[str]] = None) -> dict[str, dict]:
    """Group manifest records by task_id and summarize per-task pass status.

    Each record needs: task_id (str), iteration (int), status (str), build_ok (bool).
    Records with error_type=="api_error" are NOT real evaluations (NVAPI 429
    raised in call_llm and was recorded as five_category=compile_error /
    status=failed by run_experiment.py:397). Drop them so the task neither
    counts as a fail nor pollutes compile/build stats. A task whose only
    records are api_error is dropped entirely (treated as not-yet-evaluated).

    valid_task_ids: optional whitelist of canonical v2 task IDs (200). When set,
    records for task IDs NOT in the whitelist are dropped. Used to exclude the
    8 trimmed flash_attn variants that early model runs (4/24-4/27, before the
    build phase trim on 4/25) still have lingering manifest records for. This
    keeps cross-model pass@k denominators consistent at 200.

    Returns dict task_id -> {
        passed: bool,                   # any iteration passed
        first_pass_iter: int | None,    # earliest iter that passed (None if never)
        n_iters: int,                   # total iter records for this task
        build_ok_iter1: bool,           # was build OK on iter 1?
    }
    """
    by_task: dict[str, list[dict]] = {}
    n_api_err = 0
    n_outside_whitelist = 0
    for r in records:
        tid = r.get("task_id")
        if tid is None:
            continue
        if r.get("error_type") == "api_error":
            n_api_err += 1
            continue
        if valid_task_ids is not None and tid not in valid_task_ids:
            n_outside_whitelist += 1
            continue
        by_task.setdefault(tid, []).append(r)
    if n_api_err:
        logger.info("dropped %d api_error records (see error_type field)", n_api_err)
    if n_outside_whitelist:
        logger.info("dropped %d records for task IDs outside v2 whitelist (build phase trim)",
                    n_outside_whitelist)

    out: dict[str, dict] = {}
    for tid, recs in by_task.items():
        recs_sorted = sorted(recs, key=lambda r: r.get("iteration", 0))
        passed = any(r.get("status") == "passed" for r in recs_sorted)
        first_pass_iter = next(
            (r["iteration"] for r in recs_sorted if r.get("status") == "passed"),
            None,
        )
        iter1 = next((r for r in recs_sorted if r.get("iteration") == 1), None)
        build_ok_iter1 = bool(iter1 and iter1.get("build_ok"))
        out[tid] = {
            "passed": passed,
            "first_pass_iter": first_pass_iter,
            "n_iters": len(recs_sorted),
            "build_ok_iter1": build_ok_iter1,
        }
    return out


def compute_pooled_metrics(
    main_status: dict[str, dict],
    memcrash_status: dict[str, dict],
    main_expected: Optional[int] = None,
    memcrash_expected: Optional[int] = None,
) -> dict:
    """Pool v2 + v3 per-task statuses and compute aggregate metrics.

    main_expected / memcrash_expected: if set and the per-source count is less, mark as partial.
    """
    # Original per-source counts (for transparency in n_main / n_memcrash reporting).
    n_main = len(main_status)
    n_memcrash = len(memcrash_status)

    # memcrash is by design a re-run of a subset of v2 task_ids
    # (e.g. CUDA/123, kernelbench/level1/26). On collision, prefer the v3
    # record (newer dedicated mem_crash run). n_tasks is the union size.
    overlap = set(main_status.keys()) & set(memcrash_status.keys())
    if overlap:
        logger.info("v2/v3 task_id collision (v3 wins): %d overlapping tasks", len(overlap))
    pooled = {**main_status, **memcrash_status}  # v3 wins on key collision
    n_tasks = len(pooled)

    # Coverage label (computed from per-source counts, not pooled count).
    if n_tasks == 0:
        coverage = "empty"
    elif n_main > 0 and n_memcrash > 0:
        coverage = "main+memcrash"
    elif n_main > 0:
        coverage = "v2_only"
        if main_expected and n_main < main_expected:
            coverage = f"v2_partial({n_main}/{main_expected})"
    else:  # n_memcrash > 0
        coverage = "v3_only"
        if memcrash_expected and n_memcrash < memcrash_expected:
            coverage = f"v3_partial({n_memcrash}/{memcrash_expected})"

    if not pooled:
        return {
            "n_tasks": 0, "n_main": 0, "n_memcrash": 0, "coverage": coverage,
            "n_overlap": len(overlap),
            "n_passed": 0,
            "pass_at_1": None, "pass_at_k": None, "debug_rate_at_k": None,
            "compilation_rate_at_1": None, "avg_iters_to_pass": None,
        }

    n_passed = sum(1 for s in pooled.values() if s["passed"])
    n_pass_at_1 = sum(1 for s in pooled.values() if s["first_pass_iter"] == 1)
    n_build_ok_iter1 = sum(1 for s in pooled.values() if s["build_ok_iter1"])
    pass_iters = [s["first_pass_iter"] for s in pooled.values() if s["passed"]]

    return {
        "n_tasks": n_tasks,
        "n_main": n_main,
        "n_memcrash": n_memcrash,
        "coverage": coverage,
        "n_overlap": len(overlap),
        "n_passed": n_passed,
        "pass_at_1": n_pass_at_1 / n_tasks,
        "pass_at_k": n_passed / n_tasks,
        "debug_rate_at_k": (n_passed - n_pass_at_1) / n_tasks,
        "compilation_rate_at_1": n_build_ok_iter1 / n_tasks,
        "avg_iters_to_pass": sum(pass_iters) / len(pass_iters) if pass_iters else None,
    }
