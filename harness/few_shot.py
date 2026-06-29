"""Few-shot example pool builder + selector + prompt formatter for Exp 7.

Examples are (broken_code, initial_error, fixed_code) triples extracted
from successful single-step recovery trajectories in committee runs:

    iter-1 broken (any failure category)
    iter-2 passed

We filter to broken_lines <= 100 to keep K=5 prompts compact.

LOO contamination prevention: when selecting examples for test task X,
we exclude any pair where pair.task_id == X.

Strategy:
  - "random"                  uniform over candidates
  - "matched-error-category"  pairs with same broken_start error_category
                              (fall back to random if not enough)
  - "curated"                 hand-picked map test_task_id -> [example_task_id ...]
                              read from a JSON file. Skips entries not in pool;
                              never falls back. Returns <K if pool is missing some.
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Optional

logger = logging.getLogger("cuda_debugger_exp")

DEFAULT_MAX_BROKEN_LINES = 300
DEFAULT_MAX_RECOVERY_ITER = 5


def _code_lines(code: str) -> int:
    return sum(1 for ln in code.splitlines() if ln.strip())


def build_pair_pool(
    committee_dirs,
    max_broken_lines: int = DEFAULT_MAX_BROKEN_LINES,
    max_recovery_iter: int = DEFAULT_MAX_RECOVERY_ITER,
):
    """Scan committee model dirs and extract recovery pairs.

    For each task in each model_dir's manifest, find the FIRST iter that
    failed and the FIRST subsequent iter that passed within the first
    `max_recovery_iter` iterations. broken_code = that fail, fixed_code =
    that pass. (Was previously strict iter-1→iter-2 only — too restrictive,
    only ~11 tasks survived across 3 dirs.)

    committee_dirs: list of model-dir Paths. Each must contain
    index/manifest.jsonl.

    Returns list of dicts:
      {
        "task_id": "CUDA/55",
        "source_model": "gemma4",
        "broken_code": "...",
        "initial_error": "...",
        "fixed_code": "...",
        "error_category": "compile_error",
        "broken_lines": 78,
        "fail_iter": 1,
        "pass_iter": 3,
      }
    """
    pairs = []
    for model_dir in committee_dirs:
        model_dir = Path(model_dir)
        if not (model_dir / "index/manifest.jsonl").exists():
            continue
        iters_by_task: dict = {}
        with open(model_dir / "index/manifest.jsonl") as f:
            for line in f:
                r = json.loads(line)
                iters_by_task.setdefault(r["task_id"], []).append(r)
        for tid, recs in iters_by_task.items():
            recs.sort(key=lambda r: r.get("iteration", 0))
            recs = [r for r in recs if r.get("iteration", 0) <= max_recovery_iter]
            # First failed iter, then first subsequent passed iter
            fail_idx = next((i for i, r in enumerate(recs) if r.get("status") == "failed"), None)
            if fail_idx is None:
                continue
            pass_idx = next(
                (i for i, r in enumerate(recs) if i > fail_idx and r.get("status") == "passed"),
                None,
            )
            if pass_idx is None:
                continue
            r_fail, r_pass = recs[fail_idx], recs[pass_idx]
            try:
                broken_rel = r_fail["paths"].get("error_code")
                fixed_rel = r_pass["paths"].get("error_code")
                if not broken_rel or not fixed_rel:
                    continue
                broken_path = model_dir / broken_rel
                fixed_path = model_dir / fixed_rel
                broken_code = broken_path.read_text(encoding="utf-8")
                fixed_code = fixed_path.read_text(encoding="utf-8")
            except (KeyError, OSError):
                continue
            if _code_lines(broken_code) > max_broken_lines:
                continue
            log_path = model_dir / r_fail["paths"].get("build_log_raw", "")
            initial_error = ""
            if log_path.exists():
                initial_error = log_path.read_text(encoding="utf-8")[:1000]
            pairs.append({
                "task_id": tid,
                "source_model": model_dir.parent.name,
                "broken_code": broken_code,
                "initial_error": initial_error,
                "fixed_code": fixed_code,
                "error_category": r_fail.get("five_category", "unknown"),
                "broken_lines": _code_lines(broken_code),
                "fail_iter": r_fail.get("iteration"),
                "pass_iter": r_pass.get("iteration"),
            })
    return pairs


def select_examples(
    pool: list,
    test_task_id: str,
    test_error_category: str,
    strategy: str,
    K: int,
    seed: int = 42,
    curated_file: str | None = None,
):
    """Pick K examples from pool given strategy. LOO excludes test_task_id."""
    if K == 0:
        return []
    candidates = [p for p in pool if p["task_id"] != test_task_id]
    rng = random.Random(seed + hash(test_task_id) % 10_000)

    if strategy == "curated":
        if not curated_file:
            raise ValueError("curated strategy requires few_shot_curated_file")
        try:
            data = json.loads(Path(curated_file).read_text())
        except (OSError, json.JSONDecodeError) as e:
            raise ValueError(f"could not load curated_file {curated_file}: {e}") from e
        mappings = data.get("mappings", data)  # tolerate either {mappings: ...} or flat
        wanted = list(mappings.get(test_task_id, []))[:K]
        if not wanted:
            logger.warning(f"curated: no entry for test task {test_task_id} — empty examples")
            return []
        pool_by_tid: dict = {}
        for p in candidates:  # candidates already excludes test_task_id (LOO)
            pool_by_tid.setdefault(p["task_id"], p)  # first-seen wins (deterministic)
        out = []
        missing = []
        for tid in wanted:
            if tid == test_task_id:  # belt-and-suspenders LOO
                continue
            if tid in pool_by_tid:
                out.append(pool_by_tid[tid])
            else:
                missing.append(tid)
        if missing:
            logger.warning(
                f"curated for {test_task_id}: {len(missing)}/{len(wanted)} example task_ids "
                f"missing from pool (filtered out by recovery/length rules): {missing}"
            )
        return out

    if strategy == "random":
        if len(candidates) < K:
            logger.warning(
                f"select_examples: pool has only {len(candidates)} candidates, "
                f"K={K}; returning all"
            )
            return list(candidates)
        return rng.sample(candidates, K)

    if strategy == "matched-error-category":
        matched = [p for p in candidates if p["error_category"] == test_error_category]
        if len(matched) >= K:
            return rng.sample(matched, K)
        # Fallback: take all matched, pad with random from rest
        rest = [p for p in candidates if p["error_category"] != test_error_category]
        need = K - len(matched)
        pad = rng.sample(rest, min(need, len(rest)))
        if pad:
            logger.info(
                f"matched-cat fallback for task {test_task_id}: "
                f"{len(matched)} matched + {len(pad)} random padding"
            )
        return list(matched) + pad

    raise ValueError(f"Unknown strategy: {strategy}")


def format_examples_for_system(examples: list) -> str:
    """Render the example list into a system-message section."""
    if not examples:
        return ""
    parts = [
        "",
        f"Here are {len(examples)} examples of similar bugs and their fixes:",
        "",
    ]
    for i, ex in enumerate(examples, 1):
        parts.append(f"[Example {i}]")
        parts.append("Broken code:")
        parts.append("```cuda")
        parts.append(ex["broken_code"])
        parts.append("```")
        if ex.get("initial_error"):
            parts.append(f"Error: {ex['initial_error'][:600]}")
        parts.append("Fixed code:")
        parts.append("```cuda")
        parts.append(ex["fixed_code"])
        parts.append("```")
        parts.append("")
    parts.append("Now help fix the current task.")
    return "\n".join(parts)
