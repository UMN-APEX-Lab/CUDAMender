"""
Multi-agent analysis pipeline for post-run evaluation.

Inspired by agent-main's multi-agent architecture. Three parallel analysis
"agents" (Python functions via ThreadPoolExecutor):

  Agent 1: Failure Classifier — re-classifies build logs, detects misclassifications
  Agent 2: Quality Auditor    — checks stagnation, oscillation, progression patterns
  Agent 3: Baseline Comparator — cross-references with other model results

Usage:
    python analyze_pipeline.py results/20260402_150000
    python analyze_pipeline.py results/20260402_150000 --baseline results/20260401_120000
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .metrics import (
    CAT3_MAP,
    TaskMetrics,
    compute_task_metrics,
    compute_aggregate_metrics,
    metrics_to_dict,
)

logger = logging.getLogger("cuda_debugger_exp")


# ---------------------------------------------------------------------------
# Agent 1: Failure Classifier
# ---------------------------------------------------------------------------


def classify_failures(result_dir: Path) -> dict:
    """
    Re-analyze failure patterns from build logs.

    Returns:
        {
            "category_distribution": {"compile_stage_error": N, ...},
            "misclassification_candidates": [...],
            "environment_issues": [...],
            "common_error_patterns": [{"pattern": str, "count": int}, ...],
        }
    """
    records = _load_records(result_dir)
    if not records:
        return {"category_distribution": {}, "misclassification_candidates": [],
                "environment_issues": [], "common_error_patterns": []}

    # Category distribution (8-cat and 3-cat)
    cat8_dist = Counter()
    cat3_dist = Counter()
    for r in records:
        cat = r.get("category", r.get("five_category", "unknown"))
        cat8_dist[cat] += 1
        cat3_dist[CAT3_MAP.get(cat, cat)] += 1

    # Detect misclassification candidates by reading build logs
    misclass = []
    env_issues = []
    log_dir = result_dir / "build_log_raw"

    error_patterns = Counter()

    for r in records:
        if r["status"] == "passed":
            continue
        sample_id = r["sample_id"]
        log_file = log_dir / f"{sample_id}.txt"
        if not log_file.exists():
            continue

        log_text = log_file.read_text(encoding="utf-8", errors="replace")
        cat = r.get("category", r.get("five_category", "unknown"))

        # Check for environment issues misclassified as other errors
        env_markers = [
            "no kernel image", "cudaErrorNoKernelImageForDevice",
            "Failed to query occupancy", "cutlass", "cudnn",
        ]
        for marker in env_markers:
            if marker.lower() in log_text.lower() and cat not in (
                "environment_dependency_bug", "compile_error"
            ):
                misclass.append({
                    "sample_id": sample_id,
                    "current_category": cat,
                    "suggested_category": "environment_dependency_bug",
                    "evidence": marker,
                })
                env_issues.append({"sample_id": sample_id, "marker": marker})
                break

        # Extract common error patterns from failed build logs
        if "error:" in log_text.lower():
            for line in log_text.splitlines():
                if "error:" in line.lower():
                    # Normalize: remove file paths and line numbers
                    import re
                    normalized = re.sub(r"[^\s:]+\.\w+:\d+:\d+:", "FILE:LINE:", line.strip())
                    normalized = re.sub(r"\s+", " ", normalized)[:120]
                    error_patterns[normalized] += 1

    common_patterns = [
        {"pattern": p, "count": c}
        for p, c in error_patterns.most_common(20)
    ]

    return {
        "category_distribution_8cat": dict(cat8_dist.most_common()),
        "category_distribution_3cat": dict(cat3_dist.most_common()),
        "misclassification_candidates": misclass,
        "environment_issues": env_issues,
        "common_error_patterns": common_patterns,
    }


# ---------------------------------------------------------------------------
# Agent 2: Quality Auditor
# ---------------------------------------------------------------------------


def audit_quality(result_dir: Path) -> dict:
    """
    Audit debugging quality: stagnation, oscillation, progression patterns.

    Returns:
        {
            "task_quality": {task_id: {"stagnation": bool, "oscillation": bool, ...}},
            "aggregate": {"stagnation_rate": float, ...},
            "worst_tasks": [...],  # tasks with most wasted iterations
            "best_debugged": [...],  # tasks showing genuine progression
        }
    """
    records = _load_records(result_dir)
    if not records:
        return {"task_quality": {}, "aggregate": {}, "worst_tasks": [], "best_debugged": []}

    # Group by task
    by_task = defaultdict(list)
    for r in records:
        by_task[r["task_id"]].append(r)

    task_metrics_list = []
    task_quality = {}
    for task_id, task_records in by_task.items():
        task_records.sort(key=lambda r: r["iteration"])
        tm = compute_task_metrics(task_id, task_records)
        task_metrics_list.append(tm)
        task_quality[task_id] = {
            "status": tm.status,
            "iterations_used": tm.iterations_used,
            "stagnation": tm.stagnation,
            "oscillation": tm.oscillation,
            "progression": tm.progression,
            "trajectory": tm.trajectory,
            "unique_codes": len(set(h for h in tm.code_hashes if h)),
            "wasted_iterations": _count_wasted(tm),
        }

    # Find worst tasks (most wasted iterations)
    worst = sorted(
        [
            {"task_id": tid, **info}
            for tid, info in task_quality.items()
            if info["status"] == "failed"
        ],
        key=lambda x: -x["wasted_iterations"],
    )[:10]

    # Find best debugged tasks (progression from failure to pass)
    best = [
        {"task_id": tid, **info}
        for tid, info in task_quality.items()
        if info["status"] == "passed" and info["progression"] and info["iterations_used"] > 1
    ]
    best.sort(key=lambda x: x["iterations_used"])

    # Aggregate quality metrics
    failed_tasks = [t for t in task_metrics_list if t.status != "passed"]
    passed_after_1 = [t for t in task_metrics_list if t.status == "passed" and not t.pass_at_1]
    n_failed = len(failed_tasks) or 1

    aggregate = {
        "total_tasks": len(task_metrics_list),
        "stagnation_rate": sum(1 for t in failed_tasks if t.stagnation) / n_failed,
        "oscillation_rate": sum(1 for t in failed_tasks if t.oscillation) / n_failed,
        "progression_rate": (
            sum(1 for t in passed_after_1 if t.progression) / len(passed_after_1)
            if passed_after_1 else 0.0
        ),
        "avg_wasted_iterations": (
            sum(_count_wasted(t) for t in failed_tasks) / n_failed
        ),
    }

    return {
        "task_quality": task_quality,
        "aggregate": aggregate,
        "worst_tasks": worst,
        "best_debugged": best[:10],
    }


def _count_wasted(tm: TaskMetrics) -> int:
    """Count iterations that didn't contribute to progress."""
    if tm.status == "passed":
        return 0
    # All iterations on a failed task are "wasted" but some show progression
    # Count iterations where category didn't improve
    if len(tm.trajectory) < 2:
        return 0
    wasted = 0
    for i in range(1, len(tm.trajectory)):
        if tm.trajectory[i] == tm.trajectory[i - 1]:
            wasted += 1
    return wasted


# ---------------------------------------------------------------------------
# Agent 3: Baseline Comparator
# ---------------------------------------------------------------------------


def compare_baselines(result_dir: Path, baseline_dirs: list[Path] | None = None) -> dict:
    """
    Compare this run against baseline results.

    Returns:
        {
            "this_run": {"pass_at_1": float, "pass_at_k": float, ...},
            "comparisons": [{"name": str, "pass_at_1": float, ...}],
            "difficulty_tiers": {"hard": [...], "medium": [...], "easy": [...]},
            "model_specific_strengths": [...],
            "model_specific_weaknesses": [...],
        }
    """
    records = _load_records(result_dir)
    if not records:
        return {"this_run": {}, "comparisons": [], "difficulty_tiers": {},
                "model_specific_strengths": [], "model_specific_weaknesses": []}

    # Compute metrics for this run
    by_task = defaultdict(list)
    for r in records:
        by_task[r["task_id"]].append(r)

    task_metrics = []
    task_pass = {}  # task_id -> bool
    for task_id, task_records in by_task.items():
        task_records.sort(key=lambda r: r["iteration"])
        tm = compute_task_metrics(task_id, task_records)
        task_metrics.append(tm)
        task_pass[task_id] = tm.status == "passed"

    agg = compute_aggregate_metrics(task_metrics)
    this_run = {
        "name": _run_name(result_dir),
        "model": _detect_model(result_dir),
        **metrics_to_dict(agg),
    }

    # Compare with baselines
    comparisons = []
    all_passes = {"this": task_pass}

    if baseline_dirs:
        for bdir in baseline_dirs:
            if not bdir.exists():
                continue
            b_records = _load_records(bdir)
            b_by_task = defaultdict(list)
            for r in b_records:
                b_by_task[r["task_id"]].append(r)

            b_metrics = []
            b_pass = {}
            for task_id, task_records in b_by_task.items():
                task_records.sort(key=lambda r: r["iteration"])
                tm = compute_task_metrics(task_id, task_records)
                b_metrics.append(tm)
                b_pass[task_id] = tm.status == "passed"

            b_agg = compute_aggregate_metrics(b_metrics)
            comp = {
                "name": _run_name(bdir),
                "model": _detect_model(bdir),
                **metrics_to_dict(b_agg),
            }
            comparisons.append(comp)
            all_passes[_run_name(bdir)] = b_pass

    # Difficulty tiers based on cross-model pass rates
    all_tasks = set()
    for passes in all_passes.values():
        all_tasks.update(passes.keys())

    difficulty_tiers = {"hard": [], "medium": [], "easy": []}
    for task_id in sorted(all_tasks):
        n_models = len(all_passes)
        n_pass = sum(1 for passes in all_passes.values() if passes.get(task_id, False))
        rate = n_pass / n_models if n_models else 0

        if rate == 0:
            difficulty_tiers["hard"].append(task_id)
        elif rate < 1.0:
            difficulty_tiers["medium"].append(task_id)
        else:
            difficulty_tiers["easy"].append(task_id)

    # Model-specific strengths/weaknesses (tasks only this model solves/fails)
    strengths = []
    weaknesses = []
    if comparisons:
        for task_id in all_tasks:
            this_passed = task_pass.get(task_id, False)
            others_passed = any(
                all_passes.get(name, {}).get(task_id, False)
                for name in all_passes if name != "this"
            )
            if this_passed and not others_passed:
                strengths.append(task_id)
            elif not this_passed and others_passed:
                weaknesses.append(task_id)

    return {
        "this_run": this_run,
        "comparisons": comparisons,
        "difficulty_tiers": {k: v for k, v in difficulty_tiers.items()},
        "model_specific_strengths": strengths,
        "model_specific_weaknesses": weaknesses,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def analyze_run(
    result_dir: Path,
    baseline_dirs: list[Path] | None = None,
) -> dict:
    """
    Run all 3 analysis agents in parallel and merge results.

    Args:
        result_dir: Path to experiment results
        baseline_dirs: Optional list of baseline result dirs for comparison

    Returns:
        Merged analysis report dict
    """
    result_dir = Path(result_dir)
    logger.info(f"Analyzing: {result_dir}")

    with ThreadPoolExecutor(max_workers=3) as pool:
        f1 = pool.submit(classify_failures, result_dir)
        f2 = pool.submit(audit_quality, result_dir)
        f3 = pool.submit(compare_baselines, result_dir, baseline_dirs)

    failures = f1.result()
    quality = f2.result()
    comparison = f3.result()

    report = {
        "result_dir": str(result_dir),
        "failure_analysis": failures,
        "quality_audit": quality,
        "baseline_comparison": comparison,
    }

    # Write report
    out_file = result_dir / "analysis_report.json"
    out_file.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    logger.info(f"Analysis report written to {out_file}")

    # Print summary
    _print_summary(report)

    return report


def _print_summary(report: dict):
    """Print a human-readable summary of the analysis."""
    fa = report["failure_analysis"]
    qa = report["quality_audit"]
    bc = report["baseline_comparison"]

    print("\n" + "=" * 60)
    print("ANALYSIS REPORT")
    print("=" * 60)

    # Failure analysis
    if fa.get("category_distribution_3cat"):
        print("\n--- Error Distribution (3-category) ---")
        for cat, count in fa["category_distribution_3cat"].items():
            print(f"  {cat}: {count}")

    if fa.get("misclassification_candidates"):
        print(f"\n--- Misclassification Candidates: {len(fa['misclassification_candidates'])} ---")
        for mc in fa["misclassification_candidates"][:5]:
            print(f"  {mc['sample_id']}: {mc['current_category']} → {mc['suggested_category']} ({mc['evidence']})")

    if fa.get("common_error_patterns"):
        print(f"\n--- Top Error Patterns ---")
        for ep in fa["common_error_patterns"][:5]:
            print(f"  [{ep['count']}x] {ep['pattern'][:80]}")

    # Quality audit
    agg = qa.get("aggregate", {})
    if agg:
        print(f"\n--- Quality Metrics ---")
        print(f"  Stagnation rate:  {agg.get('stagnation_rate', 0):.1%}")
        print(f"  Oscillation rate: {agg.get('oscillation_rate', 0):.1%}")
        print(f"  Progression rate: {agg.get('progression_rate', 0):.1%}")
        print(f"  Avg wasted iters: {agg.get('avg_wasted_iterations', 0):.1f}")

    if qa.get("worst_tasks"):
        print(f"\n--- Worst Tasks (most wasted iterations) ---")
        for t in qa["worst_tasks"][:5]:
            print(f"  {t['task_id']}: {t['wasted_iterations']} wasted, trajectory={t['trajectory']}")

    # Baseline comparison
    if bc.get("this_run"):
        this = bc["this_run"]
        core = this.get("core_metrics", {})
        print(f"\n--- This Run: {this.get('name', '?')} ({this.get('model', '?')}) ---")
        print(f"  pass@1: {core.get('pass_at_1', 0):.1%}")
        print(f"  pass@k: {core.get('pass_at_k', 0):.1%}")

    diff = bc.get("difficulty_tiers", {})
    if diff:
        print(f"\n--- Difficulty Tiers ---")
        print(f"  Hard (all fail): {len(diff.get('hard', []))} tasks")
        print(f"  Medium (some pass): {len(diff.get('medium', []))} tasks")
        print(f"  Easy (all pass): {len(diff.get('easy', []))} tasks")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_records(result_dir: Path) -> list[dict]:
    """Load manifest records from result directory."""
    manifest = result_dir / "index" / "manifest.jsonl"
    if not manifest.exists():
        logger.warning(f"No manifest found at {manifest}")
        return []
    records = []
    with open(manifest, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _run_name(result_dir: Path) -> str:
    """Extract a human-readable name from the result directory."""
    return result_dir.name


def _detect_model(result_dir: Path) -> str:
    """Try to detect model name from config.yaml in result dir."""
    config_file = result_dir / "config.yaml"
    if config_file.exists():
        import yaml
        try:
            cfg = yaml.safe_load(config_file.read_text())
            return cfg.get("api", {}).get("model", "unknown")
        except Exception:
            pass
    # Try summary.json
    summary_file = result_dir / "summary.json"
    if summary_file.exists():
        try:
            s = json.loads(summary_file.read_text())
            return s.get("model", "unknown")
        except Exception:
            pass
    return "unknown"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze CUDA-Debugger experiment results")
    parser.add_argument("result_dir", type=Path, help="Path to experiment result directory")
    parser.add_argument(
        "--baseline", "-b",
        type=Path,
        nargs="*",
        help="Baseline result directories for comparison",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    analyze_run(args.result_dir, args.baseline)


if __name__ == "__main__":
    main()
