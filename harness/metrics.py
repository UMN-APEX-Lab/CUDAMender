"""
4-Layer Metrics System for CUDA-Debugger Evaluation.

Layer 1 - Core: pass@1, pass@k, compilation_rate@1
Layer 2 - Debugging: debug_rate@k, fix_rate_by_type, avg_iterations_to_pass
Layer 3 - Quality: stagnation_rate, oscillation_rate, progression_rate
Layer 4 - Error Analysis: per_category_pass_rate, error_transition_matrix, root_cause_distribution
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field


CAT3_MAP = {
    "buildability_bug": "compile_stage_error",
    "compile_error": "compile_stage_error",
    "integration_bug": "compile_stage_error",
    "environment_dependency_bug": "compile_stage_error",
    "illegal_memory_access": "runtime_crash",
    "out_of_memory": "runtime_crash",
    "memory_crash": "runtime_crash",
    "timeout": "runtime_crash",
    "others": "runtime_crash",
    "functional_correctness_bug": "wrong_result",
    "logic_error": "wrong_result",
}


@dataclass
class TaskMetrics:
    """Metrics for a single task."""
    task_id: str
    status: str  # "passed" or "failed"
    iterations_used: int
    trajectory: list[str]  # per-iteration category
    trajectory_8cat: list[str]
    code_hashes: list[str]
    pass_at_1: bool
    first_compile_iter: int | None  # first iteration where build_ok=True
    stop_reason: str  # "passed", "max_iterations", "duplicate_code", "code_cycle", "category_oscillation", "no_progress"
    stagnation: bool
    oscillation: bool
    progression: bool


@dataclass
class AggregateMetrics:
    """Full 4-layer metrics for an experiment run."""
    # Layer 1: Core
    total_tasks: int = 0
    pass_at_1: float = 0.0
    pass_at_k: float = 0.0  # k = max_iterations
    compilation_rate_at_1: float = 0.0
    task_coverage: float = 1.0
    max_iterations: int = 5

    # Layer 2: Debugging
    debug_rate_at_k: float = 0.0
    avg_iterations_to_pass: float | None = None
    fix_rate: dict[str, float] = field(default_factory=dict)
    first_compile_pass_distribution: dict[str, float] = field(default_factory=dict)

    # Layer 3: Quality
    stagnation_rate: float = 0.0
    oscillation_rate: float = 0.0
    progression_rate: float = 0.0
    unique_approach_ratio: float = 0.0
    early_stop_reasons: dict[str, int] = field(default_factory=dict)

    # Layer 4: Error Analysis
    per_category_pass_rate: dict[str, dict] = field(default_factory=dict)
    error_transition_matrix: dict[str, dict[str, float]] = field(default_factory=dict)
    top_root_causes_failed: list[dict] = field(default_factory=list)


def compute_task_metrics(
    task_id: str,
    records: list[dict],
    stop_reason: str = "max_iterations",
) -> TaskMetrics:
    """Compute metrics for a single task from its iteration records."""
    records = sorted(records, key=lambda r: r["iteration"])

    trajectory = [r.get("category", r.get("five_category", "unknown")) for r in records]
    trajectory_8cat = [r.get("category", r.get("five_category", "unknown")) for r in records]
    code_hashes = [
        r["paths"].get("error_code", "").split("/")[-1] if r["paths"].get("error_code") else ""
        for r in records
    ]
    passed = any(r["status"] == "passed" for r in records)
    pass_at_1 = records[0]["status"] == "passed" if records else False

    # First iteration where build succeeded
    first_compile = None
    for r in records:
        if r.get("build_ok", False):
            first_compile = r["iteration"]
            break

    # Stagnation: last 3 same category AND same code hash seen before
    stagnation = False
    if len(trajectory) >= 3 and not passed:
        last3_cats = set(trajectory[-3:])
        if len(last3_cats) == 1:
            stagnation = True

    # Oscillation: ≥3 category changes in last 5 iterations
    oscillation = False
    if len(trajectory) >= 5 and not passed:
        recent = trajectory[-5:]
        transitions = sum(1 for i in range(1, len(recent)) if recent[i] != recent[i - 1])
        if transitions >= 3:
            oscillation = True

    # Progression: strictly improving trajectory (compile→logic→pass or compile→pass)
    SEVERITY = {"compile_stage_error": 3, "compile_error": 3, "buildability_bug": 3,
                "runtime_crash": 2, "memory_crash": 2, "illegal_memory_access": 2,
                "timeout": 2, "out_of_memory": 2,
                "wrong_result": 1, "logic_error": 1, "functional_correctness_bug": 1,
                "passed": 0}
    progression = False
    if passed and len(trajectory) >= 2:
        mapped = [CAT3_MAP.get(c, c) for c in trajectory]
        severities = [SEVERITY.get(c, SEVERITY.get(CAT3_MAP.get(c, ""), 99)) for c in mapped]
        # Check if severity is non-increasing (allows equal)
        progression = all(severities[i] >= severities[i + 1] for i in range(len(severities) - 1))

    return TaskMetrics(
        task_id=task_id,
        status="passed" if passed else "failed",
        iterations_used=len(records),
        trajectory=trajectory,
        trajectory_8cat=trajectory_8cat,
        code_hashes=code_hashes,
        pass_at_1=pass_at_1,
        first_compile_iter=first_compile,
        stop_reason=stop_reason if not passed else "passed",
        stagnation=stagnation,
        oscillation=oscillation,
        progression=progression,
    )


def compute_aggregate_metrics(
    task_metrics: list[TaskMetrics],
    max_iterations: int = 5,
) -> AggregateMetrics:
    """Compute aggregate metrics from all task metrics."""
    n = len(task_metrics)
    if n == 0:
        return AggregateMetrics()

    # Layer 1: Core
    pass_1 = sum(1 for t in task_metrics if t.pass_at_1)
    passed = sum(1 for t in task_metrics if t.status == "passed")
    compiled_1 = sum(1 for t in task_metrics if t.first_compile_iter == 1)

    # Layer 2: Debugging
    failed_at_1 = [t for t in task_metrics if not t.pass_at_1]
    debugged = sum(1 for t in failed_at_1 if t.status == "passed")
    debug_rate = debugged / len(failed_at_1) if failed_at_1 else 0.0

    iters_to_pass = [t.iterations_used for t in task_metrics if t.status == "passed"]
    avg_iters = sum(iters_to_pass) / len(iters_to_pass) if iters_to_pass else None

    # Fix rate by error type (first error category → eventually passed?)
    fix_by_type: dict[str, list[bool]] = defaultdict(list)
    for t in task_metrics:
        if not t.pass_at_1 and t.trajectory:
            first_cat = CAT3_MAP.get(t.trajectory[0], t.trajectory[0])
            fix_by_type[first_cat].append(t.status == "passed")
    fix_rate = {cat: sum(v) / len(v) for cat, v in fix_by_type.items() if v}

    # First compile pass distribution
    compile_dist = Counter()
    for t in task_metrics:
        if t.first_compile_iter is not None:
            key = str(t.first_compile_iter) if t.first_compile_iter <= 3 else "4+"
            compile_dist[key] += 1
    compile_total = sum(compile_dist.values()) or 1
    first_compile_dist = {k: v / compile_total for k, v in sorted(compile_dist.items())}

    # Layer 3: Quality
    failed_tasks = [t for t in task_metrics if t.status != "passed"]
    stag = sum(1 for t in failed_tasks if t.stagnation)
    osc = sum(1 for t in failed_tasks if t.oscillation)
    passed_after_1 = [t for t in task_metrics if t.status == "passed" and not t.pass_at_1]
    prog = sum(1 for t in passed_after_1 if t.progression)

    unique_ratios = []
    for t in task_metrics:
        if t.status == "passed" and t.iterations_used > 0:
            unique = len(set(h for h in t.code_hashes if h))
            unique_ratios.append(unique / t.iterations_used)

    stop_reasons = Counter(t.stop_reason for t in task_metrics)

    # Layer 4: Error Analysis
    # Per-category pass rate (3 merged categories)
    cat_results: dict[str, dict] = defaultdict(lambda: {"total": 0, "passed": 0})
    for t in task_metrics:
        if t.trajectory:
            cat3 = CAT3_MAP.get(t.trajectory[0], t.trajectory[0])
            cat_results[cat3]["total"] += 1
            if t.status == "passed":
                cat_results[cat3]["passed"] += 1
    per_cat = {
        cat: {**d, "rate": d["passed"] / d["total"] if d["total"] else 0}
        for cat, d in cat_results.items()
    }

    # Error transition matrix
    transitions: dict[str, Counter] = defaultdict(Counter)
    for t in task_metrics:
        mapped = [CAT3_MAP.get(c, c) for c in t.trajectory]
        for i in range(len(mapped) - 1):
            transitions[mapped[i]][mapped[i + 1]] += 1
    matrix = {}
    for src, counts in transitions.items():
        total = sum(counts.values())
        matrix[src] = {dst: c / total for dst, c in counts.items()} if total else {}

    # Top root causes for failed tasks
    root_cause_stats: dict[str, dict] = defaultdict(lambda: {"count": 0, "fixed": 0})
    for t in task_metrics:
        if not t.pass_at_1 and t.trajectory:
            # Use first iteration's root cause tag (not available in trajectory, approximate from category)
            tag = t.trajectory[0]
            root_cause_stats[tag]["count"] += 1
            if t.status == "passed":
                root_cause_stats[tag]["fixed"] += 1
    top_causes = sorted(
        [{"tag": tag, "count": d["count"], "fix_rate": d["fixed"] / d["count"] if d["count"] else 0}
         for tag, d in root_cause_stats.items()],
        key=lambda x: -x["count"],
    )[:10]

    return AggregateMetrics(
        total_tasks=n,
        pass_at_1=pass_1 / n,
        pass_at_k=passed / n,
        compilation_rate_at_1=compiled_1 / n,
        max_iterations=max_iterations,
        debug_rate_at_k=debug_rate,
        avg_iterations_to_pass=avg_iters,
        fix_rate=fix_rate,
        first_compile_pass_distribution=first_compile_dist,
        stagnation_rate=stag / len(failed_tasks) if failed_tasks else 0,
        oscillation_rate=osc / len(failed_tasks) if failed_tasks else 0,
        progression_rate=prog / len(passed_after_1) if passed_after_1 else 0,
        unique_approach_ratio=sum(unique_ratios) / len(unique_ratios) if unique_ratios else 0,
        early_stop_reasons=dict(stop_reasons),
        per_category_pass_rate=dict(per_cat),
        error_transition_matrix=matrix,
        top_root_causes_failed=top_causes,
    )


def metrics_to_dict(agg: AggregateMetrics) -> dict:
    """Convert to the summary.json schema from EVALUATION_DESIGN.md Section 8."""
    return {
        "core_metrics": {
            "pass_at_1": round(agg.pass_at_1, 4),
            "pass_at_k": round(agg.pass_at_k, 4),
            "max_iterations": agg.max_iterations,
            "compilation_rate_at_1": round(agg.compilation_rate_at_1, 4),
            "task_coverage": round(agg.task_coverage, 4),
        },
        "debugging_metrics": {
            "debug_rate_at_k": round(agg.debug_rate_at_k, 4),
            "avg_iterations_to_pass": round(agg.avg_iterations_to_pass, 2) if agg.avg_iterations_to_pass else None,
            "fix_rate": {k: round(v, 4) for k, v in agg.fix_rate.items()},
            "first_compile_pass_distribution": {k: round(v, 4) for k, v in agg.first_compile_pass_distribution.items()},
        },
        "quality_metrics": {
            "stagnation_rate": round(agg.stagnation_rate, 4),
            "oscillation_rate": round(agg.oscillation_rate, 4),
            "progression_rate": round(agg.progression_rate, 4),
            "unique_approach_ratio": round(agg.unique_approach_ratio, 4),
            "early_stop_reasons": agg.early_stop_reasons,
        },
        "error_analysis": {
            "per_category_pass_rate": agg.per_category_pass_rate,
            "error_transition_matrix": {
                src: {dst: round(p, 4) for dst, p in dsts.items()}
                for src, dsts in agg.error_transition_matrix.items()
            },
            "top_root_causes_failed": agg.top_root_causes_failed,
        },
    }
