"""Generate agent behavior analysis from existing experiment traces.

The report uses manifest/tool_trace paths as task identities. Some runs collapse
short task IDs in summary.json, while manifest paths preserve the instance-level
keys needed for per-task accounting.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]

GPT_RUNS = [
    {
        "model": "gpt-5.4",
        "failure_type": "compile_error",
        "path": "results_compile_variant_agent_gpt5_4_low_pgate095_temp08_gpu3/20260626_041300",
        "acceptance": "run-native `status=passed`",
    },
    {
        "model": "gpt-5.4",
        "failure_type": "logic_error",
        "path": "results_logic_variant_agent_gpt5_4_low_pgate095_temp08_gpu2/20260626_041300",
        "acceptance": "run-native `status=passed`",
    },
    {
        "model": "gpt-5.4",
        "failure_type": "memory_crash",
        "path": "results_memcrash_variant_agent_gpt5_4_low_pgate095_temp08_gpu2/20260626_145854",
        "acceptance": "run-native `status=passed`",
    },
]

MODEL_RUNS = [
    {
        "model": "gemma-4-31b",
        "failure_type": "logic_error",
        "path": "logs/logic_gemma/20260620_150130",
        "report_min_speedup": 1.0,
        "acceptance": "reported high-gate result",
    },
    {
        "model": "gemma-4-31b",
        "failure_type": "memory_crash",
        "path": "logs/memcrash_all_gemma/20260618_172300",
        "report_min_speedup": 1.0,
        "acceptance": "reported high-gate result",
    },
    {
        "model": "gemma-4-31b",
        "failure_type": "compile_error",
        "path": "logs/compile_gemma_guided_think_pg10/20260625_211216",
        "report_min_speedup": 1.0,
        "exclude_perf_skips": {"candidate_runtime_invalid"},
        "acceptance": "reported high-gate result",
    },
    {
        "model": "qwen3.6-27b",
        "failure_type": "logic_error",
        "path": "logs/logic_qwen_g5_tok65k/20260627_175416",
        "report_min_speedup": 1.0,
        "acceptance": "reported high-gate result",
    },
    {
        "model": "qwen3.6-27b",
        "failure_type": "memory_crash",
        "path": "logs/memcrash_all_guided/20260617_194443",
        "report_min_speedup": 1.0,
        "acceptance": "reported high-gate result",
    },
    {
        "model": "qwen3.6-27b",
        "failure_type": "compile_error",
        "path": "logs/compile_qwen_guided_pg10/20260625_211216",
        "report_min_speedup": 1.0,
        "acceptance": "reported high-gate result",
    },
]

TOOL_CATEGORIES = {
    "source_inspection": {"read_code", "read_test_file", "read_prompt"},
    "static_analysis": {"run_static_analysis"},
    "build_test": {"build_only", "test_only"},
    "compute_sanitizer": {"run_compute_sanitizer"},
    "localization": {"locate_bug"},
    "repair_submission": {"write_solution", "apply_edit"},
}

CATEGORY_ORDER = [
    "source_inspection",
    "static_analysis",
    "build_test",
    "compute_sanitizer",
    "localization",
    "repair_submission",
]

TOOL_TO_CATEGORY = {
    tool: category for category, tools in TOOL_CATEGORIES.items() for tool in tools
}


@dataclass
class TaskRecord:
    model: str
    failure_type: str
    key: str
    trace_path: Path
    budget: int
    attempts_used: int
    fixed: bool
    fixed_attempt: int | None
    final_category: str
    tool_counts: Counter
    raw_tool_counts: Counter


def pct(numer: int | float, denom: int | float, digits: int = 1) -> str:
    if not denom:
        return "0.0%"
    return f"{100 * numer / denom:.{digits}f}%"


def num(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def read_config_scalar(run_dir: Path, key: str) -> str:
    text = (run_dir / "config.yaml").read_text()
    match = re.search(rf"^\s*{re.escape(key)}:\s*([^\n#]+)", text, re.MULTILINE)
    return match.group(1).strip() if match else ""


def acceptance_reason(record: dict, run: dict) -> tuple[bool, str]:
    if record.get("status") != "passed":
        return False, "not_passed"

    perf = record.get("performance") or {}
    skipped = perf.get("skipped")
    excluded_skips = set(run.get("exclude_perf_skips", set()))
    if skipped in excluded_skips:
        return False, f"perf_skipped_{skipped}"

    threshold = run.get("report_min_speedup")
    speedup = perf.get("speedup")
    if threshold is not None and speedup is not None and speedup < threshold:
        return False, "perf_below_report_gate"

    return True, "accepted"


def final_category_for_rejected_pass(record: dict, run: dict) -> str:
    ok, reason = acceptance_reason(record, run)
    if ok:
        return "passed"
    if reason == "perf_below_report_gate":
        return "perf_below_report_gate"
    if reason.startswith("perf_skipped_"):
        return reason.replace("perf_skipped_", "perf_unmeasured_", 1)
    return record.get("category") or record.get("five_category") or "unknown"


def load_tasks(run: dict) -> list[TaskRecord]:
    run_dir = REPO_ROOT / run["path"]
    manifest_path = run_dir / "index" / "manifest.jsonl"
    grouped: dict[str, dict] = {}
    budget = int(read_config_scalar(run_dir, "max_patch_submissions") or 5)

    with manifest_path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            trace_rel = record["paths"]["tool_trace"]
            key = os.path.splitext(os.path.basename(trace_rel))[0]
            grouped.setdefault(
                key,
                {"records": [], "trace_path": run_dir / trace_rel},
            )["records"].append(record)

    tasks: list[TaskRecord] = []
    for key, data in sorted(grouped.items()):
        records = sorted(data["records"], key=lambda item: item["iteration"])
        accepted_iters = [
            item["iteration"] for item in records if acceptance_reason(item, run)[0]
        ]
        rejected_passes = [
            item
            for item in records
            if item.get("status") == "passed" and not acceptance_reason(item, run)[0]
        ]
        attempts_used = max(item["iteration"] for item in records)
        fixed_attempt = min(accepted_iters) if accepted_iters else None

        if fixed_attempt is None and rejected_passes:
            final_category = final_category_for_rejected_pass(rejected_passes[-1], run)
        else:
            final = records[-1]
            final_category = final.get("category") or final.get("five_category") or "unknown"

        raw_counts: Counter = Counter()
        category_counts: Counter = Counter({category: 0 for category in CATEGORY_ORDER})
        with data["trace_path"].open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                entry = json.loads(line)
                tool = entry.get("tool", "unknown")
                raw_counts[tool] += 1
                category_counts[TOOL_TO_CATEGORY.get(tool, "other")] += 1

        tasks.append(
            TaskRecord(
                model=run["model"],
                failure_type=run["failure_type"],
                key=key,
                trace_path=data["trace_path"],
                budget=budget,
                attempts_used=attempts_used,
                fixed=bool(accepted_iters),
                fixed_attempt=fixed_attempt,
                final_category=final_category,
                tool_counts=category_counts,
                raw_tool_counts=raw_counts,
            )
        )
    return tasks


def load_runs(runs: list[dict]) -> list[tuple[dict, list[TaskRecord]]]:
    return [(run, load_tasks(run)) for run in runs]


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def aggregate_counts(tasks: Iterable[TaskRecord]) -> Counter:
    counts: Counter = Counter({category: 0 for category in CATEGORY_ORDER})
    for task in tasks:
        counts.update(task.tool_counts)
    return counts


def aggregate_raw_counts(tasks: Iterable[TaskRecord]) -> Counter:
    counts: Counter = Counter()
    for task in tasks:
        counts.update(task.raw_tool_counts)
    return counts


def total_category_calls(tasks: list[TaskRecord]) -> int:
    counts = aggregate_counts(tasks)
    return sum(counts[category] for category in CATEGORY_ORDER)


def result_label(tasks: list[TaskRecord]) -> str:
    fixed = sum(task.fixed for task in tasks)
    return f"{fixed}/{len(tasks)} ({pct(fixed, len(tasks))})"


def source_rows(runs_with_tasks: list[tuple[dict, list[TaskRecord]]], include_model: bool) -> list[list[str]]:
    rows: list[list[str]] = []
    for run, tasks in runs_with_tasks:
        run_dir = REPO_ROOT / run["path"]
        gate = read_config_scalar(run_dir, "min_speedup")
        budget = read_config_scalar(run_dir, "max_patch_submissions")
        manifest_records = sum(1 for _ in (run_dir / "index" / "manifest.jsonl").open())
        trace_files = len(list((run_dir / "tool_trace").glob("*.jsonl")))
        row = []
        if include_model:
            row.append(run["model"])
        row.extend(
            [
                run["failure_type"],
                result_label(tasks),
                f"`{run['path']}`",
                gate,
                budget,
                str(len(tasks)),
                str(manifest_records),
                str(trace_files),
            ]
        )
        rows.append(row)
    return rows


def attempt_rows(grouped: list[tuple[list[str], list[TaskRecord]]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for labels, group in grouped:
        fixed = [task for task in group if task.fixed]
        failed = [task for task in group if not task.fixed]
        hist = Counter(task.fixed_attempt for task in fixed)
        exhausted = sum(task.attempts_used >= task.budget for task in failed)
        rows.append(
            labels
            + [
                str(len(group)),
                f"{len(fixed)} ({pct(len(fixed), len(group))})",
                str(hist[1]),
                str(hist[2]),
                str(hist[3]),
                str(hist[4]),
                str(hist[5]),
                str(len(failed)),
                f"{exhausted}/{len(failed)}",
            ]
        )
    return rows


def cumulative_attempt_rows(tasks: list[TaskRecord]) -> list[list[str]]:
    fixed_tasks = [task for task in tasks if task.fixed]
    hist = Counter(task.fixed_attempt for task in fixed_tasks)
    cumulative = 0
    rows: list[list[str]] = []
    for attempt in range(1, 6):
        count = hist[attempt]
        cumulative += count
        rows.append(
            [
                str(attempt),
                str(count),
                pct(count, len(fixed_tasks)),
                str(cumulative),
                pct(cumulative, len(fixed_tasks)),
                pct(cumulative, len(tasks)),
            ]
        )
    return rows


def final_category_rows(grouped: list[tuple[list[str], list[TaskRecord]]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for labels, group in grouped:
        failed = [task for task in group if not task.fixed]
        final_counts = Counter(task.final_category for task in failed)
        rows.append(
            labels
            + [
                str(len(failed)),
                ", ".join(f"{name}: {count}" for name, count in final_counts.most_common()),
            ]
        )
    return rows


def composition_rows(grouped: list[tuple[list[str], list[TaskRecord]]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for labels, group in grouped:
        counts = aggregate_counts(group)
        total_calls = sum(counts[category] for category in CATEGORY_ORDER)
        for category in CATEGORY_ORDER:
            rows.append(
                labels
                + [
                    category,
                    str(counts[category]),
                    num(counts[category] / len(group)),
                    pct(counts[category], total_calls),
                ]
            )
    return rows


def behavior_rows(grouped: list[tuple[list[str], list[TaskRecord]]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for labels, group in grouped:
        fixed = [task for task in group if task.fixed]
        counts = aggregate_counts(group)
        total = sum(counts[category] for category in CATEGORY_ORDER)
        rows.append(
            labels
            + [
                str(len(group)),
                f"{len(fixed)} ({pct(len(fixed), len(group))})",
                num(counts["repair_submission"] / len(group)),
                num(counts["source_inspection"] / len(group)),
                pct(sum(task.tool_counts["source_inspection"] > 0 for task in group), len(group)),
                num(counts["static_analysis"] / len(group)),
                num(counts["compute_sanitizer"] / len(group)),
                pct(sum(task.tool_counts["compute_sanitizer"] > 0 for task in group), len(group)),
                num(counts["localization"] / len(group)),
                num(total / len(group)),
            ]
        )
    return rows


def add_gpt_section(lines: list[str]) -> None:
    runs_with_tasks = load_runs(GPT_RUNS)
    tasks = [task for _, run_tasks in runs_with_tasks for task in run_tasks]
    fixed_tasks = [task for task in tasks if task.fixed]
    failed_tasks = [task for task in tasks if not task.fixed]
    by_type = [
        ([failure_type], [task for task in tasks if task.failure_type == failure_type])
        for failure_type in ["compile_error", "logic_error", "memory_crash"]
    ]

    lines.append("## A. GPT-5.4 Latest Agent Runs")
    lines.append("")
    lines.append(
        "Scope: latest GPT-5.4 blackboard-agent runs with a 0.95 performance gate "
        "and a 5-submission repair budget. Fixed means the run-native manifest "
        "record reached `status=passed`."
    )
    lines.append("")
    lines.append(
        "Note: the memory run's `summary.json` collapses two short task IDs, so this "
        "report uses `manifest.paths.tool_trace` stems as instance-level task keys."
    )
    lines.append("")
    lines.append(
        markdown_table(
            [
                "Initial failure type",
                "Result",
                "Run directory",
                "Gate",
                "Budget",
                "Tasks",
                "Manifest records",
                "Tool traces",
            ],
            source_rows(runs_with_tasks, include_model=False),
        )
    )
    lines.append("")

    lines.append("### A1. Submission Attempt Distribution")
    lines.append("")
    lines.append(
        markdown_table(
            [
                "Submission",
                "Fixed tasks",
                "% of fixed",
                "Cumulative fixed",
                "Cumulative % of fixed",
                "Cumulative % of all tasks",
            ],
            cumulative_attempt_rows(tasks),
        )
    )
    lines.append("")
    lines.append(
        markdown_table(
            [
                "Initial failure type",
                "Tasks",
                "Fixed",
                "Sub. 1",
                "Sub. 2",
                "Sub. 3",
                "Sub. 4",
                "Sub. 5",
                "Failed",
                "Failed exhausted",
            ],
            attempt_rows(by_type + [(["overall"], tasks)]),
        )
    )
    lines.append("")
    lines.append(
        markdown_table(
            ["Initial failure type", "Failed tasks", "Final categories after budget"],
            final_category_rows(by_type),
        )
    )
    lines.append("")
    all_hist = Counter(task.fixed_attempt for task in fixed_tasks)
    lines.append(
        "Interpretation: the system is mostly early-convergent: "
        f"{pct(sum(all_hist[i] for i in (1, 2)), len(fixed_tasks))} of fixed tasks pass by the "
        "second submission, and "
        f"{pct(sum(all_hist[i] for i in (1, 2, 3)), len(fixed_tasks))} pass by the third. "
        "The long tail still matters, especially for logic and memory tasks: submissions 4-5 "
        f"contribute {sum(all_hist[i] for i in (4, 5))} additional fixes. Every failed task "
        "exhausted all five submissions, so failures are budget-limited rather than early-aborted."
    )
    lines.append("")

    lines.append("### A2. Tool-Use Composition: Fixed vs Failed")
    lines.append("")
    lines.append(
        markdown_table(
            ["Outcome", "Tool category", "Calls", "Avg calls/task", "Share within outcome"],
            composition_rows([(["fixed"], fixed_tasks), (["failed"], failed_tasks)]),
        )
    )
    lines.append("")
    fixed_total = total_category_calls(fixed_tasks)
    failed_total = total_category_calls(failed_tasks)
    lines.append(
        f"Fixed tasks average {num(fixed_total / len(fixed_tasks))} categorized tool calls; "
        f"failed tasks average {num(failed_total / len(failed_tasks))}. The failed group spends "
        "more absolute effort in every diagnostic category because it always reaches five "
        "submissions. The relative mix is similar: localization plus repair submissions are "
        "about 56-57% of calls in both outcomes, while Compute Sanitizer is only about 4%."
    )
    lines.append("")
    lines.append(
        "The explicit `build_only` / `test_only` category is nearly absent because "
        "`write_solution` automatically triggers build/test feedback after each submitted patch."
    )
    lines.append("")

    lines.append("### A3. Behavior by Failure Type")
    lines.append("")
    lines.append(
        markdown_table(
            [
                "Initial failure type",
                "Tasks",
                "Fixed",
                "Avg submissions",
                "Avg source reads",
                "Any source read",
                "Avg static",
                "Avg sanitizer",
                "Any sanitizer",
                "Avg localization",
                "Avg total calls",
            ],
            behavior_rows(by_type),
        )
    )
    lines.append("")
    lines.append(
        "Interpretation: memory_crash tasks do use Compute Sanitizer most often "
        "(46.9% of tasks, 0.59 calls/task), but the gap over logic_error is modest "
        "(40.0%, 0.56 calls/task). Logic_error tasks show the heaviest repeated "
        "localization and code inspection, not test-file inspection. Compile_error "
        "uses testbench inspection most heavily."
    )
    lines.append("")
    lines.append(
        "A notable failure-mode transition appears in the exhausted tasks: memory_crash "
        "failures rarely remain pure memory crashes after five submissions; 6/9 end as "
        "functional correctness bugs, 2/9 as buildability bugs, and only 1/9 as an "
        "illegal memory access."
    )
    lines.append("")


def add_model_section(lines: list[str]) -> None:
    runs_with_tasks = load_runs(MODEL_RUNS)
    tasks = [task for _, run_tasks in runs_with_tasks for task in run_tasks]
    by_model = [
        ([model], [task for task in tasks if task.model == model])
        for model in ["gemma-4-31b", "qwen3.6-27b"]
    ]
    by_model_type = [
        (
            [model, failure_type],
            [
                task
                for task in tasks
                if task.model == model and task.failure_type == failure_type
            ],
        )
        for model in ["gemma-4-31b", "qwen3.6-27b"]
        for failure_type in ["logic_error", "memory_crash", "compile_error"]
    ]

    lines.append("## B. Gemma vs Qwen Agent Runs")
    lines.append("")
    lines.append(
        "Scope: the six gemma/qwen log directories provided in the request. Fixed/failed "
        "uses the reported high-gate result: measured speedups must be >=1.0 when a "
        "speedup is available; skipped performance references are kept as accepted, "
        "except `candidate_runtime_invalid`, which is treated as failed. This matches "
        "the supplied result table."
    )
    lines.append("")
    lines.append(
        markdown_table(
            [
                "Model",
                "Initial failure type",
                "Result",
                "Run directory",
                "Run gate",
                "Budget",
                "Tasks",
                "Manifest records",
                "Tool traces",
            ],
            source_rows(runs_with_tasks, include_model=True),
        )
    )
    lines.append("")

    lines.append("### B1. Submission Attempt Distribution")
    lines.append("")
    lines.append(
        markdown_table(
            [
                "Submission",
                "Fixed tasks",
                "% of fixed",
                "Cumulative fixed",
                "Cumulative % of fixed",
                "Cumulative % of all tasks",
            ],
            cumulative_attempt_rows(tasks),
        )
    )
    lines.append("")
    lines.append(
        markdown_table(
            [
                "Model",
                "Initial failure type",
                "Tasks",
                "Fixed",
                "Sub. 1",
                "Sub. 2",
                "Sub. 3",
                "Sub. 4",
                "Sub. 5",
                "Failed",
                "Failed exhausted",
            ],
            attempt_rows(by_model_type),
        )
    )
    lines.append("")
    lines.append(
        markdown_table(
            [
                "Model",
                "Tasks",
                "Fixed",
                "Sub. 1",
                "Sub. 2",
                "Sub. 3",
                "Sub. 4",
                "Sub. 5",
                "Failed",
                "Failed exhausted",
            ],
            attempt_rows(by_model),
        )
    )
    lines.append("")
    lines.append(
        markdown_table(
            ["Model", "Initial failure type", "Failed tasks", "Final categories after report gate"],
            final_category_rows(by_model_type),
        )
    )
    lines.append("")

    fixed_tasks = [task for task in tasks if task.fixed]
    hist = Counter(task.fixed_attempt for task in fixed_tasks)
    lines.append(
        "Interpretation: gemma/qwen are also strongly early-convergent. Across both models, "
        f"{pct(sum(hist[i] for i in (1, 2)), len(fixed_tasks))} of accepted fixes pass by "
        f"submission 2, and {pct(sum(hist[i] for i in (1, 2, 3)), len(fixed_tasks))} pass by "
        "submission 3. The long tail is smaller than in GPT-5.4: submissions 4-5 add "
        f"{sum(hist[i] for i in (4, 5))} accepted fixes. Unlike the GPT section, some "
        "reported failures did not exhaust the 5-submission budget because the original run "
        "stopped at a lower run gate or hit an early non-exhaustive stop before this report-gate "
        "reclassification."
    )
    lines.append("")

    lines.append("### B2. Tool-Use Composition: Fixed vs Failed")
    lines.append("")
    model_outcome_groups: list[tuple[list[str], list[TaskRecord]]] = []
    for model in ["gemma-4-31b", "qwen3.6-27b"]:
        model_tasks = [task for task in tasks if task.model == model]
        model_outcome_groups.extend(
            [
                ([model, "fixed"], [task for task in model_tasks if task.fixed]),
                ([model, "failed"], [task for task in model_tasks if not task.fixed]),
            ]
        )
    lines.append(
        markdown_table(
            ["Model", "Outcome", "Tool category", "Calls", "Avg calls/task", "Share within outcome"],
            composition_rows(model_outcome_groups),
        )
    )
    lines.append("")
    for model, group in by_model:
        total = total_category_calls(group)
        lines.append(
            f"{model[0]} averages {num(total / len(group))} categorized tool calls per task "
            f"over {len(group)} tasks."
        )
    lines.append(
        "For both models, failures spend roughly 2.4-2.5x as many categorized tool calls as "
        "accepted fixes. Localization remains the largest single category, while Compute "
        "Sanitizer is much more visible here than in the GPT-5.4 run because gemma/qwen "
        "logic and memory traces call it frequently."
    )
    lines.append("")

    lines.append("### B3. Behavior by Model and Failure Type")
    lines.append("")
    lines.append(
        markdown_table(
            [
                "Model",
                "Initial failure type",
                "Tasks",
                "Fixed",
                "Avg submissions",
                "Avg source reads",
                "Any source read",
                "Avg static",
                "Avg sanitizer",
                "Any sanitizer",
                "Avg localization",
                "Avg total calls",
            ],
            behavior_rows(by_model_type),
        )
    )
    lines.append("")
    lines.append(
        "Behavioral differences: qwen spends more effort on logic tasks than gemma "
        "(19.20 vs 15.53 total calls/task), mostly from source inspection and localization, "
        "and gets a modestly higher logic result (20/45 vs 17/45). Memory_crash uses "
        "Compute Sanitizer often for both models, but gemma calls it more frequently "
        "(65.6% of memory tasks vs 53.1% for qwen) while qwen still solves one more memory "
        "task. Compile_error is similar in total effort, but gemma has the better result "
        "(51/92 vs 46/92) despite slightly fewer localization calls."
    )
    lines.append("")
    lines.append(
        "Failure transitions are consistent with the GPT-5.4 pattern: memory_crash failures "
        "usually stop as functional correctness bugs rather than remaining pure illegal-memory "
        "crashes. Gemma memory failures end as 8 functional-correctness, 3 illegal-memory, "
        "and 2 buildability cases; qwen memory failures end as 10 functional-correctness and "
        "2 illegal-memory cases."
    )
    lines.append("")


def add_appendix(lines: list[str], all_runs: list[dict]) -> None:
    all_tasks = [task for _, tasks in load_runs(all_runs) for task in tasks]
    lines.append("## Appendix: Tool Mapping and Raw Counts")
    lines.append("")
    mapping_rows = [
        [category, ", ".join(f"`{tool}`" for tool in sorted(tools))]
        for category, tools in TOOL_CATEGORIES.items()
    ]
    lines.append(markdown_table(["Category", "Tools"], mapping_rows))
    lines.append("")
    raw_counts = aggregate_raw_counts(all_tasks)
    raw_rows = [[f"`{tool}`", str(count)] for tool, count in raw_counts.most_common()]
    lines.append(markdown_table(["Tool", "Calls across all report sections"], raw_rows))
    lines.append("")


def write_report(output_path: Path) -> None:
    lines: list[str] = []
    lines.append("# Agent Behavior Analysis: Submission Attempts and Tool Use")
    lines.append("")
    lines.append(
        "This report summarizes submission-attempt distributions, tool-use composition, "
        "and behavior by failure type from existing CUDABeaver agent traces."
    )
    lines.append("")
    add_gpt_section(lines)
    add_model_section(lines)
    add_appendix(lines, GPT_RUNS + MODEL_RUNS)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "docs" / "agent_behavior_analysis.md",
        help="Markdown output path.",
    )
    args = parser.parse_args()
    output = args.output
    if not output.is_absolute():
        output = REPO_ROOT / output
    write_report(output)
    print(output)


if __name__ == "__main__":
    main()
