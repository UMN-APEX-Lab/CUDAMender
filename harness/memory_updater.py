"""
Memory system for persistent model knowledge across experiments.

Inspired by agent-main's CLAUDE.md persistence pattern. Maintains three
markdown files that accumulate knowledge:

  memory/MODEL_PROFILES.md   — Per-model strengths, weaknesses, stats
  memory/TASK_DIFFICULTY.md  — Empirical difficulty tiers from multi-model runs
  memory/KNOWN_ISSUES.md     — Environment issues, data bugs, workarounds

Auto-updated after each experiment run via update_memory().
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from .metrics import CAT3_MAP, compute_task_metrics, compute_aggregate_metrics

logger = logging.getLogger("cuda_debugger_exp")

MEMORY_DIR = Path(__file__).resolve().parent.parent / "memory"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def update_memory(
    result_dir: Path,
    config: dict | None = None,
    all_records: list[dict] | None = None,
):
    """
    Update all memory files after an experiment run.

    Args:
        result_dir: Path to experiment output directory
        config: Experiment config dict (optional, will try to load from result_dir)
        all_records: Manifest records (optional, will load from result_dir)
    """
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)

    if all_records is None:
        manifest = result_dir / "index" / "manifest.jsonl"
        if not manifest.exists():
            logger.warning("No manifest found, skipping memory update")
            return
        all_records = []
        with open(manifest, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_records.append(json.loads(line))

    if config is None:
        config_file = result_dir / "config.yaml"
        if config_file.exists():
            import yaml
            config = yaml.safe_load(config_file.read_text())
        else:
            config = {}

    model_name = config.get("api", {}).get("model", "unknown")

    # Compute metrics
    by_task = defaultdict(list)
    for r in all_records:
        by_task[r["task_id"]].append(r)

    task_metrics = []
    task_pass = {}
    for task_id, recs in by_task.items():
        recs.sort(key=lambda r: r["iteration"])
        tm = compute_task_metrics(task_id, recs)
        task_metrics.append(tm)
        task_pass[task_id] = tm.status == "passed"

    max_iters = config.get("experiment", {}).get("max_iterations", 5)
    agg = compute_aggregate_metrics(task_metrics, max_iterations=max_iters)

    # Update each memory file
    _update_model_profiles(model_name, agg, task_metrics, result_dir)
    _update_task_difficulty(model_name, task_pass)
    _update_known_issues(all_records, result_dir)

    logger.info(f"Memory updated in {MEMORY_DIR}")


# ---------------------------------------------------------------------------
# MODEL_PROFILES.md
# ---------------------------------------------------------------------------


def _update_model_profiles(
    model_name: str,
    agg,
    task_metrics: list,
    result_dir: Path,
):
    """Update or append model profile."""
    profile_file = MEMORY_DIR / "MODEL_PROFILES.md"
    existing = _read_file(profile_file)

    # Build profile section
    timestamp = datetime.now().strftime("%Y-%m-%d")

    # Find strengths (categories with high fix rate)
    strengths = []
    for cat, rate in sorted(agg.fix_rate.items(), key=lambda x: -x[1]):
        if rate >= 0.7:
            strengths.append(f"{cat} (fix rate {rate:.0%})")

    # Find weaknesses (categories with low fix rate)
    weaknesses = []
    for cat, info in sorted(agg.per_category_pass_rate.items(), key=lambda x: x[1].get("rate", 0)):
        if info.get("rate", 0) < 0.3 and info.get("total", 0) > 0:
            weaknesses.append(f"{cat} (pass rate {info['rate']:.0%}, n={info['total']})")

    # Stop reason distribution
    stop_reasons = agg.early_stop_reasons
    avg_iters_str = f"{agg.avg_iterations_to_pass:.1f}" if agg.avg_iterations_to_pass else "N/A"
    strengths_str = ", ".join(strengths) if strengths else "N/A"
    weaknesses_str = ", ".join(weaknesses) if weaknesses else "N/A"

    section = f"""## {model_name}
- **Last updated**: {timestamp} (from `{result_dir.name}`)
- **pass@1**: {agg.pass_at_1:.1%} | **pass@{agg.max_iterations}**: {agg.pass_at_k:.1%}
- **debug_rate@{agg.max_iterations}**: {agg.debug_rate_at_k:.1%}
- **compilation_rate@1**: {agg.compilation_rate_at_1:.1%}
- **avg iterations to pass**: {avg_iters_str}
- **Stagnation**: {agg.stagnation_rate:.1%} | **Oscillation**: {agg.oscillation_rate:.1%} | **Progression**: {agg.progression_rate:.1%}
- **Strong**: {strengths_str}
- **Weak**: {weaknesses_str}
- **Stop reasons**: {json.dumps(stop_reasons)}
"""

    # Replace existing section or append
    marker_start = f"## {model_name}\n"
    if marker_start in existing:
        # Find the next ## or end of file
        start_idx = existing.index(marker_start)
        rest = existing[start_idx + len(marker_start):]
        next_section = rest.find("\n## ")
        if next_section >= 0:
            end_idx = start_idx + len(marker_start) + next_section
            new_content = existing[:start_idx] + section + existing[end_idx:]
        else:
            new_content = existing[:start_idx] + section
    else:
        if not existing:
            new_content = "# Model Profiles\n\n> Auto-updated after each experiment run.\n\n" + section
        else:
            new_content = existing.rstrip() + "\n\n" + section

    profile_file.write_text(new_content, encoding="utf-8")
    logger.info(f"Updated model profile for {model_name}")


# ---------------------------------------------------------------------------
# TASK_DIFFICULTY.md
# ---------------------------------------------------------------------------


def _update_task_difficulty(model_name: str, task_pass: dict[str, bool]):
    """Update task difficulty tiers with new model data."""
    diff_file = MEMORY_DIR / "TASK_DIFFICULTY.md"

    # Load existing difficulty data
    data_file = MEMORY_DIR / ".task_difficulty_data.json"
    if data_file.exists():
        data = json.loads(data_file.read_text())
    else:
        data = {}  # task_id -> {model_name: bool, ...}

    # Merge new results
    for task_id, passed in task_pass.items():
        if task_id not in data:
            data[task_id] = {}
        data[task_id][model_name] = passed

    # Save raw data for future updates
    data_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    # Compute difficulty tiers
    hard = []    # all models fail
    medium = []  # some pass, some fail
    easy = []    # all models pass

    for task_id in sorted(data.keys()):
        results = data[task_id]
        n_models = len(results)
        n_pass = sum(1 for v in results.values() if v)

        if n_pass == 0:
            hard.append((task_id, results))
        elif n_pass == n_models:
            easy.append((task_id, results))
        else:
            medium.append((task_id, results))

    # Sort medium by pass rate (ascending = harder first)
    medium.sort(key=lambda x: sum(v for v in x[1].values()) / len(x[1]))

    # Get all model names
    all_models = sorted(set(m for results in data.values() for m in results.keys()))
    timestamp = datetime.now().strftime("%Y-%m-%d")

    lines = [
        "# Task Difficulty Tiers\n",
        f"> Auto-updated: {timestamp} | Models: {', '.join(all_models)}\n",
        f"> Total tasks: {len(data)} | Hard: {len(hard)} | Medium: {len(medium)} | Easy: {len(easy)}\n\n",
    ]

    # Hard tasks
    lines.append(f"## Hard ({len(hard)} tasks — all models fail)\n\n")
    if hard:
        for task_id, results in hard[:30]:
            models_str = ", ".join(f"{m}:{'PASS' if v else 'FAIL'}" for m, v in sorted(results.items()))
            lines.append(f"- `{task_id}` ({models_str})\n")
        if len(hard) > 30:
            lines.append(f"- ... and {len(hard) - 30} more\n")
    else:
        lines.append("(none)\n")

    # Medium tasks (discriminative)
    lines.append(f"\n## Medium ({len(medium)} tasks — models disagree)\n\n")
    if medium:
        for task_id, results in medium[:30]:
            models_str = ", ".join(f"{m}:{'PASS' if v else 'FAIL'}" for m, v in sorted(results.items()))
            lines.append(f"- `{task_id}` ({models_str})\n")
        if len(medium) > 30:
            lines.append(f"- ... and {len(medium) - 30} more\n")
    else:
        lines.append("(none)\n")

    # Easy tasks (just count)
    lines.append(f"\n## Easy ({len(easy)} tasks — all models pass)\n\n")
    lines.append(f"{len(easy)} tasks passed by all {len(all_models)} models.\n")

    diff_file.write_text("".join(lines), encoding="utf-8")
    logger.info(f"Updated task difficulty ({len(hard)} hard, {len(medium)} medium, {len(easy)} easy)")


# ---------------------------------------------------------------------------
# KNOWN_ISSUES.md
# ---------------------------------------------------------------------------


def _update_known_issues(all_records: list[dict], result_dir: Path):
    """Detect and record known issues from this run."""
    issues_file = MEMORY_DIR / "KNOWN_ISSUES.md"
    existing = _read_file(issues_file)

    new_issues = []
    timestamp = datetime.now().strftime("%Y-%m-%d")
    run_name = result_dir.name

    # Detect environment issues
    env_count = 0
    timeout_count = 0
    api_error_count = 0
    log_dir = result_dir / "build_log_raw"

    for r in all_records:
        cat = r.get("category", r.get("five_category", ""))
        if cat in ("environment_dependency_bug",):
            env_count += 1
        if cat in ("timeout",):
            timeout_count += 1
        if r.get("error_type") == "api_error":
            api_error_count += 1

    if env_count > 0:
        issue = f"- [{timestamp}] `{run_name}`: {env_count} environment dependency issues detected"
        if issue not in existing:
            new_issues.append(issue)

    if api_error_count > 0:
        issue = f"- [{timestamp}] `{run_name}`: {api_error_count} API errors"
        if issue not in existing:
            new_issues.append(issue)

    if timeout_count > len(all_records) * 0.2:
        issue = f"- [{timestamp}] `{run_name}`: High timeout rate ({timeout_count}/{len(all_records)} = {timeout_count/len(all_records):.0%})"
        if issue not in existing:
            new_issues.append(issue)

    # Check for GPU arch issues in logs
    gpu_arch_issues = set()
    if log_dir.exists():
        for r in all_records:
            if r["status"] == "passed":
                continue
            log_file = log_dir / f"{r['sample_id']}.txt"
            if not log_file.exists():
                continue
            log_text = log_file.read_text(encoding="utf-8", errors="replace")
            if "no kernel image" in log_text.lower():
                task_id = r.get("task_id", r["sample_id"])
                gpu_arch_issues.add(task_id)

    if gpu_arch_issues:
        issue = f"- [{timestamp}] `{run_name}`: {len(gpu_arch_issues)} tasks with GPU arch mismatch (no kernel image)"
        if issue not in existing:
            new_issues.append(issue)

    if not new_issues and not existing:
        content = "# Known Issues\n\n> Auto-updated after each experiment run.\n\n(no issues detected)\n"
    elif new_issues:
        if not existing:
            content = "# Known Issues\n\n> Auto-updated after each experiment run.\n\n"
        else:
            content = existing.rstrip() + "\n"
        content += "\n".join(new_issues) + "\n"
    else:
        return  # No new issues and file exists, don't touch

    issues_file.write_text(content, encoding="utf-8")
    logger.info(f"Updated known issues ({len(new_issues)} new)")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_file(path: Path) -> str:
    """Read file or return empty string if not exists."""
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""
