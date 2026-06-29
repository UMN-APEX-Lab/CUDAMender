"""NCU profile collection for Exp 3 L4 feedback.

Wraps `ncu` (NVIDIA Compute Profiler) to collect a small set of key
metrics and parse them into a dict suitable for build_feedback's
`ncu_data` argument.

Triggered only when build_ok=True AND test failed (gamma policy in spec).
For build-fail or pass cases the caller does not invoke this module.

If ncu is unavailable / blocked / times out, returns None and the caller
falls back to L3-only feedback.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger("cuda_debugger_exp")

# A compact set of metrics — keep the prompt addition under ~150 tokens.
METRICS = [
    "sm__warps_active.avg.pct_of_peak_sustained_active",  # achieved occupancy
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",  # memory throughput
    "sm__inst_executed.avg.pct_of_peak_sustained_active",  # SM efficiency
    "smsp__warp_issue_stalled_branch_resolving_per_inst_executed.ratio",  # branch stall
    "smsp__warps_eligible.avg.pct_of_peak_sustained_active",  # warp eligibility
    "smsp__warp_issue_stalled_long_scoreboard_per_inst_executed.ratio",  # mem-stall
]

METRIC_KEYS = {
    "sm__warps_active.avg.pct_of_peak_sustained_active": "achieved_occupancy_pct",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "dram_throughput_pct",
    "sm__inst_executed.avg.pct_of_peak_sustained_active": "sm_efficiency_pct",
    "smsp__warp_issue_stalled_branch_resolving_per_inst_executed.ratio": "branch_stall_pct",
    "smsp__warps_eligible.avg.pct_of_peak_sustained_active": "warp_eligibility_pct",
    "smsp__warp_issue_stalled_long_scoreboard_per_inst_executed.ratio": "long_scoreboard_pct",
}


def is_available() -> bool:
    """Return True if ncu binary is in PATH and responsive."""
    try:
        out = subprocess.run(
            ["ncu", "--version"], capture_output=True, text=True, timeout=5
        )
        return out.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _parse_ncu_csv(stdout: str) -> dict:
    """Parse ncu --csv output into {friendly_key: float_pct}."""
    result: dict = {}
    lines = stdout.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if "Metric Name" in line and "Metric Value" in line:
            header_idx = i
            break
    if header_idx is None:
        return result

    headers = [h.strip().strip('"') for h in lines[header_idx].split(",")]
    try:
        name_col = headers.index("Metric Name")
        val_col = headers.index("Metric Value")
    except ValueError:
        return result

    for line in lines[header_idx + 1:]:
        cols = [c.strip().strip('"') for c in line.split(",")]
        if len(cols) <= max(name_col, val_col):
            continue
        metric_name = cols[name_col]
        val_str = cols[val_col].rstrip("%").strip()
        if metric_name not in METRIC_KEYS:
            continue
        try:
            val = float(val_str)
        except ValueError:
            continue
        result[METRIC_KEYS[metric_name]] = val
    return result


def collect_metrics(
    workdir,
    task_stem: str,
    test_command: str = "./test.out",
    timeout: int = 60,
):
    """Run ncu against the workdir's test binary and return parsed metrics."""
    workdir = Path(workdir)
    if not is_available():
        logger.debug("ncu binary not available; skipping L4 collection")
        return None
    if not (workdir / "test.out").exists():
        logger.debug(f"no test.out in {workdir}; skipping ncu")
        return None

    cmd = [
        "ncu",
        "--csv",
        "--metrics", ",".join(METRICS),
        "--target-processes", "all",
        "--launch-skip", "0",
        "--launch-count", "1",
        test_command,
    ]
    try:
        out = subprocess.run(
            cmd, cwd=workdir, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        logger.warning(f"ncu timed out after {timeout}s for {task_stem}")
        return None
    if out.returncode != 0:
        logger.debug(f"ncu returned {out.returncode} for {task_stem}; parsing anyway")

    parsed = _parse_ncu_csv(out.stdout)
    if not parsed:
        logger.debug(f"ncu produced no parseable metrics for {task_stem}")
        return None
    return parsed
