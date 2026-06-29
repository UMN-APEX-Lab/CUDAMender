"""CLI: scan results_v* manifests, pool main+memcrash per (model, axis), write master.csv + parquet.

Usage:
    python -m harness.tools.integrated.build_master \
        --results-root evaluation \
        --out-dir evaluation/outputs_analysis/integrated
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .pool import compute_pooled_metrics, extract_per_task_status
from .scanner import Run, scan_manifests

logger = logging.getLogger("build_master")

_MAIN_EXPECTED_DEFAULT = 200
_MAIN_EXPECTED_BY_MODEL = {
    "gemma4": 192,
}
_MEMCRASH_EXPECTED = 16


def _load_main_whitelist(results_root: Path) -> set[str] | None:
    """Load canonical main task ID set from main_task_list.json. Used to filter
    out 8 trimmed flash_attn variants from early-model manifest records (Phase
    X.2 dropped 208 → 200 on 4/25). Without this, gemma4/glm_4_7/qwen36_27b
    iter would over-count denominator at 208 vs 200 elsewhere."""
    p = results_root / "main_task_list.json"
    if not p.exists():
        logger.warning("main_task_list.json not found at %s — skipping whitelist filter", p)
        return None
    data = json.load(open(p))
    ids = {e["task_id"] for e in data["entries"]}
    logger.info("loaded v2 whitelist: %d task IDs", len(ids))
    return ids

_METRICS = (
    "pass_at_1",
    "pass_at_k",
    "debug_rate_at_k",
    "compilation_rate_at_1",
    "avg_iters_to_pass",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "build_master.log"
    logging.basicConfig(
        filename=log_path, filemode="w", level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    main_whitelist = _load_main_whitelist(args.results_root)
    runs: list[Run] = scan_manifests(args.results_root)
    logger.info("found %d runs across results_v* dirs (rerun: %d)",
                len(runs), sum(1 for r in runs if r.is_rerun))

    # Group runs by (experiment, model, axis), separating main vs memcrash.
    # Each (cell, dataset) holds a LIST of Runs: main run + zero-or-more
    # rerun batches (results_v*_rerun/). Their records are concatenated
    # before pool computation; pool.py filters api_error rows so the rerun
    # successes naturally supersede the api_error main records.
    by_cell: dict[tuple[str, str, str], dict[str, list[Run]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in runs:
        cell = (r.experiment, r.model, r.axis)
        by_cell[cell][r.dataset].append(r)

    def _records_from(runs_list: list[Run]) -> list[dict]:
        # Sort: main first (is_rerun=False), rerun later — this matters
        # only if pool.py gains preferred-source dedup; today it filters api_err
        # and aggregates per-task, so order is informational.
        ordered = sorted(runs_list, key=lambda r: (r.is_rerun, r.timestamp))
        out = []
        for r in ordered:
            out.extend(r.load_records())
        return out

    rows: list[dict] = []
    coverage_counts: dict[str, int] = defaultdict(int)

    for (experiment, model, axis), per_dataset in sorted(by_cell.items()):
        main_runs = per_dataset.get("v2", [])
        memcrash_runs = per_dataset.get("v3", [])

        main_status = extract_per_task_status(_records_from(main_runs), main_whitelist) if main_runs else {}
        # memcrash supplement is by design 16 hard mem_crash IDs (subset of 200 main).
        # Apply same whitelist so the union is bounded by canonical 200.
        memcrash_status = extract_per_task_status(_records_from(memcrash_runs), main_whitelist) if memcrash_runs else {}

        main_expected = _MAIN_EXPECTED_BY_MODEL.get(model, _MAIN_EXPECTED_DEFAULT)
        try:
            metrics = compute_pooled_metrics(
                main_status, memcrash_status,
                main_expected=main_expected,
                memcrash_expected=_MEMCRASH_EXPECTED,
            )
        except ValueError as e:
            logger.warning("skipping (%s, %s, %s): %s", experiment, model, axis, e)
            continue

        key = metrics["coverage"]
        if key.startswith("main_partial"): key = "main_partial"
        elif key.startswith("memcrash_partial"): key = "memcrash_partial"
        coverage_counts[key] += 1

        for metric_name in _METRICS:
            value = metrics[metric_name]
            rows.append({
                "experiment": experiment,
                "model": model,
                "axis": axis,
                "metric": metric_name,
                "value": value,
                "n_tasks": metrics["n_tasks"],
                "n_main": metrics["n_main"],
                "n_memcrash": metrics["n_memcrash"],
                "coverage": metrics["coverage"],
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["experiment", "model", "axis", "metric"]).reset_index(drop=True)

    csv_path = args.out_dir / "master.csv"
    df.to_csv(csv_path, index=False)
    logger.info("wrote %s (%d rows)", csv_path, len(df))

    parquet_path = args.out_dir / "master.parquet"
    df.to_parquet(parquet_path, index=False)
    logger.info("wrote %s", parquet_path)

    n_models = df["model"].nunique() if not df.empty else 0
    n_experiments = df["experiment"].nunique() if not df.empty else 0
    n_cells = len(by_cell)
    coverage_str = ", ".join(f"{k}={v}" for k, v in sorted(coverage_counts.items()))
    summary = (
        f"[build_master] {n_models} models, {n_experiments} experiments, {n_cells} (model x axis) cells\n"
        f"  coverage: {coverage_str}\n"
        f"  master.csv: {len(df)} rows, {len(df.columns) if not df.empty else 0} columns\n"
        f"  log: {log_path}"
    )
    print(summary)
    logger.info(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
