#!/usr/bin/env python3
"""Retroactively measure speedup for old experiments that lack perf data.

Usage:
    CUDA_VISIBLE_DEVICES=6 python rescore_speedup.py

Reads manifests from old experiments, finds PASSED tasks without speedup,
measures their performance against the now-available reference cache,
and writes a supplementary JSON mapping task→speedup.
"""
import json
import re
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness.performance import measure_iter_perf

DATASET_DIR = Path(__file__).resolve().parent / "data"

EXPERIMENTS = {
    # (model, method, category): (experiment_root, manifest_relative)
    ("gemma", "CUDebugger", "mem"): "/tmp/memcrash_all_gemma/20260618_172300",
    ("gemma", "CUDebugger", "logic"): "/tmp/logic_gemma/20260620_150130",
    ("gemma", "CUDAForge", "mem"): "/tmp/memcrash_baseline_gemma_iterative/20260620_104626",
    ("gemma", "CUDAForge", "logic"): "/tmp/logic_gemma_iterative10/20260621_011342",
    ("gemma", "pass@5", "mem"): "/tmp/memcrash_baseline_gemma_repeated/20260620_121116",
    ("gemma", "pass@5", "logic"): "/tmp/logic_gemma_rep_pg095/20260626_192122",
    ("qwen", "CUDebugger", "mem"): "/tmp/memcrash_all_guided/20260617_194443",
    ("qwen", "CUDAForge", "mem"): "/tmp/memcrash_baseline_qwen_iterative/20260620_004626",
    ("qwen", "CUDAForge", "logic"): "/tmp/logic_qwen_iterative5/20260621_174243",
    ("qwen", "pass@5", "mem"): "/tmp/memcrash_baseline_qwen_repeated/20260620_054626",
    ("qwen", "pass@5", "logic"): "/tmp/logic_qwen_repeated5/20260621_174251",
}


def extract_instance_id(sample_id: str) -> str:
    """Extract full instance_id from sample_id by removing __iter_NNNN suffix."""
    return re.sub(r'__iter_\d+$', '', sample_id)


def extract_short_stem(instance_id: str) -> str | None:
    """Extract short task stem from instance_id, e.g. cuda_151 from cuda_151__memory_crash__abc123."""
    m = re.match(r'^(.+?)__(logic_error|memory_crash|compile_error|perf_broken|timeout)__[0-9a-f]+$', instance_id)
    return m.group(1) if m else None


def guess_solution_file(code_path: str) -> str:
    """Guess solution filename from the saved error_code path extension."""
    if code_path.endswith('.py'):
        return 'solution.py'
    return 'solution.cu'


def rescore_experiment(key, exp_root_str):
    exp_root = Path(exp_root_str)
    manifest_path = exp_root / "index" / "manifest.jsonl"
    if not manifest_path.exists():
        print(f"  SKIP: manifest not found at {manifest_path}")
        return {}

    output_path = exp_root / "index" / "rescore_speedup.json"
    existing = {}
    if output_path.exists():
        existing = json.loads(output_path.read_text())
        print(f"  Loaded {len(existing)} existing rescored entries")

    entries = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))

    # Group by task: for repeated/iterative, pick the best passing entry
    task_entries = defaultdict(list)
    for e in entries:
        instance_id = extract_instance_id(e['sample_id'])
        task_entries[instance_id].append(e)

    results = dict(existing)
    todo = []

    for instance_id, task_runs in task_entries.items():
        # Re-measure if previously got reference_not_cached (cache lookup might work with fallback now)
        if instance_id in results:
            prev = results[instance_id]
            if prev.get('skipped') not in ('reference_not_cached', 'reference_not_defined'):
                continue
            # Will retry with fallback

        # Find passing entries without speedup
        passing = [e for e in task_runs
                   if e.get('final_solution_status', e.get('status', '')) in ('PASSED', 'passed', 'pass')]
        if not passing:
            continue

        # Check if any passing entry already has speedup
        has_speedup = any(
            isinstance(e.get('performance', {}).get('speedup'), (int, float))
            for e in passing
        )
        if has_speedup:
            # Use existing speedup
            best = max(
                (e.get('performance', {}).get('speedup', 0) for e in passing
                 if isinstance(e.get('performance', {}).get('speedup'), (int, float))),
                default=None
            )
            if best is not None:
                results[instance_id] = {"speedup": best, "source": "original_manifest"}
                continue

        # Need to measure - pick the latest passing entry (most iterations)
        best_entry = passing[-1]
        code_rel = best_entry.get('paths', {}).get('error_code', '')
        if not code_rel:
            print(f"  SKIP {instance_id}: no error_code path in manifest")
            continue

        code_path = exp_root / code_rel
        if not code_path.exists():
            print(f"  SKIP {instance_id}: solution file not found at {code_path}")
            continue

        todo.append((instance_id, best_entry, code_path))

    if not todo:
        print(f"  Nothing to measure (all {len(results)} tasks already scored)")
        if results:
            output_path.write_text(json.dumps(results, indent=2) + "\n")
        return results

    print(f"  Measuring {len(todo)} tasks...")
    for i, (instance_id, entry, code_path) in enumerate(todo):
        code = code_path.read_text(encoding='utf-8')
        sol_file = guess_solution_file(str(code_path))

        short_stem = extract_short_stem(instance_id)
        manifest_tid = entry.get('task_id', '')

        print(f"  [{i+1}/{len(todo)}] {instance_id} ...", end=" ", flush=True)
        try:
            perf = measure_iter_perf(
                task_id=instance_id,
                code=code,
                solution_file=sol_file,
                gpu_id=0,
                dataset_dir=DATASET_DIR,
            )
            # Fallback: if cache miss with full instance_id, try short stem or manifest task_id
            if perf.get('skipped') in ('reference_not_cached', 'reference_not_defined'):
                for fallback_tid in [manifest_tid, short_stem]:
                    if fallback_tid and fallback_tid != instance_id:
                        perf2 = measure_iter_perf(
                            task_id=fallback_tid, code=code,
                            solution_file=sol_file, gpu_id=0,
                            dataset_dir=DATASET_DIR,
                        )
                        if perf2.get('speedup') is not None:
                            perf = perf2
                            break
            speedup = perf.get('speedup')
            skipped = perf.get('skipped')
            if speedup is not None:
                print(f"speedup={speedup:.4f}")
                results[instance_id] = {
                    "speedup": speedup,
                    "candidate_ms": perf.get('candidate_runtime_ms'),
                    "reference_ms": perf.get('reference_runtime_ms'),
                    "source": "rescored",
                }
            elif skipped:
                print(f"skipped={skipped}")
                results[instance_id] = {"speedup": None, "skipped": skipped, "source": "rescored"}
            else:
                print(f"unknown: {perf}")
                results[instance_id] = {"speedup": None, "raw": perf, "source": "rescored"}
        except Exception as e:
            print(f"ERROR: {e}")
            results[instance_id] = {"speedup": None, "error": str(e), "source": "rescored"}

        # Save after each task (resume-friendly)
        output_path.write_text(json.dumps(results, indent=2) + "\n")

    return results


def main():
    import os
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '?')
    print(f"GPU: CUDA_VISIBLE_DEVICES={gpu}")
    print(f"Dataset: {DATASET_DIR}")

    # Optional filter: pass model name(s) as argv to process only that subset
    filter_models = set(sys.argv[1:]) if len(sys.argv) > 1 else None
    if filter_models:
        print(f"Filter: {filter_models}")
    print()

    all_results = {}
    for key, exp_root in EXPERIMENTS.items():
        if filter_models and key[0] not in filter_models:
            continue
        label = f"{key[0]} / {key[1]} / {key[2]}"
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"  {exp_root}")
        print(f"{'='*60}")
        results = rescore_experiment(key, exp_root)
        all_results[label] = results

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for label, results in all_results.items():
        if not results:
            continue
        n = len(results)
        n_with_speedup = sum(1 for r in results.values() if r.get('speedup') is not None)
        speedups = [r['speedup'] for r in results.values() if r.get('speedup') is not None]
        n_ge_095 = sum(1 for s in speedups if s >= 0.95)
        n_ge_10 = sum(1 for s in speedups if s >= 1.0)
        print(f"{label}: {n} tasks, {n_with_speedup} measured, {n_ge_095} @>=0.95, {n_ge_10} @>=1.0")


if __name__ == '__main__':
    main()
