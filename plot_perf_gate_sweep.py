#!/usr/bin/env python3
"""Plot pass rate vs performance gate for 3 models × 3 methods."""
import json
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

BASE = "/home/li004074/CUDABeaver"

# GPT ran 101 compile tasks; Gemma/Qwen ran 92. Exclude the 9 GPT-only tasks for fair comparison.
GPT_ONLY_COMPILE = {
    'cuda_107', 'cuda_204', 'cusolver_22',
    'cutlass_16_ampere_tensorop_conv2dfprop',
    'cutlass_25_ampere_fprop_mainloop_fusion',
    'cutlass_26_ampere_wgrad_mainloop_fusion',
    'cutlass_28_ampere_3xtf32_fast_accurate_tensorop_fprop',
    'cutlass_34_transposed_conv2d',
    'cutlass_41_fused_multi_head_attention',
}

MANIFESTS = {
    "GPT-5.4": {
        "CUDebugger": [
            f"{BASE}/results_memcrash_variant_agent_gpt5_4_low_pgate10_temp08_gpu4/20260626_014641/index/manifest.jsonl",
            f"{BASE}/results_logic_variant_agent_gpt5_4_low_pgate095_temp08_gpu2/20260626_041300/index/manifest.jsonl",
            f"{BASE}/results_compile_variant_agent_gpt5_4_low_pgate095_temp08_gpu3/20260626_041300/index/manifest.jsonl",
        ],
        "CUDAForge": [
            f"{BASE}/outputs/iter/gpt5_4_iterative_memcrash32_low/20260625_235453/index/manifest.jsonl",
            f"{BASE}/outputs/iter/gpt5_4_iterative_logic45_low_pgate095_temp08_gpu4/20260627_003749/index/manifest.jsonl",
            f"{BASE}/outputs/iter/gpt5_4_iterative_compile101_low_pgate095_temp08_maxtok65536_gpu5/20260627_003749/index/manifest.jsonl",
        ],
        "pass@5": [
            f"{BASE}/outputs/repeated/gpt5_4_repeated_memcrash32_low/20260625_235453/index/manifest.jsonl",
            f"{BASE}/outputs/repeated/gpt5_4_repeated_logic45_low_pgate095_temp08_gpu6/20260627_003749/index/manifest.jsonl",
            f"{BASE}/outputs/repeated/gpt5_4_repeated_compile101_low_pgate095_maxtok65536_gpu3/20260626_150232/index/manifest.jsonl",
        ],
    },
    "Gemma-4-31B": {
        "CUDebugger": [
            "/tmp/memcrash_all_gemma/20260618_172300/index/manifest.jsonl",
            "/tmp/logic_gemma/20260620_150130/index/manifest.jsonl",
            "/tmp/compile_gemma_guided_think_pg10/20260625_211216/index/manifest.jsonl",
        ],
        "CUDAForge": [
            "/tmp/memcrash_baseline_gemma_iterative/20260620_104626/index/manifest.jsonl",
            "/tmp/logic_gemma_iterative10/20260621_011342/index/manifest.jsonl",
            "/tmp/compile_gemma_iter_pg095/20260626_153554/index/manifest.jsonl",
        ],
        "pass@5": [
            "/tmp/memcrash_baseline_gemma_repeated/20260620_121116/index/manifest.jsonl",
            "/tmp/logic_gemma_rep_pg095/20260626_192122/index/manifest.jsonl",
            "/tmp/compile_gemma_rep_pg095/20260626_153554/index/manifest.jsonl",
        ],
    },
    "Qwen3.6-27B": {
        "CUDebugger": [
            "/tmp/memcrash_all_guided/20260617_194443/index/manifest.jsonl",
            "/tmp/logic_qwen_g5_tok65k/20260627_175416/index/manifest.jsonl",
            "/tmp/compile_qwen_guided_pg10/20260625_211216/index/manifest.jsonl",
        ],
        "CUDAForge": [
            "/tmp/memcrash_baseline_qwen_iterative/20260620_004626/index/manifest.jsonl",
            "/tmp/logic_qwen_iterative5/20260621_174243/index/manifest.jsonl",
            "/tmp/compile_qwen_iter_pg095/20260626_153554/index/manifest.jsonl",
        ],
        "pass@5": [
            "/tmp/memcrash_baseline_qwen_repeated/20260620_054626/index/manifest.jsonl",
            "/tmp/logic_qwen_repeated5/20260621_174251/index/manifest.jsonl",
            "/tmp/compile_qwen_rep_pg095/20260626_153554/index/manifest.jsonl",
        ],
    },
}


def get_best_speedup(perf):
    """Extract speedup from a performance dict. Returns float or None."""
    if not perf or not isinstance(perf, dict):
        return float('inf')  # no perf info → passes any gate

    skipped = perf.get('skipped')
    if skipped:
        if skipped in ('reference_not_defined', 'reference_not_cached'):
            return float('inf')  # no reference → correctness-only, passes any gate
        else:
            # build_failed_on_perf, candidate_runtime_invalid, etc.
            return 0.0  # candidate-side failure → no speedup evidence

    speedup = perf.get('speedup')
    if speedup is not None and isinstance(speedup, (int, float)):
        return float(speedup)

    return float('inf')  # fallback


def load_experiment(manifest_paths, exclude_tids=None):
    """Load manifests and return {composite_key: best_speedup_among_passing} and total task count.

    Keys by instance_id (full variant ID) to avoid collisions when two memcrash variants
    share the same short task_id.
    """
    task_best = {}
    exclude_tids = exclude_tids or set()

    for mi, path in enumerate(manifest_paths):
        p = Path(path)
        if not p.exists():
            print(f"  WARNING: missing {path}")
            continue

        task_results = defaultdict(list)
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                tid = d.get('task_id', '')
                if tid in exclude_tids:
                    continue
                sample_id = d.get('sample_id', '')
                instance_id = re.sub(r'__iter_\d+$', '', sample_id) if sample_id else tid
                status = d.get('final_solution_status', d.get('status', ''))
                category = d.get('category', '')
                passed = status in ('PASSED', 'passed', 'pass')
                perf = d.get('performance', {})
                perf_gate = d.get('performance_gate', {})

                correctness_ok = passed or category == 'perf_below_threshold'
                speedup = get_best_speedup(perf) if correctness_ok else None

                # Fallback: applied-kernels/kernelbench tasks store speedup in
                # performance_gate (from test stdout) instead of performance
                if correctness_ok and speedup == float('inf') and 'speedup' in perf_gate:
                    speedup = float(perf_gate['speedup'])

                task_results[instance_id].append((correctness_ok, speedup))

        for iid, results in task_results.items():
            key = f"{mi}_{iid}"
            passing_speedups = [s for (p, s) in results if p and s is not None]
            if passing_speedups:
                task_best[key] = max(passing_speedups)
            else:
                task_best[key] = None

    return task_best


def compute_pass_rates(task_best, gates):
    """Compute pass rate at each gate threshold."""
    total = len(task_best)
    if total == 0:
        return [0.0] * len(gates)

    rates = []
    for g in gates:
        if g == 0:
            # Gate 0 = correctness only
            count = sum(1 for s in task_best.values() if s is not None)
        else:
            count = sum(1 for s in task_best.values() if s is not None and s >= g)
        rates.append(count / total * 100)
    return rates


def main():
    gates = np.arange(0, 1.0, 0.1)  # 0.0, 0.1, ..., 0.9
    gates = np.append(gates, 0.95)  # add 0.95 as final point

    # Colors and styles
    method_styles = {
        'CUDebugger': {'color': '#2196F3', 'marker': 'o', 'linewidth': 2.5},
        'CUDAForge': {'color': '#FF9800', 'marker': 's', 'linewidth': 2.5},
        'pass@5':     {'color': '#4CAF50', 'marker': '^', 'linewidth': 2.5},
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for idx, (model, methods) in enumerate(MANIFESTS.items()):
        ax = axes[idx]
        print(f"\n=== {model} ===")

        exclude = GPT_ONLY_COMPILE if model == "GPT-5.4" else set()

        for method_name, paths in methods.items():
            print(f"  {method_name}:")
            task_best = load_experiment(paths, exclude_tids=exclude)
            total = len(task_best)
            passed_any = sum(1 for s in task_best.values() if s is not None)
            print(f"    total tasks: {total}, correctness-passing: {passed_any}")

            rates = compute_pass_rates(task_best, gates)
            style = method_styles[method_name]
            ax.plot(gates * 100, rates, label=method_name,
                    color=style['color'], marker=style['marker'],
                    linewidth=style['linewidth'], markersize=7)

            # Print key gate values
            for g, r in zip(gates, rates):
                gr = round(g, 2)
                if gr in (0, 0.5, 0.7, 0.95, 1.0):
                    print(f"    gate {gr}: {r:.1f}%")

        ax.set_title(model, fontsize=15, fontweight='bold')
        ax.set_xlabel('Performance Gate (%)', fontsize=12)
        if idx == 0:
            ax.set_ylabel('Pass Rate (%)', fontsize=12)
        ax.set_xlim(-2, 100)
        ax.set_ylim(0, 100)
        ax.set_xticks(range(0, 100, 10))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11, loc='lower left')

    fig.suptitle('CUDABeaver: Pass Rate vs Performance Gate', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    out_path = f"{BASE}/perf_gate_sweep.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved to {out_path}")

    pdf_path = f"{BASE}/perf_gate_sweep.pdf"
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f"Saved to {pdf_path}")


if __name__ == '__main__':
    main()
