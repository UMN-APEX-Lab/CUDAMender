# Agent Behavior Analysis: Submission Attempts and Tool Use

This report summarizes submission-attempt distributions, tool-use composition, and behavior by failure type from existing CUDABeaver agent traces.

## A. GPT-5.4 Latest Agent Runs

Scope: latest GPT-5.4 blackboard-agent runs with a 0.95 performance gate and a 5-submission repair budget. Fixed means the run-native manifest record reached `status=passed`.

Note: the memory run's `summary.json` collapses two short task IDs, so this report uses `manifest.paths.tool_trace` stems as instance-level task keys.

| Initial failure type | Result | Run directory | Gate | Budget | Tasks | Manifest records | Tool traces |
|---|---|---|---|---|---|---|---|
| compile_error | 77/101 (76.2%) | `results_compile_variant_agent_gpt5_4_low_pgate095_temp08_gpu3/20260626_041300` | 0.95 | 5 | 101 | 247 | 101 |
| logic_error | 34/45 (75.6%) | `results_logic_variant_agent_gpt5_4_low_pgate095_temp08_gpu2/20260626_041300` | 0.95 | 5 | 45 | 139 | 45 |
| memory_crash | 23/32 (71.9%) | `results_memcrash_variant_agent_gpt5_4_low_pgate095_temp08_gpu2/20260626_145854` | 0.95 | 5 | 32 | 96 | 32 |

### A1. Submission Attempt Distribution

| Submission | Fixed tasks | % of fixed | Cumulative fixed | Cumulative % of fixed | Cumulative % of all tasks |
|---|---|---|---|---|---|
| 1 | 68 | 50.7% | 68 | 50.7% | 38.2% |
| 2 | 27 | 20.1% | 95 | 70.9% | 53.4% |
| 3 | 20 | 14.9% | 115 | 85.8% | 64.6% |
| 4 | 15 | 11.2% | 130 | 97.0% | 73.0% |
| 5 | 4 | 3.0% | 134 | 100.0% | 75.3% |

| Initial failure type | Tasks | Fixed | Sub. 1 | Sub. 2 | Sub. 3 | Sub. 4 | Sub. 5 | Failed | Failed exhausted |
|---|---|---|---|---|---|---|---|---|---|
| compile_error | 101 | 77 (76.2%) | 46 | 18 | 9 | 2 | 2 | 24 | 24/24 |
| logic_error | 45 | 34 (75.6%) | 12 | 5 | 8 | 7 | 2 | 11 | 11/11 |
| memory_crash | 32 | 23 (71.9%) | 10 | 4 | 3 | 6 | 0 | 9 | 9/9 |
| overall | 178 | 134 (75.3%) | 68 | 27 | 20 | 15 | 4 | 44 | 44/44 |

| Initial failure type | Failed tasks | Final categories after budget |
|---|---|---|
| compile_error | 24 | functional_correctness_bug: 10, buildability_bug: 7, perf_below_threshold: 4, illegal_memory_access: 3 |
| logic_error | 11 | functional_correctness_bug: 5, buildability_bug: 4, perf_below_threshold: 2 |
| memory_crash | 9 | functional_correctness_bug: 6, buildability_bug: 2, illegal_memory_access: 1 |

Interpretation: the system is mostly early-convergent: 70.9% of fixed tasks pass by the second submission, and 85.8% pass by the third. The long tail still matters, especially for logic and memory tasks: submissions 4-5 contribute 19 additional fixes. Every failed task exhausted all five submissions, so failures are budget-limited rather than early-aborted.

### A2. Tool-Use Composition: Fixed vs Failed

| Outcome | Tool category | Calls | Avg calls/task | Share within outcome |
|---|---|---|---|---|
| fixed | source_inspection | 265 | 1.98 | 22.4% |
| fixed | static_analysis | 208 | 1.55 | 17.6% |
| fixed | build_test | 1 | 0.01 | 0.1% |
| fixed | compute_sanitizer | 44 | 0.33 | 3.7% |
| fixed | localization | 403 | 3.01 | 34.1% |
| fixed | repair_submission | 262 | 1.96 | 22.1% |
| failed | source_inspection | 216 | 4.91 | 23.4% |
| failed | static_analysis | 144 | 3.27 | 15.6% |
| failed | build_test | 0 | 0.00 | 0.0% |
| failed | compute_sanitizer | 36 | 0.82 | 3.9% |
| failed | localization | 308 | 7.00 | 33.3% |
| failed | repair_submission | 220 | 5.00 | 23.8% |

Fixed tasks average 8.83 categorized tool calls; failed tasks average 21.00. The failed group spends more absolute effort in every diagnostic category because it always reaches five submissions. The relative mix is similar: localization plus repair submissions are about 56-57% of calls in both outcomes, while Compute Sanitizer is only about 4%.

The explicit `build_only` / `test_only` category is nearly absent because `write_solution` automatically triggers build/test feedback after each submitted patch.

### A3. Behavior by Failure Type

| Initial failure type | Tasks | Fixed | Avg submissions | Avg source reads | Any source read | Avg static | Avg sanitizer | Any sanitizer | Avg localization | Avg total calls |
|---|---|---|---|---|---|---|---|---|---|---|
| compile_error | 101 | 77 (76.2%) | 2.45 | 2.78 | 93.1% | 1.86 | 0.36 | 24.8% | 3.44 | 10.88 |
| logic_error | 45 | 34 (75.6%) | 3.09 | 2.62 | 95.6% | 2.24 | 0.56 | 40.0% | 5.02 | 13.56 |
| memory_crash | 32 | 23 (71.9%) | 3.00 | 2.56 | 81.2% | 1.97 | 0.59 | 46.9% | 4.31 | 12.44 |

Interpretation: memory_crash tasks do use Compute Sanitizer most often (46.9% of tasks, 0.59 calls/task), but the gap over logic_error is modest (40.0%, 0.56 calls/task). Logic_error tasks show the heaviest repeated localization and code inspection, not test-file inspection. Compile_error uses testbench inspection most heavily.

A notable failure-mode transition appears in the exhausted tasks: memory_crash failures rarely remain pure memory crashes after five submissions; 6/9 end as functional correctness bugs, 2/9 as buildability bugs, and only 1/9 as an illegal memory access.

## B. Gemma vs Qwen Agent Runs

Scope: the six gemma/qwen log directories provided in the request. Fixed/failed uses the reported high-gate result: measured speedups must be >=1.0 when a speedup is available; skipped performance references are kept as accepted, except `candidate_runtime_invalid`, which is treated as failed. This matches the supplied result table.

| Model | Initial failure type | Result | Run directory | Run gate | Budget | Tasks | Manifest records | Tool traces |
|---|---|---|---|---|---|---|---|---|
| gemma-4-31b | logic_error | 17/45 (37.8%) | `logs/logic_gemma/20260620_150130` | 0.7 | 5 | 45 | 156 | 45 |
| gemma-4-31b | memory_crash | 19/32 (59.4%) | `logs/memcrash_all_gemma/20260618_172300` | 0.7 | 5 | 32 | 93 | 32 |
| gemma-4-31b | compile_error | 51/92 (55.4%) | `logs/compile_gemma_guided_think_pg10/20260625_211216` | 1.0 | 5 | 92 | 303 | 92 |
| qwen3.6-27b | logic_error | 20/45 (44.4%) | `logs/logic_qwen_g5_tok65k/20260627_175416` | 0.7 | 5 | 45 | 147 | 45 |
| qwen3.6-27b | memory_crash | 20/32 (62.5%) | `logs/memcrash_all_guided/20260617_194443` | 0.7 | 5 | 32 | 88 | 32 |
| qwen3.6-27b | compile_error | 46/92 (50.0%) | `logs/compile_qwen_guided_pg10/20260625_211216` | 1.0 | 5 | 92 | 308 | 92 |

### B1. Submission Attempt Distribution

| Submission | Fixed tasks | % of fixed | Cumulative fixed | Cumulative % of fixed | Cumulative % of all tasks |
|---|---|---|---|---|---|
| 1 | 111 | 64.2% | 111 | 64.2% | 32.8% |
| 2 | 25 | 14.5% | 136 | 78.6% | 40.2% |
| 3 | 17 | 9.8% | 153 | 88.4% | 45.3% |
| 4 | 7 | 4.0% | 160 | 92.5% | 47.3% |
| 5 | 13 | 7.5% | 173 | 100.0% | 51.2% |

| Model | Initial failure type | Tasks | Fixed | Sub. 1 | Sub. 2 | Sub. 3 | Sub. 4 | Sub. 5 | Failed | Failed exhausted |
|---|---|---|---|---|---|---|---|---|---|---|
| gemma-4-31b | logic_error | 45 | 17 (37.8%) | 13 | 2 | 1 | 1 | 0 | 28 | 26/28 |
| gemma-4-31b | memory_crash | 32 | 19 (59.4%) | 12 | 5 | 2 | 0 | 0 | 13 | 13/13 |
| gemma-4-31b | compile_error | 92 | 51 (55.4%) | 30 | 7 | 6 | 0 | 8 | 41 | 40/41 |
| qwen3.6-27b | logic_error | 45 | 20 (44.4%) | 9 | 2 | 5 | 3 | 1 | 25 | 17/25 |
| qwen3.6-27b | memory_crash | 32 | 20 (62.5%) | 16 | 2 | 1 | 0 | 1 | 12 | 12/12 |
| qwen3.6-27b | compile_error | 92 | 46 (50.0%) | 31 | 7 | 2 | 3 | 3 | 46 | 46/46 |

| Model | Tasks | Fixed | Sub. 1 | Sub. 2 | Sub. 3 | Sub. 4 | Sub. 5 | Failed | Failed exhausted |
|---|---|---|---|---|---|---|---|---|---|
| gemma-4-31b | 169 | 87 (51.5%) | 55 | 14 | 9 | 1 | 8 | 82 | 79/82 |
| qwen3.6-27b | 169 | 86 (50.9%) | 56 | 11 | 8 | 6 | 5 | 83 | 75/83 |

| Model | Initial failure type | Failed tasks | Final categories after report gate |
|---|---|---|---|
| gemma-4-31b | logic_error | 28 | functional_correctness_bug: 19, buildability_bug: 4, perf_below_threshold: 2, perf_below_report_gate: 2, out_of_memory: 1 |
| gemma-4-31b | memory_crash | 13 | functional_correctness_bug: 8, illegal_memory_access: 3, buildability_bug: 2 |
| gemma-4-31b | compile_error | 41 | functional_correctness_bug: 16, buildability_bug: 16, perf_below_threshold: 4, illegal_memory_access: 3, timeout: 1, perf_unmeasured_candidate_runtime_invalid: 1 |
| qwen3.6-27b | logic_error | 25 | functional_correctness_bug: 20, buildability_bug: 2, perf_below_report_gate: 2, perf_below_threshold: 1 |
| qwen3.6-27b | memory_crash | 12 | functional_correctness_bug: 10, illegal_memory_access: 2 |
| qwen3.6-27b | compile_error | 46 | functional_correctness_bug: 23, buildability_bug: 15, illegal_memory_access: 4, perf_below_threshold: 3, timeout: 1 |

Interpretation: gemma/qwen are also strongly early-convergent. Across both models, 78.6% of accepted fixes pass by submission 2, and 88.4% pass by submission 3. The long tail is smaller than in GPT-5.4: submissions 4-5 add 20 accepted fixes. Unlike the GPT section, some reported failures did not exhaust the 5-submission budget because the original run stopped at a lower run gate or hit an early non-exhaustive stop before this report-gate reclassification.

### B2. Tool-Use Composition: Fixed vs Failed

| Model | Outcome | Tool category | Calls | Avg calls/task | Share within outcome |
|---|---|---|---|---|---|
| gemma-4-31b | fixed | source_inspection | 158 | 1.82 | 21.5% |
| gemma-4-31b | fixed | static_analysis | 138 | 1.59 | 18.8% |
| gemma-4-31b | fixed | build_test | 8 | 0.09 | 1.1% |
| gemma-4-31b | fixed | compute_sanitizer | 49 | 0.56 | 6.7% |
| gemma-4-31b | fixed | localization | 229 | 2.63 | 31.1% |
| gemma-4-31b | fixed | repair_submission | 154 | 1.77 | 20.9% |
| gemma-4-31b | failed | source_inspection | 348 | 4.24 | 21.1% |
| gemma-4-31b | failed | static_analysis | 248 | 3.02 | 15.0% |
| gemma-4-31b | failed | build_test | 13 | 0.16 | 0.8% |
| gemma-4-31b | failed | compute_sanitizer | 162 | 1.98 | 9.8% |
| gemma-4-31b | failed | localization | 484 | 5.90 | 29.3% |
| gemma-4-31b | failed | repair_submission | 398 | 4.85 | 24.1% |
| qwen3.6-27b | fixed | source_inspection | 168 | 1.95 | 21.8% |
| qwen3.6-27b | fixed | static_analysis | 140 | 1.63 | 18.1% |
| qwen3.6-27b | fixed | build_test | 10 | 0.12 | 1.3% |
| qwen3.6-27b | fixed | compute_sanitizer | 67 | 0.78 | 8.7% |
| qwen3.6-27b | fixed | localization | 236 | 2.74 | 30.6% |
| qwen3.6-27b | fixed | repair_submission | 151 | 1.76 | 19.6% |
| qwen3.6-27b | failed | source_inspection | 395 | 4.76 | 21.8% |
| qwen3.6-27b | failed | static_analysis | 284 | 3.42 | 15.7% |
| qwen3.6-27b | failed | build_test | 18 | 0.22 | 1.0% |
| qwen3.6-27b | failed | compute_sanitizer | 129 | 1.55 | 7.1% |
| qwen3.6-27b | failed | localization | 590 | 7.11 | 32.6% |
| qwen3.6-27b | failed | repair_submission | 392 | 4.72 | 21.7% |

gemma-4-31b averages 14.14 categorized tool calls per task over 169 tasks.
qwen3.6-27b averages 15.27 categorized tool calls per task over 169 tasks.
For both models, failures spend roughly 2.4-2.5x as many categorized tool calls as accepted fixes. Localization remains the largest single category, while Compute Sanitizer is much more visible here than in the GPT-5.4 run because gemma/qwen logic and memory traces call it frequently.

### B3. Behavior by Model and Failure Type

| Model | Initial failure type | Tasks | Fixed | Avg submissions | Avg source reads | Any source read | Avg static | Avg sanitizer | Any sanitizer | Avg localization | Avg total calls |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma-4-31b | logic_error | 45 | 17 (37.8%) | 3.47 | 2.89 | 88.9% | 2.51 | 2.11 | 86.7% | 4.53 | 15.53 |
| gemma-4-31b | memory_crash | 32 | 19 (59.4%) | 2.91 | 3.09 | 100.0% | 1.78 | 1.09 | 65.6% | 3.88 | 12.75 |
| gemma-4-31b | compile_error | 92 | 51 (55.4%) | 3.29 | 3.01 | 81.5% | 2.35 | 0.88 | 41.3% | 4.18 | 13.93 |
| qwen3.6-27b | logic_error | 45 | 20 (44.4%) | 3.27 | 4.91 | 100.0% | 3.51 | 1.82 | 73.3% | 5.56 | 19.20 |
| qwen3.6-27b | memory_crash | 32 | 20 (62.5%) | 2.75 | 3.28 | 87.5% | 2.34 | 0.97 | 53.1% | 4.50 | 13.84 |
| qwen3.6-27b | compile_error | 92 | 46 (50.0%) | 3.35 | 2.58 | 82.6% | 2.08 | 0.90 | 46.7% | 4.70 | 13.84 |

Behavioral differences: qwen spends more effort on logic tasks than gemma (19.20 vs 15.53 total calls/task), mostly from source inspection and localization, and gets a modestly higher logic result (20/45 vs 17/45). Memory_crash uses Compute Sanitizer often for both models, but gemma calls it more frequently (65.6% of memory tasks vs 53.1% for qwen) while qwen still solves one more memory task. Compile_error is similar in total effort, but gemma has the better result (51/92 vs 46/92) despite slightly fewer localization calls.

Failure transitions are consistent with the GPT-5.4 pattern: memory_crash failures usually stop as functional correctness bugs rather than remaining pure illegal-memory crashes. Gemma memory failures end as 8 functional-correctness, 3 illegal-memory, and 2 buildability cases; qwen memory failures end as 10 functional-correctness and 2 illegal-memory cases.

## Appendix: Tool Mapping and Raw Counts

| Category | Tools |
|---|---|
| source_inspection | `read_code`, `read_prompt`, `read_test_file` |
| static_analysis | `run_static_analysis` |
| build_test | `build_only`, `test_only` |
| compute_sanitizer | `run_compute_sanitizer` |
| localization | `locate_bug` |
| repair_submission | `apply_edit`, `write_solution` |

| Tool | Calls across all report sections |
|---|---|
| `locate_bug` | 2250 |
| `write_solution` | 1577 |
| `run_static_analysis` | 1162 |
| `read_code` | 1162 |
| `run_compute_sanitizer` | 487 |
| `read_test_file` | 385 |
| `build_only` | 39 |
| `test_only` | 11 |
| `read_prompt` | 3 |

