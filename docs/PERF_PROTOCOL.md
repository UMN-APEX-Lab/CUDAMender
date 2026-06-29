# Performance Measurement Protocol

Each candidate kernel measured via:
- KernelBench eval_kernel_against_ref (cuda_event timing) for kernelbench tasks
- nsys-instrumented NVTX bench_region for curated CUDA tasks
- 10 outer benchmark runs x 10 internal num_perf_trials = 100 trials per task
- mean = mean of 10 outer means; CV = stddev(10 outer means) / mean
- CV gate: cv_ok = (cv < 0.03)
- speedup = reference_runtime_ms / candidate_runtime_ms

Reference baselines pre-cached using num_perf_trials=10 with 2 correctness trials.

Per-GPU perf flock prevents nsys/CUDA driver mutex contention; one perf
process per GPU.
