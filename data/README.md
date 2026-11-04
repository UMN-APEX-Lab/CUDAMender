# CUDAMender 151-Task Dataset

This artifact intentionally ships the 151 tasks used in the paper experiments for "CUDAMender: Automated Agentic Performance-Aware CUDA Program Repair":

- 74 `compile_error` instances
- 45 `logic_error` instances
- 32 `memory_crash` instances

Flat layout directories (`input/`, `prompts/`, `testbench/`, `references/`) always include full instance-id aliases. Short-stem aliases are included only when the stem is unambiguous within this 151-task dataset.

The full CUDABeaver repository is available as the top-level `CUDABeaver/` submodule. Use that submodule if you need historical `perf_broken`/`timeout` tasks or full-dataset configs.
