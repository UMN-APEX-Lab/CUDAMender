# Reproducing Paper Tables

## One-time setup

```bash
bash setup_env.sh
source .venv/bin/activate
bash scripts/prepare_benchmark.sh
```

`setup_env.sh` creates a venv, installs `requirements.txt`, and registers
the vendored upstream `kernelbench` package (`harness/_vendor/kernelbench/`,
MIT-licensed) onto the venv's `site-packages` via a `.pth` file so
`import kernelbench` works.

`prepare_benchmark.sh` materializes the flat dataset layout the harness
needs (`data/{input,prompts,testbench,references}/` symlinks pointing at
each per-instance dir, plus per-testbench `reference_sources/` symlinks
into the bundled CUTLASS / ThunderKittens headers). Idempotent; re-runnable.

Generated caches are not part of the artifact. Before performance-gated
experiments, measure reference baselines on the target machine:

```bash
python scripts/measure_logic_baselines.py \
  --dataset-root data \
  --task-list benchmark/task_list.json \
  --gpu <gpu-id> \
  --out outputs/reference_baselines_gpu<gpu-id>.jsonl
```

This populates `data/evaluation/references_cache/<gpu_signature>/`. The
applied-kernels evaluator may also create `harness/.cache/ref_builds/` while
compiling CUDA references; it is local scratch state and can be deleted.

## LLM endpoints — REQUIRED swap

Each cfg's `llm:` block has two fields a reviewer must point at their own
infrastructure before running the experiment:

- `base_url` — an OpenAI-compatible HTTPS endpoint. The harness uses any
  cloud API that follows the OpenAI Chat Completions schema, or a self-hosted
  vLLM server (`http://<host>:<port>/v1`).
- `api_key_env` — name of the env var that holds the API key. For local vLLM
  serving, set this var to any non-empty placeholder.

Configs ship with the endpoints + env-var names we used to generate the paper
results. Edit them, or use a launcher that overrides the cfg, to point at your
own cloud-API account or vLLM server before running. Configs targeting a local
vLLM server will silently fail until reachable.

## Family-specific runtime requirements

Most instances build with just `nvcc` from the CUDA toolkit. A few families
need additional setup; all required code/headers are bundled:

| Family                                  | n  | Bundled at                                          |
|-----------------------------------------|----|-----------------------------------------------------|
| cuda, cublas, cufft, cusolver, cusparse | 70 | (nvcc only)                                         |
| cutlass                                 | 20 | `data/_external/cutlass`     |
| thunderkittens                          |  0 | excluded from default tests; requires Hopper sm_90a |
| kernelbench                             | 84 | `harness/_vendor/kernelbench/` (registered via .pth)|
| flashattention, layernorm, cufft_samples| 35 | `harness/applied_kernels_runner/` + reference_sources |

## Reproducing tables

The harness's `--mode debug` path needs a legacy-format task list that maps
each cfg-side stem to its `broken_start` metadata. We ship one regenerated
from the public dataset at `benchmark/task_list.json` (see
`benchmark/README.md` for how it was built).

## Table 1 (iter axis, 7 models)

```bash
for cfg in configs/iter/*.yaml; do
  python -m harness.run_experiment \
    --config "$cfg" \
    --v2-task-list benchmark/task_list.json \
    --mode debug
done
python -m analysis.build_master --results-root outputs/iter --out-dir analysis_out
python -m analysis.pivot_tables --master analysis_out/master.csv --out-dir analysis_out
```

Output: `analysis_out/table_iter.tex` matches paper Table 1.

(Repeat for axes: repeated, feedback, history, fewshot.)
