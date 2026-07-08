# Artifact for CUDAMender: Automated Agentic Performance-Aware CUDA Program Repair

This artifact packages the code and the 169-task benchmark data needed to reproduce the experiments for "CUDAMender: Automated Agentic Performance-Aware CUDA Program Repair".
The repository has two layers:

- Top-level artifact files: a runnable snapshot of the harness, configs, dataset, scripts, tests, and analysis code needed for reproduction.
- `CUDABeaver/`: a git submodule pointing at the full upstream CUDABeaver benchmark repository for provenance and complete source history.

## Layout

```text
CUDAMender/
  README.md                 artifact instructions
  harness/                  experiment runner, agent loop, tools, compiler, scoring
  harness/cudeubgger/       blackboard / ReAct CUDAMender agent implementation
  configs/                  agent, iterative, repeated, feedback, history, few-shot configs
  data/                     169 paper-task instances, references, prompts, testbenches, bundled headers
  benchmark/task_list.json  legacy task-list used by --mode debug
  scripts/                  setup and dataset materialization helpers
  analysis/                 aggregation/report scripts
  tests/                    smoke/unit tests that do not require an LLM key
  CUDABeaver/               full CUDABeaver repo as a git submodule
```

No `logs/`, `outputs/`, or `results_*` directories are included in the artifact copy. Re-running experiments will create fresh result directories according to each config's `experiment.output_dir`.


## Tasks

The top-level `data/` directory contains the 169 tasks used in the paper experiments: 92 `compile_error`, 45 `logic_error`, and 32 `memory_crash` instances.

## Submodule

After cloning this artifact, initialize the full CUDABeaver submodule:

```bash
git submodule update --init --recursive
```

The top-level artifact is sufficient for the commands below. The submodule is included so reviewers can inspect the full original repository state and compare against upstream.

## Requirements

Hardware/software assumptions:

- Linux machine with an NVIDIA GPU.
- CUDA toolkit with `nvcc` available on `PATH`.
- Python 3.10+.
- PyTorch with CUDA support.
- Optional but recommended for memory-crash analysis: NVIDIA Compute Sanitizer (`compute-sanitizer`).
- For local model serving: an OpenAI-compatible vLLM/SGLang endpoint.
- For cloud models: an OpenAI-compatible API key and endpoint.

The dataset bundles CUTLASS, ThunderKittens headers, and KernelBench support files used by the harness.

## Environment Setup

From the artifact root:

```bash
cd ~/CUDAMender
bash setup_env.sh
source .venv/bin/activate
bash scripts/prepare_benchmark.sh
```

Alternative conda path:

```bash
conda env create -f environment.yml
conda activate cuda-debugger-bench
bash scripts/post_conda_setup.sh
bash scripts/prepare_benchmark.sh
```

`prepare_benchmark.sh` materializes the flat dataset layout used by the harness. It is idempotent and can be rerun.

Generated reference caches are intentionally not shipped. Performance-gated runs create `data/evaluation/references_cache/<gpu_signature>/` lazily on the reviewer's own GPU/CUDA stack when a reference baseline is first needed. Applied-kernels runs may also create `harness/.cache/ref_builds/` while compiling CUDA references; that directory is generated scratch state.

## Sanity Checks

Run tests that do not need an LLM endpoint:

```bash
pytest tests/test_smoke.py -q
pytest tests/test_agent_tools.py tests/test_task_registry.py -q
```

If CUDA or PyTorch is not configured correctly, GPU-dependent tests may fail before any LLM calls are made.

## Configuring LLM Endpoints

Experiment configs use an `api:` block. Before running, edit the selected config or create a copy and change:

```yaml
api:
  backend: openai        # or vllm
  model: <model-name>
  base_url: <https-or-local-openai-compatible-endpoint>/v1
  api_key_env: <ENV_VAR_NAME>
```

Then export the key. For local vLLM/SGLang, any non-empty placeholder is usually enough:

```bash
export OPENAI_API_KEY=<your-api-key>
# or for local vLLM:
export LOCAL_VLLM_KEY=dummy
```

Configs use either localhost defaults or placeholder OpenAI-compatible endpoints. Replace them with your endpoint before use.

## Running a Single Experiment

General form:

```bash
python -m harness.run_experiment \
  --config configs/agent/<config>.yaml \
  --v2-task-list benchmark/task_list.json \
  --mode debug
```

Examples:

```bash
# GPT-style blackboard agent on logic-error variants
python -m harness.run_experiment \
  --config configs/agent/logic_variant_agent_gpt5_4_low_pgate095_temp08_gpu2.yaml \
  --v2-task-list benchmark/task_list.json \
  --mode debug

# Qwen memory-crash guided repair
python -m harness.run_experiment \
  --config configs/agent/memcrash_basic_agent_qwen36_27b_converge_30007_no_maxtok_gpu7.yaml \
  --v2-task-list benchmark/task_list.json \
  --mode debug

# Iterative baseline
python -m harness.run_experiment \
  --config configs/iter/gpt5_4_iterative_memcrash32_low.yaml \
  --v2-task-list benchmark/task_list.json \
  --mode debug
```

The agent workflow is implemented mainly in:

- `harness/agent_loop.py`
- `harness/agent_tools.py`
- `harness/cudeubgger/blackboard.py`
- `harness/cudeubgger/runtime.py`
- `harness/cudeubgger/submission.py`
- `harness/diagnostics.py`

## Reproducing the Main Agent Comparisons

The configs used for the agent behavior analysis are included under `configs/agent/`, `configs/iter/`, and `configs/repeated/`. The important knobs are:

```yaml
experiment:
  method: agent          # agent / iterative / repeated
  max_iterations: 5
agent:
  workflow: blackboard
  max_patch_submissions: 5
performance_gate:
  enabled: true
  min_speedup: 0.95      # or 0.7 / 1.0 depending on the run
```

Run each selected config with `harness.run_experiment`; the harness writes per-task manifests, raw build logs, error-code snapshots, summaries, and tool traces into the configured output directory.

## Analyzing Results

After experiments finish, aggregate results with:

```bash
python -m analysis.build_master --results-root outputs/iter --out-dir analysis_out
python -m analysis.pivot_tables --master analysis_out/master.csv --out-dir analysis_out
```

For the tool-use and submission-attempt report used during artifact preparation:

```bash
python analysis/agent_behavior_report.py --output analysis_out/agent_behavior_analysis.md
```

That script expects the specific result directories named in the script. If you regenerate experiments into different output directories, update the run list at the top of `analysis/agent_behavior_report.py`.

## Reproducibility Notes

- This artifact excludes logs and prior result directories by design.
- Generated caches such as `data/evaluation/references_cache/` and `harness/.cache/ref_builds/` are excluded and should be regenerated locally.
- Configs may need endpoint and GPU-ID edits for your machine.
- Performance-gated results can vary with GPU model, CUDA version, timing backend, and model serving stack.
- `write_solution` submissions automatically trigger build/test feedback; explicit `build_only` and `test_only` tool calls are optional diagnostics.
- Compute Sanitizer is used through the constrained `run_compute_sanitizer` tool, not arbitrary shell execution.
