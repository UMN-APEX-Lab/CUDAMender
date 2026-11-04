# CUDAMender: Automated Agentic Performance-Aware CUDA Program Repair

Official implementation of **CUDAMender: Automated Agentic Performance-Aware CUDA Program Repair**. CUDAMender uses an LLM agent to repair CUDA compilation errors, logic errors, and memory errors with correctness and performance feedback.

## Requirements

- Linux with an NVIDIA GPU and a compatible NVIDIA driver.
- CUDA Toolkit with `nvcc` on `PATH`, a C++ compiler, and `make`.
- Python 3.10+ and CUDA-enabled PyTorch compatible with the installed CUDA Toolkit.
- An OpenAI-compatible LLM endpoint, either a cloud API or a local vLLM/SGLang service.
- NVIDIA Compute Sanitizer (`compute-sanitizer`) is recommended for memory-error diagnosis.

## Installation

```bash
git clone https://github.com/UMN-APEX-Lab/CUDAMender.git
cd CUDAMender
bash setup_env.sh
source .venv/bin/activate
bash scripts/prepare_benchmark.sh
```

The setup script installs the Python dependencies. The benchmark data and required library headers are included in this repository.

Alternatively, use Conda:

```bash
conda env create -f environment.yml
conda activate cuda-debugger-bench
bash scripts/post_conda_setup.sh
bash scripts/prepare_benchmark.sh
```

## Configuration

Copy an agent configuration as a starting point:

```bash
cp configs/agent/logic_variant_agent_gpt5_4_low_pgate095_temp08_gpu2.yaml \
  configs/agent/cudamender.yaml
```

Edit the following fields in `configs/agent/cudamender.yaml` to match your model service and output directory, keeping the remaining settings:

```yaml
api:
  backend: openai
  model: your-model-name
  base_url: https://your-api-endpoint/v1
  api_key_env: OPENAI_API_KEY

experiment:
  error_category: null  # Run all benchmark tasks.
  output_dir: results_cudamender
```

Set the API key in your shell:

```bash
export OPENAI_API_KEY="your-api-key"
```

For a local vLLM/SGLang service, set `api.backend` to `vllm`, use its endpoint (for example, `http://localhost:8000/v1`), and provide the served model name. An unauthenticated local service can use `OPENAI_API_KEY=EMPTY`.

## Running CUDAMender

Run the benchmark on GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 python -m harness.run_experiment \
  --config configs/agent/cudamender.yaml \
  --v2-task-list benchmark/task_list.json \
  --mode debug
```

To run selected tasks, append `--tasks cublas_applications_154` to the command. Task identifiers are listed in [`benchmark/task_list.json`](benchmark/task_list.json).

Results, repaired programs, and execution logs are saved under `experiment.output_dir`, in a separate directory for each run.
