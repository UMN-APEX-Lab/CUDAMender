"""Adapter for `applied_kernels_class` tasks (Class 2 def.py-based).

These tasks have empty `build_command` / `test_command` in input.json — the
build is deferred to a pybind11 + torch.utils.cpp_extension pipeline that
needs `def.py` (function signature, input/output generators, reference fn,
tolerances) and `reference.cu`. The harness's `compiler.py` detects these
via `is_applied_kernels_task(input_meta)` and dispatches here.

Output contract (matches `harness/classifier.py` parsing expectations):
- BUILD_OK={True,False} on first line
- "Kernel time: N.NNNN ms"
- "Speedup: N.NNNNx"  (perf-gate trigger for `perf_broken` instances)
- exit code 0/2/3 for pass/build_fail/logic_fail
"""
import os
import subprocess
import sys
from pathlib import Path

from .subprocess_util import run_pg


def is_applied_kernels_task(input_meta: dict) -> bool:
    """Detect Class 2 def.py-based tasks via `level` field."""
    return input_meta.get("level") == "applied_kernels_class"


def applied_kernels_build(task_id: str, workdir: Path, timeout: int = 300) -> tuple[bool, str]:
    """Class 2 build is deferred to the runner during test. Just sanity-check
    that `solution.cu` and `def.py` are present so we fail fast on missing files."""
    sol = workdir / "solution.cu"
    if not sol.exists():
        return False, "BUILD_OK=False\n[BUILD OUTPUT]\nsolution.cu missing"
    defpy = workdir / "def.py"
    if not defpy.exists():
        return False, f"BUILD_OK=False\n[BUILD OUTPUT]\ndef.py missing in {workdir}"
    return True, "BUILD_OK=True\n\n[BUILD OUTPUT]\n(deferred to applied-kernels runner)"


def applied_kernels_test(task_id: str, workdir: Path, timeout: int = 300) -> tuple[int | None, str]:
    """Run the def.py-based eval pipeline in a subprocess (isolated CUDA context).

    Constructs a TaskConfig in-memory from the workdir layout (no task.yaml
    needed) and calls applied_kernels_runner.run_task. Prints `Speedup: X` so
    the standard classifier's perf-gate works.
    """
    sol_path = (workdir / "solution.cu").resolve()
    runner_root = Path(__file__).resolve().parent.parent  # = code package root

    script = f"""
import sys, json
sys.path.insert(0, {str(runner_root)!r})
from harness.applied_kernels_runner.runner import run_task
from harness.applied_kernels_runner.task_schema import (
    TaskConfig, RunnerConfig, BuildConfig, ExecuteConfig,
    CorrectnessConfig, PerformanceConfig, HardwareConfig,
)
config = TaskConfig(
    task_id={task_id!r},
    name={task_id!r},
    task_class=2,
    domain="ml",
    hardware=HardwareConfig(min_sm=80),
    runner=RunnerConfig(
        backend="class2_defpy",
        workdir={str(workdir.resolve())!r},
        solution_file="solution.cu",
        timeout_sec={int(timeout)},
    ),
    correctness=CorrectnessConfig(
        mode="tensor_compare",
        tolerances={{"atol": 5e-2, "rtol": 5e-2}},
    ),
    performance=PerformanceConfig(enabled=True),
)
tr = run_task(config, {str(sol_path)!r}, measure_perf=True, verbose=False)
if not tr.compiled:
    print(f"BUILD_FAIL: {{tr.error_msg or 'compile failed'}}")
    sys.exit(2)
if not tr.correct:
    print(f"LOGIC_FAIL: {{tr.error_msg or 'tensor mismatch'}}")
    sys.exit(3)
sp = tr.speedup if tr.speedup is not None else 0.0
lat = tr.latency_mean_ms or 0.0
print(f"Kernel time: {{lat:.4f}} ms")
print(f"Speedup: {{sp:.4f}}x")
sys.exit(0)
"""
    try:
        proc = run_pg(
            [sys.executable, "-c", script],
            cwd=str(workdir),
            env=os.environ.copy(),
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None, f"[TEST OUTPUT]\n[TEST TIMEOUT] after {timeout}s"
    out = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
    return proc.returncode, f"[TEST OUTPUT]\n{out}"
