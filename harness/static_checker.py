"""Anti-cheat static checker for v2 debug solutions.

Reduced from the original FrontierKernel-style generation-anti-cheat to a
debug-benchmark-appropriate policy. As a DEBUG benchmark, we trust the
testbench (correctness + perf-gate ≥0.3x) to be the real judge. Models
that legitimately fix a bug by switching to cuBLAS/cuFFT/CUTLASS and
producing fast correct output are accepted — that's a valid debug
strategy, not cheating, in this context.

What we still block (genuine sandbox-escape / out-of-band work):
  - dlopen / dlsym / dlfcn (runtime library loading)
  - NVRTC (runtime compilation)
  - cuLibraryLoad / cuModuleLoad / cuModuleLoadData (driver-API module loading)
  - system() / popen() / exec*() (shell escape)
  - TRIVIAL stub solutions (<5 non-empty non-comment lines)

What we no longer block (was generation-style; over-strict for debug):
  - CUTLASS / cuBLAS / cuDNN / cuFFT headers and APIs
  - flash_attn / flash_fwd / flash_bwd / layernorm headers
  - torch:: / at:: / c10:: PyTorch C++ APIs
  - try/catch blocks
  - "must contain __global__ kernel" structural requirement

This change was made in 2026-04-25 after analysis of the v2 results showed
that ~73 of 74 "stuck" tasks were stuck on these generation-style rules
rather than on real debug difficulty.
"""
import re
from dataclasses import dataclass, field


@dataclass
class CheckResult:
    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# Anti-sandbox-escape / out-of-band-work patterns: reject if matched.
# Library use (CUTLASS / cuBLAS / cuFFT / cuDNN / torch::) is INTENTIONALLY
# allowed — for a debug benchmark, switching to a working library call is a
# valid debug strategy. The testbench (correctness + perf gate) is the judge.
BLOCKED = [
    # Dynamic library loading (could load arbitrary .so at runtime)
    (r"#include\s*[<\"]dlfcn", "Includes dlfcn.h"),
    (r"\bdlopen\s*\(", "Uses dlopen (runtime library loading)"),
    (r"\bdlsym\s*\(", "Uses dlsym (runtime symbol resolution)"),
    # NVRTC runtime compilation (could compile/load arbitrary code)
    (r"#include\s*[<\"]nvrtc", "Includes NVRTC headers"),
    (r"\bnvrtc[A-Z][a-zA-Z_]*\s*\(", "Calls NVRTC API (runtime compilation)"),
    # CUDA Driver API module loading (load PTX/cubin at runtime)
    (r"\bcuLibraryLoad", "Uses CUDA Driver cuLibraryLoad"),
    (r"\bcuModuleLoad", "Uses CUDA Driver cuModuleLoad"),
    (r"\bcuModuleLoadData", "Uses CUDA Driver cuModuleLoadData"),
    # Shell escape (could exec arbitrary binary)
    (r"\bsystem\s*\(", "Uses system() (shell escape)"),
    (r"\bpopen\s*\(", "Uses popen() (shell escape)"),
    (r"\bexecv?p?\s*\(", "Uses exec*() (shell escape)"),
]

# REQUIRED_CUDA structural enforcement is intentionally empty under the
# unified debug-skill policy. Models may legitimately fix a bug by switching
# to a library call, with no user-written __global__ kernel left in the file.
REQUIRED_CUDA: list = []


def _is_cuda_source(filename: str) -> bool:
    return filename.endswith((".cu", ".cuh", ".h"))


def _is_python_solution(filename: str) -> bool:
    return filename.endswith(".py")


def _extract_inline_cuda(py_src: str) -> str:
    """Extract triple-quoted CUDA source strings from a .py solution.
    kernelbench solutions use load_inline(cuda_sources='''...'''). Return the
    concatenated content of all triple-quoted strings, so we can run BLOCKED
    + REQUIRED_CUDA checks against the actual CUDA code.
    """
    chunks = re.findall(r'"{3}(.*?)"{3}', py_src, flags=re.DOTALL)
    chunks += re.findall(r"'{3}(.*?)'{3}", py_src, flags=re.DOTALL)
    return "\n".join(chunks)


def _uses_thrust_algorithm(code: str) -> bool:
    """Solutions that delegate to a thrust algorithm (any_of/find_if/
    transform/reduce/...) are legitimate even without a user-written
    __global__ kernel — thrust generates kernels internally. Detect this
    so we don't reject pure-thrust implementations."""
    return bool(re.search(
        r"\bthrust::(any_of|all_of|none_of|find|find_if|count|count_if|"
        r"transform|reduce|transform_reduce|inclusive_scan|exclusive_scan|"
        r"sort|stable_sort|copy_if|remove_if|unique|partition|for_each|"
        r"min_element|max_element|inner_product)\b",
        code,
    ))


def validate_solution(code: str, solution_file: str = "solution.cu") -> CheckResult:
    res = CheckResult()

    # Trivial-solution guard: reject stub kernels like `__global__ void k() {}`
    # submitted to satisfy structural checks while doing no real work.
    non_trivial_lines = [
        ln.strip() for ln in code.splitlines()
        if ln.strip() and not ln.strip().startswith("//")
    ]
    if len(non_trivial_lines) < 5:
        res.errors.append(
            f"TRIVIAL: solution has only {len(non_trivial_lines)} non-empty/non-comment lines (min 5)"
        )

    # Always run BLOCKED across the full solution text (catches Python-side
    # imports of c10::, torch::, etc., as well as CUDA-side #include cutlass)
    for pat, msg in BLOCKED:
        if re.search(pat, code):
            res.errors.append(msg)

    # For .cu — require __global__ + launch in the file itself,
    # UNLESS the solution legitimately delegates to thrust algorithms.
    if _is_cuda_source(solution_file):
        if not _uses_thrust_algorithm(code):
            for pat, msg in REQUIRED_CUDA:
                if not re.search(pat, code):
                    res.errors.append(msg)
    # For .py — extract embedded triple-quoted strings, treat as CUDA src
    elif _is_python_solution(solution_file):
        inline = _extract_inline_cuda(code)
        if not inline.strip():
            res.warnings.append("No inline CUDA source string found in .py")
        elif not _uses_thrust_algorithm(inline):
            for pat, msg in REQUIRED_CUDA:
                if not re.search(pat, inline):
                    res.errors.append(f"{msg} (in inline cuda_sources)")
            # Also re-run BLOCKED on the inline text — sometimes the import
            # happens inside the cuda_sources """...""" string and the outer
            # Python doesn't show it
            for pat, msg in BLOCKED:
                if re.search(pat, inline):
                    res.errors.append(f"{msg} (inside inline cuda_sources)")

    res.valid = len(res.errors) == 0
    return res
