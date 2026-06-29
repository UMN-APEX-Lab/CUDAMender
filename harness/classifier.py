"""
Eight-category classifier for CUDA build/test results.
Aligned with CUDA-Debugger benchmark taxonomy (docs/error_taxonomy.md).

Categories (mutually exclusive, checked in priority order):
  1. environment_dependency_bug — GPU arch mismatch, driver issues, missing system libs
  2. integration_bug            — wrong file layout, harness contract mismatch
  3. compile_error              — code-intrinsic build failure (syntax, type, API misuse)
  4. out_of_memory              — cudaMalloc / OOM at runtime
  5. illegal_memory_access      — segfault, illegal address, double free
  6. timeout                    — execution exceeded time limit
  7. functional_correctness_bug — compiled & ran, wrong result
  8. passed                     — all checks passed
"""

import re

# --- Environment / GPU arch mismatch (checked FIRST to avoid misclassification) ---
ENVIRONMENT_PATTERNS = [
    r"no kernel image is available for execution on the device",
    r"cudaErrorNoKernelImageForDevice",
    r"Failed to query occupancy",
    r"CUDA driver version is insufficient",
    r"cudaErrorInsufficientDriver",
    r"CUDA_ERROR_NO_DEVICE",
    r"CUDA_ERROR_INVALID_DEVICE",
    r"GPU arch mismatch",
    r"the provided PTX was compiled with an unsupported toolchain",
]

# --- Illegal memory access (runtime crash) ---
ILLEGAL_MEMORY_PATTERNS = [
    r"Segmentation fault",
    r"SIGSEGV",
    r"illegal memory access",
    r"cudaErrorIllegalAddress",
    r"double free",
    r"an illegal memory access was encountered",
]

# --- Out of memory ---
OOM_PATTERNS = [
    r"out of memory",
    r"cudaErrorMemoryAllocation",
    r"CUDA_ERROR_OUT_OF_MEMORY",
    r"cudaMalloc failed",
]

# --- Timeout ---
TIMEOUT_PATTERNS = [
    r"timed out",
    r"SIGKILL",
    r"killed",
    r"\[TEST TIMEOUT",
    r"Time limit exceeded",
]


def classify_result(
    build_ok: bool,
    build_output: str,
    test_output: str | None,
    test_returncode: int | None,
    performance_gate: dict | None = None,
) -> dict:
    """
    Classify a single iteration result into one of eight categories.

    Returns dict with keys:
      - five_category: str   (kept for backward compat, maps 8-cat to 5-cat)
      - category: str        (8-category label)
      - error_type: str | None
      - build_ok: bool
      - status: str          (failed | passed)
      - logic_error_category: str | None
      - logic_error_detail: str | None
    """
    combined = (build_output or "") + "\n" + (test_output or "")
    perf_gate = _normalize_performance_gate(performance_gate)

    # --- 1. Environment dependency (check both build and test output) ---
    env_match = _match_patterns(combined, ENVIRONMENT_PATTERNS)
    if env_match:
        return _result(
            category="environment_dependency_bug",
            five_category="logic_error",  # legacy compat
            error_type="environment",
            build_ok=build_ok,
            detail=f"Environment: {env_match}",
        )

    # --- 2. Build failure ---
    if not build_ok:
        # Always classify as compile error when build failed,
        # regardless of whether specific error patterns match.
        # _has_compile_error may miss non-C++ errors (e.g., Python SyntaxError).
        return _result(
            category="buildability_bug",
            five_category="compile_error",
            error_type="compile_only",
            build_ok=False,
            detail=build_output[:500] if build_output else None,
        )

    # --- 3. Build succeeded, check test ---
    if test_returncode == 0 and test_output is not None:
        if (not _match_patterns(test_output, ILLEGAL_MEMORY_PATTERNS)
                and not _match_patterns(test_output, OOM_PATTERNS)
                and not _match_patterns(test_output, TIMEOUT_PATTERNS)):
            # applied-kernels output may include "Speedup:" lines. Treat them
            # as a gate only when YAML performance_gate is enabled.
            kh_speedups = _extract_kh_speedups(test_output)
            if kh_speedups:
                import statistics
                mean_sp = statistics.mean(kh_speedups)
                if perf_gate:
                    threshold = perf_gate["min_speedup"]
                    gate_meta = _gate_meta(
                        threshold, mean_sp, source="applied_kernels_stdout",
                    )
                    if mean_sp < threshold:
                        return _result(
                            category="perf_below_threshold",
                            five_category="logic_error",
                            error_type="perf_gate",
                            build_ok=True,
                            detail=(
                                f"correctness OK but mean speedup {mean_sp:.3f}x "
                                f"< {threshold:.3f}x reference (per-size: {kh_speedups})"
                            ),
                            performance_gate=gate_meta,
                        )
                    return _result(
                        category="passed",
                        five_category="passed",
                        build_ok=True,
                        status="passed",
                        detail=f"mean_speedup={mean_sp:.3f}x  per-size={kh_speedups}",
                        performance_gate=gate_meta,
                    )
                return _result(
                    category="passed",
                    five_category="passed",
                    build_ok=True,
                    status="passed",
                    detail=f"mean_speedup={mean_sp:.3f}x  per-size={kh_speedups}",
                )
            if perf_gate:
                kb_speedup = _extract_kernelbench_speedup(test_output)
                if kb_speedup is not None:
                    gate_meta = _gate_meta(
                        perf_gate["min_speedup"], kb_speedup,
                        source="kernelbench_runtime",
                    )
                    if kb_speedup < perf_gate["min_speedup"]:
                        return _result(
                            category="perf_below_threshold",
                            five_category="logic_error",
                            error_type="perf_gate",
                            build_ok=True,
                            detail=(
                                f"correctness OK but speedup {kb_speedup:.3f}x "
                                f"< {perf_gate['min_speedup']:.3f}x reference"
                            ),
                            performance_gate=gate_meta,
                        )
                    return _result(
                        category="passed",
                        five_category="passed",
                        build_ok=True,
                        status="passed",
                        detail=f"speedup={kb_speedup:.3f}x",
                        performance_gate=gate_meta,
                    )
                if perf_gate.get("fail_on_skipped"):
                    return _result(
                        category="perf_unmeasured",
                        five_category="logic_error",
                        error_type="perf_gate_unmeasured",
                        build_ok=True,
                        detail="correctness OK but speedup was not measured",
                        performance_gate={
                            "enabled": True,
                            "min_speedup": perf_gate["min_speedup"],
                            "passed": False,
                            "skipped": "speedup_not_found",
                        },
                    )
            return _result(
                category="passed",
                five_category="passed",
                build_ok=True,
                status="passed",
            )

    test_output = test_output or ""

    # --- 4. OOM ---
    oom_match = _match_patterns(test_output, OOM_PATTERNS)
    if oom_match:
        return _result(
            category="out_of_memory",
            five_category="memory_crash",
            error_type="test_only",
            build_ok=True,
            detail=f"OOM: {oom_match}",
        )

    # --- 5. Illegal memory access ---
    mem_match = _match_patterns(test_output, ILLEGAL_MEMORY_PATTERNS)
    if mem_match:
        return _result(
            category="illegal_memory_access",
            five_category="memory_crash",
            error_type="test_only",
            build_ok=True,
            detail=f"Memory: {mem_match}",
        )

    # --- 6. Timeout ---
    if _match_patterns(test_output, TIMEOUT_PATTERNS):
        return _result(
            category="timeout",
            five_category="timeout",
            error_type="test_only",
            build_ok=True,
            detail="Execution timed out",
        )

    # --- 7. Functional correctness bug ---
    return _result(
        category="functional_correctness_bug",
        five_category="logic_error",
        error_type="test_only",
        build_ok=True,
        detail=test_output[:500] if test_output else "No test output",
    )


def _result(
    category: str,
    five_category: str,
    build_ok: bool = False,
    error_type: str | None = None,
    status: str = "failed",
    detail: str | None = None,
    performance_gate: dict | None = None,
) -> dict:
    result = {
        "category": category,
        "five_category": five_category,
        "error_type": error_type,
        "build_ok": build_ok,
        "status": status,
        "logic_error_category": category if status == "failed" else None,
        "logic_error_detail": detail,
    }
    if performance_gate is not None:
        result["performance_gate"] = performance_gate
    return result


def _extract_kh_speedups(output: str) -> list[float]:
    """Extract `Speedup: X.XXXXx` numbers from applied-kernels ./test stdout.
    Returns empty list if no Speedup lines found (non-applied-kernels task).
    """
    return [float(m) for m in re.findall(r"Speedup:\s*([\d.]+)x", output or "")]


def _extract_kernelbench_speedup(output: str) -> float | None:
    """Extract ref/candidate speedup from KernelBench eval repr output."""
    text = output or ""
    runtime = _extract_named_float(text, "runtime")
    ref_runtime = _extract_named_float(text, "ref_runtime")
    if runtime is None or ref_runtime is None or runtime <= 0 or ref_runtime <= 0:
        return None
    return ref_runtime / runtime


def _extract_named_float(text: str, name: str) -> float | None:
    match = re.search(rf"\b{name}=(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)", text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _normalize_performance_gate(gate: dict | None) -> dict | None:
    if not gate or not gate.get("enabled", True):
        return None
    min_speedup = gate.get("min_speedup", gate.get("threshold"))
    if min_speedup is None:
        return None
    return {
        "min_speedup": float(min_speedup),
        "fail_on_skipped": bool(gate.get("fail_on_skipped", False)),
    }


def _gate_meta(min_speedup: float, speedup: float, source: str) -> dict:
    return {
        "enabled": True,
        "min_speedup": float(min_speedup),
        "speedup": float(speedup),
        "passed": speedup >= min_speedup,
        "source": source,
    }


def _match_patterns(output: str, patterns: list[str]) -> str | None:
    """Return the first matching line, or None."""
    for line in output.splitlines():
        for pattern in patterns:
            if re.search(pattern, line, re.IGNORECASE):
                return line.strip()[:200]
    return None


def _has_compile_error(output: str) -> bool:
    for line in output.splitlines():
        if re.search(r"\berror\b", line, re.IGNORECASE):
            if not re.search(r"\bwarning\b", line, re.IGNORECASE):
                return True
    return False
