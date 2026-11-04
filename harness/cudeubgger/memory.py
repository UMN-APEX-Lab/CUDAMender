"""Typed debugging memory for the CUDAMender blackboard workflow.

This module holds the deterministic pieces of the context construction

    P_R = V_R(B) ∪ A_R(B) ∪ S(M, B) ∪ L_K(M)

that do not belong to the role transcript itself:

  - ``attempt_summary`` / ``attempt_memory``: the cross-attempt part of
    S(M, B). Old attempts T_old are folded into typed AttSum records; the
    latest completed attempt T_last stays verbatim because the next role
    must react to its validation feedback.
  - ``select_anchors``: A_R(B), the global CUDA evidence anchors chosen by
    evidence type and agent role (line-level localizations, kernel
    signatures, launch expressions, sanitizer output, verified root causes,
    failed patch intents, validation outcomes, performance-gate feedback).
  - ``parallel_structure`` / ``parallel_structure_diff``: the intended GPU
    execution strategy of a candidate and how a submitted patch changed it,
    used to turn a performance-gate failure into debugging evidence.

The in-iteration part of S(M, B) (InvSum) and the local window L_K live in
``harness.context_manager``.
"""

from __future__ import annotations

import json
import re

_KERNEL_RE = re.compile(r"__global__")
_LAUNCH_RE = re.compile(r"[A-Za-z_][\w:<>,\s]*?<<<[^;]*?>>>")
_SHARED_RE = re.compile(r"__shared__|extern\s+__shared__")
_SYNC_RE = re.compile(r"__syncthreads\s*\(|__syncwarp\s*\(|\.sync\s*\(\s*\)|grid\.sync")
_TENSOR_CORE_RE = re.compile(
    r"\bwmma::|\bmma\.sync\b|\bmma_sync\b|\bwgmma\b|\btcgen05\b|__hmma|"
    r"\bldmatrix\b|cutlass::gemm|cute::gemm|\bmma\s*\("
)
_SHUFFLE_RE = re.compile(r"__shfl\w*\s*\(|__reduce_\w+_sync\s*\(")
_ATOMIC_RE = re.compile(r"\batomic[A-Z]\w*\s*\(")
_SERIAL_LAUNCH_RE = re.compile(r"<<<\s*1\s*,\s*1\s*[,>]")


def _clip(text, max_chars: int) -> str:
    text = "" if text is None else str(text)
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + " ... " + text[-half:]


# ---------------------------------------------------------------------------
# Parallel execution strategy
# ---------------------------------------------------------------------------


def kernel_signatures(code: str, max_items: int = 8) -> list[dict]:
    """``__global__`` signatures with 1-indexed line numbers."""
    lines = (code or "").splitlines()
    out: list[dict] = []
    for idx, line in enumerate(lines):
        if not _KERNEL_RE.search(line):
            continue
        sig = line.strip()
        j = idx
        while ")" not in sig and j + 1 < len(lines) and j - idx < 4:
            j += 1
            sig += " " + lines[j].strip()
        sig = sig.split("{")[0].strip()
        out.append({"line": idx + 1, "signature": _clip(sig, 240)})
        if len(out) >= max_items:
            break
    return out


def launch_expressions(code: str, max_items: int = 8) -> list[dict]:
    """Kernel launch expressions (``k<<<grid, block, smem>>>``) with lines."""
    out: list[dict] = []
    for idx, line in enumerate((code or "").splitlines(), start=1):
        for match in _LAUNCH_RE.finditer(line):
            out.append({"line": idx, "launch": _clip(match.group(0).strip(), 200)})
        if len(out) >= max_items:
            break
    return out[:max_items]


def parallel_structure(code: str) -> dict:
    """Coarse fingerprint of a candidate's GPU execution strategy."""
    code = code or ""
    return {
        "kernels": len(kernel_signatures(code, max_items=64)),
        "launches": [item["launch"] for item in launch_expressions(code, max_items=64)],
        "shared_memory_decls": len(_SHARED_RE.findall(code)),
        "barriers": len(_SYNC_RE.findall(code)),
        "tensor_core_ops": len(_TENSOR_CORE_RE.findall(code)),
        "warp_shuffles": len(_SHUFFLE_RE.findall(code)),
        "atomics": len(_ATOMIC_RE.findall(code)),
        "serial_launch": bool(_SERIAL_LAUNCH_RE.search(code)),
    }


def parallel_structure_diff(old_code: str, new_code: str) -> list[str]:
    """Describe how ``new_code`` changed the parallel behaviour of ``old_code``.

    The statements are evidence for the next repair round after a
    performance-gate failure, not proofs: they point the Verifier and Coder
    at structural changes that commonly cost GPU throughput.
    """
    old = parallel_structure(old_code)
    new = parallel_structure(new_code)
    notes: list[str] = []
    if new["serial_launch"] and not old["serial_launch"]:
        notes.append(
            "a kernel is now launched with a single block and a single thread "
            "(<<<1, 1>>>), which serializes the work"
        )
    if old["launches"] != new["launches"]:
        removed = [l for l in old["launches"] if l not in new["launches"]]
        added = [l for l in new["launches"] if l not in old["launches"]]
        if removed or added:
            notes.append(
                "launch configuration changed: "
                f"removed {removed[:3] or '[]'}, added {added[:3] or '[]'}"
            )
    if old["kernels"] and new["kernels"] < old["kernels"]:
        notes.append(f"__global__ kernels decreased {old['kernels']} -> {new['kernels']}")
    if old["shared_memory_decls"] > new["shared_memory_decls"]:
        notes.append(
            "__shared__ declarations decreased "
            f"{old['shared_memory_decls']} -> {new['shared_memory_decls']} "
            "(block-level cooperation through shared memory was reduced)"
        )
    if old["tensor_core_ops"] > new["tensor_core_ops"]:
        notes.append(
            "Tensor Core instructions decreased "
            f"{old['tensor_core_ops']} -> {new['tensor_core_ops']}"
        )
    if old["warp_shuffles"] > new["warp_shuffles"]:
        notes.append(
            "warp-level shuffles/reductions decreased "
            f"{old['warp_shuffles']} -> {new['warp_shuffles']}"
        )
    if new["barriers"] > old["barriers"]:
        notes.append(
            f"barriers increased {old['barriers']} -> {new['barriers']} "
            "(extra synchronization can serialize warps)"
        )
    if new["atomics"] > old["atomics"]:
        notes.append(
            f"atomic operations increased {old['atomics']} -> {new['atomics']} "
            "(possible contention or serialized reduction)"
        )
    return notes


def describe_strategy(code: str) -> str:
    """One-line description of the candidate's execution strategy."""
    s = parallel_structure(code)
    parts = [f"{s['kernels']} kernel(s)"]
    if s["launches"]:
        parts.append("launches " + "; ".join(s["launches"][:3]))
    if s["shared_memory_decls"]:
        parts.append(f"{s['shared_memory_decls']} __shared__ decl(s)")
    if s["barriers"]:
        parts.append(f"{s['barriers']} barrier(s)")
    if s["tensor_core_ops"]:
        parts.append("Tensor Core path")
    if s["warp_shuffles"]:
        parts.append("warp-shuffle reduction")
    if s["atomics"]:
        parts.append(f"{s['atomics']} atomic(s)")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Attempt memory (S_T)
# ---------------------------------------------------------------------------


def _outcome(attempt: dict) -> str | None:
    cls = attempt.get("classification") or {}
    return cls.get("category") or cls.get("five_category") or cls.get("status")


def _failure_reason(attempt: dict, max_chars: int = 400) -> str | None:
    cls = attempt.get("classification") or {}
    if cls.get("status") == "passed" or cls.get("five_category") == "passed":
        return None
    gate = attempt.get("performance_gate") or {}
    if gate and gate.get("passed") is False:
        perf = attempt.get("performance") or {}
        reason = (
            f"performance gate failed: speedup={perf.get('speedup')} "
            f"< {gate.get('min_speedup')}"
        )
        changes = attempt.get("parallel_changes") or []
        if changes:
            reason += "; " + "; ".join(changes[:3])
        return _clip(reason, max_chars)
    detail = (
        cls.get("logic_error_detail")
        or attempt.get("test_output_excerpt")
        or attempt.get("build_output_excerpt")
        or cls.get("error_type")
    )
    if not detail:
        return None
    lines = [ln.strip() for ln in str(detail).splitlines() if ln.strip()]
    key = [
        ln for ln in lines
        if any(t in ln.lower() for t in ("error", "assert", "fail", "mismatch", "illegal", "expected"))
    ]
    return _clip(" | ".join((key or lines)[:3]), max_chars)


def attempt_summary(attempt: dict) -> dict:
    """AttSum(t): the typed summary of one completed repair attempt."""
    located = attempt.get("located") or {}
    perf = attempt.get("performance") or {}
    gate = attempt.get("performance_gate") or {}
    summary = {
        "iteration": attempt.get("iteration"),
        "attempted_fix": _clip(attempt.get("fix_direction") or attempt.get("rationale"), 300),
        "verified_localization": (
            {
                "file": located.get("file"),
                "line": located.get("line"),
                "root_cause": _clip(
                    located.get("bug_description") or located.get("root_cause"), 240
                ),
            }
            if located else None
        ),
        "patch_intent": _clip(attempt.get("patch_intent") or attempt.get("rationale"), 300),
        "validation_outcome": _outcome(attempt),
        "failure_reason": _failure_reason(attempt),
    }
    if gate:
        summary["performance"] = {
            "speedup": perf.get("speedup"),
            "min_speedup": gate.get("min_speedup"),
            "passed": gate.get("passed"),
        }
    if attempt.get("route"):
        summary["routing_decision"] = attempt["route"]
    return summary


def attempt_memory(attempts: list[dict], latest_chars: int = 6000) -> str:
    """Render S_T: AttSum for T_old plus the latest attempt T_last verbatim."""
    if not attempts:
        return "[]"
    older = [attempt_summary(a) for a in attempts[:-1]]
    latest = attempts[-1]
    latest_text = json.dumps(latest, indent=2, ensure_ascii=False, default=str)
    if len(latest_text) > latest_chars:
        half = latest_chars // 2
        latest_text = latest_text[:half] + "\n... [truncated] ...\n" + latest_text[-half:]
    blocks = []
    if older:
        blocks.append(
            "[Compressed earlier attempts — typed AttSum records]\n"
            + json.dumps(older, indent=2, ensure_ascii=False, default=str)
        )
    blocks.append("[Latest completed attempt — verbatim]\n" + latest_text)
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Global CUDA evidence anchors (A_R)
# ---------------------------------------------------------------------------

# Anchor types each role keeps explicit across iterations.
ROLE_ANCHOR_TYPES = {
    "analyzer": (
        "kernel_signatures", "launch_expressions", "prior_localizations",
        "failed_patch_intents", "validation_outcomes",
    ),
    "executor": (
        "kernel_signatures", "launch_expressions", "localizations",
        "sanitizer_output", "validation_outcomes",
    ),
    "verifier": (
        "localizations", "sanitizer_output", "verified_root_causes",
        "failed_patch_intents", "validation_outcomes", "performance_feedback",
    ),
    "coder": (
        "kernel_signatures", "launch_expressions", "verified_root_causes",
        "failed_patch_intents", "performance_feedback",
    ),
    "router": (
        "verified_root_causes", "failed_patch_intents", "validation_outcomes",
        "performance_feedback",
    ),
}


def _localization(item: dict) -> dict:
    return {
        "source": item.get("source"),
        "file": item.get("file"),
        "line": item.get("line"),
        "root_cause": _clip(item.get("root_cause") or item.get("bug_description"), 200),
    }


def _sanitizer_anchors(board) -> list[dict]:
    results = (board.runtime_trace or {}).get("tool_results") or []
    out = []
    for r in results:
        if r.get("tool") != "run_compute_sanitizer":
            continue
        excerpt = r.get("content_excerpt") or ""
        key = [
            ln.strip() for ln in excerpt.splitlines()
            if any(t in ln for t in ("=========", "Invalid", "Hazard", "Uninitialized", "ERROR SUMMARY", "at 0x", "by thread"))
        ]
        out.append({
            "mode": (r.get("meta") or {}).get("tool") or (r.get("args") or {}).get("tool"),
            "ok": r.get("ok"),
            "findings": _clip("\n".join(key[:8]) or excerpt, 700),
        })
    return out[-2:]


def select_anchors(board, role: str, code: str) -> dict:
    """A_R(B): role-selected CUDA evidence anchors from board and candidate."""
    wanted = ROLE_ANCHOR_TYPES.get(role, ())
    anchors: dict = {}
    if "kernel_signatures" in wanted:
        anchors["kernel_signatures"] = kernel_signatures(code)
    if "launch_expressions" in wanted:
        anchors["launch_expressions"] = launch_expressions(code)
    if "localizations" in wanted:
        anchors["localizations"] = [_localization(s) for s in board.suspected_lines[-8:]]
    if "prior_localizations" in wanted:
        anchors["prior_localizations"] = [
            _localization(s) for s in board.prior_suspected_lines[-8:]
        ]
    if "sanitizer_output" in wanted:
        anchors["sanitizer_output"] = _sanitizer_anchors(board)
    if "verified_root_causes" in wanted:
        causes = [
            _localization(s) for s in board.suspected_lines if s.get("source") == "verifier"
        ]
        for diag in board.previous_diagnoses[-3:]:
            for item in diag.get("line_level_bug_report") or []:
                if isinstance(item, dict):
                    causes.append(_localization({**item, "source": "previous_verifier"}))
        anchors["verified_root_causes"] = causes[-8:]
    if "failed_patch_intents" in wanted:
        anchors["failed_patch_intents"] = [
            {
                "iteration": a.get("iteration"),
                "fix_direction": a.get("attempted_fix"),
                "patch_intent": a.get("patch_intent"),
            }
            for a in board.attempts
            if a.get("validation_outcome") not in (None, "passed")
        ][-5:]
    if "validation_outcomes" in wanted:
        anchors["validation_outcomes"] = [
            {
                "iteration": a.get("iteration"),
                "outcome": a.get("validation_outcome"),
                "failure_reason": a.get("failure_reason"),
            }
            for a in board.attempts
        ][-5:]
    if "performance_feedback" in wanted:
        anchors["performance_feedback"] = board.performance_feedback()
        anchors["execution_strategy"] = board.execution_strategy
    return {k: v for k, v in anchors.items() if v}


def anchors_block(board, role: str, code: str, max_chars: int = 6000) -> str:
    anchors = select_anchors(board, role, code)
    if not anchors:
        return "{}"
    return _clip(json.dumps(anchors, indent=2, ensure_ascii=False, default=str), max_chars)
