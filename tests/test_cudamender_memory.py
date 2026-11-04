#!/usr/bin/env python3
"""Unit tests for CUDAMender's typed debugging memory and context construction.

Pure logic — no LLM, GPU, or nvcc required.
"""

import json

from harness.context_manager import get_budget_nudge
from harness.context_summary import rule_based_summary
from harness.cudeubgger.common import (
    ErrorBlackboard,
    _LoopState,
    _attempt_history_prompt,
    _performance_feedback_text,
)
from harness.cudeubgger.memory import (
    attempt_summary,
    kernel_signatures,
    launch_expressions,
    parallel_structure_diff,
    select_anchors,
)
from harness.cudeubgger.prompts import (
    CODER_SYSTEM_PROMPT,
    with_performance_target,
)

KERNEL = """\
#include <cuda_runtime.h>
#define TILE 32
__global__ void reduce(const float* x, float* out, int n) {
  __shared__ float sdata[TILE];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  sdata[threadIdx.x] = i < n ? x[i] : 0.0f;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd(out, sdata[0]);
}
void launch(const float* x, float* out, int n) {
  reduce<<<(n + TILE - 1) / TILE, TILE>>>(x, out, n);
}
"""

SERIAL = """\
__global__ void reduce(const float* x, float* out, int n) {
  float acc = 0.0f;
  for (int i = 0; i < n; ++i) acc += x[i];
  *out = acc;
}
void launch(const float* x, float* out, int n) {
  reduce<<<1, 1>>>(x, out, n);
}
"""


def _attempt(iteration, outcome, **extra):
    attempt = {
        "iteration": iteration,
        "rationale": f"rationale {iteration}",
        "patch_intent": f"intent {iteration}",
        "fix_direction": f"fix {iteration}",
        "located": {"file": "solution.cu", "line": 7, "bug_description": "missing barrier"},
        "classification": {"category": outcome, "five_category": outcome},
        "test_output_excerpt": "check 3 FAILED: expected 1.0 got 0.5",
    }
    attempt.update(extra)
    return attempt


# ---------------------------------------------------------------------------
# Parallel structure / performance evidence
# ---------------------------------------------------------------------------


def test_kernel_signatures_and_launches_carry_line_numbers():
    sigs = kernel_signatures(KERNEL)
    assert sigs[0]["line"] == 3
    assert "reduce(const float* x, float* out, int n)" in sigs[0]["signature"]
    launches = launch_expressions(KERNEL)
    assert launches[0]["line"] == 15
    assert "<<<(n + TILE - 1) / TILE, TILE>>>" in launches[0]["launch"]


def test_parallel_structure_diff_flags_serialized_patch():
    notes = " | ".join(parallel_structure_diff(KERNEL, SERIAL))
    assert "<<<1, 1>>>" in notes
    assert "__shared__ declarations decreased" in notes
    assert "launch configuration changed" in notes


def test_performance_feedback_reports_parallel_changes():
    gate = {
        "min_speedup": 0.95,
        "passed": False,
        "intended_strategy": "1 kernel(s)",
        "submitted_strategy": "1 kernel(s)",
        "parallel_changes": parallel_structure_diff(KERNEL, SERIAL),
    }
    text = _performance_feedback_text({"speedup": 0.05}, gate)
    assert "Required speedup: 0.950x" in text
    assert "How the patch changed the expected parallel behavior" in text
    assert "<<<1, 1>>>" in text


def test_prompts_use_configured_performance_target():
    cfg = {"agent": {"performance_gate": {"enabled": True, "min_speedup": 0.7}}}
    prompt = with_performance_target(CODER_SYSTEM_PROMPT, cfg)
    assert "0.70x" in prompt
    assert "<PERF_TARGET>" not in prompt
    assert "0.95x" not in prompt


# ---------------------------------------------------------------------------
# Attempt memory S_T
# ---------------------------------------------------------------------------


def test_attempt_summary_has_typed_fields():
    summary = attempt_summary(_attempt(1, "logic_error"))
    assert summary["attempted_fix"] == "fix 1"
    assert summary["verified_localization"]["line"] == 7
    assert summary["patch_intent"] == "intent 1"
    assert summary["validation_outcome"] == "logic_error"
    assert "expected 1.0 got 0.5" in summary["failure_reason"]


def test_attempt_history_compresses_old_and_keeps_latest_verbatim():
    state = _LoopState()
    state.attempt_history = [
        _attempt(
            1, "logic_error",
            test_output_excerpt="OLD_RAW_LOG\n" * 200 + "assert failed: expected 2 got 3\n",
        ),
        _attempt(2, "logic_error"),
        _attempt(3, "memory_crash", test_output_excerpt="LATEST_RAW_LOG"),
    ]
    text = _attempt_history_prompt(state)
    assert "[Compressed earlier attempts" in text
    assert "[Latest completed attempt — verbatim]" in text
    # Old raw logs are folded into typed summaries; the latest stays raw.
    assert "OLD_RAW_LOG" not in text
    assert "expected 2 got 3" in text
    assert "LATEST_RAW_LOG" in text
    assert '"iteration": 1' in text and '"iteration": 2' in text


def test_attempt_summary_explains_perf_gate_failure():
    summary = attempt_summary(_attempt(
        1, "perf_below_threshold",
        performance={"speedup": 0.1},
        performance_gate={"passed": False, "min_speedup": 0.95},
        parallel_changes=["a kernel is now launched with a single block"],
    ))
    assert summary["performance"] == {"speedup": 0.1, "min_speedup": 0.95, "passed": False}
    assert "performance gate failed" in summary["failure_reason"]
    assert "single block" in summary["failure_reason"]


# ---------------------------------------------------------------------------
# Blackboard records, views, anchors
# ---------------------------------------------------------------------------


def _board_with_history() -> ErrorBlackboard:
    board = ErrorBlackboard(task_stem="cuda_demo")
    board.error_type = "logic_error"
    board.add_suspected_line({
        "source": "verifier", "file": "solution.cu", "line": 9,
        "root_cause": "reduction reads past the active half",
        "evidence": ["test mismatch"],
    })
    board.verifier_payload = {"error_type": "logic_error", "next_fix": "guard s"}
    board.next_fix = "guard s"
    board.record_attempt(attempt_summary(_attempt(
        1, "perf_below_threshold",
        performance={"speedup": 0.1},
        performance_gate={"passed": False, "min_speedup": 0.95},
    )))
    board.record_route({"next_action": "reanalyze", "diagnosis": "wrong_strategy"}, 1)
    board.record_report("analyzer", {
        "summary": "s", "key_points": [], "ruled_out": ["launch bounds"],
        "execution_strategy": "1D grid, shared-memory tree reduction",
    })
    return board


def test_begin_analysis_round_archives_instead_of_dropping():
    board = _board_with_history()
    board.begin_analysis_round()
    assert board.suspected_lines == []
    assert board.next_fix is None
    assert board.prior_suspected_lines[0]["line"] == 9
    assert board.previous_diagnoses[0]["next_fix"] == "guard s"
    # Attempt and routing records persist across rounds.
    assert board.attempts[0]["routing_decision"]["next_action"] == "reanalyze"
    assert board.ruled_out == ["launch bounds"]
    assert board.execution_strategy == "1D grid, shared-memory tree reduction"


def test_analyzer_view_matches_retrieval_policy():
    board = _board_with_history()
    board.begin_analysis_round()
    board.candidate_meta = {"lines": 16}
    view = json.loads(board.view_for("analyzer"))
    assert view["candidate_meta"] == {"lines": 16}
    assert view["prior_suspected_lines"][0]["line"] == 9
    assert view["failed_fix_directions"][0]["fix_direction"] == "fix 1"
    assert "static_flags" not in view
    assert "executor_report" not in view


def test_verifier_and_coder_views_carry_performance_constraints():
    board = _board_with_history()
    verifier = json.loads(board.view_for("verifier"))
    assert verifier["performance_feedback"]["speedup"] == 0.1
    assert verifier["failed_fix_directions"]
    assert verifier["execution_strategy"].startswith("1D grid")
    coder = json.loads(board.view_for("coder"))
    assert coder["performance_constraints"]["execution_strategy"].startswith("1D grid")
    assert coder["latest_failed_patch"]["outcome"] == "perf_below_threshold"


def test_router_view_has_diagnosis_attempt_and_alert():
    board = _board_with_history()
    board.repeated_failure_alert = "REPEATED FAILURE ALERT: same signature"
    view = json.loads(board.view_for("router"))
    assert view["current_diagnosis"]["next_fix"] == "guard s"
    assert view["latest_attempt"]["patch_intent"] == "intent 1"
    assert view["repeated_failure_alert"].startswith("REPEATED FAILURE ALERT")
    assert view["routing_decisions"][0]["diagnosis"] == "wrong_strategy"


def test_anchors_are_selected_by_role():
    board = _board_with_history()
    coder = select_anchors(board, "coder", KERNEL)
    assert coder["kernel_signatures"][0]["line"] == 3
    assert coder["launch_expressions"]
    assert coder["verified_root_causes"][0]["line"] == 9
    assert coder["failed_patch_intents"][0]["patch_intent"] == "intent 1"
    assert "sanitizer_output" not in coder
    router = select_anchors(board, "router", KERNEL)
    assert "kernel_signatures" not in router
    assert router["validation_outcomes"][0]["outcome"] == "perf_below_threshold"


def test_executor_anchors_keep_sanitizer_findings():
    board = ErrorBlackboard(task_stem="cuda_demo")
    board.runtime_trace = {"tool_results": [{
        "tool": "run_compute_sanitizer", "ok": False, "meta": {"tool": "memcheck"},
        "content_excerpt": (
            "noise\n========= Invalid __global__ write of size 4 bytes\n"
            "=========     by thread (31,0,0) in block (2,0,0)\nmore noise\n"
        ),
    }]}
    anchors = select_anchors(board, "executor", KERNEL)
    findings = anchors["sanitizer_output"][0]["findings"]
    assert "Invalid __global__ write" in findings
    assert "by thread (31,0,0)" in findings
    assert "noise" not in findings.replace("=========", "")


# ---------------------------------------------------------------------------
# Type-aware summarization and budget control
# ---------------------------------------------------------------------------


def test_source_summary_preserves_signatures_launches_and_line_numbers():
    numbered = "\n".join(
        f"{i:5d} | {line}" for i, line in enumerate(KERNEL.splitlines() * 20, start=1)
    )
    out = rule_based_summary(numbered, "code", 800)
    assert out.startswith("[ctx-compressed code (typed)")
    assert "    3 | __global__ void reduce" in out
    assert "<<<(n + TILE - 1) / TILE, TILE>>>" in out
    assert "#define TILE 32" in out


def test_test_log_summary_preserves_expected_vs_actual():
    log = "progress\n" * 300 + "check 7 FAILED: expected 4.0 got 3.5\n" + "progress\n" * 300
    out = rule_based_summary(log, "test_log", 400)
    assert "expected 4.0 got 3.5" in out
    assert "progress" not in out


def test_force_nudge_requests_localization_for_diagnosis_roles():
    for role in ("analyzer", "executor"):
        nudge = get_budget_nudge("force", role=role)
        assert "locate_bug" in nudge and "final_report" in nudge
    assert "write_solution" in get_budget_nudge("force", role="coder")
