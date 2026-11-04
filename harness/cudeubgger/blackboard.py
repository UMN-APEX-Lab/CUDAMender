"""Blackboard multi-role workflow for Cudeubgger."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from ..agent_tools import ToolEnv, call_tool, schemas_for_config
from . import deps
from .common import (
    ErrorBlackboard,
    _LoopState,
    _RoleResult,
    _FIX_PHASE_TOOLS,
    _accumulate_token_usage,
    _action_to_tool_call,
    _append_action_result_message,
    _attempt_history_prompt,
    _build_ctx_config,
    _chars_per_token,
    _extract_json_action,
    _forced_action_context,
    _full_file_submissions,
    _is_recoverable_coder_stop,
    _join_decision_constraints,
    _json_action_enabled,
    _json_action_request,
    _json_role_action_request,
    _max_coder_non_submission_rounds,
    _max_patch_submissions,
    _maybe_make_summarizer,
    _normalize_structured_report,
    _patch_submission_limit_stop_reason,
    _record_coder_non_submission,
    _repeated_failure_alert,
    _sanitize_assistant_content,
    _scoped_views_enabled,
    _strip_thinking,
    _structured_reports_enabled,
    _token_usage_summary,
    _tried_strategies_prompt,
    _tool_decision_overrides,
    _tool_names,
    _truncate_text,
)
from .memory import (
    anchors_block,
    attempt_summary,
    describe_strategy,
    kernel_signatures,
    launch_expressions,
)
from .prompts import (
    ANALYZER_SYSTEM_PROMPT,
    CODER_SYSTEM_PROMPT,
    EXECUTOR_SYSTEM_PROMPT,
    ROUTE_DECISION_SYSTEM_PROMPT,
    STRUCTURED_REPORT_PROMPT,
    VERIFIER_SYSTEM_PROMPT,
    with_performance_target,
)
from .runtime import (
    _append_assistant_message,
    _auto_submit_after_apply_edit,
    _build_tool_env,
    _call_json_action_or_record_api_error,
    _call_llm_or_record_api_error,
    _dispatch_tool_calls,
    _pre_turn_context_mgmt,
    _seed_broken_start,
    _write_tool_trace,
)
from ..task_registry import load_input_meta, load_prompt_text

logger = logging.getLogger("cuda_debugger_exp")


def _update_blackboard_from_tool(
    board: ErrorBlackboard,
    role_name: str,
    tool_name: str,
    args: dict,
    result,
) -> None:
    if tool_name == "locate_bug" and result.ok and result.meta.get("located"):
        board.add_suspected_line({
            "source": role_name,
            "file": result.meta.get("file"),
            "line": result.meta.get("line"),
            "bug_description": result.meta.get("bug_description"),
            "evidence": result.meta.get("evidence"),
        })
        return

    if tool_name == "run_static_analysis" and result.ok:
        for flag in result.meta.get("flags") or []:
            item = dict(flag)
            item["source"] = role_name
            board.static_flags.append(item)
        return

    if tool_name in {"build_only", "test_only", "run_compute_sanitizer"}:
        board.runtime_trace.setdefault("tool_results", []).append({
            "source": role_name,
            "tool": tool_name,
            "args": args,
            "ok": result.ok,
            "error": result.error,
            "meta": result.meta,
            "content_excerpt": _truncate_text(result.content, 2000),
        })


def _board_block(board: ErrorBlackboard, role: str, config: dict) -> str:
    """Render the board for a role: a scoped projection when
    ``agent.blackboard.scoped_views`` is on, otherwise the full dump."""
    if _scoped_views_enabled(config):
        return board.view_for(role)
    return board.to_prompt_block()


def _force_final_report(
    client,
    config: dict,
    messages: list[dict],
    state: _LoopState,
    task_stem: str,
    input_meta: dict,
    role_name: str,
    structured: bool,
    max_tokens,
    temperature,
    json_response_format: bool,
    asst_content_cap: int,
):
    """Make one constrained call asking the role to summarize now (gap 2).

    Called when a role used all its investigation turns without emitting a
    `final_report`. Returns a structured dict (when structured reports are on),
    a free-text string, or None on API error. Bounded to a single LLM call.
    """
    if structured:
        schema = (
            '{"tool":"final_report","args":{"summary":"<=1 sentence",'
            '"key_points":["<=4 bullets"],"ruled_out":["<=4 rejected hypotheses"]}}'
        )
    else:
        schema = '{"tool":"final_report","args":{"report":"compact evidence summary"}}'
    request = (
        f"FINAL REPORT REQUIRED ({role_name})\n"
        "You have used your investigation turns. Summarize your findings now so "
        "the verifier can act on them. Return exactly one compact JSON object, "
        "nothing else:\n"
        f"{schema}\n"
    )
    request_messages = messages + [{"role": "user", "content": request}]
    resp = _call_json_action_or_record_api_error(
        client, config, request_messages, state, task_stem, input_meta,
        max_tokens=max_tokens, temperature=temperature,
        response_format=json_response_format, role=role_name,
    )
    if resp is None:
        return None
    raw = resp.get("content") or ""
    try:
        action = _extract_json_action(raw)
    except ValueError:
        # Model ignored the schema — keep its prose as a free-text report.
        return _sanitize_assistant_content(raw, max_chars=asst_content_cap)
    tool_name = action.get("tool") or action.get("name") or action.get("action")
    args = action.get("args") if isinstance(action.get("args"), dict) else {}
    if tool_name != "final_report":
        return _sanitize_assistant_content(raw, max_chars=asst_content_cap)
    if structured:
        payload = args.get("report")
        if not isinstance(payload, dict):
            payload = args
        return _normalize_structured_report(payload)
    return str(args.get("report") or "")


def _run_tool_role_agent(
    role_name: str,
    system_prompt: str,
    user_prompt: str,
    allowed_tools: list[str],
    client,
    config: dict,
    env: ToolEnv,
    state: _LoopState,
    board: ErrorBlackboard | None,
    task_stem: str,
    input_meta: dict,
    max_turns: int,
    max_tool_calls: int,
    asst_content_cap: int,
    ctx_cfg=None,
    summarizer=None,
) -> _RoleResult:
    structured = _structured_reports_enabled(config)
    if structured:
        system_prompt = system_prompt + "\n" + STRUCTURED_REPORT_PROMPT
    system_prompt = with_performance_target(system_prompt, config)
    ctx_cfg = ctx_cfg or _build_ctx_config(config.get("agent", {}) or {})
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    reports: list[str] = []
    structured_report: dict | None = None

    def _resolve_report():
        """The role's report: the structured object if one was produced,
        else the joined free-text reports."""
        if structured_report is not None:
            return structured_report
        return "\n\n".join(reports)

    tools = schemas_for_config(allowed_tools)
    tool_decision_max_tokens, tool_decision_temperature = (
        _tool_decision_overrides(config.get("agent", {}) or {})
    )
    use_json_action = _json_action_enabled(config)
    json_response_format = bool(
        (config.get("agent", {}) or {}).get("json_action_response_format", True)
    )

    for turn in range(max_turns):
        # Keep the local window verbatim, fold older tool groups into typed
        # summaries, and attach the budget nudge before every turn.
        _pre_turn_context_mgmt(
            messages, ctx_cfg, summarizer, system_prompt, role=role_name,
        )
        if use_json_action:
            action_request = _json_role_action_request(
                tools,
                role_name,
                allow_final_report=True,
                extra_context=_forced_action_context(state),
                structured_report=structured,
            )
            request_messages = messages + [{"role": "user", "content": action_request}]
            resp = _call_json_action_or_record_api_error(
                client,
                config,
                request_messages,
                state,
                task_stem,
                input_meta,
                max_tokens=tool_decision_max_tokens,
                temperature=tool_decision_temperature,
                response_format=json_response_format,
                role=role_name,
            )
            if resp is None:
                return _RoleResult(stop_reason="api_error")
            raw_content = resp.get("content") or ""
            messages.append({
                "role": "assistant",
                "content": _sanitize_assistant_content(raw_content, max_chars=asst_content_cap),
            })
            try:
                action = _extract_json_action(raw_content)
            except ValueError as e:
                _append_action_result_message(
                    messages,
                    "parse_json_action",
                    f"[invalid JSON action] {e}\nRaw response:\n{raw_content[:1000]}",
                    ok=False,
                )
                continue

            tool_name = action.get("tool") or action.get("name") or action.get("action")
            if tool_name == "final_report":
                args = action.get("args") if isinstance(action.get("args"), dict) else {}
                if structured:
                    # Accept either flat args ({summary, key_points, ruled_out})
                    # or a nested {report: {...}} — whichever the model emits.
                    payload = args.get("report")
                    if not isinstance(payload, dict):
                        payload = args
                    structured_report = _normalize_structured_report(payload)
                else:
                    report = str(args.get("report") or "")
                    if report:
                        reports.append(report)
                break

            try:
                tc = _action_to_tool_call(
                    action,
                    _tool_names(tools),
                    call_id=f"{role_name}_json_action_{turn + 1}",
                )
            except ValueError as e:
                _append_action_result_message(
                    messages,
                    "parse_json_action",
                    f"[invalid JSON action] {e}",
                    ok=False,
                )
                continue

            if state.tool_calls_used >= max_tool_calls:
                _append_action_result_message(
                    messages,
                    tc["function"]["name"],
                    f"[tool budget exhausted: {state.tool_calls_used}/{max_tool_calls}]",
                    ok=False,
                )
                return _RoleResult(
                    report=_resolve_report(),
                    stop_reason="tool_budget_exhausted",
                )

            state.tool_calls_used += 1
            name = tc["function"]["name"]
            args_str = tc["function"]["arguments"]
            try:
                args = json.loads(args_str) if isinstance(args_str, str) and args_str else {}
            except json.JSONDecodeError:
                args = {}
            if state.force_read_code_reason and name != "read_code":
                _append_action_result_message(
                    messages,
                    name,
                    (
                        "[action rejected] A previous apply_edit failed. "
                        "Read the full current code before attempting another "
                        "patch."
                    ),
                    ok=False,
                )
                continue
            if (
                state.force_read_code_reason
                and name == "read_code"
                and state.force_read_code_full_file
                and not bool(args.get("full_file"))
            ):
                _append_action_result_message(
                    messages,
                    name,
                    (
                        "[action rejected] A previous apply_edit failed because "
                        "old_string did not match. Call read_code with "
                        "`{ \"full_file\": true }` before any further patch."
                    ),
                    ok=False,
                )
                continue

            result = call_tool(name, args_str, env, iteration=state.iteration + 1)
            if name == "read_code" and result.ok and state.force_read_code_reason:
                if (
                    not state.force_read_code_full_file
                    or bool(result.meta.get("full_file"))
                ):
                    state.force_read_code_reason = None
                    state.force_read_code_full_file = False
            if board is not None:
                _update_blackboard_from_tool(board, role_name, name, args, result)
            _append_action_result_message(
                messages, name, result.to_message_content(), result.ok
            )
            continue

        resp = _call_llm_or_record_api_error(
            client, config, messages, tools, state, task_stem, input_meta,
            max_tokens=tool_decision_max_tokens,
            temperature=tool_decision_temperature,
            role=role_name,
        )
        if resp is None:
            return _RoleResult(stop_reason="api_error")

        content = _sanitize_assistant_content(
            resp.get("content"), max_chars=asst_content_cap,
        )
        if content:
            reports.append(content)
        _append_assistant_message(messages, resp, asst_content_cap)

        tool_calls = resp.get("tool_calls") or []
        if not tool_calls:
            break

        for tc in tool_calls:
            if state.tool_calls_used >= max_tool_calls:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": (
                        f"[tool budget exhausted: "
                        f"{state.tool_calls_used}/{max_tool_calls}]"
                    ),
                })
                return _RoleResult(
                    report=_resolve_report(),
                    stop_reason="tool_budget_exhausted",
                )

            state.tool_calls_used += 1
            name = tc["function"]["name"]
            args_str = tc["function"]["arguments"]
            try:
                args = json.loads(args_str) if isinstance(args_str, str) and args_str else {}
            except json.JSONDecodeError:
                args = {}
            if state.force_read_code_reason and name != "read_code":
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": (
                        "[action rejected] A previous apply_edit failed. "
                        "Read the full current code before attempting another patch."
                    ),
                })
                continue
            if (
                state.force_read_code_reason
                and name == "read_code"
                and state.force_read_code_full_file
                and not bool(args.get("full_file"))
            ):
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": (
                        "[action rejected] A previous apply_edit failed because "
                        "old_string did not match. Call read_code with "
                        "`{ \"full_file\": true }` before any further patch."
                    ),
                })
                continue

            result = call_tool(name, args_str, env, iteration=state.iteration + 1)
            if name == "read_code" and result.ok and state.force_read_code_reason:
                if (
                    not state.force_read_code_full_file
                    or bool(result.meta.get("full_file"))
                ):
                    state.force_read_code_reason = None
                    state.force_read_code_full_file = False
            if board is not None:
                _update_blackboard_from_tool(board, role_name, name, args, result)
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result.to_message_content(),
            })

    # The role used all its investigation turns without summarizing — force a
    # final report so the verifier always has this role's conclusion (gap 2).
    if structured_report is None and not reports:
        forced = _force_final_report(
            client, config, messages, state, task_stem, input_meta,
            role_name, structured,
            tool_decision_max_tokens, tool_decision_temperature,
            json_response_format, asst_content_cap,
        )
        if isinstance(forced, dict):
            structured_report = forced
        elif forced:
            reports.append(forced)

    return _RoleResult(report=_resolve_report())


def _collect_runtime_observation(
    env: ToolEnv,
    task_stem: str,
    dataset_dir: Path,
    input_meta: dict,
    config: dict,
) -> dict:
    workdir = Path(config["build"]["workdir"])
    build_timeout = int(config["build"]["timeout"])
    test_timeout = int(config["build"]["test_timeout"])
    build_ok, build_output = deps.compile_code(
        env.current_code, task_stem, dataset_dir, input_meta, workdir, build_timeout,
    )
    test_returncode, test_output = None, None
    if build_ok:
        test_returncode, test_output = deps.run_test(
            task_stem, input_meta, workdir, test_timeout,
        )
    classification = deps.classify_result(
        build_ok, build_output, test_output, test_returncode,
    )
    return {
        "build_ok": build_ok,
        "build_output_excerpt": _truncate_text(build_output, 3000),
        "test_returncode": test_returncode,
        "test_output_excerpt": _truncate_text(test_output, 3000),
        "classification": classification,
    }


def _extract_json_payload(text: str) -> dict | None:
    raw = _strip_thinking(text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    if not raw.startswith("{"):
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match:
            raw = match.group(0)
    try:
        obj = json.loads(raw)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _candidate_meta(code: str, input_meta: dict) -> dict:
    """Candidate source metadata for the Analyzer view."""
    return {
        "solution_file": input_meta.get("solution_file", "solution.cu"),
        "lines": len((code or "").splitlines()),
        "kernel_signatures": kernel_signatures(code),
        "launch_expressions": launch_expressions(code),
        "execution_strategy": describe_strategy(code),
    }


def _role_context(
    board: ErrorBlackboard,
    role: str,
    config: dict,
    code: str,
    attempt_history: str,
) -> str:
    """V_R(B) ∪ A_R(B) ∪ S_T: the blackboard-derived part of a role prompt.

    The role transcript adds the in-iteration summaries and the local window
    (S_I and L_K), maintained by the context manager before every turn.
    """
    return (
        f"## Blackboard View ({role})\n"
        f"{_board_block(board, role, config)}\n\n"
        "## CUDA Evidence Anchors\n"
        f"{anchors_block(board, role, code)}\n\n"
        "## Previous Failed Attempts\n"
        f"{attempt_history}\n\n"
    )


def _parse_verifier_payload(report: str) -> tuple[dict | None, str | None, str | None]:
    """Parse verifier JSON and normalize the single next-fix direction."""
    parsed = _extract_json_payload(report)
    if not parsed:
        return None, None, None
    error_type = str(parsed["error_type"]) if parsed.get("error_type") else None
    next_fix = parsed.get("next_fix")
    if not next_fix:
        suggestions = parsed.get("fix_suggestions") or []
        if isinstance(suggestions, str):
            suggestions = [suggestions]
        next_fix = suggestions[0] if suggestions else None
    return parsed, error_type, str(next_fix).strip() if next_fix else None


def _run_blackboard_verifier(
    client,
    config: dict,
    state: _LoopState,
    board: ErrorBlackboard,
    prompt_text: str,
    current_code: str,
    attempt_history: str = "[]",
    repeated_failure_alert: str | None = None,
    tried_strategies: str | None = None,
    route_diagnosis: dict | None = None,
) -> _RoleResult:
    repeat_section = (
        f"## Repeated Failure Alert\n{repeated_failure_alert}\n\n"
        if repeated_failure_alert
        else ""
    )
    tried_section = (
        f"## {tried_strategies}\n\n" if tried_strategies else ""
    )
    # When the previous attempt failed because the coder mis-implemented an
    # otherwise-sound strategy, keep the direction and fix the implementation.
    route_section = ""
    if route_diagnosis:
        route_section = (
            "## Routing Diagnosis (from the last failed attempt)\n"
            f"{json.dumps(route_diagnosis, indent=2, ensure_ascii=False)}\n\n"
        )
    messages = [
        {"role": "system", "content": with_performance_target(VERIFIER_SYSTEM_PROMPT, config)},
        {
            "role": "user",
            "content": (
                "## Task Prompt\n"
                f"{_truncate_text(prompt_text, 4000)}\n\n"
                "## Current Candidate\n"
                "```cuda\n"
                f"{_truncate_text(current_code, 8000)}\n"
                "```\n\n"
                f"{_role_context(board, 'verifier', config, current_code, attempt_history)}"
                f"{tried_section}"
                f"{route_section}"
                f"{repeat_section}"
            ),
        },
    ]
    tool_decision_max_tokens, tool_decision_temperature = _tool_decision_overrides(
        config.get("agent", {}) or {}
    )
    try:
        raw = deps.call_llm(
            client,
            config,
            messages,
            max_tokens=tool_decision_max_tokens,
            temperature=tool_decision_temperature,
        )
    except Exception as e:
        logger.error(f"  [blackboard verifier] API error: {e}")
        return _RoleResult(stop_reason="api_error")

    # call_llm returns plain text (no usage object), so only the estimated
    # prompt-token cost is recorded for the verifier.
    _accumulate_token_usage(
        state, "verifier", messages, None, _chars_per_token(config)
    )

    board.verifier_report = _sanitize_assistant_content(raw, max_chars=12000) or ""
    parsed, error_type, next_fix = _parse_verifier_payload(board.verifier_report)
    if parsed:
        board.verifier_payload = parsed
        if error_type:
            board.error_type = error_type
        reports = parsed.get("line_level_bug_report") or parsed.get("bug_report") or []
        if isinstance(reports, dict):
            reports = [reports]
        for item in reports:
            if not isinstance(item, dict):
                continue
            board.add_suspected_line({
                "source": "verifier",
                "file": item.get("file", "solution.cu"),
                "line": item.get("line"),
                "confidence": item.get("confidence"),
                "root_cause": item.get("root_cause") or item.get("bug_description"),
                "evidence": item.get("evidence"),
            })
        # Single fix direction (gap 4). Prefer `next_fix`; fall back to the
        # first legacy `fix_suggestions` entry so older outputs still work.
        board.next_fix = next_fix
        # Keep fix_suggestions populated (as the single direction) for any
        # consumer that still reads it.
        board.fix_suggestions = [board.next_fix] if board.next_fix else []
        # The execution strategy the fix must preserve becomes a repair
        # constraint for the coder.
        if parsed.get("performance_constraints"):
            board.execution_strategy = _truncate_text(
                str(parsed["performance_constraints"]), 400
            )
        # Stash the direction so the submission recorder can log it into the
        # tried-strategy registry (gap 6).
        state.current_fix_direction = board.next_fix

    return _RoleResult(report=board.verifier_report)


def _run_route_decision(
    client,
    config: dict,
    state: _LoopState,
    candidate_code: str,
    located: dict | None,
    next_fix: str | None,
    latest_attempt: dict | None,
    tried_strategies: str | None,
    board: ErrorBlackboard | None = None,
    repeated_failure_alert: str | None = None,
) -> dict:
    """Router: after a failed submission, retry the fix or re-analyze.

    The prompt includes the submitted patch intent, the localization, the
    attempted fix strategy, the submitted code, the validation result, the
    attempt history, and any repeated-failure alert, so the router can tell
    a wrong *strategy* from a coder that simply mis-implemented a sound one
    (gap 5b). Returns ``{next_action, diagnosis, reason}``; defaults to
    ``reanalyze`` when the response cannot be parsed (the safe,
    evidence-gathering choice).
    """
    latest = latest_attempt or {}
    cls = latest.get("classification") or {}
    result_summary = {
        "category": cls.get("category"),
        "five_category": cls.get("five_category"),
        "build_ok": cls.get("build_ok"),
        "build_output_excerpt": latest.get("build_output_excerpt"),
        "test_output_excerpt": latest.get("test_output_excerpt"),
        "performance_gate": latest.get("performance_gate"),
    }
    attempt_history = _attempt_history_prompt(state)
    if board is not None:
        history_section = _role_context(
            board, "router", config, candidate_code, attempt_history
        )
    else:
        history_section = f"## Previous Failed Attempts\n{attempt_history}\n\n"
    repeat_section = (
        f"## Repeated Failure Alert\n{repeated_failure_alert}\n\n"
        if repeated_failure_alert
        else ""
    )
    messages = [
        {"role": "system", "content": with_performance_target(ROUTE_DECISION_SYSTEM_PROMPT, config)},
        {
            "role": "user",
            "content": (
                "A patch was just submitted and FAILED. Decide the next step.\n\n"
                "## Submitted patch intent\n"
                f"{latest.get('patch_intent') or latest.get('rationale') or '(none recorded)'}\n\n"
                "## Attempted fix strategy (verifier's direction)\n"
                f"{next_fix or '(none recorded)'}\n\n"
                "## Localization used\n"
                f"{json.dumps(located, indent=2, ensure_ascii=False)}\n\n"
                "## Submitted candidate code\n"
                "```cuda\n"
                f"{_truncate_text(candidate_code, 8000)}\n"
                "```\n\n"
                "## Validation result\n"
                f"{json.dumps(result_summary, indent=2, ensure_ascii=False, default=str)}\n\n"
                f"{history_section}"
                f"{repeat_section}"
                f"## {tried_strategies or 'ALREADY TRIED: (none)'}\n"
            ),
        },
    ]
    tool_decision_max_tokens, tool_decision_temperature = _tool_decision_overrides(
        config.get("agent", {}) or {}
    )
    try:
        raw = deps.call_llm(
            client,
            config,
            messages,
            max_tokens=tool_decision_max_tokens,
            temperature=tool_decision_temperature,
        )
    except Exception as e:
        logger.error(f"  [blackboard router] API error: {e}; defaulting to reanalyze")
        return {"next_action": "reanalyze", "diagnosis": None, "reason": "api_error"}

    _accumulate_token_usage(
        state, "router", messages, None, _chars_per_token(config)
    )
    parsed = _extract_json_payload(_sanitize_assistant_content(raw, max_chars=4000) or "")
    next_action = (parsed or {}).get("next_action")
    if next_action not in ("retry_fix", "reanalyze"):
        next_action = "reanalyze"  # safe default: gather fresh evidence
    return {
        "next_action": next_action,
        "diagnosis": (parsed or {}).get("diagnosis"),
        "reason": (parsed or {}).get("reason"),
    }


def _attach_blackboard_to_latest_record(
    state: _LoopState,
    board: ErrorBlackboard,
) -> None:
    if not state.records:
        return
    state.records[-1]["agent_workflow"] = "blackboard"
    state.records[-1]["error_blackboard"] = board.to_dict()


def _verified_localization(board: ErrorBlackboard) -> dict | None:
    """The localization record a Coder submission is tied to.

    Prefers the Verifier's line-level bug report; falls back to the
    Analyzer/Executor ``locate_bug`` records on the board. Returns None when
    the board holds no usable line, in which case the Coder must localize
    itself before the repair phase unlocks.
    """
    verified = [s for s in board.suspected_lines if s.get("source") == "verifier"]
    located = [
        s for s in board.suspected_lines if s.get("source") in ("analyzer", "executor")
    ]
    for item in verified or located:
        if item.get("line") is None:
            continue
        evidence = item.get("evidence")
        if isinstance(evidence, list):
            evidence = "; ".join(str(e) for e in evidence)
        return {
            "file": item.get("file") or "solution.cu",
            "line": item.get("line"),
            "bug_description": item.get("root_cause") or item.get("bug_description") or "",
            "evidence": evidence or "",
            "source": item.get("source"),
        }
    return None


def _record_board_attempt(state: _LoopState, board: ErrorBlackboard) -> None:
    if state.attempt_history:
        board.record_attempt(attempt_summary(state.attempt_history[-1]))


def _run_blackboard_coder_once(
    client,
    config: dict,
    env: ToolEnv,
    state: _LoopState,
    board: ErrorBlackboard,
    task_stem: str,
    dataset_dir: Path,
    input_meta: dict,
    output_dir: Path,
    prompt_text: str,
    max_tool_calls: int,
    asst_content_cap: int,
    ctx_cfg=None,
    summarizer=None,
) -> str | None:
    agent_cfg = config.get("agent", {}) or {}
    bb_cfg = agent_cfg.get("blackboard") or {}
    max_turns = int(bb_cfg.get("max_coder_turns", 8))
    coder_tools = bb_cfg.get("coder_tools") or [
        "read_code",
        "read_test_file",
        "read_prompt",
        "build_only",
        "apply_edit",
        "write_solution",
    ]
    # locate_bug is a diagnosis tool; the coder only falls back to it when
    # the board holds no localization record.
    fix_tools = [t for t in coder_tools if t != "locate_bug"]
    full_file_mode = _full_file_submissions(config)
    ctx_cfg = ctx_cfg or _build_ctx_config(agent_cfg)
    tool_decision_max_tokens, tool_decision_temperature = (
        _tool_decision_overrides(agent_cfg)
    )
    use_json_action = _json_action_enabled(config)
    json_response_format = bool(agent_cfg.get("json_action_response_format", True))
    force_locate_first = bool(
        bb_cfg.get(
            "coder_force_locate_first",
            agent_cfg.get("coder_force_locate_first", True),
        )
    )
    state.located_this_iteration = False
    state.pending_location = None
    state.applied_edit_this_iteration = False
    localization = _verified_localization(board)
    if localization is not None:
        # The repair phase starts from the verified localization record;
        # write_solution submissions are tied to it.
        state.located_this_iteration = True
        state.pending_location = localization
    local_no_ops = 0
    repeat_section = (
        f"## Repeated Failure Alert\n{board.repeated_failure_alert}\n\n"
        if board.repeated_failure_alert
        else ""
    )
    base_system = with_performance_target(CODER_SYSTEM_PROMPT, config)
    messages: list[dict] = [
        {"role": "system", "content": base_system},
        {
            "role": "user",
            "content": (
                "Repair the candidate based on this verified blackboard.\n\n"
                + "## Task Prompt\n"
                + f"{_truncate_text(prompt_text, 4000)}\n\n"
                + "## Current Candidate (full file)\n"
                + "```cuda\n"
                + f"{env.current_code}\n"
                + "```\n\n"
                + "## Localization Record\n"
                + (
                    f"{json.dumps(localization, indent=2, ensure_ascii=False)}\n\n"
                    if localization is not None
                    else "(none on the blackboard — call `locate_bug` first)\n\n"
                )
                + _role_context(
                    board, "coder", config, env.current_code,
                    _attempt_history_prompt(state),
                )
                + repeat_section
            ),
        },
    ]

    for turn in range(max_turns):
        last_turn = turn == max_turns - 1
        in_locate_phase = (
            bool(env.current_code.strip()) and not state.located_this_iteration
        )
        staged = state.applied_edit_this_iteration
        if state.force_read_code_reason:
            names = [t for t in ("read_code", "write_solution") if t in coder_tools]
            action_context = _forced_action_context(state)
        elif in_locate_phase:
            names = (
                ["locate_bug"]
                if force_locate_first
                else [t for t in fix_tools if t not in _FIX_PHASE_TOOLS] + ["locate_bug"]
            )
            action_context = _join_decision_constraints(
                _forced_action_context(state),
                "No verified localization is on the blackboard. Call "
                "`locate_bug` with the exact file, line, bug, and evidence "
                "before repairing.",
            )
        elif last_turn:
            names = [t for t in ("apply_edit", "write_solution") if t in fix_tools]
            action_context = _join_decision_constraints(
                _forced_action_context(state),
                "This is your last coder turn. Submit now with `write_solution` "
                + (
                    "(the staged edits, or a complete revised `code` file)."
                    if staged
                    else "and a complete revised `code` file."
                ),
            )
        else:
            names = fix_tools
            action_context = _forced_action_context(state)
        tools = schemas_for_config(
            names,
            require_full_solution_code=full_file_mode and not staged,
        )
        prev_iter = state.iteration
        _pre_turn_context_mgmt(
            messages, ctx_cfg, summarizer, base_system,
            in_locate_phase=in_locate_phase, role="coder",
        )
        if use_json_action:
            action_request = _json_action_request(
                tools,
                in_locate_phase,
                extra_context=action_context,
            )
            request_messages = messages + [{"role": "user", "content": action_request}]
            resp = _call_json_action_or_record_api_error(
                client,
                config,
                request_messages,
                state,
                task_stem,
                input_meta,
                max_tokens=tool_decision_max_tokens,
                temperature=tool_decision_temperature,
                response_format=json_response_format,
                role="coder",
            )
            if resp is None:
                return "api_error"
            raw_content = resp.get("content") or ""
            messages.append({
                "role": "assistant",
                "content": _sanitize_assistant_content(
                    raw_content, max_chars=asst_content_cap
                ),
            })
            try:
                action = _extract_json_action(raw_content)
                tc = _action_to_tool_call(
                    action,
                    _tool_names(tools),
                    call_id=f"coder_json_action_{len(env.trace) + 1}",
                )
                resp = {"tool_calls": [tc]}
            except ValueError as e:
                local_no_ops += 1
                _append_action_result_message(
                    messages,
                    "parse_json_action",
                    f"[invalid JSON action] {e}\nRaw response:\n{raw_content[:1000]}",
                    ok=False,
                )
                if local_no_ops >= 2:
                    return "coder_action_parse_error"
                continue
        else:
            resp = _call_llm_or_record_api_error(
                client, config, messages, tools, state, task_stem, input_meta,
                max_tokens=tool_decision_max_tokens,
                temperature=tool_decision_temperature,
                role="coder",
            )
            if resp is None:
                return "api_error"
            _append_assistant_message(messages, resp, asst_content_cap)

        if not resp["tool_calls"]:
            local_no_ops += 1
            if local_no_ops >= 2:
                return "coder_no_op"
            messages.append({
                "role": "user",
                "content": (
                    "Continue with tool calls: call `locate_bug` first."
                    if in_locate_phase
                    else (
                        "Continue with tool calls: apply the verified fix and "
                        "submit it with `write_solution`. Do not answer in "
                        "prose only."
                    )
                ),
            })
            continue
        local_no_ops = 0

        stop = _dispatch_tool_calls(
            resp["tool_calls"], env, state, messages,
            max_tool_calls,
            task_stem, dataset_dir, input_meta, config, output_dir,
            client, prompt_text,
            json_action=use_json_action,
            allow_staged_edits=True,
        )
        if state.iteration > prev_iter:
            _record_board_attempt(state, board)
            _attach_blackboard_to_latest_record(state, board)
            if stop is not None:
                return stop
            return "submitted"
        if stop is not None:
            return stop

        if last_turn:
            # A staged edit is submitted even if the coder omitted
            # write_solution; earlier turns may keep editing or build first.
            guard_stop = _auto_submit_after_apply_edit(
                env, state, messages,
                max_tool_calls,
                task_stem, dataset_dir, input_meta, config, output_dir,
                client, prompt_text,
                json_action=use_json_action,
                allow_staged_edits=True,
            )
            if state.iteration > prev_iter:
                _record_board_attempt(state, board)
                _attach_blackboard_to_latest_record(state, board)
                if guard_stop is not None:
                    return guard_stop
                return "submitted"
            if guard_stop is not None:
                return guard_stop

    return "coder_no_submission"


def run_task_blackboard_agent(
    task_stem: str,
    dataset_dir: Path,
    output_dir: Path,
    config: dict,
    client,
    broken_start: dict | None = None,
) -> tuple[list[dict], str]:
    """Run analyzer -> executor -> verifier -> coder -> router over a shared board."""
    logger.info(f"=== [blackboard agent] Starting task: {task_stem} ===")
    if not broken_start:
        raise ValueError(
            "run_task_blackboard_agent requires `broken_start` — the agent "
            "loop is debug-only."
        )

    input_meta = load_input_meta(task_stem, dataset_dir)
    prompt_text = load_prompt_text(task_stem, dataset_dir)

    agent_cfg = config.get("agent", {}) or {}
    bb_cfg = agent_cfg.get("blackboard") or {}
    env = _build_tool_env(task_stem, dataset_dir, input_meta, config, agent_cfg)
    state = _LoopState()
    _seed_broken_start(
        broken_start, prompt_text, env, state,
        config, task_stem, dataset_dir, input_meta,
    )
    ctx_cfg = _build_ctx_config(agent_cfg)
    summarizer = _maybe_make_summarizer(agent_cfg, client, config)

    max_patch_submissions = _max_patch_submissions(config)
    max_tool_calls = int(agent_cfg.get("max_tool_calls_per_task", 40))
    asst_content_cap = int(agent_cfg.get("assistant_content_cap", 2000))
    max_role_turns = int(bb_cfg.get("max_role_turns", 3))
    max_non_submission_rounds = _max_coder_non_submission_rounds(config)
    analyzer_tools = bb_cfg.get("analyzer_tools") or [
        "read_code", "read_test_file", "read_prompt", "run_static_analysis", "locate_bug",
    ]
    executor_tools = bb_cfg.get("executor_tools") or [
        "read_code", "read_test_file", "build_only", "test_only",
        "run_compute_sanitizer", "locate_bug",
    ]

    state.stop_reason = _patch_submission_limit_stop_reason(config)

    # Guided repair loop. The first round always analyzes; after a failed
    # submission the router decides whether the next round re-analyzes
    # (analyzer + executor) or just retries the fix (verifier + coder) using
    # the existing evidence. The board persists across rounds: a new
    # analysis round archives the previous bug records instead of dropping
    # them, and attempt records accumulate.
    need_analysis = True
    board = ErrorBlackboard(task_stem=task_stem)
    # Seed the intended execution strategy from the starting candidate; the
    # analyzer and verifier refine it.
    board.execution_strategy = describe_strategy(env.current_code)
    route_diagnosis: dict | None = None

    while state.iteration < max_patch_submissions:
        repeated_failure_alert = _repeated_failure_alert(state)
        board.repeated_failure_alert = repeated_failure_alert
        tried_strategies = _tried_strategies_prompt(state)
        repeat_section = (
            f"## Repeated Failure Alert\n{repeated_failure_alert}\n\n"
            if repeated_failure_alert
            else ""
        )

        if need_analysis:
            board.begin_analysis_round()
            runtime = _collect_runtime_observation(
                env, task_stem, dataset_dir, input_meta, config,
            )
            board.runtime_trace["initial_observation"] = runtime
            cls = runtime.get("classification") or {}
            board.error_type = cls.get("category") or cls.get("five_category")
            board.candidate_meta = _candidate_meta(env.current_code, input_meta)

            analyzer_prompt = (
                "Populate the static side of the shared blackboard.\n\n"
                "## Task Prompt\n"
                f"{_truncate_text(prompt_text, 4000)}\n\n"
                "## Current Candidate\n"
                "```cuda\n"
                f"{_truncate_text(env.current_code, 12000)}\n"
                "```\n\n"
                f"{_role_context(board, 'analyzer', config, env.current_code, _attempt_history_prompt(state))}"
                f"{repeat_section}"
                "Read code/test/prompt as needed, call run_static_analysis, "
                "then locate concrete suspected lines and finish with a report."
            )
            analyzer = _run_tool_role_agent(
                "analyzer", ANALYZER_SYSTEM_PROMPT, analyzer_prompt,
                analyzer_tools, client, config, env, state, board,
                task_stem, input_meta, max_role_turns, max_tool_calls,
                asst_content_cap, ctx_cfg=ctx_cfg, summarizer=summarizer,
            )
            board.record_report("analyzer", analyzer.report)
            if analyzer.stop_reason:
                state.stop_reason = analyzer.stop_reason
                break

            executor_prompt = (
                "Populate the runtime side of the shared blackboard.\n\n"
                "## Current Candidate\n"
                "```cuda\n"
                f"{_truncate_text(env.current_code, 12000)}\n"
                "```\n\n"
                "## Initial Runtime Observation\n"
                f"{json.dumps(runtime, indent=2, ensure_ascii=False)}\n\n"
                f"{_role_context(board, 'executor', config, env.current_code, _attempt_history_prompt(state))}"
                f"{repeat_section}"
                "Choose the Compute Sanitizer mode based on the error category "
                "and hypothesis. Combine sanitizer output with static evidence; "
                "do not treat a clean sanitizer run as proof of correctness. "
                "Finish with a report."
            )
            executor = _run_tool_role_agent(
                "executor", EXECUTOR_SYSTEM_PROMPT, executor_prompt,
                executor_tools, client, config, env, state, board,
                task_stem, input_meta, max_role_turns, max_tool_calls,
                asst_content_cap, ctx_cfg=ctx_cfg, summarizer=summarizer,
            )
            board.record_report("executor", executor.report)
            if executor.stop_reason:
                state.stop_reason = executor.stop_reason
                break

        verifier = _run_blackboard_verifier(
            client,
            config,
            state,
            board,
            prompt_text,
            env.current_code,
            attempt_history=_attempt_history_prompt(state),
            repeated_failure_alert=repeated_failure_alert,
            tried_strategies=tried_strategies,
            route_diagnosis=route_diagnosis,
        )
        if verifier.stop_reason:
            state.stop_reason = verifier.stop_reason
            break

        coder_stop = _run_blackboard_coder_once(
            client, config, env, state, board,
            task_stem, dataset_dir, input_meta, output_dir, prompt_text,
            max_tool_calls, asst_content_cap,
            ctx_cfg=ctx_cfg, summarizer=summarizer,
        )
        if coder_stop == "submitted":
            # A patch was submitted but did not pass. Let the router decide
            # whether to retry the fix or re-analyze (gap 5b).
            state.non_submission_rounds = 0
            latest_attempt = (
                state.attempt_history[-1] if state.attempt_history else None
            )
            board.repeated_failure_alert = _repeated_failure_alert(state)
            route = _run_route_decision(
                client, config, state,
                candidate_code=env.current_code,
                located=(latest_attempt or {}).get("located"),
                next_fix=board.next_fix,
                latest_attempt=latest_attempt,
                tried_strategies=_tried_strategies_prompt(state),
                board=board,
                repeated_failure_alert=board.repeated_failure_alert,
            )
            board.record_route(route, state.iteration)
            if latest_attempt is not None:
                latest_attempt["route"] = route
            need_analysis = route["next_action"] == "reanalyze"
            route_diagnosis = None if need_analysis else route
            logger.info(
                "  [blackboard router] next_action=%s diagnosis=%s reason=%s",
                route["next_action"], route.get("diagnosis"), route.get("reason"),
            )
            continue
        if _is_recoverable_coder_stop(coder_stop):
            _record_coder_non_submission(state, coder_stop or "coder_no_submission")
            _record_board_attempt(state, board)
            _write_tool_trace(
                output_dir,
                task_stem,
                env,
                state,
                stop_reason=coder_stop or "coder_no_submission",
            )
            logger.warning(
                "  [blackboard coder] no patch submitted: %s (%d/%d); "
                "continuing with a fresh diagnosis",
                coder_stop,
                state.non_submission_rounds,
                max_non_submission_rounds,
            )
            if state.non_submission_rounds >= max_non_submission_rounds:
                state.stop_reason = coder_stop or "coder_no_submission"
                break
            need_analysis = True       # fresh diagnosis after a non-submission
            route_diagnosis = None
            continue
        if coder_stop is not None:
            state.stop_reason = coder_stop
            break

    if state.records:
        state.records[-1]["stop_reason"] = state.stop_reason
        state.records[-1]["token_usage"] = _token_usage_summary(state)

    _write_tool_trace(output_dir, task_stem, env, state)
    return state.records, state.stop_reason
