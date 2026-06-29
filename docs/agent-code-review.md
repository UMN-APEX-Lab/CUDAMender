# Agent subsystem — code review

Review of the uncommitted agent subsystem under `harness/`. Scope:

| File | LOC | Role |
|---|---|---|
| `agent_loop.py` | 2320 | ReAct + blackboard orchestration, the loop |
| `agent_tools.py` | 988 | Tool registry, executors, quota dispatcher |
| `context_manager.py` | 503 | 4-layer context compaction |
| `context_summary.py` | 197 | LLM-backed summariser |
| `debug_assist.py` | 865 | Precheck, localization, patch-scope, patch-verifier |
| `diagnostics.py` | 440 | compute-sanitizer wrapper + option allow-list |
| `performance.py` | 117 | perf gate measurement |
| `trace_io.py` | 238 | stage-trace + sanitizer-summary serialisation |

Reviewed against `tests/test_agent_loop.py`, `test_agent_tools.py`, `test_diagnostics.py`.

Overall this is well-structured code: the two-phase locate→fix design is enforced *structurally* (tools are withheld, not just rejected), the tool layer is cleanly separated from orchestration, and the sanitizer wrapper exposes a broad option surface without giving the model arbitrary shell. The findings below are mostly in the *optional* debug-assist paths and in the interaction between the JSON-action protocol and the context manager — areas the test suite doesn't exercise.

Severity legend: **[H]** high (silently breaks a feature / unbounded cost), **[M]** medium, **[L]** low, **[N]** nit.

---

## High severity

### H1 — `verify_failed_patch` is called with the wrong signature; the patch-verifier never runs
`agent_loop.py:1094-1105` calls:

```python
verifier_result = verify_failed_patch(
    client=..., config=..., task_stem=..., sample_id=..., iteration=...,
    prompt_text=..., broken_code=..., candidate_code=...,
    localization=state.pending_location, classification=classification,
)   # <-- no build_output, no test_output
```

But the definition (`debug_assist.py:386-399`) declares `build_output: str` and `test_output: str | None` as **required positional parameters** (no defaults), before `trace_writer=None`. The call therefore raises `TypeError: verify_failed_patch() missing 2 required positional arguments` on every invocation. That exception is swallowed by the surrounding `try/except` at `agent_loop.py:1106-1107`, which only logs `[agent] patch verifier failed: …` and leaves `verifier_result = None`.

**Impact:** when `debug_tools.patch_verifier.enabled` is set, the feature is silently dead — it never produces guidance, it just logs a warning each failed iteration. Both `build_output` and `test_output` are already in local scope at the call site, so the fix is to pass them.

**Fix:** add `build_output=build_output, test_output=test_output` to the call (and see H2).

### H2 — the verifier (and the located-bug record) reads `state.pending_location` *after* it's been reset to `None`
In `_handle_submission`, the localisation is cleared at `agent_loop.py:1066-1070`:

```python
state.located_this_iteration = False
state.pending_location = None        # <-- cleared here
```

…but `verify_failed_patch` is called ~25 lines later (`:1103`) with `localization=state.pending_location`, which is now `None`. So even after fixing H1, the verifier always receives `localization=None`. The downstream `format_localization_report(config, None)` returns `""`, so the verifier loses the agent's file/line/evidence entirely.

**Fix:** capture the location before the reset (e.g. `located = state.pending_location` near `:1038` where it's already read for the record) and pass that captured value to both `analyze_patch_scope` and `verify_failed_patch`.

### H3 — context compaction (L1/L2/L3) is inert in JSON-action mode
This is the most consequential finding because the shipped smoke config (`configs/agent/agent_smoke.yaml`) sets `action_protocol: json`.

The context manager keys all three structural layers off the **native tool-calling** message shape:
- `cap_tool_results` (L1) only inspects messages with `role == "tool"` (`context_manager.py:277`).
- `slide_recent_exchanges` (L2) / `_tool_groups` only fold an `assistant` message that has `tool_calls` plus the following `tool` messages (`context_manager.py:152-158`).
- `compact_old_iterations` (L3) only fires when it finds `[BUILD/TEST RESULT]` markers (`_is_marker`, `context_manager.py:300-305`).

In JSON-action mode none of those shapes exist:
- Tool results are appended as `role: "user"` wrapped in `ACTION_RESULT_TEMPLATE`, whose content starts with `"[DETERMINISTIC ACTION RESULT]"` (`agent_loop.py:177-183`, `_append_action_result_message`). So there are **no `role: "tool"` messages** (L1 no-ops) and the build/test feedback **no longer starts with `[BUILD/TEST RESULT]`** (L3's marker scan finds nothing).
- Assistant messages carry plain text, **never `tool_calls`** (`agent_loop.py:2171-2176`). So `_tool_groups` returns `[]` and L2 no-ops.

Net: in JSON-action mode only L4 (the budget *nudge*) does anything — the message list is never actually compacted. On top of that, every turn re-appends the full `ACTION REQUEST` block, which embeds the entire compact tool schema (`agent_loop.py:2170`), and those user messages are never folded either. So JSON mode both **grows fastest** (≈3 messages/turn including a schema dump) and **has no compaction at all**. On a long task this risks a runaway token bill and eventually overflowing the model window despite the context manager being "configured."

**Fix options:**
- Make `_is_marker` also recognise the `ACTION RESULT` wrapper (or have `_append_action_result_message` keep the `[BUILD/TEST RESULT]` prefix at the start), and teach `_tool_groups`/`cap_tool_results` to treat an (assistant-text, action-result-user) pair as a foldable group in JSON mode; and
- Drop/replace the prior `ACTION REQUEST` message instead of appending a new one each turn (it's regenerated every turn anyway — only the latest is needed).
- At minimum, document that JSON-action mode has no context compaction and rely on a tight `token_budget`.

A unit test that runs `manage_context` over a synthetic JSON-action transcript and asserts `tokens_after < tokens_before` would lock this down.

---

## Medium severity

### M1 — patch-scope guard can never detect a "missed suspicious line" (shape mismatch)
`analyze_patch_scope` extracts the located lines from `localization["parsed"]["suspicious_lines"]` (`debug_assist.py:346-350`). But the agent loop passes `state.pending_location`, which has the shape `{file, line, bug_description, evidence}` (set in `_handle_locate_bug`, `agent_loop.py:876-881`) — there is **no `parsed` key**. So `suspicious` is always `[]`, which means:
- `overlap` is always empty, and
- `missed = bool(suspicious) and not overlap and not large_rewrite` is always `False`.

The `large_rewrite` heuristic still works, but the headline "did the patch actually touch the line the agent located?" check is dead — even though the data (`state.pending_location["line"]`) is right there. This guard only matters when `debug_tools.patch_scope_guard.enabled`, which is why it's gone unnoticed.

**Fix:** either adapt `analyze_patch_scope` to accept the `{file, line, …}` shape, or wrap the location: `{"parsed": {"suspicious_lines": [{"line": loc["line"]}]}}` at the call site.

### M2 — perf measurement hardcodes `gpu_id=0`
`performance.measure_iter_perf` calls `_bench_run_candidate(task_id, files, gpu_id=0)` (`performance.py:55`). This is correct for the single-GPU `CUDA_VISIBLE_DEVICES=N` pinning the configs use, but if the perf path is ever driven under `parallel_runner.py` with multiple workers that distribute by *logical* GPU id rather than by masking `CUDA_VISIBLE_DEVICES`, every worker's perf build lands on physical GPU 0 and they contend. Worth either threading the worker's GPU id through or asserting the `CUDA_VISIBLE_DEVICES` assumption.

### M3 — token estimator is fixed at 4.0 chars/token and under-counts CUDA source
`ContextConfig.chars_per_token` defaults to `4.0`, but `_build_ctx_config` (`agent_loop.py:599-608`) never reads a `chars_per_token` override from `agent_cfg`, so it's effectively hard-coded. Dense CUDA/C++ tokenises closer to ~3 chars/token, so `estimate_tokens` *under*-estimates by ~25% on code-heavy transcripts — the warn/force nudges fire later than the configured ratios imply, and on a model with a hard window this can overshoot. Either wire the override through `_build_ctx_config`, or lower the default for this code-heavy workload. (Note the api block already carries a separate `service_limit_chars_per_token: 2.0`; the two knobs being unrelated is itself a readability trap.)

---

## Low severity / efficiency

### L1 — a passing candidate is compiled twice
On pass, `_handle_submission` has already run `compile_code` + `run_test` (`agent_loop.py:953-960`); `measure_iter_perf` then rebuilds the same code through `_bench_run_candidate` (`performance.py:53-55`) for timing. If the benchmark backend can time an already-built artifact, reusing it would save a full nvcc compile per passing task. At minimum worth a comment explaining why the rebuild is unavoidable.

### L2 — blackboard runs an extra full build+test every round
`run_task_blackboard_agent` calls `_collect_runtime_observation` (a full `compile_code` + `run_test`) at the top of every `while` iteration (`agent_loop.py:1959`), in addition to the coder's `write_solution` build later in the same round. That's up to 2× builds per submission. Consider reusing the most recent build result from the prior round when the candidate hasn't changed.

### L3 — `read_code` returns an empty body (no error) for an out-of-range range
In `_tool_read_code` (`agent_tools.py:541-554`), `start=500` on a 100-line file gives `s=500, e=100`; the slice `lines[499:100]` is empty and the tool returns `ok=True` with an empty body and no diagnostic. The model can't tell "empty file region" from "I asked for the wrong range." Return an explicit out-of-range message (as `locate_bug` already does at `agent_tools.py:776-778`). The same asymmetry exists in `_safe_read`: the `MAX_READ_LINES` clamp only applies when *both* or *neither* of `start`/`end` are given (`agent_tools.py:525-528`), so `start`-only reads aren't line-capped (only char-capped).

### L4 — `read_test_file` cannot read the bundled CUTLASS / ThunderKittens headers
`_tool_read_test_file` resolves the path and rejects anything that escapes `testbench_dir` (`agent_tools.py:562-567`). But `prepare_benchmark.sh` populates `testbench/<stem>/reference_sources/{cutlass,ThunderKittens}` as **absolute symlinks** into `data/_external/`. After `resolve()`, those land outside the testbench root and `relative_to` raises → "path escapes testbench root." So for cutlass/TK tasks the agent can't read the very headers it needs. Confinement is good, but consider allow-listing `data/_external` (read-only) or resolving symlinks before the containment check.

### L5 — budget "force-submit" can deadlock against the LOCATE phase
The L4 force nudge says *"You MUST call `write_solution` on the next turn"* (`context_manager.py:498-502`), but if the agent hasn't called `locate_bug` yet, `write_solution` is withheld from the tool list (`_tools_for_phase`). In JSON mode that yields an "tool not available in this phase" rejection loop until the no-op/parse-error counter trips. The nudge should be phase-aware (tell the model to `locate_bug` *then* submit when still in LOCATE).

### L6 — agent records inconsistently stamp `method`/file extensions
- The API-error record in `_call_*_or_record_api_error` (`agent_loop.py:779-784`, `808-813`) doesn't set `record["method"] = "agent"`, unlike the submission record (`:1041`), so failed-API tasks look method-less in analysis.
- `_persist_iter_artifacts` always writes `error_code/<hash>.cu` and `_make_iter_record` always records a `.cu` path (`agent_loop.py:859`, `:2299`) even for `solution.py` (kernelbench/applied-kernels) tasks. Harmless for scoring, but mislabels Python solutions in the artifact tree.

---

## Nits

- **N1** `agent_loop.py:1090` — `debug_tools_enabled(config) and section_enabled(config, "patch_verifier", …)` is redundant; `section_enabled` already short-circuits on `debug_tools_enabled` (`debug_assist.py:23-29`).
- **N2** `agent_loop.py:1180` increments `state.tool_calls_used` *before* the `force_read_code_reason` rejection at `:1183`, so a rejected (never-executed) action still consumes the global tool budget. Defensible as a loop-breaker, but slightly surprising.
- **N3** `_extract_json_action` (`agent_loop.py:352-369`) scans every `{` and calls `raw_decode` from each — O(n²) on a large blurb. Fine for compact actions; a guard on length wouldn't hurt.
- **N4** `_tool_run_static_analysis` defines `add()` as a closure *inside* the per-line loop (`agent_tools.py:608`), rebuilding it every iteration. Cosmetic.

---

## Test-coverage gaps

The bugs above survive because the affected paths aren't tested. There are **no** dedicated tests for `context_manager.py`, `context_summary.py`, `debug_assist.py`, `trace_io.py`, or `performance.py`; `test_agent_loop.py` exercises the happy paths and the native/JSON loops but not the optional debug-assist hooks.

Highest-value additions, each of which would have caught a finding above:
1. **`verify_failed_patch` integration** with `debug_tools.patch_verifier.enabled` and a stub client → catches **H1** + **H2** (the call would raise / receive `None` localization).
2. **`manage_context` over a JSON-action transcript** asserting real compaction occurs → catches **H3**.
3. **`analyze_patch_scope` fed the agent's real `pending_location` shape**, asserting `overlap`/`missed_suspicious_lines` are populated when the patch misses the located line → catches **M1**.
4. A `context_manager` round-trip test asserting the message list stays **API-valid** (no orphaned `tool` messages) after L2/L3 — the `_tool_groups` whole-group design is good and worth pinning.

---

## What's done well (keep)

- **Structural two-phase enforcement.** Withholding `apply_edit`/`write_solution` until `locate_bug` (`_tools_for_phase`, `_FIX_PHASE_TOOLS`) makes "locate first" a property of the tool surface, with a backstop for backends that ignore the filter (`agent_loop.py:913-936`). Clean.
- **Whole-group folding.** `_tool_groups` deliberately folds an assistant-with-tool_calls plus *all* its tool replies as one unit to avoid orphaned `tool` messages (→ API 400). The reasoning is documented and correct; multi-tool-call turns are handled.
- **Sanitizer option allow-list.** `diagnostics._build_sanitizer_options` validates/renders a large flag surface and rejects unknown keys (`:225-227`) while the executable stays the benchmark's own `test_command` — broad capability, no arbitrary shell.
- **Summariser robustness.** Content-hash cache + call budget + graceful truncation fallback on budget exhaustion or API error (`context_summary.py:149-192`), with the rationale (avoid the "[middle truncated]" re-read loop) documented.
- **Defensive API-error handling.** Every LLM call wrapper stamps an `api_error` iteration record and sets `stop_reason` instead of crashing the task (`agent_loop.py:755-814`).
- **Tool trace.** `tool_trace.jsonl` with per-call timing/quotas is a genuinely useful audit artifact for analysing agent behaviour.
