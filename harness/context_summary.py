"""
LLM-backed summarisation for context management.

When a tool result (or a batch of aged-out exchanges) is too large to keep
verbatim, we ask the *same* agent model to compress it rather than
blind-truncating.

Why not truncate: a head+tail truncation drops the middle of a file and
leaves a "[middle truncated]" marker. The agent reads that as "my earlier
read was incomplete" and re-reads the same file in a loop (observed on
CUDA_10's first smoke run — test_main.cu read 6×). An LLM summary carries
no incompleteness signal — it reads as a deliberate, complete digest, so
the agent acts on it instead of re-fetching.

`make_summarizer(client, config)` returns a callable

    summarize(text: str, kind: str, target_chars: int) -> str

with:
  - content-hash caching (same file summarised once, reused after)
  - a call budget (falls back to truncation when exhausted / on API error)
  - kind-specific prompts that tell the model what to preserve

`make_rule_based_summarizer()` returns a deterministic summariser with the
same signature and the same per-kind preservation schema, used when
LLM summarisation is disabled (``agent.use_llm_summary: false``).

Kinds:
  code          — read_code / read_test_file / read_prompt output
  diagnostic    — compute-sanitizer / cuda-gdb output
  build_log     — build_only / write_solution feedback
  test_log      — test_only output
  investigation — a batch of tool exchanges aged out of the recent window
  attempt       — one completed iteration's tool activity (cross-iter fold)
  generic       — fallback
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Callable

logger = logging.getLogger("cuda_debugger_exp")


SUMMARY_SYSTEM = """\
You are a context compressor for an automated CUDA debugging agent. You \
receive a chunk of context (source code, tool output, or an investigation \
log) that is too large to keep verbatim. Produce a dense, faithful summary \
the agent can still act on. Never invent facts. Never add commentary. \
Output ONLY the summary text."""


# kind -> extra "preserve exactly" instruction appended to the user prompt
_KIND_INSTRUCTIONS = {
    "code": (
        "This is CUDA/C++/Python source. PRESERVE EXACTLY: every function, "
        "kernel, and method signature; struct/class definitions; key "
        "constants and macros; launch-config expressions; and the LINE "
        "NUMBERS of anything a debugger would care about. Keep line-number "
        "references in the form `L<n>`. Drop boilerplate and repetitive "
        "blocks, but never drop a signature."
    ),
    "diagnostic": (
        "This is compute-sanitizer or cuda-gdb output. PRESERVE EXACTLY: "
        "the error kind, the faulting thread/block coordinates, the faulting "
        "address, the source file:line of the fault, the signal name, and "
        "the total error count. Drop repeated per-thread duplicates."
    ),
    "build_log": (
        "This is a compiler / test log. PRESERVE EXACTLY: every distinct "
        "error message with its file:line, the first failing assertion, and "
        "any Expected/Got/Tolerance values. Drop warnings and repeated "
        "instantiation noise unless they are the only diagnostic."
    ),
    "test_log": (
        "This is functional-test output. PRESERVE EXACTLY: every failing "
        "assertion or check, Expected-versus-actual values (Expected/Got, "
        "ref/out, tolerance), the failing index or shape, and any runtime "
        "error with its file:line. Drop passing checks and progress noise."
    ),
    "investigation": (
        "This is a log of tool calls the agent already made in the CURRENT "
        "debugging iteration. Summarise WHAT was investigated, WHAT was "
        "found (concrete facts: signatures seen, error coordinates, line "
        "numbers), and WHAT remains unknown. The agent will use this to "
        "avoid re-investigating."
    ),
    "attempt": (
        "This is the tool activity of one COMPLETED debugging iteration "
        "(an earlier fix attempt). Summarise: what fix was attempted, what "
        "the build/test outcome was, and the single most important finding. "
        "Two or three sentences — this is stale background, not live state."
    ),
    "generic": (
        "Summarise faithfully, preserving any error messages, identifiers, "
        "numbers, and file:line references."
    ),
}


def _build_user_prompt(text: str, kind: str, target_chars: int) -> str:
    instruction = _KIND_INSTRUCTIONS.get(kind, _KIND_INSTRUCTIONS["generic"])
    return (
        f"{instruction}\n\n"
        f"Target length: about {target_chars} characters (hard limit "
        f"{target_chars * 2}).\n\n"
        f"--- CONTENT TO SUMMARISE ---\n{text}\n--- END CONTENT ---"
    )


def _truncate(text: str, cap: int) -> str:
    """Head+tail fallback used when summarisation is unavailable."""
    if len(text) <= cap:
        return text
    half = max(1, (cap - 80) // 2)
    return (
        text[:half]
        + "\n... [middle truncated — summariser unavailable] ...\n"
        + text[-half:]
    )


def make_summarizer(
    client,
    config: dict,
    *,
    max_calls: int = 40,
    max_output_tokens: int = 4096,
) -> Callable[[str, str, int], str]:
    """Build an LLM-backed summarise() callable.

    `client` / `config` are the same objects the agent loop uses, so the
    summariser runs on the same model. A derived config lowers temperature
    for determinism.

    `max_output_tokens` caps the summary length. It must be comfortably
    above the token-equivalent of the largest `target_chars` the caller
    passes — a too-low cap silently truncates the summary mid-sentence,
    which is worse than a longer-but-complete digest.
    """
    from .llm_client import call_llm

    # Derived config: low temperature, summary-sized output. Same
    # backend/model/base_url — no extra endpoint needed.
    api = dict(config.get("api", {}))
    api["temperature"] = 0.2
    api["max_tokens"] = int(max_output_tokens)
    api["max_completion_tokens"] = int(max_output_tokens)
    # A reasoning model would burn the whole budget thinking; summarisation
    # is a shallow task, so drop reasoning_effort if present.
    api.pop("reasoning_effort", None)
    api.pop("seed", None)
    summ_config = {**config, "api": api}

    cache: dict[str, str] = {}
    state = {"calls": 0, "cache_hits": 0, "fallbacks": 0}

    def summarize(text: str, kind: str = "generic", target_chars: int = 2000) -> str:
        if not text:
            return text
        key = hashlib.sha256(
            f"{kind}|{target_chars}|{text}".encode("utf-8")
        ).hexdigest()
        if key in cache:
            state["cache_hits"] += 1
            return cache[key]

        if state["calls"] >= max_calls:
            state["fallbacks"] += 1
            logger.warning(
                f"  [summarizer] call budget {max_calls} exhausted — "
                f"falling back to truncation for {kind}"
            )
            return _truncate(text, target_chars)

        messages = [
            {"role": "system", "content": SUMMARY_SYSTEM},
            {"role": "user", "content": _build_user_prompt(text, kind, target_chars)},
        ]
        try:
            state["calls"] += 1
            raw = call_llm(client, summ_config, messages)
        except Exception as e:
            state["fallbacks"] += 1
            logger.warning(
                f"  [summarizer] LLM call failed ({type(e).__name__}: {e}) — "
                f"falling back to truncation"
            )
            return _truncate(text, target_chars)

        summary = (raw or "").strip()
        if not summary:
            state["fallbacks"] += 1
            return _truncate(text, target_chars)

        labelled = (
            f"[ctx-compressed {kind}: {len(text)}→{len(summary)} chars]\n"
            + summary
        )
        cache[key] = labelled
        return labelled

    # Expose counters for logging / tests
    summarize.state = state          # type: ignore[attr-defined]
    summarize.cache = cache          # type: ignore[attr-defined]
    return summarize


# ---------------------------------------------------------------------------
# Deterministic type-aware summariser
# ---------------------------------------------------------------------------

_CODE_KEEP_RE = re.compile(
    r"__global__|__device__|__host__|<<<|>>>|#define|#include|constexpr|"
    r"\bconst\s+\w+\s+\w+\s*=|\bstruct\b|\bclass\b|\btemplate\b|"
    r"__shared__|__syncthreads|__syncwarp|blockIdx|blockDim|threadIdx|gridDim|"
    r"\bdim3\b|cudaMalloc|cudaMemcpy|\bdef\s|\bTORCH_|PYBIND11"
)
_DIAG_KEEP_RE = re.compile(
    r"Invalid|Hazard|Uninitialized|ERROR SUMMARY|Barrier|Race|race|"
    r"thread \(|block \(|Address|address|at 0x|by thread|\.cu:\d+|\.cuh:\d+|"
    r"line \d+|\bL\d+\b|severity|flag|signal|SIG[A-Z]+|error",
)
_LOG_KEEP_RE = re.compile(
    r"error|Error|ERROR|assert|Assert|FAIL|Fail|fail|mismatch|Mismatch|"
    r"expected|Expected|got|Got|actual|Actual|tolerance|Tolerance|"
    r"max_abs|max_diff|diff|illegal|Illegal|undefined|\.cu\(\d+\)|\.cu:\d+|"
    r"build_ok=|category=|speedup|Speedup|PERFORMANCE GATE|timeout|Timeout"
)
_KIND_KEEP_RE = {
    "code": _CODE_KEEP_RE,
    "diagnostic": _DIAG_KEEP_RE,
    "build_log": _LOG_KEEP_RE,
    "test_log": _LOG_KEEP_RE,
}
# Tool name -> summary kind for results inside investigation/attempt digests.
_RESULT_KIND = {
    "read_code": "code",
    "read_test_file": "code",
    "read_prompt": "code",
    "run_static_analysis": "diagnostic",
    "run_compute_sanitizer": "diagnostic",
    "build_only": "build_log",
    "test_only": "test_log",
    "write_solution": "build_log",
    "locate_bug": "generic",
    "apply_edit": "generic",
}


def _keep_lines(text: str, kind: str, max_lines: int) -> list[str]:
    pattern = _KIND_KEEP_RE.get(kind)
    seen: set[str] = set()
    kept: list[str] = []
    for line in text.splitlines():
        stripped = line.rstrip()
        if not stripped.strip():
            continue
        if pattern is not None and not pattern.search(stripped):
            continue
        key = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", stripped.strip())
        if key in seen:
            continue
        seen.add(key)
        kept.append(stripped)
        if len(kept) >= max_lines:
            break
    if not kept:
        kept = [ln.rstrip() for ln in text.splitlines() if ln.strip()][: min(4, max_lines)]
    return kept


def _summarise_tool_log(text: str, kind: str, target_chars: int) -> str:
    """InvSum / AttSum over a ``TOOL: name(args)\nRESULT:\n...`` log."""
    blocks = re.split(r"(?m)^(?=TOOL: )", text)
    inspected: list[str] = []
    facts: list[str] = []
    outcomes: list[str] = []
    for block in blocks:
        if not block.startswith("TOOL: "):
            continue
        head, _, result = block.partition("\nRESULT:\n")
        call = head[len("TOOL: "):].strip()
        name = call.split("(", 1)[0]
        inspected.append(call[:200])
        result_kind = _RESULT_KIND.get(name, "generic")
        for line in _keep_lines(result, result_kind, 4):
            facts.append(f"[{name}] {line.strip()[:200]}")
        if "[BUILD/TEST RESULT]" in result:
            outcomes.extend(
                ln.strip() for ln in result.splitlines()
                if ln.startswith("build_ok=")
            )
    label = "Attempt" if kind == "attempt" else "Investigation"
    parts = [f"{label} — tool calls: " + "; ".join(inspected)]
    if facts:
        parts.append("Facts:\n" + "\n".join(f"- {f}" for f in facts))
    if outcomes:
        parts.append("Validation outcome: " + "; ".join(outcomes))
    return _truncate("\n".join(parts), target_chars)


def rule_based_summary(text: str, kind: str = "generic", target_chars: int = 2000) -> str:
    """Summarize(r, Kind(r)) without an LLM.

    Keeps exactly the fields the per-kind preservation schema names:
    signatures, constants, launch expressions and line numbers for source;
    error kinds, coordinates, addresses, locations and counts for
    diagnostics; distinct errors, failing assertions and expected-versus-
    actual values for build/test logs; inspected calls, concrete facts and
    validation outcomes for investigation/attempt digests.
    """
    if not text:
        return text
    if kind in ("investigation", "attempt"):
        body = _summarise_tool_log(text, kind, target_chars)
    else:
        max_lines = max(8, target_chars // 60)
        body = _truncate("\n".join(_keep_lines(text, kind, max_lines)), target_chars)
    return f"[ctx-compressed {kind} (typed): {len(text)}→{len(body)} chars]\n" + body


def make_rule_based_summarizer() -> Callable[[str, str, int], str]:
    state = {"calls": 0, "cache_hits": 0, "fallbacks": 0}

    def summarize(text: str, kind: str = "generic", target_chars: int = 2000) -> str:
        state["calls"] += 1
        return rule_based_summary(text, kind, target_chars)

    summarize.state = state          # type: ignore[attr-defined]
    return summarize
