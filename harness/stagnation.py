"""
Multi-signal stagnation detector.

Detects 4 signals that the LLM is stuck and should stop iterating:
1. duplicate_code    — identical code hash as previous iteration
2. code_cycle        — code hash seen in any earlier iteration (not just prev)
3. category_oscillation — ≥3 category transitions in last 5 iterations
4. no_progress       — same category AND same error for 3+ consecutive rounds
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass
class IterationRecord:
    """Minimal record for stagnation checking."""
    iteration: int
    code_hash: str
    category: str
    error_detail: str  # first 200 chars of error for dedup


def check_stagnation(history: list[IterationRecord]) -> tuple[bool, str]:
    """
    Check if the model is stuck.

    Returns:
        (should_stop, reason)
        reason is one of: "", "duplicate_code", "code_cycle",
                          "category_oscillation", "no_progress"
    """
    if len(history) < 2:
        return False, ""

    # Signal 1: Exact duplicate with previous iteration
    if history[-1].code_hash == history[-2].code_hash:
        return True, "duplicate_code"

    # Signal 2: Code hash seen in any earlier iteration (cycle detection)
    seen = {r.code_hash for r in history[:-1]}
    if history[-1].code_hash in seen:
        return True, "code_cycle"

    # Signal 3: Category oscillation — ≥3 transitions in last 5 iters
    if len(history) >= 5:
        recent = [r.category for r in history[-5:]]
        transitions = sum(
            1 for i in range(1, len(recent)) if recent[i] != recent[i - 1]
        )
        if transitions >= 3:
            return True, "category_oscillation"

    # Signal 4: Same category for 3+ consecutive rounds (regardless of error detail)
    if len(history) >= 3:
        last3 = history[-3:]
        cats = set(r.category for r in last3)
        if len(cats) == 1:
            return True, "no_progress"

    return False, ""


def make_record(
    iteration: int,
    code: str,
    classification: dict,
) -> IterationRecord:
    """Create an IterationRecord from raw iteration data."""
    code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]
    category = classification.get("category", classification.get("five_category", "unknown"))
    detail = classification.get("logic_error_detail", "") or ""
    return IterationRecord(
        iteration=iteration,
        code_hash=code_hash,
        category=category,
        error_detail=detail[:200],
    )


def _detail_hash(detail: str) -> str:
    """Hash error detail for comparison (ignores minor text differences)."""
    # Normalize: strip whitespace, lowercase, remove line numbers
    import re
    normalized = re.sub(r"\d+", "N", detail.lower().strip())
    return hashlib.md5(normalized.encode()).hexdigest()[:8]
