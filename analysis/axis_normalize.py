"""Canonicalize axis names from config basenames.

A config basename has the shape `<model>[_<suffix>]*.yaml` where the optional
suffixes encode (in any order):
  - _memcrash         (the 16-instance mem_crash supplement subset)
  - _p<port>          (vLLM port routing)
  - _rerun_<ts>       (rerun-with-timestamp)
  - <axis-keyword>... (repeated | feedback_l<N> | history_h<N> | fewshot_<...>)

Examples:
    gemma4.yaml                            -> iter
    gemma4_memcrash.yaml                   -> iter
    gemma4_repeated.yaml                   -> repeated
    gemma4_feedback_l0.yaml                -> L0
    gemma4_feedback_l3_raw.yaml            -> L3_raw
    gemma4_feedback_l3_memcrash.yaml       -> L3
    gemma4_history_h1.yaml                 -> H1
    gemma4_fewshot_matched_K1.yaml         -> matched_K1
    gemma4_fewshot_matched_K1_p8007.yaml   -> matched_K1
    glm_4_7_feedback_l1_rerun_<ts>.yaml    -> L1
"""
from __future__ import annotations

import re

# Strip these suffixes (in any position before .yaml) to canonicalize.
_SUFFIX_PATTERNS = [
    re.compile(r"_p\d{4}"),
    re.compile(r"_rerun_\d{8}_\d{6}"),
    re.compile(r"_memcrash"),
]

_KEYWORD_RE = re.compile(r"_(repeated|feedback_l|history_h|fewshot_)(.*)$")


def _strip_suffixes(name: str) -> str:
    """Drop port-routing, rerun-timestamp, and memcrash markers from the basename."""
    for pat in _SUFFIX_PATTERNS:
        name = pat.sub("", name)
    return name


def _parse_model_and_rest(config_basename: str) -> tuple[str, str | None]:
    """Extract (model, rest) from `<model>[_<rest>].yaml`.

    `rest` is the portion AFTER the model name; for the bare-iter case it is
    None.
    """
    if not config_basename.endswith(".yaml"):
        raise ValueError(f"Unrecognized config basename format: {config_basename!r}")
    full = config_basename[:-len(".yaml")]
    kw_match = _KEYWORD_RE.search(full)
    if kw_match:
        model = full[:kw_match.start()]
        rest = full[kw_match.start() + 1:]
        return model, rest
    return full, None


def normalize_axis(config_basename: str) -> str:
    """Map a config basename to its canonical axis name."""
    name = _strip_suffixes(config_basename)
    _model, rest = _parse_model_and_rest(name)
    if rest is None:
        return "iter"
    if rest == "repeated":
        return "repeated"
    if rest.startswith("feedback_l"):
        suffix = rest[len("feedback_l"):]
        if suffix.endswith("_raw"):
            return f"L{suffix[:-len('_raw')]}_raw"
        return f"L{suffix}"
    if rest.startswith("history_h"):
        return f"H{rest[len('history_h'):]}"
    if rest.startswith("fewshot_"):
        return rest[len("fewshot_"):]
    raise ValueError(f"Unrecognized rest portion {rest!r} in {config_basename!r}")


def parse_experiment(config_basename: str) -> str:
    """Map config basename to one of: iter, repeated, feedback, history, fewshot."""
    name = _strip_suffixes(config_basename)
    _model, rest = _parse_model_and_rest(name)
    if rest is None:
        return "iter"
    if rest == "repeated":
        return "repeated"
    if rest.startswith("feedback_"):
        return "feedback"
    if rest.startswith("history_"):
        return "history"
    if rest.startswith("fewshot_"):
        return "fewshot"
    raise ValueError(f"Unrecognized rest portion {rest!r} in {config_basename!r}")
