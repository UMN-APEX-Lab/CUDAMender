"""Walk results_v* directories and yield Run objects for each (root, model, axis)."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Map result root dir name -> (dataset, experiment_filter)
_ROOT_MAP = {
    "outputs":                    ("v2", "iter"),
    "outputs_repeated":           ("v2", "repeated"),
    "outputs_feedback":           ("v2", "feedback"),
    "outputs_history_ablation":   ("v2", "history"),
    "outputs_few_shot":           ("v2", "fewshot"),
    "outputs_rerun":           ("v2", "_rerun"),  # axis inferred per-dir; merged into main cell
    "outputs":                    ("v3", None),  # holds iter and repeated by subdir
    "outputs_feedback":           ("v3", "feedback"),
    "outputs_history_ablation":   ("v3", "history"),
    "outputs_few_shot":           ("v3", "fewshot"),
    "outputs_rerun":           ("v3", "_rerun"),
}

_TS_RE = re.compile(r"^\d{8}_\d{6}$")

# Canonical model names for outputs/outputs legacy ablation directory mapping.
# Any directory under outputs/ or outputs/ that does not match (or start with one of)
# these names is skipped. Directories like "gpt5_4_feedback_l0" are split into
# (model=gpt5_4, experiment=feedback, axis=L0) via _classify_legacy_dir.
_CANONICAL_MODELS = {
    "gpt5_4", "qwen36_27b", "gemma4", "glm_4_7",
    "kimi", "qwen3_6plus", "minimax_m2_7",
}


def _classify_legacy_dir(dir_name: str) -> tuple[str, str, str] | None:
    """Map legacy outputs/<dir_name>/ to (model, experiment, axis), or None to skip.

    Returns None if dir_name is not one of the canonical models or canonical model + suffix.
    """
    # Find longest canonical model that is an exact match or prefix
    matched_model = None
    for m in sorted(_CANONICAL_MODELS, key=len, reverse=True):
        if dir_name == m:
            return (m, "iter", "iter")
        if dir_name.startswith(m + "_"):
            matched_model = m
            break
    if matched_model is None:
        return None
    suffix = dir_name[len(matched_model) + 1:]
    if suffix == "repeated":
        return (matched_model, "repeated", "repeated")
    if suffix == "iter":
        return (matched_model, "iter", "iter")
    if suffix in ("kernelbench_clean", "retest_x1"):
        return None  # subset run, skip
    if suffix.startswith("feedback_l"):
        rest = suffix[len("feedback_l"):]
        if rest.endswith("_raw"):
            return (matched_model, "feedback", f"L{rest[:-len('_raw')]}_raw")
        return (matched_model, "feedback", f"L{rest}")
    if suffix.startswith("history_h"):
        return (matched_model, "history", f"H{suffix[len('history_h'):]}")
    if suffix.startswith("fewshot_"):
        return (matched_model, "fewshot", suffix[len("fewshot_"):])
    return None  # unknown suffix, skip


@dataclass(frozen=True)
class Run:
    dataset: str          # 'main' or 'memcrash'
    model: str
    experiment: str       # iter, repeated, feedback, history, fewshot
    axis: str             # iter, L0, H4, matched_K1, ...
    timestamp: str
    manifest_path: Path
    is_rerun: bool = False   # True if from results_v*_rerun/ supplement

    def load_records(self) -> list[dict]:
        out = []
        with self.manifest_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning("parse error in %s: %s", self.manifest_path, e)
        return out


def _latest_timestamp_dir(parent: Path) -> Path | None:
    """Return the timestamped subdir with the latest mtime on its manifest, or None."""
    candidates = []
    for sub in parent.iterdir():
        if not sub.is_dir() or not _TS_RE.match(sub.name):
            continue
        m = sub / "index" / "manifest.jsonl"
        if m.exists():
            candidates.append((m.stat().st_mtime, sub))
        else:
            logger.debug("skipping %s: no manifest", sub)
    if not candidates:
        return None
    return max(candidates)[1]


def _all_timestamp_dirs(parent: Path) -> list[Path]:
    """Return ALL timestamped subdirs with manifest. Used by rerun roots
    where each retry batch is a fresh ts subdir and we want them all merged."""
    out = []
    for sub in parent.iterdir():
        if not sub.is_dir() or not _TS_RE.match(sub.name):
            continue
        if (sub / "index" / "manifest.jsonl").exists():
            out.append(sub)
    return out


def scan_manifests(root: Path) -> list[Run]:
    """Walk root looking for results_v* dirs and emit Run objects.

    Layout per root:
      outputs/<model>/<ts>/index/manifest.jsonl                      (iter)
      outputs_<exp>/<model>/<axis>/<ts>/index/manifest.jsonl         (other v2)
      outputs/<model_or_axis>/<ts>/index/manifest.jsonl              (iter + repeated)
      outputs_<exp>/<model>/<axis>/<ts>/index/manifest.jsonl         (other v3)
      results_v[23]_rerun: mirrors above with multiple ts batches; all are
        emitted with is_rerun=True and merged into the parent (model, axis)
        cell by build_master.
    """
    runs: list[Run] = []
    for root_name, (dataset, exp_filter) in _ROOT_MAP.items():
        rd = root / root_name
        if not rd.is_dir():
            continue
        is_rerun_root = exp_filter == "_rerun"

        for model_dir in sorted(rd.iterdir()):
            if not model_dir.is_dir():
                continue
            model_name = model_dir.name

            # Special handling: outputs and outputs have <model>/<ts>/ directly
            # (and legacy ablation dirs like gpt5_4_feedback_l0/<ts>/...).
            # rerun roots ALSO carry legacy-style dirs (e.g. gpt5_4_history_h1/<ts>/),
            # in addition to clean <model>/<axis>/ structure.
            if root_name in ("outputs", "outputs") or is_rerun_root:
                classified = _classify_legacy_dir(model_name)
                if classified is not None:
                    real_model, experiment, axis = classified
                    ts_dirs = _all_timestamp_dirs(model_dir) if is_rerun_root else (
                        [_latest_timestamp_dir(model_dir)] if _latest_timestamp_dir(model_dir) else []
                    )
                    for ts in ts_dirs:
                        manifest = ts / "index" / "manifest.jsonl"
                        runs.append(Run(
                            dataset=dataset, model=real_model, experiment=experiment,
                            axis=axis, timestamp=ts.name, manifest_path=manifest,
                            is_rerun=is_rerun_root,
                        ))
                    if not is_rerun_root:
                        continue
                # rerun: also walk <model>/<axis>/<ts>/ structure even for canonical models
                if is_rerun_root and model_name in _CANONICAL_MODELS:
                    for axis_dir in sorted(model_dir.iterdir()):
                        if not axis_dir.is_dir() or _TS_RE.match(axis_dir.name):
                            continue  # skip ts dirs already handled above
                        axis_name = axis_dir.name
                        # axis_name like L0 / H1 / matched_K1 — infer experiment
                        if axis_name.startswith("L"):
                            exp = "feedback"
                        elif axis_name.startswith("H"):
                            exp = "history"
                        elif "_K" in axis_name:
                            exp = "fewshot"
                        else:
                            continue  # unrecognised
                        for ts in _all_timestamp_dirs(axis_dir):
                            runs.append(Run(
                                dataset=dataset, model=model_name, experiment=exp,
                                axis=axis_name, timestamp=ts.name,
                                manifest_path=ts / "index" / "manifest.jsonl",
                                is_rerun=True,
                            ))
                continue

            if root_name == "outputs_repeated":
                if model_name not in _CANONICAL_MODELS:
                    continue
                ts = _latest_timestamp_dir(model_dir)
                if ts is None:
                    continue
                runs.append(Run(
                    dataset="v2", model=model_name, experiment="repeated",
                    axis="repeated", timestamp=ts.name,
                    manifest_path=ts / "index" / "manifest.jsonl",
                ))
                continue

            # Generic: <model>/<axis>/<ts>/index/manifest.jsonl
            for axis_dir in sorted(model_dir.iterdir()):
                if not axis_dir.is_dir():
                    continue
                axis_name = axis_dir.name
                ts = _latest_timestamp_dir(axis_dir)
                if ts is None:
                    continue
                manifest = ts / "index" / "manifest.jsonl"
                experiment = exp_filter or "iter"
                runs.append(Run(
                    dataset=dataset, model=model_name, experiment=experiment,
                    axis=axis_name, timestamp=ts.name, manifest_path=manifest,
                ))
    return runs
