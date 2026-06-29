"""Smoke test: verifies harness loads + classifier runs on bundled data.

Does NOT require an LLM API key. Uses a static stderr fixture that the
classifier should label as a compile error.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
INSTANCE_DIR = REPO_ROOT / "data"


def test_smoke_classifier_runs():
    """Smoke: import harness modules + classify a static error message."""
    sys.path.insert(0, str(REPO_ROOT))
    from harness.classifier import classify_result  # raise on import error

    sample_stderr = "kernel.cu:5:1: error: expected ';' before 'return'"
    result = classify_result(
        build_ok=False,
        build_output=sample_stderr,
        test_output=None,
        test_returncode=None,
    )
    assert isinstance(result, dict)
    assert "category" in result
    assert "five_category" in result
    # Build failed -> must classify as some non-passed category.
    assert result["status"] != "passed"


def test_smoke_instance_dir_present():
    """Smoke: at least 1 instance is bundled under data/."""
    inst_dirs = [p for p in INSTANCE_DIR.iterdir() if (p / "instance.json").exists()]
    assert len(inst_dirs) >= 1, "no instance bundled"
    inst = inst_dirs[0]
    assert (inst / "instance.json").exists(), f"missing instance.json in {inst}"
    assert (inst / "input.json").exists(), f"missing input.json in {inst}"
