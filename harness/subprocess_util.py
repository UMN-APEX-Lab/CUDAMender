"""Subprocess helpers that don't leak orphan grandchildren on timeout.

`subprocess.run(..., shell=True, timeout=...)` only SIGKILLs the direct
child (the shell). When the shell launches `python -c "..."` (e.g., a
KernelBench `eval_kernel_against_ref` call), that python becomes an
orphan, reparents to init, and continues holding GPU memory until the
CUDA call returns or the GPU is reset.

Use `run_pg` instead — it puts the child in a new process group and
SIGKILLs the whole group on timeout.
"""
from __future__ import annotations

import os
import signal
import subprocess
from typing import Sequence


def run_pg(
    cmd: str | Sequence[str],
    *,
    cwd: str | os.PathLike | None = None,
    env: dict | None = None,
    timeout: int,
    shell: bool = False,
    text: bool = True,
) -> subprocess.CompletedProcess:
    """`subprocess.run` substitute that group-kills on timeout.

    Returns a CompletedProcess on success. Raises subprocess.TimeoutExpired
    on timeout (after killing the whole process group).
    """
    proc = subprocess.Popen(
        cmd,
        shell=shell,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=text,
        start_new_session=True,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
        return subprocess.CompletedProcess(cmd, proc.returncode, out, err)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
        try:
            out, err = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            out, err = "", "killed (process group did not exit within 10s)"
        raise subprocess.TimeoutExpired(cmd, timeout, output=out, stderr=err)
