"""Queue daemon — auto-launches sequential v2 experiment runs per model.

Usage (background):
    nohup python evaluation/scripts/queue_daemon.py \
        --queue /tmp/queue_gemma4.txt \
        --gpu 3 \
        --label gemma4 > /tmp/queue_daemon_gemma4.log 2>&1 &

Queue file format: one absolute config path per line, '#' for comments,
blank lines OK. Empty queue → daemon exits.

Behavior:
  - Polls running PID (set via --watch-pid). When it exits (or none set
    initially), pops the next config from the queue and launches it.
  - New PID becomes the next watch target.
  - Records each launch's exit status to <queue>.log.json (append-only).
  - Reads /etc/{cuda-debugger}/.env-style .env if present, plus any keys
    already in os.environ. We assume caller exported keys before launch.
  - Context-overflow handling is delegated to llm_client.py's retry
    layer; the daemon does not care.

Stop the daemon: kill -INT <daemon_pid>. Currently-running experiment
keeps going.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PYTHON = "/mnt/data/anonymous/envs/cuda-debugger-exp/bin/python"
TS_RE = re.compile(r"^\d{8}_\d{6}$")


def find_resume_dir(config_path: str) -> Path | None:
    """Look for the latest partial run dir for this config that we can resume.

    Reads `experiment.output_dir` from the YAML, scans for timestamped subdirs
    with index/manifest.jsonl present, returns the newest. Returns None on no
    candidate (fresh run).
    """
    out_root = None
    try:
        for line in Path(config_path).read_text().splitlines():
            m = re.match(r"^\s*output_dir:\s*(\S+)\s*$", line)
            if m:
                # Strip optional YAML quotes (single or double)
                raw = m.group(1).strip().strip("'\"")
                out_root = REPO / "evaluation" / raw
                break
    except OSError:
        return None
    if out_root is None or not out_root.exists():
        return None
    candidates = []
    for child in out_root.iterdir():
        if child.is_dir() and TS_RE.match(child.name) and (child / "index" / "manifest.jsonl").exists():
            candidates.append(child)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.name)


def read_queue(queue_path: Path) -> list:
    out = []
    if not queue_path.exists():
        return out
    for line in queue_path.read_text().splitlines():
        ln = line.strip()
        if not ln or ln.startswith("#"):
            continue
        out.append(ln)
    return out


def pop_first_unstarted(queue_path: Path, status_log_path: Path) -> str | None:
    """Read queue, find first config that's neither alive nor recently-completed.

    A config is skipped if EITHER:
      - its recorded pid is currently alive (still running), OR
      - its most-recent status_log entry has exit_code=0 AND the queue file
        hasn't been modified since that completion (success → don't relaunch)

    To force relaunch of a completed config, touch the queue file.
    Crashed children (exit_code != 0 or missing) are eligible for relaunch.
    """
    queue = read_queue(queue_path)
    alive_started = set()
    completed = {}  # config -> latest_completion_ts (iso str)
    if status_log_path.exists():
        for line in status_log_path.read_text().splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            cfg = e.get("config")
            if not cfg:
                continue
            pid = e.get("pid")
            if pid and os.path.exists(f"/proc/{pid}"):
                alive_started.add(cfg)
            if "exit_code" in e and e["exit_code"] == 0:
                ts = e.get("completed_at", "")
                if ts > completed.get(cfg, ""):
                    completed[cfg] = ts
    try:
        queue_mtime_iso = datetime.fromtimestamp(queue_path.stat().st_mtime).isoformat()
    except OSError:
        queue_mtime_iso = ""
    for cfg in queue:
        if cfg in alive_started:
            continue
        # Skip if completed AFTER queue file was last modified
        if cfg in completed and completed[cfg] > queue_mtime_iso:
            continue
        return cfg
    return None


def append_log(status_log_path: Path, entry: dict) -> None:
    with open(status_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def is_alive(pid: int) -> bool:
    """True iff pid exists AND is not a zombie.

    Daemon-launched children become zombies on exit because we never call
    waitpid() (we only stored proc.pid, not the Popen object). Plain
    `os.kill(pid, 0)` reports zombies as alive — which would deadlock the
    queue. Read /proc/<pid>/status to spot State=Z, and reap so it doesn't
    linger.
    """
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("State:"):
                    if line.split()[1] == "Z":
                        try:
                            os.waitpid(pid, os.WNOHANG)
                        except OSError:
                            pass
                        return False
                    break
    except FileNotFoundError:
        return False
    return True


def launch(config_path: str, gpu: int, label: str, log_dir: Path,
           visible_gpus: str | None = None, extra_args: list[str] | None = None) -> subprocess.Popen:
    """Launch run_experiment.py for the given config. Return Popen object."""
    log_path = log_dir / f"{Path(config_path).stem}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = visible_gpus if visible_gpus else str(gpu)
    env["PYTHONPATH"] = str(REPO)
    cmd = [
        PYTHON,
        str(REPO / "evaluation/scripts/run_experiment.py"),
        "--config", config_path,
        "--mode", "debug",
        "--v2-task-list", str(REPO / "evaluation/v2_task_list.json"),
    ]
    if extra_args:
        cmd += extra_args
    resume_dir = find_resume_dir(config_path)
    if resume_dir is not None:
        cmd += ["--resume", str(resume_dir)]
        print(f"[{datetime.now()}] resuming {Path(config_path).name} from {resume_dir.name}", flush=True)
    fh = open(log_path, "ab")
    proc = subprocess.Popen(
        cmd, env=env, cwd=str(REPO),
        stdout=fh, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return proc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True, help="Path to queue file (one config per line)")
    ap.add_argument("--gpu", type=int, required=True, help="GPU for compile + perf")
    ap.add_argument("--label", required=True, help="Daemon label (used in logs)")
    ap.add_argument("--watch-pid", type=int, default=None,
                    help="Wait for this PID to exit before launching first config")
    ap.add_argument("--poll-sec", type=int, default=60, help="Poll interval")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="Max concurrent run_experiment children (different configs in parallel)")
    ap.add_argument("--visible-gpus", type=str, default=None,
                    help="Override CUDA_VISIBLE_DEVICES for children (e.g. '6,1'). Defaults to --gpu.")
    ap.add_argument("--extra-args", type=str, default=None,
                    help="Extra args appended to each run_experiment.py child command (e.g. '--n-workers 2 --gpus 6,1').")
    args = ap.parse_args()
    extra_args_list = args.extra_args.split() if args.extra_args else None

    queue_path = Path(args.queue)
    log_dir = Path(f"/tmp/queue_daemon_{args.label}_runs")
    log_dir.mkdir(exist_ok=True)
    status_log = Path(f"/tmp/queue_daemon_{args.label}_status.jsonl")

    print(f"[{datetime.now()}] daemon {args.label} started, queue={queue_path}, gpu={args.gpu}, concurrency={args.concurrency}", flush=True)

    if args.watch_pid:
        print(f"[{datetime.now()}] waiting for initial PID {args.watch_pid} to exit", flush=True)
        while is_alive(args.watch_pid):
            time.sleep(args.poll_sec)
        print(f"[{datetime.now()}] watch PID {args.watch_pid} exited", flush=True)

    # Track running children: list of (config, pid, popen)
    running: list[tuple[str, int, subprocess.Popen]] = []

    while True:
        # Reap dead children — record exit_code so completed configs aren't relaunched
        survivors: list[tuple[str, int, subprocess.Popen]] = []
        for cfg, pid, proc in running:
            rc = proc.poll()
            if rc is None and is_alive(pid):
                survivors.append((cfg, pid, proc))
            else:
                if rc is None:
                    rc = proc.wait(timeout=1)
                append_log(status_log, {
                    "config": cfg,
                    "completed_at": datetime.now().isoformat(),
                    "pid": pid,
                    "exit_code": rc,
                })
                print(f"[{datetime.now()}] child PID {pid} ({Path(cfg).name}) exited rc={rc}", flush=True)
        if len(survivors) < len(running):
            print(f"[{datetime.now()}] {len(running) - len(survivors)} child(ren) exited; {len(survivors)} still running", flush=True)
        running = survivors

        # Fill empty slots up to concurrency
        while len(running) < args.concurrency:
            next_cfg = pop_first_unstarted(queue_path, status_log)
            if next_cfg is None:
                break
            print(f"[{datetime.now()}] launching {next_cfg}", flush=True)
            try:
                proc = launch(next_cfg, args.gpu, args.label, log_dir,
                              visible_gpus=args.visible_gpus, extra_args=extra_args_list)
            except Exception as e:
                print(f"[{datetime.now()}] launch FAILED for {next_cfg}: {e}", flush=True)
                append_log(status_log, {
                    "config": next_cfg,
                    "started_at": datetime.now().isoformat(),
                    "pid": None,
                    "launch_error": str(e),
                })
                continue
            append_log(status_log, {
                "config": next_cfg,
                "started_at": datetime.now().isoformat(),
                "pid": proc.pid,
            })
            print(f"[{datetime.now()}] launched PID {proc.pid}", flush=True)
            running.append((next_cfg, proc.pid, proc))

        # Exit when nothing running and queue is exhausted
        if not running:
            print(f"[{datetime.now()}] queue drained, daemon exiting", flush=True)
            return

        time.sleep(args.poll_sec)


if __name__ == "__main__":
    main()
