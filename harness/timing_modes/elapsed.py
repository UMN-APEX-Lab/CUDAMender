"""Wall-clock elapsed time parser.

Looks for `Elapsed: X.XXX s` or `Time: X.XXX ms` in stdout. If neither is
found, returns NaN — caller should handle.
"""
import re


class TotalElapsed:
    def wrap_command(self, cmd: str, workdir: str) -> str:
        # Wrap with /usr/bin/time -p to get an "Elapsed" line
        return f"/usr/bin/env bash -c 'start=$(date +%s.%N); {cmd}; end=$(date +%s.%N); echo Elapsed: $(echo $end - $start | bc) s'"

    def parse(self, stdout: str, stderr: str) -> dict:
        text = stdout + "\n" + stderr
        m_s = re.search(r"Elapsed:\s*([\d.eE+-]+)\s*s\b", text)
        m_ms = re.search(r"Time:\s*([\d.eE+-]+)\s*ms\b", text)
        if m_s:
            mean_ms = float(m_s.group(1)) * 1000.0
        elif m_ms:
            mean_ms = float(m_ms.group(1))
        else:
            mean_ms = float("nan")
        return {
            "method": "total_elapsed",
            "mean_ms": mean_ms,
            "p50_ms": mean_ms,
            "p95_ms": mean_ms,
            "stddev_ms": 0.0,
            "raw_samples_ms": [],
        }
