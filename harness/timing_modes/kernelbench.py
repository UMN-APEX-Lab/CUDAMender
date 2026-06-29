"""Parse the repr of a kernelbench KernelExecResult to extract timing.

Sample input (from print(result)):
    compiled=True correctness=True metadata={...}
      runtime=11.0
      runtime_stats={'mean': 11.0, 'std': 5.13, 'min': 8.84, 'max': 26.1, 'num_trials': 10}
      ref_runtime=10.4
      ref_runtime_stats={...}
"""
import ast
import re


class KernelbenchEvalResult:
    def wrap_command(self, cmd: str, workdir: str) -> str:
        return cmd  # kernelbench prints stats directly; no wrapping

    def parse(self, stdout: str, stderr: str) -> dict:
        # Match runtime_stats={...} (NOT ref_runtime_stats — use negative lookbehind)
        m = re.search(r"(?<!ref_)runtime_stats\s*=\s*(\{[^}]*\})", stdout)
        if not m:
            raise ValueError(
                f"could not find runtime_stats in stdout:\n{stdout[-500:]}"
            )
        stats = ast.literal_eval(m.group(1))
        mean = float(stats.get("mean", 0))
        return {
            "method": "kernelbench_eval_result",
            "mean_ms": mean,
            "p50_ms": mean,  # kernelbench doesn't emit p50; use mean as approximation
            "p95_ms": float(stats.get("max", mean)),
            "stddev_ms": float(stats.get("std", 0)),
            "raw_samples_ms": [],
            "n_trials": int(stats.get("num_trials", 0)),
        }
