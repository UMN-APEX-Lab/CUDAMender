# Benchmark task list

`task_list.json` in this directory is a legacy-format task list reconstructed
from the dataset's `manifest.json` and per-instance `instance.json` files.
It is consumed by `harness.run_experiment --mode debug` via the
`--v2-task-list` argument to look up each cfg-stem's `broken_start` metadata.
See `docs/REPRODUCE.md` for the exact command.
