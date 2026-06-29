# Smoke tests

Run from the repo root:

```bash
pytest tests/test_smoke.py -v
```

These verify the harness loads correctly and a classifier can be invoked,
without requiring an LLM API key. For a full single-task end-to-end run, see
`examples/` and `docs/REPRODUCE.md`.
