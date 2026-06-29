# Evaluation Protocol

See companion paper Section 3 (Benchmark Design). Summary:

- Default protocol: K=5 iterations, H=4 history rounds, feedback level L3
  (G+B+T aware), iterative method, T=0.7
- 5 evaluation axes:
  - **iter** (Exp 1): single iterative run per (model, task)
  - **repeated** (Exp 2): K independent attempts, T=1.0, no inter-attempt feedback
  - **feedback** (Exp 3): L0=none, L1=G, L2=G+B, L3=G+B+T aware, L3-raw, L4=G+B+T+P (ncu)
  - **history** (Exp 4): H1, H2, H3, H4 (default)
  - **fewshot** (Exp 5): {matched, random, curated} x K {1, 3, 5}

5-bucket error categories: compile_error, logic_error, memory_crash,
perf_broken, timeout.

Pass criterion: pass@k with performance gate p=0.7
(candidate_runtime <= reference / 0.7).
