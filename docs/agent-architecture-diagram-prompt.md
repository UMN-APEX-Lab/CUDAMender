# Agent Architecture Diagram Prompt

Use the following prompt to generate an architecture diagram focused only on
the current blackboard-based CUDABeaver agent system.

```text
Create a clean software architecture diagram for a blackboard-based CUDA
benchmark debugging agent system. The diagram should show system components
and interactions, not individual source files.

Style:
- Technical architecture diagram, not marketing.
- Use rectangular modules, grouped by layer.
- White background, high contrast, readable labels.
- Use arrows to show control flow and data flow.
- Keep it compact but complete.
- Use different colors for: experiment driver, blackboard workflow, shared memory, tools/runtime, external systems, artifacts.
- Do not list source filenames as boxes. Use component names and short responsibility labels.

Diagram content:

Top layer: "Experiment Orchestrator"
- Responsibilities: load benchmark tasks, read run configuration, initialize the LLM backend, schedule tasks, collect results.
- Arrow to "Blackboard Debugging Agent".

Main workflow layer: "Blackboard Multi-Role Agent"
- Arrange roles left to right as the main horizontal flow:
  Analyzer -> Executor -> Verifier -> Coder -> Submission/Evaluation
- Add a Router after failed Submission/Evaluation.
- Router has two loop-back arrows:
  - retry implementation -> Coder
  - reanalyze diagnosis -> Analyzer / Executor / Verifier

Roles:
- Analyzer
  Label: static reasoning over prompt, current code, testbench, and static analysis.
- Executor
  Label: runtime evidence from build, test, sanitizer, and logs.
- Verifier
  Label: merges evidence, identifies bug type, suspected lines, and one next fix.
- Coder
  Label: produces the complete revised solution file.
- Router
  Label: after failed patch, decide retry same fix or reanalyze.

Shared memory component:
- Put a large central component under the roles: "Shared Error Blackboard"
- Contents:
  suspected lines, static flags, runtime trace, verifier report,
  error type, next fix, tried strategies, repeated-failure alerts.
- Draw bidirectional arrows between Analyzer, Executor, Verifier, Coder, Router and the Shared Error Blackboard.

Tool layer: "Restricted Tool Interface"
- Label: controlled actions with schemas, quotas, result caps, and trace logging
- List tools inside:
  read task/code/test context,
  run static checks,
  build and test candidate,
  run sanitizer,
  locate bug,
  edit or submit full solution.
- Arrows from Analyzer, Executor, and Coder to Restricted Tool Interface.
- Verifier mainly reads the Shared Error Blackboard and calls the LLM to produce a structured diagnosis.

Runtime/evaluation layer:
- Component: "Task Workspace"
  Label: seeded with broken_start, current candidate code, copied testbench.
- Component: "Tool Runtime"
  Label: dispatches tool actions, mutates candidate code, records tool traces.
- Component: "CUDA Build/Test Sandbox"
  Label: compile, run tests, capture logs.
- Component: "Diagnostics"
  Label: sanitizer and static/runtime debugging signals.
- Component: "Result Classifier"
  Label: compile error, memory crash, logic error, timeout, pass.
- Component: "Performance Gate"
  Label: optional speedup threshold check.
- Arrows:
  Restricted Tool Interface -> Tool Runtime
  Tool Runtime -> Task Workspace
  Tool Runtime -> CUDA Build/Test Sandbox
  CUDA Build/Test Sandbox -> Diagnostics
  CUDA Build/Test Sandbox -> Result Classifier
  Result Classifier -> Performance Gate
  Result Classifier / Performance Gate -> Submission/Evaluation

LLM and context layer:
- Component: "LLM Backend"
  Label: OpenAI-compatible or vLLM model, native tool calls or JSON action protocol.
- Component: "Context Manager"
  Label: sliding window, cross-iteration compaction, token budget, summaries.
- Arrows:
  Analyzer/Executor/Verifier/Coder/Router -> LLM Backend
  Analyzer/Executor/Verifier/Coder/Router -> Context Manager
  Context Manager loops back into role prompts.

External inputs:
- Benchmark data:
  task prompt, input metadata, broken start code, testbench, reference.
- Run configuration:
  model endpoint, workflow settings, tool budgets, performance gate, timeouts.
- LLM service:
  OpenAI-compatible API or vLLM server
- GPU/CUDA toolchain:
  nvcc, test binary, compute-sanitizer, CUDA runtime

Artifacts/output:
- Component: "Run Artifacts"
  Contents: per-iteration manifest, candidate code snapshots, build/test logs, tool traces, category indexes, aggregate summary.
- Arrows from Experiment Orchestrator, Tool Runtime, and Submission/Evaluation to Run Artifacts.

Important annotations:
- The agent system is debug-only: it starts from broken_start, not from scratch generation.
- Show only the blackboard workflow; do not include the legacy ReAct workflow.
- Tools are intentionally narrow; no arbitrary shell.
- Final candidate must pass build/test/classifier and optional performance gate.
- Main visual emphasis should be the shared blackboard coordinating specialized roles, not the implementation file layout.
```
