import os
import sys
sys.path.insert(0, "/home/li004074/AutoFigure")

from autofigure import AutoFigureAgent, Config
from dotenv import load_dotenv

load_dotenv("/home/li004074/CUDABeaver/.env")

api_key = os.environ.get("OPEN_API_KEY2")
if not api_key:
    print("ERROR: OPEN_API_KEY2 not found in .env")
    sys.exit(1)

config = Config(
    generation_api_key=api_key,
    generation_provider="openrouter",
    generation_base_url="https://api.openai.com/v1",
    generation_model="gpt-5.5",
    output_dir="./CUdebugger-paper/figure/autofigure_output",
)

agent = AutoFigureAgent(config)

description = """
Generate a publication-ready scientific illustration (NeurIPS/ICLR style) showing the Context Compaction Pipeline in a multi-agent CUDA debugging system.

Story: During long-horizon debugging, the raw repair transcript grows to ~15K tokens per iteration—mixing useful evidence with obsolete hypotheses, failed patches, repeated logs, and intermediate reasoning. The compaction pipeline applies four layers to compress this into ~2K structured tokens stored in a shared blackboard, implementing an attention-style graded memory: recent evidence stays at full resolution, distant evidence is kept in a lower-resolution but semantically structured form.

Layout: A top-to-bottom or left-to-right flow diagram. Use rounded-corner boxes with colored headers (dark header bar, light body), clean directional arrows, and nested detail panels where needed. The diagram should show the raw transcript entering from one end, passing through four compaction layers in sequence, and producing the compact blackboard state at the other end. Use progressively narrowing visual elements (e.g., narrower bars, tapering flow, or decreasing density) to convey compression across the stages.

The four layers (show each as a distinct processing stage with a brief annotation):

Layer 1 — Tool-Result Summarization: Each tool result exceeding a character cap C_tool is replaced by a model-generated, type-aware summary. Show a small inset or annotation illustrating the type-specific preservation rules:
- Source -> function signatures, kernel names, key constants, line numbers
- Diagnostic -> error kinds, faulting thread/block coordinates, addresses, error counts
- Build/Test -> distinct compiler errors, failing assertions, expected-vs-actual values

Layer 2 — Intra-Iteration Attention Window: Within the current repair iteration, the most recent K=6 tool-call groups are kept verbatim (high resolution). Older groups from the same iteration are folded into an "investigation summary" recording: what was inspected, what concrete facts were found, and what remains unknown. Visually distinguish the verbatim window (full detail) from the folded region (compressed).

Layer 3 — Cross-Iteration Folding: Completed iterations older than the most recent one are replaced by "previous-attempt summaries." Each summary preserves five fields: attempted fix, verified localization, patch intent, validation outcome, and failure reason. The most recent completed iteration stays verbatim because the next agent typically reacts to that feedback directly. Show multiple past iterations collapsing into compact summary blocks.

Layer 4 — Token Budget Nudge: The system estimates total context size. At a warning threshold (>=80%), it nudges the agent to move toward a concrete action. At a force threshold (>=95%), it compacts aggressively and requests immediate localization or patch submission. Show this as a gauge or threshold indicator.

Output: The final compacted transcript feeds into the shared ErrorBlackboard (~2K structured tokens) with five typed slots: Bug State, Diagnosis, Attempt History, Performance, Context Summary.

Visual style: Clean academic illustration. Light pastel backgrounds for processing stages, darker colored headers for stage titles. Minimal text, maximum clarity. No emojis. Sans-serif font.
"""

print("Starting AutoFigure generation...")
print(f"Model: {config.generation_model}")
print(f"Output dir: {config.output_dir}")

result = agent.generate(
    description=description,
    max_iterations=5,
    output_format="svg",
    topic="paper",
)

if result.success:
    print(f"\nSUCCESS!")
    print(f"SVG: {result.svg_path}")
    print(f"Preview: {result.preview_path}")
    print(f"Score: {result.final_score}/10")
else:
    print(f"\nFAILED: {result.error}")
