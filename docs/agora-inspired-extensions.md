# 借鉴 Agora 的三项 Agent 扩展：自生成测试 / Memory / 通信简化

> 状态：**方案讨论中，尚未改动任何代码**。本文档对齐 Agora（NeurIPS/ICML 2026 投稿）的三项设计，
> 映射到 CUDABeaver 的 debug-only agent 子系统，并列出待决策点。
> 相关代码：`harness/agent_loop.py`、`harness/agent_tools.py`、`harness/cudeubgger/`、
> `harness/context_manager.py`、`harness/debug_assist.py`、`configs/agent/*.yaml`。

---

## 0. 根本差异：为什么不能照搬 Agora

| 维度 | Agora | CUDABeaver |
|---|---|---|
| 任务 | 开放代码库的**未知 bug 发现**（zero-day） | 固定 testbench 下的**已知 broken 内核修复** |
| Oracle | **没有** ground truth，必须自造测试当 oracle（HDT 四元组 H=(C,A,E,O)，O 就是断言） | testbench 已是 oracle（pass/fail + mismatch/sanitizer 输出） |
| 测试生成的价值 | 充当判定真伪的唯一手段 | **廉价验证假设 + 定位 + 正式提交前预检**，不替代官方判定 |
| Memory 的风险 | 纯收益（跨 repo 复用知识） | 跨任务记忆 = **任务间信息泄漏**，必须作为受控实验轴 |
| 通信 | 从零设计三 agent | 已有 blackboard hub，只需做"精简化"增量 |

**结论**：三项都要换动机落地，不能直接搬。

Agora 关键证据（用于排优先级）：ablation 中去掉 `reflection-loop`（自测试迭代）→ **0 bug**；
去掉 `state-analyzer`（防重复探索）→ **0 bug 且 token 飙到 79.6M**；去掉 `scenario-generator` →
回退到 baseline，token 233M（×3.98）。即"自测试"与"防重复"是命门组件。

---

## 方案 A：Agent 自生成测试（对应 Agora TestGen + reflection loop）

### 动机
testbench 的反馈往往很粗（`mismatch at element N` / sanitizer 崩在某地址）。让 agent 写一个
小型独立 `.cu` micro-test（例如只跑 33 个元素去戳"reduction 尾块未处理"假设），
能在**消耗一次正式 patch submission 之前**便宜地验证/证伪假设、缩小定位、预检修复。
杠杆点：用便宜的诊断测试把昂贵的正式提交命中率拉高。

### 工具设计
在 `agent_tools.py` 的 `TOOL_SCHEMAS` 新增 `run_hypothesis_test`，**强制 HDT 结构**：

```jsonc
{
  "hypothesis":      "string",  // 我怀疑的 bug 是什么
  "expected_if_buggy":"string", // 若 bug 存在，这个测试预期输出什么
  "test_code":       "string"   // 独立 .cu 测试程序，#include "candidate.cu"
}
```

### 执行语义
- 写入 `task_workdir/agent_tests/test_<n>.cu`，用**任务同款 nvcc flags / include path**（复用
  `compiler.py` 的 build dispatch）编译并运行，返回编译+运行输出。
- 候选代码以 `#include "candidate.cu"` 暴露给测试程序；测试程序**只读不写**候选代码，
  写入严格限制在 `agent_tests/` scratch 目录（路径沙箱，同 `read_test_file` 的 suffix 白名单思路）。
- 编译/运行失败 → agent 自行修改自己的测试再跑 = **reflection loop**，在现有工具循环里天然成立，
  只需 quota 约束（建议 `4` 个测试程序 / `8` 次运行）。
- **不动正式判定**：`write_solution` 仍走官方 build+test，自测试纯诊断，**不计入 patch submission**。

### 阶段与 workflow 集成
- LOCATE 阶段可用（取证性质，配合 `locate_bug`）；FIX 阶段可用（`apply_edit` 后、`write_solution` 前预检）。
- react workflow：作为普通工具直接进 registry。
- blackboard workflow：先挂给 `executor`（`executor_tools` 加白名单）；独立 TestGen 角色作为后续可选。

### 副产品
强制 `hypothesis` 字段后，`tool_trace.jsonl` 可直接做"假设质量 × 修复成功率"分析，写进 paper。

### 范围切割（重要）
- **第一期只做 standalone nvcc 任务**。
- Class-1/Class-2 PyTorch-extension 任务（kernelbench / flashattention / layernorm，走
  `applied_kernels_runner/`）的 Python 端 scratch test 列为**第二期**，第一期砍掉。

### 配置
新增 `agent.self_tests.{enabled, max_programs, max_runs}`，作为一条 ablation 轴。

---

## 方案 B：Memory 系统（对应 Agora §3.3/3.4 + TestGen 的 lesson 工具）

Agora memory 两层：静态知识库（repo knowledge + bug patterns，~360KB，启动读一次）+
动态 lessons（`add_lesson` / `find_lessons`，错误→解法对）。按**风险从低到高**分三步：

### B1. 任务内 tried-fix registry（先做，零污染风险）
- 现状：已有 `state.attempt_history` + repeating-failure alert，但失败 patch **无去重签名**。
- 做法：每次失败 patch 取 normalized diff 签名存入 `_LoopState`；每轮以
  "以下修法已试过且失败，禁止重复"注入 coder。
- 对应 Agora 的 state-analyzer（防重复探索）—— ablation 显示这是命门（去掉直接 0 bug、token 暴涨）。

### B2. 静态知识卡（人工策划，小而可控）
- 按 task family 写卡片：cutlass / ThunderKittens / kernelbench / flashattention 各自的构建系统、
  testbench 判对错方式、常见坑。
- 按五类错误写 CUDA debug pattern 清单：漏 `__syncthreads`、reduction 尾块、grid-stride 缺失、
  atomic 误用、bank conflict……
- **污染红线**：pattern 必须是通用 CUDA 知识，**严禁从本数据集的 bug 提炼**（同 `few_shot.py`
  的 leave-one-out 逻辑，否则审稿人判数据泄漏）。

### B3. 跨 run lesson store（作为实验轴，默认关闭）
- 存储：JSONL，键 `(task_family, error_category, error_signature)` → lesson。
- 写入：任务结束时由一个 LLM 蒸馏 pass 从 `attempt_history` + 最终结果提炼。
- 读取：下个任务启动按签名检索注入。
- **必须配套**：leave-one-out 隔离、固定任务顺序（否则结果 order-dependent）、报告开/关两组数据。
- 工程量比 B1/B2 大一个量级 —— 是否真做取决于决策点 ③。

---

## 方案 C：简化角色间通信（对应 Agora §3.6）

Agora 原则：全局状态只在 Orchestrator 一份且精简、worker 无状态、memory 只读一次、只传必要结构化信息。
你的 blackboard 已是 hub，以下是四个低效点与对策：

| 现状问题 | 对策 |
|---|---|
| 四角色各收完整 board dump（≤14K chars），一个 cycle 重复付 4 次 token | `board.view_for(role)`：coder 只拿 verified suspects + fix_suggestions + 最近 test 输出 + tried-fix registry，不拿 analyzer 原始 static flags |
| analyzer/executor 报告是自由文本，coder 要重新理解 | 角色间唯一契约改成小 JSON 产物（verifier 已是 JSON）；自由文本只进 trace 留审计 |
| 工具原始输出直接堆进 `runtime_trace`，board 越滚越大 | **按引用传递**：board 只存指针 + 一两行摘要，全量留在 trace 文件，需要时角色用工具回读 |
| 固定流水线 analyzer→executor→verifier→coder 每轮全跑 | orchestrator 用**确定性代码**做条件路由：compile_error 跳过 sanitizer executor；同一 suspect 第二轮已确认则直接进 coder。省略整段角色调用是最大 token 项 |

度量：manifest + tool_trace 已能算 tokens/task、tokens/solved、submissions used，
analysis pipeline 加一列即可。

---

## 待确认的决策点

**① 优先级**
建议顺序：**A（自测试工具）→ C 的条件路由 + role-scoped view → B1（tried-fix registry）→ B2/B3**。
A 故事性最强（Agora ablation 去掉 reflection-loop 直接 0 bug）。
> 待确认：是否同意先做 A？

**② A 的形态**
做成普通工具（react + blackboard 都受益，改动小），还是 blackboard 里独立 TestGen 角色？
> 倾向：先做工具，角色化作为后续。待确认。

**③ B3 跨 run memory 的定位**
真做进 benchmark 实验轴（需配套隔离协议、固定顺序、双组数据），还是只在论文 discussion 提及？
> 这决定工程量差一个量级。待确认。

**④ PyTorch-extension 任务的自测试（Python 端）是否第一期就要？**
> 建议第一期砍掉，只做 standalone nvcc。待确认。

---

## 确认后产出
方向定了之后再出**文件级改动清单**（具体到函数与 config key）。

---

## 进度（Progress）

### 决策记录（已确认）
- **① 优先级**：先做方案 C（通信优化）。
- **方案 C 落地形态**：config 开关 + **默认关**（保留现状作 ablation 对照）。
- **coder 视图**：**严格**（只给 verifier 产物 + fix_suggestions + 最近一条 runtime 输出）。
- **首批 PR 范围**：Step 1（role-scoped views）+ 度量（token 计量）。

### PR-1：role-scoped views + token 度量 —— ✅ 已实现并测试通过

**Step 1 — role-scoped views（默认关）**
- `harness/cudeubgger/common.py`：
  - `ErrorBlackboard.view_for(role)` + 私有 helper `_compact_static_flags` / `_runtime_summary` /
    `_verified_suspects` / `_latest_runtime`。
  - **coder 严格视图**：error_type + verifier 确认的 suspected_lines + fix_suggestions + 最近一条
    runtime 输出；**不含** analyzer/executor/verifier 自由文本、原始 static_flags、完整 runtime_trace。
  - executor 视图压缩 static_flags；verifier 视图保留全部证据但 excerpt 由 2000 → 800 字符。
  - 未知角色回退完整 dump（向后兼容）。
  - `_scoped_views_enabled(config)` 读 `agent.blackboard.scoped_views`（默认 `False` = 完全保持现状）。
- `harness/cudeubgger/blackboard.py`：新增 `_board_block(board, role, config)`，替换 executor/verifier/
  coder 三处 `to_prompt_block()`。

**Step 2 — token 度量**
- `_LoopState.token_usage`：按角色累计 `{calls, est_prompt_tokens, prompt_tokens, completion_tokens,
  total_tokens}`。
- `_accumulate_token_usage()`：永远记估算 prompt token（chars/4，复用 `context_manager.estimate_tokens`），
  endpoint 返回 `usage` 时叠加真实值；估算逻辑永不抛异常打断主循环。
- 接入 `runtime.py` 两个 LLM wrapper（加 `role` 参数）；verifier 单独估算（`call_llm` 不返回 usage）；
  react 模式自动以 `role="agent"` 计量。
- `harness/llm_client.py`：`call_llm_with_tools` 返回值补 `usage`（原本只有 `call_llm_json_action` 有）。
- 任务结束把 `token_usage`（per_role + total）写入 manifest 最后一条记录（blackboard + react 都写）。

**配置 + 测试**
- `configs/agent/agent_smoke.yaml`：新增带注释的 `agent.blackboard.scoped_views: false`。
- `tests/test_blackboard_views.py`：10 个纯逻辑测试（视图严格性 / static_flags 压缩 / excerpt 裁剪 /
  回退 / config gate / token 估算+真实值+对象形态+汇总），无需 GPU/LLM。
- 全量：新增 10 + 原有 70 测试通过。

**如何验证收益**
同一 config 跑两次（`scoped_views: true` / `false`），对比 manifest 里 `token_usage.total` 与修复率，
即得 efficiency ablation 数据。

### PR-2：结构化角色契约（Step 3）—— ✅ 已实现并测试，⚠️ 真机验证不充分

**决策**：保留 `ruled_out` 字段（Agora state-analyzer 防冗余探索思路）。

**改动**
- `common.py`：`_structured_reports_enabled(config)` 开关（默认关）；`_normalize_structured_report(raw)`
  防御性裁剪（summary≤400、列表各≤4，非 dict 降级为 summary-only，永不抛异常）；
  `_json_role_action_request` 加 `structured_report` 参数；`_RoleResult.report` 改 `str | dict`；
  `ErrorBlackboard.{analyzer,executor}_report` 改 `str | dict | None`。
- `prompts.py`：`STRUCTURED_REPORT_PROMPT` 契约（**扁平字段**：summary/key_points/ruled_out 直接进 args）。
- `blackboard.py`：`_run_tool_role_agent` 条件追加契约说明、请求结构化 schema、归一化 `final_report`；
  **dual-accept** 解析（扁平 args 或嵌套 `report:{}` 都接受）；`_resolve_report()` 统一三处返回。
- `configs/agent/agent_smoke.yaml`：`structured_reports: false`（gated，默认关）。
- 测试：`test_blackboard_views.py` 增至 15 例（normalize 裁剪/降级、gate、request-builder、扁平=嵌套
  dual-accept、view 序列化），全量 85 通过；off-path 措辞与 legacy 逐字一致。

**真机验证发现（重要、诚实记录）**
- cuda_110 单任务（scoped off，structured off→on）：**结果不可下结论**。
  - executor_report 在 off 跑里有值、在 structured 跑里为空 → 说明该跑里 role **没成功 emit final_report**
    （否则会存成 dict）。analyzer_report 在 **off 基线里也是空** → `max_role_turns: 2` 下 final_report
    本就 emit 不稳定，与 Step 3 无关。
  - completion_tokens +483% 属 thinking-mode 方差，N=1 confounded，**token 结论不成立**。
  - 已据此把 schema 由嵌套改扁平 + dual-accept + prompt 加「turns 用尽前务必 final_report」。
- **结论**：机制正确（单测覆盖），但收益依赖 role 真的调用 final_report；需要
  (a) 全量多任务对照测量，(b) 可能调大 `max_role_turns` 或令 final_report 对 role 强制。

### PR-3：guided-repair 工作流（补齐 4 个缺口）—— ✅ 已实现并测试

**目标工作流**（用户定义）：analyzer/executor 静态+运行时定位后**都必须给 report** → verifier 汇总并
**一次只给一个**具体修复方向 → coder 改 → 看结果：修好就结束；没修好 verifier **看本轮代码+定位+策略**
决定「直接换方向」还是「重跑 analyzer+executor」，并**记录已试策略避免重复**。

**决策**：不加开关（直接成为 blackboard 默认行为）；路由用「失败后独立轻量调用」。

**四个缺口的实现**
- **缺口2 强制报告**：`_run_tool_role_agent` 调查轮用尽仍无 report 时，`_force_final_report` 追加一次
  「只允许 final_report」的调用（≤1 次额外调用）。解决之前 `max_role_turns=2` 下 report 常为空的问题。
- **缺口4 单一方向**：verifier 输出改为单个 `next_fix`（dual-accept 旧 `fix_suggestions[0]`）；
  新增 `board.next_fix`；coder view 只给这一个；改 `VERIFIER_SYSTEM_PROMPT`。
- **缺口6 registry**：`_LoopState.current_fix_direction` + attempt_history 记 `fix_direction`；
  `_tried_strategies_prompt` 列出**每个**失败尝试的 `{believed_cause, fix_direction, outcome}`，
  注入 verifier「ALREADY TRIED, do not repeat」（比旧的同签名软提示强）。
- **缺口5b 路由**：`run_task_blackboard_agent` 重构为 guided 循环（`need_analysis` 标志）；
  失败提交后 `_run_route_decision`（`ROUTE_DECISION_SYSTEM_PROMPT`）**输入本轮提交代码 + 定位 +
  next_fix + 最新执行结果 + tried-registry**，输出 `{next_action: retry_fix|reanalyze, diagnosis:
  coder_error|wrong_strategy, reason}`。retry_fix → 跳过 analyzer/executor、复用 board、verifier 带
  `route_diagnosis` 再给方向；reanalyze → 重跑全链。解析失败/ API 错 → 安全默认 reanalyze。

**测试**：全量 98 通过。新增路由解析+默认、tried-registry、next_fix 视图、强制报告等单测；
更新 perf-gate 重试测试（失败后多一次路由响应）。token 计量新增 `router` 角色。

### 后续（待做）
- **全量对照工具**（跑整个 task list，输出 tokens/task、tokens/solved），验证 Step 1/3 + guided-repair。
- 方案 A（自生成测试）/ 方案 B（memory，registry 已是 B1 雏形）。
