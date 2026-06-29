# Guided-repair vs. Baselines — memory_crash & logic_error 实验结果

> 对比新做的 **guided-repair**(blackboard 多角色 agent + 强制 report / 单一 next_fix / verifier 路由 /
> tried-strategy registry)与 benchmark 自带的 **iterative**、**pass@5** baseline,在 memory_crash
> 和 logic_error 两类任务、qwen3.6-27b 与 gemma-4-31b 两个模型上的修复通过率。

## 实验口径(所有实验共用)

- **模式**:debug(从 `broken_start` 起修)。
- **采样温度**:guided-repair / iterative = `temperature 0`;pass@5 = `0.7`(独立采样需多样性)。
- **性能门**:`performance_gate.min_speedup = 0.7`(correctness 通过后才判;多数 memory_crash 任务无 perf
  reference → `skipped_allowed`,即不卡)。
- **任务集**:memory_crash 32 个、logic_error 45 个,均按 **instance-id** 建 flat layout + task list
  (避免 stem 冲突;`/tmp/memcrash_task_list.json`、`/tmp/logic_task_list.json`)。
- **预算**:guided-repair 计 `max_patch_submissions`(提交次数);iterative 计轮数;pass@5 = 5 次独立尝试。
- **端点**:
  - gemma-4-31b-it:本地 vLLM `:8032`(全程)。
  - qwen3.6-27b:**logic_error 全用远端 vLLM** `cs-u-converge.cs.umn.edu:30005`;
    **memory_crash 的 iterative/repeated baseline 用本地 sglang** `:8050`(同模型不同栈,见“注意”)。

## 结果总表

| 类型 | 模型 | workflow | 预算 | 通过 | % | 状态 |
|---|---|---|---|---|---|---|
| memory_crash | qwen3.6-27b | **guided-repair** | 5 | **20/32** | 62 | done |
| memory_crash | qwen3.6-27b | iterative | 5 | 19/32 | 59 | done |
| memory_crash | qwen3.6-27b | pass@5 | 5 | 18/32 | 56 | done |
| memory_crash | gemma-4-31b | **guided-repair** | 5 | **19/32** | 59 | done |
| memory_crash | gemma-4-31b | iterative | 5 | 18/32 | 56 | done |
| memory_crash | gemma-4-31b | pass@5 | 5 | 12/32 | 38 | done |
| logic_error | gemma-4-31b | **guided-repair** | 5 | 19/45 | 42 | done |
| logic_error | gemma-4-31b | **guided-repair** | 10 | **22/45** | 49 | done |
| logic_error | gemma-4-31b | iterative | 10 | 16/45 | 36 | done |
| logic_error | qwen3.6-27b | guided-repair | 5 | 16/45 | 36 | done |
| logic_error | qwen3.6-27b | iterative | 5 | **21/45** | 47 | done |
| logic_error | qwen3.6-27b | pass@5 | 5 | 18/45 | 40 | done |

> 全部 12 个实验已完成。

### memory_crash 按 family(已完成 6 个)

| 模型 | workflow | cuda | cudabench | kernelbench | 总计 |
|---|---|---|---|---|---|
| qwen | guided-repair | 8/15 | 5/5 | 7/12 | 20/32 |
| qwen | iterative | 7/15 | 5/5 | 7/12 | 19/32 |
| qwen | pass@5 | 6/15 | 5/5 | 7/12 | 18/32 |
| gemma | guided-repair | 8/15 | 5/5 | 6/12 | 19/32 |
| gemma | iterative | 5/15 | 5/5 | 8/12 | 18/32 |
| gemma | pass@5 | 3/15 | 5/5 | 4/12 | 12/32 |

## 关键发现

1. **guided-repair 的价值高度任务相关。**
   - **logic_error(gemma)**:guided ≫ iterative —— 22/45 vs 16/45 @10 预算;且 guided 用**半预算**
     (5)就赢 iterative 双预算(19 vs 16)。逻辑 bug 隐蔽、需反复定位换方向,结构化工作流收益大。
   - **memory_crash**:三种 workflow 接近(qwen 20/19/18,gemma 19/18/12)。瓶颈在“模型能否写对硬核内核”
     (tensor-core MMA / 卷积布局 / 手写 PTX),编排撬不动。

2. **工作流收益还与模型相关(反转)。**
   - gemma logic:guided > iterative。
   - **qwen logic:iterative 反超 guided(21 vs 16)**。提示 guided-repair 的多角色 + JSON-action 结构
     对 qwen3.6 不如对 gemma 契合(待深挖:是否预算被多角色分散 / JSON parse 失败 / 思考发散)。

3. **pass@5 整体最弱**,尤其 **gemma memory_crash(12/32)**——独立重采样对需要逐步定位的 bug 帮助有限;
   qwen 采样更发散,repeated 相对没那么差。

4. **guided-repair 5→10 预算**(gemma logic):19→22,有增益但边际递减。

## memory_crash 失败模式分析(gemma,11 个 guided/iterative 共同失败)

- **7/11:崩溃修好但结果错**(最终 `logic_error`)——硬核内核数值逻辑写不对(cuda_144 m16n8k8 MMA、
  kb_59/60/86 卷积布局、cuda_227 odd-even sort 等);per-iter category 显示 `illegal_memory_access →
  functional_correctness_bug`,即 agent 消除了崩溃但产出错误结果。
- **2/11:卡在编译**(cuda_173 手写 `mma.sync` PTX 内联汇编、cuda_1 未定义标识符)。
- **2/11:崩溃始终没修好**(cuda_21 sliding-window 越界、kb_81 atomic scatter)。
- 启示:对“崩溃修好但结果错”这类,**加编排无用**,应给 agent 正确性验证手段(→ 方案 A 自生成测试)。

## 注意 / 已知混淆变量

- **qwen memory_crash baseline 用本地 :8050 sglang,而 qwen memory_crash guided-repair 用远端 :30005
  vLLM** —— 同模型不同 serving 栈,对比 qwen memcrash 三方时是已知混淆。logic_error 的 qwen 三方全用
  :30005,无此问题;gemma 全程 :8032,无此问题。
- **iter-5old(7/43)** 是更早一次用旧 stem 任务集(43 个)跑的 gemma logic iterative-5,与本表 instance-id
  集(45)不完全对齐,未纳入主表(仅作历史参考)。
- temp 0 ≠ 完全可复现:thinking 模型经 vLLM/sglang 即使 temp 0 也有非确定性(曾见同任务重跑结果翻转)。

## 不同 performance gate 下的成功率(%)

回溯式重打分:所有实验**实际都在 gate=0.7 下跑**,这里拿它们产出的解重新套用 gate 0.5 / 0.7 / 1.0
(成功 = 存在一个 correctness 通过的解,且其 speedup ≥ gate,或该任务无 perf 参考→任何 gate 都按 correctness 判)。

| 类型 | 模型 | workflow | n | @0.5 | @0.7 | @1.0 |
|---|---|---|---|---|---|---|
| memory_crash | qwen3.6-27b | guided-repair | 32 | 62% | 62% | 62% |
| memory_crash | qwen3.6-27b | iterative-5 | 32 | 59% | 59% | 41% |
| memory_crash | qwen3.6-27b | pass@5 | 32 | 56% | 56% | 38% |
| memory_crash | gemma-4-31b | guided-repair | 32 | 59% | 59% | 59% |
| memory_crash | gemma-4-31b | iterative-5 | 32 | 56% | 56% | 44% |
| memory_crash | gemma-4-31b | pass@5 | 32 | 41% | 38% | 34% |
| logic_error | gemma-4-31b | guided-repair-5 | 45 | 42% | 42% | 38% |
| logic_error | gemma-4-31b | guided-repair-10 | 45 | 51% | 49% | 44% |
| logic_error | gemma-4-31b | iterative-10 | 45 | 40% | 36% | 24% |
| logic_error | qwen3.6-27b | guided-repair-5 | 45 | 36% | 36% | 29% |
| logic_error | qwen3.6-27b | iterative-5 | 45 | 47% | 47% | 29% |
| logic_error | qwen3.6-27b | pass@5 | 45 | 44% | 40% | 29% |

观察:
- **0.5 ≈ 0.7**:几乎所有正确解的 speedup 本就 ≥0.7,卡在 0.5–0.7 的极少。
- **1.0 是分水岭,且对 guided 友好**:收紧到 1.0,iterative/repeated 掉得多(gemma iter-10 36%→24%;qwen
  iter-5 47%→29%),guided 掉得少甚至不掉(memcrash guided 不变;gemma guided-10 49%→44%)——guided 的修复更
  常达到 ≥1.0×。
- **memcrash 多与 gate 无关**(cuda/cudabench 无 perf 参考→skipped);仅 kernelbench 子集在 1.0 时显现差异。

> ⚠️ **方法学警告**:`@1.0` 是**低估**——agent 在 speedup 达 0.7 即判过并停止优化,真用 `min_speedup: 1.0`
> 重跑会继续往 ≥1.0 推,实际成功率更高。`@0.5` 基本准。要拿真实的 1.0-gate 成功率需用 `min_speedup: 1.0` 重跑。

## 数据位置

- memory_crash:`/tmp/memcrash_all_guided`、`/tmp/memcrash_all_gemma`、
  `/tmp/memcrash_baseline_{qwen,gemma}_{iterative,repeated}`
- logic_error:`/tmp/logic_gemma`、`/tmp/logic_gemma_guided10`、`/tmp/logic_gemma_iterative10`、
  `/tmp/logic_qwen_remote`、`/tmp/logic_qwen_iterative5`、`/tmp/logic_qwen_repeated5`
- 每个目录 `<timestamp>/index/manifest.jsonl`,用 `sample_id` 去 `__iter_NNNN` 后缀区分同 stem 多实例。
