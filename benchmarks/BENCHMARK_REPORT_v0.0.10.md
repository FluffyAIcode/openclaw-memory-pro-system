# Memory Pro System — Benchmark 总报告 v0.0.10

> 版本：v0.0.10
> 日期：2026-04-05
> LLM Provider: xAI Grok-4-1-fast-non-reasoning (Collision/Inference), DeepSeek (LongMemEval)

---

## 1. 测试矩阵总览

本次 benchmark 包含两套互补的测试框架，共覆盖 **600+ 测试用例**：

| 框架 | 测试用例 | 覆盖维度 | 得分 |
|------|---------|---------|------|
| **LongMemEval** (外部标准) | 500 | Recall 准确率、时序推理、知识更新、多会话推理 | 38.6% |
| **Extended Benchmark** (内部) | 100 | Skill 抽象、灵感碰撞涌现、KG 稳定性、全管线集成 | 88.0% |

### 为什么需要两套测试？

LongMemEval 是学术界标准 benchmark，测的是 **"记忆能不能被找回来"**——这只是记忆系统的基础能力。

真正的记忆系统竞争力在于：
- **Knowledge Graph 稳定性** — 事实之间的关系能否正确传播、矛盾能否被检测
- **Skill 抽象能力** — 能否从对话中提炼可复用的程序化知识
- **Intuitive 涌现能力** — 碰撞引擎能否从不相关记忆中产生有价值的灵感
- **Anti-hacking 安全性** — PII 过滤、投毒检测、安全门控

---

## 2. LongMemEval 测试结果

**数据集**: Oracle Dataset (500 questions), **耗时**: 47 分钟

| 能力维度 | 准确率 | 评价 |
|---------|--------|------|
| Information Extraction | 46.8% | 单会话事实提取，基本合格 |
| Temporal Reasoning | 44.4% | 时序推理，合格 |
| Multi-Session Reasoning | 34.6% | 跨会话聚合，需改进 |
| Knowledge Update | 19.2% | 知识更新识别，**严重不足** |

### 根因分析

Knowledge Update 得分低的核心原因：系统尚未实现 **时间戳感知的 reranking** 和 **新旧信息冲突标注**。当同一实体有多个版本时，recall 无法优先返回最新版本。

---

## 3. Extended Benchmark 测试结果

### 3.1 总分: 88/100 (88%)

| 模块 | 得分 | 通过率 |
|------|------|--------|
| **Skill Registry** (35 cases) | 30/35 | 86% |
| **Collision Engine** (35 cases) | 33/35 | 94% |
| **E2E Integration** (30 cases) | 25/30 | 83% |

### 3.2 模块 A: Skill Registry (30/35)

测试 Skill 从学习、匹配、到生命周期管理的全链路。

**通过的关键能力：**
- ✅ SK-01~10: 基础注册（含中文、长内容、批量、tags、prerequisites）— 10/10
- ✅ SK-11,13,14: 精确语义匹配、跨语言匹配、关键词回退 — 3/3
- ✅ SK-17~19: L1 注入验证、merged 输出、procedures 格式 — 3/3
- ✅ SK-20~22: DRAFT/DEPRECATED 过滤、5上限控制 — 3/3
- ✅ SK-26~30,32,34,35: 生命周期（promote、cooldown、feedback、版本递增、stats）— 8/8

**失败用例分析：**

| Case | 问题 | 根因 | 严重度 |
|------|------|------|--------|
| SK-12 | 同义词匹配失败 | embedding 模型对中文"ping不通"→"网络调试"语义距离偏大 | Medium |
| SK-16 | 不相关查询仍返回 skill | 缺少 match_score 阈值过滤 | High |
| SK-24 | 偏好类 skill 匹配弱 | "编辑器主题"与"深色主题"语义关联不够强 | Low |
| SK-31 | rewrite 后 uses 重置 | `_trigger_rewrite` 重置了 `total_uses` 导致条件不满足 | Medium |
| SK-33 | deprecated 可被 re-promote | `promote()` 未检查 DEPRECATED 状态 | **Critical** |

### 3.3 模块 B: Collision Engine (33/35)

测试 7 种碰撞策略、insight 质量、跨层发现能力。

**通过的关键能力：**
- ✅ CL-03~14: 5/7 策略覆盖（chronos_crossref, digest_bridge, dormant_revival, temporal_echo, contradiction_based, blind_spot_based）
- ✅ CL-15~25: Insight 质量全部通过（novelty 范围、connection/ideas 非空、emotional_relevance、自碰撞过滤、空/单条目处理、markdown 格式、持久化、轮次限制）— 11/11
- ✅ CL-26~35: 跨层发现全部通过（Chronos/Digest 跨层、HTTP collide、KG 矛盾传播、线索发现、多策略轮次、attention focus、deep_collide、insight 可召回）— 10/10

**失败用例分析：**

| Case | 问题 | 根因 | 严重度 |
|------|------|------|--------|
| CL-01/02 | semantic_bridge 未触发 | 种子数据 embedding 相似度未达到 semantic_bridge 的阈值 | Low |

### 3.4 模块 C: E2E Integration (25/30)

测试 Skill + KG + Evidence 三层协同、安全门控、预算控制。

**通过的关键能力：**
- ✅ E2E-01,02,06~10: L1 Skill 注入（Docker/PyPI skill、多skill匹配、executable_prompt、DRAFT 过滤、budget 控制）— 7/7
- ✅ E2E-11~20: L2/L4 KG Relations（GraphQL/Rust 召回、锻炼冲突检测、supports/contradicts 验证、推理引擎矛盾扫描）— 10/10
- ✅ E2E-21,22,25~30: Full Pipeline（三层协同、碰撞 insight 召回、intent 分类、PII 检测、budget 控制、fallback、feedback 闭环、**管线延迟 460ms**）— 8/8

**失败用例分析：**

| Case | 问题 | 根因 | 严重度 |
|------|------|------|--------|
| E2E-03/04 | 偏好类 skill 匹配弱 | 与 SK-24 同根因——embedding 语义距离 | Low |
| E2E-05 | 不相关查询注入 skill | 与 SK-16 同根因——缺少阈值 | High |
| E2E-23 | 知识更新识别失败 | seed 数据用 skip_hooks 绕过了完整索引 | Medium |
| E2E-24 | 多会话聚合不完整 | 同上 | Medium |

---

## 4. 发现的关键问题（按严重度排序）

### P0 — Critical

1. **SK-33: Deprecated Skill 可被 Re-promote**
   - `SkillRegistry.promote()` 未检查 `DEPRECATED` 状态
   - 修复方案：添加 `if skill.status == SkillStatus.DEPRECATED: return None`

### P1 — High

2. **SK-16/E2E-05: Skill 匹配缺少相关度阈值**
   - 不相关查询（如"今天天气"）仍返回 5 个 skill
   - 修复方案：在 `context_composer.py` 的 skill 注入阶段增加 `min_score` 阈值

3. **Knowledge Update 全面薄弱**（LongMemEval 19.2%）
   - 修复方案：recall 阶段增加时间戳感知 reranking，新信息权重 boost

### P2 — Medium

4. **SK-31: Skill Rewrite 后 uses 计数重置**
   - `_trigger_rewrite` 更新内容后重置了 feedback 计数器
   - 修复方案：rewrite 时保留 historical 统计

5. **SK-12: 中文同义词匹配精度不足**
   - "ping不通如何排查" 匹配到 "用户访谈技巧"
   - 修复方案：考虑补充关键词 fallback 策略或 fine-tune embedding

6. **CL-30: discover_threads 耗时 650s**（3424 节点图）
   - 修复方案：限制社区命名的并发批次或缓存

### P3 — Low

7. **CL-01/02: semantic_bridge 策略触发率低**
   - 修复方案：调低 semantic_bridge 相似度阈值

8. **SK-24/E2E-03/04: 偏好类 Skill 匹配弱**
   - 修复方案：偏好类 skill 增加场景关键词扩展

---

## 5. 改进建议路线图

### 第一优先级：修复 Critical/High（影响生产安全和核心功能）

```
v0.0.11 目标：
├── fix: SkillRegistry.promote() 拒绝 DEPRECATED 状态
├── feat: skill matching 增加 min_score 阈值过滤
└── feat: recall 增加时间戳感知 reranking
```

### 第二优先级：提升竞争力（Knowledge Update + Multi-Session）

```
v0.0.12 目标：
├── feat: 知识更新检测（实体版本追踪）
├── feat: 多会话聚合增强（话题聚类 + 去重合并）
├── fix: skill rewrite 保留历史统计
└── perf: discover_threads 批量命名优化
```

### 第三优先级：Benchmark 自身增强

```
v0.1.0 目标：
├── feat: CI 集成 — benchmark 作为 PR gate
├── feat: 增加 anti-hacking 测试（投毒检测、prompt injection、PII 泄漏）
├── feat: 增加 semantic_bridge 策略的专项测试数据
└── feat: benchmark 结果趋势追踪（版本间对比）
```

---

## 6. 性能基线

| 指标 | 数值 | 目标 |
|------|------|------|
| 全管线 Recall 延迟 | 460ms | < 3000ms ✅ |
| MRR (向量召回) | 0.883 | > 0.8 ✅ |
| Skill 注册延迟 | < 10ms | < 100ms ✅ |
| 碰撞引擎单轮延迟 | ~100s | < 60s ⚠️ |
| KG 节点数 | 3424 | — |
| KG 边数 | 2100 | — |
| 线索发现 (communities) | 945 | — |

---

## 7. 测试框架架构

```
benchmarks/
├── __init__.py                    # 包入口
├── __main__.py                    # CLI runner
├── runner.py                      # 通用 runner 基类
├── test_dataset.py                # Phase 1-4 种子数据
│
├── phase1_recall.py               # Phase 1: 召回质量
├── phase2_faithfulness.py         # Phase 2: 忠实度
├── phase3_ab_test.py              # Phase 3: A/B 增益
├── phase4_context_efficiency.py   # Phase 4: 上下文效率
├── search_eval.py                 # 搜索评估
│
├── longmemeval_bench.py           # LongMemEval 标准 benchmark
├── extended_testdata.py           # Extended 100-case 种子数据
├── extended_bench.py              # Extended benchmark runner
│
├── BENCHMARK_REPORT_CN.md         # v0.0.6 历史报告
├── BENCHMARK_REPORT_EN.md         # v0.0.6 英文报告
└── BENCHMARK_REPORT_v0.0.10.md    # 本次总报告
```

### 运行方式

```bash
# LongMemEval benchmark (需要 LLM API)
SKIP_AUTO_INGEST=1 python -m benchmarks.longmemeval_bench

# Extended benchmark (100 cases, 需要 LLM API)
SKIP_AUTO_INGEST=1 LLM_PROVIDER=xai python -m benchmarks.extended_bench

# Extended benchmark (跳过碰撞引擎, 不需要 LLM)
SKIP_AUTO_INGEST=1 python -m benchmarks.extended_bench --skip-collision

# 仅清理测试数据
python -m benchmarks.extended_bench --cleanup-only

# 原始 Phase 1-4 benchmark
python -m benchmarks
```

---

## 8. Benchmark 设计哲学

> Recall 效率和准确率只是第一步。更重要的是 Knowledge Graph 稳定性、Skill 抽象能力、Intuitive 涌现能力和 Anti-hacking 安全性。

一个好的记忆系统 benchmark 应该回答：

| 层次 | 问题 | 本框架覆盖 |
|------|------|-----------|
| L0: Recall | 存进去的能找回来吗？ | ✅ LongMemEval + Phase 1 |
| L1: Knowledge Graph | 事实关系能正确传播吗？矛盾能被检测吗？ | ✅ E2E-14~17, CL-29~30 |
| L2: Skill Abstraction | 能从对话中提炼可复用知识吗？ | ✅ SK-01~35, E2E-01~10 |
| L3: Intuitive Emergence | 不相关记忆能碰撞出灵感吗？ | ✅ CL-01~35, E2E-22 |
| L4: Security | PII 不泄漏？投毒能检测？ | ⚠️ E2E-26 (基础), 需扩展 |

下一个版本需要重点加强 **L4: Security** 的测试覆盖，包括：
- Prompt injection 攻击向量
- 记忆投毒检测率
- PII/敏感信息泄漏防护
- 对抗性查询过滤
