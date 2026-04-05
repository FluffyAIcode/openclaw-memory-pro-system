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

## 4. 发现的关键问题与修改建议（按严重度排序）

### P0 — Critical

**1. SK-33: Deprecated Skill 可被 Re-promote**

- **现象**：`registry.deprecate(id)` 后再次 `registry.promote(id, force=True)` 能成功，状态变回 `active`
- **根因**：`skill_registry/registry.py` 第 285~329 行 `promote()` 方法只检查了 cooldown 和 injection safety，没有检查当前状态是否为 `DEPRECATED`
- **修改建议**：

```python
# skill_registry/registry.py — promote() 方法，在 cooldown 检查之前添加
def promote(self, skill_id: str, force: bool = False) -> Optional[Skill]:
    skills = self._load()
    skill = skills.get(skill_id)
    if not skill:
        return None

    # ← 新增：拒绝已废弃的 skill
    if skill.status == SkillStatus.DEPRECATED:
        logger.warning("Skill promote blocked: %s is DEPRECATED", skill.name)
        return None

    if not force:
        # ... 原有 cooldown 检查 ...
```

- **验证用例**：SK-33 应从 FAIL 变为 PASS

---

### P1 — High

**2. SK-16 / E2E-05: Skill 匹配缺少相关度阈值**

- **现象**：查询"今天天气怎么样"返回 5 个完全不相关的 skill；"今天吃什么"也注入 skill 到 L1
- **根因链路**：
  - `memory_hub.py` 第 608 行 `_recall_skills_vector()` 的阈值 `sim > 0.25` 太低
  - `context_composer.py` 第 602~614 行 `_build_core_layer()` 对 skill 设置了 `composite_score: 1.0` 并 `force_add`，完全不考虑 match_score
- **修改建议（两处联动）**：

```python
# memory_hub.py — _recall_skills_vector()，提高匹配阈值
if sim > 0.45:  # 原值 0.25，提高到 0.45 过滤无关 skill
    scored.append((sim, skill))

# context_composer.py — _build_core_layer()，使用实际 match_score 而非硬编码 1.0
for s in raw.get("skills", []):
    match_score = s.get("match_score", 0)
    if match_score < 0.4:  # ← 新增：低分 skill 不进 L1
        continue
    # ...
    entry = {
        "content": f"[Skill] {s.get('name', '?')}: {body}",
        "score": match_score,            # ← 改用实际分数
        "composite_score": match_score,   # ← 改用实际分数
        # ...
    }
```

- **验证用例**：SK-16 应返回 0 个 skill，E2E-05 应不注入 skill

---

**3. Knowledge Update 全面薄弱（LongMemEval 19.2%）**

- **现象**：同一实体有新旧两个版本时（如"3 月买了 Tesla → 4 月换了 BMW"），recall 无法优先返回最新版本
- **根因**：
  - `context_composer.py` 的 `score_item()` 虽然有 `_recency_score()` 衰减，但权重配比中 recency 权重仅 0.15~0.3，不足以让新信息排在旧信息前面
  - 系统没有 **实体级版本追踪** — 不知道"Tesla Model 3"和"BMW iX3"指向同一个槽位（用户的车）
- **修改建议（分两步）**：

第一步（v0.0.11，快速收益）— 在 `_build_core_layer` 中增加 **同实体时间戳去重**：

```python
# context_composer.py — _build_core_layer()，evidence 排序后增加时间戳去重
def _dedupe_by_entity(items: list) -> list:
    """当多条 evidence 描述同一实体时，只保留最新的一条。
    判定标准：两条内容的 embedding 相似度 > 0.85 时视为同一实体的不同版本。
    """
    # 按 timestamp 降序排，遍历时跳过与已选 item 高度相似的旧条目
    ...
```

第二步（v0.0.12，根本方案）— 在 KG 中引入 **实体版本链**：

```
Node("我的车=Tesla Model 3", type=FACT, valid_from="2026-03-01")
   ──[superseded_by]──▶
Node("我的车=BMW iX3", type=FACT, valid_from="2026-04-01")
```

recall 时通过 KG 查询实体的 latest_version，将旧版本自动降权或标注为 `[已更新]`

---

### P2 — Medium

**4. SK-31: Skill Rewrite 后 uses 计数重置**

- **现象**：utility 为 0%（3 次失败 / 3 次使用）触发 rewrite 后，`total_uses` 被重置为 1，不满足 `MIN_USES_FOR_REWRITE >= 3` 的后续检查条件
- **根因**：`skill_registry/registry.py` 第 462~465 行，`_trigger_rewrite()` 中硬重置了 `successes = 0` 和 `failures = 0`
- **修改建议**：

```python
# skill_registry/registry.py — _trigger_rewrite()
# 改为：保留历史统计，另起一个 "rewrite 后" 的计数周期
if rewritten and len(rewritten.strip()) > 50:
    # ... 安全检查 ...
    skill.content = rewritten.strip()
    skill.version += 1
    # ← 保留总量，新增 rewrite_epoch 字段记录从哪个版本开始计数
    skill.rewrite_epoch_successes = 0
    skill.rewrite_epoch_failures = 0
    # 不再重置 skill.successes 和 skill.failures
    skill.updated_at = datetime.now().isoformat()
    self._save_all()
```

或更简单的方案：只重置 `failures`，保留 `successes`，避免 `total_uses` 跌到 1：

```python
    skill.failures = 0       # 给 rewrite 后的内容一个"重新证明"的机会
    # skill.successes = 0    # ← 删除这行，保留历史成功次数
```

- **验证用例**：SK-31 中 rewrite 后 `total_uses` 应 >= 3

---

**5. SK-12: 中文同义词匹配精度不足**

- **现象**：查询"容器之间 ping 不通如何排查"匹配到"用户访谈技巧"而非"Docker 网络调试"
- **根因**：`nomic-embed-text-v1.5` 对中文口语化表达的语义捕捉不够精确，"ping 不通" ↔ "网络调试"的向量距离大于"ping 不通" ↔ "访谈技巧"
- **修改建议**：

方案 A（推荐）— 在 `_recall_skills_vector` 中增加 **二阶段 rerank**：

```python
# memory_hub.py — _recall_skills_vector()
# 第一阶段：embedding 粗筛（保留 top 10）
# 第二阶段：cross-encoder rerank（使用已有的 reranker.py）
from reranker import rerank
candidates = [(sim, skill) for sim, skill in scored if sim > 0.2]
if candidates:
    texts = [f"{s.name}: {s.content[:200]}" for _, s in candidates]
    reranked = rerank(query, texts, top_k=5)
    return [(reranked[i].score, candidates[i][1]) for i in reranked.indices]
```

方案 B — Skill 注册时生成 **搜索扩展词**：

```python
# skill_registry/registry.py — add() 时自动生成
skill.search_aliases = ["Docker 网络", "容器通信", "ping 不通", "网络排查"]
# _recall_skills_keyword 中同时搜索 aliases
```

---

**6. CL-30: discover_threads 耗时 650s（3424 节点图）**

- **现象**：`inference_engine.discover_threads()` 对 945 个社区逐一调用 LLM 命名，单次 ~0.7s × 945 = 660s
- **根因**：`second_brain/inference.py` 第 321~362 行，`_name_thread()` 为每个 community 单独调用 `llm_client.generate()`
- **修改建议**：

方案 A（推荐）— **批量命名 + 缓存**：

```python
# second_brain/inference.py — discover_threads()
# 1. 缓存：已命名的 community 不重复调用 LLM
# 2. 批量：将多个 community 合并成一个 prompt，一次 LLM 调用命名 10 个
# 3. 裁剪：跳过 size < 3 的小社区，直接用启发式命名

def _name_threads_batch(self, communities: list, batch_size: int = 10):
    unnamed = [c for c in communities if self._thread_cache.get(id(c)) is None]
    for batch in chunks(unnamed, batch_size):
        prompt = "\n\n".join(
            f"社区 {i+1}:\n" + "\n".join(f"- {n.content[:60]}" for n in c[:5])
            for i, c in enumerate(batch)
        )
        prompt += "\n\n为每个社区各取一个5-10字标题，格式：1. 标题\\n2. 标题\\n..."
        raw = llm_client.generate(prompt, model=llm_client.FAST_MODEL, ...)
        # 解析并缓存
```

预期效果：945 社区 → ~95 次 LLM 调用（每次命名 10 个），从 650s 降到 ~70s

方案 B — **限制社区数量上限**：

```python
communities = self._kg.get_communities()
communities.sort(key=len, reverse=True)
communities = communities[:100]  # 只命名最大的 100 个社区
```

---

### P3 — Low

**7. CL-01/02: semantic_bridge 策略触发率低**

- **现象**：5 轮碰撞中 `semantic_bridge` 从未触发
- **根因**：种子数据的 21 条记忆虽然跨 3 个领域（fitness/coding/reading），但 `semantic_bridge` 策略在挑选配对时要求 embedding 相似度在 0.3~0.7 之间（既不太近也不太远），种子数据未命中此区间
- **修改建议**：

```python
# second_brain/collision.py — _semantic_bridge()
# 降低下界阈值，扩大配对候选范围
MIN_SIM = 0.15  # 原值约 0.3，降低以捕捉更多跨领域关联
MAX_SIM = 0.75  # 原值约 0.7，略微放宽
```

同时在 `extended_testdata.py` 中增加专门设计的 semantic_bridge 测试对（已有 `COLLISION_SEMANTIC_PAIRS`，但未被碰撞引擎单独路由）

---

**8. SK-24 / E2E-03 / E2E-04: 偏好类 Skill 匹配弱**

- **现象**：查询"帮我配置编辑器"无法匹配到"偏好:dark mode"skill
- **根因**：skill 名称/内容是"深色主题 dark mode"，与"配置编辑器"的语义距离较大
- **修改建议**：

```python
# skill_registry/registry.py — add() 时为偏好类 skill 自动扩展场景词
if "偏好" in name or any(t in tags for t in ["preference", "偏好"]):
    skill.applicable_scenarios = (
        skill.applicable_scenarios + " " +
        _expand_preference_scenarios(skill.content)
    )
    # 例如：content="深色主题" → 扩展 "编辑器主题, IDE配置, 终端配色, 浏览器外观"

# memory_hub.py — _recall_skills_vector() 中用扩展后的文本做 embedding
text = f"{skill.name} {skill.content[:500]} {skill.applicable_scenarios}"
```

---

## 5. 改进建议路线图

### 第一优先级：修复 Critical/High（v0.0.11，影响生产安全和核心功能）

| 修改项 | 文件 | 行数 | 预估工作量 |
|--------|------|------|-----------|
| deprecated skill 拒绝 re-promote | `skill_registry/registry.py` | ~3 行 | 5 min |
| skill matching 提高阈值 + 传递 match_score | `memory_hub.py` + `context_composer.py` | ~10 行 | 15 min |
| knowledge update: 同实体时间戳去重 | `context_composer.py` | ~30 行 | 1 hour |

### 第二优先级：提升竞争力（v0.0.12）

| 修改项 | 文件 | 预估工作量 |
|--------|------|-----------|
| 实体版本链 (KG superseded_by) | `second_brain/knowledge_graph.py` + `relation_extractor.py` | 4 hours |
| skill rewrite 保留历史统计 | `skill_registry/registry.py` | 15 min |
| skill recall 增加 cross-encoder rerank | `memory_hub.py` + `reranker.py` | 1 hour |
| discover_threads 批量命名 + 缓存 | `second_brain/inference.py` | 1 hour |

### 第三优先级：Benchmark 自身增强（v0.1.0）

| 修改项 | 文件 | 预估工作量 |
|--------|------|-----------|
| CI 集成 — benchmark 作为 PR gate | 新建 `.github/workflows/benchmark.yml` | 2 hours |
| Anti-hacking 测试扩展（投毒/injection/PII） | `benchmarks/extended_testdata.py` | 3 hours |
| semantic_bridge 专项测试数据 | `benchmarks/extended_testdata.py` | 30 min |
| benchmark 版本间趋势对比 | `benchmarks/trend_tracker.py` | 2 hours |

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
