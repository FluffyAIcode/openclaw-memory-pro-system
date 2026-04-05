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

## 4. 改进方案总表（13 项，统一编号）

两套 benchmark 共发现 13 项可行动的改进。下面按 **投入产出比** 统一排序，编号 F-01 ~ F-13。每项标注影响的 benchmark 指标、涉及的源码位置、代码示例和预估工作量。

| 编号 | 改进项 | 影响指标 | 涉及文件 | Sprint |
|------|--------|---------|---------|--------|
| F-01 | 时间戳感知排序 + update 信号检测 | KU +25pp, Temp +10pp | `context_composer.py` | 1 |
| F-02 | Answer Prompt 结构化 | Overall +5pp | `longmemeval_bench.py` | 1 |
| F-03 | Deprecated Skill 拒绝 Re-promote | SK-33 fix | `registry.py` | 1 |
| F-04 | Skill 匹配阈值 + match_score 传递 | SK-16, E2E-05 fix | `memory_hub.py` + `context_composer.py` | 1 |
| F-05 | 聚合查询 Multi-hop Recall | MS +15pp | `memory_hub.py` | 2 |
| F-06 | 时间线注入（Temporal 查询） | Temp +12pp, KU +8pp | `context_composer.py` | 2 |
| F-07 | Skill Recall 二阶段 Rerank | SK-12 fix, Skill 精度 | `memory_hub.py` + `reranker.py` | 2 |
| F-08 | Skill Rewrite 保留历史统计 | SK-31 fix | `registry.py` | 2 |
| F-09 | discover_threads 批量命名 + 缓存 | CL-30 perf 650s→70s | `inference.py` | 2 |
| F-10 | 偏好标签提取 + L1 主动注入 | Pref +20pp, SK-24/E2E-03/04 | `chronos/learner.py` + `context_composer.py` | 3 |
| F-11 | KG 关系在 Recall 中实际启用 | MS +8pp | `context_composer.py` + `knowledge_graph.py` | 3 |
| F-12 | 实体版本链 (KG superseded_by) | KU 根本方案 | `knowledge_graph.py` + `relation_extractor.py` | 3 |
| F-13 | semantic_bridge 阈值放宽 | CL-01/02 fix | `collision.py` | 3 |

> 缩写说明：KU = Knowledge Update, Temp = Temporal, MS = Multi-Session, Pref = Preference

---

### F-01 时间戳感知排序 + update 信号检测

**预计提升**：Knowledge Update +25pp, Temporal +10pp
**Sprint 1 | 工作量 1~2h | `context_composer.py`**

**问题**：`score_item()` 的 recency 权重仅 0.15~0.3，同一主题的新旧记忆分数相近（如"在 A 公司工作" vs "跳槽到 B 公司"），LLM 无法区分。

**改动 1** — `score_item()`（第 285~315 行）增加 update 信号 bonus：

```python
def score_item(item: dict, query_intent: str) -> float:
    relevance = item.get("score", 0.0)
    recency = _recency_score(item.get("timestamp", ""))
    importance = item.get("metadata", {}).get("importance", 0.5)

    update_signals = ['换了', '改成', '更新', '不再', '现在', '搬到', '跳槽',
                      'changed', 'updated', 'now', 'switched', 'moved to',
                      'instead', 'actually', 'no longer']
    content = item.get("content", "").lower()
    update_bonus = 0.1 if any(s in content for s in update_signals) else 0.0

    # ... 原有 w_rel/w_rec/w_imp 权重计算 ...
    composite = w_rel * relevance + w_rec * recency + w_imp * importance
    return round(composite + update_bonus, 4)
```

**改动 2** — `_build_core_layer()`（第 616~637 行）evidence 输出注入时间戳前缀：

```python
for ev in scored_evidence:
    ts = ev.get("timestamp", "")[:10]
    if ts:
        ev["content"] = f"[{ts}] {ev['content']}"
```

效果：LLM 能看到 `[2026-03-01] 买了 Tesla` 和 `[2026-04-01] 换成 BMW`，自然选择最新值。

**验证**：E2E-23 (Knowledge Update) 应从 FAIL 变为 PASS

---

### F-02 Answer Prompt 结构化

**预计提升**：Overall +5pp
**Sprint 1 | 工作量 30min | `benchmarks/longmemeval_bench.py`**

**问题**：生成答案的 prompt 过于简单，没有引导 LLM 正确使用时序和聚合信息。

```python
# 原有 prompt：
#   "Memory: {context[:2000]}\nQ: {question}\nA:"

# 改进为：
system = """You are a personal assistant with access to the user's long-term memory.
IMPORTANT RULES:
- For "how many" questions: count ALL distinct instances across ALL memories
- For questions about current state: use the MOST RECENT memory (check dates)
- For preference questions: look for expressions of like/dislike/preference
- If memories conflict, state the most recent one and note the change
- Always ground your answer in the provided memories"""

prompt = f"""Memory Context (sorted by date, newest first):
{context}

Question: {question}
Answer (be specific and cite memory dates when relevant):"""
```

---

### F-03 Deprecated Skill 拒绝 Re-promote

**严重度**：Critical（数据完整性）
**Sprint 1 | 工作量 5min | `skill_registry/registry.py`**

**问题**：`promote()` 第 285~329 行只检查了 cooldown 和 injection safety，未检查 `DEPRECATED` 状态。

```python
def promote(self, skill_id: str, force: bool = False) -> Optional[Skill]:
    skills = self._load()
    skill = skills.get(skill_id)
    if not skill:
        return None

    if skill.status == SkillStatus.DEPRECATED:
        logger.warning("Skill promote blocked: %s is DEPRECATED", skill.name)
        return None

    if not force:
        # ... 原有 cooldown 检查 ...
```

**验证**：SK-33 应从 FAIL 变为 PASS

---

### F-04 Skill 匹配阈值 + match_score 传递

**严重度**：High（不相关 skill 污染 L1）
**Sprint 1 | 工作量 15min | `memory_hub.py` + `context_composer.py`**

**问题**：`_recall_skills_vector()` 阈值 `sim > 0.25` 太低；`_build_core_layer()` 对 skill 硬编码 `composite_score: 1.0`。

```python
# memory_hub.py 第 608 行：提高阈值
if sim > 0.45:  # 原值 0.25
    scored.append((sim, skill))

# context_composer.py 第 602~614 行：使用实际分数
for s in raw.get("skills", []):
    match_score = s.get("match_score", 0)
    if match_score < 0.4:
        continue
    entry = {
        "content": f"[Skill] {s.get('name', '?')}: {body}",
        "score": match_score,
        "composite_score": match_score,
        # ...
    }
```

**验证**：SK-16（"今天天气"返回 0 skill）、E2E-05（"今天吃什么"不注入 skill）

---

### F-05 聚合查询 Multi-hop Recall

**预计提升**：Multi-Session +15pp
**Sprint 2 | 工作量 3~5h | `memory_hub.py`**

**问题**：问 "How many model kits?" 时 `top_k=8` 不足以覆盖散布在多个会话中的不同实例。

```python
def recall(self, query, top_k=8, ...):
    agg_patterns = [
        r'how many', r'how much', r'list all', r'what are all',
        r'everything .* about', r'all .* (i|my)',
        r'有多少', r'列出所有', r'哪些', r'都有什么',
    ]
    is_aggregation = any(re.search(p, query.lower()) for p in agg_patterns)

    if is_aggregation:
        expanded = self._vector_search(query, top_k=top_k * 3)
        results = self._diverse_selection(expanded, target=top_k)  # MMR 去重
    else:
        results = self._vector_search(query, top_k=top_k)
```

**验证**：E2E-24（"去过哪些地方旅行"应找到 3/3 目的地）

---

### F-06 时间线注入（Temporal 查询）

**预计提升**：Temporal +12pp, Knowledge Update +8pp
**Sprint 2 | 工作量 1~2h | `context_composer.py`**

**问题**：上下文是扁平文本，LLM 难以做时间推理（"最近一次"、"多少天前"）。

```python
def _inject_timeline(self, merged: list, query: str) -> list:
    temporal_signals = ['when', 'last time', 'first time', 'how long',
                        'how many days', 'before', 'after', 'recently',
                        '什么时候', '多久', '最近', '上次', '第一次']
    if not any(s in query.lower() for s in temporal_signals):
        return merged

    time_entries = sorted(
        [m for m in merged if m.get("timestamp")],
        key=lambda e: e.get("timestamp", ""))

    timeline = "Timeline (chronological):\n"
    for e in time_entries:
        date = e.get("timestamp", "")[:10]
        timeline += f"  {date}: {e['content'][:80]}...\n"

    return [{"content": timeline, "score": 1.0, "system": "timeline",
             "layer": "core"}] + merged
```

---

### F-07 Skill Recall 二阶段 Rerank

**Sprint 2 | 工作量 1h | `memory_hub.py` + `reranker.py`**

**问题**：SK-12 "ping不通如何排查" 匹配到"用户访谈技巧"而非"Docker网络调试"，因为 `nomic-embed-text-v1.5` 对中文口语表达语义距离不精确。

```python
# memory_hub.py — _recall_skills_vector()
# 第一阶段：embedding 粗筛 top 10
# 第二阶段：cross-encoder rerank（使用已有的 reranker.py）
from reranker import rerank
candidates = [(sim, skill) for sim, skill in scored if sim > 0.2]
if candidates:
    texts = [f"{s.name}: {s.content[:200]}" for _, s in candidates]
    reranked = rerank(query, texts, top_k=5)
    return [(reranked[i].score, candidates[i][1]) for i in reranked.indices]
```

**验证**：SK-12 应匹配到"Docker网络调试"

---

### F-08 Skill Rewrite 保留历史统计

**Sprint 2 | 工作量 15min | `skill_registry/registry.py`**

**问题**：`_trigger_rewrite()`（第 462~465 行）硬重置 `successes = 0, failures = 0`，导致 `total_uses` 跌到 1。

```python
# 修改 _trigger_rewrite()：
skill.content = rewritten.strip()
skill.version += 1
skill.failures = 0       # 给 rewrite 后的内容"重新证明"的机会
# skill.successes = 0    # ← 删除这行，保留历史成功次数
skill.updated_at = datetime.now().isoformat()
self._save_all()
```

**验证**：SK-31 中 rewrite 后 `total_uses` 应 >= 3

---

### F-09 discover_threads 批量命名 + 缓存

**Sprint 2 | 工作量 1h | `second_brain/inference.py`**

**问题**：`discover_threads()` 对 945 个社区逐一调用 LLM，0.7s × 945 = 650s。

```python
def _name_threads_batch(self, communities: list, batch_size: int = 10):
    # 1. 跳过 size < 3 的小社区，用启发式命名
    # 2. 批量：10 个社区合并成一个 prompt
    # 3. 缓存：已命名的不重复调用
    for batch in chunks(unnamed, batch_size):
        prompt = "\n\n".join(
            f"社区 {i+1}:\n" + "\n".join(f"- {n.content[:60]}" for n in c[:5])
            for i, c in enumerate(batch)
        )
        prompt += "\n\n为每个社区各取一个5-10字标题，格式：1. 标题\\n2. 标题\\n..."
        raw = llm_client.generate(prompt, model=llm_client.FAST_MODEL, ...)
```

预期：945 → ~95 次 LLM 调用，从 650s 降到 ~70s

---

### F-10 偏好标签提取 + L1 主动注入

**预计提升**：Preference +20pp
**Sprint 3 | 工作量 1~2 周 | `chronos/learner.py` + `context_composer.py`**

**问题**：用户偏好散落在对话中（"I love spicy food"），与推荐类查询（"suggest a restaurant"）向量距离远。同时 SK-24/E2E-03/04 偏好类 skill 匹配弱。

```python
# chronos/learner.py — 偏好提取
PREFERENCE_PATTERNS = [
    (r"i (love|like|prefer|enjoy|am into)\s+(.+)", "positive"),
    (r"i (hate|dislike|don't like|avoid)\s+(.+)", "negative"),
    (r"my favorite (.+) is (.+)", "favorite"),
    (r"我(喜欢|爱|偏好|习惯)\s*(.+)", "positive"),
    (r"我(讨厌|不喜欢|避免|不想)\s*(.+)", "negative"),
]

# context_composer.py — _build_core_layer()
# 当 intent in ("suggestion", "planning") 时主动注入偏好：
prefs = get_relevant_preferences(query)
for p in prefs[:3]:
    items.append({"content": f"[偏好] {p['type']}: {p['value']}",
                   "score": 0.8, "system": "chronos", "layer": "core"})
```

同时在 `skill_registry/registry.py` 的 `add()` 中为偏好类 skill 自动扩展 `applicable_scenarios`（解决 SK-24/E2E-03/04）：

```python
if "偏好" in name or any(t in tags for t in ["preference"]):
    skill.applicable_scenarios += " " + _expand_preference_scenarios(skill.content)
    # "深色主题" → "编辑器主题, IDE配置, 终端配色, 浏览器外观"
```

---

### F-11 KG 关系在 Recall 中实际启用

**预计提升**：Multi-Session +8pp
**Sprint 3 | 工作量 3~5h | `context_composer.py` + `knowledge_graph.py`**

**问题**：E2E-11 报告 `kg_rels=0`，KG 在 recall 中未实际参与。

三步行动：
1. `_build_concept_layer()` 增加 debug logging，确认 KG 查询是否被调用
2. `knowledge_graph.py` 的 `get_related_nodes()` 改用 embedding 匹配（而非纯字符串）
3. 对聚合查询，KG 直接提供 `(user)-[has]->(item_1), (item_2)...` 完整列表

---

### F-12 实体版本链 (KG superseded_by)

**Sprint 3 | 工作量 4h | `knowledge_graph.py` + `relation_extractor.py`**

**问题**：Knowledge Update 的根本方案。F-01 是"让 LLM 看到时间戳"的快速修复，F-12 是在 KG 层面真正追踪实体版本。

```
Node("我的车=Tesla Model 3", type=FACT, valid_from="2026-03-01")
   ──[superseded_by]──▶
Node("我的车=BMW iX3", type=FACT, valid_from="2026-04-01")
```

recall 时通过 KG 查询实体 latest_version，旧版本自动降权或标注 `[已更新]`。

---

### F-13 semantic_bridge 阈值放宽

**Sprint 3 | 工作量 15min | `second_brain/collision.py`**

**问题**：CL-01/02 semantic_bridge 5 轮未触发，因为相似度阈值窗口 [0.3, 0.7] 太窄。

```python
# collision.py — _semantic_bridge()
MIN_SIM = 0.15  # 原值 ~0.3
MAX_SIM = 0.75  # 原值 ~0.7
```

同时在 `extended_testdata.py` 增加 semantic_bridge 专项测试数据。

---

## 5. Sprint 路线图

### Sprint 1 — v0.0.11（1~2 天）：改动最小、收益最大

| 编号 | 改进项 | 文件 | 工作量 | 预期收益 |
|------|--------|------|--------|---------|
| F-01 | 时间戳感知排序 + update 信号 | `context_composer.py` | 1~2h | KU +25pp |
| F-02 | Answer Prompt 结构化 | `longmemeval_bench.py` | 30min | Overall +5pp |
| F-03 | Deprecated 拒绝 re-promote | `registry.py` | 5min | SK-33 fix |
| F-04 | Skill 匹配阈值 + match_score | `memory_hub.py` + `context_composer.py` | 15min | SK-16, E2E-05 fix |

**Sprint 1 完成后预期**：LongMemEval Overall 38.6% → ~43%, Extended 88% → ~92%

### Sprint 2 — v0.0.12（3~5 天）：recall pipeline 逻辑增强

| 编号 | 改进项 | 文件 | 工作量 | 预期收益 |
|------|--------|------|--------|---------|
| F-05 | 聚合查询 Multi-hop | `memory_hub.py` | 3~5h | MS +15pp |
| F-06 | 时间线注入 | `context_composer.py` | 1~2h | Temp +12pp |
| F-07 | Skill Rerank | `memory_hub.py` + `reranker.py` | 1h | SK-12 fix |
| F-08 | Rewrite 保留统计 | `registry.py` | 15min | SK-31 fix |
| F-09 | Threads 批量命名 | `inference.py` | 1h | CL-30 650s→70s |

**Sprint 2 完成后预期**：LongMemEval Overall ~43% → ~53%, Extended ~92% → ~96%

### Sprint 3 — v0.1.0（1~2 周）：子系统增强

| 编号 | 改进项 | 文件 | 工作量 | 预期收益 |
|------|--------|------|--------|---------|
| F-10 | 偏好标签提取 + L1 注入 | `chronos/learner.py` + `context_composer.py` | 1~2w | Pref +20pp |
| F-11 | KG 关系实际启用 | `context_composer.py` + `knowledge_graph.py` | 3~5h | MS +8pp |
| F-12 | 实体版本链 | `knowledge_graph.py` + `relation_extractor.py` | 4h | KU 根本方案 |
| F-13 | semantic_bridge 阈值 | `collision.py` | 15min | CL-01/02 fix |
| — | CI benchmark 集成 | `.github/workflows/benchmark.yml` | 2h | — |
| — | Anti-hacking 测试扩展 | `extended_testdata.py` | 3h | — |

**Sprint 3 完成后预期**：LongMemEval Overall ~53% → ~60%, Extended ~96% → ~99%

### 预期提升效果（累积）

| 能力维度 | 当前 | Sprint 1 | Sprint 2 | Sprint 3 |
|---------|------|----------|----------|----------|
| Knowledge Update | 19.2% | ~45% | ~53% | ~58% |
| Multi-Session | 34.6% | ~37% | ~55% | ~60% |
| Temporal | 44.4% | ~52% | ~62% | ~65% |
| Preference | 30.0% | ~30% | ~30% | ~55% |
| Info Extraction | 46.8% | ~50% | ~55% | ~60% |
| **LongMemEval Overall** | **38.6%** | **~43%** | **~53%** | **~60%** |
| **Extended Overall** | **88%** | **~92%** | **~96%** | **~99%** |

### 验证节奏

每个 Sprint 完成后重跑 benchmark 验证效果：

```bash
# LongMemEval（~47 分钟）
SKIP_AUTO_INGEST=1 python -m benchmarks.longmemeval_bench

# Extended（~33 分钟）
SKIP_AUTO_INGEST=1 LLM_PROVIDER=xai python -m benchmarks.extended_bench
```

对比版本间得分变化，确认改进方向正确。

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
