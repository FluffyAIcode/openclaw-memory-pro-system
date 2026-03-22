# Second Brain v3 — 知识图谱认知引擎

## 概述

Second Brain 是 OpenClaw 的智能层，位于语料库（Memora/MSA）之上，提供知识织网、蒸馏总结和灵感碰撞。它与 RAG 有本质区别：不仅能"找到相似的"，还能"推理相关的"、"发现缺失的"、"追踪矛盾的"。

## 核心能力

### 1. 知识图谱 (Knowledge Graph)
- 记忆不再是扁平向量列表，而是**有向图**：节点=知识单元，边=逻辑关系
- 节点类型: fact(事实), decision(决策), preference(偏好), goal(目标), question(问题)
- 边类型: supports(支持), contradicts(矛盾), extends(延伸), depends_on(依赖), alternative_to(替代), addresses(回答)
- 每次 `/remember` 写入时，自动抽取知识节点和关系（通过 LLM）

### 2. 蒸馏总结 (Digest)
- 定期读取近期 daily 记录，通过 LLM 生成结构化长期摘要
- 摘要存储在 `memory/second_brain/long_term/`
- 通过 `memory-cli digest --days 7` 手动触发，或 Scheduler 自动执行

### 3. 三种 RAG 做不到的推理
- **矛盾检测**: 扫描图谱中 `contradicts` 边，找到有矛盾证据的决策并评估风险
- **缺失推理**: 对每个决策生成"应考虑的维度"，然后检查哪些维度图谱中没有覆盖
- **前向传播**: 新事实加入时，沿 `depends_on`/`contradicts` 边传播影响

### 4. 思维线索 (Thought Threads)
- 通过图谱社区检测算法自动发现相关知识的聚类
- 每个聚类就是一条"思维线索"，有自动命名和状态评估

### 5. 灵感碰撞（7 策略自适应）
- **RAG-based**: Semantic Bridge, Dormant Revival, Temporal Echo, Chronos Cross-Reference, Digest Bridge
- **KG-driven**: Contradiction-Based, Blind Spot-Based
- 评分反馈闭环：用户评分 → 策略权重调整 → 更好的碰撞

### 6. 内化管道
- 高成熟度的知识图谱节点自动标记为"内化候选"
- Chronos 巩固时从 KG 拉取稳定模式，增强 PERSONALITY.yaml 生成
- Chronos Distiller 导出 JSONL 训练数据（为 Nebius 云端微调准备）

## 命令

```bash
# 灵感碰撞（7种策略，自适应权重选择）
memory-cli collide

# 矛盾检测
memory-cli contradictions

# 盲区扫描
memory-cli blindspots

# 思维线索
memory-cli threads

# 知识图谱状态
memory-cli graph-status

# 灵感评分
memory-cli rate <insight_id> <1-5> [-s strategy] [-c "评论"]

# 策略统计
memory-cli insight-stats

# 每日简报
memory-cli briefing

# 记忆活力
memory-cli vitality

# 记忆生命周期
memory-cli inspect "关键词"

# 沉睡记忆
memory-cli review-dormant

# 记忆总结
memory-cli digest --days 7
```

## 数据存储

- 知识图谱: `memory/kg/nodes.jsonl` + `memory/kg/edges.jsonl`
- 访问日志: `memory/tracker/access_log.jsonl`
- 灵感文件: `memory/insights/YYYY-MM-DD.md`
- 策略权重: `memory/insights/strategy_weights.json`
- 评分记录: `memory/insights/ratings.jsonl`
- 每日摘要: `memory/second_brain/daily/`
- 长期摘要: `memory/second_brain/long_term/`

## 架构位置

```
碎片输入 → Memora/MSA (语料库)
                │
     ┌──────────┼──────────┐
     ↓          ↓          ↓
 [KG 织网]  [蒸馏/总结]  [碰撞/联想]   ← Second Brain
     │          │          │
     └──────────┼──────────┘
                ↓
         Skill Registry → Chronos (训练管线)
```
