# Second Brain v3 — 知识图谱认知引擎

## 概述

Second Brain 是 OpenClaw 的认知引擎，从 v1 的"被动档案馆"升级为 v3 的"知识图谱驱动认知伙伴"。它与 RAG 有本质区别：不仅能"找到相似的"，还能"推理相关的"、"发现缺失的"、"追踪矛盾的"。

## 核心能力

### 1. 知识图谱 (Knowledge Graph)
- 记忆不再是扁平向量列表，而是**有向图**：节点=知识单元，边=逻辑关系
- 节点类型: fact(事实), decision(决策), preference(偏好), goal(目标), question(问题)
- 边类型: supports(支持), contradicts(矛盾), extends(延伸), depends_on(依赖), alternative_to(替代), addresses(回答)
- 每次 `/remember` 写入时，自动抽取知识节点和关系（通过 LLM）

### 2. 三种 RAG 做不到的推理
- **矛盾检测**: 扫描图谱中 `contradicts` 边，找到有矛盾证据的决策并评估风险
- **缺失推理**: 对每个决策生成"应考虑的维度"，然后检查哪些维度图谱中没有覆盖
- **前向传播**: 新事实加入时，沿 `depends_on`/`contradicts` 边传播影响

### 3. 思维线索 (Thought Threads)
- 通过图谱社区检测算法自动发现相关知识的聚类
- 每个聚类就是一条"思维线索"，有自动命名和状态评估

### 4. 反馈闭环
- 灵感评分机制 (1-5 分)
- 7 种碰撞策略的权重自适应
- 好策略被选中概率更高，差策略降权

### 5. 内化管道
- 高成熟度的知识图谱节点自动标记为"内化候选"
- Chronos 巩固时从 KG 拉取稳定模式，增强 PERSONALITY.yaml 生成
- 生成 LoRA 训练数据（为未来本地模型微调准备）

## 命令

```bash
# 碰撞（7种策略，自适应权重选择）
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

# 其他（不变）
memory-cli briefing
memory-cli vitality
memory-cli inspect "关键词"
memory-cli review-dormant
```

## 数据存储

- 知识图谱: `memory/kg/nodes.jsonl` + `memory/kg/edges.jsonl`
- 访问日志: `memory/tracker/access_log.jsonl`
- 灵感文件: `memory/insights/YYYY-MM-DD.md`
- 策略权重: `memory/insights/strategy_weights.json`
- 评分记录: `memory/insights/ratings.jsonl`
- 训练数据: `memory/training_data/pairs_YYYY-MM-DD.jsonl`

## 架构关系

```
Memora/MSA → 关系抽取器 → 知识图谱 → 推理引擎 → 矛盾/盲区/传播
                                    ↓
                              内化管理器 → Chronos → PERSONALITY.yaml
```
