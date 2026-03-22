---
name: second-brain
description: "智能层 — KG 织网、蒸馏总结、灵感碰撞"
metadata:
  openclaw:
    emoji: "🧪"
    requires:
      anyBins: ["python3"]
---

# Second Brain — 智能层

位于语料库之上，提供知识图谱推理、蒸馏总结和灵感碰撞。与 RAG 的本质区别：能推理关系、发现缺失、追踪矛盾。

## 命令速查

| 场景 | 命令 |
|------|------|
| 灵感碰撞 (7 策略) | `memory-cli collide` |
| 矛盾检测 | `memory-cli contradictions` |
| 盲区扫描 | `memory-cli blindspots` |
| 思维线索 | `memory-cli threads` |
| 知识图谱统计 | `memory-cli graph-status` |
| 灵感评分 | `memory-cli rate <id> <1-5>` |
| 策略统计 | `memory-cli insight-stats` |
| 每日简报 | `memory-cli briefing` |
| 记忆活力 | `memory-cli vitality` |
| 记忆生命周期 | `memory-cli inspect "关键词"` |
| 沉睡记忆 | `memory-cli review-dormant` |
| 记忆总结 | `memory-cli digest --days 7` |

## 碰撞策略 (7 种)

RAG-based: Semantic Bridge, Dormant Revival, Temporal Echo, Chronos Cross-Ref, Digest Bridge
KG-driven: Contradiction-Based, Blind Spot-Based

策略权重自适应：用户评分 → 权重调整 → 更好碰撞。

## 数据

- KG: `memory/kg/nodes.jsonl` + `edges.jsonl`
- 灵感: `memory/insights/YYYY-MM-DD.md`
- 权重: `memory/insights/strategy_weights.json`
