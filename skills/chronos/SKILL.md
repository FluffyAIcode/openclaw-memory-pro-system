---
name: chronos
description: "训练管线 — replay buffer + distiller + 人格档案"
metadata:
  openclaw:
    emoji: "⚡"
    requires:
      anyBins: ["python3"]
---

# Chronos — 训练管线

从记忆中提取训练数据，生成人格档案，为 Nebius 云端微调准备数据集。

## 何时涉及

| 场景 | 方式 |
|------|------|
| 高重要性记忆自动编码 | `memory-cli remember "内容" -i 0.95`（Hub 自动路由） |
| 导出训练数据 | `memory-cli training-export` |
| 人格档案生成 | Scheduler 每 6 小时自动巩固 → `PERSONALITY.yaml` |
| 查看状态 | `memory-cli status` |

## 何时不用

- 语义搜索 → Memora（`recall` / `search`）
- KG 推理 → Second Brain（`contradictions` / `blindspots`）

## 数据

- Replay Buffer: `memory/chronos/replay_buffer.jsonl`
- 人格档案: `PERSONALITY.yaml`
