---
name: chronos
description: "训练管线 + 人格档案生成。使用场景: (1) 将重要记忆编码到训练缓冲区, (2) 导出 JSONL 训练数据集, (3) 执行巩固并生成 PERSONALITY.yaml, (4) 查看训练管线状态。NOT for: 日常日文件读写(用文件操作), 语义搜索(用 memora), KG推理(用 second-brain)。"
metadata:
  openclaw:
    emoji: "⚡"
    requires:
      anyBins: ["python3"]
---

# Chronos — 训练管线 + 人格档案系统

Chronos 是 OpenClaw 记忆系统的训练层。核心职责：从记忆中提取结构化训练数据，生成人格档案，为未来 Nebius 云端微调准备数据集。

## v3.0 更新 (v0.0.3-beta → v0.0.4)

- **EWC/LoRA 已废弃**: 不再模拟持续学习，转为真实训练管线
- **Distiller**: 从 digest + replay buffer 生成 JSONL 训练数据集 (`chronos_lora_row_v1` 格式)
- **Nebius Client**: 云端微调骨架（skeleton，待接入 Nebius AI Cloud API）
- **Personality Profile**: 巩固时通过 LLM 从高重要性记忆 + KG 模式生成 `PERSONALITY.yaml`

### 工作流程

```
记忆输入 → 重要性编码 → Replay Buffer
                              ↓
                  定期巩固 (Scheduler 每6小时)
                              ↓
                ┌─────────────┼─────────────┐
                ↓             ↓             ↓
          LLM 分析记忆   Distiller 导出   Nebius Client
                ↓             ↓             ↓
        PERSONALITY.yaml  JSONL 数据集    (未来)微调
```

### PERSONALITY.yaml 包含

- `core_beliefs`: 从记忆中提炼的核心信念
- `communication_style`: 交流风格偏好
- `knowledge_anchors`: 关键知识锚点
- `learned_preferences`: 学习到的用户偏好
- `important_decisions`: 重要决策记录

## 五子系统对比

| 系统 | 层级 | 适用场景 |
|------|------|----------|
| **Memora** | 统一语料库 | 语义搜索、去重存储 |
| **MSA** | 统一语料库 | 长文档存储、多跳推理 |
| **Second Brain** | 智能层 | KG 织网、蒸馏总结、灵感碰撞 |
| **Skill Registry** | 技能层 | 可操作技能管理 |
| **Chronos** | 训练层 | 训练数据、人格档案、Nebius 微调 |

## 使用方法

### 通过 Memory Server（推荐）

```bash
# 记忆自动路由（高重要性内容自动编码到缓冲区）
memory-cli remember "核心知识内容" -i 0.95 --tag thought

# 导出训练数据
memory-cli training-export

# 查看状态
memory-cli status
```

### 何时涉及 Chronos

- **自动**: Memory Server 的 Scheduler 每 6 小时自动巩固，生成 PERSONALITY.yaml
- **手动导出**: `memory-cli training-export` 生成 JSONL 数据集
- **路由**: 通过 `memory-cli remember` 时指定 `force_systems=["chronos"]` 可强制路由

## 数据位置

- Replay Buffer: `memory/chronos/replay_buffer.jsonl`
- 训练数据集: `memory/chronos/training_*.jsonl`
- 人格档案: `PERSONALITY.yaml`
- Nebius 配置: `chronos/config.yaml` (api_key, base_model 等)
