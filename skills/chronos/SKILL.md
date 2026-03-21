---
name: chronos
description: "持续学习记忆系统 + 人格档案生成。使用场景: (1) 将重要记忆内化到经验缓冲区, (2) 执行记忆巩固并生成 PERSONALITY.yaml 人格档案, (3) 查看持续学习状态。NOT for: 日常日文件读写(用文件操作), 语义搜索(用 memora)。"
metadata:
  openclaw:
    emoji: "⚡"
    requires:
      anyBins: ["python3"]
---

# Chronos — 持续学习 + 人格档案系统

Chronos 是 OpenClaw 的第三套记忆系统。核心理念：**以算代存**。

## v2.0 更新：Personality Profile

Chronos 现在在每次巩固（consolidate）时，会通过 xAI Grok API 从高重要性记忆中**生成 PERSONALITY.yaml 人格档案**。这个档案在 Agent 启动时被读取，直接影响 Agent 的行为和风格。

### 工作流程

```
记忆输入 → 重要性评分 → 经验回放缓冲区
                              ↓
                  定期巩固 (每6小时)
                              ↓
                 LLM 分析高重要性记忆
                              ↓
                 生成 PERSONALITY.yaml
                              ↓
          Agent 启动时读取 → 影响行为
```

### PERSONALITY.yaml 包含

- `core_beliefs`: 从记忆中提炼的核心信念
- `communication_style`: 交流风格偏好
- `knowledge_anchors`: 关键知识锚点
- `learned_preferences`: 学习到的用户偏好
- `important_decisions`: 重要决策记录

## 四套记忆系统对比

| 系统 | 方式 | 适用场景 |
|------|------|----------|
| **原生 memory/** | 文件读写 | 日常日志、快速笔记 |
| **Memora** | 向量 RAG 检索 | 语义搜索历史记忆 |
| **Chronos** | 经验回放 + 人格生成 | 人格演化、行为影响 |
| **MSA** | 稀疏路由 + 多跳推理 | 大文档分析、跨段落推理 |

## 使用方法

### 学习新记忆

```bash
# 通过 Memory Server（推荐，会自动路由）
memory-cli remember "用户的核心理念是以算代存" -i 0.95

# 直接 Chronos CLI
python3 -m chronos learn "核心知识内容" -i 0.9 -s openclaw
```

### 强制巩固 + 生成人格档案

```bash
python3 -m chronos consolidate
```

巩固后检查生成的人格档案：

```bash
cat PERSONALITY.yaml
```

### 查看状态

```bash
python3 -m chronos status
python3 -m chronos report
```

## 何时使用 Chronos

**用 Chronos 当：**
- 用户表达核心偏好、价值观、人格特征
- 做出会长期影响交互的重大决策
- 想让 AI "真正记住"并影响后续行为
- heartbeat 中执行周期性记忆巩固

**用 Memora 当：**
- 需要语义搜索过去的对话内容

**用 MSA 当：**
- 需要跨文档推理

## 数据位置

- 回放缓冲区：`~/.openclaw/workspace/memory/chronos/replay_buffer.jsonl`
- 系统状态：`~/.openclaw/workspace/memory/chronos/state/`
- 人格档案：`~/.openclaw/workspace/PERSONALITY.yaml`
- 配置文件：`~/.openclaw/workspace/chronos/config.yaml`
