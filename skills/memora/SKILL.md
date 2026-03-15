---
name: memora
description: "增强型记忆管理。使用场景: (1) 保存重要对话/决策到长期记忆, (2) 搜索历史记忆获取上下文, (3) 执行记忆提炼生成长期摘要, (4) 查看记忆系统状态。NOT for: 普通的 memory/ 日文件读写(直接用文件操作)。"
metadata:
  openclaw:
    emoji: "🧠"
    requires:
      anyBins: ["python3"]
---

# Memora — 增强型记忆管理系统

Memora 是你的高级记忆层，与 OpenClaw 原生 `memory/` 目录双写同步。它提供向量语义搜索、自动提炼、长期记忆管理等原生记忆系统不具备的能力。

## 核心能力

| 能力 | 说明 |
|------|------|
| **双写保存** | 同时写入 Memora 向量库 + OpenClaw memory/ 目录 |
| **语义搜索** | 基于向量相似度搜索历史记忆，比关键词匹配更智能 |
| **记忆提炼** | 读取近期 daily 记录，合并生成长期摘要 |
| **系统状态** | 查看向量条目数、存储路径、配置信息 |

## 使用方法

所有命令通过 `bash` 工具执行，工作目录为 `~/.openclaw/workspace`。

### 保存重要记忆

当对话中出现值得长期保存的内容（重要决策、关键发现、用户偏好等），使用双写保存：

```bash
python3 -m memora add "用户决定使用 xAI Grok 作为主要 LLM" -i 0.9 -s openclaw
```

参数说明：
- 第一个参数：记忆内容（引号包裹）
- `-i`：重要性 0.0-1.0（默认 0.7，重要决策建议 0.8-1.0）
- `-s`：来源标记（默认 cli，推荐用 openclaw/telegram/user 等标注来源）

### 搜索历史记忆

需要回忆过去的上下文时，用语义搜索：

```bash
python3 -m memora search "用户的 LLM 偏好"
```

结果按相似度得分排序，score 越高越相关。

### 执行记忆提炼

定期（建议每周一次或在 heartbeat 中触发）将近期 daily 记录提炼为长期摘要：

```bash
python3 -m memora digest --days 7
```

提炼结果写入 `memory/long_term/digest_*.md`。

### 查看系统状态

```bash
python3 -m memora status
```

### 初始化（首次或目录丢失时）

```bash
python3 -m memora init
```

## 何时使用 Memora

**用 Memora（双写）当：**
- 对话中出现重要决策或用户明确的偏好
- 需要跨会话持久保存的关键上下文
- 用户说"记住这个"或类似指令
- heartbeat 中执行周期性记忆提炼

**直接写 memory/ 当：**
- 普通的 daily 日志记录
- 临时的会话笔记
- 已有的 MEMORY.md 更新

## 数据位置

- 向量数据库：`~/.openclaw/workspace/memory/vector_db/`
- 日记录：`~/.openclaw/workspace/memory/daily/`
- 长期摘要：`~/.openclaw/workspace/memory/long_term/`
- 配置文件：`~/.openclaw/workspace/memora/config.yaml`
