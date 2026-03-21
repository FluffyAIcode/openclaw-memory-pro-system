---
name: memora
description: "增强型记忆管理。使用场景: (1) 保存重要对话/决策到长期记忆, (2) 搜索历史记忆获取上下文, (3) 执行记忆提炼生成长期摘要(现接入LLM), (4) 查看记忆系统状态。NOT for: 普通的 memory/ 日文件读写(直接用文件操作)。"
metadata:
  openclaw:
    emoji: "🧠"
    requires:
      anyBins: ["python3"]
---

# Memora — 增强型记忆管理系统

Memora 是你的高级记忆层，与 OpenClaw 原生 `memory/` 目录双写同步。它提供向量语义搜索、自动提炼、长期记忆管理等原生记忆系统不具备的能力。

## v2.0 更新

- **Memory Server 加速**: 通过常驻服务，响应从 10-30s 降到 <100ms
- **LLM 驱动摘要**: `digest` 现在调用 xAI Grok 生成 AI 摘要，而非简单拼接
- **共享嵌入模型**: Memora 和 MSA 共用同一个 SentenceTransformer，节省内存

## 核心能力

| 能力 | 说明 |
|------|------|
| **双写保存** | 同时写入 Memora 向量库 + OpenClaw memory/ 目录 |
| **语义搜索** | 基于向量相似度搜索历史记忆，比关键词匹配更智能 |
| **LLM 记忆提炼** | 读取近期 daily 记录，通过 LLM 生成结构化长期摘要 |
| **系统状态** | 查看向量条目数、存储路径、配置信息 |

## 使用方法

### 推荐：通过 Memory Server（快速）

确保服务在运行（`memory-cli health`），然后：

```bash
# 语义搜索
memory-cli search "用户的 LLM 偏好"

# 保存记忆
memory-cli add "用户决定使用 xAI Grok 作为主要 LLM" -i 0.9 -s openclaw

# 智能路由保存（自动分发到 Memora/MSA/Chronos）
memory-cli remember "重要内容" -i 0.8

# 记忆提炼（现在带 AI 摘要）
memory-cli digest --days 7

# 查看状态
memory-cli status
```

### 备选：直接 CLI（较慢，每次加载模型）

```bash
python3 -m memora search "关键词"
python3 -m memora add "内容" -i 0.9 -s openclaw
python3 -m memora digest --days 7
python3 -m memora status
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
