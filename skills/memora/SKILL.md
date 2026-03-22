---
name: memora
description: "统一语料库 — 向量存储与语义搜索。使用场景: (1) 保存重要内容到向量库, (2) 语义搜索历史记忆, (3) 合并召回 (Memora + MSA), (4) 查看记忆状态。NOT for: 普通的 memory/ 日文件读写(直接用文件操作), 长文档推理(用 deep-recall/MSA), KG推理(用 second-brain)。"
metadata:
  openclaw:
    emoji: "🧠"
    requires:
      anyBins: ["python3"]
---

# Memora — 统一语料库 (向量存储)

Memora 是记忆系统的主存储层。所有内容首先进入 Memora 向量库（nomic-embed-text-v1.5, JSONL），提供语义搜索和去重。

## 核心能力

| 能力 | 说明 |
|------|------|
| **智能摄入** | 通过 Memory Hub 自动路由：短文本 → Memora，长文本 → Memora + MSA |
| **语义搜索** | 基于 nomic-embed-text 向量相似度搜索，自动添加 `search_query:` / `search_document:` 前缀 |
| **去重** | 内容哈希 + 向量库 `contains()` 双重去重，防止重复写入 |
| **合并召回** | `recall` 同时搜索 Memora + MSA，合并排序返回 |
| **Ingest Tag** | `--tag thought/share/reference/to_verify` 标记内容认知意图 |

## 使用方法

通过 Memory Server（`memory-cli health` 确认在运行）：

```bash
# 记住（自动路由到 Memora + 可选 MSA + KG 抽取）
memory-cli remember "用户决定使用 xAI Grok 作为主要 LLM" -i 0.9 --tag thought

# 语义搜索（仅 Memora）
memory-cli search "用户的 LLM 偏好"

# 合并搜索（Memora + MSA）
memory-cli recall "用户的 LLM 偏好"

# 直接添加到向量库（不走 Hub 路由）
memory-cli add "内容" -i 0.9 -s openclaw

# 记忆总结
memory-cli digest --days 7

# 查看状态
memory-cli status
```

## 何时使用 Memora

**用 `remember`（Hub 路由）当：**
- 对话中出现重要决策或偏好
- 用户说"记住这个"或类似指令
- 需要同时触发 KG 关系抽取

**用 `recall`（合并搜索）当：**
- 需要回忆用户之前说过什么
- 回复前检查历史上下文

**用 `search`（仅 Memora）当：**
- 只需快速向量搜索，不需要 MSA 长文档结果

## 数据位置

- 向量库: `memory/vectorstore.jsonl`
- 每日日志: `memory/YYYY-MM-DD.md`
- 长期摘要: `memory/second_brain/long_term/` (由 Second Brain digest 生成)
