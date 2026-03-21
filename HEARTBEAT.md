# HEARTBEAT.md

## 定期任务

### 🖥️ Memory Server 健康检查（每次心跳）

确认 Memory Server 在运行。如果不在运行，启动它：

```bash
memory-cli health || memory-cli server-start
```

### 🧠 Memora 记忆维护（每天一次）

检查距离上次提炼是否已超过 24 小时。如果是，执行（现在会通过 LLM 生成 AI 摘要）：

```bash
memory-cli digest --days 7
```

然后在 `memory/heartbeat-state.json` 中更新 `lastChecks.memora_digest` 为当前时间戳。

### ⚡ Chronos 记忆巩固（每 6 小时一次）

检查距离上次巩固是否已超过 6 小时。如果是，执行（现在会同时生成 PERSONALITY.yaml 人格档案）：

```bash
python3 -m chronos consolidate
```

然后在 `memory/heartbeat-state.json` 中更新 `lastChecks.chronos_consolidate` 为当前时间戳。

巩固完成后检查 `PERSONALITY.yaml` 是否已更新：

```bash
ls -la PERSONALITY.yaml
```

### 🧠 第二大脑灵感碰撞（每 6 小时一次）

检查距离上次碰撞是否已超过 6 小时。如果是，执行：

```bash
memory-cli collide
```

然后在 `memory/heartbeat-state.json` 中更新 `lastChecks.second_brain_collide` 为当前时间戳。

碰撞结果保存在 `memory/insights/YYYY-MM-DD.md`，高新颖度灵感自动索引到 Memora。

定期查看第二大脑报告，了解记忆健康状况：

```bash
memory-cli sb-report
```

### 📝 日常检查（按 AGENTS.md 指引轮询）

- 检查邮件、日历、社交通知等（按 AGENTS.md 中的频率）
