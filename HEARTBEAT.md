# HEARTBEAT.md

## 定期任务

### 🧠 Memora 记忆维护（每天一次）

检查距离上次提炼是否已超过 24 小时。如果是，执行：

```bash
python3 -m memora digest --days 7
```

然后在 `memory/heartbeat-state.json` 中更新 `lastChecks.memora_digest` 为当前时间戳。

### 📝 日常检查（按 AGENTS.md 指引轮询）

- 检查邮件、日历、社交通知等（按 AGENTS.md 中的频率）
