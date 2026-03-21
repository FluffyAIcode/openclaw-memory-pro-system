# HEARTBEAT.md

## 定期任务

### 🖥️ Memory Server 健康检查（每次心跳）

确认 Memory Server 在运行。如果不在运行，启动它：

```bash
memory-cli health || memory-cli server-start
```

### ☀️ 晨间简报推送（每天 1 次，早上首次心跳时）

检查 `lastChecks.daily_briefing`。如果距离上次推送超过 20 小时，执行：

```bash
memory-cli briefing
```

将简报内容发送给用户（通过当前会话渠道）。简报包含：
- 昨夜灵感碰撞结果
- 沉睡记忆提醒
- 近期趋势话题
- 记忆活力概况

然后更新 `lastChecks.daily_briefing` 时间戳。

### 🧠 Memora 记忆维护（每天一次）

检查距离上次提炼是否已超过 24 小时。如果是，执行：

```bash
memory-cli digest --days 7
```

然后更新 `lastChecks.memora_digest` 时间戳。

### ⚡ Chronos 记忆巩固（每 6 小时一次）

检查距离上次巩固是否已超过 6 小时。如果是，执行：

```bash
python3 -m chronos consolidate
```

然后更新 `lastChecks.chronos_consolidate` 时间戳。巩固完成后检查 `PERSONALITY.yaml` 是否已更新。

### 🧠 第二大脑灵感碰撞 + 推送（每 6 小时一次）

检查距离上次碰撞是否已超过 6 小时。如果是，执行：

```bash
memory-cli collide
```

然后更新 `lastChecks.second_brain_collide` 时间戳。

**灵感推送规则：** 碰撞完成后，检查输出中是否有 novelty >= 4 的灵感。如果有，主动发送给用户：

```
💡 刚刚的灵感碰撞发现了一条高新颖度的联系：
[策略名] 联系: ...
灵感: ...
```

不需要等到晨间简报，有好灵感就立即分享。

### 💤 沉睡记忆提醒（每 3 天一次）

检查 `lastChecks.dormant_check`。如果距离上次超过 3 天，执行：

```bash
memory-cli review-dormant
```

如果有沉睡记忆，挑选最重要的 1-2 条，以对话方式提醒用户：

```
你之前提到过 X（两周前），最近没有再讨论过。是否需要跟进？
```

然后更新 `lastChecks.dormant_check` 时间戳。

### 🕸️ 知识图谱维护（每天 1 次）

检查 `lastChecks.kg_maintenance`。如果距离上次超过 24 小时，执行：

```bash
memory-cli graph-status
```

如果节点数 > 0，同时运行矛盾扫描：

```bash
memory-cli contradictions
```

如果发现高风险矛盾（risk > 0.5），在下次与用户对话时主动提醒。

然后更新 `lastChecks.kg_maintenance` 时间戳。

### 🔍 盲区扫描（每周 1 次）

检查 `lastChecks.blindspot_scan`。如果距离上次超过 7 天，执行：

```bash
memory-cli blindspots
```

如果发现有盲区的重要决策，生成提醒推送给用户。

然后更新 `lastChecks.blindspot_scan` 时间戳。

### 📝 日常检查（按 AGENTS.md 指引轮询）

- 检查邮件、日历、社交通知等（按 AGENTS.md 中的频率）
