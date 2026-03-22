# HEARTBEAT.md

> Memory Server 内置 Scheduler 自动执行：晨间简报、灵感碰撞、Chronos 巩固、Digest、KG 扫描。心跳时只需确保服务在线。

## 每次心跳

```bash
memory-cli health || memory-cli server-start
```

## 有好灵感时推送

碰撞结果中 novelty >= 4 的灵感，主动分享给用户（不用等晨间简报）。

## 按 AGENTS.md 轮询

按频率轮检邮件、日历、社交通知。Track in `memory/heartbeat-state.json`。

Late night (23:00-08:00) 除非紧急，保持安静。
