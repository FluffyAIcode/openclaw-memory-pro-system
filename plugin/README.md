# openclaw-plugin-memory-pro

OpenClaw Gateway plugin that adds an enhanced AI memory system — vector store (Memora), document-level MSA, knowledge graph, 7-strategy collision engine, and executable skills with closed-loop evolution.

## Installation

```bash
openclaw plugins install openclaw-plugin-memory-pro
```

## Configuration

Add to your `openclaw.json`:

```json
{
  "plugins": {
    "entries": {
      "memory-pro": {
        "enabled": true,
        "config": {
          "pythonPath": "python3",
          "memoryServerPort": 18790,
          "workspacePath": "/path/to/memory-pro-workspace",
          "autoStart": true,
          "contextInjection": true,
          "contextMaxTokens": 2000
        }
      }
    }
  }
}
```

### Config Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `pythonPath` | string | `"python3"` | Path to the Python 3 interpreter |
| `memoryServerPort` | number | `18790` | Port the memory server listens on |
| `workspacePath` | string | `""` | Path to the memory-pro workspace (auto-detected if empty) |
| `autoStart` | boolean | `true` | Auto-start the memory server on gateway load |
| `contextInjection` | boolean | `true` | Inject recalled memory into agent prompts |
| `contextMaxTokens` | number | `2000` | Max tokens for injected memory context |

## Agent Tools

The plugin registers 5 agent tools:

| Tool | Description |
|------|-------------|
| `memory_remember` | Store information with automatic routing to Memora, MSA, and Chronos |
| `memory_recall` | Three-layer assembled retrieval with token budget control |
| `memory_deep_recall` | LLM-powered multi-hop deep recall |
| `memory_collide` | 7-strategy attention-aware inspiration collision |
| `memory_skills` | List active skills with utility stats and executable prompts |

## Context Injection

When `contextInjection` is enabled, the plugin hooks into `before_prompt_build` to automatically recall relevant memories and prepend them to the agent's context.

## HTTP Routes

| Route | Auth | Description |
|-------|------|-------------|
| `GET /memory-pro/health` | gateway | Plugin + memory server health |
| `GET /memory-pro/status` | gateway | Full memory system status |

## Prerequisites

- Python 3.9+
- The [memory-pro workspace](https://github.com/FluffyAIcode/openclaw-memory-pro-system) cloned and set up
- An LLM API key (OpenRouter preferred, xAI fallback)

## License

MIT
