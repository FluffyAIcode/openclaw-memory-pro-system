/**
 * OpenClaw Gateway Plugin — Memory Pro System v0.0.7
 *
 * Provides agent tools, a background service managing the Python memory server,
 * a before_prompt_build hook for automatic memory injection, and HTTP status routes.
 */

import { spawn, ChildProcess } from "child_process";
import { resolve, dirname } from "path";
import { existsSync, readFileSync } from "fs";
import { IncomingMessage, ServerResponse } from "http";
// Tool parameters use plain JSON Schema (TypeBox compatible)

// ---------------------------------------------------------------------------
// Config shape (mirrors openclaw.plugin.json configSchema)
// ---------------------------------------------------------------------------

interface MemoryProConfig {
  pythonPath?: string;
  memoryServerPort?: number;
  workspacePath?: string;
  autoStart?: boolean;
  contextInjection?: boolean;
  contextMaxTokens?: number;
}

const DEFAULTS: Required<MemoryProConfig> = {
  pythonPath: "python3",
  memoryServerPort: 18790,
  workspacePath: "",
  autoStart: true,
  contextInjection: true,
  contextMaxTokens: 4000,
};

// ---------------------------------------------------------------------------
// Internal state
// ---------------------------------------------------------------------------

let serverProcess: ChildProcess | null = null;
let serverReady = false;
let authToken: string | null = null;

function resolvedConfig(pluginConfig?: Record<string, unknown>): Required<MemoryProConfig> {
  const raw = pluginConfig ?? {};
  const inner = (typeof raw.config === "object" && raw.config !== null)
    ? raw.config as Record<string, unknown>
    : raw;
  return { ...DEFAULTS, ...inner } as Required<MemoryProConfig>;
}

function baseUrl(cfg: Required<MemoryProConfig>): string {
  return `http://127.0.0.1:${cfg.memoryServerPort}`;
}

function workspaceDir(cfg: Required<MemoryProConfig>): string {
  if (cfg.workspacePath) return cfg.workspacePath;
  return resolve(dirname(__filename), "..");
}

// ---------------------------------------------------------------------------
// Auth token
// ---------------------------------------------------------------------------

function loadAuthToken(cfg: Required<MemoryProConfig>): string | null {
  if (authToken !== null) return authToken || null;
  const wsDir = workspaceDir(cfg);
  const tokenPath = resolve(wsDir, "memory", "security", ".auth_token");
  try {
    if (existsSync(tokenPath)) {
      authToken = readFileSync(tokenPath, "utf-8").trim();
      return authToken || null;
    }
  } catch {
    // token file not readable — fall through
  }
  authToken = "";
  return null;
}

function authHeaders(cfg: Required<MemoryProConfig>): Record<string, string> {
  const token = loadAuthToken(cfg);
  if (token) {
    return { "Content-Type": "application/json", "Authorization": `Bearer ${token}` };
  }
  return { "Content-Type": "application/json" };
}

// ---------------------------------------------------------------------------
// HTTP helpers
// ---------------------------------------------------------------------------

async function memoryPost(
  cfg: Required<MemoryProConfig>,
  path: string,
  body: Record<string, unknown>,
): Promise<unknown> {
  const url = `${baseUrl(cfg)}${path}`;
  const resp = await fetch(url, {
    method: "POST",
    headers: authHeaders(cfg),
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`Memory server ${path} returned ${resp.status}: ${text}`);
  }
  return resp.json();
}

async function memoryGet(cfg: Required<MemoryProConfig>, path: string): Promise<unknown> {
  const url = `${baseUrl(cfg)}${path}`;
  const hdrs = authHeaders(cfg);
  const resp = await fetch(url, { headers: hdrs });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`Memory server ${path} returned ${resp.status}: ${text}`);
  }
  return resp.json();
}

// ---------------------------------------------------------------------------
// Tool helpers
// ---------------------------------------------------------------------------

function textResult(text: string) {
  return {
    content: [{ type: "text" as const, text }],
    details: {},
  };
}

async function toolExec(
  cfg: Required<MemoryProConfig>,
  method: "GET" | "POST",
  path: string,
  body?: Record<string, unknown>,
) {
  try {
    const data = method === "POST"
      ? await memoryPost(cfg, path, body ?? {})
      : await memoryGet(cfg, path);
    return textResult(JSON.stringify(data, null, 2));
  } catch (e) {
    return textResult(`Error: ${e}`);
  }
}

// ---------------------------------------------------------------------------
// Plugin entry — OpenClawPluginDefinition shape
// ---------------------------------------------------------------------------

export const id = "memory-pro";
export const name = "Memory Pro System";
export const version = "0.0.7";
export const description =
  "Enhanced AI memory — vector store, MSA, KG, collision engine, executable skills.";

export async function register(api: any): Promise<void> {
  const logger = api.logger;
  const cfg = resolvedConfig(api.pluginConfig);

  logger.info(`Memory Pro plugin v0.0.7 — registering... (contextInjection=${cfg.contextInjection}, maxTokens=${cfg.contextMaxTokens})`);

  // -----------------------------------------------------------------------
  // 1. Background service: spawn/manage memory_server.py
  // -----------------------------------------------------------------------
  api.registerService({
    id: "memory-server",

    async start(ctx: any) {
      if (serverProcess && !serverProcess.killed) {
        logger.info("Memory server already running.");
        return;
      }

      const ws = workspaceDir(cfg);
      const serverScript = resolve(ws, "memory_server.py");

      if (!existsSync(serverScript)) {
        logger.error(`memory_server.py not found at ${serverScript}`);
        return;
      }

      logger.info(`Starting memory server: ${cfg.pythonPath} ${serverScript} --port ${cfg.memoryServerPort}`);

      serverProcess = spawn(
        cfg.pythonPath,
        [serverScript, "--port", String(cfg.memoryServerPort)],
        {
          cwd: ws,
          stdio: ["ignore", "pipe", "pipe"],
          env: { ...process.env, PYTHONUNBUFFERED: "1" },
        },
      );

      serverProcess.stdout?.on("data", (chunk: Buffer) => {
        const line = chunk.toString().trim();
        if (line) logger.info(`[mem-srv] ${line}`);
        if (line.includes("ready") || line.includes("Listening")) {
          serverReady = true;
        }
      });

      serverProcess.stderr?.on("data", (chunk: Buffer) => {
        const line = chunk.toString().trim();
        if (line) logger.warn(`[mem-srv] ${line}`);
      });

      serverProcess.on("exit", (code: number | null) => {
        logger.warn(`Memory server exited with code ${code}`);
        serverProcess = null;
        serverReady = false;
      });

      const deadline = Date.now() + 60_000;
      while (!serverReady && Date.now() < deadline) {
        await new Promise((r) => setTimeout(r, 1000));
        try {
          const resp = await fetch(`${baseUrl(cfg)}/health`);
          if (resp.ok) {
            serverReady = true;
            break;
          }
        } catch {
          // not ready yet
        }
      }

      if (serverReady) {
        logger.info("Memory server is ready.");
      } else {
        logger.warn("Memory server did not become ready within 60s.");
      }
    },

    async stop() {
      if (serverProcess && !serverProcess.killed) {
        logger.info("Stopping memory server...");
        serverProcess.kill("SIGTERM");
        await new Promise((r) => setTimeout(r, 2000));
        if (serverProcess && !serverProcess.killed) {
          serverProcess.kill("SIGKILL");
        }
        serverProcess = null;
        serverReady = false;
      }
    },
  });

  // -----------------------------------------------------------------------
  // 2. Agent tools (AnyAgentTool shape: name, description, parameters, label, execute)
  // -----------------------------------------------------------------------

  api.registerTool({
    name: "memory_remember",
    description:
      "Store information in long-term memory. Routes to Memora (vector), MSA (long/important text), and Chronos (high-importance).",
    label: "Memory Remember",
    parameters: {
      type: "object",
      required: ["content"],
      properties: {
        content: { type: "string", description: "The text content to remember." },
        source: { type: "string", description: "Origin label.", default: "agent" },
        importance: { type: "number", description: "Importance 0.0-1.0. >=0.85 triggers MSA + Chronos.", default: 0.7 },
        tags: { type: "string", description: "Comma-separated tags.", default: "" },
      },
    },
    async execute(_toolCallId: string, params: any) {
      return toolExec(cfg, "POST", "/remember", {
        content: params.content,
        source: params.source ?? "agent",
        importance: params.importance ?? 0.7,
        tags: params.tags ?? "",
      });
    },
  });

  api.registerTool({
    name: "memory_recall",
    description:
      "Recall relevant memories — three-layer assembled retrieval: skills, KG relations, and evidence with token budget control.",
    label: "Memory Recall",
    parameters: {
      type: "object",
      required: ["query"],
      properties: {
        query: { type: "string", description: "What to recall." },
        top_k: { type: "number", description: "Max results.", default: 8 },
        max_tokens: { type: "number", description: "Token budget.", default: 4000 },
      },
    },
    async execute(_toolCallId: string, params: any) {
      return toolExec(cfg, "POST", "/recall", {
        query: params.query,
        top_k: params.top_k ?? 8,
        max_tokens: params.max_tokens ?? 4000,
      });
    },
  });

  api.registerTool({
    name: "memory_deep_recall",
    description:
      "LLM-powered multi-hop deep recall — iteratively retrieves and reasons until sufficient answer.",
    label: "Memory Deep Recall",
    parameters: {
      type: "object",
      required: ["query"],
      properties: {
        query: { type: "string", description: "Complex question for deep recall." },
        max_hops: { type: "number", description: "Max retrieval-reasoning hops.", default: 3 },
      },
    },
    async execute(_toolCallId: string, params: any) {
      return toolExec(cfg, "POST", "/deep-recall", {
        query: params.query,
        max_hops: params.max_hops ?? 3,
      });
    },
  });

  api.registerTool({
    name: "memory_collide",
    description:
      "Run an inspiration collision round — 7 attention-aware strategies to generate novel insights from existing memories.",
    label: "Memory Collide",
    parameters: {
      type: "object",
      properties: {
        rounds: { type: "number", description: "Number of collision rounds.", default: 1 },
      },
    },
    async execute(_toolCallId: string, params: any) {
      return toolExec(cfg, "POST", "/second-brain/collide", {
        rounds: params.rounds ?? 1,
      });
    },
  });

  api.registerTool({
    name: "memory_skills",
    description:
      "List active skills from the Skill Registry, with utility stats and executable prompts.",
    label: "Memory Skills",
    parameters: { type: "object", properties: {} },
    async execute() {
      return toolExec(cfg, "GET", "/skills/active");
    },
  });

  // -----------------------------------------------------------------------
  // 3. Context hook: inject recalled memory into agent prompts
  // -----------------------------------------------------------------------

  if (cfg.contextInjection) {
    api.on(
      "before_prompt_build",
      async (event: { prompt: string; messages: unknown[] }, _ctx: any) => {
        const userMessage = event.prompt ?? "";
        if (!userMessage) return;

        try {
          const result = (await memoryPost(cfg, "/recall", {
            query: userMessage,
            top_k: 8,
            max_tokens: cfg.contextMaxTokens,
          })) as {
            merged?: Array<{ content?: string; system?: string; layer?: string; score?: number; composite_score?: number }>;
            intent?: string;
            layers?: Record<string, Array<{ content?: string; system?: string; score?: number; composite_score?: number }>>;
            composer_warnings?: string[];
          };

          const layers = result.layers ?? {};
          const intent = result.intent ?? "";
          const hasContent = Object.values(layers).some((items) => items.length > 0);
          if (!hasContent && !(result.merged ?? []).length) return;

          const sections: string[] = [];

          const intentLabel: Record<string, string> = {
            factual: "事实查询",
            thinking: "深度思考",
            planning: "规划决策",
            review: "回顾总结",
          };
          sections.push(`## Recalled Memory Context (${intentLabel[intent] ?? intent})`);
          sections.push("");

          const coreItems = layers.core ?? [];
          if (coreItems.length > 0) {
            sections.push("### Active Knowledge");
            for (const item of coreItems) {
              sections.push(`- ${item.content ?? ""}`);
            }
            sections.push("");
          }

          const conceptItems = layers.concept ?? [];
          if (conceptItems.length > 0) {
            sections.push("### Related Concepts");
            for (const item of conceptItems) {
              sections.push(`- ${item.content ?? ""}`);
            }
            sections.push("");
          }

          const bgItems = layers.background ?? [];
          if (bgItems.length > 0) {
            sections.push("### Background");
            for (const item of bgItems) {
              sections.push(`- ${item.content ?? ""}`);
            }
            sections.push("");
          }

          const conflictItems = layers.conflict ?? [];
          if (conflictItems.length > 0) {
            sections.push("### ⚠ Conflicts");
            for (const item of conflictItems) {
              sections.push(`- ${item.content ?? ""}`);
            }
            sections.push("");
          }

          if (sections.length <= 2) {
            const merged = result.merged ?? [];
            if (merged.length === 0) return;
            for (const m of merged) {
              const tag = m.system ?? m.layer ?? "memory";
              sections.push(`- [${tag}] ${m.content ?? ""}`);
            }
            sections.push("");
          }

          return { prependContext: sections.join("\n") };
        } catch (e) {
          logger.warn(`Context injection failed: ${e}`);
          return;
        }
      },
      { priority: 10 },
    );
  }

  // -----------------------------------------------------------------------
  // 4. HTTP routes (auth: "gateway" uses the gateway's own auth)
  // -----------------------------------------------------------------------

  api.registerHttpRoute({
    path: "/memory-pro/health",
    auth: "gateway",
    async handler(req: IncomingMessage, res: ServerResponse) {
      try {
        const data = await memoryGet(cfg, "/health");
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ plugin: "memory-pro", version: "0.0.7", server: data }));
      } catch (e) {
        res.writeHead(502, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ plugin: "memory-pro", error: "Memory server unreachable", detail: String(e) }));
      }
    },
  });

  api.registerHttpRoute({
    path: "/memory-pro/status",
    auth: "gateway",
    async handler(req: IncomingMessage, res: ServerResponse) {
      try {
        const data = await memoryGet(cfg, "/status");
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ plugin: "memory-pro", version: "0.0.7", status: data }));
      } catch (e) {
        res.writeHead(502, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ plugin: "memory-pro", error: "Memory server unreachable", detail: String(e) }));
      }
    },
  });

  // -----------------------------------------------------------------------
  // 5. Auto-start if configured
  // -----------------------------------------------------------------------

  if (cfg.autoStart) {
    logger.info("Auto-starting memory server...");
  }

  logger.info("Memory Pro plugin registered successfully.");
}
