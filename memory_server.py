#!/usr/bin/env python3
"""
Memory Server — persistent HTTP service for OpenClaw's memory systems.

Keeps the SentenceTransformer model loaded in memory, shared across
Memora and MSA. Agent calls via curl or memory_cli.py instead of
spawning a new python3 process each time.

Zero external dependencies beyond what's already installed.

Usage:
    python3 memory_server.py                   # foreground
    python3 memory_server.py --port 18790      # custom port
    python3 memory_server.py --daemon           # background (writes PID file)
"""

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Any, Callable, Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [memory-server] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("memory-server")

_WORKSPACE = Path(__file__).parent
_START_TIME = time.time()
_EMBEDDER_TYPE = "unknown"


def _load_env():
    """Load .env from OpenClaw directory."""
    env_path = Path.home() / ".openclaw" / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def _setup_hf_offline():
    """Configure HuggingFace for offline/mirror mode."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _force_cpu_for_daemon():
    """MPS (Apple GPU) is unreliable after setsid(); force CPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTORCH_MPS_ENABLED"] = "0"
    try:
        import torch
        torch.set_default_device("cpu")
    except Exception:
        pass


def _load_shared_embedder():
    """Load SentenceTransformer once, set as shared embedder."""
    global _EMBEDDER_TYPE

    import shared_embedder

    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        model_name = "nomic-ai/nomic-embed-text-v1.5"
        logger.info("Loading SentenceTransformer: %s ...", model_name)
        device = "cpu" if os.environ.get("PYTORCH_MPS_ENABLED") == "0" else None
        model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
        logger.info("Model loaded successfully")

        class SharedSentenceEmbedder:
            """Wraps SentenceTransformer with the embed/embed_batch interface."""
            def __init__(self, m, dim=768):
                self._model = m
                self.dimension = dim

            def embed(self, text: str):
                return self._model.encode(text, normalize_embeddings=True).tolist()

            def embed_batch(self, texts):
                return self._model.encode(texts, normalize_embeddings=True).astype(np.float32)

            def embed_np(self, text: str):
                return self._model.encode(text, normalize_embeddings=True).astype(np.float32)

        emb = SharedSentenceEmbedder(model)
        shared_embedder.set(emb)
        _EMBEDDER_TYPE = "SentenceTransformer"
        logger.info("Shared embedder ready (SentenceTransformer)")
        return emb

    except Exception as e:
        logger.warning("SentenceTransformer load failed (%s), using MockEmbedder", e)
        import hashlib
        import numpy as np

        class SharedMockEmbedder:
            def __init__(self, dim=768):
                self.dimension = dim

            def embed(self, text: str):
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                raw = [int(digest[i:i+2], 16) / 255.0 for i in range(0, min(len(digest), self.dimension*2), 2)]
                while len(raw) < self.dimension:
                    raw.append(0.5)
                vec = np.array(raw[:self.dimension], dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec /= norm
                return vec.tolist()

            def embed_batch(self, texts):
                return np.array([np.array(self.embed(t), dtype=np.float32) for t in texts])

            def embed_np(self, text: str):
                return np.array(self.embed(text), dtype=np.float32)

        emb = SharedMockEmbedder()
        shared_embedder.set(emb)
        _EMBEDDER_TYPE = "MockEmbedder"
        logger.info("Shared embedder ready (MockEmbedder fallback)")
        return emb


_hub = None

def _get_hub():
    global _hub
    if _hub is None:
        from memory_hub import hub
        _hub = hub
    return _hub


def _track_access_async(results: list, query: str):
    """Record access events for returned search results (non-blocking)."""
    import threading
    def _do_track():
        try:
            from second_brain.bridge import bridge as sb_bridge
            for r in results[:5]:
                content = r.get("content", "")
                sb_bridge.track_access(
                    memory_id=r.get("metadata", {}).get("source", ""),
                    content=content,
                    query=query,
                )
        except Exception:
            pass
    threading.Thread(target=_do_track, daemon=True).start()


class TaskManager:
    """In-memory async task queue for long-running operations."""

    def __init__(self, max_workers: int = 3):
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._tasks: Dict[str, dict] = {}
        self._lock = threading.Lock()

    def submit(self, name: str, fn: Callable, *args, **kwargs) -> str:
        task_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._tasks[task_id] = {
                "id": task_id,
                "name": name,
                "status": "running",
                "submitted_at": time.time(),
                "result": None,
                "error": None,
                "completed_at": None,
            }

        def _run():
            try:
                result = fn(*args, **kwargs)
                with self._lock:
                    self._tasks[task_id]["status"] = "done"
                    self._tasks[task_id]["result"] = result
                    self._tasks[task_id]["completed_at"] = time.time()
                elapsed = self._tasks[task_id]["completed_at"] - self._tasks[task_id]["submitted_at"]
                logger.info("Task %s (%s) completed in %.1fs", task_id, name, elapsed)
            except Exception as e:
                with self._lock:
                    self._tasks[task_id]["status"] = "error"
                    self._tasks[task_id]["error"] = str(e)
                    self._tasks[task_id]["completed_at"] = time.time()
                logger.error("Task %s (%s) failed: %s", task_id, name, e, exc_info=True)

        self._pool.submit(_run)
        return task_id

    def get(self, task_id: str) -> Optional[dict]:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            info = {
                "id": task["id"],
                "name": task["name"],
                "status": task["status"],
                "submitted_at": task["submitted_at"],
                "elapsed": round(
                    (task["completed_at"] or time.time()) - task["submitted_at"], 1
                ),
            }
            if task["status"] == "done":
                info["result"] = task["result"]
            elif task["status"] == "error":
                info["error"] = task["error"]
            return info

    def list_tasks(self) -> list:
        with self._lock:
            result = []
            for task in self._tasks.values():
                info = {
                    "id": task["id"],
                    "name": task["name"],
                    "status": task["status"],
                    "elapsed": round(
                        (task["completed_at"] or time.time()) - task["submitted_at"], 1
                    ),
                }
                if task["status"] == "error":
                    info["error"] = task["error"]
                result.append(info)
            return result

    def cleanup(self, max_age: float = 3600):
        now = time.time()
        with self._lock:
            expired = [
                tid for tid, t in self._tasks.items()
                if t["status"] in ("done", "error")
                and t["completed_at"]
                and now - t["completed_at"] > max_age
            ]
            for tid in expired:
                del self._tasks[tid]
        if expired:
            logger.info("Cleaned up %d expired tasks", len(expired))


_task_manager = TaskManager(max_workers=3)

_SLOW_ENDPOINTS = frozenset({
    "/second-brain/collide",
    "/second-brain/deep-collide",
    "/deep-recall",
    "/msa/interleave",
    "/chronos/consolidate",
    "/digest",
})


def _execute_endpoint(path: str, body: dict) -> dict:
    """Core dispatch logic. Stateless — safe to call from any thread."""
    hub = _get_hub()

    if path == "/remember":
        return hub.remember(
            content=body.get("content", ""),
            source=body.get("source", "openclaw"),
            importance=body.get("importance", 0.7),
            doc_id=body.get("doc_id"),
            title=body.get("title"),
            force_systems=body.get("force_systems"),
        )

    elif path == "/recall":
        result = hub.recall(
            query=body.get("query", ""),
            top_k=body.get("top_k", 8),
        )
        _track_access_async(result.get("merged", []), body.get("query", ""))
        return result

    elif path == "/deep-recall":
        return hub.deep_recall(
            query=body.get("query", ""),
            max_rounds=body.get("max_rounds", 3),
        )

    elif path == "/search":
        from memora.vectorstore import vector_store
        results = vector_store.search(
            query=body.get("query", ""),
            limit=body.get("limit", 8),
        )
        _track_access_async(results, body.get("query", ""))
        return {"results": results}

    elif path == "/add":
        from memora.collector import collector
        from memora.vectorstore import vector_store
        content = body.get("content", "")
        source = body.get("source", "cli")
        importance = body.get("importance", 0.7)
        entry = collector.collect(content, source=source, importance=importance)
        vector_store.add(content, metadata={
            "source": source,
            "importance": importance,
            "timestamp": entry["timestamp"],
        })
        return entry

    elif path == "/digest":
        from memora.digest import digest_memories
        days = body.get("days", 7)
        ok = digest_memories(days=days)
        return {"success": ok, "days": days}

    elif path == "/chronos/learn":
        from chronos.bridge import bridge as chronos_bridge
        result = chronos_bridge.learn_and_save(
            content=body.get("content", ""),
            source=body.get("source", "openclaw"),
            importance=body.get("importance", 0.75),
        )
        return {"importance": result.importance, "timestamp": result.timestamp.isoformat()}

    elif path == "/chronos/consolidate":
        from chronos.bridge import bridge as chronos_bridge
        return chronos_bridge.consolidate()

    elif path == "/msa/ingest":
        from msa.bridge import bridge as msa_bridge
        return msa_bridge.ingest_and_save(
            content=body.get("content", ""),
            source=body.get("source", "openclaw"),
            doc_id=body.get("doc_id"),
            metadata=body.get("metadata"),
        )

    elif path == "/msa/query":
        from msa.bridge import bridge as msa_bridge
        return msa_bridge.query_memory(
            question=body.get("query", ""),
            top_k=body.get("top_k"),
        )

    elif path == "/msa/interleave":
        from msa.bridge import bridge as msa_bridge
        return msa_bridge.interleave_query(
            question=body.get("query", ""),
            max_rounds=body.get("max_rounds"),
        )

    elif path == "/second-brain/collide":
        from second_brain.bridge import bridge as sb_bridge
        return sb_bridge.collide()

    elif path == "/second-brain/deep-collide":
        from second_brain.bridge import bridge as sb_bridge
        return sb_bridge.deep_collide(topic=body.get("topic", ""))

    elif path == "/second-brain/track":
        from second_brain.bridge import bridge as sb_bridge
        sb_bridge.track_access(
            memory_id=body.get("memory_id", ""),
            content=body.get("content", ""),
            query=body.get("query", ""),
        )
        return {"ok": True}

    else:
        raise ValueError(f"Unknown endpoint: {path}")


class MemoryHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the memory API."""

    def do_POST(self):
        try:
            self._handle_post()
        except Exception:
            logger.exception("Unhandled POST error on %s", self.path)
            try:
                self._respond(500, {"error": "Internal server error"})
            except Exception:
                pass

    def _handle_post(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length)) if content_length else {}
        except (json.JSONDecodeError, ValueError):
            self._respond(400, {"error": "Invalid JSON"})
            return

        use_async = body.pop("async", False)

        if use_async and self.path in _SLOW_ENDPOINTS:
            self._dispatch_async(body)
            return

        self._dispatch_sync(body)

    def _dispatch_async(self, body: dict):
        """Submit the request to the task manager and return immediately."""
        path = self.path

        def _run():
            return _execute_endpoint(path, body)

        task_id = _task_manager.submit(path, _run)
        logger.info("Async task %s submitted for %s", task_id, path)
        self._respond(202, {
            "task_id": task_id,
            "status": "running",
            "endpoint": path,
            "poll": f"/task/{task_id}",
        })

    def _dispatch_sync(self, body: dict):
        """Execute the request synchronously and return the result."""
        try:
            result = _execute_endpoint(self.path, body)
        except ValueError as e:
            self._respond(404, {"error": str(e)})
            return
        self._respond(200, result)

    def do_GET(self):
        try:
            self._handle_get()
        except Exception:
            logger.exception("Unhandled GET error on %s", self.path)
            try:
                self._respond(500, {"error": "Internal server error"})
            except Exception:
                pass

    def _handle_get(self):
        if self.path == "/health":
            _task_manager.cleanup()
            self._respond(200, {
                "status": "ok",
                "uptime_seconds": round(time.time() - _START_TIME),
                "embedder": _EMBEDDER_TYPE,
                "pid": os.getpid(),
            })

        elif self.path == "/status":
            hub = _get_hub()
            result = hub.status()
            result["server"] = {
                "uptime_seconds": round(time.time() - _START_TIME),
                "embedder": _EMBEDDER_TYPE,
            }
            self._respond(200, result)

        elif self.path.startswith("/task/"):
            task_id = self.path.split("/task/", 1)[1]
            task = _task_manager.get(task_id)
            if task is None:
                self._respond(404, {"error": f"Task {task_id} not found"})
            else:
                self._respond(200, task)

        elif self.path == "/tasks":
            tasks = _task_manager.list_tasks()
            self._respond(200, {"tasks": tasks, "count": len(tasks)})

        elif self.path == "/second-brain/report":
            from second_brain.bridge import bridge as sb_bridge
            result = sb_bridge.report()
            self._respond(200, result)

        elif self.path == "/second-brain/status":
            from second_brain.bridge import bridge as sb_bridge
            result = sb_bridge.status()
            self._respond(200, result)

        else:
            self._respond(404, {"error": f"Unknown endpoint: {self.path}"})

    def _respond(self, code: int, data: Any):
        body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        logger.info("%s %s", self.command, self.path)


class ThreadedServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def _daemonize(pid_file: Path, log_dir: Path):
    """
    Proper Unix double-fork daemon.

    1st fork  — detach from parent terminal
    setsid    — become session leader
    2nd fork  — prevent re-acquiring a controlling terminal
    Redirect  — stdin→/dev/null, stdout/stderr→log files
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = log_dir / "server-stdout.log"
    stderr_log = log_dir / "server-stderr.log"

    pid = os.fork()
    if pid > 0:
        # Parent waits briefly for child to write PID, then exits.
        time.sleep(0.3)
        if pid_file.exists():
            child_pid = pid_file.read_text().strip()
            print(f"Memory server started (PID {child_pid}, port 18790)")
        else:
            print(f"Memory server forked (PID {pid})")
        sys.exit(0)

    os.setsid()

    pid2 = os.fork()
    if pid2 > 0:
        os._exit(0)

    # Now in grandchild — the actual daemon process.
    # Redirect standard file descriptors.
    sys.stdin.close()
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.close(devnull)

    out_fd = os.open(str(stdout_log), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    err_fd = os.open(str(stderr_log), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(out_fd, 1)
    os.dup2(err_fd, 2)
    os.close(out_fd)
    os.close(err_fd)

    # Reattach Python file objects to new FDs.
    sys.stdout = os.fdopen(1, "w", buffering=1)
    sys.stderr = os.fdopen(2, "w", buffering=1)

    # Reconfigure logging to use the new stderr.
    root_logger = logging.getLogger()
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [memory-server] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    ))
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)

    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()))


def main():
    parser = argparse.ArgumentParser(description="OpenClaw Memory Server")
    parser.add_argument("--port", type=int, default=18790, help="Listen port")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address")
    parser.add_argument("--daemon", action="store_true", help="Run in background")
    args = parser.parse_args()

    _load_env()
    _setup_hf_offline()

    # Ignore SIGPIPE — prevents broken-pipe from killing the daemon
    # when a client disconnects mid-response.
    signal.signal(signal.SIGPIPE, signal.SIG_IGN)

    pid_file = _WORKSPACE / "memory" / "server.pid"
    log_dir = _WORKSPACE / "memory"

    if args.daemon:
        _daemonize(pid_file, log_dir)
        _force_cpu_for_daemon()

    _server_ref = None

    def _shutdown(signum, frame):
        logger.info("Shutting down (signal %s)...", signum)
        if _server_ref is not None:
            _server_ref.shutdown()
        if pid_file.exists():
            pid_file.unlink()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()))

    logger.info("Loading shared embedder...")
    _load_shared_embedder()

    server = ThreadedServer((args.host, args.port), MemoryHandler)
    _server_ref = server
    logger.info("Memory Server listening on %s:%d (PID %d)",
                args.host, args.port, os.getpid())
    logger.info("Endpoints: /remember /recall /deep-recall /status /health /search /add /digest")
    logger.info("           /chronos/learn /chronos/consolidate")
    logger.info("           /msa/ingest /msa/query /msa/interleave")
    logger.info("           /second-brain/collide /second-brain/report /second-brain/status /second-brain/track")
    logger.info("           /task/<id> /tasks  (async: POST with {\"async\":true})")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    except Exception:
        logger.exception("Fatal server error")
    finally:
        server.server_close()
        if pid_file.exists():
            pid_file.unlink()
        logger.info("Server stopped")


if __name__ == "__main__":
    main()
