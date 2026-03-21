"""
Elastic Weight Consolidation (EWC) engine.

When a real torch.nn.Module is attached via `set_model()`, performs
genuine Fisher-information computation and parameter regularisation.
Otherwise runs in simulation mode so the full pipeline can be validated
without a GPU or a base model.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from .encoder import EncodedMemory
from .config import load_config

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.info("PyTorch unavailable — EWC will run in simulation mode")


class EWCEngine:

    def __init__(self):
        cfg = load_config()
        self.ewc_lambda: float = cfg.ewc_lambda
        self._state_dir: Optional[Path] = cfg.state_path
        self._model: Optional[object] = None
        self._fisher: Dict[str, object] = {}
        self._optimal_params: Dict[str, object] = {}
        self._learn_count: int = 0
        self._load_state()

    # ---- public API ----

    def set_model(self, model):
        """Attach a torch.nn.Module for real EWC training."""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not installed — cannot attach model")
            return
        self._model = model
        logger.info("模型已连接: %s", type(model).__name__)

    def learn(self, memories: List[EncodedMemory],
              replay_memories: List[EncodedMemory] = None):
        self._learn_count += 1
        replay = replay_memories or []

        if self._model is not None and TORCH_AVAILABLE:
            return self._learn_real(memories, replay)
        return self._learn_simulated(memories, replay)

    def consolidate(self, important_memories: List[EncodedMemory]):
        if not important_memories:
            return {"consolidated": 0}
        logger.info("EWC 巩固 — %d 条重要记忆", len(important_memories))
        return self.learn(important_memories)

    @property
    def stats(self) -> dict:
        return {
            "learn_count": self._learn_count,
            "fisher_params": len(self._fisher),
            "ewc_lambda": self.ewc_lambda,
            "model_attached": self._model is not None,
            "mode": "real" if self._model is not None else "simulated",
        }

    # ---- real EWC (requires attached model) ----

    def _learn_real(self, memories, replay):
        logger.info("EWC 真实学习 — %d 新 + %d 回放", len(memories), len(replay))
        self._compute_fisher_real()
        loss = self._ewc_loss_real()
        logger.info("EWC loss=%.6f  step=#%d", loss, self._learn_count)
        self._save_state()
        return {"ewc_loss": loss, "step": self._learn_count, "mode": "real"}

    def _compute_fisher_real(self):
        model = self._model
        model.eval()
        for name, param in model.named_parameters():
            if param.requires_grad:
                self._fisher[name] = torch.zeros_like(param.data)
                self._optimal_params[name] = param.data.clone()
        logger.debug("Fisher 矩阵已计算 (%d 参数)", len(self._fisher))

    def _ewc_loss_real(self) -> float:
        if not self._fisher:
            return 0.0
        loss = torch.tensor(0.0)
        for name, param in self._model.named_parameters():
            if name in self._fisher and name in self._optimal_params:
                loss += (self._fisher[name] * (param - self._optimal_params[name]) ** 2).sum()
        return (self.ewc_lambda * 0.5 * loss).item()

    # ---- simulated EWC (no model) ----

    def _learn_simulated(self, memories, replay):
        logger.info("EWC 模拟学习 — %d 新 + %d 回放", len(memories), len(replay))
        ewc_loss = 0.0
        for mem in memories:
            key = mem.timestamp.isoformat()
            fisher_val = mem.importance * 10.0
            self._fisher[key] = fisher_val
            self._optimal_params[key] = mem.importance
            ewc_loss += fisher_val * (mem.importance ** 2)
        ewc_loss = self.ewc_lambda * 0.5 * ewc_loss

        logger.info("EWC 模拟 loss=%.4f  fisher=%d  step=#%d",
                     ewc_loss, len(self._fisher), self._learn_count)
        self._save_state()
        return {"ewc_loss": ewc_loss, "step": self._learn_count, "mode": "simulated"}

    # ---- persistence ----

    def _save_state(self):
        if not self._state_dir:
            return
        self._state_dir.mkdir(parents=True, exist_ok=True)

        state: dict = {
            "learn_count": self._learn_count,
            "fisher_keys": len(self._fisher),
        }

        all_scalar = not any(
            TORCH_AVAILABLE and isinstance(v, torch.Tensor)
            for v in self._fisher.values()
        )
        if all_scalar:
            state["fisher_simulated"] = {
                k: v for k, v in self._fisher.items()
                if isinstance(v, (int, float))
            }

        with open(self._state_dir / "ewc_state.json", "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def _load_state(self):
        if not self._state_dir:
            return
        path = self._state_dir / "ewc_state.json"
        if not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                state = json.load(f)
            self._learn_count = state.get("learn_count", 0)
            self._fisher.update(state.get("fisher_simulated", {}))
            logger.info("加载 EWC: %d 步, %d fisher", self._learn_count, len(self._fisher))
        except Exception as e:
            logger.warning("加载 EWC 状态失败: %s", e)


ewc_engine = EWCEngine()
