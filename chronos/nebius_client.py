"""
Nebius AI Cloud fine-tuning client — skeleton implementation.

All API paths, payload formats, and auth mechanisms are placeholders.
See docs/NEBIUS_FINETUNE_INTEGRATION_SKETCH.md and https://docs.nebius.com/
for the real API spec.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class NebiusConfig:
    def __init__(self):
        self.api_base = os.environ.get("NEBIUS_API_BASE", "")
        self.api_key = os.environ.get("NEBIUS_API_KEY", "")
        self.base_model = os.environ.get(
            "NEBIUS_BASE_MODEL", "meta-llama-3.1-8b-instruct"
        )

    @property
    def is_configured(self) -> bool:
        return bool(self.api_base and self.api_key)


class NebiusClient:
    """Thin wrapper for Nebius fine-tuning API.

    Lifecycle:
      1. upload_dataset(jsonl_path) → dataset_id
      2. create_job(dataset_id, **hyperparams) → job_id
      3. poll_job(job_id) → status dict with adapter_uri or endpoint
    """

    def __init__(self):
        self._config = NebiusConfig()

    @property
    def is_configured(self) -> bool:
        return self._config.is_configured

    def upload_dataset(self, local_path: Path) -> str:
        if not self._config.is_configured:
            raise RuntimeError("NEBIUS_API_BASE and NEBIUS_API_KEY not set")
        if not local_path.exists():
            raise FileNotFoundError(f"Dataset not found: {local_path}")

        # TODO: implement actual upload — multipart or presigned URL
        # See Nebius docs: dataset upload API
        logger.info("Nebius: would upload %s (%d bytes)",
                     local_path, local_path.stat().st_size)
        raise NotImplementedError(
            "Nebius dataset upload not yet implemented. "
            "See docs/NEBIUS_FINETUNE_INTEGRATION_SKETCH.md"
        )

    def create_job(self, dataset_id: str, *,
                   lora_r: int = 16,
                   lora_alpha: int = 16,
                   epochs: int = 3,
                   learning_rate: float = 2e-4) -> str:
        if not self._config.is_configured:
            raise RuntimeError("NEBIUS_API_BASE and NEBIUS_API_KEY not set")

        # TODO: implement actual job creation
        # See Nebius docs: create fine-tune job
        logger.info("Nebius: would create job (model=%s, dataset=%s, "
                     "lora_r=%d, epochs=%d)",
                     self._config.base_model, dataset_id, lora_r, epochs)
        raise NotImplementedError(
            "Nebius job creation not yet implemented. "
            "See docs/NEBIUS_FINETUNE_INTEGRATION_SKETCH.md"
        )

    def poll_job(self, job_id: str, timeout: int = 7200,
                 interval: int = 30) -> dict:
        # TODO: implement actual polling
        raise NotImplementedError(
            "Nebius job polling not yet implemented. "
            "See docs/NEBIUS_FINETUNE_INTEGRATION_SKETCH.md"
        )

    def status(self) -> dict:
        return {
            "configured": self._config.is_configured,
            "api_base": self._config.api_base[:30] + "..." if self._config.api_base else "",
            "base_model": self._config.base_model,
        }


nebius_client = NebiusClient()
