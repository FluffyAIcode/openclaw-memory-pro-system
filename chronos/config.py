import os
import logging
import yaml
from pathlib import Path
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from typing import Optional

logger = logging.getLogger(__name__)

_PACKAGE_DIR = Path(__file__).parent

_config_cache: Optional["ChronosConfig"] = None


def _resolve_base() -> Path:
    env = os.environ.get("CHRONOS_BASE_DIR")
    if env:
        return Path(env).resolve()
    workspace = _PACKAGE_DIR.parent
    return (workspace / "memory" / "chronos").resolve()


class ChronosConfig(BaseSettings):
    model_config = {"env_prefix": "CHRONOS_"}

    base_dir: Path = Field(default_factory=_resolve_base)
    buffer_path: Optional[Path] = None
    state_path: Optional[Path] = None

    max_buffer_size: int = 2000
    importance_threshold: float = 0.6
    consolidation_interval_hours: int = 6

    ewc_lambda: float = 5000.0
    max_lora_adapters: int = 32
    lora_rank: int = 8
    base_model_name: str = "Qwen/Qwen2.5-7B"

    def model_post_init(self, __context):
        if self.buffer_path is None:
            self.buffer_path = self.base_dir / "replay_buffer.jsonl"
        if self.state_path is None:
            self.state_path = self.base_dir / "state"

    @field_validator("importance_threshold")
    @classmethod
    def validate_importance(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("importance_threshold must be between 0.0 and 1.0")
        return v

    @field_validator("consolidation_interval_hours")
    @classmethod
    def validate_hours(cls, v):
        if v < 1:
            raise ValueError("consolidation_interval_hours must be at least 1")
        return v

    def ensure_dirs(self):
        for d in [self.base_dir, self.state_path]:
            if d is not None:
                d.mkdir(parents=True, exist_ok=True)


def load_config() -> ChronosConfig:
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    config_path = _PACKAGE_DIR / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f) or {}
            cfg = ChronosConfig(**yaml_config)
    else:
        cfg = ChronosConfig()
    cfg.ensure_dirs()
    _config_cache = cfg
    return cfg
