import os
import logging
import yaml
from pathlib import Path
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from typing import Optional

logger = logging.getLogger(__name__)

_PACKAGE_DIR = Path(__file__).parent
_config_cache: Optional["MSAConfig"] = None


def _resolve_base() -> Path:
    env = os.environ.get("MSA_BASE_DIR")
    if env:
        return Path(env).resolve()
    workspace = _PACKAGE_DIR.parent
    return (workspace / "memory" / "msa").resolve()


class MSAConfig(BaseSettings):
    model_config = {"env_prefix": "MSA_"}

    base_dir: Path = Field(default_factory=_resolve_base)
    routing_keys_path: Optional[Path] = None
    content_store_path: Optional[Path] = None
    state_path: Optional[Path] = None

    top_k: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 64
    max_interleave_rounds: int = 3
    similarity_threshold: float = 0.3

    embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"
    embedding_dimension: int = 768

    def model_post_init(self, __context):
        if self.routing_keys_path is None:
            self.routing_keys_path = self.base_dir / "routing_index.jsonl"
        if self.content_store_path is None:
            self.content_store_path = self.base_dir / "content"
        if self.state_path is None:
            self.state_path = self.base_dir / "state.json"

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, v):
        if v < 1:
            raise ValueError("top_k must be at least 1")
        return v

    @field_validator("similarity_threshold")
    @classmethod
    def validate_threshold(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        return v

    @field_validator("chunk_size")
    @classmethod
    def validate_chunk_size(cls, v):
        if v < 32:
            raise ValueError("chunk_size must be at least 32")
        return v

    def ensure_dirs(self):
        for d in [self.base_dir, self.content_store_path]:
            if d is not None:
                d.mkdir(parents=True, exist_ok=True)


def load_config() -> MSAConfig:
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    config_path = _PACKAGE_DIR / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f) or {}
            cfg = MSAConfig(**yaml_config)
    else:
        cfg = MSAConfig()
    cfg.ensure_dirs()
    _config_cache = cfg
    return cfg
