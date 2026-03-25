import os
import logging
import yaml
from pathlib import Path
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from typing import Optional

logger = logging.getLogger(__name__)

_PACKAGE_DIR = Path(__file__).parent
_config_cache: Optional["MemoraConfig"] = None


def _resolve_base() -> Path:
    env = os.environ.get("MEMORA_BASE_DIR")
    if env:
        return Path(env).resolve()
    workspace = _PACKAGE_DIR.parent
    return (workspace / "memory").resolve()


class MemoraConfig(BaseSettings):
    model_config = {"env_prefix": "MEMORA_"}

    base_dir: Path = Field(default_factory=_resolve_base)
    raw_dir: Optional[Path] = None
    daily_dir: Optional[Path] = None
    long_term_dir: Optional[Path] = None
    vector_db_path: Optional[Path] = None

    embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"
    embedding_dimension: int = 768

    importance_threshold: float = 0.6
    digest_interval_days: int = 7
    digest_batch_size: int = 50

    use_zfs_snapshot: bool = False

    def model_post_init(self, __context):
        base = self.base_dir
        if self.raw_dir is None:
            self.raw_dir = base / "raw"
        if self.daily_dir is None:
            self.daily_dir = base / "daily"
        if self.long_term_dir is None:
            self.long_term_dir = base / "long_term"
        if self.vector_db_path is None:
            self.vector_db_path = base / "vector_db"

    @field_validator("importance_threshold")
    @classmethod
    def validate_importance(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("importance_threshold must be between 0.0 and 1.0")
        return v

    @field_validator("digest_interval_days")
    @classmethod
    def validate_days(cls, v):
        if v < 1:
            raise ValueError("digest_interval_days must be at least 1")
        return v

    def ensure_dirs(self):
        for d in [self.base_dir, self.raw_dir, self.daily_dir,
                  self.long_term_dir, self.vector_db_path]:
            if d is not None:
                d.mkdir(parents=True, exist_ok=True)


def load_config() -> MemoraConfig:
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    config_path = _PACKAGE_DIR / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f) or {}
            cfg = MemoraConfig(**yaml_config)
    else:
        cfg = MemoraConfig()
    cfg.ensure_dirs()
    _config_cache = cfg
    return cfg
