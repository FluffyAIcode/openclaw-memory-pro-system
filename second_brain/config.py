import os
import logging
import yaml
from pathlib import Path
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from typing import Optional

logger = logging.getLogger(__name__)

_PACKAGE_DIR = Path(__file__).parent
_config_cache: Optional["SecondBrainConfig"] = None


def _resolve_base() -> Path:
    env = os.environ.get("SB_BASE_DIR")
    if env:
        return Path(env).resolve()
    workspace = _PACKAGE_DIR.parent
    return (workspace / "memory").resolve()


class SecondBrainConfig(BaseSettings):
    model_config = {"env_prefix": "SB_"}

    base_dir: Path = Field(default_factory=_resolve_base)

    tracker_path: Optional[Path] = None
    insights_path: Optional[Path] = None

    vitality_half_life_days: float = 30.0
    dormancy_importance_threshold: float = 0.7
    dormancy_age_days: int = 14
    trend_window_days: int = 3
    trend_top_k: int = 5

    collision_interval_hours: float = 6.0
    collisions_per_round: int = 3
    semantic_bridge_low: float = 0.35
    semantic_bridge_high: float = 0.65
    insight_novelty_threshold: int = 4

    max_search_candidates: int = 20

    @field_validator("vitality_half_life_days")
    @classmethod
    def validate_half_life(cls, v):
        if v <= 0:
            raise ValueError("vitality_half_life_days must be positive")
        return v

    @field_validator("semantic_bridge_low", "semantic_bridge_high")
    @classmethod
    def validate_bridge_range(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("semantic bridge bounds must be between 0.0 and 1.0")
        return v

    def model_post_init(self, __context):
        if self.tracker_path is None:
            self.tracker_path = self.base_dir / "tracker"
        if self.insights_path is None:
            self.insights_path = self.base_dir / "insights"

    def ensure_dirs(self):
        for d in [self.tracker_path, self.insights_path]:
            if d is not None:
                d.mkdir(parents=True, exist_ok=True)


def load_config() -> SecondBrainConfig:
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    config_path = _PACKAGE_DIR / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f) or {}
            cfg = SecondBrainConfig(**yaml_config)
    else:
        cfg = SecondBrainConfig()
    cfg.ensure_dirs()
    _config_cache = cfg
    return cfg
