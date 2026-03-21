"""
训练数据蒸馏器 — 从长期记忆和 replay buffer 生成微调训练数据集。

Moved from memora/ to chronos/ as part of the architecture refactor:
training data preparation belongs with the training pipeline (Nebius/Axolotl).
"""

import json
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import load_config
from .encoder import EncodedMemory

logger = logging.getLogger(__name__)
config = load_config()


class TrainingDistiller:
    """Prepares JSONL training datasets from memory sources.

    Sources:
      - Digest files (long_term/digest_*.md) → instruction-response pairs
      - Chronos replay buffer (EncodedMemory) → structured memory rows

    Output: chronos_lora_row_v1 JSONL (see docs/NEBIUS_FINETUNE_INTEGRATION_SKETCH.md)
    """

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or (config.base_dir / "training")

    def prepare_from_digests(self, digest_dir: Path = None) -> Path:
        """Generate training rows from digest summary files."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = self.output_dir / f"digest_{datetime.now().strftime('%Y%m%d')}.jsonl"

        if digest_dir is None:
            workspace = Path(__file__).parent.parent
            digest_dir = workspace / "memory" / "long_term"

        if not digest_dir.exists():
            logger.warning("Digest 目录不存在: %s", digest_dir)
            return dataset_path

        samples = []
        for digest_file in sorted(digest_dir.glob("digest_*.md")):
            content = digest_file.read_text(encoding="utf-8").strip()
            if content:
                samples.append({
                    "instruction": "基于以下记忆上下文回答问题",
                    "input": f"记忆摘要来源: {digest_file.stem}",
                    "output": content[:2000],
                    "source": "second_brain_digest",
                    "source_id": digest_file.stem,
                    "importance": 0.7,
                    "timestamp": datetime.now().isoformat(),
                })

        with open(dataset_path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        logger.info("Digest 数据集已生成: %s (%d 条样本)", dataset_path, len(samples))
        return dataset_path

    def prepare_from_buffer(self, buffer_path: Path = None) -> Path:
        """Generate training rows from Chronos replay buffer."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = self.output_dir / f"buffer_{datetime.now().strftime('%Y%m%d')}.jsonl"

        if buffer_path is None:
            buffer_path = config.buffer_path

        if not buffer_path or not buffer_path.exists():
            logger.warning("Buffer 文件不存在: %s", buffer_path)
            return dataset_path

        samples = []
        with open(buffer_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    m = json.loads(line)
                    samples.append(self._encoded_memory_to_row(m))
                except (json.JSONDecodeError, KeyError) as e:
                    logger.debug("Skipped bad buffer line: %s", e)

        with open(dataset_path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        logger.info("Buffer 数据集已生成: %s (%d 条样本)", dataset_path, len(samples))
        return dataset_path

    def prepare_merged(self, digest_dir: Path = None,
                       buffer_path: Path = None) -> Path:
        """Merge digest + buffer into a single training dataset."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        merged_path = self.output_dir / f"merged_{datetime.now().strftime('%Y%m%d')}.jsonl"

        digest_ds = self.prepare_from_digests(digest_dir)
        buffer_ds = self.prepare_from_buffer(buffer_path)

        rows = []
        for ds in [digest_ds, buffer_ds]:
            if ds.exists():
                for line in ds.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        rows.append(line)

        with open(merged_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(r + "\n")

        logger.info("合并数据集已生成: %s (%d 条)", merged_path, len(rows))
        return merged_path

    @staticmethod
    def _encoded_memory_to_row(m: dict) -> dict:
        ts = m.get("timestamp", "")
        raw = m.get("raw_text", "")
        sid = hashlib.sha256(f"{ts}:{raw[:200]}".encode()).hexdigest()[:12]

        structured = []
        for key in ("facts", "preferences", "emotions", "causal_links"):
            for item in m.get(key) or []:
                structured.append(f"[{key}] {item}")

        inp = "\n".join(structured) if structured else raw[:1500]

        return {
            "instruction": "根据结构化记忆，生成一句可注入系统提示的用户偏好摘要。",
            "input": inp,
            "output": raw[:2000],
            "source": "chronos_replay",
            "source_id": sid,
            "importance": m.get("importance", 0.5),
            "timestamp": ts,
        }


distiller = TrainingDistiller()
