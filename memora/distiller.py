"""
LoRA 蒸馏器 — 从长期记忆生成训练数据集。
实际 LoRA 训练需对接 Axolotl / Unsloth。
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from .config import load_config

logger = logging.getLogger(__name__)
config = load_config()


class LoRADistiller:
    def __init__(self):
        self.output_dir = config.long_term_dir / "lora_training"
        self.base_model = "opus-4.6-base"
        self.target_model = "memora-personal-v1"

    def prepare_dataset(self) -> Path:
        """从长期记忆中提取训练样本并写入 JSONL"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = self.output_dir / f"dataset_{datetime.now().strftime('%Y%m%d')}.jsonl"

        long_term_dir = config.long_term_dir
        if not long_term_dir.exists():
            logger.warning("长期记忆目录不存在: %s", long_term_dir)
            return dataset_path

        samples = []
        for digest_file in sorted(long_term_dir.glob("digest_*.md")):
            content = digest_file.read_text(encoding="utf-8").strip()
            if content:
                samples.append({
                    "instruction": "基于以下记忆上下文回答问题",
                    "input": f"记忆摘要来源: {digest_file.stem}",
                    "output": content[:2000],
                })

        with open(dataset_path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        logger.info("数据集已生成: %s (%d 条样本)", dataset_path, len(samples))
        return dataset_path

    def start_training(self, epochs: int = 3, learning_rate: float = 2e-4):
        """启动 LoRA 训练（需要外部框架）"""
        dataset = self.prepare_dataset()
        logger.info(
            "LoRA 训练参数: model=%s, epochs=%d, lr=%s, dataset=%s",
            self.target_model, epochs, learning_rate, dataset,
        )
        # TODO: 对接 Axolotl / Unsloth
        logger.info("LoRA 训练流程已就绪，等待外部框架对接")
        return True


distiller = LoRADistiller()
