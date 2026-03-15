import logging
import subprocess
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class ZFSIntegration:
    def __init__(self):
        self.enabled = False
        self.dataset = None

    def check_zfs_available(self) -> bool:
        try:
            result = subprocess.run(
                ["zfs", "list"],
                capture_output=True, text=True, timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as e:
            logger.debug("ZFS 不可用: %s", e)
            return False

    def create_snapshot(self, name: str = None):
        if not self.enabled:
            logger.warning("ZFS 支持未启用")
            return False

        snapshot_name = name or f"memora@{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if not self.dataset:
            logger.error("ZFS dataset 未配置")
            return False

        try:
            result = subprocess.run(
                ["zfs", "snapshot", f"{self.dataset}@{snapshot_name}"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                logger.info("创建 ZFS 快照: %s", snapshot_name)
                return True
            logger.error("ZFS 快照失败: %s", result.stderr.strip())
            return False
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as e:
            logger.error("ZFS 快照异常: %s", e)
            return False


zfs = ZFSIntegration()
