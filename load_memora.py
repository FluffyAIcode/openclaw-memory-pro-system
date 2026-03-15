#!/usr/bin/env python3
"""
Memora 加载器 - 一键初始化 Memora 记忆系统
"""

from datetime import datetime

print("🚀 Memora 记忆系统加载器")
print("=" * 50)

try:
    from memora.bridge import bridge
    from memora.config import load_config

    config = load_config()
    config.ensure_dirs()

    for attr in ["raw_dir", "daily_dir", "long_term_dir", "vector_db_path"]:
        d = getattr(config, attr, None)
        if d:
            print(f"✓ 目录已就绪: {d}")

    print(f"\n✅ Memora 桥接器加载成功！")
    print(f"   版本: 1.0.0")
    print(f"   当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   记忆存储位置: {config.base_dir}")

    test_msg = f"系统初始化于 {datetime.now().strftime('%H:%M:%S')}"
    bridge.save_to_both(test_msg, source="loader", importance=0.6)

    print("\n🎉 Memora 系统初始化完成，可以正常使用！")
    print("可用命令示例：")
    print('   python3 -m memora add "你的记忆内容"')

except Exception as e:
    print(f"❌ 加载失败: {type(e).__name__}: {e}")
    print("\n尝试修复建议：")
    print("   1. 确保在 ~/.openclaw/workspace 目录下运行")
    print("   2. 运行: python3 -m pip install -e .")
    import sys
    sys.exit(1)
