"""
AlphaQCM 内部历史遗留的 alphagen 实现（不推荐用于当前训练脚本）。

说明：
- 本文件的目的不是让这套实现“变成主版本”，而是为了阻断 namespace package 的混合导入问题。
- 训练入口 `train_alphagen_crypto.py` 会通过 sys.path 优先级强制使用仓库根目录的 `alphagen/` 子模块。
"""

from __future__ import annotations

