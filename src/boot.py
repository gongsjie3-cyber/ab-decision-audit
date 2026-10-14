"""
启动引导：把 matplotlib 的缓存目录指到工作区内。

**为什么必须单独一个模块**：matplotlib 在 `import` 的那一刻就读取
`MPLCONFIGDIR`，之后再 `os.environ.setdefault` 已经无效。因此任何用到
matplotlib 的模块，都必须在 `import matplotlib` **之前**先 `import src.boot`。

宿主机的 `~/.matplotlib` 常常不可写，若不重定向，每次运行都会打印
"Matplotlib created a temporary cache directory..." 警告。
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_CACHE = ROOT / ".mplcache"
_CACHE.mkdir(exist_ok=True)
# 用 get 而非 setdefault：setdefault 只在键"不存在"时生效，
# 若外部传了空字符串（MPLCONFIGDIR=）仍会走 matplotlib 的默认路径。
if not os.environ.get("MPLCONFIGDIR"):
    os.environ["MPLCONFIGDIR"] = str(_CACHE)
