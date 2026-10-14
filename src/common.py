"""公共配置：路径、绘图风格（中文字体）、结果落盘工具。"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- 路径
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
REPORTS = ROOT / "reports"
for _p in (DATA_PROC, RESULTS, FIGURES, REPORTS):
    _p.mkdir(parents=True, exist_ok=True)

# matplotlib 缓存放工作区（宿主 ~/.matplotlib 不可写）
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplcache"))
(ROOT / ".mplcache").mkdir(exist_ok=True)

SEED = 20240501

# ---------------------------------------------------------------- 绘图风格
_CJK = ["Arial Unicode MS", "PingFang HK", "Hiragino Sans GB", "Heiti TC", "Songti SC", "DejaVu Sans"]
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 190,
    "savefig.bbox": "tight",
    "font.sans-serif": _CJK,
    "font.family": "sans-serif",
    "axes.unicode_minus": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 12,
    "axes.labelsize": 10.5,
    "legend.frameon": False,
    "figure.facecolor": "white",
})

PALETTE = {
    "bad": "#C0392B",     # 失衡 / 高风险
    "good": "#1E8449",    # 干净 / 推荐
    "base": "#2C3E50",    # 主色
    "accent": "#B7791F",  # 强调
    "muted": "#7F8C8D",   # 次要
    "blue": "#2471A3",
    "purple": "#6C3483",
}
RULE_COLORS = {
    "R1_argmax_observed": "#7F8C8D",
    "R2_significance_gated": "#2471A3",
    "R3_eb_argmax": "#B7791F",
    "R4_pbest_threshold": "#6C3483",
    "R5_voi_threshold": "#1E8449",
}
RULE_LABELS = {
    "R1_argmax_observed": "R1 选观察CTR最高（业界默认）",
    "R2_significance_gated": "R2 显著性门控(p<0.05)",
    "R3_eb_argmax": "R3 经验贝叶斯收缩后选优",
    "R4_pbest_threshold": "R4 最优臂概率阈值 κ=0.8",
    "R5_voi_threshold": "R5 成本最优阈值（值信息）",
}

# 随机化失衡窗口（由本项目的 SRM 审计独立检出，并与官方 2024 披露一致）
BROKEN_START = pd.Timestamp("2013-06-25")
BROKEN_END = pd.Timestamp("2014-01-10")


def savefig(fig, name, subdir=""):
    out = FIGURES / subdir if subdir else FIGURES
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[figure] {path.relative_to(ROOT)}")
    return path


class Timer:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.t0 = time.time()
        print(f"[start] {self.label}", flush=True)
        return self

    def __exit__(self, *a):
        print(f"[done ] {self.label}  ({time.time() - self.t0:.1f}s)", flush=True)


def dump_json(obj, name):
    path = RESULTS / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)
    print(f"[result] {path.relative_to(ROOT)}")
    return path


def load_json(name):
    with open(RESULTS / name, encoding="utf-8") as f:
        return json.load(f)


def dump_table(df, name, **kw):
    path = RESULTS / name
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, **kw)
    print(f"[table ] {path.relative_to(ROOT)}")
    return path


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def load_arms():
    """读取构建好的臂级数据集。"""
    df = pd.read_csv(DATA_PROC / "arms.csv.gz", parse_dates=["created_at"], low_memory=False)
    return df


def load_tests():
    """读取构建好的实验级数据集（统一把实验 ID 列命名为 test_id）。"""
    df = pd.read_csv(DATA_PROC / "tests.csv.gz", parse_dates=["date"], low_memory=False)
    if "clickability_test_id" in df.columns:
        df = df.rename(columns={"clickability_test_id": "test_id"})
    return df
