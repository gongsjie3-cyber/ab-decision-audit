"""
数据获取脚本
============

从 OSF（Open Science Framework）下载 Upworthy 研究档案的两个**公开**子样本。
两者实验 ID 零重叠，均为按周分层的随机样本，本项目分别用作训练集与留出集。

    档案主页 : https://osf.io/jd64p/
    数据论文 : Matias et al. (2021) Scientific Data 8:155
    许可证   : CC BY 4.0

档案的三个子集（exploratory + holdout + confirmatory）合计正好是官方宣称的
32,487 个实验，且**均已公开**，无需申请即可下载。

分析纪律：模块 1–5 只在 exploratory + holdout 上开发（先完成代码与结论），
confirmatory 子集是在**全部结论写完之后**才下载的，模块 6 只用于纯样本外确认，
不重新估计任何参数。时间戳证据见 reports/report.md 第 10 节。

    python scripts/fetch_data.py            # 下载并校验
    python scripts/fetch_data.py --force    # 强制重新下载
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

FILES = [
    {
        "name": "upworthy-exploratory-packages.csv",
        "url": "https://osf.io/download/3vqmp/",
        "sha256": "8368313b060f4015a0c6fb34e6d788163cee29554144aba9390b14922eb9d8ce",
        "role": "训练集：4,873 个实验 / 22,666 个臂（用于估计先验与规则参数）",
    },
    {
        "name": "upworthy-holdout-packages.csv",
        "url": "https://osf.io/download/ynf3k/",
        "sha256": "93832660b831aca16bbe314c60dba4a6815b0a5f9cdded1217671cfbfa5186ea",
        "role": "留出集：4,871 个实验 / 22,600 个臂（仅用于评估）",
    },
    {
        "name": "upworthy-confirmatory-packages.csv",
        "url": "https://osf.io/download/vy8mj/",
        "sha256": "b2a88288c88f2b67d40413c9bdbebe79a1c0b5d9a9354180549f59f3bfff685f",
        "role": "确认集：22,743 个实验 / 105,551 个臂（模块 6 的纯样本外检验集）",
    },
]


def sha256_of(path: Path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="即使已存在也重新下载")
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    ok = True
    for spec in FILES:
        path = RAW / spec["name"]
        print(f"\n=== {spec['name']} ===\n  {spec['role']}")
        if path.exists() and not args.force:
            digest = sha256_of(path)
            if digest == spec["sha256"]:
                print(f"  已存在且校验通过（sha256 {digest[:16]}…），跳过")
                continue
            print("  已存在但校验不符，重新下载")
        print(f"  下载 {spec['url']} …")
        try:
            urllib.request.urlretrieve(spec["url"], path)
        except Exception as e:  # noqa: BLE001
            print(f"  下载失败：{e}")
            ok = False
            continue
        digest = sha256_of(path)
        if digest == spec["sha256"]:
            print(f"  ✔ 校验通过：{path}  ({path.stat().st_size / 1e6:.1f} MB)")
        else:
            print(f"  ✘ 校验失败：期望 {spec['sha256'][:16]}…，实际 {digest[:16]}…")
            ok = False
    print()
    if not ok:
        print("存在未通过校验的文件，请检查网络后重试。")
        sys.exit(1)
    print("数据就绪。下一步：python -m src.run_all")


if __name__ == "__main__":
    main()
