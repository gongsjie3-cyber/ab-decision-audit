"""
模块 0：重建实验数据集
======================

输入：Upworthy 研究档案的两个**互不相交**的公开子集（均为按周分层的随机样本）
      * exploratory：4,873 个实验 / 22,666 个实验臂
      * holdout    ：4,871 个实验 / 22,600 个实验臂
      （档案另有 22,743 个实验的 confirmatory 子集，需通过同行评审申请才能获取，
        本项目未使用，以保证完全可复现）

输出：
      data/processed/arms.csv.gz   臂级明细（曝光、点击、logit 变换、方差、素材指纹）
      data/processed/tests.csv.gz  实验级汇总（臂数、总量、SRM 检验、平台打分、平台选择）

设计说明
--------
* 只保留 k>=2 且总曝光>0 的实验；
* 素材指纹 package_key = slug + eyecatcher_id，用于识别"同一素材被重复投放"，
  这是后文做**样本外复现检验**的基础；
* SRM 检验在实验级完成，同时记录效应量 Cramér's V，避免只看 p 值；
* 保留原档案的 significance / first_place / winner 三列，用于逆向工程平台自身的
  决策规则（作为 R1 规则的现实依据，而不是我们自己假想的基线）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exptdecide import srm as srm_mod          # noqa: E402
from src.common import BROKEN_END, BROKEN_START, DATA_PROC, DATA_RAW, Timer, dump_json  # noqa: E402

RAW = {
    "expl": DATA_RAW / "upworthy-exploratory-packages.csv",
    "hold": DATA_RAW / "upworthy-holdout-packages.csv",
}


def load_raw():
    frames = []
    for split, path in RAW.items():
        if not path.exists():
            raise FileNotFoundError(f"缺少原始数据 {path}，请先运行 scripts/fetch_data.py")
        df = pd.read_csv(path)
        df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
        df["split"] = split
        df["source_file"] = path.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build():
    with Timer("读取原始档案"):
        d = load_raw()
    print(f"  原始行数 {len(d):,}，实验数 {d.clickability_test_id.nunique():,}")

    # ---- 基本清洗
    d = d[d.impressions > 0].copy()
    d["created_at"] = pd.to_datetime(d.created_at, format="mixed")
    d["ctr"] = d.clicks / d.impressions
    d["package_key"] = d.slug.fillna("NA") + "|" + d.eyecatcher_id.fillna("NA")
    d["in_broken_window"] = (d.created_at >= BROKEN_START) & (d.created_at <= BROKEN_END)

    # ---- 只保留 k >= 2 的实验
    k_of_test = d.groupby("clickability_test_id")["impressions"].transform("size")
    d = d[k_of_test >= 2].copy()
    k_of_test = d.groupby("clickability_test_id")["impressions"].transform("size")
    d["k"] = k_of_test

    # ---- 实验内的臂顺序（按归档行序稳定）
    d = d.sort_values(["split", "clickability_test_id", "created_at"]).reset_index(drop=True)
    d["arm"] = d.groupby("clickability_test_id").cumcount()

    # ---- logit 变换与方差（供经验贝叶斯使用）
    from exptdecide.effect import arm_logit
    z, v = arm_logit(d.clicks.values, d.impressions.values)
    d["z"], d["v"] = z, v

    # ---- 实验级汇总
    g = d.groupby("clickability_test_id")
    tests = pd.DataFrame({
        "k": g.size(),
        "N": g["impressions"].sum(),
        "clicks": g["clicks"].sum(),
        "date": g["created_at"].min(),
        "test_week": g["test_week"].first(),
        "split": g["split"].first(),
        "n_distinct_packages": g["package_key"].nunique(),
        "n_dup_arms": g.size() - g["package_key"].nunique(),
        "in_broken_window": g["in_broken_window"].first(),
    })
    tests["pooled_ctr"] = tests.clicks / tests.N
    tests["max_pkg_share"] = g.apply(lambda x: x.impressions.max() / x.impressions.sum(),
                                     include_groups=False)
    tests["ctr_obs_spread"] = g["ctr"].max() - g["ctr"].min()
    tests["ctr_obs_spread_rel"] = tests.ctr_obs_spread / tests.pooled_ctr

    # ---- SRM 检验（实验级）
    rows = []
    for tid, sub in d.groupby("clickability_test_id", sort=False):
        r = srm_mod.chi2_srm(sub.impressions.values)
        r["test_id"] = tid
        rows.append(r)
    srm_df = pd.DataFrame(rows).set_index("test_id")
    tests = tests.join(srm_df[["chi2", "dof", "p", "cramer_v", "max_share", "max_dev", "ratio"]])
    tests = tests.rename(columns={"p": "srm_p"})

    # ---- 平台自身的排序/选择（逆向工程用）
    d["is_obs_best"] = d.ctr == g["ctr"].transform("max")
    d["sig_rank_in_test"] = d.groupby("clickability_test_id")["significance"].rank(
        ascending=False, method="min")
    tests = tests.join(d[d.first_place].groupby("clickability_test_id")["arm"].first().rename("first_place_arm"))
    tests = tests.join(d[d.winner].groupby("clickability_test_id")["arm"].first().rename("winner_arm"))
    d["is_first_place"] = d.first_place
    tests["obs_best_arm"] = d[d.is_obs_best].groupby("clickability_test_id")["arm"].first()

    # ---- 平台 first_place 与观察最优臂吻合度（作为 R1 基线的现实依据）
    both = tests.dropna(subset=["first_place_arm", "obs_best_arm"])
    agree = float((both.first_place_arm == both.obs_best_arm).mean())

    # ---- 组内 logit 与 significance 的秩相关（确认 significance 的真实含义）
    from scipy import stats as st
    rr = d.groupby("clickability_test_id").apply(
        lambda x: x.significance.corr(x.ctr, method="spearman")
        if x.significance.nunique() > 1 and x.ctr.nunique() > 1 else np.nan,
        include_groups=False)
    sig_rho = float(np.nanmean(rr.values))

    d.to_csv(DATA_PROC / "arms.csv.gz", index=False, compression="gzip")
    tests.reset_index().to_csv(DATA_PROC / "tests.csv.gz", index=False, compression="gzip")

    summary = {
        "n_tests": int(len(tests)),
        "n_arms": int(len(d)),
        "impressions_total": int(d.impressions.sum()),
        "clicks_total": int(d.clicks.sum()),
        "overall_ctr": float(d.clicks.sum() / d.impressions.sum()),
        "tests_by_split": tests.split.value_counts().to_dict(),
        "arms_by_split": d.split.value_counts().to_dict(),
        "mean_arms_per_test": float(tests.k.mean()),
        "median_arm_impressions": float(d.impressions.median()),
        "date_min": str(d.created_at.min()),
        "date_max": str(d.created_at.max()),
        "srm_flagged_p001": int((tests.srm_p < 0.001).sum()),
        "srm_flagged_p001_share": float((tests.srm_p < 0.001).mean()),
        "srm_flagged_p05_share": float((tests.srm_p < 0.05).mean()),
        "median_max_dev_pp": float(tests.max_dev.median()),
        "p99_max_dev_pp": float(tests.max_dev.quantile(0.99)),
        "max_arm_share_mean": float(tests.max_share.mean()),
        "tests_with_dup_arms": int((tests.n_dup_arms > 0).sum()),
        "tests_all_arms_identical": int((tests.n_distinct_packages == 1).sum()),
        "platform_first_place_eq_obs_best": agree,
        "platform_significance_spearman_with_ctr": sig_rho,
        "holdout_tests_with_winner_flag": int(tests.winner_arm.notna().sum()),
    }
    dump_json(summary, "00_dataset_summary.json")

    print("\n=== 数据集摘要 ===")
    for kk, vv in summary.items():
        print(f"  {kk:42s} {vv}")
    return tests, d


if __name__ == "__main__":
    build()
