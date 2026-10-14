"""
模块 1：随机化失衡（SRM）审计
=============================

问题：这批实验"能不能被当作随机实验用"？

做法：
1. 对全部 9,744 个实验做臂级曝光分配的多项分布卡方检验，记录 p 值与 Cramér's V；
2. 用 BH-FDR 控制多重比较；
3. **不预设**失衡时间窗，用数据自己扫描出"窗口内/窗口外失衡率差异最大"的区间；
4. 与官方 2024 年披露的窗口（2013-06-25 ~ 2014-01-10）对照；
5. 检验"缓存分流"特征：失衡是否与实验上线的小时相关。

产出：results/01_audit_*.csv/json 与 figures/m1_*.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt  # noqa: E402
from exptdecide import srm as srm_mod  # noqa: E402
from src.common import (BROKEN_END, BROKEN_START, PALETTE, Timer, dump_json,  # noqa: E402
                        dump_table, load_tests, savefig)

ALPHA = 0.001          # 单实验 SRM 判定阈值
FDR_Q = 0.001          # 多重比较 FDR 水平


def scan_window(dates, flags, min_frac=0.05, grid="W"):
    """不预设窗口，用数据扫描出使"窗口内外失衡率差异"最大的连续区间。

    以二项对数似然为目标：score = logL(窗口内 p1) + logL(窗口外 p0) - logL(全样本 p)。
    这等价于一个两样本比例检验的似然比统计量，扫描所有起止组合取最大。
    """
    df = pd.DataFrame({"d": pd.to_datetime(dates).values, "f": flags.astype(float)})
    df["g"] = df.d.dt.to_period(grid).dt.start_time
    agg = df.groupby("g").agg(n=("f", "size"), k=("f", "sum")).reset_index().sort_values("g")
    n, k = agg.n.values, agg.k.values
    cum_n = np.concatenate([[0], np.cumsum(n)])
    cum_k = np.concatenate([[0], np.cumsum(k)])
    tot_n, tot_k = cum_n[-1], cum_k[-1]
    p_all = tot_k / tot_n

    def ll(kk, nn):
        if nn == 0:
            return 0.0
        p = kk / nn
        if p <= 0 or p >= 1:
            return 0.0
        return kk * np.log(p) + (nn - kk) * np.log(1 - p)

    base = ll(tot_k, tot_n)
    m = len(n)
    best = (-np.inf, None, None)
    nmin = max(int(np.ceil(min_frac * m)), 2)
    for i in range(m - nmin + 1):
        for j in range(i + nmin, m + 1):
            kin, nin = cum_k[j] - cum_k[i], cum_n[j] - cum_n[i]
            kout, nout = tot_k - kin, tot_n - nin
            score = ll(kin, nin) + ll(kout, nout) - base
            if score > best[0]:
                best = (score, i, j)
    score, i, j = best
    inside = agg.iloc[i:j]
    outside_mask = np.ones(m, bool)
    outside_mask[i:j] = False
    out = agg[outside_mask]
    return {
        "lr_stat": float(2 * score),
        "start": str(inside.g.iloc[0].date()),
        "end": str((inside.g.iloc[-1] + pd.Timedelta(days=6)).date()),
        "n_inside": int(inside.n.sum()), "flag_inside": float(inside.k.sum() / inside.n.sum()),
        "n_outside": int(out.n.sum()), "flag_outside": float(out.k.sum() / out.n.sum()),
    }


def main():
    tests = load_tests()
    tests["date"] = pd.to_datetime(tests.date)
    tests["hour"] = tests.date.dt.hour
    tests["flag"] = tests.srm_p < ALPHA
    tests["flag_fdr"] = srm_mod.bh_flag(tests.srm_p.values, q=FDR_Q)
    tests["official_window"] = (tests.date >= BROKEN_START) & (tests.date <= BROKEN_END)

    with Timer("SRM 检验汇总"):
        overall = {
            "n_tests": int(len(tests)),
            "flag_rate_p001": float(tests.flag.mean()),
            "flag_rate_p05": float((tests.srm_p < 0.05).mean()),
            "flag_rate_fdr001": float(tests.flag_fdr.mean()),
            "flag_rate_window_official": float(tests.loc[tests.official_window, "flag"].mean()),
            "flag_rate_outside_official": float(tests.loc[~tests.official_window, "flag"].mean()),
            "cramer_v_median_flagged": float(tests.loc[tests.flag, "cramer_v"].median()),
            "cramer_v_median_clean": float(tests.loc[~tests.flag, "cramer_v"].median()),
            "max_share_p99_flagged": float(tests.loc[tests.flag, "max_share"].quantile(0.99)),
            "max_share_p99_clean": float(tests.loc[~tests.flag, "max_share"].quantile(0.99)),
            "max_dev_pp_median_flagged": float(tests.loc[tests.flag, "max_dev"].median()),
            "max_dev_pp_median_clean": float(tests.loc[~tests.flag, "max_dev"].median()),
            "missing_exp_in_window_share": float(
                tests.loc[tests.official_window, "flag"].mean()),
        }

    with Timer("数据驱动窗口扫描"):
        win = scan_window(tests.date, tests.flag)
        win["official_start"] = str(BROKEN_START.date())
        win["official_end"] = str(BROKEN_END.date())
        # 与官方窗口的重合度（按实验计数）
        det = (tests.date >= pd.Timestamp(win["start"])) & (
            tests.date <= pd.Timestamp(win["end"]) + pd.Timedelta(days=1))
        win["overlap_with_official"] = float((det & tests.official_window).sum() / max(det.sum(), 1))
        win["recall_of_official"] = float((det & tests.official_window).sum() / max(tests.official_window.sum(), 1))

    # ---------- 时间维度 ----------
    tests["quarter"] = tests.date.dt.to_period("Q").astype(str)
    tests["month"] = tests.date.dt.to_period("M").astype(str)
    by_month = tests.groupby("month").agg(
        n_tests=("flag", "size"), flag_rate=("flag", "mean"),
        v_median=("cramer_v", "median"), max_dev_median=("max_dev", "median")).reset_index()
    by_quarter = tests.groupby("quarter").agg(
        n_tests=("flag", "size"), flag_rate=("flag", "mean")).reset_index()

    # ---------- 缓存特征：小时 ----------
    hour_tbl = (tests.groupby(["hour", "official_window"])
                .agg(n=("flag", "size"), flag_rate=("flag", "mean")).reset_index())
    h_in = hour_tbl[hour_tbl.official_window].set_index("hour")
    h_out = hour_tbl[~hour_tbl.official_window].set_index("hour")
    common_hours = sorted(set(h_in.index) & set(h_out.index))
    rate_in = h_in.loc[common_hours, "flag_rate"]
    rate_out = h_out.loc[common_hours, "flag_rate"]

    # 窗口内失衡率的离散度（缓存假设预测：随小时显著波动）
    from scipy import stats as st
    def _cont(h):
        n = h.loc[common_hours, "n"].values.astype(float)
        p = h.loc[common_hours, "flag_rate"].values
        return np.column_stack([n * p, n * (1 - p)])

    chi_in = st.chi2_contingency(_cont(h_in))
    chi_out = st.chi2_contingency(_cont(h_out))
    hour_stats = {
        "flag_rate_in_window_sd": float(rate_in.std()),
        "flag_rate_out_window_sd": float(rate_out.std()),
        "hour_chi2_in_window": float(chi_in.statistic),
        "hour_chi2_p_in_window": float(chi_in.pvalue),
        "hour_chi2_out_window": float(chi_out.statistic),
        "hour_chi2_p_out_window": float(chi_out.pvalue),
        "corr_within_vs_outside": float(np.corrcoef(rate_in.values, rate_out.values)[0, 1]),
        "argmax_hour_in_window": int(rate_in.idxmax()),
        "max_hour_rate_in_window": float(rate_in.max()),
        "min_hour_rate_in_window": float(rate_in.min()),
    }

    # ---------- 图 1：时间序列 ----------
    wk = tests.copy()
    wk["week"] = wk.date.dt.to_period("W").dt.start_time
    wagg = wk.groupby("week").agg(n=("flag", "size"), flag_rate=("flag", "mean"),
                                  v=("cramer_v", "median")).reset_index()
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1.15]})
    ax = axes[0]
    ax.axvspan(pd.Timestamp(win["start"]), pd.Timestamp(win["end"]), color="#F5B7B1", alpha=.45,
               label=f"数据自行检出的失衡区间\n{win['start']} ~ {win['end']}")
    ax.plot(wagg.week, wagg.flag_rate, color=PALETTE["base"], lw=1.6, marker="o", ms=2.6,
            label="每周 SRM 标记率 (χ² p<0.001)")
    ax.axhline(overall["flag_rate_outside_official"], color=PALETTE["good"], ls="--", lw=1.2,
               label=f"窗口外基准 {overall['flag_rate_outside_official']:.1%}")
    ax.axvline(BROKEN_START, color=PALETTE["bad"], ls=":", lw=1.6)
    ax.axvline(BROKEN_END, color=PALETTE["bad"], ls=":", lw=1.6)
    ax.annotate("官方 2024 年披露的窗口：\n2013-06-25 ~ 2014-01-10",
                xy=(BROKEN_START, .99), xytext=(pd.Timestamp("2013-01-12"), .80),
                fontsize=8.5, color=PALETTE["bad"],
                arrowprops=dict(arrowstyle="->", color=PALETTE["bad"], lw=1))
    ax.set_ylabel("分流失衡实验占比")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("图 1 ｜ 9,744 个真实 A/B 实验的随机化失衡率：数据自行检出 2013-06 ~ 2014-01 的异常期")
    ax.legend(loc="upper right", fontsize=8.2, ncol=1)
    ax2 = axes[1]
    ax2.plot(wagg.week, wagg.v, color=PALETTE["accent"], lw=1.4, marker="o", ms=2.4)
    ax2.set_ylabel("Cramér's V 中位数")
    ax2.set_xlabel("实验上线时间")
    ax2.axvspan(pd.Timestamp(win["start"]), pd.Timestamp(win["end"]), color="#F5B7B1", alpha=.45)
    savefig(fig, "fig01_srm_timeseries", "m1")

    # ---------- 图 2：小时特征 ----------
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    ax.plot(common_hours, rate_in.values, marker="o", ms=4, color=PALETTE["bad"],
            label=f"官方披露窗口内（n={int(h_in.loc[common_hours,'n'].sum()):,} 个实验）")
    ax.plot(common_hours, rate_out.values, marker="s", ms=3.6, color=PALETTE["good"],
            label=f"窗口外（n={int(h_out.loc[common_hours,'n'].sum()):,} 个实验）")
    ax.set_xlabel("实验上线的小时（0–23）")
    ax.set_ylabel("SRM 标记率")
    ax.set_title("图 2 ｜ 失衡率随上线小时剧烈波动，符合「缓存按整点批量放量」的分流故障特征")
    ax.legend(fontsize=9)
    savefig(fig, "fig02_srm_by_hour", "m1")

    # ---------- 图 3：效应量分布 ----------
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.1))
    bins = np.linspace(0, 0.35, 45)
    axes[0].hist(tests.loc[tests.flag, "cramer_v"], bins=bins, color=PALETTE["bad"], alpha=.75,
                 label=f"被标记 (n={int(tests.flag.sum()):,})", density=True)
    axes[0].hist(tests.loc[~tests.flag, "cramer_v"], bins=bins, color=PALETTE["good"], alpha=.6,
                 label=f"通过 (n={int((~tests.flag).sum()):,})", density=True)
    axes[0].set_xlabel("Cramér's V（曝光分配偏离度）")
    axes[0].set_ylabel("密度")
    axes[0].legend(fontsize=9)
    axes[0].set_title("实验臂曝光分配偏离度")
    mx = tests.max_share
    axes[1].hist(mx[~tests.flag].clip(0, 1), bins=np.linspace(0.15, 0.8, 45), color=PALETTE["good"],
                 alpha=.6, density=True, label="通过")
    axes[1].hist(mx[tests.flag].clip(0, 1), bins=np.linspace(0.15, 0.8, 45), color=PALETTE["bad"],
                 alpha=.75, density=True, label="被标记")
    axes[1].axvline(1 / tests.k.mean(), color=PALETTE["base"], ls="--", lw=1.4,
                    label=f"设计值 1/k ≈ {1/tests.k.mean():.3f}")
    axes[1].set_xlabel("单个实验臂的最大曝光占比")
    axes[1].set_ylabel("密度")
    axes[1].legend(fontsize=9)
    axes[1].set_title("最大臂流量占比（理论值 1/k）")
    fig.suptitle("图 3 ｜ 被标记实验的失衡是「实质性」的：最大臂拿到了远超设计比例的流量", y=1.03)
    savefig(fig, "fig03_srm_effect_size", "m1")

    # ---------- 落盘 ----------
    dump_table(by_month, "01_audit_by_month.csv")
    dump_table(by_quarter, "01_audit_by_quarter.csv")
    dump_table(hour_tbl, "01_audit_by_hour.csv")
    dump_table(tests[["test_id", "date", "split", "k", "N", "srm_p", "cramer_v",
                      "max_share", "max_dev", "flag", "flag_fdr", "official_window"]],
               "01_srm_per_test.csv")
    res = {"overall": overall, "detected_window": win, "hour_signature": hour_stats,
           "quarterly": by_quarter.to_dict("records")}
    dump_json(res, "01_audit_summary.json")

    print("\n=== 随机化失衡审计 ===")
    print(f"  被标记实验（χ² p<0.001）: {overall['flag_rate_p001']:.1%}"
          f"  ({int(tests.flag.sum()):,} / {len(tests):,})")
    print(f"  官方窗口内 {overall['flag_rate_window_official']:.1%}  vs  窗口外 "
          f"{overall['flag_rate_outside_official']:.1%}")
    print(f"  数据自行检出窗口: {win['start']} ~ {win['end']}"
          f"  (窗口内 {win['flag_inside']:.1%} vs 外 {win['flag_outside']:.1%}, LR={win['lr_stat']:.0f})")
    print(f"  与官方窗口重合: 精确率 {win['overlap_with_official']:.1%}, 召回率 {win['recall_of_official']:.1%}")
    print(f"  窗口内失衡率随小时波动 sd={hour_stats['flag_rate_in_window_sd']:.3f} "
          f"vs 窗口外 sd={hour_stats['flag_rate_out_window_sd']:.3f}")
    return res


if __name__ == "__main__":
    main()
