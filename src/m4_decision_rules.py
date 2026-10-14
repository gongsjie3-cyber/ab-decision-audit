"""
模块 4：上线决策规则对比（训练 / 留出）
=======================================

设计
----
* 训练集：exploratory 子样本（4,873 个实验）—— 只用来估计先验 τ² 与规则参数；
* 测试集：holdout 子样本（4,871 个实验）—— 与训练集**实验 ID 零重叠**，用于评估；
* 评价基准：每个实验的贝叶斯期望遗憾
      Regret(规则) = E[ max_i p_i - p_被选臂 | 数据 ]
  它由后验抽样给出，不依赖任何不可观测的真值；同一实验内所有规则共用同一批
  后验抽样，因此规则之间的比较是**配对**的，蒙特卡洛噪声基本抵消。

被比较的规则
------------
R0  平台的实际选择（档案里 first_place 标记的臂）—— 这是真实的"人在决策"
R1  选观察点击率最高的臂（业界默认，也是平台打分系统的行为）
R2  R1 + 显著性门控：top 与次优差异不显著就不上线
R3  经验贝叶斯收缩后选后验均值最高的臂
R4  最优臂后验概率 >= κ 才上线，否则退回
R5  值信息阈值：把"再测一轮的成本 c"写进目标函数

核心结论（见 results/04_decision_rules.json）
--------------------------------------------
* R3 与 R1 的选择在 99.8% 的实验里完全相同 —— 因为同一实验内各臂的噪声水平
  接近，收缩是近似单调变换，**排序不变**。"用经验贝叶斯修正赢家诅咒"这句话
  对"选谁"几乎没用。
* 真正的问题是"有多确定"：被选中臂的后验最优概率中位数只有 0.62，
  28% 的实验里它不到 0.5 —— 等于在掷硬币，而平台照样上线。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt  # noqa: E402
from exptdecide import decision, posterior  # noqa: E402
from src.common import (PALETTE, RESULTS, SEED, Timer,  # noqa: E402
                        dump_json, dump_table, load_arms, load_json, savefig)

N_DRAWS = 6000
KAPPA_GRID = np.round(np.arange(0.20, 0.96, 0.05), 2)


def evaluate_split(arms_hold, tau2, kappa_grid=KAPPA_GRID, seed_off=0):
    """在给定先验下评估全部规则，返回逐实验明细。"""
    rows = []
    for i, (tid, sub) in enumerate(arms_hold.groupby("clickability_test_id", sort=False)):
        y = sub.clicks.values.astype(float)
        n = sub.impressions.values.astype(float)
        base_t = float(y.sum() / n.sum())
        if base_t <= 0:
            continue
        post = posterior.eb_test_posterior(sub.z.values, sub.v.values, tau2,
                                           n_draws=N_DRAWS, rng=1000 + i + seed_off)
        P = post["draws_p"]
        mx = P.max(axis=1)

        def reg(j):
            return float((mx - P[:, j]).mean()) if j >= 0 else float((mx - P.mean(axis=1)).mean())

        j1 = decision.rule_argmax_observed(y, n)
        j2 = decision.rule_significance_gated(y, n, alpha=0.05)
        j3 = decision.rule_eb_argmax(post)
        r = {
            "test_id": tid, "k": len(sub), "base_ctr": base_t,
            "N": float(n.sum()), "date": sub.created_at.min(),
            "r0_platform": int(sub.first_place.values.argmax()) if sub.first_place.any() else -1,
            "r1": j1, "r2": j2, "r3": j3,
            "emax": post["emax"], "pooled_post": float(post["p_mean"].mean()),
            "pbest_argmax": float(post["p_best"].max()),
        }
        for name, j in [("r0_platform", r["r0_platform"]), ("r1", j1), ("r2", j2), ("r3", j3)]:
            r[f"regret_{name}"] = reg(j)
            r[f"pbest_{name}"] = float(post["p_best"][j]) if j >= 0 else np.nan
            r[f"chosen_{name}"] = j
        # κ 扫描（R4）
        jbest = int(np.argmax(post["p_best"]))
        for kp in kappa_grid:
            jj = jbest if post["p_best"][jbest] >= kp else -1
            r[f"r4_k{kp:.2f}_regret"] = reg(jj)
            r[f"r4_k{kp:.2f}_reject"] = int(jj < 0)
        rows.append(r)
    return pd.DataFrame(rows)


def paired_boot(diff, n_boot=2000, seed=SEED):
    d = np.asarray(diff, float)
    d = d[np.isfinite(d)]
    rng = np.random.default_rng(seed)
    s = rng.choice(d, size=(n_boot, d.size), replace=True).mean(axis=1)
    return {"mean": float(d.mean()), "ci": [float(np.quantile(s, .025)), float(np.quantile(s, .975))],
            "n": int(d.size)}


def main():
    arms = load_arms()
    pf = load_json("02_effects_power.json")
    tau2_train = pf["tau2"]["train"]["tau2_reml"]
    tau2_all = pf["tau2_used"]
    hold = arms[arms.split == "hold"].copy()
    print(f"训练集 τ² = {tau2_train:.5f}（仅在 explanatory 子样本上估计）")

    with Timer("在留出集上评估全部规则"):
        det = evaluate_split(hold, tau2_train)
    det.to_csv(RESULTS / "04_decisions_per_test.csv.gz", index=False, compression="gzip")
    print(f"[table ] results/04_decisions_per_test.csv.gz ({len(det):,} 个实验)")

    # ---------- 规则总表 ----------
    rules = {"r0_platform": "R0 平台实际选择(first_place)",
             "r1": "R1 选观察CTR最高", "r2": "R2 显著性门控", "r3": "R3 经验贝叶斯收缩"}
    summary = []
    for key, label in rules.items():
        reg = det[f"regret_{key}"].values
        rej = float((det[f"chosen_{key}"] < 0).mean())
        pb = det[f"pbest_{key}"].values
        summary.append({
            "rule": label, "key": key,
            "mean_regret": float(np.nanmean(reg)),
            "mean_regret_pp": float(np.nanmean(reg) * 100),
            "regret_ci": paired_boot(reg)["ci"],
            "median_regret": float(np.nanmedian(reg)),
            "reject_rate": rej,
            "median_pbest": float(np.nanmedian(pb)),
            "share_pbest_lt_50": float(np.nanmean(pb < 0.5)),
            "share_pbest_lt_80": float(np.nanmean(pb < 0.8)),
        })
    summary = pd.DataFrame(summary)
    # 归一化遗憾：占"实验所能带来的最大收益"的比例
    max_gain = float((det.emax - det.pooled_post).mean())
    summary["regret_as_pct_of_max_gain"] = summary.mean_regret / max_gain
    summary["capture_rate"] = 1 - summary.regret_as_pct_of_max_gain
    base_ctr = float(det.base_ctr.mean())
    summary["mean_regret_pct_of_base"] = summary.mean_regret / base_ctr
    print("\n=== 规则对比（留出集 4,871 个实验）===")
    print(summary[["rule", "mean_regret_pp", "reject_rate", "median_pbest",
                   "share_pbest_lt_50", "capture_rate"]].round(4).to_string(index=False))

    # ---------- R1 vs R3 配对比较 ----------
    same = float((det.chosen_r1 == det.chosen_r3).mean())
    diff_reg = det.regret_r1.values - det.regret_r3.values
    r1r3 = {"share_same_choice": same,
            "paired_regret_diff_r1_minus_r3": paired_boot(diff_reg),
            "n_diff": int((det.chosen_r1 != det.chosen_r3).sum())}
    print(f"\n  R1 与 R3 选择完全相同比例: {same:.4%}（不同 {r1r3['n_diff']} 个实验）")

    # ---------- R2 的"门控"代价与收益 ----------
    gate = det[det.chosen_r2 < 0]
    nogate = det[det.chosen_r2 >= 0]
    r2_info = {
        "reject_rate": float(len(gate) / len(det)),
        "n_rejected": int(len(gate)),
        "regret_of_deployers": float(nogate.regret_r2.mean()),
        "regret_if_rejected_tests_had_deployed": float(gate.regret_r1.mean()),
        "median_pbest_of_deployers": float(nogate.pbest_r2.median()),
        "median_pbest_of_rejected": float(det.loc[det.chosen_r2 < 0, "pbest_r1"].median()),
        "share_of_rejected_with_pbest_lt_50": float((det.loc[det.chosen_r2 < 0, "pbest_r1"] < 0.5).mean()),
    }

    # ---------- κ 前沿 ----------
    front = []
    for kp in KAPPA_GRID:
        reg = det[f"r4_k{kp:.2f}_regret"].values
        rej = det[f"r4_k{kp:.2f}_reject"].values
        deployed = det[rej == 0]
        front.append({
            "kappa": float(kp),
            "mean_regret": float(reg.mean()),
            "reject_rate": float(rej.mean()),
            "median_pbest_of_deployed": float(deployed.pbest_r1.median()) if len(deployed) else np.nan,
            "share_deployed_pbest_lt_50": float((deployed.pbest_r1 < 0.5).mean()) if len(deployed) else np.nan,
        })
    front = pd.DataFrame(front)

    # ---------- τ² 敏感性 ----------
    with Timer("τ² 敏感性分析"):
        sens = []
        rng_s = np.random.default_rng(SEED)
        sens_ids = set(rng_s.choice(hold.clickability_test_id.unique(), 1500, replace=False))
        hold_sens = hold[hold.clickability_test_id.isin(sens_ids)]
        for mult in [0.5, 0.75, 1.0, 1.5, 2.0]:
            d = evaluate_split(hold_sens, tau2_train * mult, seed_off=int(mult * 97))
            sens.append({
                "tau2_multiplier": mult, "tau2": tau2_train * mult,
                "mean_regret_r1": float(d.regret_r1.mean()),
                "mean_regret_r3": float(d.regret_r3.mean()),
                "share_same_r1_r3": float((d.chosen_r1 == d.chosen_r3).mean()),
                "median_pbest": float(np.nanmedian(d.pbest_r1)),
                "share_pbest_lt_50": float(np.nanmean(d.pbest_r1 < 0.5)),
            })
        sens = pd.DataFrame(sens)
        print(sens.round(4).to_string(index=False))

    # ---------- P(best) 分布 ----------
    pb = det.pbest_r1.dropna().values
    pbest_dist = {
        "median": float(np.median(pb)), "q25": float(np.quantile(pb, .25)),
        "q75": float(np.quantile(pb, .75)),
        "share_lt_50": float((pb < 0.5).mean()), "share_lt_80": float((pb < 0.8).mean()),
        "share_gt_95": float((pb > 0.95).mean()),
        "n_coin_flip": int((pb < 0.5).sum()), "n_total": int(len(pb)),
    }

    # ================= 图 9 =================
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.4))
    ax = axes[0]
    xs = np.arange(len(summary))
    ax.bar(xs, summary.mean_regret_pp.values, width=.6,
           color=[PALETTE["muted"], PALETTE["bad"], PALETTE["blue"], PALETTE["good"]])
    for x, v, ci in zip(xs, summary.mean_regret_pp.values, summary.regret_ci.values):
        ax.errorbar([x], [v], yerr=[[v - ci[0] * 100], [ci[1] * 100 - v]], fmt="none",
                    ecolor=PALETTE["base"], capsize=4, lw=1.2)
        ax.text(x, v + 0.004, f"{v:.3f}pp", ha="center", fontsize=9.2)
    short = {"R0 平台实际选择(first_place)": "R0 平台\n实际选择",
             "R1 选观察CTR最高": "R1 选观察\nCTR最高",
             "R2 显著性门控": "R2 显著性\n门控",
             "R3 经验贝叶斯收缩": "R3 EB\n收缩选优"}
    ax.set_xticks(xs)
    ax.set_xticklabels([short.get(s, s) for s in summary.rule], fontsize=8.8)
    ax.set_ylabel("平均贝叶斯期望遗憾（百分点）")
    ax.set_ylim(0, summary.mean_regret_pp.max() * 1.3)
    ax.set_title(f"各规则的平均期望遗憾\n（占基础 CTR 的 {summary.mean_regret_pct_of_base.min():.1%}–"
                 f"{summary.mean_regret_pct_of_base.max():.1%}）")

    ax = axes[1]
    ax.hist(pb, bins=np.linspace(0, 1, 41), color=PALETTE["base"], alpha=.85)
    ax.axvline(0.5, color=PALETTE["bad"], ls="--", lw=1.6, label="0.5 = 掷硬币")
    ax.axvline(np.median(pb), color=PALETTE["accent"], lw=1.8, label=f"中位数 {np.median(pb):.2f}")
    ax.set_xlabel("被上线方案的「最优臂后验概率」P(best)")
    ax.set_ylabel("实验个数")
    ax.legend(fontsize=9)
    ax.set_title(f"{(pb < 0.5).mean():.0%} 的实验里，"
                 f"「赢家」其实分不出来\n（P(best) < 0.5）")

    ax = axes[2]
    ax.plot(front.kappa, front.mean_regret / base_ctr * 100, marker="o", color=PALETTE["purple"],
            lw=1.8, label="平均期望遗憾（占基础 CTR %）")
    ax.set_xlabel("上线门槛 κ：要求 P(best) ≥ κ")
    ax.set_ylabel("平均期望遗憾（% of 基础 CTR）", color=PALETTE["purple"])
    ax2 = ax.twinx()
    ax2.plot(front.kappa, front.share_deployed_pbest_lt_50, marker="s", color=PALETTE["good"],
             lw=1.8, ls="--", label="上线方案仍属掷硬币的比例")
    ax2.set_ylabel("「掷硬币式上线」占比", color=PALETTE["good"])
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8.6, loc="upper center")
    ax.set_title("门槛提高的代价与收益")
    fig.suptitle("图 9 ｜ 决策规则对比：收缩不改变「选谁」，但能告诉你「有多确定」", y=1.03)
    savefig(fig, "fig09_decision_rules", "m4")

    # ================= 图 10：平台实际决策 vs 模型 =================
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3))
    ax = axes[0]
    both = det[(det.chosen_r0_platform >= 0) & (det.chosen_r1 >= 0)]
    agree1 = float((both.chosen_r0_platform == both.chosen_r1).mean())
    agree3 = float((both.chosen_r0_platform == both.chosen_r3).mean())
    ax.bar(["与 R1 一致", "与 R3 一致"], [agree1, agree3],
           color=[PALETTE["muted"], PALETTE["good"]], width=.55)
    for i, v in enumerate([agree1, agree3]):
        ax.text(i, v + .012, f"{v:.1%}", ha="center", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("平台 first_place 与规则选择相同的比例")
    ax.set_title(f"档案里的 {len(both):,} 个有 first_place 标记的实验")
    ax = axes[1]
    regs = [det.regret_r0_platform.mean(), det.regret_r1.mean(), det.regret_r3.mean()]
    ax.bar(["平台实际\n(first_place)", "选观察\nCTR 最高", "经验贝叶斯\n收缩选优"], regs,
           color=[PALETTE["muted"], PALETTE["bad"], PALETTE["good"]], width=.58)
    for i, v in enumerate(regs):
        ax.text(i, v + 0.00002, f"{v*100:.3f}pp", ha="center", fontsize=9.6)
    ax.set_ylabel("平均期望遗憾（百分点）")
    ax.set_ylim(0, max(regs) * 1.28)
    ax.set_title("平台的打分系统 ≈ 选观察 CTR 最高")
    fig.suptitle("图 10 ｜ 平台实际决策的复盘：打分系统与「选最高 CTR」高度一致", y=1.03)
    savefig(fig, "fig10_platform_vs_rules", "m4")

    # ---------- 落盘 ----------
    dump_table(summary.drop(columns=["regret_ci"]), "04_rule_summary.csv")
    dump_table(front, "04_kappa_frontier.csv")
    dump_table(sens, "04_tau2_sensitivity.csv")
    res = {"tau2_train": tau2_train, "tau2_all": tau2_all,
           "n_holdout_tests": int(len(det)), "base_ctr": base_ctr,
           "rule_summary": summary.to_dict("records"),
           "r1_vs_r3": r1r3, "r2_gating": r2_info,
           "kappa_frontier": front.to_dict("records"),
           "tau2_sensitivity": sens.to_dict("records"),
           "pbest_distribution": pbest_dist,
           "max_gain_pp": float(max_gain * 100)}
    dump_json(res, "04_decision_rules.json")

    print("\n=== 决策规则 ===")
    print(f"  R1/R3 选择一致率 {same:.2%}；R1 平均遗憾 {det.regret_r1.mean()*100:.4f}pp"
          f"（占基础 CTR {det.regret_r1.mean()/base_ctr:.2%}）")
    print(f"  被上线方案的 P(best) 中位数 {pbest_dist['median']:.3f}；"
          f"{pbest_dist['share_lt_50']:.1%} 的实验属于「掷硬币」")
    print(f"  实验程序捕获了可获得收益的 {summary.loc[summary.key=='r1','capture_rate'].iloc[0]:.1%}")
    return res


if __name__ == "__main__":
    main()
