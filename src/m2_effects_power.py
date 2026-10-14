"""
模块 2：效应量先验、功效分析与噪声基准
=======================================

三个问题：
Q1  "换标题"这件事到底有多大真实价值？—— 估计组内先验方差 tau^2，并换算成
    真实点击率的分布。
Q2  实验中看到的臂间差距，有多少是真实的、多少是抽样噪声？—— 与"纯噪声基准"
    对照，并做方差分解。
Q3  这批实验在设计上有没有能力得出结论？—— MDE / 所需样本量 / 实际功效。
Q4  模型可信吗？—— 在留出集上做"留一臂"后验预测覆盖率检验。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import boot  # noqa: E402  必须在 import matplotlib 之前导入
import matplotlib.pyplot as plt  # noqa: E402
from exptdecide import effect, posterior, power  # noqa: E402
from exptdecide.effect import expit  # noqa: E402
from src.common import PALETTE, SEED, Timer, dump_json, dump_table, load_arms, savefig  # noqa: E402

RNG = np.random.default_rng(SEED)


def fit_tau2(arms, tag=""):
    z, v, g = arms.z.values, arms.v.values, arms.clickability_test_id.values
    mom = effect.tau2_moment(z, v, g)
    reml = effect.tau2_reml(z, v, g)
    out = {"tag": tag, "tau2_mom": mom["tau2"], "tau2_reml": reml["tau2"],
           "tau2_reml_se": reml["se"], "Q": mom["Q"], "dof": mom["dof"], "I2": mom["I2"]}
    print(f"  tau^2 [{tag}] MoM={mom['tau2']:.5f}  REML={reml['tau2']:.5f} (±{reml['se']:.5f})  I²={mom['I2']:.3f}")
    return out, reml


def implied_ctr_distribution(tau2, base_ctr, n=400000, rng=None):
    rng = np.random.default_rng(SEED if rng is None else rng)
    mu = np.log(base_ctr / (1 - base_ctr))
    eps = rng.standard_normal(n) * np.sqrt(tau2)
    p = expit(mu + eps)
    return p


def main():
    arms = load_arms()
    base_ctr = float(arms.clicks.sum() / arms.impressions.sum())
    med_n = float(arms.impressions.median())

    # ---------------- Q1: tau^2 ----------------
    with Timer("估计 tau^2（全样本 / 训练集 / 留出集）"):
        all_fit, reml_all = fit_tau2(arms, "all")
        expl_fit, reml_expl = fit_tau2(arms[arms.split == "expl"], "train(expl)")
        hold_fit, reml_hold = fit_tau2(arms[arms.split == "hold"], "holdout(hold)")
    tau2 = reml_all["tau2"]
    p_true = implied_ctr_distribution(tau2, base_ctr)
    prior_q = {q: float(np.quantile(p_true, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}
    prior_stats = {"mean": float(p_true.mean()), "sd": float(p_true.std()),
                   "cv_pct": float(p_true.std() / p_true.mean() * 100), **{f"q{int(q*100)}": v for q, v in prior_q.items()}}

    # ---------------- Q2: 方差分解 + 纯噪声基准 ----------------
    with Timer("方差分解与纯噪声基准"):
        # 精确恒等式：SS_within = tau^2 * dof + sum_i (1-1/k_i) v_i
        #   "真实文案差异" 与 "抽样噪声" 的占比可直接读出（见 effect.within_test_ss）。
        #   用无偏的方法矩 tau^2 时该恒等式两边严格相等，故分解用它。
        vd = effect.variance_decomposition(arms.z.values, arms.v.values,
                                           arms.clickability_test_id.values,
                                           tau2=None)
        vd_reml = effect.variance_decomposition(arms.z.values, arms.v.values,
                                                arms.clickability_test_id.values, tau2=tau2)
        decomp = {
            "ss_within": vd["ss_within"], "signal_part": vd["signal_part"],
            "noise_part": vd["noise_part"], "dof": vd["dof"],
            "tau2_estimates": {"contrast_moment_unbiased": vd["tau2_implied"],
                               "dl_moment": all_fit["tau2_mom"],
                               "reml_used_in_model": tau2},
            "signal_share": vd["signal_share"], "noise_share": vd["noise_share"],
            "signal_share_using_reml": vd_reml["signal_share"],
            # CTR 尺度的直观表述
            "ctr_sd_true_pp": float(p_true.std() * 100),
            "ctr_sd_noise_pp": float(np.sqrt(np.mean(arms.ctr.values * (1 - arms.ctr.values)
                                                  / arms.impressions.values)) * 100),
        }
        decomp["ctr_signal_share"] = decomp["ctr_sd_true_pp"] ** 2 / (
            decomp["ctr_sd_true_pp"] ** 2 + decomp["ctr_sd_noise_pp"] ** 2)
        spread_obs = arms.groupby("clickability_test_id").apply(
            lambda x: x.ctr.max() - x.ctr.min(), include_groups=False)
        mean_k = int(round(arms.groupby("clickability_test_id").size().mean()))
        nz = power.noise_only_spread(base_ctr, int(round(med_n)), mean_k)
        noise_bench = {"observed_mean_spread": float(spread_obs.mean()),
                       "observed_median_spread": float(spread_obs.median()),
                       "pure_noise_mean_spread": nz["mean"], "pure_noise_median_spread": nz["median"],
                       "ratio_mean": float(spread_obs.mean() / nz["mean"]),
                       "mean_k": mean_k}

    # ---------------- Q3: 功效 ----------------
    with Timer("功效分析"):
        deltas = [0.10, 0.20, 0.30, 0.50]
        power_tbl = []
        for d in deltas:
            need = power.required_n_per_arm(base_ctr, d)
            power_tbl.append({
                "delta_rel": d,
                "required_n_per_arm": need,
                "actual_median_n": med_n,
                "shortfall_x": need / med_n,
                "power_at_actual_n": power.power_two_prop(med_n, base_ctr, d),
            })
        power_tbl = pd.DataFrame(power_tbl)
        # 每个实验按自己的中位臂样本量算功效（更贴近实际）
        per_test_n = arms.groupby("clickability_test_id")["impressions"].median()
        per_test_p = arms.groupby("clickability_test_id").apply(
            lambda x: x.clicks.sum() / x.impressions.sum(), include_groups=False)
        n_v, p_v = per_test_n.values.astype(float), per_test_p.values.astype(float)
        valid = (n_v > 0) & (p_v > 0) & (p_v < 1)
        n_v, p_v = n_v[valid], p_v[valid]
        pw20 = np.asarray(power.power_two_prop(n_v, p_v, 0.20), dtype=float)
        pw10 = np.asarray(power.power_two_prop(n_v, p_v, 0.10), dtype=float)
        mde_v = np.asarray(power.mde_relative(n_v, p_v), dtype=float)
        pw = {
            "n_tests_used": int(valid.sum()),
            "share_power_ge_80pct_for_20pct_lift": float(np.mean(pw20 >= 0.8)),
            "median_power_for_20pct_lift": float(np.median(pw20)),
            "share_power_ge_80pct_for_10pct_lift": float(np.mean(pw10 >= 0.8)),
            "median_power_for_10pct_lift": float(np.median(pw10)),
            "median_mde_rel": float(np.median(mde_v)),
            "p90_mde_rel": float(np.quantile(mde_v, 0.9)),
        }
        print(f"  20% 提升下，功效>=80% 的实验占比: {pw['share_power_ge_80pct_for_20pct_lift']:.2%}"
              f"（中位功效 {pw['median_power_for_20pct_lift']:.2f}）")
        print(f"  中位实验的 MDE：{pw['median_mde_rel']:.1%} 相对提升")

        # 选择准确率（模拟真值）：k 从 2 到 10
        sel = []
        for kk in [2, 3, 4, 5, 6, 8, 10]:
            r = power.selection_accuracy(tau2, base_ctr, kk, int(round(med_n)), n_sim=20000)
            sel.append({"k": kk, **r})
        sel_tbl = pd.DataFrame(sel)

    # ---------------- Q4: 留一臂后验预测校准（只用留出集） ----------------
    with Timer("留一臂后验预测覆盖率检验（holdout）"):
        hold = arms[arms.split == "hold"]
        rng_sel = np.random.default_rng(SEED)
        test_ids = hold.clickability_test_id.unique()
        sampled = rng_sel.choice(test_ids, size=min(3000, len(test_ids)), replace=False)
        zscores = []
        for tid in sampled:
            sub = hold[hold.clickability_test_id == tid]
            zz, vv = sub.z.values, sub.v.values
            for i in range(len(sub)):
                mask = np.ones(len(sub), bool)
                mask[i] = False
                post = posterior.eb_test_posterior(zz[mask], vv[mask], tau2, n_draws=100)
                # 新臂 theta_new = mu + eps_new；其后验预测分布为
                #   z_new ~ N(E[mu], Var(mu) + tau^2 + v_new)
                pred_sd = np.sqrt(post["var_mu"] + tau2 + vv[i])
                zscores.append((zz[i] - post["mean_mu"]) / pred_sd)
        zscores = np.asarray(zscores)
    cover = {
        "n_holdout_arms": int(zscores.size),
        "coverage_95": float(np.mean(np.abs(zscores) <= 1.96)),
        "coverage_90": float(np.mean(np.abs(zscores) <= 1.645)),
        "coverage_80": float(np.mean(np.abs(zscores) <= 1.282)),
        "sd_of_zscores": float(zscores.std()),
        "mean_abs_z": float(np.abs(zscores).mean()),
    }
    print(f"  95% 预测区间实测覆盖率 {cover['coverage_95']:.3f}（名义 0.95）; z 的标准差 {cover['sd_of_zscores']:.3f}")

    # ---------------- 图 4：tau^2 似然 + 隐含真实 CTR 分布 ----------------
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2))
    grid = reml_all["grid"]
    ll = reml_all["loglik"]
    axes[0].plot(grid, ll - ll.max(), color=PALETTE["base"], lw=1.8)
    axes[0].axvline(tau2, color=PALETTE["bad"], ls="--", lw=1.4,
                    label=f"REML 估计 τ²={tau2:.4f}")
    axes[0].axhline(-1.92, color=PALETTE["muted"], ls=":", lw=1.2, label="近似 95% 似然区间")
    axes[0].set_xlim(0, max(0.15, tau2 * 3))
    axes[0].set_ylim(-60, 2)
    axes[0].set_xlabel("τ²（实验内 logit 尺度的真实方差）")
    axes[0].set_ylabel("对数似然（相对最大值）")
    axes[0].legend(fontsize=9)
    axes[0].set_title("τ² 的限制极大似然曲线")
    axes[1].hist(p_true, bins=160, color=PALETTE["base"], alpha=.8, density=True)
    axes[1].axvline(base_ctr, color=PALETTE["bad"], ls="--", lw=1.5,
                    label=f"整体平均 CTR {base_ctr:.4f}")
    axes[1].set_xlim(0, float(np.quantile(p_true, 0.995)))
    axes[1].set_xlabel("真实点击率（先验隐含分布）")
    axes[1].set_ylabel("密度")
    axes[1].legend(fontsize=9)
    axes[1].set_title(f"文案真实点击率的先验分布（95% 区间 "
                      f"{prior_q[0.05]:.4f}–{prior_q[0.95]:.4f}）")
    fig.suptitle("图 4 ｜ 同一文章内不同标题的真实点击率确实存在系统性差异（τ² 显著大于 0）", y=1.02)
    savefig(fig, "fig04_tau2_and_prior", "m2")

    # ---------------- 图 5：噪声基准 ----------------
    fig, ax = plt.subplots(figsize=(9.4, 4.4))
    bins = np.linspace(0, 0.03, 70)
    ax.hist(spread_obs.values, bins=bins, density=True, color=PALETTE["base"], alpha=.75,
            label=f"实测臂间极差（中位 {spread_obs.median():.4f}）")
    rng2 = np.random.default_rng(SEED)
    sim = rng2.binomial(int(round(med_n)), base_ctr,
                        size=(60000, int(round(arms.groupby('clickability_test_id').size().mean()))))
    sim_spread = (sim / med_n).max(axis=1) - (sim / med_n).min(axis=1)
    ax.hist(sim_spread, bins=bins, density=True, color=PALETTE["accent"], alpha=.55,
            label=f"纯噪声基准（中位 {nz['median']:.4f}）")
    ax.axvline(spread_obs.median(), color=PALETTE["base"], ls="--", lw=1.3)
    ax.axvline(nz["median"], color=PALETTE["accent"], ls="--", lw=1.3)
    ax.set_xlabel("同一实验内最高与最低臂的点击率之差")
    ax.set_ylabel("密度")
    ax.set_xlim(0, 0.03)
    ax.set_title(f"图 5 ｜ 实测臂间差距是纯噪声基准的 {noise_bench['ratio_mean']:.2f} 倍："
                 f"真实效应存在，但噪声占比约 {decomp['noise_share']:.0%}")
    ax.legend(fontsize=9)
    savefig(fig, "fig05_noise_benchmark", "m2")

    # ---------------- 图 6：功效 ----------------
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2))
    ns = np.logspace(2.7, 5.6, 200)
    for d, c in zip(deltas, [PALETTE["good"], PALETTE["blue"], PALETTE["accent"], PALETTE["bad"]]):
        axes[0].plot(ns, [power.power_two_prop(x, base_ctr, d) for x in ns], lw=1.9, color=c,
                     label=f"真实提升 {d:.0%}")
    axes[0].axhline(0.8, color=PALETTE["muted"], ls="--", lw=1.2)
    axes[0].axvline(med_n, color=PALETTE["base"], ls=":", lw=1.6)
    axes[0].annotate(f"本档案实际中位样本量\n{med_n:,.0f} 次曝光/臂", xy=(med_n, .55),
                     xytext=(med_n * 1.25, .60), fontsize=8.6,
                     arrowprops=dict(arrowstyle="->", lw=1, color=PALETTE["base"]))
    axes[0].set_xscale("log")
    axes[0].set_xlabel("每臂曝光量（对数轴）")
    axes[0].set_ylabel("检验功效")
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(fontsize=9, loc="upper left")
    axes[0].set_title("两臂等样本量、双侧 α=0.05 的功效曲线")
    axes[1].hist(pw20, bins=50, color=PALETTE["blue"], alpha=.8)
    axes[1].axvline(0.8, color=PALETTE["bad"], ls="--", lw=1.5, label="功效 0.8 目标线")
    axes[1].axvline(float(np.median(pw20)), color=PALETTE["accent"], ls="-", lw=1.8,
                    label=f"实测中位功效 {np.median(pw20):.2f}")
    axes[1].set_xlabel("每个实验在其实际样本量下、检出 20% 提升的功效")
    axes[1].set_ylabel("实验个数")
    axes[1].legend(fontsize=9)
    axes[1].set_title(f"仅 {pw['share_power_ge_80pct_for_20pct_lift']:.1%} 的实验达到 80% 功效")
    fig.suptitle("图 6 ｜ 功效审计：这批实验在设计上就不足以检出它自己要找的效应", y=1.02)
    savefig(fig, "fig06_power_audit", "m2")

    # ---------------- 落盘 ----------------
    dump_table(power_tbl, "02_power_table.csv")
    dump_table(sel_tbl, "02_selection_accuracy.csv")
    dump_table(pd.DataFrame([{"quantile": k, "ctr": v} for k, v in prior_q.items()]),
               "02_prior_ctr_quantiles.csv")
    res = {"tau2": {"all": all_fit, "train": expl_fit, "holdout": hold_fit},
           "tau2_used": tau2, "prior_ctr_distribution": prior_stats,
           "variance_decomposition": decomp, "noise_benchmark": noise_bench,
           "power": pw, "power_table": power_tbl.to_dict("records"),
           "selection_accuracy": sel_tbl.to_dict("records"),
           "predictive_calibration": cover,
           "base_ctr": base_ctr, "median_arm_impressions": med_n}
    dump_json(res, "02_effects_power.json")

    print("\n=== 效应量与功效 ===")
    print(f"  τ²={tau2:.5f} → 真实 CTR 的 90% 区间 [{prior_q[0.05]:.4f}, {prior_q[0.95]:.4f}]，"
          f"变异系数 {prior_stats['cv_pct']:.0f}%")
    print(f"  臂级观测方差中，真实文案差异占 {decomp['signal_share']:.1%}，抽样噪声占 {decomp['noise_share']:.1%}")
    print(f"  留一臂 95% 预测区间覆盖率 {cover['coverage_95']:.3f}（名义 0.95）")
    return res


if __name__ == "__main__":
    main()
