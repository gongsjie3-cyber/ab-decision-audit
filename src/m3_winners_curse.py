"""
模块 3：赢家诅咒的量化与「跨投放可迁移性」验证
==============================================

赢家诅咒（winner's curse）：被选中的方案之所以被选中，部分原因是它的噪声为正，
所以**它观测到的提升系统性地高于它的真实提升**。

本模块用两条独立路径量化它，并给出一个更狠的结论：

路径 A（模型内）
    用经验贝叶斯后验给出被选臂的"诚实"估计，与观测值对比。

路径 B（模型外，本项目的关键创新）
    档案里有大量**同一素材（标题+配图）被投放到多个互相独立的实验**的情形，
    这是天然的重复测量设计。记 u = 组内中心化的 logit 观测，则

        Cov(u_a, u_b) = Var(素材的稳定效应)

    因为它对两次测量各自的抽样噪声是稳健的。于是可以完全绕开分层模型，
    直接回答两个问题：
      (1) 实验内部 τ²_within 里的差异，有多少是"素材固有"、能迁移到下一次投放？
      (2) 预测下一次投放时，最优的收缩系数到底应该是多少？

    结论（见 results/03_winners_curse.json）是反直觉的：可迁移份额很小，
    模型隐含的收缩强度远远不够，需要收缩到约 0.1 才最优。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt  # noqa: E402
from exptdecide import decision, effect, posterior  # noqa: E402
from src.common import (PALETTE, RESULTS, SEED, Timer, dump_json, dump_table,  # noqa: E402
                        load_arms, load_json, savefig)

MIN_BASE_CTR = 1e-6      # 过滤掉零点击实验（否则相对提升无定义）
N_DRAWS = 4000


# --------------------------------------------------------------- 工具
def dedupe_by_key(df, key):
    """把 (key, 实验) 合并成一条观测，再做组内中心化得到 u。

    既避免同一素材在同一实验里被重复计数（伪重复），又让 u 消掉文章层面的
    基线差异——不同实验的基线本来就不可比。
    """
    agg = (df.groupby([key, "clickability_test_id", "split"], as_index=False)
             .agg(impressions=("impressions", "sum"), clicks=("clicks", "sum"),
                  date=("created_at", "first"), headline=("headline", "first")))
    z, v = effect.arm_logit(agg.clicks.values, agg.impressions.values)
    agg["z"], agg["v"] = z, v
    agg["ctr"] = agg.clicks / agg.impressions
    agg["u"] = agg.z - agg.groupby("clickability_test_id")["z"].transform("mean")
    return agg


def build_pairs(df, key, min_tests=2):
    """构造同一 key 在不同实验间的重复测量配对。"""
    cnt = df.groupby(key)["clickability_test_id"].nunique()
    sub = df[df[key].isin(cnt[cnt >= min_tests].index)]
    rows = []
    for k, g in sub.groupby(key, sort=False):
        rec = g.drop_duplicates("clickability_test_id").to_dict("records")
        for i in range(len(rec)):
            for j in range(i + 1, len(rec)):
                rows.append({
                    "key": k, "ua": rec[i]["u"], "ub": rec[j]["u"],
                    "va": rec[i]["v"], "vb": rec[j]["v"],
                    "ctra": rec[i]["ctr"], "ctrb": rec[j]["ctr"],
                    "ta": rec[i]["date"], "tb": rec[j]["date"],
                    "tid_a": rec[i]["clickability_test_id"],
                    "tid_b": rec[j]["clickability_test_id"],
                    "split_a": rec[i]["split"], "split_b": rec[j]["split"],
                    "n_a": rec[i]["impressions"], "n_b": rec[j]["impressions"],
                })
    p = pd.DataFrame(rows)
    if len(p):
        p["gap_days"] = (pd.to_datetime(p.tb) - pd.to_datetime(p.ta)).dt.total_seconds().abs() / 86400
        p["cross_split"] = p.split_a != p.split_b
    return p


def replication_estimates(pairs, n_boot=2000, seed=SEED):
    """由重复测量对估计可迁移方差 Cov(u_a,u_b)、最优收缩斜率及 bootstrap 区间。"""
    ua, ub = pairs.ua.values, pairs.ub.values
    ma, mb = ua.mean(), ub.mean()
    cov = float(np.mean((ua - ma) * (ub - mb)))
    var_a = float(np.mean((ua - ma) ** 2))
    var_b = float(np.mean((ub - mb) ** 2))
    slope = cov / var_a if var_a > 0 else np.nan

    rng = np.random.default_rng(seed)
    n = len(pairs)
    bs_cov, bs_slope = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        s = rng.integers(0, n, n)
        a, c = ua[s], ub[s]
        cv = np.mean((a - a.mean()) * (c - c.mean()))
        vv = np.mean((a - a.mean()) ** 2)
        bs_cov[b] = cv
        bs_slope[b] = cv / vv if vv > 0 else np.nan
    return {
        "n_pairs": int(n), "n_keys": int(pairs.key.nunique()),
        "cov_ua_ub": cov,
        "cov_ci": [float(np.quantile(bs_cov, .025)), float(np.quantile(bs_cov, .975))],
        "var_ua": var_a, "var_ub": var_b,
        "corr": float(cov / np.sqrt(var_a * var_b)) if var_a * var_b > 0 else np.nan,
        "slope_ols": float(slope),
        "slope_ci": [float(np.nanquantile(bs_slope, .025)), float(np.nanquantile(bs_slope, .975))],
        "mean_va": float(pairs.va.mean()),
        "median_gap_days": float(pairs.gap_days.median()) if "gap_days" in pairs else np.nan,
    }


def boot_mean(x, n_boot=1500, seed=SEED):
    v = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    s = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
    return [float(np.quantile(s, .025)), float(np.quantile(s, .975))]


def stratify_cov(pairs, mask, label):
    s = pairs[mask]
    if len(s) < 25:
        return {"stratum": label, "n_pairs": int(len(s)), "cov": np.nan, "corr": np.nan}
    c = float(np.mean((s.ua - s.ua.mean()) * (s.ub - s.ub.mean())))
    va, vb = float(s.ua.var()), float(s.ub.var())
    return {"stratum": label, "n_pairs": int(len(s)), "cov": c,
            "var_ua": va, "var_ub": vb,
            "corr": float(c / np.sqrt(va * vb)) if va * vb > 0 else np.nan}


# --------------------------------------------------------------- 主流程
def main():
    arms = load_arms()
    pf = load_json("02_effects_power.json")
    tau2 = pf["tau2_used"]
    arms["package_key"] = arms.package_key.astype(str)
    arms["date"] = pd.to_datetime(arms.created_at)
    arms["broken"] = (arms.date >= pd.Timestamp("2013-06-25")) & (arms.date <= pd.Timestamp("2014-01-10"))

    # ============ 路径 A：模型内的赢家诅咒 ============
    rows = []
    n_drop = 0
    with Timer("逐实验经验贝叶斯后验（全部 9,744 个实验）"):
        for i, (tid, sub) in enumerate(arms.groupby("clickability_test_id", sort=False)):
            y = sub.clicks.values.astype(float)
            n = sub.impressions.values.astype(float)
            base_t = float(y.sum() / n.sum())
            if base_t <= MIN_BASE_CTR:
                n_drop += 1
                continue
            post = posterior.eb_test_posterior(sub.z.values, sub.v.values, tau2,
                                               n_draws=N_DRAWS, rng=i)
            j = decision.rule_argmax_observed(y, n)
            P = post["draws_p"]
            rows.append({
                "test_id": tid, "split": sub.split.iloc[0], "k": len(sub),
                "date": sub.date.min(), "N": float(n.sum()), "base_ctr": base_t,
                "sel_arm": j, "sel_obs_ctr": float(y[j] / n[j]),
                "sel_shrunk_ctr": float(post["p_mean"][j]),
                "sel_pbest": float(post["p_best"][j]), "emax": post["emax"],
                "regret_r1": float((P.max(axis=1) - P[:, j]).mean()),
                "p_below_base": float((P[:, j] < base_t).mean()),
            })
    est = pd.DataFrame(rows)
    est.to_csv(RESULTS / "03_per_test_estimates.csv.gz", index=False, compression="gzip")
    print(f"[table ] results/03_per_test_estimates.csv.gz ({len(est):,} 行，零点击实验剔除 {n_drop})")

    naive_lift = est.sel_obs_ctr / est.base_ctr - 1
    honest_lift = est.sel_shrunk_ctr / est.base_ctr - 1
    oracle_lift = est.emax / est.base_ctr - 1

    winner = {
        "n_tests": int(len(est)), "n_tests_dropped_zero_clicks": int(n_drop),
        "mean_sel_obs_ctr": float(est.sel_obs_ctr.mean()),
        "mean_sel_shrunk_ctr": float(est.sel_shrunk_ctr.mean()),
        "overstatement_abs": float(est.sel_obs_ctr.mean() - est.sel_shrunk_ctr.mean()),
        "overstatement_pct_of_shrunk": float(est.sel_obs_ctr.mean() / est.sel_shrunk_ctr.mean() - 1),
        "mean_base_ctr": float(est.base_ctr.mean()),
        "naive_lift": float(naive_lift.mean()), "naive_lift_ci": boot_mean(naive_lift),
        "honest_lift": float(honest_lift.mean()), "honest_lift_ci": boot_mean(honest_lift),
        "oracle_lift": float(oracle_lift.mean()), "oracle_lift_ci": boot_mean(oracle_lift),
        "lift_overstatement_pp": float((naive_lift.mean() - honest_lift.mean()) * 100),
        "lift_shrink_ratio": float(naive_lift.mean() / honest_lift.mean()),
        "mean_regret_r1": float(est.regret_r1.mean()),
        "mean_regret_r1_ci": boot_mean(est.regret_r1),
        "capture_rate": float(honest_lift.mean() / oracle_lift.mean()),
        "p_below_base_mean": float(est.p_below_base.mean()),
        "share_p_below_base_gt_20pct": float((est.p_below_base > 0.20).mean()),
    }
    print(f"\n  被选中臂：观测 CTR {winner['mean_sel_obs_ctr']:.5f} → 收缩后 "
          f"{winner['mean_sel_shrunk_ctr']:.5f}（高估 {winner['overstatement_pct_of_shrunk']:.2%}）")
    print(f"  声称提升 {winner['naive_lift']:.1%} → 诚实提升 {winner['honest_lift']:.1%}"
          f"（上限 {winner['oracle_lift']:.1%}，捕获率 {winner['capture_rate']:.1%}）")

    # ============ 路径 B：跨投放可迁移性 ============
    with Timer("构造重复投放配对（素材 / slug / 标题 三种粒度）"):
        frames = {}
        for name, key in [("package", "package_key"), ("slug", "slug"), ("headline", "hl_key")]:
            base = arms.copy()
            if key == "hl_key":
                base["hl_key"] = base.headline.astype(str)
            base[key] = base[key].astype(str)
            d = dedupe_by_key(base, key)
            frames[name] = (d, build_pairs(d, key))
    for name, (d, p) in frames.items():
        print(f"  {name:9s}: {len(p):,} 对配对，{p.key.nunique():,} 个 key 被重复投放")

    pkg_df, pkg_pairs = frames["package"]
    rep = {name: replication_estimates(p) for name, (d, p) in frames.items()}

    # 与可迁移性对比的"同情境内"方差：用重复投放所涉及的那些实验重新估 τ²
    tinfo = arms.groupby("clickability_test_id").agg(broken=("broken", "first"),
                                                     k=("impressions", "size"))
    pkg_pairs["broken_a"] = pkg_pairs.tid_a.map(tinfo.broken).astype(bool)
    pkg_pairs["broken_b"] = pkg_pairs.tid_b.map(tinfo.broken).astype(bool)
    rep_tests = set(pkg_pairs.tid_a) | set(pkg_pairs.tid_b)
    sub_df = pkg_df[pkg_df.clickability_test_id.isin(rep_tests)]
    tau2_sub = effect.tau2_reml(sub_df.z.values, sub_df.v.values,
                                sub_df.clickability_test_id.values)["tau2"]
    tau2_all_pkg = effect.tau2_reml(pkg_df.z.values, pkg_df.v.values,
                                    pkg_df.clickability_test_id.values)["tau2"]
    transfer = {
        "tau2_within_full": tau2_all_pkg,
        "tau2_within_replication_subsample": tau2_sub,
        "tau2_transferable_package": rep["package"]["cov_ua_ub"],
        "transferable_share_vs_subsample": rep["package"]["cov_ua_ub"] / tau2_sub,
        "transferable_share_vs_full": rep["package"]["cov_ua_ub"] / tau2_all_pkg,
        "per_design": {},
    }
    for name, (d, p) in frames.items():
        ids = set(p.tid_a) | set(p.tid_b)
        sd = d[d.clickability_test_id.isin(ids)]
        t2 = effect.tau2_reml(sd.z.values, sd.v.values, sd.clickability_test_id.values)["tau2"]
        transfer["per_design"][name] = {
            "tau2_within_subsample": t2, "n_tests_involved": len(ids),
            "cov": rep[name]["cov_ua_ub"], "cov_ci": rep[name]["cov_ci"],
            "transferable_share": rep[name]["cov_ua_ub"] / t2 if t2 > 0 else np.nan,
            "slope_ols": rep[name]["slope_ols"], "slope_ci": rep[name]["slope_ci"],
        }

    # 分层：故障窗口 / 跨子样本 / 对手集合重叠
    pkgset = arms.groupby("clickability_test_id")["package_key"].apply(set)
    pkg_pairs["jaccard"] = [len(pkgset[a] & pkgset[b]) / len(pkgset[a] | pkgset[b])
                            for a, b in zip(pkg_pairs.tid_a, pkg_pairs.tid_b)]
    strata = [
        stratify_cov(pkg_pairs, ~pkg_pairs.broken_a & ~pkg_pairs.broken_b, "两次投放都在故障窗口外"),
        stratify_cov(pkg_pairs, pkg_pairs.broken_a | pkg_pairs.broken_b, "至少一次在故障窗口内"),
        stratify_cov(pkg_pairs, pkg_pairs.cross_split, "跨两个子样本(严格独立)"),
        stratify_cov(pkg_pairs, pkg_pairs.jaccard >= 0.8, "对手集合几乎相同 (J≥0.8)"),
        stratify_cov(pkg_pairs, pkg_pairs.jaccard < 0.5, "对手集合差异大 (J<0.5)"),
        stratify_cov(pkg_pairs, pkg_pairs.gap_days > 7, "两次投放间隔 >7 天"),
    ]
    strata = pd.DataFrame(strata)

    # 时间间隔敏感性
    tstab = []
    for lo, hi in [(0, 0.5), (0.5, 7), (7, 1e9)]:
        m = (pkg_pairs.gap_days >= lo) & (pkg_pairs.gap_days < hi)
        if m.sum() >= 25:
            tstab.append(stratify_cov(pkg_pairs, m, f"{lo:g}–{hi:g} 天"))
    tstab = pd.DataFrame(tstab)

    # 重复投放素材是否有选择偏差
    sel_bias = {
        "mean_ctr_repeated": float(arms.loc[arms.package_key.isin(set(pkg_pairs.key)), "ctr"].mean()),
        "mean_ctr_other": float(arms.loc[~arms.package_key.isin(set(pkg_pairs.key)), "ctr"].mean()),
        "share_in_broken_window_repeated": float(arms.loc[arms.package_key.isin(set(pkg_pairs.key)),
                                                      "broken"].mean()),
        "share_in_broken_window_other": float(arms.loc[~arms.package_key.isin(set(pkg_pairs.key)),
                                                     "broken"].mean()),
    }

    # 样本外预测：用一次投放预测同一素材的下一次投放
    with Timer("样本外预测误差比较（不同收缩强度）"):
        ua, ub, va = pkg_pairs.ua.values, pkg_pairs.ub.values, pkg_pairs.va.values
        B_eb = effect.shrink_factor(va, tau2)
        B_model_w = float(np.average(B_eb, weights=ua ** 2))
        Bs = np.linspace(0, 1, 401)
        mse_curve = [float(np.mean((ub - B * ua) ** 2)) for B in Bs]
        B_star = float(Bs[int(np.argmin(mse_curve))])
        preds = {
            "不预测（预测为 0）": np.zeros_like(ua),
            "原始观测不收缩 (B=1)": ua,
            f"分层模型隐含收缩 (B={B_model_w:.2f})": B_eb * ua,
            f"重复投放估出的最优收缩 (B={B_star:.2f})": B_star * ua,
        }
        oos = pd.DataFrame([{
            "predictor": k, "mse": float(np.mean((ub - p) ** 2)),
            "mae": float(np.mean(np.abs(ub - p))),
            "r2_vs_naive": float(1 - np.mean((ub - p) ** 2) / np.mean((ub - ua) ** 2)),
        } for k, p in preds.items()]).sort_values("mse")
        print(oos.to_string(index=False))

        shrink_cmp = {
            "B_model_weighted": B_model_w,
            "B_model_mean": float(B_eb.mean()),
            "B_optimal_from_replication": B_star,
            "B_optimal_slope_ci": rep["package"]["slope_ci"],
            "mse_curve_B": Bs.tolist(), "mse_curve": mse_curve,
            "mse_raw": float(np.mean((ub - ua) ** 2)),
            "mse_zero": float(np.mean(ub ** 2)),
            "mse_model_B": float(np.mean((ub - B_eb * ua) ** 2)),
            "mse_at_B_optimal": float(np.mean((ub - B_star * ua) ** 2)),
        }

    # ============ 图 7：赢家诅咒 ============
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3))
    ax = axes[0]
    ax.scatter(est.sel_obs_ctr * 100, est.sel_shrunk_ctr * 100, s=5, alpha=.16,
               color=PALETTE["base"], edgecolors="none")
    lim = [0, min(5.0, float(est.sel_obs_ctr.quantile(.999)) * 100)]
    ax.plot(lim, lim, color=PALETTE["bad"], ls="--", lw=1.5, label="无偏线 y = x")
    ax.scatter([est.sel_obs_ctr.mean() * 100], [est.sel_shrunk_ctr.mean() * 100], s=120,
               color=PALETTE["accent"], zorder=5,
               label=f"均值：{est.sel_obs_ctr.mean()*100:.2f}% → {est.sel_shrunk_ctr.mean()*100:.2f}%")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("被选中方案：观测点击率 (%)")
    ax.set_ylabel("被选中方案：收缩后的诚实估计 (%)")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_title("被上线的方案，真实点击率系统性低于观测值")

    ax = axes[1]
    fracs = [winner["naive_lift"], winner["honest_lift"], winner["oracle_lift"]]
    cis = [winner["naive_lift_ci"], winner["honest_lift_ci"], winner["oracle_lift_ci"]]
    vals = [f * 100 for f in fracs]
    err = [[(f - c[0]) * 100 for f, c in zip(fracs, cis)],
           [(c[1] - f) * 100 for f, c in zip(fracs, cis)]]
    labels = ["观测到的提升\n（平台会这样汇报）", "经验贝叶斯修正后\n（诚实估计）", "事后可知的上限\n（真·最优臂）"]
    bars = ax.bar(labels, vals, color=[PALETTE["bad"], PALETTE["good"], PALETTE["muted"]], width=.62)
    ax.errorbar(range(3), vals, yerr=err, fmt="none", ecolor=PALETTE["base"], capsize=4, lw=1.2)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"+{v:.1f}%", ha="center", fontsize=10)
    ax.set_ylabel("相对同一实验平均臂的提升 (%)")
    ax.set_ylim(0, max(vals) * 1.28)
    ax.set_title(f"「最优臂」的收益被高估 {winner['lift_overstatement_pp']:.1f} 个百分点"
                 f"（约 {winner['lift_shrink_ratio']:.2f} 倍）")
    fig.suptitle("图 7 ｜ 赢家诅咒的量化：9,744 个真实实验中被选中方案的系统性高估", y=1.03)
    savefig(fig, "fig07_winners_curse", "m3")

    # ============ 图 8：可迁移性 ============
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.4))
    ax = axes[0]
    ax.scatter(pkg_pairs.ua, pkg_pairs.ub, s=15, alpha=.4, color=PALETTE["blue"], edgecolors="none")
    xs = np.linspace(-0.7, 0.7, 50)
    ax.plot(xs, xs, color=PALETTE["bad"], ls="--", lw=1.5, label="不收缩（斜率 1.0）")
    ax.plot(xs, rep["package"]["slope_ols"] * xs, color=PALETTE["good"], lw=2.2,
            label=f"重复投放估出的最优斜率 {rep['package']['slope_ols']:.2f}")
    ax.plot(xs, shrink_cmp["B_model_weighted"] * xs, color=PALETTE["accent"], lw=1.6, ls="-.",
            label=f"分层模型隐含 {shrink_cmp['B_model_weighted']:.2f}")
    ax.set_xlim(-0.6, 0.6); ax.set_ylim(-0.6, 0.6)
    ax.set_xlabel("同一素材在实验 A 中的组内中心化 logit")
    ax.set_ylabel("同一素材在实验 B 中的组内中心化 logit")
    ax.legend(fontsize=8.2, loc="upper left")
    ax.set_title(f"重复投放素材：{rep['package']['n_pairs']:,} 对独立测量")

    ax = axes[1]
    lab = ["同一实验内\nτ²_within\n(留一臂覆盖率 94% 验证)",
           "跨投放可迁移部分\nCov(u_a,u_b)\n(素材级)",
           "跨投放可迁移部分\n(标题级, 4.8k 对)"]
    hts = [tau2_sub, rep["package"]["cov_ua_ub"], rep["headline"]["cov_ua_ub"]]
    cls = [PALETTE["base"], PALETTE["good"], PALETTE["purple"]]
    ax.bar(lab, hts, color=cls, width=.6)
    ax.errorbar([1], [hts[1]],
                yerr=[[hts[1] - rep["package"]["cov_ci"][0]], [rep["package"]["cov_ci"][1] - hts[1]]],
                fmt="none", ecolor=PALETTE["base"], capsize=5, lw=1.3)
    ax.errorbar([2], [hts[2]],
                yerr=[[hts[2] - rep["headline"]["cov_ci"][0]], [rep["headline"]["cov_ci"][1] - hts[2]]],
                fmt="none", ecolor=PALETTE["base"], capsize=5, lw=1.3)
    for i, v in enumerate(hts):
        ax.text(i, v + max(hts) * .03, f"{v:.4f}", ha="center", fontsize=9.5)
    ax.set_ylabel("方差分量（logit 尺度）")
    ax.set_ylim(0, max(hts) * 1.32)
    ax.set_title(f"只有约 {transfer['transferable_share_vs_subsample']:.0%} 的臂间差异可迁移")

    ax = axes[2]
    Bs = np.array(shrink_cmp["mse_curve_B"])
    ax.plot(Bs, shrink_cmp["mse_curve"], color=PALETTE["base"], lw=2)
    ax.axvline(shrink_cmp["B_model_weighted"], color=PALETTE["accent"], ls="-.", lw=1.6,
               label=f"分层模型隐含 B={shrink_cmp['B_model_weighted']:.2f}")
    ax.axvline(shrink_cmp["B_optimal_from_replication"], color=PALETTE["good"], ls="--", lw=1.8,
               label=f"重复投放估出的最优 B={shrink_cmp['B_optimal_from_replication']:.2f}")
    ax.axhline(shrink_cmp["mse_zero"], color=PALETTE["muted"], ls=":", lw=1.4,
               label="完全不预测的误差")
    ax.set_xlabel("收缩系数 B（预测 = B × 上次观测偏差）")
    ax.set_ylabel("预测下一次投放的均方误差")
    ax.legend(fontsize=8.6)
    ax.set_title(f"预测「下一次投放」时最优 B 仅为模型隐含值的 {shrink_cmp['B_optimal_from_replication']/shrink_cmp['B_model_weighted']:.2f} 倍", fontsize=10)
    fig.suptitle("图 8 ｜ 跨投放验证：实验内的「最优文案」优势，大部分无法迁移到下一次投放", y=1.03)
    savefig(fig, "fig08_transferability", "m3")

    # ============ 落盘 ============
    dump_table(oos, "03_oos_prediction.csv")
    rep_tbl = pd.DataFrame(rep).T.reset_index().rename(columns={"index": "design"})
    dump_table(rep_tbl, "03_replication_estimates.csv")
    dump_table(strata, "03_transferability_strata.csv")
    if len(tstab):
        dump_table(tstab, "03_time_stability.csv")
    res = {"winners_curse": winner, "replication": rep, "transferability": transfer,
           "strata": strata.to_dict("records"),
           "time_stability": tstab.to_dict("records") if len(tstab) else [],
           "selection_bias_check": sel_bias,
           "oos_prediction": oos.to_dict("records"),
           "shrinkage_comparison": {k: v for k, v in shrink_cmp.items()
                                    if k not in ("mse_curve_B", "mse_curve")},
           "tau2_model": tau2}
    dump_json(res, "03_winners_curse.json")

    print("\n=== 跨投放可迁移性 ===")
    print(f"  同一实验内 τ²_within = {tau2_sub:.4f}（重复投放子样本）"
          f" vs {tau2_all_pkg:.4f}（全体）")
    print(f"  可迁移部分 Cov = {rep['package']['cov_ua_ub']:.4f} "
          f"[{rep['package']['cov_ci'][0]:.4f}, {rep['package']['cov_ci'][1]:.4f}]"
          f"  → 仅占 {transfer['transferable_share_vs_subsample']:.1%}")
    print(f"  最优收缩系数 B* = {shrink_cmp['B_optimal_from_replication']:.3f} "
          f"[{rep['package']['slope_ci'][0]:.3f}, {rep['package']['slope_ci'][1]:.3f}]"
          f"  vs 模型隐含 {shrink_cmp['B_model_weighted']:.3f}")
    return res


if __name__ == "__main__":
    main()
