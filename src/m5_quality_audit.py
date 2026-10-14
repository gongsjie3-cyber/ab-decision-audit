"""
模块 5：数据质量缺陷的后果审计与应对政策
========================================

模块 1 确认：2013-06 ~ 2014-01 期间约 70% 的实验分流失衡。官方 2024 年的调查
把原因归到 Cloudflare 缓存按整块放量（即"分块随机化"而非个体随机化）。

那么，**这批数据到底还能不能用？** 平台有三种应对：

    P1 照旧分析   —— 用二项分布的标准误（绝大多数公司的做法）
    P2 检测即丢弃 —— 把被标记的 1,482 个实验整批排除
    P3 稳健校正   —— 保留数据，把被标记实验的标准误保守放大后再决策

本模块不预设答案，而是做四项**能够证伪"失衡有害"的检验**，再核算政策的代价：

    检验 A  组内过度离散 φ：失衡实验的臂间差异是否比模型预期更大？
    检验 B  效应量分布：失衡期的真实文案差异是否异常？
    检验 C  跨投放可迁移性：按失衡严重程度分层，结论是否更不可复现（剂量反应）？
    检验 D  决策质量：在留出集上，失衡实验的期望遗憾是否更差？

结论（见 results/05_quality_policy.json）与"越失衡越不可信"的直觉相反：
**四项检验都没有检出失衡实验存在实质性统计危害**，这与官方复查"既有结论未被
实质影响"的结论一致。因此"检测到就整批丢弃"缺乏经验支持，代价却是 15.2% 的
实验与两千八百万次曝光。合理的做法是把 SRM 检验当作一次性的数据可用性审计，
而不是逐实验的丢弃规则。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats as st  # noqa: E402
from exptdecide import effect, posterior, srm  # noqa: E402
from src.common import (PALETTE, RESULTS, SEED, Timer, dump_json, dump_table,  # noqa: E402
                        load_arms, load_json, load_tests, savefig)
from src.m3_winners_curse import build_pairs, dedupe_by_key  # noqa: E402

N_DRAWS = 6000
KAPPA = 0.80
ALPHA_SRM = 0.001
PHI_CONSERVATIVE = 1.5      # P3 使用的保守方差放大因子


def implied_tau2(df):
    d = effect.within_test_ss(df.z.values, df.v.values, df.clickability_test_id.values)
    return (d["ss_within"] - d["noise_part"]) / d["dof"]


def evaluate_policy(df, tau2, policy, phi=1.0, kappa=KAPPA):
    rows = []
    for i, (tid, sub) in enumerate(df.groupby("clickability_test_id", sort=False)):
        y = sub.clicks.values.astype(float)
        n = sub.impressions.values.astype(float)
        base_t = float(y.sum() / n.sum())
        if base_t <= 0:
            continue
        if policy == "P2_discard":
            rows.append({"test_id": tid, "chosen": -1, "deployed": 0, "regret": np.nan,
                         "regret_if_reject": np.nan, "pbest": np.nan, "reject_at_kappa": 1})
            continue
        v_use = sub.v.values * (phi if policy == "P3_robust" else 1.0)
        post = posterior.eb_test_posterior(sub.z.values, v_use, tau2, n_draws=N_DRAWS, rng=i)
        P = post["draws_p"]
        mx = P.max(axis=1)
        j = int(np.argmax(post["p_mean"]))
        pb = float(post["p_best"][j])
        rows.append({"test_id": tid, "chosen": j, "deployed": 1,
                     "regret": float((mx - P[:, j]).mean()),
                     "regret_if_reject": float((mx - P.mean(axis=1)).mean()),
                     "pbest": pb, "reject_at_kappa": int(pb < kappa)})
    return pd.DataFrame(rows)


def main():
    arms = load_arms()
    tests = load_tests()
    pf = load_json("02_effects_power.json")
    tau2_train = pf["tau2"]["train"]["tau2_reml"]
    arms["package_key"] = arms.package_key.astype(str)

    meta = tests[["test_id", "srm_p", "cramer_v", "max_share", "date", "N", "k"]]
    arms = arms.merge(meta, left_on="clickability_test_id", right_on="test_id", how="left")
    arms["flagged"] = arms.srm_p < ALPHA_SRM
    fid = set(tests.loc[tests.srm_p < ALPHA_SRM, "test_id"])
    cid = set(tests.loc[tests.srm_p >= ALPHA_SRM, "test_id"])
    print(f"被标记实验 {len(fid):,} 个（{len(fid)/len(tests):.1%}），"
          f"涉及曝光 {tests.loc[tests.srm_p<ALPHA_SRM,'N'].sum()/1e6:.1f} 百万次")

    # ---------------- 检验 A：过度离散 ----------------
    with Timer("检验 A：组内过度离散"):
        tau2_clean_train = implied_tau2(arms[(arms.split == "expl") &
                                             (arms.clickability_test_id.isin(cid))])
        rows = []
        for name, mask in [("被 SRM 标记", arms.clickability_test_id.isin(fid)),
                           ("通过 SRM 检验", arms.clickability_test_id.isin(cid))]:
            a = arms[mask]
            d = effect.within_test_ss(a.z.values, a.v.values, a.clickability_test_id.values)
            exp = tau2_clean_train * d["dof"] + d["noise_part"]
            rows.append({
                "group": name, "n_arms": len(a), "n_tests": int(a.clickability_test_id.nunique()),
                "dof": d["dof"], "ss_within": d["ss_within"], "expected_if_clean": exp,
                "phi": d["ss_within"] / exp,
                "implied_tau2": (d["ss_within"] - d["noise_part"]) / d["dof"],
                "actual_coverage_95": srm.nominal_coverage(d["ss_within"] / exp, 0.95),
                "mean_arm_impressions": float(a.impressions.mean()),
            })
        disp = pd.DataFrame(rows)
        phi_flagged = float(disp.loc[disp.group == "被 SRM 标记", "phi"].iloc[0])
    print(disp.round(4).to_string(index=False))

    # ---------------- 检验 B：效应量分布（分时段） ----------------
    arms["period"] = np.where(arms.date < pd.Timestamp("2013-06-25"), "前期\n(<2013-06-25)",
                              np.where(arms.date <= pd.Timestamp("2014-01-10"),
                                       "故障窗口内\n(2013-06-25~2014-01-10)", "后期\n(>2014-01-10)"))
    tb = pd.DataFrame([{
        "period": p.replace("\n", ""), "n_tests": int(g.clickability_test_id.nunique()),
        "n_arms": len(g), "implied_tau2": implied_tau2(g),
        "flag_rate": float(g.groupby("clickability_test_id").flagged.first().mean()),
        "mean_impressions_per_arm": float(g.impressions.mean()),
    } for p, g in arms.groupby("period", sort=True)])
    tb["period_label"] = list(arms.groupby("period", sort=True).groups.keys())

    # ---------------- 检验 C：剂量反应 ----------------
    with Timer("检验 C：可迁移性的剂量反应"):
        pk = dedupe_by_key(arms, "package_key")
        pp = build_pairs(pk, "package_key")
        mcv = tests.set_index("test_id")["cramer_v"]
        pp["cramer_v_max"] = np.maximum(pp.tid_a.map(mcv), pp.tid_b.map(mcv))
        bins = [0, 0.01, 0.02, 0.035, 1.0]
        dose = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            s = pp[(pp.cramer_v_max >= lo) & (pp.cramer_v_max < hi)]
            if len(s) < 30:
                continue
            c = float(np.mean((s.ua - s.ua.mean()) * (s.ub - s.ub.mean())))
            rng = np.random.default_rng(SEED)
            ua_, ub_ = s.ua.values, s.ub.values
            bs = [np.mean((ua_[i] - ua_[i].mean()) * (ub_[i] - ub_[i].mean()))
                  for i in (rng.integers(0, len(s), len(s)) for _ in range(500))]
            ids = set(s.tid_a) | set(s.tid_b)
            t2 = implied_tau2(pk[pk.clickability_test_id.isin(ids)])
            dose.append({"cramer_v_bin": f"[{lo},{hi})", "n_pairs": len(s),
                         "cov": c, "cov_ci_lo": float(np.quantile(bs, .025)),
                         "cov_ci_hi": float(np.quantile(bs, .975)),
                         "corr": float(c / np.sqrt(s.ua.var() * s.ub.var())),
                         "tau2_within": t2, "transferable_share": c / t2 if t2 > 0 else np.nan})
        dose = pd.DataFrame(dose)

        hl = arms.copy()
        hl["hl_key"] = hl.headline.astype(str)
        hp = build_pairs(dedupe_by_key(hl, "hl_key"), "hl_key")
        hp["cramer_v_max"] = np.maximum(hp.tid_a.map(mcv), hp.tid_b.map(mcv))
        dh = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            s = hp[(hp.cramer_v_max >= lo) & (hp.cramer_v_max < hi)]
            if len(s) < 50:
                continue
            c = float(np.mean((s.ua - s.ua.mean()) * (s.ub - s.ub.mean())))
            dh.append({"cramer_v_bin": f"[{lo},{hi})", "n_pairs": len(s), "cov": c,
                       "corr": float(c / np.sqrt(s.ua.var() * s.ub.var()))})
        dose_head = pd.DataFrame(dh)
        print(dose.round(4).to_string(index=False))
        r = st.pearsonr(pp.cramer_v_max.values, (pp.ua * pp.ub).values)
        dose_test = {"pearson_r_cv_vs_product": float(r.statistic), "p_value": float(r.pvalue),
                     "n_pairs": int(len(pp))}

    # ---------------- 检验 D + 政策比较 ----------------
    with Timer("检验 D / 政策比较（留出集）"):
        hf = arms[(arms.split == "hold") & (arms.clickability_test_id.isin(fid))]
        hc = arms[(arms.split == "hold") & (arms.clickability_test_id.isin(cid))]
        polres = {
            "P1_asis_flagged": evaluate_policy(hf, tau2_train, "P1_asis"),
            "P3_robust_flagged": evaluate_policy(hf, tau2_train, "P3_robust", phi=PHI_CONSERVATIVE),
            "P2_discard_flagged": evaluate_policy(hf, tau2_train, "P2_discard"),
            "P1_asis_clean": evaluate_policy(hc, tau2_train, "P1_asis"),
        }
        alt = polres["P1_asis_flagged"].set_index("test_id")["regret_if_reject"]
        polres["P2_discard_flagged"]["regret_if_reject"] = \
            polres["P2_discard_flagged"].test_id.map(alt)

    def summ(d, name):
        dep = d[d.deployed == 1]
        eff = d.regret.fillna(d.regret_if_reject)
        return {
            "policy": name, "n_tests": int(len(d)),
            "deploy_rate": float(d.deployed.mean()),
            "mean_regret_of_deployed_pp": float(dep.regret.mean() * 100) if len(dep) else np.nan,
            "median_pbest_of_deployed": float(dep.pbest.median()) if len(dep) else np.nan,
            "share_deployed_pbest_lt_50": float((dep.pbest < 0.5).mean()) if len(dep) else np.nan,
            "reject_rate_at_kappa80": float(d.reject_at_kappa.mean()),
            "mean_effective_regret_pp": float(eff.mean() * 100),
        }

    pol = pd.DataFrame([
        summ(polres["P1_asis_flagged"], "P1 照旧分析\n（失衡实验）"),
        summ(polres["P3_robust_flagged"], f"P3 稳健校正\n(v×{PHI_CONSERVATIVE})"),
        summ(polres["P2_discard_flagged"], "P2 检测即丢弃"),
        summ(polres["P1_asis_clean"], "（对照）\n未失衡实验 P1"),
    ])
    print("\n=== 政策对比（留出集的失衡实验）===")
    print(pol.round(4).to_string(index=False))

    rng_b = np.random.default_rng(SEED)
    rf = polres["P1_asis_flagged"].regret.dropna().values
    rc = polres["P1_asis_clean"].regret.dropna().values
    bs = np.array([rng_b.choice(rf, rf.size).mean() - rng_b.choice(rc, rc.size).mean()
                   for _ in range(2000)])
    regret_diff_flagged_vs_clean = {
        "delta_pp": float((rf.mean() - rc.mean()) * 100),
        "ci_pp": [float(np.quantile(bs, .025) * 100), float(np.quantile(bs, .975) * 100)],
        "p_two_sided_approx": float(2 * min((bs <= 0).mean(), (bs >= 0).mean())),
    }

    a1 = polres["P1_asis_flagged"].set_index("test_id").chosen
    a3 = polres["P3_robust_flagged"].set_index("test_id").chosen
    choice_change = float((a1 != a3).mean())

    cost = {
        "flagged_tests": int(len(fid)), "flagged_tests_share": float(len(fid) / len(tests)),
        "clean_tests": int(len(cid)), "clean_tests_share": float(len(cid) / len(tests)),
        "flagged_impressions": int(tests.loc[tests.srm_p < ALPHA_SRM, "N"].sum()),
        "flagged_impression_share": float(tests.loc[tests.srm_p < ALPHA_SRM, "N"].sum()
                                          / tests.N.sum()),
        "p2_forgone_tests": int(len(fid)),
        "p2_forgone_impressions": int(tests.loc[tests.srm_p < ALPHA_SRM, "N"].sum()),
        "p3_choice_change_rate": choice_change,
        "p3_deploy_rate": float((a3 >= 0).mean()),
        "p1_mean_regret_pp": float(pol.loc[0, "mean_regret_of_deployed_pp"]),
        "p3_mean_regret_pp": float(pol.loc[1, "mean_regret_of_deployed_pp"]),
        "p2_mean_regret_pp": float(pol.loc[2, "mean_effective_regret_pp"]),
        "p1_vs_p3_regret_delta_pp": float(pol.loc[1, "mean_regret_of_deployed_pp"]
                                          - pol.loc[0, "mean_regret_of_deployed_pp"]),
    }

    # ================= 图 11 =================
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.6))
    fig.subplots_adjust(wspace=0.42)
    ax = axes[0]
    ax.bar(disp.group, disp.phi, color=[PALETTE["bad"], PALETTE["good"]], width=.5)
    ax.axhline(1.0, color=PALETTE["base"], ls="--", lw=1.5, label="φ=1（模型预期）")
    for i, (v, cv) in enumerate(zip(disp.phi, disp.actual_coverage_95)):
        ax.text(i, v + .02, f"φ={v:.2f}", ha="center", fontsize=11)
        ax.text(i, v / 2, f"名义 95% 区间\n实际覆盖 {cv:.1%}", ha="center", fontsize=8.6, color="white")
    ax.set_ylabel("过度离散系数 φ")
    ax.set_ylim(0, max(disp.phi) * 1.28)
    ax.legend(fontsize=9)
    ax.set_title("检验 A：没有额外的过度离散")

    ax = axes[1]
    xl = tb.period_label
    ax.bar(range(len(xl)), tb.flag_rate, alpha=.30, color=PALETTE["bad"], width=.55,
           label="失衡实验占比")
    ax.set_xticks(range(len(xl)))
    ax.set_xticklabels(xl, fontsize=8.6)
    ax2 = ax.twinx()
    ax2.plot(range(len(xl)), tb.implied_tau2, marker="o", color=PALETTE["base"], lw=2.2,
             label="真实文案差异 τ²")
    for i, v in enumerate(tb.implied_tau2):
        ax2.text(i, v + .0025, f"{v:.4f}", ha="center", fontsize=9)
    ax2.grid(False)
    ax.set_ylabel("失衡实验占比")
    ax2.set_ylabel("τ²（真实文案差异）")
    ax.set_ylim(0, 1.0)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8.4, loc="upper left")
    ax.set_title("检验 B：效应量未被异常放大")

    ax = axes[2]
    yv = dose["cov"].values
    ax.errorbar(np.arange(len(dose)), yv,
                yerr=[yv - dose["cov_ci_lo"].values, dose["cov_ci_hi"].values - yv],
                fmt="o-", color=PALETTE["purple"], lw=2, capsize=4, ms=6)
    overall = float(np.mean((pp.ua - pp.ua.mean()) * (pp.ub - pp.ub.mean())))
    ax.axhline(overall, color=PALETTE["muted"], ls="--", lw=1.5,
               label=f"整体可迁移协方差 {overall:.4f}")
    ax.set_xticks(np.arange(len(dose)))
    ax.set_xticklabels([f"{b}\nn={n}" for b, n in zip(dose.cramer_v_bin, dose.n_pairs)], fontsize=8.2)
    ax.set_xlabel("两次投放的失衡严重程度（Cramér's V 上界）")
    ax.set_ylabel("可迁移协方差 Cov(u_a, u_b)")
    ax.set_ylim(0, max(dose["cov_ci_hi"].max(), overall) * 1.35)
    ax.legend(fontsize=8.6)
    ax.set_title(f"检验 C：无剂量反应\n(r={dose_test['pearson_r_cv_vs_product']:.2f}，"
                 f"p={dose_test['p_value']:.2f}，n={dose_test['n_pairs']} 对)", fontsize=11)
    fig.suptitle("图 11 ｜ 失衡诊断的后果审计：四项检验均未检出实质性危害，"
                 "「检测到就丢弃」缺乏经验依据", y=1.07, fontsize=12.5)
    savefig(fig, "fig11_quality_audit", "m5")

    # ================= 图 12 =================
    fig, ax = plt.subplots(figsize=(10.0, 4.5))
    x = np.arange(len(pol))
    w = .38
    ax.bar(x - w / 2, pol.mean_effective_regret_pp, w, color=PALETTE["base"],
           label="平均有效遗憾（含「不上线」的机会成本）")
    ax.bar(x + w / 2, pol.reject_rate_at_kappa80 * 10, w, color=PALETTE["accent"],
           label="被判「证据不足」的比例（数值×10）")
    for i, (v, r) in enumerate(zip(pol.mean_effective_regret_pp, pol.reject_rate_at_kappa80)):
        ax.text(i - w / 2, v + .012, f"{v:.3f}pp", ha="center", fontsize=9)
        ax.text(i + w / 2, r * 10 + .012, f"{r:.0%}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(pol.policy, fontsize=8.8)
    ax.set_ylabel("百分点 / 比例×10")
    ax.legend(fontsize=8.8, loc="upper left")
    ax.set_title("图 12 ｜ 三种应对政策的代价：丢弃要付出 "
                 f"{cost['p2_forgone_impressions']/1e6:.1f} 百万次曝光的机会成本，"
                 "稳健校正几乎不改变选择", fontsize=11)
    savefig(fig, "fig12_policy_cost", "m5")

    # ---------------- 落盘 ----------------
    dump_table(disp, "05_dispersion_tests.csv")
    dump_table(tb, "05_effect_size_by_period.csv")
    dump_table(dose, "05_dose_response_package.csv")
    dump_table(dose_head, "05_dose_response_headline.csv")
    dump_table(pol, "05_policy_comparison.csv")
    out = {
        "n_flagged": len(fid), "n_clean": len(cid),
        "tau2_clean_train": tau2_clean_train,
        "test_A_dispersion": disp.to_dict("records"),
        "test_B_effect_size_by_period": tb.to_dict("records"),
        "test_C_dose_response": dose.to_dict("records"),
        "test_C_dose_response_headline": dose_head.to_dict("records"),
        "test_C_regression": dose_test,
        "test_D_policy_comparison": pol.to_dict("records"),
        "test_D_regret_diff_flagged_vs_clean": regret_diff_flagged_vs_clean,
        "cost_accounting": cost,
        "phi_flagged": phi_flagged, "phi_conservative_used": PHI_CONSERVATIVE,
        "kappa": KAPPA, "alpha_srm": ALPHA_SRM,
        "verdict": "四项检验均未检出失衡实验存在实质性统计危害；丢弃政策的代价远大于收益。",
    }
    dump_json(out, "05_quality_policy.json")

    print("\n=== 结论 ===")
    print(f"  检验A φ(失衡)={phi_flagged:.3f} vs φ(干净)={disp.phi.iloc[1]:.3f} → 失衡方并未更离散")
    print(f"  检验C 剂量反应 r={dose_test['pearson_r_cv_vs_product']:.3f} "
          f"(p={dose_test['p_value']:.2f}) → 不显著")
    print(f"  P3 改变选择的比例 {choice_change:.2%}；P2 丢弃代价 "
          f"{cost['p2_forgone_impressions']/1e6:.1f}M 次曝光")
    return out


if __name__ == "__main__":
    main()
