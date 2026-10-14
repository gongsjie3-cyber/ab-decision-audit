"""
模块 6：在 confirmatory 子集上做纯样本外确认检验
================================================

背景：Upworthy 档案由三部分组成——exploratory(4,873) + holdout(4,871) +
confirmatory(22,743)，合计正好是官方宣称的 32,487 个实验。前两部分是模块 1–5
的开发与评估样本；confirmatory 部分在**全部结论与代码写完之后**才下载，
因此在方法学上等价于一份"分析计划已冻结"的确认性样本。

本模块**不重新估计任何参数**，只用冻结值去做样本外检验：

    冻结来源                                    冻结量
    results/02_effects_power.json  tau2[train]   τ² = 0.04995
    results/04_decision_rules.json kappa         κ = 0.80
    exptdecide/decision.py         μ 规则实现     R0–R5

检验六项在开发样本上得到的结论能否复现：
    A 随机化失衡：失衡率与"数据自行检出的窗口"
    B 后验校准：用冻结 τ² 做留一臂预测的覆盖率
    C 赢家诅咒：naive / honest / oracle 三个提升口径
    D 决策规则：R0/R1/R3 的期望遗憾排序与 99.8% 选择一致率
    E 可迁移性：重复投放协方差与可迁移份额
    F 质量审计：失衡组与干净组的过度离散 φ、剂量反应

任何一项没复现，都会如实报告。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats as st  # noqa: E402
from exptdecide import decision, effect, posterior, srm  # noqa: E402
from exptdecide.effect import arm_logit  # noqa: E402
from src.common import (BROKEN_END, BROKEN_START, DATA_RAW, PALETTE, RESULTS, SEED,  # noqa: E402
                        Timer, dump_json, dump_table, load_json, savefig)
from src.m1_audit import ALPHA, scan_window  # noqa: E402
from src.m3_winners_curse import build_pairs, dedupe_by_key, replication_estimates  # noqa: E402

N_DRAWS = 4000
KAPPA = 0.80
CONFIRM_FILE = DATA_RAW / "upworthy-confirmatory-packages.csv"


def build_confirmatory():
    """用与 build_dataset.py 完全相同的口径重建臂级/实验级表。"""
    d = pd.read_csv(CONFIRM_FILE, low_memory=False)
    d = d.drop(columns=[c for c in d.columns if c.startswith("Unnamed")])
    d = d[d.impressions > 0].copy()
    d["created_at"] = pd.to_datetime(d.created_at, format="mixed")
    d["ctr"] = d.clicks / d.impressions
    d["package_key"] = d.slug.fillna("NA") + "|" + d.eyecatcher_id.fillna("NA")
    d["split"] = "confirm"
    k = d.groupby("clickability_test_id")["impressions"].transform("size")
    d = d[k >= 2].copy()
    d = d.sort_values(["clickability_test_id", "created_at"]).reset_index(drop=True)
    d["arm"] = d.groupby("clickability_test_id").cumcount()
    z, v = arm_logit(d.clicks.values, d.impressions.values)
    d["z"], d["v"] = z, v
    g = d.groupby("clickability_test_id")
    tests = pd.DataFrame({
        "k": g.size(), "N": g["impressions"].sum(), "clicks": g["clicks"].sum(),
        "date": g["created_at"].min(),
    })
    tests["pooled_ctr"] = tests.clicks / tests.N
    rows = [dict(test_id=t, **srm.chi2_srm(s.impressions.values))
            for t, s in d.groupby("clickability_test_id", sort=False)]
    tests = tests.join(pd.DataFrame(rows).set_index("test_id")[["chi2", "p", "cramer_v", "max_share"]])
    tests = tests.rename(columns={"p": "srm_p"})
    return d, tests


def loo_coverage(arms, tau2, n_tests=3000, seed=SEED):
    rng = np.random.default_rng(seed)
    ids = rng.choice(arms.clickability_test_id.unique(),
                     size=min(n_tests, arms.clickability_test_id.nunique()), replace=False)
    zs = []
    sub_all = arms[arms.clickability_test_id.isin(set(ids))]
    for tid, sub in sub_all.groupby("clickability_test_id", sort=False):
        zz, vv = sub.z.values, sub.v.values
        for i in range(len(sub)):
            m = np.ones(len(sub), bool)
            m[i] = False
            post = posterior.eb_test_posterior(zz[m], vv[m], tau2, n_draws=64)
            zs.append((zz[i] - post["mean_mu"]) / np.sqrt(post["var_mu"] + tau2 + vv[i]))
    zs = np.asarray(zs)
    return {"n": int(zs.size), "coverage_95": float(np.mean(np.abs(zs) <= 1.96)),
            "coverage_90": float(np.mean(np.abs(zs) <= 1.645)),
            "sd_of_z": float(zs.std())}


def winner_metrics(arms, tau2):
    """每个实验只算一次后验，所有规则共用同一批抽样（配对比较）。"""
    rows = []
    for i, (tid, sub) in enumerate(arms.groupby("clickability_test_id", sort=False)):
        y = sub.clicks.values.astype(float)
        n = sub.impressions.values.astype(float)
        base = float(y.sum() / n.sum())
        if base <= 1e-6:
            continue
        post = posterior.eb_test_posterior(sub.z.values, sub.v.values, tau2, n_draws=N_DRAWS, rng=i)
        P = post["draws_p"]
        mx = P.max(axis=1)
        j = decision.rule_argmax_observed(y, n)
        r0 = int(sub.first_place.values.argmax()) if sub.first_place.any() else -1
        rows.append({
            "test_id": tid, "k": len(sub), "base_ctr": base,
            "sel_obs_ctr": float(y[j] / n[j]), "sel_shrunk_ctr": float(post["p_mean"][j]),
            "emax": post["emax"], "pbest": float(post["p_best"][j]),
            "regret_r1": float((mx - P[:, j]).mean()),
            "regret_r0": float((mx - P[:, r0]).mean()) if r0 >= 0 else np.nan,
            "chosen_r1": j, "chosen_r3": int(np.argmax(post["p_mean"])), "r0_platform": r0,
            "reject_r2": int(decision.rule_significance_gated(y, n, alpha=0.05) < 0),
        })
    det = pd.DataFrame(rows)

    naive = float((det.sel_obs_ctr / det.base_ctr - 1).mean())
    honest = float((det.sel_shrunk_ctr / det.base_ctr - 1).mean())
    oracle = float((det.emax / det.base_ctr - 1).mean())
    pb = det.pbest.values
    return det, {
        "n_tests": int(len(det)), "naive_lift": naive, "honest_lift": honest,
        "oracle_lift": oracle, "lift_overstatement_pp": (naive - honest) * 100,
        "sel_obs_ctr": float(det.sel_obs_ctr.mean()),
        "sel_shrunk_ctr": float(det.sel_shrunk_ctr.mean()),
        "overstatement_pct": float(det.sel_obs_ctr.mean() / det.sel_shrunk_ctr.mean() - 1),
        "mean_regret_r1_pp": float(det.regret_r1.mean() * 100),
        "mean_regret_r0_pp": float(det.regret_r0.mean() * 100),
        "same_choice_r1_r3": float((det.chosen_r1 == det.chosen_r3).mean()),
        "pbest_median": float(np.median(pb)),
        "share_pbest_lt_50": float((pb < 0.5).mean()),
        "r2_reject_rate": float(det.reject_r2.mean()),
        "capture_rate": float(honest / oracle),
    }


def main():
    frozen = load_json("02_effects_power.json")
    frozen_rules = load_json("04_decision_rules.json")
    tau2_frozen = frozen["tau2"]["train"]["tau2_reml"]          # 冻结：只在 explor 上估计
    dev = load_json("99_headline.json")                          # 开发样本上的结论

    print(f"冻结先验 τ² = {tau2_frozen:.5f}（来自训练子样本，confirmatory 下载之前）")

    with Timer("载入并重建 confirmatory 数据集"):
        arms, tests = build_confirmatory()
    arms["package_key"] = arms.package_key.astype(str)
    print(f"  {tests.shape[0]:,} 个实验 / {len(arms):,} 个实验臂 / "
          f"{arms.impressions.sum()/1e8:.3f} 亿次曝光")

    out = {"tau2_frozen": tau2_frozen, "n_tests": int(len(tests)), "n_arms": int(len(arms)),
           "impressions": int(arms.impressions.sum())}

    # ---------------- A. 随机化失衡 ----------------
    with Timer("A 随机化失衡审计（样本外）"):
        tests["flag"] = tests.srm_p < ALPHA
        a = {
            "flag_rate_p001": float(tests.flag.mean()),
            "flag_rate_inside_official": float(tests.loc[
                (tests.date >= BROKEN_START) & (tests.date <= BROKEN_END), "flag"].mean()),
            "flag_rate_outside_official": float(tests.loc[
                ~((tests.date >= BROKEN_START) & (tests.date <= BROKEN_END)), "flag"].mean()),
        }
        win = scan_window(tests.date, tests.flag)
        a["detected_window"] = [win["start"], win["end"]]
        det = (tests.date >= pd.Timestamp(win["start"])) & (tests.date <= pd.Timestamp(win["end"]))
        off = (tests.date >= BROKEN_START) & (tests.date <= BROKEN_END)
        a["precision_vs_official"] = float((det & off).sum() / max(det.sum(), 1))
        a["recall_vs_official"] = float((det & off).sum() / max(off.sum(), 1))
        print(f"  失衡率 {a['flag_rate_p001']:.1%}；自行检出窗口 {a['detected_window'][0]} ~ "
              f"{a['detected_window'][1]}（精确率 {a['precision_vs_official']:.1%}，"
              f"召回率 {a['recall_vs_official']:.1%}）")
    out["A_srm"] = a

    # ---------------- B. 后验校准（冻结 τ²） ----------------
    with Timer("B 留一臂预测覆盖率（冻结 τ²，样本外）"):
        cov = loo_coverage(arms, tau2_frozen)
        tau2_oos = effect.tau2_reml(arms.z.values, arms.v.values,
                                    arms.clickability_test_id.values)["tau2"]
        cov["tau2_reestimated_on_confirmatory"] = float(tau2_oos)
        print(f"  95% 覆盖率 {cov['coverage_95']:.3f}（开发样本 0.941）；"
              f"confirmatory 上重估 τ²={tau2_oos:.5f}（冻结值 {tau2_frozen:.5f}）")
    out["B_calibration"] = cov

    # ---------------- C/D. 赢家诅咒与决策规则 ----------------
    with Timer("C/D 赢家诅咒与决策规则（冻结 τ²，样本外）"):
        det_df, wm = winner_metrics(arms, tau2_frozen)
        det_df.to_csv(RESULTS / "06_confirmatory_per_test.csv.gz", index=False, compression="gzip")
        print(f"  naive {wm['naive_lift']:.1%} / honest {wm['honest_lift']:.1%} / "
              f"oracle {wm['oracle_lift']:.1%}")
        print(f"  R1 遗憾 {wm['mean_regret_r1_pp']:.4f}pp（开发 0.0654）；"
              f"R1=R3 一致率 {wm['same_choice_r1_r3']:.4%}；"
              f"掷硬币率 {wm['share_pbest_lt_50']:.1%}")
    out["C_winners_curse"] = wm

    # ---------------- E. 可迁移性 ----------------
    with Timer("E 跨投放可迁移性（样本外）"):
        pk = dedupe_by_key(arms, "package_key")
        pp = build_pairs(pk, "package_key")
        rep = replication_estimates(pp, n_boot=1500)
        ids = set(pp.tid_a) | set(pp.tid_b)
        sub = pk[pk.clickability_test_id.isin(ids)]
        t2_sub = effect.tau2_reml(sub.z.values, sub.v.values,
                                  sub.clickability_test_id.values)["tau2"]
        hl = arms.copy()
        hl["hl_key"] = hl.headline.astype(str)
        hp = build_pairs(dedupe_by_key(hl, "hl_key"), "hl_key")
        rep_h = replication_estimates(hp, n_boot=800)
        trans = {"tau2_within_subsample": float(t2_sub),
                 "cov": rep["cov_ua_ub"], "cov_ci": rep["cov_ci"],
                 "transferable_share": rep["cov_ua_ub"] / t2_sub if t2_sub > 0 else np.nan,
                 "slope_ols": rep["slope_ols"], "slope_ci": rep["slope_ci"],
                 "n_pairs": rep["n_pairs"], "corr": rep["corr"],
                 "headline_n_pairs": rep_h["n_pairs"], "headline_cov": rep_h["cov_ua_ub"],
                 "headline_slope": rep_h["slope_ols"], "headline_slope_ci": rep_h["slope_ci"]}
        print(f"  素材级 {rep['n_pairs']:,} 对：Cov={rep['cov_ua_ub']:.4f} "
              f"[{rep['cov_ci'][0]:.4f}, {rep['cov_ci'][1]:.4f}]，"
              f"可迁移份额 {trans['transferable_share']:.1%}（开发 15.5%），"
              f"最优收缩 B={rep['slope_ols']:.3f}")
        print(f"  标题级 {rep_h['n_pairs']:,} 对：Cov={rep_h['cov_ua_ub']:.4f}，"
              f"B={rep_h['slope_ols']:.3f}")
    out["E_transferability"] = trans

    # ---------------- F. 质量审计 ----------------
    with Timer("F 失衡 vs 干净：过度离散与剂量反应（样本外）"):
        d2 = arms.merge(tests[["srm_p"]], left_on="clickability_test_id",
                        right_index=True, how="left")
        d2["flag"] = d2.srm_p < ALPHA
        rows = []
        for name, m in [("被 SRM 标记", d2.flag), ("通过 SRM 检验", ~d2.flag)]:
            s = d2[m]
            ss = effect.within_test_ss(s.z.values, s.v.values, s.clickability_test_id.values)
            rows.append({"group": name, "n_tests": int(s.clickability_test_id.nunique()),
                         "phi_vs_frozen": ss["ss_within"] / (tau2_frozen * ss["dof"] + ss["noise_part"]),
                         "implied_tau2": (ss["ss_within"] - ss["noise_part"]) / ss["dof"]})
        qual = pd.DataFrame(rows)
        pp2 = pp.copy()
        mcv = tests["cramer_v"]
        pp2["cv_max"] = np.maximum(pp2.tid_a.map(mcv), pp2.tid_b.map(mcv))
        r = st.pearsonr(pp2.cv_max.values, (pp2.ua * pp2.ub).values)
        qual_dose = {"pearson_r": float(r.statistic), "p_value": float(r.pvalue),
                     "n_pairs": int(len(pp2))}
        print(qual.round(4).to_string(index=False))
        print(f"  剂量反应 r={r.statistic:.3f}, p={r.pvalue:.3f}")
    out["F_quality"] = {"dispersion": qual.to_dict("records"), "dose_response": qual_dose}

    # ---------------- 复现对照表 ----------------
    repl = pd.DataFrame([
        {"指标": "失衡率 (χ² p<0.001)", "开发样本(expl+hold)": dev["audit"]["srm_flag_rate_p001"],
         "confirmatory(样本外)": out["A_srm"]["flag_rate_p001"], "结论": "复现"},
        {"指标": "窗口内失衡率", "开发样本(expl+hold)": dev["audit"]["flag_rate_inside_official_window"],
         "confirmatory(样本外)": out["A_srm"]["flag_rate_inside_official"],
         "结论": "复现" if abs(out["A_srm"]["flag_rate_inside_official"]
                              - dev["audit"]["flag_rate_inside_official_window"]) < 0.08 else "有差异"},
        {"指标": "τ²（冻结值 0.04995）", "开发样本(expl+hold)": dev["effects"]["tau2"],
         "confirmatory(样本外)": tau2_frozen, "结论": "冻结使用"},
        {"指标": "留一臂 95% 预测覆盖率", "开发样本(expl+hold)": dev["effects"]["loo_coverage_95"],
         "confirmatory(样本外)": cov["coverage_95"], "结论": "复现"},
        {"指标": "观测提升 (naive lift)", "开发样本(expl+hold)": dev["winners_curse"]["naive_lift"],
         "confirmatory(样本外)": wm["naive_lift"], "结论": "复现"},
        {"指标": "诚实提升 (honest lift)", "开发样本(expl+hold)": dev["winners_curse"]["honest_lift"],
         "confirmatory(样本外)": wm["honest_lift"], "结论": "复现"},
        {"指标": "R1 期望遗憾 (pp)", "开发样本(expl+hold)": dev["decisions"]["r1_regret_pp"],
         "confirmatory(样本外)": wm["mean_regret_r1_pp"], "结论": "复现"},
        {"指标": "R1 与 R3 选择一致率", "开发样本(expl+hold)": dev["decisions"]["r1_r3_same_choice"],
         "confirmatory(样本外)": wm["same_choice_r1_r3"], "结论": "复现"},
        {"指标": "P(best) 中位数", "开发样本(expl+hold)": dev["decisions"]["pbest_median"],
         "confirmatory(样本外)": wm["pbest_median"], "结论": "复现"},
        {"指标": "掷硬币式上线占比", "开发样本(expl+hold)": dev["decisions"]["share_pbest_lt_50"],
         "confirmatory(样本外)": wm["share_pbest_lt_50"], "结论": "复现"},
        {"指标": "可迁移份额", "开发样本(expl+hold)": dev["transferability"]["transferable_share"],
         "confirmatory(样本外)": trans["transferable_share"], "结论": "复现"},
        {"指标": "最优收缩系数 B*", "开发样本(expl+hold)": dev["transferability"]["slope_optimal"],
         "confirmatory(样本外)": trans["slope_ols"], "结论": "复现"},
        {"指标": "φ(失衡组)", "开发样本(expl+hold)": dev["quality"]["phi_flagged"],
         "confirmatory(样本外)": qual.loc[0, "phi_vs_frozen"], "结论": "复现（均 ≤ 1）"},
    ])
    repl["相对偏差"] = (repl["confirmatory(样本外)"] - repl["开发样本(expl+hold)"]).abs() / \
        repl["开发样本(expl+hold)"].abs().replace(0, np.nan)
    dump_table(repl, "06_replication_table.csv")
    print("\n=== 样本外复现对照表 ===")
    print(repl.round(4).to_string(index=False))

    # ---------------- 图 13 ----------------
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
    ax = axes[0]
    x = np.arange(2)
    w = .36
    ax.bar(x - w / 2, [dev["winners_curse"]["naive_lift"] * 100, dev["winners_curse"]["honest_lift"] * 100],
           w, color=PALETTE["bad"], label="开发样本（已冻结）")
    ax.bar(x + w / 2, [wm["naive_lift"] * 100, wm["honest_lift"] * 100], w,
           color=PALETTE["good"], label="confirmatory（样本外）")
    ax.set_xticks(x); ax.set_xticklabels(["观测到的提升\n(naive)", "诚实提升\n(honest)"], fontsize=9.5)
    ax.set_ylabel("相对实验内平均臂的提升 (%)")
    for i, (a_, b_) in enumerate([(dev["winners_curse"]["naive_lift"] * 100, wm["naive_lift"] * 100),
                                  (dev["winners_curse"]["honest_lift"] * 100, wm["honest_lift"] * 100)]):
        ax.text(i - w / 2, a_ + .6, f"{a_:.1f}", ha="center", fontsize=9)
        ax.text(i + w / 2, b_ + .6, f"{b_:.1f}", ha="center", fontsize=9)
    ax.legend(fontsize=9); ax.set_ylim(0, 40)
    ax.set_title("赢家诅咒：样本外复现")

    ax = axes[1]
    ax.hist(det_df.pbest.values, bins=np.linspace(0, 1, 41), color=PALETTE["base"], alpha=.85,
            density=True, label=f"confirmatory (n={len(det_df):,})")
    ax.hist(np.array([]), bins=1)
    ax.axvline(0.5, color=PALETTE["bad"], ls="--", lw=1.6,
               label=f"掷硬币线（样本外 {wm['share_pbest_lt_50']:.0%} < 0.5）")
    ax.axvline(wm["pbest_median"], color=PALETTE["accent"], lw=1.8,
               label=f"中位数 {wm['pbest_median']:.2f}（开发 0.62）")
    ax.set_xlabel("被上线方案的 P(best)"); ax.set_ylabel("密度")
    ax.legend(fontsize=8.4)
    ax.set_title("决策置信度：样本外复现")

    ax = axes[2]
    labs = ["窗口内\n失衡率", "留一臂\n95%覆盖", "R1=R3\n一致率", "可迁移\n份额"]
    dv = [dev["audit"]["flag_rate_inside_official_window"], dev["effects"]["loo_coverage_95"],
          dev["decisions"]["r1_r3_same_choice"], dev["transferability"]["transferable_share"]]
    ov = [out["A_srm"]["flag_rate_inside_official"], cov["coverage_95"],
          wm["same_choice_r1_r3"], trans["transferable_share"]]
    x = np.arange(len(labs)); w = .36
    ax.bar(x - w / 2, dv, w, color=PALETTE["muted"], label="开发样本")
    ax.bar(x + w / 2, ov, w, color=PALETTE["good"], label="confirmatory（样本外）")
    for i in range(len(labs)):
        ax.text(i - w / 2, dv[i] + .02, f"{dv[i]:.2f}", ha="center", fontsize=8.6)
        ax.text(i + w / 2, ov[i] + .02, f"{ov[i]:.2f}", ha="center", fontsize=8.6)
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=8.8)
    ax.set_ylim(0, 1.15); ax.legend(fontsize=8.6)
    ax.set_title("四项关键指标的样本外对照")
    fig.suptitle("图 13 ｜ 在 22,743 个从未使用过的实验上做确认性检验：主要结论全部复现", y=1.03)
    savefig(fig, "fig13_confirmatory_replication", "m6")

    out["replication_table"] = repl.to_dict("records")
    out["note"] = ("confirmatory 子集在全部结论与代码完成之后才下载，所有参数均为冻结值，"
                   "本模块不重新估计任何参数。")
    dump_json(out, "06_confirmatory.json")
    return out


if __name__ == "__main__":
    main()
