"""
一键复跑整个项目。

    python -m src.run_all

会依次执行：
    0  重建实验数据集
    1  随机化失衡审计
    2  效应量先验 / 功效 / 模型校准
    3  赢家诅咒与跨投放可迁移性
    4  上线决策规则对比（训练/留出）
    5  数据质量缺陷的后果审计与政策评估
最后汇总所有关键指标到 results/99_headline.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import (build_dataset, m1_audit, m2_effects_power, m3_winners_curse,  # noqa: E402
                 m4_decision_rules, m5_quality_audit, m6_confirmatory)
from src.common import RESULTS, dump_json  # noqa: E402


def collect_headline():
    """把所有模块的结论压缩成一份"关键指标表"，供报告与简历引用。"""
    def J(name):
        with open(RESULTS / name, encoding="utf-8") as f:
            return json.load(f)

    ds = J("00_dataset_summary.json")
    a1 = J("01_audit_summary.json")
    m2 = J("02_effects_power.json")
    m3 = J("03_winners_curse.json")
    m4 = J("04_decision_rules.json")
    m5 = J("05_quality_policy.json")
    m6 = J("06_confirmatory.json")

    rs = {r["key"]: r for r in m4["rule_summary"]}
    h = {
        "data": {
            "n_tests": ds["n_tests"], "n_arms": ds["n_arms"],
            "impressions": ds["impressions_total"], "clicks": ds["clicks_total"],
            "overall_ctr": ds["overall_ctr"],
            "date_range": [ds["date_min"], ds["date_max"]],
            "splits": ds["tests_by_split"],
        },
        "audit": {
            "srm_flag_rate_p001": a1["overall"]["flag_rate_p001"],
            "flag_rate_inside_official_window": a1["overall"]["flag_rate_window_official"],
            "flag_rate_outside_official_window": a1["overall"]["flag_rate_outside_official"],
            "detected_window": [a1["detected_window"]["start"], a1["detected_window"]["end"]],
            "official_window": ["2013-06-25", "2014-01-10"],
            "precision_vs_official": a1["detected_window"]["overlap_with_official"],
            "recall_vs_official": a1["detected_window"]["recall_of_official"],
            "hour_sd_inside": a1["hour_signature"]["flag_rate_in_window_sd"],
            "hour_sd_outside": a1["hour_signature"]["flag_rate_out_window_sd"],
        },
        "effects": {
            "tau2": m2["tau2_used"],
            "tau2_estimates": m2["variance_decomposition"]["tau2_estimates"],
            "true_ctr_q05": m2["prior_ctr_distribution"]["q5"],
            "true_ctr_q95": m2["prior_ctr_distribution"]["q95"],
            "true_ctr_cv_pct": m2["prior_ctr_distribution"]["cv_pct"],
            "signal_share_logit": m2["variance_decomposition"]["signal_share"],
            "signal_share_ctr": m2["variance_decomposition"]["ctr_signal_share"],
            "noise_benchmark_ratio": m2["noise_benchmark"]["ratio_mean"],
            "observed_spread_median": m2["noise_benchmark"]["observed_median_spread"],
            "noise_spread_median": m2["noise_benchmark"]["pure_noise_median_spread"],
            "power_median_for_20pct": m2["power"]["median_power_for_20pct_lift"],
            "share_power_ge80_for_20pct": m2["power"]["share_power_ge_80pct_for_20pct_lift"],
            "median_mde_rel": m2["power"]["median_mde_rel"],
            "loo_coverage_95": m2["predictive_calibration"]["coverage_95"],
            "loo_coverage_90": m2["predictive_calibration"]["coverage_90"],
            "selection_accuracy_k4": [r for r in m2["selection_accuracy"] if r["k"] == 4][0]["P_correct"],
        },
        "winners_curse": {
            "naive_lift": m3["winners_curse"]["naive_lift"],
            "honest_lift": m3["winners_curse"]["honest_lift"],
            "oracle_lift": m3["winners_curse"]["oracle_lift"],
            "lift_overstatement_pp": m3["winners_curse"]["lift_overstatement_pp"],
            "observed_to_honest_overstatement_pct": m3["winners_curse"]["overstatement_pct_of_shrunk"],
            "capture_rate_all": m3["winners_curse"]["capture_rate"],
            "mean_regret_r1": m3["winners_curse"]["mean_regret_r1"],
        },
        "transferability": {
            "tau2_within_subsample": m3["transferability"]["tau2_within_replication_subsample"],
            "tau2_transferable": m3["transferability"]["tau2_transferable_package"],
            "transferable_share": m3["transferability"]["transferable_share_vs_subsample"],
            "transfer_ci": m3["replication"]["package"]["cov_ci"],
            "corr_replication": m3["replication"]["package"]["corr"],
            "slope_optimal": m3["shrinkage_comparison"]["B_optimal_from_replication"],
            "slope_model": m3["shrinkage_comparison"]["B_model_weighted"],
            "mse_raw": m3["shrinkage_comparison"]["mse_raw"],
            "mse_zero": m3["shrinkage_comparison"]["mse_zero"],
            "mse_model_B": m3["shrinkage_comparison"]["mse_model_B"],
            "mse_optimal_B": m3["shrinkage_comparison"]["mse_at_B_optimal"],
            "per_design": m3["transferability"]["per_design"],
        },
        "decisions": {
            "n_holdout": m4["n_holdout_tests"],
            "r0_platform_regret_pp": rs["r0_platform"]["mean_regret_pp"],
            "r0_capture": rs["r0_platform"]["capture_rate"],
            "r1_regret_pp": rs["r1"]["mean_regret_pp"],
            "r1_capture": rs["r1"]["capture_rate"],
            "r3_regret_pp": rs["r3"]["mean_regret_pp"],
            "r3_capture": rs["r3"]["capture_rate"],
            "r1_r3_same_choice": m4["r1_vs_r3"]["share_same_choice"],
            "r2_reject_rate": m4["r2_gating"]["reject_rate"],
            "r2_median_pbest_of_deployers": m4["r2_gating"]["median_pbest_of_deployers"],
            "pbest_median": m4["pbest_distribution"]["median"],
            "share_pbest_lt_50": m4["pbest_distribution"]["share_lt_50"],
            "share_pbest_lt_80": m4["pbest_distribution"]["share_lt_80"],
            "max_gain_pp": m4["max_gain_pp"],
            "tau2_sensitivity_same_choice_min": min(r["share_same_r1_r3"] for r in m4["tau2_sensitivity"]),
        },
        "confirmatory_oos": {
            "n_tests": m6["n_tests"], "n_arms": m6["n_arms"],
            "impressions": m6["impressions"],
            "detected_window": m6["A_srm"]["detected_window"],
            "precision_vs_official": m6["A_srm"]["precision_vs_official"],
            "recall_vs_official": m6["A_srm"]["recall_vs_official"],
            "flag_rate_oos": m6["A_srm"]["flag_rate_p001"],
            "loo_coverage_95": m6["B_calibration"]["coverage_95"],
            "tau2_reestimated": m6["B_calibration"]["tau2_reestimated_on_confirmatory"],
            "naive_lift": m6["C_winners_curse"]["naive_lift"],
            "honest_lift": m6["C_winners_curse"]["honest_lift"],
            "oracle_lift": m6["C_winners_curse"]["oracle_lift"],
            "r1_regret_pp": m6["C_winners_curse"]["mean_regret_r1_pp"],
            "r0_regret_pp": m6["C_winners_curse"]["mean_regret_r0_pp"],
            "same_choice_r1_r3": m6["C_winners_curse"]["same_choice_r1_r3"],
            "share_pbest_lt_50": m6["C_winners_curse"]["share_pbest_lt_50"],
            "transfer_pairs": m6["E_transferability"]["n_pairs"],
            "transfer_cov": m6["E_transferability"]["cov"],
            "transfer_cov_ci": m6["E_transferability"]["cov_ci"],
            "transfer_share": m6["E_transferability"]["transferable_share"],
            "slope_optimal": m6["E_transferability"]["slope_ols"],
            "headline_pairs": m6["E_transferability"]["headline_n_pairs"],
            "headline_slope": m6["E_transferability"]["headline_slope"],
            "headline_slope_ci": m6["E_transferability"]["headline_slope_ci"],
            "n_replicated": sum(1 for r in m6["replication_table"] if "复现" in r["结论"]),
            "n_metrics": len(m6["replication_table"]),
        },
        "quality": {
            "n_flagged": m5["n_flagged"], "n_clean": m5["n_clean"],
            "phi_flagged": m5["phi_flagged"],
            "phi_clean": m5["test_A_dispersion"][1]["phi"],
            "coverage_flagged": m5["test_A_dispersion"][0]["actual_coverage_95"],
            "coverage_clean": m5["test_A_dispersion"][1]["actual_coverage_95"],
            "dose_response_r": m5["test_C_regression"]["pearson_r_cv_vs_product"],
            "dose_response_p": m5["test_C_regression"]["p_value"],
            "flagged_impressions": m5["cost_accounting"]["flagged_impressions"],
            "flagged_impression_share": m5["cost_accounting"]["flagged_impression_share"],
            "p2_forgone_impressions": m5["cost_accounting"]["p2_forgone_impressions"],
            "p1_vs_p3_choice_change": m5["cost_accounting"]["p3_choice_change_rate"],
            "regret_diff_flagged_vs_clean": m5["test_D_regret_diff_flagged_vs_clean"],
        },
    }
    dump_json(h, "99_headline.json")
    return h


def main():
    t0 = time.time()
    print("=" * 72)
    print(" 在线实验决策审计 —— 全流程复跑")
    print("=" * 72)
    build_dataset.build()
    m1_audit.main()
    m2_effects_power.main()
    m3_winners_curse.main()
    m4_decision_rules.main()
    m5_quality_audit.main()
    m6_confirmatory.main()
    h = collect_headline()
    print("\n" + "=" * 72)
    print(f" 全部完成，用时 {time.time() - t0:.0f}s")
    print("=" * 72)
    for section, vals in h.items():
        print(f"\n[{section}]")
        for k, v in vals.items():
            if isinstance(v, float):
                print(f"  {k:42s} {v:,.6g}")
            elif isinstance(v, (int, str)):
                print(f"  {k:42s} {v}")
    return h


if __name__ == "__main__":
    main()
