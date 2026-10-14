"""
exptdecide 单元测试。

运行：  python -m exptdecide.tests.test_exptdecide      （无需 pytest）
或      pytest exptdecide/tests -q

重点验证三件容易出错的事：
1. Helmert 对照确实把实验固定效应消掉、且保持正交；
2. tau^2 的两个估计量能从模拟数据里把已知真值找回来；
3. 后验自洽：E[max p] >= max E[p]，p_best 归一，收缩方向正确。
"""

from __future__ import annotations

import numpy as np

from exptdecide import decision, effect, posterior, power, srm


def _simulate(tau2=0.05, p=0.015, k=4, n=3000, n_tests=400, seed=0):
    rng = np.random.default_rng(seed)
    mu = np.log(p / (1 - p))
    Z, V, G, Y, N = [], [], [], [], []
    for t in range(n_tests):
        eps = rng.standard_normal(k) * np.sqrt(tau2)
        pt = 1 / (1 + np.exp(-(mu + eps)))
        y = rng.binomial(n, pt)
        z, v = effect.arm_logit(y, np.full(k, n))
        Z.append(z); V.append(v); G.append(np.full(k, t)); Y.append(y); N.append(np.full(k, n))
    return (np.concatenate(Z), np.concatenate(V), np.concatenate(G),
            np.concatenate(Y), np.concatenate(N))


def test_helmert():
    for k in (2, 3, 5, 9, 14):
        C = effect.helmert(k)
        assert C.shape == (k - 1, k)
        assert np.allclose(C @ C.T, np.eye(k - 1), atol=1e-12), "行需单位正交"
        assert np.allclose(C @ np.ones(k), 0, atol=1e-12), "行需与全 1 正交"
    print("  test_helmert OK")


def test_arm_logit():
    z, v = effect.arm_logit(np.array([0, 1, 50, 3000]), np.array([3000, 3000, 3000, 3000]))
    assert np.all(np.isfinite(z)) and np.all(np.isfinite(v))
    assert z[0] < z[1] < z[2] < z[3], "logit 应单调"
    assert np.all(v > 0)
    print("  test_arm_logit OK")


def test_tau2_recovery():
    tau2_true = 0.05
    z, v, g, _, _ = _simulate(tau2=tau2_true, n_tests=1500)
    mom = effect.tau2_moment(z, v, g)["tau2"]
    reml = effect.tau2_reml(z, v, g)["tau2"]
    assert abs(mom - tau2_true) < 0.25 * tau2_true, f"MoM 偏差过大: {mom:.4f}"
    assert abs(reml - tau2_true) < 0.25 * tau2_true, f"REML 偏差过大: {reml:.4f}"
    # tau2 = 0 时不应系统性估出正的大值
    z0, v0, g0, _, _ = _simulate(tau2=0.0, n_tests=1500)
    mom0 = effect.tau2_moment(z0, v0, g0)["tau2"]
    reml0 = effect.tau2_reml(z0, v0, g0)["tau2"]
    assert mom0 < 0.01 and reml0 < 0.01, f"零效应下高估: {mom0:.4f}, {reml0:.4f}"
    print(f"  test_tau2_recovery OK  (MoM={mom:.4f}, REML={reml:.4f}, 真值={tau2_true}; "
          f"零效应下 {mom0:.5f}/{reml0:.5f})")


def test_gh_matches_mc():
    rng = np.random.default_rng(0)
    m = rng.normal(-4, 0.3, 6)
    s = rng.uniform(0.05, 0.3, 6)
    gh = posterior.gh_expect_logistic(m, s)
    mc = effect.expit(m[None, :] + rng.standard_normal((400000, 6)) * s[None, :]).mean(axis=0)
    assert np.allclose(gh, mc, atol=3e-4), f"GH 与 MC 不一致: {gh} vs {mc}"
    print(f"  test_gh_matches_mc OK  (最大偏差 {np.abs(gh - mc).max():.2e})")


def test_posterior_consistency():
    z, v, g, y, n = _simulate(n_tests=60)
    for t in np.unique(g)[:60]:
        m_ = g == t
        post = posterior.eb_test_posterior(z[m_], v[m_], 0.05, n_draws=8000, rng=t)
        assert post["emax"] >= post["p_mean"].max() - 1e-12, "E[max p] 必须 >= max E[p]"
        assert abs(post["p_best"].sum() - 1) < 1e-12, "p_best 需归一"
        assert np.all(np.diff(post["p_mean"][np.argsort(post["m"])]) >= -1e-12), "后验均值应单调于 m"
        # 配对遗憾估计：非负
        r = posterior.regret_from_draws(post, int(np.argmax(post["p_mean"])))
        assert r >= -1e-9, f"遗憾不应为负: {r}"
    # tau2 -> 0：真实效应不存在，所有臂的最佳估计都退化为实验内合并点击率
    t = 0
    m_ = g == t
    post0 = posterior.eb_test_posterior(z[m_], v[m_], 1e-10, n_draws=500)
    pooled = effect.expit((z[m_] / v[m_]).sum() / (1.0 / v[m_]).sum())
    assert np.allclose(post0["p_mean"], pooled, atol=2e-3), "tau2->0 应收缩到实验均值"
    assert post0["p_mean"].max() - post0["p_mean"].min() < 1e-3
    # tau2 -> inf：先验不提供任何信息，后验均值回到原始 logit 变换
    post1 = posterior.eb_test_posterior(z[m_], v[m_], 1e8, n_draws=500)
    assert np.allclose(post1["p_mean"], effect.expit(z[m_]), atol=2e-3), "tau2->inf 应不收缩"
    print("  test_posterior_consistency OK")


def test_srm():
    ok = srm.chi2_srm([2500, 2480, 2520, 2500])
    bad = srm.chi2_srm([5000, 1000, 1000, 1000])
    assert ok["p"] > 0.05 and bad["p"] < 1e-10
    assert bad["cramer_v"] > ok["cramer_v"]
    assert np.isnan(srm.chi2_srm([10])["p"]) and np.isnan(srm.chi2_srm([1, 0])["p"])
    print(f"  test_srm OK  (均衡 p={ok['p']:.3f}, 失衡 p={bad['p']:.2e}, V={bad['cramer_v']:.3f})")


def test_dispersion_and_coverage():
    rng = np.random.default_rng(0)
    r = rng.standard_normal(100000)
    phi = srm.dispersion_from_residuals(r)
    assert abs(phi - 1) < 0.02, f"标准正态残差 phi 应约为 1，实际 {phi:.3f}"
    assert abs(srm.nominal_coverage(1.0) - 0.95) < 1e-9
    assert abs(srm.nominal_coverage(1.6) - 0.879) < 0.01, srm.nominal_coverage(1.6)
    print(f"  test_dispersion_and_coverage OK  (phi={phi:.3f}, phi=1.6 时覆盖率={srm.nominal_coverage(1.6):.3f})")


def test_decision_rules():
    # 明显胜出：回归"上线"
    y = np.array([200, 100, 95]); n = np.array([2000, 2000, 2000])
    post = posterior.eb_test_posterior(*effect.arm_logit(y, n), tau2=0.05, n_draws=3000, rng=0)
    assert decision.rule_argmax_observed(y, n) == 0
    assert decision.rule_significance_gated(y, n, alpha=0.05) == 0
    assert decision.rule_eb_argmax(post) == 0
    assert decision.rule_pbest_threshold(post, kappa=0.8) == 0
    # 三臂几乎无差异：显著性门控应退回
    y2 = np.array([100, 101, 99]); n2 = np.array([2000, 2000, 2000])
    assert decision.rule_significance_gated(y2, n2, alpha=0.05) == -1
    post2 = posterior.eb_test_posterior(*effect.arm_logit(y2, n2), tau2=0.05, n_draws=3000, rng=0)
    assert decision.rule_pbest_threshold(post2, kappa=0.8) == -1, "赢家不确定时不应上线"
    print("  test_decision_rules OK")


def test_power_consistency():
    p = 0.015
    n = power.required_n_per_arm(p, 0.20, power=0.8)
    assert abs(power.power_two_prop(n, p, 0.20) - 0.80) < 0.01, power.power_two_prop(n, p, 0.20)
    assert power.required_n_per_arm(p, 0.10) > power.required_n_per_arm(p, 0.20)
    r = power.selection_accuracy(0.05, p, 4, 3000, n_sim=4000)
    assert 0.5 < r["P_correct"] < 0.95 and r["E_regret"] > 0
    print(f"  test_power_consistency OK  (20% 提升需 n={n:,.0f}/臂；k=4 时选对率 {r['P_correct']:.3f})")


if __name__ == "__main__":
    print("exptdecide 单元测试")
    for fn in [test_helmert, test_arm_logit, test_tau2_recovery, test_gh_matches_mc,
               test_posterior_consistency, test_srm, test_dispersion_and_coverage,
               test_decision_rules, test_power_consistency]:
        fn()
    print("全部通过 ✔")
