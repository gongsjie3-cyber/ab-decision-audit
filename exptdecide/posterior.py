"""
精确高斯经验贝叶斯后验。

给定先验方差 tau^2 和实验内各臂的 (z_ti, v_ti)，求 (mu_t, eps_t) 的联合后验。
因为模型在 logit 尺度上是线性高斯，后验有解析解，不需要 MCMC：

    z = mu * 1 + eps + e ,  eps ~ N(0, tau^2 I) ,  e ~ N(0, diag(v))
    先验精度:  diag(0, (1/tau^2) I)          # mu 用平坦先验
    似然精度:  X' V^{-1} X ,  X = [1, I]
    后验:      N( (X'V^{-1}X + P)^{-1} X'V^{-1} z ,  (X'V^{-1}X + P)^{-1} )

得到 theta_i = mu + eps_i 的后验均值 m_i 与协方差 Sigma。
再做两件事（都对 logit 尺度做，最后映射回点击率尺度）：
  * E[p_i | data]  —— 用 Gauss–Hermite 求 E[logistic(theta_i)]，不近似为 logistic(E[theta_i])
  * P(arm i 最优) 与 E[max_i p_i] —— 从 N(m, Sigma) 抽样，保留臂间相关性
    （臂间相关性来自共享的 mu，忽略它会把"最优臂概率"算得过自信）

E[max_i p_i] 是本项目的核心评价量：它给出"这次实验里最好的方案"的事后期望点击率，
任何决策规则的期望遗憾 = E[max p] - E[p_chosen]。该量只依赖后验，不依赖任何
不可观测的真值。
"""

from __future__ import annotations

import numpy as np

from .effect import expit

__all__ = ["eb_test_posterior", "gh_expect_logistic", "posterior_summary"]

# Gauss–Hermite 节点（20 点足够：logistic 光滑、后验近似正态）
_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(24)
_GH_W = _GH_W / _GH_W.sum()


def gh_expect_logistic(m, s):
    """E[logistic(theta)], theta ~ N(m, s^2)，用 Gauss–Hermite 积分。

    `s` 是**标准差**（不是方差）。hermegauss 给出的是关于权重 exp(-x^2/2) 的
    节点，故取样点 theta = m + s * x，再对权重求和即得期望。
    """
    m = np.asarray(m, float)
    s = np.asarray(s, float)
    if np.all(s <= 0):
        return expit(m)
    x = _GH_X[None, :] * np.maximum(s, 0.0)[:, None] + m[:, None]
    return (expit(x) * _GH_W[None, :]).sum(axis=1)


def eb_test_posterior(z, v, tau2, n_draws=2000, rng=None):
    """单个实验的经验贝叶斯后验。

    Parameters
    ----------
    z, v : (k,) array
        臂级 logit 观测与抽样方差。
    tau2 : float
        组内先验方差。
    n_draws : int
        计算 P(最优)/E[max p] 时的后验抽样次数。

    Returns
    -------
    dict
        m       : (k,) 后验均值（logit 尺度）
        sd      : (k,) 后验标准差（logit 尺度）
        Sigma   : (k,k) 后验协方差（logit 尺度）
        p_mean  : (k,) E[p_i | data]（点击率尺度）
        p_best  : (k,) P(arm i 是最优臂 | data)
        emax    : float  E[max_i p_i | data]
        draws_p : (n_draws,k) 点击率尺度后验样本（可选复用）
    """
    k = z.size
    prec = 1.0 / v                                  # (k,)
    A = np.zeros((k + 1, k + 1))
    A[0, 0] = prec.sum()
    A[0, 1:] = prec
    A[1:, 0] = prec
    A[1:, 1:] = np.diag(prec)
    A[1:, 1:] += np.eye(k) / max(tau2, 1e-12)
    b = np.concatenate([[float((z * prec).sum())], z * prec])
    cov = np.linalg.inv(A)
    mean = cov @ b
    m = mean[0] + mean[1:]
    Sigma = (cov[0, 0]
             + cov[0, 1:][None, :]
             + cov[1:, 0][:, None]
             + cov[1:, 1:])
    Sigma = 0.5 * (Sigma + Sigma.T)

    sd = np.sqrt(np.clip(np.diag(Sigma), 0, None))
    p_mean = gh_expect_logistic(m, sd)

    rng = np.random.default_rng(0 if rng is None else rng)
    try:
        L = np.linalg.cholesky(Sigma + 1e-10 * np.eye(k))
    except np.linalg.LinAlgError:
        w, Q = np.linalg.eigh(Sigma)
        L = Q @ np.diag(np.sqrt(np.clip(w, 0, None)))
    theta = m[None, :] + rng.standard_normal((n_draws, k)) @ L.T
    P = expit(theta)
    best_idx = np.argmax(P, axis=1)
    p_best = np.bincount(best_idx, minlength=k) / n_draws
    emax_mc = float(P.max(axis=1).mean())
    # 数学上必有 E[max_i p_i] >= max_i E[p_i]。蒙特卡洛估计 E[max] 的标准误
    # 与这个差额同量级，因此把它投影回可行域（这是有偏但方差小得多的估计，
    # 且不会违反不等式）。原始 MC 值保留在 emax_mc 供审计。
    emax = max(emax_mc, float(p_mean.max()))
    return {"m": m, "sd": sd, "Sigma": Sigma, "p_mean": p_mean,
            "p_best": p_best, "emax": emax, "emax_mc": emax_mc, "draws_p": P,
            "mean_mu": float(mean[0]), "var_mu": float(cov[0, 0])}


def regret_from_draws(post, chosen):
    """配对估计的期望遗憾 E[max_i p_i - p_chosen | data]。

    用**同一批后验抽样**同时计算 max 与被选臂，两项高度正相关，方差远小于
    "E[max] - E[p_chosen]" 的差。chosen = -1 表示"拒绝上线"（用实验内平均
    点击率作为不改变现状的基准）。
    """
    P = post["draws_p"]
    mx = P.max(axis=1)
    if chosen is None or chosen < 0:
        base = post.get("p_reference", float(post["p_mean"].mean()))
        return float((mx - base).mean())
    return float((mx - P[:, chosen]).mean())


def posterior_summary(post, chosen=None):
    """把后验压成一行摘要，便于批量写表。"""
    out = {
        "p_mean_max": float(post["p_mean"].max()),
        "p_best_max": float(post["p_best"].max()),
        "emax": post["emax"],
        "emax_mc": post.get("emax_mc", post["emax"]),
        "p_best_entropy": float(-(np.log(np.clip(post["p_best"], 1e-12, None)) * post["p_best"]).sum()),
    }
    if chosen is not None:
        out["regret"] = regret_from_draws(post, chosen)
        out["is_pbest_argmax"] = int(np.argmax(post["p_best"]) == chosen) if chosen >= 0 else np.nan
    else:
        out["regret"] = np.nan
        out["is_pbest_argmax"] = np.nan
    return out
