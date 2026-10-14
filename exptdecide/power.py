"""
样本量规划与"实验是否具备结论能力"的判定。

在线实验最常见的失败不是分析做错了，而是**样本量根本不够**：在 1.5% 的
基础点击率下、每臂三千次曝光，两臂差异的标准误约 0.3 个百分点，而臂间真实
差异的中位数只有零点几个百分点 —— 实验从设计上就不可能有结论。
"""

from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = ["mde_relative", "power_two_prop", "required_n_per_arm",
           "selection_accuracy", "noise_only_spread"]


def mde_relative(n_arm, p, alpha=0.05, power=0.8):
    """给定每臂曝光 n 与基础点击率 p，可检出的**最小相对提升**（两臂比较）。"""
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    delta_abs = (z_a + z_b) * np.sqrt(2.0 * p * (1 - p) / np.maximum(n_arm, 1))
    out = delta_abs / p
    return out.item() if np.ndim(out) == 0 else out


def power_two_prop(n_arm, p, delta_rel, alpha=0.05):
    """两臂等样本量、双侧检验的检验功效。"""
    p1 = p * (1 + delta_rel)
    delta_abs = p1 - p
    se = np.sqrt(2.0 * p * (1 - p) / np.maximum(n_arm, 1))
    z_a = stats.norm.ppf(1 - alpha / 2)
    out = stats.norm.sf(z_a - delta_abs / se) + stats.norm.cdf(-z_a - delta_abs / se)
    return out.item() if np.ndim(out) == 0 else out


def required_n_per_arm(p, delta_rel, alpha=0.05, power=0.8):
    """要检出相对提升 delta_rel 所需的每臂曝光量。"""
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    delta_abs = p * delta_rel
    if delta_abs <= 0:
        return np.inf
    return float(2.0 * p * (1 - p) * (z_a + z_b) ** 2 / delta_abs ** 2)


def selection_accuracy(tau2, p_base, k, n_arm, n_sim=20000, rng=None):
    """在模型下模拟：'选观察点击率最高的臂' 能选中真实最优臂的概率。

    同时返回该规则的期望遗憾（相对真实最优臂的点击率损失）。
    """
    rng = np.random.default_rng(0 if rng is None else rng)
    mu = np.log(p_base / (1 - p_base))
    eps = rng.standard_normal((n_sim, k)) * np.sqrt(tau2)
    p_true = 1.0 / (1.0 + np.exp(-(mu + eps)))
    y = rng.binomial(n_arm, p_true)
    phat = y / n_arm
    pick = np.argmax(phat, axis=1)
    rows = np.arange(n_sim)
    regret = p_true[rows, np.argmax(p_true, axis=1)] - p_true[rows, pick]
    return {
        "P_correct": float((pick == np.argmax(p_true, axis=1)).mean()),
        "E_regret": float(regret.mean()),
        "E_regret_pct_of_base": float(regret.mean() / p_base * 100),
    }


def noise_only_spread(p, n_arm, k, q=0.5, n_sim=20000, rng=None):
    """**纯噪声基准**：若所有臂真实点击率完全相同，观察到的极差分布。

    把实测的臂间极差与这个基准对照，可以直接回答"实验里看到的差距到底是
    文案差异还是抽样噪声"。
    """
    rng = np.random.default_rng(1 if rng is None else rng)
    y = rng.binomial(n_arm, p, size=(n_sim, k))
    spread = (y / n_arm).max(axis=1) - (y / n_arm).min(axis=1)
    return {"mean": float(spread.mean()), "median": float(np.median(spread)),
            "q": float(np.quantile(spread, q)), "q25": float(np.quantile(spread, .25)),
            "q75": float(np.quantile(spread, .75))}
