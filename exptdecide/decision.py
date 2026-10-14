"""
上线决策规则：从实验数据到"选哪个方案上线 / 是否上线"。

四类规则覆盖了业界从最朴素到决策论最优的做法：

R1 argmax_observed      观察点击率最高者上线（业界默认做法，也是本档案平台的实际
                        行为：其"first_place"标记与臂级 CTR 最大者的吻合率为 88.5%）
R2 significance_gated   R1 + 显著性门控（top vs 次优 差值不显著就不上线）
R3 eb_argmax            后验均值最高者上线（用 tau^2 收缩掉部分噪声后排序）
R4 pbest_threshold      后验"最优臂概率"超过阈值 kappa 才上线，否则退回再测
R5 voi_threshold        值信息（Value of Information）阈值：把"再测一轮的成本 c"
                        显式写进目标函数，得到随成本比变化的最优上线门槛

所有规则的产出都是臂下标，或 -1 表示"本次不上线 / 退回再测"。
"""

from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = ["ctr", "two_proportion_z", "rule_argmax_observed", "rule_significance_gated",
           "rule_eb_argmax", "rule_pbest_threshold", "rule_voi_threshold", "RULES"]


def ctr(y, n):
    y = np.asarray(y, float)
    n = np.asarray(n, float)
    return y / np.maximum(n, 1.0)


def two_proportion_z(y1, n1, y0, n0):
    """两臂点击率差异的 z 统计量（合并方差）。"""
    p1, p0 = y1 / max(n1, 1.0), y0 / max(n0, 1.0)
    p = (y1 + y0) / max(n1 + n0, 1.0)
    se = np.sqrt(p * (1 - p) * (1.0 / max(n1, 1.0) + 1.0 / max(n0, 1.0)))
    if se <= 0:
        return 0.0
    return float((p1 - p0) / se)


def rule_argmax_observed(y, n, **kw):
    """R1：选观察点击率最高的臂。"""
    return int(np.argmax(ctr(y, n)))


def rule_significance_gated(y, n, alpha=0.05, **kw):
    """R2：top 与次优的差异在 alpha 水平上显著才上线，否则退回。"""
    p = ctr(y, n)
    order = np.argsort(-p)
    top, second = int(order[0]), int(order[1])
    z = two_proportion_z(y[top], n[top], y[second], n[second])
    return top if stats.norm.sf(abs(z)) < alpha else -1


def rule_eb_argmax(post, **kw):
    """R3：后验均值 E[p_i|data] 最高者上线。"""
    return int(np.argmax(post["p_mean"]))


def rule_pbest_threshold(post, kappa=0.8, **kw):
    """R4：最优臂后验概率 >= kappa 才上线。"""
    j = int(np.argmax(post["p_best"]))
    return j if post["p_best"][j] >= kappa else -1


def rule_voi_threshold(post, emax_prior, c, **kw):
    """R5：值信息阈值。

    立即上线的收益 = max_i E[p_i | data]；
    退回再测一轮的收益 = E[max_i p_i | 先验] - c（c 为再测一轮的等效点击率成本）。
    仅当上线的后验收益超过再测的期望收益时才上线。
    """
    j = int(np.argmax(post["p_mean"]))
    return j if post["p_mean"][j] > emax_prior - c else -1


RULES = {
    "R1_argmax_observed": rule_argmax_observed,
    "R2_significance_gated": rule_significance_gated,
    "R3_eb_argmax": rule_eb_argmax,
    "R4_pbest_threshold": rule_pbest_threshold,
    "R5_voi_threshold": rule_voi_threshold,
}
