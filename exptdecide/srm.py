"""
样本比例失衡（Sample Ratio Mismatch, SRM）检验与过度离散诊断。

在线实验里，如果各实验臂的**曝光量分配比例**显著偏离设计比例，通常意味着
分流链路（缓存、埋点丢失、机器人流量、代码回滚）出了问题，此时实验的因果
解释不成立。本模块只做"体检"，不做因果推断。

主要接口
--------
chi2_srm(n, expected_share=None)   -> dict  卡方/SRM 检验 + 效应量
bh_flag(pvals, q)                  -> ndarray  Benjamini-Hochberg FDR 控制
dispersion_from_residuals(resid)   -> float   准似然过度离散系数 phi
nominal_coverage(phi, level)       -> float   名义置信区间的实际覆盖率
"""

from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = ["chi2_srm", "bh_flag", "dispersion_from_residuals", "nominal_coverage"]


def chi2_srm(impressions, expected_share=None):
    """对单个实验的臂级曝光量做分配比例检验。

    H0: 各臂曝光量服从多项分布 Multinomial(N, w)，w 为设计分配比例（默认等分）。

    Parameters
    ----------
    impressions : array-like of int
        各臂曝光量。
    expected_share : array-like, optional
        设计分配比例，默认等分。

    Returns
    -------
    dict
        chi2  : 卡方统计量
        dof   : 自由度 k-1
        p     : p 值
        n     : 总曝光
        k     : 臂数
        cramer_v : Cramér's V（效应量，0=完全均衡）
        max_share : 最大臂曝光占比
        max_dev   : 相对设计比例的最大绝对偏离（百分点）
        ratio     : 最大臂/最小臂 曝光比
    """
    n = np.asarray(impressions, dtype=float)
    k = n.size
    N = n.sum()
    out = {
        "chi2": np.nan, "dof": max(k - 1, 0), "p": np.nan, "n": N, "k": k,
        "cramer_v": np.nan, "max_share": np.nan, "max_dev": np.nan, "ratio": np.nan,
    }
    if k < 2 or N <= 0:
        return out
    if expected_share is None:
        w = np.full(k, 1.0 / k)
    else:
        w = np.asarray(expected_share, dtype=float)
        w = w / w.sum()
    exp = N * w
    if np.any(exp <= 0):
        return out
    stat = float(((n - exp) ** 2 / exp).sum())
    # 期望频数过小时卡方近似不可靠，退回蒙特卡洛式保守处理：标记为不可判定
    p = float(stats.chi2.sf(stat, k - 1)) if exp.min() >= 5 else np.nan
    out.update(
        chi2=stat,
        p=p,
        cramer_v=float(np.sqrt(stat / (N * (k - 1)))),
        max_share=float(n.max() / N),
        max_dev=float(np.max(np.abs(n / N - w)) * 100),
        ratio=float(n.max() / max(n.min(), 1e-9)),
    )
    return out


def bh_flag(pvals, q=0.01):
    """Benjamini-Hochberg FDR 控制，返回被标记的布尔数组。

    对 9k+ 个实验同时做 SRM 检验需要多重比较控制，否则假阳性会淹没信号。
    """
    p = np.asarray(pvals, dtype=float)
    ok = np.isfinite(p)
    flag = np.zeros(p.size, dtype=bool)
    if ok.sum() == 0:
        return flag
    pv = p[ok]
    m = pv.size
    order = np.argsort(pv)
    thresh = q * (np.arange(1, m + 1)) / m
    passed = pv[order] <= thresh
    if passed.any():
        kmax = np.max(np.where(passed)[0])
        cutoff = pv[order][kmax]
        idx = np.where(ok)[0]
        flag[idx[pv <= cutoff]] = True
    return flag


def dispersion_from_residuals(residuals, n_params=0):
    """准似然过度离散系数 phi = sum(r^2) / (N - p)。

    phi ≈ 1 说明残差与二项分布假设一致；phi > 1 说明存在额外方差
    （例如按时间分块的缓存分流、同一用户重复曝光），此时名义标准误被低估
    约 sqrt(phi) 倍，置信区间过窄、p 值过小。
    """
    r = np.asarray(residuals, dtype=float)
    r = r[np.isfinite(r)]
    dof = r.size - n_params
    if dof <= 0:
        return np.nan
    return float((r ** 2).sum() / dof)


def nominal_coverage(phi, level=0.95):
    """在真实方差被放大 phi 倍时，名义 level 置信区间的**实际**覆盖率。

    实际标准误 = sqrt(phi) * 名义标准误，因此区间相当于 z / sqrt(phi) 分位。
    """
    if not np.isfinite(phi) or phi <= 0:
        return np.nan
    z = stats.norm.ppf(0.5 + level / 2.0)
    return float(2 * stats.norm.cdf(z / np.sqrt(phi)) - 1.0)
