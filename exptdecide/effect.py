"""
臂级效应量的方差稳定化与组内经验贝叶斯先验方差 tau^2 的估计。

模型
----
对第 t 个实验的第 i 个实验臂，观测到 n_ti 次曝光、y_ti 次点击：

    y_ti | p_ti ~ Binomial(n_ti, p_ti)
    logit(p_ti) = mu_t + eps_ti ,   eps_ti ~ N(0, tau^2)

mu_t 是实验 t 的基线水平（讨厌参数），tau^2 是**同一实验内不同文案之间
真实点击率差异**的方差——它是整个项目最关键的一个数：tau^2 越大，说明
"换标题"这件事越有真实价值，"赢家诅咒"越轻；tau^2 越小，说明实验观测到的
臂间差距大部分是噪声。

估计方法（两条独立路径，互为校验）
----------------------------------
1. `tau2_moment`：DerSimonian–Laird 型方法矩估计（组内中心化）。
2. `tau2_reml`  ：用 Helmert 正交对照把 mu_t 精确消掉，对 tau^2 做限制极大
   似然估计（REM L）。对照变换后 u_t = C_t z_t ~ N(0, tau^2 I + C_t diag(v_t) C_t')，
   是对角结构之和，似然可解析、可向量化，无需 MCMC。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize

__all__ = [
    "logit", "expit", "arm_logit", "helmert", "tau2_moment", "tau2_reml",
    "shrink_factor", "arm_table_from_counts",
]


def logit(p):
    p = np.asarray(p, dtype=float)
    return np.log(p / (1.0 - p))


def expit(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def arm_logit(y, n, cc=0.5, max_var=1e6):
    """带连续性校正的 logit 变换 + Delta 方法方差。

    z = log((y+cc)/(n-y+cc)),  v = 1/(y+cc) + 1/(n-y+cc)

    cc=0.5（Haldane 校正）保证 y=0 或 y=n 时数值可用；对 1.5% 左右的基础
    点击率、几千次曝光的臂，该变换的偏差远小于后续要估计的 tau。
    """
    y = np.asarray(y, dtype=float)
    n = np.asarray(n, dtype=float)
    z = np.log((y + cc) / (n - y + cc))
    v = 1.0 / (y + cc) + 1.0 / (n - y + cc)
    v = np.clip(v, 1e-12, max_var)
    return z, v


def helmert(k):
    """(k-1, k) 的 Helmert 正交对照矩阵：行单位正交且与全 1 向量正交。

    用它左乘 z_t 可以**精确**地消掉实验固定效应 mu_t。
    """
    C = np.zeros((k - 1, k))
    for i in range(1, k):
        C[i - 1, :i] = 1.0
        C[i - 1, i] = -float(i)
        C[i - 1] /= np.sqrt(i * (i + 1.0))
    return C


def _groups_from_ids(gids):
    """按出现顺序把 group id 转成 (codes, sizes, order)，假设同组相邻或返回索引。"""
    gids = np.asarray(gids)
    uniq, inv = np.unique(gids, return_inverse=True)
    return uniq, inv


def tau2_moment(z, v, group_ids):
    """DerSimonian–Laird 型方法矩估计（组内中心化）。

    Q   = sum_t sum_i w_ti (z_ti - zbar_t)^2 ,  w = 1/v
    dof = sum_t (k_t - 1)
    C   = sum_t (sum_i w_ti - sum_i w_ti^2 / sum_i w_ti)
    tau^2 = max(0, (Q - dof) / C)
    """
    z = np.asarray(z, float)
    v = np.asarray(v, float)
    w = 1.0 / v
    df = pd.DataFrame({"z": z, "w": w, "g": np.asarray(group_ids)})
    g = df.groupby("g", sort=False)
    sw = g["w"].transform("sum")
    zbar = (df["w"] * df["z"]).groupby(df["g"]).transform("sum") / sw
    Q = float((df["w"] * (df["z"] - zbar) ** 2).sum())
    sizes = g.size()
    dof = float((sizes - 1).sum())
    C = float((sw - (df["w"] ** 2).groupby(df["g"]).transform("sum") / sw).groupby(df["g"]).first().sum())
    tau2 = max(0.0, (Q - dof) / C) if C > 0 else 0.0
    return {"tau2": tau2, "Q": Q, "dof": dof, "C": C, "I2": max(0.0, 1 - dof / Q) if Q > 0 else 0.0}


def _loglik_contrast(lam, q2, grid):
    """sum_m sum_j [ -0.5*log(t + lam_mj) - 0.5*q2_mj/(t + lam_mj) ]，对 grid 向量化。"""
    t = grid[:, None, None]                 # (G,1,1)
    L = lam[None, :, :]                     # (1,m,j)
    Q2 = q2[None, :, :]
    ll = -0.5 * np.log(t + L) - 0.5 * Q2 / (t + L)
    return ll.sum(axis=(1, 2))


def tau2_reml(z, v, group_ids, tau_grid=None, refine=True):
    """用 Helmert 对照 + 限制极大似然估计 tau^2。

    返回 dict(tau2, loglik_curve, grid, se)。se 由观测信息量数值二阶差分给出。
    """
    z = np.asarray(z, float)
    v = np.asarray(v, float)
    gids = np.asarray(group_ids)
    uniq, inv = np.unique(gids, return_inverse=True)
    sizes = np.bincount(inv)

    if tau_grid is None:
        tau_grid = np.concatenate([[0.0], np.logspace(-5, 1.0, 400)])

    total_ll = np.zeros(tau_grid.size)
    for k in np.unique(sizes):
        m = int((sizes == k).sum())
        if k < 2 or m == 0:
            continue
        sel = np.where(sizes == k)[0]
        # 取该组所有实验的臂数据（同一 k，行数一致）
        idx = np.concatenate([np.where(inv == gi)[0] for gi in sel])
        Z = z[idx].reshape(m, k)
        V = v[idx].reshape(m, k)
        C = helmert(k)                                   # (k-1, k)
        U = Z @ C.T                                      # (m, k-1)
        S = np.einsum("ij,mj,lj->mil", C, V, C)          # (m, k-1, k-1)
        lam, Q = np.linalg.eigh(S)                       # lam (m,k-1) 升序
        lam = np.clip(lam, 1e-12, None)
        q = np.einsum("mij,mj->mi", Q, U)
        total_ll += _loglik_contrast(lam, q ** 2, tau_grid)

    best = int(np.argmax(total_ll))
    t_hat = float(tau_grid[best])
    if refine and 0 < best < tau_grid.size - 1:
        x = tau_grid[best - 1:best + 2]
        y = total_ll[best - 1:best + 2]
        try:
            p = np.polyfit(x, y, 2)
            if p[0] < 0:
                cand = -p[1] / (2 * p[0])
                if x[0] <= cand <= x[2]:
                    t_hat = float(cand)
        except Exception:
            pass
    # 观测信息量 -> 标准误（对 tau^2 的曲率）
    se = np.nan
    if 0 < best < tau_grid.size - 1:
        h = tau_grid[best + 1] - tau_grid[best]
        d2 = (total_ll[best + 1] - 2 * total_ll[best] + total_ll[best - 1]) / (h ** 2)
        if d2 < 0:
            se = float(1.0 / np.sqrt(-d2))
    return {"tau2": max(t_hat, 0.0), "tau2_raw": t_hat, "se": se,
            "grid": tau_grid, "loglik": total_ll}


def shrink_factor(v, tau2):
    """收缩因子 B = tau^2 / (tau^2 + v)：观测偏离被保留的比例。"""
    v = np.asarray(v, float)
    if tau2 <= 0:
        return np.zeros_like(v)
    return tau2 / (tau2 + v)


def within_test_ss(z, v, group_ids):
    """用 Helmert 对照把组内变异精确分解为"真实 + 噪声"两部分。

    对第 t 个实验的每个对照行 c（单位正交、与 1 正交）：
        c'z = c'eps + c'e ,  E[(c'z)^2] = tau^2 + sum_i c_i^2 v_i
    求和即得
        SS_within = tau^2 * dof + sum_i (1 - 1/k_i) * v_i
                      └ 真实信号 ┘   └── 抽样噪声 ──┘
    这是一个**精确恒等式**（不是近似），因此可以直接读出信号占比，
    也给出 tau^2 的另一个无偏估计。
    """
    z = np.asarray(z, float)
    v = np.asarray(v, float)
    gids = np.asarray(group_ids)
    uniq, inv = np.unique(gids, return_inverse=True)
    sizes = np.bincount(inv)
    ss = 0.0
    noise = 0.0
    dof = 0.0
    for k in np.unique(sizes):
        if k < 2:
            continue
        sel = np.where(sizes == k)[0]
        idx = np.concatenate([np.where(inv == gi)[0] for gi in sel])
        Z = z[idx].reshape(-1, k)
        V = v[idx].reshape(-1, k)
        C = helmert(k)
        U = Z @ C.T
        ss += float((U ** 2).sum())
        noise += float((V.sum(axis=1) * (1.0 - 1.0 / k)).sum())
        dof += float((k - 1) * Z.shape[0])
    return {"ss_within": ss, "noise_part": noise, "dof": dof,
            "tau2_implied": max(0.0, (ss - noise) / dof) if dof else np.nan}


def variance_decomposition(z, v, group_ids, tau2=None):
    """返回信号/噪声占比（基于精确恒等式）。tau2 缺省时用无偏估计。"""
    d = within_test_ss(z, v, group_ids)
    t2 = d["tau2_implied"] if tau2 is None else float(tau2)
    signal = t2 * d["dof"]
    out = dict(d)
    out["tau2_used"] = t2
    out["signal_part"] = signal
    out["signal_share"] = signal / d["ss_within"] if d["ss_within"] > 0 else np.nan
    out["noise_share"] = d["noise_part"] / d["ss_within"] if d["ss_within"] > 0 else np.nan
    return out


def arm_table_from_counts(n, y, group_ids):
    """把计数表转成 (z, v) 并附带基本诊断。"""
    z, v = arm_logit(y, n)
    return pd.DataFrame({"z": z, "v": v, "n": n, "y": y, "g": group_ids})
