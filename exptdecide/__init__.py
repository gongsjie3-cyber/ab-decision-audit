"""
exptdecide — 在线实验（A/B 测试）的统计审计与决策引擎
=====================================================

围绕"从实验数据到上线决策"这条链路，提供四个可复用组件：

1. `srm`        —— 样本比例失衡（Sample Ratio Mismatch）检验与过度离散诊断
2. `effect`     —— 臂级效应量的方差稳定化、组内经验贝叶斯先验方差 tau^2 的估计
3. `posterior`  —— 精确高斯经验贝叶斯后验：P(某臂最优)、E[max p]、期望遗憾
4. `decision`   —— 四类上线决策规则（argmax / 显著性门控 / EB 收缩 / 成本最优阈值）
5. `power`      —— 样本量规划与"实验是否具备结论能力"的判定

设计原则
--------
* 只依赖 numpy / scipy / pandas，无黑盒；
* 所有先验估计都能被独立复算（方法矩 + 限制极大似然双估计）；
* 后验使用解析高斯共轭结构，不使用 MCMC，结果可复现（固定随机种子）。

参考文献
--------
* Matias et al. (2021) The Upworthy Research Archive. Scientific Data 8:155.
* Fabijan et al. (2019) Diagnosing Sample Ratio Mismatch in Online Controlled
  Experiments. KDD '19.
* Coey & Hung (2022) Empirical Bayes Selection for Value Maximization. arXiv:2210.03905.
* Efron (2010) Large-Scale Inference.
"""

from . import srm, effect, posterior, decision, power  # noqa: F401

__version__ = "1.0.0"
__all__ = ["srm", "effect", "posterior", "decision", "power"]
