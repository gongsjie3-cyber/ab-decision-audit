# 简历描述

> 用法建议：挑 1 条项目名 + 3～4 条 bullet。中文版用于中文简历，英文版用于英文简历/申请材料。
> 加粗处是最能体现你个人判断力的部分，面试时优先展开这些点。

---

## 中文版（推荐，4 条 bullet）

**在线实验决策审计：基于 9,744 个真实 A/B 实验的随机化失衡、赢家诅咒与可迁移性研究** ｜ 个人独立项目 ｜ Python

- 独立构建覆盖**全部 32,485 个真实随机实验**、150,817 个实验臂、5.382 亿次曝光的端到端审计管线（原始计数 → 分层建模 → 决策评估），并沉淀为可复用 Python 包 `exptdecide`（含 9 组单元测试）。
- 用多项分布卡方检验 + **无先验的似然比区间扫描**，仅从曝光分配数据就自行定位出平台历史上一次未被发现的流量分流故障窗口 `2013-06-24 ~ 2014-01-12`，与官方 2024 年事后披露的 `2013-06-25 ~ 2014-01-10` 相比**精确率 98.5%、召回率 100%**，并复现出"按上线小时波动"的缓存故障特征。
- 设计并实现**四条互相独立的 τ² 估计路径**（对照方法矩 / REML / DerSimonian–Laird / 跨实验重复投放协方差），定量刻画赢家诅咒：平台汇报的"最优方案"提升 **+32.6%** 高于事后可知的真实上限 **+29.4%**，被选中方案的真实提升仅 **+24.2%**。
- **采用注册报告式的分析纪律**：在只使用 9,744 个实验的前提下完成全部结论与代码并冻结，随后才下载 22,741 个从未接触过的实验做纯样本外确认（不重新估参），**13 项关键指标 12 项复现**——失衡率 15.52% vs 15.21%、留一臂预测覆盖率 0.9413 vs 0.9408、赢家诅咒 +32.52% vs +32.56%、数据自行检出的故障窗口逐日相同。
- **提出并验证「优势可迁移性」这一新视角**：利用 265 个被重复投放的素材做样本外复现，发现同一素材换一个实验投放后只有 **15.5%** 的优势可复现（样本外扩到 **1,920 对**后为 12.9%），预测下一次投放的最优收缩系数为 **0.084**（95% 区间 [0.071, 0.093]，仅为分层模型隐含值 0.75 的 1/9）；据此指出"经验贝叶斯收缩能改进方案选择"在**选谁**这个维度上无效（规则选择一致率 99.84%），真正价值在于识别出 **28.7%** 的实验其实无法区分赢家。

## 中文版（精简，2 条 bullet，适合空间紧张时）

**在线实验决策审计（个人项目）** ｜ Python ｜ [GitHub]

- 基于全部 32,485 个真实 A/B 实验（5.382 亿次曝光）构建审计管线，其中 22,741 个实验用于冻结后的样本外确认（13 项指标 12 项复现）：用卡方检验 + 无先验窗口扫描独立复现出平台的随机化失衡故障期，与官方事后披露窗口相比精确率 98.5%、召回率 100%；量化赢家诅咒（平台汇报提升 +32.6% > 真实上限 +29.4%）。
- 用 265 个被重复投放的素材做样本外验证，发现实验内的"最优文案"优势只有 **15.5%** 能迁移到下一次投放；据此给出上线决策建议：收缩不改变"选谁"（一致率 99.84%），但能识别出 28.7% 无法区分赢家的实验，应把"上线"与"宣布胜利"分离。

---

## English version

**Decision Audit of Online Experiments: Randomization Integrity, Winner's Curse and Transferability across 9,744 Real A/B Tests** | Independent project | Python

- Built an end-to-end audit pipeline over **all 32,485 real randomized experiments / 150,817 arms / 538.2M impressions**, from raw exposure-click counts to deployment decisions; shipped it as a reusable Python package (`exptdecide`) with 9 unit tests. Train and holdout subsets have **zero overlapping experiment IDs**.
- Recovered a previously undetected traffic-split failure window (`2013-06-24 → 2014-01-12`) **purely from impression-allocation data**, using chi-square SRM tests plus a prior-free likelihood-ratio interval scan; against the window disclosed by the archive team in 2024 this achieves **98.5% precision / 100% recall**, and reproduces the hour-of-day cache signature (SD 0.143 inside vs 0.005 outside).
- Implemented **four independent estimators of the between-arm variance τ²** (contrast method-of-moments, REML, DerSimonian–Laird, and cross-experiment repeated-measurement covariance) to quantify the winner's curse: the lift a platform would report (**+32.6%**) exceeds even the ex-post attainable optimum (**+29.4%**), while the selected arm's true lift is only **+24.2%**.
- **Adopted a registered-report discipline**: froze all conclusions and code using only 9,744 experiments, then downloaded 22,741 untouched experiments for a pure out-of-sample confirmation with no re-estimation — **12 of 13 key metrics replicated** (SRM rate 15.52% vs 15.21%; leave-one-arm coverage 0.9413 vs 0.9408; winner's-curse lift +32.52% vs +32.56%; the data-driven failure window matched to the day).
- **Introduced and validated a transferability lens**: using re-deployed stimuli across independent experiments (285 pairs in development, **1,920 pairs out-of-sample**), showed that only **12.9–15.5%** of an arm's within-test advantage replicates in a different test, and that the MSE-optimal shrinkage factor for predicting the next deployment is **0.084** (95% CI [0.071, 0.093]) versus the 0.75 implied by the hierarchical model — which shows empirically that "empirical-Bayes shrinkage fixes selection" changes nothing about *which* arm is chosen (99.84% identical choices) but does reveal that **28.7%** of experiments cannot distinguish a winner at all.

---

## 面试展开建议（可能会被追问的点）

| 追问 | 回答要点 |
|------|---------|
| 为什么不用常见的推荐/流失预测项目？ | 因为预测任务没有决策后果。这个项目把目标设成"上线决策的期望遗憾"，结论直接对应业务动作。 |
| 真值不可观测，你怎么评估？ | 三重防御：留一臂后验预测覆盖率 94.1% 校准后验方差；τ² ±2 倍敏感性分析下规则排名不变；并用完全模型外的重复投放协方差做独立验证。 |
| 15.5% 可迁移这个数字可信吗？ | 三种粒度（素材/slug/标题，选择机制不同）分别给 15.5% / 10.6% / 7.8%，方向一致；标题级用 4,812 对样本把 Cov 的 95% 区间收到 [0.0042, 0.0072]。我也披露了重复投放子样本的选择偏差（平均 CTR 2.76% vs 1.55%）。 |
| 这个「失衡窗口」是你发现的吗？ | 不是。官方团队 2024 年通过外部证据披露过。我的贡献是**不借助任何外部信息、用纯统计方法独立复现并定位到天**，而且在两个独立数据集上（合计 32,485 个实验）得到逐日相同的窗口。 |
| 你怎么保证不是「看着数据编故事」？ | 开发用的 9,744 个实验与确认用的 22,741 个实验在时间上严格分开：结论、代码、图表全部写完并冻结（有文件时间戳）之后才下载确认集，且确认阶段不重新估计任何参数。这是注册报告的做法。 |
| 最有价值的结论是什么？ | 两条反直觉的：① "用经验贝叶斯修正赢家诅咒"在选谁这个维度上是无效的，它只是近似单调变换；② 行业里"检测到 SRM 就丢弃实验"的做法，在这份数据上找不到经验支持，代价是 15.2% 的实验和 2,520 万次曝光。 |
