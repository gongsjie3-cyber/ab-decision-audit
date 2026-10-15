# Decision Audit of Online Experiments

**English** · [简体中文](./README.zh-CN.md)

> An audit of **32,485 real A/B tests** in the Upworthy Research Archive — 150,815 experiment arms, 538.3M impressions (the full archive minus two single-arm experiments) — covering randomization integrity, the winner's curse, and cross-deployment transferability.
>
> 9,744 experiments were used for development and then **frozen**; the remaining **22,741 were kept untouched until after the freeze** and used for pure out-of-sample confirmation — **12 of 13 key metrics replicated**.

🌐 **[Project page](https://gongsjie3-cyber.github.io/ab-decision-audit/)** · 📄 **[Full technical report (PDF, Chinese)](reports/report.pdf)** · [Quick start](#quick-start)

[![unit tests](https://img.shields.io/badge/unit%20tests-9%20passed-brightgreen)](#quick-start)
[![data](https://img.shields.io/badge/data-CC%20BY%204.0-blue)](https://osf.io/jd64p/)
[![confirmatory](https://img.shields.io/badge/out--of--sample-12%2F13%20replicated-success)](#six-headline-findings)

---

## What this project is about

A/B testing is the primary causal-inference tool of the modern internet company, yet the decision that actually matters — **which variant do we ship?** — is rarely audited. This project targets that decision and asks three questions:

1. **Is the data even usable?** — sample-ratio-mismatch (SRM) auditing
2. **How real is the result?** — effect-size estimation, the winner's curse, out-of-sample model calibration
3. **What is the best decision rule?** — comparing deployment rules and data-quality policies under explicit loss

Everything is reproducible from public data in about a minute.

## Six headline findings

| # | Finding | Key numbers |
|---|---------|-------------|
| 1 | Randomization failures can be detected and **dated without any prior knowledge** | A prior-free likelihood-ratio scan over impression allocation alone recovers `2013-06-24 → 2014-01-12`; the archive team disclosed `2013-06-25 → 2014-01-10` in 2024 → **98.5% precision, 100% recall** |
| 2 | Real headline effects exist, and the model survives out-of-sample checks | τ² = 0.052; 90% interval for true CTR 1.04%–2.18%; leave-one-arm posterior predictive coverage **94.1%** (nominal 95%) |
| 3 | The winner's curse is quantifiable — and worse than intuition suggests | The lift a platform would report is **+32.6%**, higher even than the *ex-post attainable optimum* (+29.4%); the selected arm's true lift is only **+24.2%** |
| 4 | **Most of a "winning headline"'s advantage does not transfer** *(core contribution)* | Only **15.5%** of an arm's within-test advantage replicates when the same stimulus is re-deployed in a different experiment; the MSE-optimal shrinkage factor for predicting the next deployment is **0.14**, one-fifth of the 0.75 implied by the hierarchical model |
| 5 | Shrinkage does **not** change *which* arm you pick — it tells you *how sure you should be* | Empirical-Bayes selection and "take the highest CTR" agree in **99.84%** of experiments; yet in **28.7%** of experiments the "winner" is statistically indistinguishable from the rest (P(best) < 0.5) |
| 6 | **Every major conclusion replicates on 22,741 untouched experiments** | SRM rate 15.52% vs 15.21%; leave-one-arm coverage **0.9413 vs 0.9408**; winner's-curse lift +32.52% vs +32.56%; the data-driven failure window matched **to the day**; transferability pairs grew from 285 to **1,920** |

Two further results that translate directly into practice:

- **Human override is worse than the mechanical rule.** The archive's own `first_place` decision has **19% higher** expected regret than simply taking the highest CTR, giving up 3.4 percentage points of attainable value.
- **"Discard every experiment that fails an SRM check" is not supported by the evidence.** Four falsifiable tests (overdispersion, effect-size distribution, dose–response, decision quality) all fail to detect material harm in the flagged experiments — while discarding costs **6.5×** more than leaving them in.

## Technical highlights

- **Closed-form empirical Bayes.** The whole posterior is analytic (linear-Gaussian conjugacy on the logit scale): 9,744 experiments in ~3 seconds, no MCMC. `P(arm i is best)` is computed by sampling the *joint* posterior so that **between-arm correlation is preserved**, and `E[p | data]` uses Gauss–Hermite quadrature rather than the lazy `logistic(E[θ])`.
- **Four independent estimators of τ²**: contrast method-of-moments, REML, DerSimonian–Laird, and **cross-experiment repeated-measurement covariance** — the last one bypasses the hierarchical model entirely and is therefore a genuine external check.
- **Exact variance decomposition.** Helmert orthogonal contrasts give the identity `SS_within = τ²·dof + Σ(1−1/k)v`, splitting arm-level variation cleanly into "real headline differences, 62.8%" and "sampling noise, 37.2%".
- **Paired evaluation protocol.** Priors are estimated on the training split only; all rules within an experiment share the same posterior draws, so Monte-Carlo noise cancels out of rule comparisons; τ² is stress-tested over ±2×.
- **Registered-report-style discipline.** The confirmatory subset was never touched during development; it was downloaded only after every conclusion, figure and line of code had been frozen, and the confirmation step re-estimates nothing.
- **9 unit tests** covering the places where this kind of code actually breaks: contrast orthogonality, τ² recovery from simulated data, Gauss–Hermite vs Monte Carlo, the posterior invariant `E[max p] ≥ max E[p]`, and shrinkage behaviour in both limits.

## Quick start

```bash
python -m pip install -r requirements.txt

python scripts/fetch_data.py                  # download + sha256-verify the data (~95 MB)
python -m exptdecide.tests.test_exptdecide    # unit tests
python -m src.run_all                         # full pipeline, ~45 seconds
```

When the pipeline finishes, every headline metric is collected in `results/99_headline.json`, figures land in `figures/`, and the report is at `reports/report.pdf`.

> Rebuilding the Word version of the report needs the optional `python-docx` dependency:
> `pip install python-docx && python -m src.make_report_docx` (the committed PDF does not require it).

## Repository layout

```
ab-decision-audit/
├── exptdecide/                reusable Python package (the technical core)
│   ├── srm.py                 sample-ratio-mismatch tests, BH-FDR, overdispersion
│   ├── effect.py              variance stabilisation, two τ² estimators, exact decomposition
│   ├── posterior.py           analytic Gaussian empirical-Bayes posterior
│   ├── decision.py            five deployment decision rules
│   ├── power.py               sample-size planning, selection accuracy, noise benchmark
│   └── tests/                 unit tests
├── src/
│   ├── build_dataset.py       rebuild the experiment/arm dataset
│   ├── m1_audit.py            randomization-integrity audit (prior-free window scan)
│   ├── m2_effects_power.py    effect-size prior, power analysis, posterior calibration
│   ├── m3_winners_curse.py    winner's curse and cross-deployment transferability
│   ├── m4_decision_rules.py   rule comparison (train / holdout)
│   ├── m5_quality_audit.py    consequences of data-quality defects, policy evaluation
│   ├── m6_confirmatory.py     frozen-parameter out-of-sample confirmation
│   └── run_all.py             one-command re-run + summary
├── scripts/fetch_data.py      data acquisition with sha256 verification
├── results/                   all result tables and JSONs
├── figures/                   13 figures
└── reports/                   technical report (PDF / Markdown)
```

## Data

From the [Upworthy Research Archive](https://osf.io/jd64p/) (CC BY 4.0). All three subsets are public — no application required — and together they account for **all 32,487 experiments** the archive advertises. After dropping single-arm tests and zero-impression rows, the pipeline analyses **32,485 experiments / 150,815 arms / 538.3M impressions**:

| Subset | Experiments | Arms | Impressions | Role here |
|--------|------------|------|-------------|-----------|
| exploratory | 4,873 | 22,666 | 81.0M | training (prior estimation, tuning) |
| holdout | 4,871 | 22,600 | 80.7M | testing (rule evaluation, zero ID overlap with training) |
| confirmatory | 22,743 | 105,551 | 376.5M | **pure out-of-sample confirmation** |

**Analysis discipline.** Modules 1-5 were completed and frozen using only the exploratory and holdout subsets. The confirmatory subset was downloaded *after* all conclusions were written, and the confirmation module re-estimates nothing. Timestamps documenting this ordering are recorded in section 9 of the technical report — the design is methodologically equivalent to a registered report.

## Citation

If you use this work, please cite both the underlying archive and this repository (see `CITATION.cff`).

```bibtex
@article{matias2021upworthy,
  title   = {The Upworthy Research Archive, a time series of 32,487 experiments in U.S. media},
  author  = {Matias, J. Nathan and Munger, Kevin and Aubin Le Quere, Marianne and Ebersole, Charles},
  journal = {Scientific Data},
  volume  = {8},
  number  = {1},
  pages   = {155},
  year    = {2021}
}
```

Released under the MIT License (code and documentation only; the data remains CC BY 4.0 — see [NOTICE.md](./NOTICE.md)).
