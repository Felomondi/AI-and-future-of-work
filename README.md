# AI, Gender, and the Future of Work

Predicting Occupational Exposure to Artificial Intelligence and analyzing its equity implications across gender and racial demographics.

**Authors:** Felix Omondi & Ezekiel Ekanem

## Overview

This project investigates two questions:

- **RQ1:** Can we predict an occupation's level of AI automation exposure based on its task characteristics, skill requirements, and work context features?
- **RQ2:** Does the demographic composition of an occupation (gender and race) significantly predict AI exposure after controlling for job tasks and skills?

We use the Felten et al. (2021) AI Occupational Exposure (AIOE) index as our target, with O*NET task features as predictors and BLS demographic and wage data for the equity analysis.

## Key Findings

- Lasso (L1) regression and XGBoost both predict AIOE at test R² ≈ 0.96. Physical abilities (Extent Flexibility, Multilimb Coordination) are the strongest protective factors; cognitive-textual skills (Written Comprehension, Memorization) are the strongest risk factors.
- Bivariate correlations show meaningful demographic differences in AIOE: women face elevated exposure to language-modeling AI (LMOE r = +0.18), Hispanic workers face the largest protective effect (matched-sample r = −0.61), Asian workers show modestly elevated exposure, and Black workers show a modest protective pattern.
- Both Baron-Kenny mediation and Double Machine Learning (DML) confirm that the demographic–AIOE relationships are almost entirely mediated by occupational task structure. Once tasks are controlled for, residual demographic effects are statistically indistinguishable from zero.


## Data Sources

- **O*NET Database (v30.2):** Skills, abilities, work activities, work context features
- **Felten et al. AIOE Index (2021):** General AIOE, Language Modeling AIOE (LMOE), Image Generation AIOE (IGOE)
- **BLS Current Population Survey (2025):** Occupational demographic shares
- **BLS Occupational Employment and Wage Statistics (2024):** Wages and employment counts

All datasets are linked at the 6-digit Standard Occupational Classification (SOC) code level.

## Methods

**RQ1 (predictive modeling):** OLS baseline → Lasso & Ridge regression → Random Forest & XGBoost → SHAP for feature interpretation → UMAP for visual structure.

**RQ2 (demographic analysis):** Pearson/Spearman correlations → employment-weighted exposure → regression with wage controls → LMOE/IGOE disaggregation → Baron-Kenny mediation → Double Machine Learning robustness check.

## Reproducing the Analysis

### Requirements

```
pandas, numpy, scikit-learn, xgboost, shap, umap-learn,
matplotlib, seaborn, scipy, doubleml
```
