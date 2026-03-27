# NBME CBSE Early Warning Tool

> **Predicting CBSE performance from organ-system block exams to enable early academic intervention in medical education.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Background

Medical students at U.S. allopathic schools take the **NBME Comprehensive Basic Science Examination (CBSE)** as a readiness check for USMLE Step 1. Students who score below the institutional threshold often require remediation, yet intervention typically happens *after* the CBSE — when time is limited.

**This project asks: Can we identify at-risk students *before* they take the CBSE, using only their organ-system block exam scores?**

## Key Findings

### 1. Block exam scores strongly predict CBSE performance

| Metric | Value | Confidence |
|--------|-------|------------|
| Model R-squared (CV) | 0.52 | HIGH (95% CI: 0.49 - 0.55) |
| Mean Absolute Error | 5.9 EPC | HIGH (95% CI: 5.8 - 6.2) |
| Pass/Fail Accuracy | 91.2% | — |
| Sensitivity (catch failures) | 88.4% | — |

> 22 of 23 individual block exams are statistically significant predictors of CBSE (p < 0.001).

### 2. Not all blocks predict equally

**Strongest predictors:** Neuroscience Final (r = 0.70), Repro Midterm (r = 0.67), MSK Final (r = 0.65)
**Weakest predictor:** Foundation Final (r = 0.28) — however, the Foundation *Formative NBME* exam (r = 0.48) is much more predictive, likely because it uses the same item format as the CBSE.

### 3. Early warning is possible after just 2 courses

| Courses Completed | Prediction Accuracy (R-squared) |
|---|---|
| Foundation only | 0.01 (not useful) |
| **+ MSK** | **0.33 (actionable)** |
| + GI + Heme + CV | 0.40 (peak) |
| All 11 courses | 0.28 (overfit - too many features) |

**Implication:** Schools can flag at-risk students as early as the second block — months before the CBSE.

### 4. Retaking CBSE without intervention does not improve scores

Among students who retook the CBSE, the average score change from attempt 1 to attempt 2 was **+0.8 EPC (p = 0.49, not significant)**. Simply retaking the exam is not enough — structured remediation before the retake is essential.

### 5. Midterm-to-final improvement does not predict CBSE success

Students who improved the most from midterm to final within blocks did **not** score higher on the CBSE (r = -0.16, p = 0.09). This is explained by regression to the mean: students with low midterms have more room to improve, but their absolute performance remains below peers. **Absolute scores matter more than improvement trends.**

## Architecture

```
NBME CBSE Early Warning Tool/
|
|-- main.py                          # Synthetic data demo pipeline
|-- src/
|   |-- generate_synthetic_data.py   # Generates realistic training data
|   |-- data_loader.py               # CSV ingestion and merging
|   |-- preprocessing.py             # Missing values, feature engineering
|   |-- models.py                    # Linear Regression, Random Forest, XGBoost
|   |-- feature_importance.py        # Coefficients, Permutation, SHAP
|   |-- early_warning.py             # Threshold-based risk flagging
|   |
|   |-- extract_real_data.py         # Real data ETL (block CSVs + CBSE PDFs)
|   |-- extract_enhanced.py          # Adds formative + remediation features
|   |-- run_real_model.py            # Model training on real data
|   |-- enhanced_model_and_atrisk.py # Enhanced model + at-risk list generation
|   |-- early_warning_timing.py      # When can we start predicting?
|   |-- retake_analysis.py           # CBSE retake score change analysis
|   |-- improvement_analysis.py      # Midterm->Final trend vs CBSE
|   |-- confidence_analysis.py       # Statistical confidence for all conclusions
|   |-- generate_presentation.py     # Publication-ready figures
|
|-- data/
|   |-- raw/                         # Synthetic CSVs (safe to share)
|   |-- real/                         # Real student data (gitignored, FERPA)
|
|-- output/                          # Synthetic data results
|-- docs/
|   |-- learning_notes.md            # ML/DS concepts with Healthcare AI PM annotations
|
|-- notebooks/
|   |-- 01_data_exploration.ipynb    # Interactive EDA
```

## Quick Start (Synthetic Data Demo)

```bash
pip install -r requirements.txt
python main.py
```

Generates synthetic medical student data, trains 3 models, produces feature importance analysis, and flags at-risk students.

## Methodology

### Data Pipeline

1. **Ingestion**: 11 organ-system block CSVs (Canvas grade exports) + 3 CBSE roster PDFs (NBME reports)
2. **Feature extraction**: Summative exam Final Scores (midterm + end-of-course), formative NBME scores, remediation flags
3. **Merging**: Join on student ID across all data sources; identify students with missing CBSE (potential selection bias)

### Feature Engineering

| Feature | Rationale |
|---------|-----------|
| Individual exam scores (23) | Direct performance measures per block |
| Formative NBME scores (3) | Same item format as CBSE; captures NBME-specific reasoning ability |
| Remediation flags (6) | Prior academic difficulty is a risk signal |
| Block average | Overall ability level |
| Block variance | Consistency across systems (high variance = weak spots) |
| Lowest exam score | "Weakest link" effect |
| Midterm-to-final change | Study adaptation signal (found to be non-predictive) |

### Models

| Model | CV R-squared | Train-Test Gap | Notes |
|-------|-------------|----------------|-------|
| Ridge Regression | 0.49 | 0.04 (OK) | Best stability, interpretable |
| Random Forest (depth=5) | 0.52 | 0.10 (OK) | Best accuracy, limited depth to prevent overfit |
| XGBoost | — | 0.77 (overfit) | Too complex for n=114 |

### Evaluation

- **Repeated 10x5-fold cross-validation** (50 train/test splits) for robust R-squared and MAE estimates
- **95% confidence intervals** reported for all key metrics
- **p-values** for all correlation claims
- **Fisher z-test** for comparing predictor rankings

## Confidence Levels

| Conclusion | Confidence | Evidence |
|---|---|---|
| Block scores predict CBSE | VERY HIGH (>99%) | 22/23 exams p < 0.001 |
| Neuro is among the strongest predictors | HIGH | r = 0.70, 95% CI: 0.59-0.78 |
| Foundation is the weakest predictor | HIGH | r = 0.28, far below all others |
| Model R-squared ~ 0.52 | HIGH | 95% CI: 0.49-0.55 |
| Early warning works after course 2 | MODERATE-HIGH | 95% CI R-squared: 0.23-0.38 |
| Retake alone does not help | HIGH | p = 0.49, not significant |
| Improvement trend is non-predictive | MODERATE | r = -0.16, p = 0.09 |

## Limitations & Future Work

### Current Limitations

- **Single institution, single cohort (n = 114)** — all findings require validation on future cohorts
- **No external validation** — model has not been tested on a held-out year of students
- **Selection bias** — 5 students with block data but no CBSE (possibly dismissed) are excluded from training
- **CBSE day-of variability** — the standard error of measurement is 4 EPC points (per NBME documentation)

### Recommended Additional Data Sources

| Data Source | Expected Impact | Rationale |
|---|---|---|
| UWorld / question bank performance | HIGH | Most studied predictor of Step 1 readiness |
| NBME practice exam (self-assessment) scores | HIGH | Direct CBSE analog |
| Study behavior data (Anki stats, hours) | MODERATE | Captures effort and consistency |
| Prior-year cohort data | HIGH | Increases sample size 3-4x, enables temporal validation |
| Attendance / engagement metrics | LOW-MODERATE | Non-compliance signal |

## Data Privacy

This repository contains **only synthetic data and analysis code**. No real student records, names, scores, or identifiable information are included. The pipeline is designed for deployment in FERPA-compliant institutional environments where real data remains on secure local systems.

## Technical Stack

Python 3.10+ | pandas | scikit-learn | XGBoost | SHAP | matplotlib | seaborn | scipy

## Learning Notes

This project includes detailed [ML/DS learning notes](docs/learning_notes.md) documenting each concept encountered during development, annotated with **Healthcare AI Product Manager relevance ratings** (1-3 stars) and practical implications.
