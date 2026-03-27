# NBME CBSE Early Warning Tool

> **Predicting CBSE performance from organ-system block exams to enable early academic intervention in medical education.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Background

Medical students at U.S. allopathic schools take the **NBME Comprehensive Basic Science Examination (CBSE)** as a readiness check for USMLE Step 1. Students who score below the institutional threshold often require remediation, yet intervention typically happens *after* the CBSE — when time is limited.

**This project asks: Can we identify at-risk students *before* they take the CBSE, using only their organ-system block exam scores?**

## What This Tool Does

1. **Predicts** each student's CBSE score based on their block exam performance
2. **Identifies** which organ systems are the strongest predictors of CBSE success
3. **Flags** students at risk of failing before they sit for the exam
4. **Determines** how early in the curriculum reliable prediction is possible

## Architecture

```
NBME CBSE Early Warning Tool/
|
|-- main.py                          # End-to-end pipeline (synthetic data demo)
|-- src/
|   |-- generate_synthetic_data.py   # Generates realistic training data
|   |-- data_loader.py               # CSV ingestion and merging
|   |-- preprocessing.py             # Missing values, feature engineering
|   |-- models.py                    # Linear Regression, Random Forest, XGBoost
|   |-- feature_importance.py        # Coefficients, Permutation, SHAP
|   |-- early_warning.py             # Threshold-based risk flagging
|
|-- data/
|   |-- raw/                         # Synthetic CSVs (safe to share)
|
|-- output/                          # Generated charts and reports
|-- docs/
|   |-- learning_notes.md            # ML/DS concepts with Healthcare AI PM annotations
```

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

Generates synthetic medical student data (500 historical + 100 current), trains 3 models, produces feature importance analysis, and flags at-risk students. All output goes to `output/`.

## Methodology

### Data Pipeline

1. **Ingestion**: Block exam scores (one CSV per organ system) + CBSE results
2. **Feature extraction**: Summative exam scores (midterm + final per block)
3. **Merging**: Join on student ID across all data sources

### Feature Engineering

| Feature | Rationale |
|---------|-----------|
| Individual block exam scores | Direct performance measures per organ system |
| Block average | Overall ability level |
| Block variance | Consistency across systems (high variance = weak spots) |
| Lowest block score | "Weakest link" effect |

### Models Compared

| Model | Strengths | When It Wins |
|-------|-----------|--------------|
| Linear Regression / Ridge | Fully interpretable coefficients | Small datasets (n < 500), linear relationships |
| Random Forest | Captures non-linear patterns, built-in feature importance | Medium datasets, complex interactions |
| XGBoost | Often highest accuracy | Large datasets (n > 1000) |

**Key insight from this project**: With medical school cohort sizes (~100-200 students), simpler models (Ridge Regression) often outperform complex ones (XGBoost) because complex models overfit on small samples.

### Evaluation Approach

- **Repeated K-fold cross-validation** for robust performance estimates
- **95% confidence intervals** on all metrics
- **Train vs Test gap monitoring** to detect overfitting
- **Multiple importance methods** (coefficients, permutation, SHAP) to cross-validate feature rankings

### Early Warning Timing Analysis

The pipeline includes a temporal analysis: at each curriculum checkpoint (after course 1, after course 2, ...), how accurately can we predict CBSE? This answers the practical question: **how early can schools intervene?**

## Designed for Real Data

This demo runs on synthetic data. To use with real institutional data:

1. Place block exam CSVs in `data/raw/`
2. Place CBSE results in `data/raw/cbse_results.csv`
3. Ensure a shared `student_id` column across files
4. Run `python main.py`

The pipeline auto-detects numeric columns as features. No code changes needed for different organ system names or block structures.

## Data Privacy

This repository contains **only synthetic data and analysis code**. No real student records, names, scores, or identifiable information are included. The pipeline is designed for deployment in FERPA-compliant institutional environments where real data remains on secure local systems.

## Technical Stack

Python 3.10+ | pandas | scikit-learn | XGBoost | SHAP | matplotlib | seaborn | scipy

## Learning Notes

This project includes detailed [ML/DS learning notes](docs/learning_notes.md) documenting each concept encountered during development, annotated with **Healthcare AI Product Manager relevance ratings** (1-3 stars) and practical implications for building clinical AI products.
