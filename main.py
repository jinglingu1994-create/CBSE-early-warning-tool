"""
NBME CBSE Early Warning Tool
=============================
End-to-end pipeline: load data → preprocess → train → evaluate → predict → flag at-risk students

Usage:
    python main.py              # Run with synthetic data
    python main.py --generate   # Regenerate synthetic data first, then run

Replace CSV files in data/raw/ with real data to use real student scores.
"""

import argparse
from pathlib import Path

from src.generate_synthetic_data import generate_and_save
from src.data_loader import load_and_merge, load_current_cohort, get_feature_columns
from src.preprocessing import preprocess_pipeline
from src.models import train_all_models, split_data
from src.feature_importance import run_feature_importance
from src.early_warning import run_early_warning


def main(regenerate: bool = False):
    print("=" * 60)
    print("  NBME CBSE Early Warning Tool")
    print("=" * 60)

    # 0. Generate synthetic data if needed
    data_dir = Path("data/raw")
    if regenerate or not (data_dir / "block_scores.csv").exists():
        print("\n[Step 0] Generating synthetic data...")
        generate_and_save()

    # 1. Load and merge data
    print("\n" + "=" * 60)
    print("[Step 1] Loading data...")
    print("=" * 60)
    df = load_and_merge()
    feature_cols = get_feature_columns(df)

    # 2. Preprocess
    print("\n" + "=" * 60)
    print("[Step 2] Preprocessing...")
    print("=" * 60)
    X, y, feature_names, scaler = preprocess_pipeline(df, feature_cols)

    # 3. Train models
    print("\n" + "=" * 60)
    print("[Step 3] Training models...")
    print("=" * 60)
    best_model, models, comparison, (X_train, X_test, y_train, y_test) = train_all_models(X, y)

    # 4. Feature importance (using Linear Regression for interpretability)
    print("\n" + "=" * 60)
    print("[Step 4] Feature importance analysis...")
    print("=" * 60)
    lr_model = models["Linear Regression"]
    run_feature_importance(lr_model, X_test, y_test, feature_names)

    # 5. Early warning
    print("\n" + "=" * 60)
    print("[Step 5] Early warning system...")
    print("=" * 60)
    current = load_current_cohort()
    report = run_early_warning(lr_model, current, feature_names, scaler)

    # Summary
    print("\n" + "=" * 60)
    print("  COMPLETE")
    print("=" * 60)
    print(f"\n  Output files:")
    print(f"    output/model_metrics.txt        - Model performance comparison")
    print(f"    output/feature_importance.png    - Which blocks predict CBSE best")
    print(f"    output/linear_coefficients.png   - Linear regression coefficients")
    print(f"    output/permutation_importance.png- Permutation importance")
    print(f"    output/shap_summary.png          - SHAP per-student explanations")
    print(f"    output/at_risk_students.csv      - Flagged at-risk students")
    print(f"    output/correlation_heatmap.png   - Block score correlations")
    print(f"    output/distributions.png         - Score distributions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NBME CBSE Early Warning Tool")
    parser.add_argument("--generate", action="store_true", help="Regenerate synthetic data")
    args = parser.parse_args()
    main(regenerate=args.generate)
