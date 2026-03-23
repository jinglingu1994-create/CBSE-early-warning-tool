"""
Early Warning System
====================
Predict CBSE scores for current cohort and flag at-risk students.

This is the actionable output of the entire project:
a list of students who need intervention BEFORE they take the CBSE.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def predict_current_cohort(
    model,
    current_df: pd.DataFrame,
    feature_cols: list[str],
    scaler=None,
) -> pd.DataFrame:
    """
    Use trained model to predict CBSE scores for current students
    who haven't taken the CBSE yet.
    """
    # Prepare features (same preprocessing as training data)
    X = current_df[feature_cols].copy() if all(c in current_df.columns for c in feature_cols) else None

    if X is None:
        # Need to engineer features first
        from src.data_loader import get_feature_columns
        raw_features = get_feature_columns(current_df)
        X = current_df[raw_features].copy()

        # Engineer same features as training
        X["block_average"] = X[raw_features].mean(axis=1).round(2)
        X["block_variance"] = X[raw_features].var(axis=1).round(2)
        X["lowest_block"] = X[raw_features].min(axis=1)

    # Scale if scaler was used during training
    if scaler is not None:
        # Only scale columns that exist in the scaler
        cols_to_scale = [c for c in X.columns if c in scaler.feature_names_in_]
        X[cols_to_scale] = scaler.transform(X[cols_to_scale])

    predictions = model.predict(X)

    result = current_df[["student_id", "name"]].copy()
    result["predicted_cbse"] = predictions.round(0).astype(int)

    return result


def flag_at_risk(
    predictions_df: pd.DataFrame,
    threshold: int = 194,
) -> pd.DataFrame:
    """
    Flag students whose predicted CBSE score is below the passing threshold.

    Default threshold = 194 (USMLE Step 1 old passing score reference).

    Risk score = how far below threshold (higher = more at risk):
      risk_score = (threshold - predicted) / threshold
      0 = exactly at threshold
      > 0 = below threshold (at risk)
      < 0 = above threshold (safe)
    """
    df = predictions_df.copy()
    df["threshold"] = threshold
    df["risk_score"] = ((threshold - df["predicted_cbse"]) / threshold).round(3)
    df["at_risk"] = df["predicted_cbse"] < threshold

    # Sort by risk (highest risk first)
    df = df.sort_values("risk_score", ascending=False)

    n_at_risk = df["at_risk"].sum()
    n_total = len(df)
    print(f"At-risk students: {n_at_risk}/{n_total} ({n_at_risk/n_total*100:.1f}%)")
    print(f"Passing threshold: {threshold}")

    return df


def generate_report(
    flagged_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_cols: list[str],
    save_path: str = "output/at_risk_students.csv",
):
    """
    Generate the early warning report with block scores included.
    This is what a dean or academic advisor would review.
    """
    # Merge predictions with original block scores
    report = flagged_df.merge(
        current_df.drop(columns=["name"], errors="ignore"),
        on="student_id",
        how="left",
    )

    # Save full report
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)
    print(f"Saved report to {output_path}")

    # Print summary of at-risk students
    at_risk = report[report["at_risk"]]
    if len(at_risk) > 0:
        print(f"\n=== Top 10 Highest-Risk Students ===")
        top10 = at_risk.head(10)
        for _, row in top10.iterrows():
            print(f"  {row['student_id']:10s} | Predicted CBSE: {row['predicted_cbse']:3d} | Risk Score: {row['risk_score']:.3f}")

    return report


def run_early_warning(model, current_df, feature_cols, scaler=None, threshold=194):
    """Full early warning pipeline."""
    print("=== Early Warning System ===\n")

    # 1. Predict
    print("1. Predicting CBSE scores for current cohort...")
    predictions = predict_current_cohort(model, current_df, feature_cols, scaler)
    print(f"   Predicted {len(predictions)} students")
    print(f"   Score range: {predictions['predicted_cbse'].min()} - {predictions['predicted_cbse'].max()}")
    print()

    # 2. Flag at-risk
    print("2. Flagging at-risk students...")
    flagged = flag_at_risk(predictions, threshold)
    print()

    # 3. Generate report
    print("3. Generating report...")
    report = generate_report(flagged, current_df, feature_cols)

    return report


if __name__ == "__main__":
    from src.data_loader import load_and_merge, load_current_cohort, get_feature_columns
    from src.preprocessing import preprocess_pipeline
    from src.models import split_data, train_linear_regression

    # Train model on historical data
    df = load_and_merge()
    feature_cols = get_feature_columns(df)
    X, y, feature_names, scaler = preprocess_pipeline(df, feature_cols)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_linear_regression(X_train, y_train)

    # Load current cohort and run early warning
    current = load_current_cohort()
    run_early_warning(model, current, feature_names, scaler)
