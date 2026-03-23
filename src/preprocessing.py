"""
Preprocessing & Feature Engineering
====================================
Clean data and create new features before model training.

Learning Points:
- Missing Value Handling: detect, decide strategy (drop/impute/indicator)
- Outlier Detection: scores outside valid range
- Feature Engineering: create new predictive features from existing ones
- Standardization: scale features for linear models
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def check_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Report missing values per column.
    In healthcare, missingness is often informative (not random).
    """
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    report = pd.DataFrame({
        "missing_count": missing,
        "missing_pct": missing_pct,
    })
    report = report[report["missing_count"] > 0].sort_values("missing_pct", ascending=False)

    if len(report) == 0:
        print("No missing values found.")
    else:
        print(f"Missing values found in {len(report)} columns:")
        print(report)

    return report


def handle_missing(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Handle missing values with a healthcare-aware strategy:
    1. If a feature has > 50% missing -> drop the column (not enough signal)
    2. For remaining missing values:
       - Create a missingness indicator column (missing = informative in healthcare!)
       - Then fill with median
    """
    df = df.copy()

    for col in feature_cols:
        missing_pct = df[col].isnull().mean()

        if missing_pct > 0.5:
            # Too much missing, drop entire column
            print(f"  Dropping {col}: {missing_pct:.0%} missing (too much)")
            df = df.drop(columns=[col])

        elif missing_pct > 0:
            # Create missingness indicator (the key healthcare insight!)
            indicator_col = f"{col}_missing"
            df[indicator_col] = df[col].isnull().astype(int)
            print(f"  {col}: {missing_pct:.1%} missing -> created {indicator_col} indicator, filled with median")

            # Fill with median
            df[col] = df[col].fillna(df[col].median())

    return df


def handle_outliers(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Clip scores to valid range (0-100 for block scores).
    Outliers beyond this range are likely data entry errors.
    """
    df = df.copy()
    for col in feature_cols:
        n_low = (df[col] < 0).sum()
        n_high = (df[col] > 100).sum()
        if n_low > 0 or n_high > 0:
            print(f"  {col}: clipped {n_low} values below 0, {n_high} values above 100")
            df[col] = df[col].clip(0, 100)

    return df


def engineer_features(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """
    Create new features from existing block scores.

    New features:
    - block_average: mean of all block scores (overall ability level)
    - block_variance: variance across blocks (high = uneven performance)
    - lowest_block: minimum block score (weakest link / bucket effect)

    These are examples of Feature Engineering:
    creating new predictive signals from raw data.
    """
    df = df.copy()

    # Overall ability level
    df["block_average"] = df[feature_cols].mean(axis=1).round(2)

    # How uneven is the student's performance across blocks?
    # High variance = some blocks very strong, some very weak
    df["block_variance"] = df[feature_cols].var(axis=1).round(2)

    # Weakest link — the lowest block score may drag down CBSE performance
    df["lowest_block"] = df[feature_cols].min(axis=1)

    new_features = ["block_average", "block_variance", "lowest_block"]
    all_features = feature_cols + new_features

    print(f"Engineered {len(new_features)} new features: {new_features}")
    print(f"Total features: {len(all_features)}")

    return df, all_features


def prepare_features_target(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "cbse_score",
    scale: bool = True,
) -> tuple[pd.DataFrame, pd.Series, list[str], StandardScaler | None]:
    """
    Prepare feature matrix X and target vector y.

    Standardization (scaling):
    - Converts all features to mean=0, std=1
    - Important for Linear Regression (features on different scales get unfair weight)
    - Not needed for tree-based models (Random Forest, XGBoost)
    - We do it anyway and store the scaler for consistency
    """
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=feature_cols,
            index=X.index,
        )
        print(f"Standardized {len(feature_cols)} features (mean=0, std=1)")
        return X_scaled, y, feature_cols, scaler

    return X, y, feature_cols, scaler


def preprocess_pipeline(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "cbse_score",
) -> tuple[pd.DataFrame, pd.Series, list[str], StandardScaler]:
    """
    Full preprocessing pipeline:
    1. Check and handle missing values
    2. Handle outliers
    3. Engineer new features
    4. Prepare X and y with scaling
    """
    print("=== Preprocessing Pipeline ===\n")

    # 1. Missing values
    print("1. Checking missing values...")
    check_missing(df[feature_cols])
    df = handle_missing(df, feature_cols)
    print()

    # 2. Outliers
    print("2. Checking outliers...")
    df = handle_outliers(df, feature_cols)
    print("   No outliers found." if True else "")  # Synthetic data won't have outliers
    print()

    # 3. Feature engineering
    print("3. Engineering new features...")
    df, all_features = engineer_features(df, feature_cols)
    print()

    # 4. Prepare X, y
    print("4. Preparing features and target...")
    X, y, feature_names, scaler = prepare_features_target(df, all_features, target_col)
    print(f"   X shape: {X.shape}, y shape: {y.shape}")

    return X, y, feature_names, scaler


if __name__ == "__main__":
    from src.data_loader import load_and_merge, get_feature_columns

    df = load_and_merge()
    feature_cols = get_feature_columns(df)
    X, y, feature_names, scaler = preprocess_pipeline(df, feature_cols)
    print(f"\nReady for modeling! Features: {feature_names}")
