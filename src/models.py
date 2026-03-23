"""
Model Training & Evaluation
============================
Train multiple models and compare performance.

Models:
1. Linear Regression (baseline, most interpretable)
2. Random Forest (captures non-linear patterns)
3. XGBoost (gradient boosting, often best performance)

Learning Points:
- Train/Test Split: why we hold out data the model has never seen
- Cross-Validation: more robust than a single split
- Regression Metrics: MAE, RMSE, R-squared
- Overfitting: training score >> test score = red flag
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Split data into training and test sets.

    Why 80/20?
    - 80% for training: model learns patterns from this data
    - 20% for testing: simulate "new unseen data" to check real performance
    - If we only looked at training performance, we'd never catch overfitting

    random_state = fixed seed so results are reproducible
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
    )
    print(f"Train: {len(X_train)} students | Test: {len(X_test)} students")
    return X_train, X_test, y_train, y_test


def train_linear_regression(X_train, y_train) -> LinearRegression:
    """
    Linear Regression — the baseline model.

    Why start here?
    - Most interpretable: each coefficient = "1 point increase in block X
      leads to N point change in CBSE score"
    - If this already works well, you don't need a complex model
    - If this doesn't work, the relationship might be non-linear
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train) -> RandomForestRegressor:
    """
    Random Forest — ensemble of decision trees.

    Why use this?
    - Captures non-linear relationships (linear regression can't)
    - Built-in feature importance
    - Less prone to overfitting than a single decision tree
    - Doesn't need feature scaling (unlike linear regression)
    """
    model = RandomForestRegressor(
        n_estimators=100,   # 100 trees in the forest
        max_depth=10,       # Limit depth to prevent overfitting
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train) -> XGBRegressor:
    """
    XGBoost — gradient boosted trees.

    Why use this?
    - Often the best performing model in tabular data competitions
    - Builds trees sequentially, each one correcting the previous one's errors
    - More sophisticated than Random Forest
    """
    model = XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
    )
    model.fit(X_train, y_train, verbose=False)
    return model


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name: str) -> dict:
    """
    Evaluate model on both training and test sets.

    Why both?
    - Training score shows how well the model "memorized" the data
    - Test score shows how well it generalizes to new data
    - Big gap between them = OVERFITTING (red flag!)

    Metrics:
    - MAE: average prediction error in score points (easiest to explain)
    - RMSE: like MAE but penalizes large errors more
    - R-squared: proportion of variance explained (0 = useless, 1 = perfect)
    """
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    results = {
        "model": model_name,
        "train_MAE": mean_absolute_error(y_train, train_pred),
        "test_MAE": mean_absolute_error(y_test, test_pred),
        "train_RMSE": root_mean_squared_error(y_train, train_pred),
        "test_RMSE": root_mean_squared_error(y_test, test_pred),
        "train_R2": r2_score(y_train, train_pred),
        "test_R2": r2_score(y_test, test_pred),
    }

    # Check for overfitting
    r2_gap = results["train_R2"] - results["test_R2"]
    overfitting = "YES - check overfitting!" if r2_gap > 0.15 else "OK"

    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    print(f"  {'Metric':<12} {'Train':>10} {'Test':>10} {'Gap':>10}")
    print(f"  {'MAE':<12} {results['train_MAE']:>10.2f} {results['test_MAE']:>10.2f}")
    print(f"  {'RMSE':<12} {results['train_RMSE']:>10.2f} {results['test_RMSE']:>10.2f}")
    print(f"  {'R-squared':<12} {results['train_R2']:>10.3f} {results['test_R2']:>10.3f} {r2_gap:>10.3f}")
    print(f"  Overfitting: {overfitting}")

    return results


def cross_validate(model, X, y, model_name: str, cv: int = 5) -> float:
    """
    K-Fold Cross-Validation.

    Why not just one train/test split?
    - One split might be "lucky" or "unlucky" depending on which students
      end up in test set
    - CV splits data K times (5 by default), trains K separate models,
      averages the results
    - More reliable estimate of real-world performance
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
    mean_r2 = scores.mean()
    std_r2 = scores.std()
    print(f"  {model_name} CV R-squared: {mean_r2:.3f} (+/- {std_r2:.3f})")
    return mean_r2


def compare_models(results_list: list[dict]) -> pd.DataFrame:
    """Create a comparison table of all models."""
    df = pd.DataFrame(results_list)
    df = df.sort_values("test_R2", ascending=False)
    return df


def save_metrics(comparison_df: pd.DataFrame, filepath: str = "output/model_metrics.txt"):
    """Save model comparison to a text file."""
    with open(filepath, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("  NBME CBSE Prediction - Model Comparison\n")
        f.write("=" * 60 + "\n\n")

        for _, row in comparison_df.iterrows():
            f.write(f"Model: {row['model']}\n")
            f.write(f"  Test MAE:  {row['test_MAE']:.2f} points\n")
            f.write(f"  Test RMSE: {row['test_RMSE']:.2f} points\n")
            f.write(f"  Test R2:   {row['test_R2']:.3f}\n")
            f.write(f"  Train R2:  {row['train_R2']:.3f}\n")
            gap = row['train_R2'] - row['test_R2']
            f.write(f"  Overfit gap: {gap:.3f}")
            f.write(f" {'(WARNING)' if gap > 0.15 else '(OK)'}\n\n")

        best = comparison_df.iloc[0]
        f.write(f"Best model: {best['model']} (Test R2 = {best['test_R2']:.3f})\n")

    print(f"\nSaved metrics to {filepath}")


def train_all_models(X, y):
    """
    Full training pipeline: split, train 3 models, evaluate, compare.
    Returns the best model and the comparison dataframe.
    """
    print("=== Model Training Pipeline ===\n")

    # Split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train and evaluate each model
    models = {
        "Linear Regression": train_linear_regression(X_train, y_train),
        "Random Forest": train_random_forest(X_train, y_train),
        "XGBoost": train_xgboost(X_train, y_train),
    }

    results = []
    for name, model in models.items():
        result = evaluate_model(model, X_train, y_train, X_test, y_test, name)
        results.append(result)

    # Cross-validation
    print(f"\n{'='*50}")
    print(f"  5-Fold Cross-Validation")
    print(f"{'='*50}")
    for name, model in models.items():
        cross_validate(model, X, y, name)

    # Compare
    comparison = compare_models(results)
    save_metrics(comparison)

    # Select best model (by test R2)
    best_name = comparison.iloc[0]["model"]
    best_model = models[best_name]
    print(f"\nBest model: {best_name} (Test R2 = {comparison.iloc[0]['test_R2']:.3f})")

    return best_model, models, comparison, (X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    from src.data_loader import load_and_merge, get_feature_columns
    from src.preprocessing import preprocess_pipeline

    df = load_and_merge()
    feature_cols = get_feature_columns(df)
    X, y, feature_names, scaler = preprocess_pipeline(df, feature_cols)
    best_model, models, comparison, splits = train_all_models(X, y)
