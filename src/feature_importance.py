"""
Feature Importance Analysis
============================
Answer: "Which blocks are the strongest predictors of CBSE score?"

Methods:
1. Linear Regression Coefficients (most interpretable)
2. Permutation Importance (model-agnostic)
3. SHAP Values (per-student explanations)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance


def plot_linear_coefficients(model, feature_names: list[str], save_path: str = "output/linear_coefficients.png"):
    """
    Plot Linear Regression coefficients.

    Each coefficient = "if this block score goes up by 1 std,
    CBSE score changes by this many points."

    Positive = higher block score → higher CBSE
    Negative = higher block score → lower CBSE (unusual, might indicate multicollinearity)
    """
    coefs = pd.Series(model.coef_, index=feature_names).sort_values()

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#e74c3c" if c < 0 else "#2ecc71" for c in coefs.values]
    coefs.plot(kind="barh", ax=ax, color=colors)
    ax.set_title("Linear Regression Coefficients\n(Impact on CBSE Score per 1 Std Increase)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Coefficient (CBSE score points)")
    ax.axvline(0, color="black", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved linear coefficients plot to {save_path}")

    return coefs


def plot_permutation_importance(model, X_test, y_test, feature_names: list[str], save_path: str = "output/permutation_importance.png"):
    """
    Permutation Importance: shuffle one feature at a time, see how much
    model performance drops.

    Big drop = that feature was important.
    No drop = that feature didn't matter.
    """
    result = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42)
    importance = pd.Series(result.importances_mean, index=feature_names).sort_values()

    fig, ax = plt.subplots(figsize=(10, 6))
    importance.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Permutation Importance\n(How Much Performance Drops When Feature is Shuffled)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Decrease in R-squared")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved permutation importance plot to {save_path}")

    return importance


def plot_shap_summary(model, X_test, feature_names: list[str], save_path: str = "output/shap_summary.png"):
    """
    SHAP Summary Plot: shows how each feature pushes predictions
    higher or lower for each student.

    Red dots = high feature value
    Blue dots = low feature value
    Right side = pushes CBSE prediction UP
    Left side = pushes CBSE prediction DOWN
    """
    import shap

    explainer = shap.Explainer(model, X_test)
    shap_values = explainer(X_test)

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.beeswarm(shap_values, show=False)
    plt.title("SHAP Summary: Feature Impact on CBSE Score Prediction", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP summary plot to {save_path}")

    return shap_values


def plot_combined_importance(coefs, perm_importance, feature_names: list[str], save_path: str = "output/feature_importance.png"):
    """
    Combined feature importance chart — the main output.
    Shows both methods side by side for comparison.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Feature Importance: Which Blocks Best Predict CBSE Score?", fontsize=16, fontweight="bold")

    # Left: Linear coefficients (absolute value, ranked)
    abs_coefs = coefs.abs().sort_values(ascending=True)
    abs_coefs.plot(kind="barh", ax=ax1, color="steelblue")
    ax1.set_title("Linear Regression\n|Coefficient|", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Absolute Coefficient")

    # Right: Permutation importance
    perm_sorted = perm_importance.sort_values(ascending=True)
    perm_sorted.plot(kind="barh", ax=ax2, color="coral")
    ax2.set_title("Permutation Importance\nR-squared Drop", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Decrease in R-squared")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved combined feature importance to {save_path}")


def run_feature_importance(model, X_test, y_test, feature_names: list[str]):
    """Full feature importance pipeline."""
    print("=== Feature Importance Analysis ===\n")

    # 1. Linear coefficients
    print("1. Linear Regression Coefficients")
    coefs = plot_linear_coefficients(model, feature_names)
    print()

    # 2. Permutation importance
    print("2. Permutation Importance")
    perm = plot_permutation_importance(model, X_test, y_test, feature_names)
    print()

    # 3. SHAP
    print("3. SHAP Values")
    try:
        shap_values = plot_shap_summary(model, X_test, feature_names)
    except Exception as e:
        print(f"   SHAP failed (non-critical): {e}")
        shap_values = None
    print()

    # 4. Combined plot
    print("4. Combined Importance Plot")
    plot_combined_importance(coefs, perm, feature_names)

    # Print ranking
    print("\n=== Feature Ranking (by |Linear Coefficient|) ===")
    ranking = coefs.abs().sort_values(ascending=False)
    for i, (feat, val) in enumerate(ranking.items(), 1):
        print(f"  {i}. {feat:25s}  |coef| = {val:.2f}")

    return coefs, perm, shap_values


if __name__ == "__main__":
    from src.data_loader import load_and_merge, get_feature_columns
    from src.preprocessing import preprocess_pipeline
    from src.models import split_data, train_linear_regression

    df = load_and_merge()
    feature_cols = get_feature_columns(df)
    X, y, feature_names, scaler = preprocess_pipeline(df, feature_cols)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_linear_regression(X_train, y_train)
    run_feature_importance(model, X_test, y_test, feature_names)
