"""
Core prediction model: Block scores -> CBSE first attempt score.
Runs on REAL data (local only, never push output to GitHub).
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import warnings
warnings.filterwarnings('ignore')

# Paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data", "real")
OUTPUT_DIR = os.path.join(DATA_DIR)  # Output stays in data/real (local only)

# Load data
df = pd.read_csv(os.path.join(DATA_DIR, "merged_all_data.csv"))
df_model = df[df['cbse_first_score'].notna()].copy()

# Feature columns: all exam scores
exam_cols = [c for c in df.columns if any(x in c for x in ['midterm', 'final'])
             and 'cbse' not in c and 'date' not in c]

# Drop rows with missing exam scores
df_clean = df_model.dropna(subset=exam_cols)
print(f"Students with complete data: {len(df_clean)} (dropped {len(df_model)-len(df_clean)} with missing blocks)")

# Engineer features
df_clean = df_clean.copy()
df_clean['block_average'] = df_clean[exam_cols].mean(axis=1)
df_clean['block_variance'] = df_clean[exam_cols].var(axis=1)
df_clean['lowest_exam'] = df_clean[exam_cols].min(axis=1)
df_clean['highest_exam'] = df_clean[exam_cols].max(axis=1)

# Midterm-to-final improvement for each block
blocks = ['foundation', 'msk', 'gi', 'heme', 'cv', 'pulm', 'renal',
          'neuro', 'behavioral_science', 'endo', 'repro']
for block in blocks:
    mid_cols = [c for c in exam_cols if c.startswith(block) and 'midterm' in c]
    fin_cols = [c for c in exam_cols if c.startswith(block) and 'final' in c and 'midterm' not in c]
    if mid_cols and fin_cols:
        df_clean[f'{block}_improvement'] = df_clean[fin_cols[0]] - df_clean[mid_cols[0]]

improvement_cols = [c for c in df_clean.columns if 'improvement' in c]
engineered_cols = ['block_average', 'block_variance', 'lowest_exam', 'highest_exam'] + improvement_cols
all_feature_cols = exam_cols + engineered_cols

X = df_clean[all_feature_cols].values
y = df_clean['cbse_first_score'].values

print(f"Total features: {len(all_feature_cols)} ({len(exam_cols)} exam + {len(engineered_cols)} engineered)")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# Scale
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ============================================================
# Model 1: Linear Regression
# ============================================================
lr = LinearRegression()
lr.fit(X_train_s, y_train)
y_pred_lr_train = lr.predict(X_train_s)
y_pred_lr_test = lr.predict(X_test_s)

print(f"\n{'='*60}")
print("Model 1: Linear Regression")
print("=" * 60)
lr_train_r2 = r2_score(y_train, y_pred_lr_train)
lr_test_r2 = r2_score(y_test, y_pred_lr_test)
lr_test_mae = mean_absolute_error(y_test, y_pred_lr_test)
print(f"  Train R2: {lr_train_r2:.3f}  MAE: {mean_absolute_error(y_train, y_pred_lr_train):.1f}")
print(f"  Test  R2: {lr_test_r2:.3f}  MAE: {lr_test_mae:.1f}")
gap = lr_train_r2 - lr_test_r2
print(f"  Gap: {gap:.3f} {'-> OK' if gap < 0.2 else '-> OVERFITTING'}")

# ============================================================
# Model 2: Random Forest (depth limited)
# ============================================================
rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train_s, y_train)
y_pred_rf_train = rf.predict(X_train_s)
y_pred_rf_test = rf.predict(X_test_s)

print(f"\n{'='*60}")
print("Model 2: Random Forest (depth=5)")
print("=" * 60)
rf_train_r2 = r2_score(y_train, y_pred_rf_train)
rf_test_r2 = r2_score(y_test, y_pred_rf_test)
rf_test_mae = mean_absolute_error(y_test, y_pred_rf_test)
print(f"  Train R2: {rf_train_r2:.3f}  MAE: {mean_absolute_error(y_train, y_pred_rf_train):.1f}")
print(f"  Test  R2: {rf_test_r2:.3f}  MAE: {rf_test_mae:.1f}")
gap = rf_train_r2 - rf_test_r2
print(f"  Gap: {gap:.3f} {'-> OK' if gap < 0.2 else '-> OVERFITTING'}")

# ============================================================
# Cross-validation
# ============================================================
print(f"\n{'='*60}")
print("5-Fold Cross-Validation")
print("=" * 60)

X_all_s = scaler.fit_transform(X)
results = {}
for name, model in [('Linear Regression', LinearRegression()),
                     ('Random Forest (d=5)', RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42))]:
    cv_r2 = cross_val_score(model, X_all_s, y, cv=5, scoring='r2')
    cv_mae = -cross_val_score(model, X_all_s, y, cv=5, scoring='neg_mean_absolute_error')
    results[name] = {'cv_r2': cv_r2.mean(), 'cv_mae': cv_mae.mean()}
    print(f"  {name}:")
    print(f"    CV R2:  {cv_r2.mean():.3f} (+/- {cv_r2.std():.3f})")
    print(f"    CV MAE: {cv_mae.mean():.1f} (+/- {cv_mae.std():.1f}) EPC points")

# ============================================================
# Feature Importance
# ============================================================
print(f"\n{'='*60}")
print("Feature Importance (Top 15 by |coefficient|)")
print("=" * 60)

lr_full = LinearRegression()
lr_full.fit(X_all_s, y)
coef_df = pd.DataFrame({'feature': all_feature_cols, 'coef': lr_full.coef_})
coef_df['abs_coef'] = coef_df['coef'].abs()
coef_df = coef_df.sort_values('abs_coef', ascending=False)

for _, row in coef_df.head(15).iterrows():
    direction = '+' if row['coef'] > 0 else '-'
    print(f"  {direction}{row['abs_coef']:.2f}  {row['feature']}")

# ============================================================
# Visualization
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Plot 1: Actual vs Predicted (LR)
axes[0, 0].scatter(y_test, y_pred_lr_test, alpha=0.7, s=50)
axes[0, 0].plot([40, 90], [40, 90], 'r--', label='Perfect prediction')
axes[0, 0].set_xlabel('Actual CBSE (EPC)')
axes[0, 0].set_ylabel('Predicted CBSE (EPC)')
axes[0, 0].set_title(f'Linear Regression: Actual vs Predicted\n(Test R2={lr_test_r2:.3f}, MAE={lr_test_mae:.1f})')
axes[0, 0].legend()

# Plot 2: Actual vs Predicted (RF)
axes[0, 1].scatter(y_test, y_pred_rf_test, alpha=0.7, s=50, color='green')
axes[0, 1].plot([40, 90], [40, 90], 'r--', label='Perfect prediction')
axes[0, 1].set_xlabel('Actual CBSE (EPC)')
axes[0, 1].set_ylabel('Predicted CBSE (EPC)')
axes[0, 1].set_title(f'Random Forest: Actual vs Predicted\n(Test R2={rf_test_r2:.3f}, MAE={rf_test_mae:.1f})')
axes[0, 1].legend()

# Plot 3: Top 15 features
top15 = coef_df.head(15)
colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in top15['coef']]
axes[1, 0].barh(range(len(top15)), top15['coef'].values, color=colors)
axes[1, 0].set_yticks(range(len(top15)))
short_names = [n.replace('behavioral_science', 'behav').replace('_improvement', '_impr')
               .replace('_midterm', '_mid').replace('_final', '_fin') for n in top15['feature']]
axes[1, 0].set_yticklabels(short_names, fontsize=9)
axes[1, 0].set_xlabel('Coefficient (standardized)')
axes[1, 0].set_title('Top 15 Features by Impact on CBSE\n(Green=positive, Red=negative)')
axes[1, 0].invert_yaxis()

# Plot 4: Residual distribution
residuals = y_test - y_pred_lr_test
axes[1, 1].hist(residuals, bins=15, edgecolor='black', alpha=0.7)
axes[1, 1].axvline(x=0, color='red', linestyle='--')
axes[1, 1].set_xlabel('Prediction Error (Actual - Predicted EPC)')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_title(f'Prediction Error Distribution\n(Mean={residuals.mean():.1f}, Std={residuals.std():.1f})')

plt.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, "model_results_real.png")
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"\nSaved: {fig_path}")

# ============================================================
# Summary
# ============================================================
best_name = max(results, key=lambda k: results[k]['cv_r2'])
best = results[best_name]
print(f"\n{'='*60}")
print("SUMMARY")
print("=" * 60)
print(f"Best model: {best_name}")
print(f"CV R2: {best['cv_r2']:.3f} -> block scores explain ~{best['cv_r2']*100:.0f}% of CBSE variance")
print(f"CV MAE: {best['cv_mae']:.1f} -> average prediction off by ~{best['cv_mae']:.0f} EPC points")
print(f"Features: {len(all_feature_cols)} total")
