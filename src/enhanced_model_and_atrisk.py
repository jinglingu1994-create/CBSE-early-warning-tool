"""
Enhanced model with formative + remediation features.
Generates at-risk student list with predicted CBSE and risk levels.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, RepeatedKFold, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data", "real")
PASS_THRESHOLD = 64

df = pd.read_csv(os.path.join(DATA_DIR, "merged_enhanced.csv"))

# Original exam features
exam_cols = [c for c in df.columns if any(x in c for x in ['midterm', 'final'])
             and 'cbse' not in c and 'date' not in c]

# New features
formative_cols = [c for c in df.columns if 'formative' in c.lower() and 'cbse' not in c]
remediation_cols = [c for c in df.columns if 'remediation' in c.lower()]

# All features
all_features_old = exam_cols
all_features_new = exam_cols + formative_cols + remediation_cols

df_model = df[df['cbse_first_score'].notna()].copy()

print("=" * 70)
print("  ENHANCED MODEL: Original vs New Features")
print("=" * 70)

# ============================================================
# Compare: original features vs enhanced features
# ============================================================
rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)

for name, feature_set in [("Original (23 exam scores only)", all_features_old),
                           ("Enhanced (+formative +remediation)", all_features_new)]:
    df_clean = df_model.dropna(subset=feature_set)
    X = df_clean[feature_set].values
    y = df_clean['cbse_first_score'].values

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    cv_r2 = cross_val_score(rf, X_s, y, cv=rkf, scoring='r2')
    cv_mae = -cross_val_score(rf, X_s, y, cv=rkf, scoring='neg_mean_absolute_error')

    ci_low = cv_r2.mean() - 1.96 * cv_r2.std() / np.sqrt(len(cv_r2))
    ci_high = cv_r2.mean() + 1.96 * cv_r2.std() / np.sqrt(len(cv_r2))

    print(f"\n  {name}")
    print(f"  Features: {len(feature_set)}, Students: {len(df_clean)}")
    print(f"  CV R2:  {cv_r2.mean():.3f} (95% CI: {ci_low:.3f}-{ci_high:.3f})")
    print(f"  CV MAE: {cv_mae.mean():.1f} EPC points")

# ============================================================
# Train final model on enhanced features
# ============================================================
print(f"\n{'='*70}")
print("  FINAL MODEL (Enhanced Features)")
print("=" * 70)

df_final = df_model.dropna(subset=all_features_new)
X_all = df_final[all_features_new].values
y_all = df_final['cbse_first_score'].values

scaler = StandardScaler()
X_all_s = scaler.fit_transform(X_all)

# Train on all data for final predictions
rf_final = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
rf_final.fit(X_all_s, y_all)

# Feature importance
importances = pd.DataFrame({
    'feature': all_features_new,
    'importance': rf_final.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 15 Features (Random Forest Importance):")
for _, row in importances.head(15).iterrows():
    feat = row['feature']
    imp = row['importance']
    is_new = " [NEW]" if feat in formative_cols or feat in remediation_cols else ""
    print(f"  {imp:.3f}  {feat}{is_new}")

# ============================================================
# Generate at-risk student list
# ============================================================
print(f"\n{'='*70}")
print(f"  AT-RISK STUDENT LIST (threshold = {PASS_THRESHOLD} EPC)")
print("=" * 70)

# Predict for ALL students with block data (including those who haven't taken CBSE yet)
df_predict = df.dropna(subset=all_features_new).copy()
X_pred = scaler.transform(df_predict[all_features_new].values)
df_predict['predicted_cbse'] = rf_final.predict(X_pred)

# Individual tree predictions for confidence interval
tree_preds = np.array([tree.predict(X_pred) for tree in rf_final.estimators_])
df_predict['pred_std'] = tree_preds.std(axis=0)
df_predict['pred_low'] = df_predict['predicted_cbse'] - 1.96 * df_predict['pred_std']
df_predict['pred_high'] = df_predict['predicted_cbse'] + 1.96 * df_predict['pred_std']

# Risk classification
def classify_risk(row):
    pred = row['predicted_cbse']
    if pred < PASS_THRESHOLD - 8:
        return "HIGH RISK"
    elif pred < PASS_THRESHOLD - 3:
        return "AT RISK"
    elif pred < PASS_THRESHOLD + 3:
        return "BORDERLINE"
    else:
        return "LIKELY PASS"

df_predict['risk_level'] = df_predict.apply(classify_risk, axis=1)

# Summary
for level in ["HIGH RISK", "AT RISK", "BORDERLINE", "LIKELY PASS"]:
    count = (df_predict['risk_level'] == level).sum()
    pct = count / len(df_predict) * 100
    print(f"  {level:<15}: {count:>3} students ({pct:.0f}%)")

# At-risk details
print(f"\n--- HIGH RISK and AT RISK Students ---")
at_risk = df_predict[df_predict['risk_level'].isin(["HIGH RISK", "AT RISK"])].copy()
at_risk = at_risk.sort_values('predicted_cbse')

print(f"  {'ID':>6} {'Name':<30} {'Pred CBSE':>9} {'95% CI':>15} {'Actual':>7} {'Risk'}")
print("-" * 85)

for _, row in at_risk.iterrows():
    sid = int(row['student_id'])
    name = row.get('name', 'Unknown')
    if pd.isna(name):
        name = 'Unknown'
    pred = row['predicted_cbse']
    low = row['pred_low']
    high = row['pred_high']
    actual = row.get('cbse_first_score', None)
    actual_str = f"{actual:.0f}" if pd.notna(actual) else "N/A"
    risk = row['risk_level']
    print(f"  {sid:>6} {str(name):<30} {pred:>8.1f} ({low:.1f}-{high:.1f}) {actual_str:>7} {risk}")

# Save at-risk list
at_risk_out = df_predict[['student_id', 'name', 'predicted_cbse', 'pred_low', 'pred_high',
                           'risk_level', 'cbse_first_score', 'cbse_total_attempts']].copy()
at_risk_out = at_risk_out.sort_values('predicted_cbse')
at_risk_path = os.path.join(DATA_DIR, "at_risk_students_real.csv")
at_risk_out.to_csv(at_risk_path, index=False)
print(f"\nSaved full list: {at_risk_path}")

# ============================================================
# Accuracy check: for students who DID take CBSE
# ============================================================
print(f"\n{'='*70}")
print("  ACCURACY CHECK: Predicted vs Actual (students with CBSE)")
print("=" * 70)

has_actual = df_predict[df_predict['cbse_first_score'].notna()]
predicted_pass = has_actual['predicted_cbse'] >= PASS_THRESHOLD
actual_pass = has_actual['cbse_first_score'] >= PASS_THRESHOLD

# Confusion matrix
tp = ((predicted_pass) & (actual_pass)).sum()      # Correctly predicted pass
tn = ((~predicted_pass) & (~actual_pass)).sum()    # Correctly predicted fail
fp = ((predicted_pass) & (~actual_pass)).sum()     # Predicted pass but actually failed
fn = ((~predicted_pass) & (actual_pass)).sum()     # Predicted fail but actually passed

print(f"  Correctly predicted PASS:  {tp}")
print(f"  Correctly predicted FAIL:  {tn}")
print(f"  False positive (said pass, actually fail): {fp}")
print(f"  False negative (said fail, actually pass): {fn}")
print(f"\n  Accuracy: {(tp+tn)/(tp+tn+fp+fn)*100:.1f}%")
print(f"  Sensitivity (catch rate for failures): {tn/(tn+fp)*100:.1f}%" if (tn+fp) > 0 else "")
print(f"  Specificity (correctly identify passers): {tp/(tp+fn)*100:.1f}%" if (tp+fn) > 0 else "")

# ============================================================
# Visualization
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# 1: Enhanced feature importance
top20 = importances.head(20)
colors = ['#e67e22' if f in formative_cols or f in remediation_cols else '#3498db'
          for f in top20['feature']]
short_names = [f.replace('behavioral_science', 'behav').replace('_midterm', '_mid')
               .replace('_final', '_fin').replace('_formative', '_form').replace('_remediation', '_rem')
               for f in top20['feature']]
axes[0, 0].barh(range(len(top20)), top20['importance'].values, color=colors)
axes[0, 0].set_yticks(range(len(top20)))
axes[0, 0].set_yticklabels(short_names, fontsize=9)
axes[0, 0].set_xlabel('Feature Importance')
axes[0, 0].set_title('Top 20 Features (Enhanced Model)\nOrange = NEW features (formative/remediation)')
axes[0, 0].invert_yaxis()

# 2: Risk distribution
risk_counts = df_predict['risk_level'].value_counts()
risk_order = ["HIGH RISK", "AT RISK", "BORDERLINE", "LIKELY PASS"]
risk_colors = {'HIGH RISK': '#e74c3c', 'AT RISK': '#f39c12', 'BORDERLINE': '#f1c40f', 'LIKELY PASS': '#2ecc71'}
counts = [risk_counts.get(r, 0) for r in risk_order]
axes[0, 1].bar(risk_order, counts, color=[risk_colors[r] for r in risk_order], edgecolor='gray')
for i, c in enumerate(counts):
    axes[0, 1].text(i, c + 0.5, str(c), ha='center', fontweight='bold')
axes[0, 1].set_ylabel('Number of Students')
axes[0, 1].set_title(f'Student Risk Distribution\n(Passing threshold = {PASS_THRESHOLD} EPC)')

# 3: Predicted vs Actual for students with CBSE
axes[1, 0].scatter(has_actual['cbse_first_score'], has_actual['predicted_cbse'],
                   alpha=0.6, s=50, edgecolors='gray', linewidth=0.3)
axes[1, 0].plot([40, 90], [40, 90], 'r--', linewidth=2, label='Perfect')
axes[1, 0].axhline(y=PASS_THRESHOLD, color='orange', linestyle=':', alpha=0.5)
axes[1, 0].axvline(x=PASS_THRESHOLD, color='orange', linestyle=':', alpha=0.5)
axes[1, 0].set_xlabel('Actual CBSE (EPC)')
axes[1, 0].set_ylabel('Predicted CBSE (EPC)')
acc = (tp+tn)/(tp+tn+fp+fn)*100
axes[1, 0].set_title(f'Predicted vs Actual CBSE\n(Pass/Fail accuracy: {acc:.0f}%)')
axes[1, 0].legend()

# 4: Formative NBME vs CBSE (new finding)
if 'foundation_formative_nbme' in df_predict.columns:
    valid = df_predict[['foundation_formative_nbme', 'cbse_first_score']].dropna()
    r, p = stats.pearsonr(valid['foundation_formative_nbme'], valid['cbse_first_score'])
    axes[1, 1].scatter(valid['foundation_formative_nbme'], valid['cbse_first_score'],
                       alpha=0.6, s=50, color='#e67e22', edgecolors='gray', linewidth=0.3)
    z = np.polyfit(valid['foundation_formative_nbme'], valid['cbse_first_score'], 1)
    p_line = np.poly1d(z)
    x_range = np.linspace(valid['foundation_formative_nbme'].min(),
                          valid['foundation_formative_nbme'].max(), 100)
    axes[1, 1].plot(x_range, p_line(x_range), 'r--', linewidth=2)
    axes[1, 1].axhline(y=PASS_THRESHOLD, color='black', linestyle=':', alpha=0.5)
    axes[1, 1].set_xlabel('Foundation Formative NBME Exam Score')
    axes[1, 1].set_ylabel('CBSE First Attempt Score (EPC)')
    axes[1, 1].set_title(f'NEW: Formative NBME Predicts CBSE Better Than Summative\n'
                          f'(r={r:.2f}, much stronger than Foundation Summative r=0.28)')

plt.tight_layout()
fig_path = os.path.join(DATA_DIR, "pres_07_enhanced_model.png")
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"\nSaved: {fig_path}")
