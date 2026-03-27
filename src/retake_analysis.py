"""
Retake Analysis:
1. Score changes across attempts
2. Who improves, who doesn't
3. Can block scores predict who will need to retake?
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, RepeatedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data", "real")

df = pd.read_csv(os.path.join(DATA_DIR, "merged_all_data.csv"))

exam_cols = [c for c in df.columns if any(x in c for x in ['midterm', 'final'])
             and 'cbse' not in c and 'date' not in c]

# ============================================================
# Part 1: Retake score changes
# ============================================================
print("=" * 70)
print("  RETAKE ANALYSIS")
print("=" * 70)

# Students with multiple attempts
df_retake = df[df['cbse_total_attempts'] >= 2].copy()
df_3x = df[df['cbse_total_attempts'] == 3].copy()

print(f"\nStudents who took CBSE once: {(df['cbse_total_attempts'] == 1).sum()}")
print(f"Students who retook (2x): {(df['cbse_total_attempts'] == 2).sum()}")
print(f"Students who retook (3x): {(df['cbse_total_attempts'] == 3).sum()}")

# Score change from attempt 1 to 2
has_a1_a2 = df[df['cbse_attempt1_score'].notna() & df['cbse_attempt2_score'].notna()].copy()
if len(has_a1_a2) > 0:
    has_a1_a2['change_1to2'] = has_a1_a2['cbse_attempt2_score'] - has_a1_a2['cbse_attempt1_score']

    print(f"\n--- Attempt 1 -> Attempt 2 (n={len(has_a1_a2)}) ---")
    print(f"  Average change: {has_a1_a2['change_1to2'].mean():+.1f} EPC points")
    print(f"  Median change:  {has_a1_a2['change_1to2'].median():+.1f}")
    print(f"  Improved: {(has_a1_a2['change_1to2'] > 0).sum()} ({(has_a1_a2['change_1to2'] > 0).mean()*100:.0f}%)")
    print(f"  Same/Worse: {(has_a1_a2['change_1to2'] <= 0).sum()} ({(has_a1_a2['change_1to2'] <= 0).mean()*100:.0f}%)")

    # Is the improvement statistically significant?
    t_stat, p_val = stats.ttest_1samp(has_a1_a2['change_1to2'], 0)
    print(f"  t-test (is change != 0?): t={t_stat:.2f}, p={p_val:.4f}")
    if p_val < 0.05:
        print(f"  -> Statistically significant change")
    else:
        print(f"  -> NOT significant (retaking may not help on average)")

# Score change from attempt 2 to 3
has_a2_a3 = df[df['cbse_attempt2_score'].notna() & df['cbse_attempt3_score'].notna()].copy()
if len(has_a2_a3) > 0:
    has_a2_a3['change_2to3'] = has_a2_a3['cbse_attempt3_score'] - has_a2_a3['cbse_attempt2_score']

    print(f"\n--- Attempt 2 -> Attempt 3 (n={len(has_a2_a3)}) ---")
    print(f"  Average change: {has_a2_a3['change_2to3'].mean():+.1f} EPC points")
    print(f"  Improved: {(has_a2_a3['change_2to3'] > 0).sum()} ({(has_a2_a3['change_2to3'] > 0).mean()*100:.0f}%)")
    print(f"  Same/Worse: {(has_a2_a3['change_2to3'] <= 0).sum()} ({(has_a2_a3['change_2to3'] <= 0).mean()*100:.0f}%)")

# ============================================================
# Part 2: Can block scores predict who needs to retake?
# ============================================================
print(f"\n{'='*70}")
print("  Can block scores predict who needs to retake?")
print("=" * 70)

# Define: needed_retake = took more than 1 attempt
df_with_cbse = df[df['cbse_total_attempts'].notna() & df['cbse_total_attempts'] > 0].copy()
df_with_cbse = df_with_cbse.dropna(subset=exam_cols)
df_with_cbse['needed_retake'] = (df_with_cbse['cbse_total_attempts'] > 1).astype(int)

print(f"\nStudents with complete data: {len(df_with_cbse)}")
print(f"  Needed retake: {df_with_cbse['needed_retake'].sum()} ({df_with_cbse['needed_retake'].mean()*100:.0f}%)")
print(f"  Passed first time: {(df_with_cbse['needed_retake'] == 0).sum()} ({(df_with_cbse['needed_retake'] == 0).mean()*100:.0f}%)")

X = df_with_cbse[exam_cols].values
y_retake = df_with_cbse['needed_retake'].values

scaler = StandardScaler()
X_s = scaler.fit_transform(X)

# Logistic Regression for retake prediction
lr = LogisticRegression(max_iter=1000, random_state=42)
rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
cv_auc = cross_val_score(lr, X_s, y_retake, cv=rkf, scoring='roc_auc')
cv_acc = cross_val_score(lr, X_s, y_retake, cv=rkf, scoring='accuracy')

mean_auc = cv_auc.mean()
ci_low_auc = mean_auc - 1.96 * cv_auc.std() / np.sqrt(len(cv_auc))
ci_high_auc = mean_auc + 1.96 * cv_auc.std() / np.sqrt(len(cv_auc))

print(f"\n  Logistic Regression (predicting retake yes/no):")
print(f"  CV AUC: {mean_auc:.3f} (95% CI: {ci_low_auc:.3f}-{ci_high_auc:.3f})")
print(f"  CV Accuracy: {cv_acc.mean():.3f}")
print(f"  (AUC=0.5 is random, AUC=1.0 is perfect)")

if mean_auc > 0.8:
    print(f"  -> GOOD: Block scores can identify students likely to need retake")
elif mean_auc > 0.7:
    print(f"  -> MODERATE: Useful signal but not definitive")
else:
    print(f"  -> WEAK: Block scores alone are not enough to predict retake")

# Which block scores best distinguish retakers from non-retakers?
print(f"\n--- Block scores: Retakers vs Non-retakers ---")
print(f"  {'Exam':<35} {'Non-retake':>10} {'Retake':>10} {'Diff':>8} {'p-value':>10}")
print("-" * 75)

sig_diffs = []
for col in exam_cols:
    group0 = df_with_cbse[df_with_cbse['needed_retake'] == 0][col].dropna()
    group1 = df_with_cbse[df_with_cbse['needed_retake'] == 1][col].dropna()
    if len(group0) > 5 and len(group1) > 5:
        t, p = stats.ttest_ind(group0, group1)
        diff = group0.mean() - group1.mean()
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        short = col.replace('behavioral_science', 'behav')
        print(f"  {short:<35} {group0.mean():>9.1f} {group1.mean():>9.1f} {diff:>+7.1f} {p:>9.4f} {sig}")
        sig_diffs.append((col, diff, p))

# ============================================================
# Part 3: Predict dismissed students
# ============================================================
print(f"\n{'='*70}")
print("  PREDICT: What would dismissed students have scored on CBSE?")
print("=" * 70)

df_no_cbse = df[df['cbse_total_attempts'].isna() | (df['cbse_total_attempts'] == 0)].copy()
df_no_cbse = df_no_cbse.dropna(subset=exam_cols)

if len(df_no_cbse) > 0:
    from sklearn.ensemble import RandomForestRegressor

    # Train on all students with CBSE
    df_train = df.dropna(subset=exam_cols + ['cbse_first_score'])
    X_train = scaler.fit_transform(df_train[exam_cols].values)
    y_train = df_train['cbse_first_score'].values

    rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)

    X_dismissed = scaler.transform(df_no_cbse[exam_cols].values)
    predictions = rf.predict(X_dismissed)

    print(f"\nDismissed/LOA students and their predicted CBSE scores:")
    for i, (_, row) in enumerate(df_no_cbse.iterrows()):
        name = row.get('name', 'Unknown')
        sid = int(row['student_id'])
        pred = predictions[i]
        risk = "HIGH RISK" if pred < 60 else "AT RISK" if pred < 65 else "BORDERLINE" if pred < 70 else "LIKELY OK"
        print(f"  {sid} {name:<30} Predicted CBSE: {pred:.1f} [{risk}]")
else:
    print("  No dismissed students with complete block data found.")

# ============================================================
# Visualization
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Plot 1: Score change distribution (attempt 1 -> 2)
if len(has_a1_a2) > 0:
    changes = has_a1_a2['change_1to2']
    colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in changes]
    axes[0, 0].bar(range(len(changes)), changes.sort_values().values,
                   color=['#2ecc71' if c > 0 else '#e74c3c' for c in changes.sort_values().values])
    axes[0, 0].axhline(y=0, color='black', linewidth=1)
    axes[0, 0].axhline(y=changes.mean(), color='blue', linestyle='--', label=f'Mean: {changes.mean():+.1f}')
    axes[0, 0].set_xlabel('Students (sorted)')
    axes[0, 0].set_ylabel('Score Change (EPC)')
    improved_pct = (changes > 0).mean() * 100
    axes[0, 0].set_title(f'CBSE Score Change: Attempt 1 -> 2\n{improved_pct:.0f}% improved, mean change: {changes.mean():+.1f}')
    axes[0, 0].legend()

# Plot 2: Retaker vs Non-retaker block averages
df_with_cbse['block_avg'] = df_with_cbse[exam_cols].mean(axis=1)
g0 = df_with_cbse[df_with_cbse['needed_retake'] == 0]['block_avg']
g1 = df_with_cbse[df_with_cbse['needed_retake'] == 1]['block_avg']
axes[0, 1].hist(g0, bins=15, alpha=0.6, label=f'Passed 1st time (n={len(g0)})', color='#2ecc71')
axes[0, 1].hist(g1, bins=15, alpha=0.6, label=f'Needed retake (n={len(g1)})', color='#e74c3c')
axes[0, 1].set_xlabel('Average Block Exam Score')
axes[0, 1].set_ylabel('Count')
axes[0, 1].set_title(f'Block Score Distribution: Retakers vs Non-retakers\nAUC={mean_auc:.3f}')
axes[0, 1].legend()

# Plot 3: Top differences between retakers and non-retakers
sig_sorted = sorted(sig_diffs, key=lambda x: abs(x[1]), reverse=True)[:12]
diff_names = [s[0].replace('behavioral_science', 'behav').replace('_midterm', '_mid').replace('_final', '_fin') for s in sig_sorted]
diff_vals = [s[1] for s in sig_sorted]
diff_colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in diff_vals]
axes[1, 0].barh(range(len(diff_vals)), diff_vals, color=diff_colors)
axes[1, 0].set_yticks(range(len(diff_names)))
axes[1, 0].set_yticklabels(diff_names, fontsize=9)
axes[1, 0].set_xlabel('Score Difference (Non-retake minus Retake)')
axes[1, 0].set_title('Where Retakers Score Lower\n(Bigger bar = bigger gap)')
axes[1, 0].invert_yaxis()

# Plot 4: First attempt scores - retakers vs one-timers
df_a1 = df[df['cbse_attempt1_score'].notna()].copy()
df_a1['retook'] = df_a1['cbse_total_attempts'] > 1
g_once = df_a1[~df_a1['retook']]['cbse_attempt1_score']
g_retook = df_a1[df_a1['retook']]['cbse_attempt1_score']
axes[1, 1].hist(g_once, bins=15, alpha=0.6, label=f'One-time (n={len(g_once)})', color='#3498db')
axes[1, 1].hist(g_retook, bins=15, alpha=0.6, label=f'Retook later (n={len(g_retook)})', color='#e67e22')
axes[1, 1].set_xlabel('CBSE First Attempt Score (EPC)')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_title('First Attempt CBSE: Who Retook vs Who Didn\'t')
axes[1, 1].legend()

plt.tight_layout()
fig_path = os.path.join(DATA_DIR, "retake_analysis.png")
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"\nSaved: {fig_path}")
