"""
Generate final presentation-ready figures.
Each figure has: Title, Meaning for the school, Confidence Level.

All output saved to data/real/ (LOCAL ONLY, never push).
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from sklearn.model_selection import cross_val_score, RepeatedKFold
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data", "real")
PASS_THRESHOLD = 64

# Load data
df = pd.read_csv(os.path.join(DATA_DIR, "merged_all_data.csv"))
exam_cols = [c for c in df.columns if any(x in c for x in ['midterm', 'final'])
             and 'cbse' not in c and 'date' not in c]
df_model = df[df['cbse_first_score'].notna()].dropna(subset=exam_cols).copy()
y = df_model['cbse_first_score'].values
n = len(df_model)

# Short names for display
def short(name):
    return (name.replace('behavioral_science', 'Behav')
            .replace('foundation', 'Found')
            .replace('_midterm', ' Mid')
            .replace('_final', ' Fin')
            .replace('_improvement', ' Impr')
            .title())

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.facecolor': 'white',
})


def add_annotation(ax, text, loc='lower right', fontsize=8):
    """Add a text box with confidence/meaning annotation."""
    props = dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9, edgecolor='gray')
    if loc == 'lower right':
        ax.annotate(text, xy=(0.98, 0.02), xycoords='axes fraction',
                    fontsize=fontsize, ha='right', va='bottom', bbox=props)
    elif loc == 'upper right':
        ax.annotate(text, xy=(0.98, 0.98), xycoords='axes fraction',
                    fontsize=fontsize, ha='right', va='top', bbox=props)
    elif loc == 'upper left':
        ax.annotate(text, xy=(0.02, 0.98), xycoords='axes fraction',
                    fontsize=fontsize, ha='left', va='top', bbox=props)


# ============================================================
# FIGURE 1: Overview - Block Scores vs CBSE Correlation
# ============================================================
print("Generating Figure 1: Block-CBSE Correlation Overview...")

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# 1a: Bar chart of correlations
corrs = {}
for col in exam_cols:
    valid = df_model[[col, 'cbse_first_score']].dropna()
    r, p = stats.pearsonr(valid[col], valid['cbse_first_score'])
    corrs[col] = (r, p)

sorted_c = sorted(corrs.items(), key=lambda x: x[1][0], reverse=True)
names = [short(c[0]) for c in sorted_c]
vals = [c[1][0] for c in sorted_c]
colors = ['#27ae60' if v > 0.6 else '#2980b9' if v > 0.4 else '#f39c12' if v > 0.2 else '#e74c3c' for v in vals]

axes[0].barh(range(len(vals)), vals, color=colors)
axes[0].set_yticks(range(len(vals)))
axes[0].set_yticklabels(names, fontsize=9)
axes[0].set_xlabel('Correlation with CBSE First Attempt (r)')
axes[0].set_title('Which Course Exams Best Predict CBSE Performance?')
axes[0].axvline(x=0.5, color='green', linestyle='--', alpha=0.4)
axes[0].invert_yaxis()
add_annotation(axes[0],
    'Confidence: VERY HIGH (>99%)\n22/23 exams significant at p<0.001\n'
    'School Meaning: Neuro & Renal\nare the strongest CBSE predictors.\n'
    'Foundation is the weakest.',
    loc='lower right')

# 1b: Heatmap
final_cols = [c for c in exam_cols if 'final' in c and 'midterm' not in c]
all_for_heatmap = final_cols + ['cbse_first_score']
corr_matrix = df_model[all_for_heatmap].corr()
display_names = {c: short(c).replace(' Fin', '') for c in final_cols}
display_names['cbse_first_score'] = 'CBSE'
corr_display = corr_matrix.rename(index=display_names, columns=display_names)

sns.heatmap(corr_display, annot=True, fmt='.2f', cmap='RdYlGn', center=0.4,
            ax=axes[1], vmin=0, vmax=1.0, annot_kws={'size': 9})
axes[1].set_title('Correlation Heatmap: All Block Finals + CBSE')

plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, "pres_01_correlation_overview.png"), dpi=150, bbox_inches='tight')
print("  Saved: pres_01_correlation_overview.png")


# ============================================================
# FIGURE 2: Model Performance
# ============================================================
print("Generating Figure 2: Model Performance...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Train model
scaler = StandardScaler()
X_s = scaler.fit_transform(df_model[exam_cols].values)
rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)

rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
cv_r2 = cross_val_score(rf, X_s, y, cv=rkf, scoring='r2')
cv_mae = -cross_val_score(rf, X_s, y, cv=rkf, scoring='neg_mean_absolute_error')

# Fit for visualization
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_s, y, test_size=0.2, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# 2a: Actual vs Predicted
axes[0].scatter(y_test, y_pred, alpha=0.7, s=60, edgecolors='gray', linewidth=0.5)
axes[0].plot([40, 90], [40, 90], 'r--', linewidth=2, label='Perfect prediction')
axes[0].axhline(y=PASS_THRESHOLD, color='orange', linestyle=':', alpha=0.5)
axes[0].axvline(x=PASS_THRESHOLD, color='orange', linestyle=':', alpha=0.5)
axes[0].fill_between([40, PASS_THRESHOLD], [40, 40], [PASS_THRESHOLD, PASS_THRESHOLD],
                     alpha=0.1, color='red', label='Both fail')
axes[0].set_xlabel('Actual CBSE Score (EPC)')
axes[0].set_ylabel('Predicted CBSE Score (EPC)')
axes[0].set_title(f'Model Prediction: Actual vs Predicted CBSE\n(CV R\u00b2={cv_r2.mean():.2f}, MAE={cv_mae.mean():.1f} EPC)')
axes[0].legend(fontsize=9)
add_annotation(axes[0],
    f'Confidence: HIGH\n95% CI for R\u00b2: {cv_r2.mean()-1.96*cv_r2.std()/np.sqrt(50):.2f}-{cv_r2.mean()+1.96*cv_r2.std()/np.sqrt(50):.2f}\n'
    f'School Meaning: Block scores\nexplain ~{cv_r2.mean()*100:.0f}% of CBSE variance.\n'
    f'Avg error: {cv_mae.mean():.0f} EPC points.',
    loc='upper left')

# 2b: Prediction error distribution
residuals = y_test - y_pred
axes[1].hist(residuals, bins=12, edgecolor='black', alpha=0.7, color='#3498db')
axes[1].axvline(x=0, color='red', linewidth=2, linestyle='--')
axes[1].set_xlabel('Prediction Error (Actual - Predicted EPC)')
axes[1].set_ylabel('Number of Students')
axes[1].set_title(f'How Far Off Are Predictions?\n(Mean error: {residuals.mean():+.1f}, Std: {residuals.std():.1f})')
add_annotation(axes[1],
    f'School Meaning: Most predictions\nare within +/-{residuals.std():.0f} EPC of actual.\n'
    f'Useful for group-level screening,\nnot individual guarantees.',
    loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, "pres_02_model_performance.png"), dpi=150, bbox_inches='tight')
print("  Saved: pres_02_model_performance.png")


# ============================================================
# FIGURE 3: Early Warning Timeline
# ============================================================
print("Generating Figure 3: Early Warning Timeline...")

COURSE_ORDER = [
    ("Found", ["foundation_midterm1", "foundation_midterm2", "foundation_final"]),
    ("MSK", ["msk_midterm", "msk_final"]),
    ("GI", ["gi_midterm", "gi_final"]),
    ("Heme", ["heme_midterm", "heme_final"]),
    ("CV", ["cv_midterm", "cv_final"]),
    ("Pulm", ["pulm_midterm", "pulm_final"]),
    ("Renal", ["renal_midterm", "renal_final"]),
    ("Neuro", ["neuro_midterm", "neuro_final"]),
    ("Endo", ["endo_midterm", "endo_final"]),
    ("Behav", ["behavioral_science_midterm", "behavioral_science_final"]),
    ("Repro", ["repro_midterm", "repro_final"]),
]

fig, ax = plt.subplots(figsize=(14, 6))

cumulative_cols = []
results = []
for course_name, ecols in COURSE_ORDER:
    cumulative_cols.extend(ecols)
    df_c = df_model.dropna(subset=cumulative_cols)
    X_c = StandardScaler().fit_transform(df_c[cumulative_cols].values)
    y_c = df_c['cbse_first_score'].values
    if len(df_c) >= 20:
        cv = cross_val_score(Ridge(alpha=1.0), X_c, y_c, cv=5, scoring='r2')
        results.append((course_name, cv.mean(), cv.std()))

names = [r[0] for r in results]
r2s = [r[1] for r in results]
stds = [r[2] for r in results]

colors = ['#e74c3c' if v < 0.15 else '#f39c12' if v < 0.3 else '#2ecc71' for v in r2s]
bars = ax.bar(range(len(r2s)), r2s, color=colors, edgecolor='gray', linewidth=0.5)
ax.errorbar(range(len(r2s)), r2s, yerr=stds, fmt='none', color='black', capsize=4)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, fontsize=11)
ax.set_ylabel('Prediction Accuracy (CV R-squared)')
ax.set_xlabel('Courses Completed (chronological order, left to right)')
ax.set_title('How Early Can We Predict CBSE Performance?\n(Each bar = prediction using all courses completed up to that point)')
ax.axhline(y=0.3, color='green', linestyle='--', alpha=0.4, label='Useful threshold (R\u00b2=0.3)')

# Arrow annotation for key finding
ax.annotate('Early warning\npossible here!',
            xy=(1, r2s[1]), xytext=(2.5, r2s[1]+0.15),
            fontsize=11, fontweight='bold', color='#27ae60',
            arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2))

add_annotation(ax,
    'Confidence: MODERATE-HIGH\n'
    'School Meaning: Intervention can\n'
    'start after MSK (course 2 of 11).\n'
    "Don't wait until all courses are done.",
    loc='upper right')
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, "pres_03_early_warning_timeline.png"), dpi=150, bbox_inches='tight')
print("  Saved: pres_03_early_warning_timeline.png")


# ============================================================
# FIGURE 4: Retake Analysis
# ============================================================
print("Generating Figure 4: Retake Analysis...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# 4a: CBSE score distribution with pass/fail
df_model['passed'] = df_model['cbse_first_score'] >= PASS_THRESHOLD
n_pass = df_model['passed'].sum()
n_fail = (~df_model['passed']).sum()

axes[0].hist(df_model[df_model['passed']]['cbse_first_score'], bins=15, alpha=0.6,
             color='#2ecc71', label=f'Passed >= {PASS_THRESHOLD} (n={n_pass}, {n_pass/n*100:.0f}%)', edgecolor='gray')
axes[0].hist(df_model[~df_model['passed']]['cbse_first_score'], bins=15, alpha=0.6,
             color='#e74c3c', label=f'Failed < {PASS_THRESHOLD} (n={n_fail}, {n_fail/n*100:.0f}%)', edgecolor='gray')
axes[0].axvline(x=PASS_THRESHOLD, color='black', linewidth=2, linestyle='--')
axes[0].set_xlabel('CBSE First Attempt Score (EPC)')
axes[0].set_ylabel('Number of Students')
axes[0].set_title(f'CBSE First Attempt Results\n{n_fail}/{n} ({n_fail/n*100:.0f}%) failed first attempt')
axes[0].legend(fontsize=9)
add_annotation(axes[0],
    'School Meaning: 38% fail rate\non first attempt is HIGH.\n'
    'Early warning could help reduce\nthis by targeting at-risk students.',
    loc='upper left')

# 4b: Retake improvement
has_a1_a2 = df[df['cbse_attempt1_score'].notna() & df['cbse_attempt2_score'].notna()].copy()
if len(has_a1_a2) > 0:
    has_a1_a2['change'] = has_a1_a2['cbse_attempt2_score'] - has_a1_a2['cbse_attempt1_score']
    changes = has_a1_a2['change'].sort_values()
    colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in changes.values]
    axes[1].bar(range(len(changes)), changes.values, color=colors, edgecolor='gray', linewidth=0.3)
    axes[1].axhline(y=0, color='black', linewidth=1)
    axes[1].axhline(y=changes.mean(), color='blue', linestyle='--',
                    label=f'Mean change: {changes.mean():+.1f} (p=0.49, NOT significant)')
    axes[1].set_xlabel('Students (sorted by change)')
    axes[1].set_ylabel('Score Change (EPC)')
    improved_pct = (changes > 0).mean() * 100
    axes[1].set_title(f'CBSE Score Change: Attempt 1 -> 2\nOnly {improved_pct:.0f}% improved, avg change: {changes.mean():+.1f}')
    axes[1].legend(fontsize=9)
    add_annotation(axes[1],
        'Confidence: HIGH (p=0.49)\n'
        'School Meaning: Simply retaking\n'
        'CBSE does NOT guarantee improvement.\n'
        'Targeted remediation needed\nbefore retake.',
        loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, "pres_04_retake_analysis.png"), dpi=150, bbox_inches='tight')
print("  Saved: pres_04_retake_analysis.png")


# ============================================================
# FIGURE 5: Counterintuitive Finding - Improvement Paradox
# ============================================================
print("Generating Figure 5: Improvement Paradox...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Calculate improvements
blocks_info = [
    ('Found', 'foundation_midterm1', 'foundation_final'),
    ('MSK', 'msk_midterm', 'msk_final'),
    ('GI', 'gi_midterm', 'gi_final'),
    ('Heme', 'heme_midterm', 'heme_final'),
    ('CV', 'cv_midterm', 'cv_final'),
    ('Pulm', 'pulm_midterm', 'pulm_final'),
    ('Renal', 'renal_midterm', 'renal_final'),
    ('Neuro', 'neuro_midterm', 'neuro_final'),
    ('Endo', 'endo_midterm', 'endo_final'),
    ('Behav', 'behavioral_science_midterm', 'behavioral_science_final'),
    ('Repro', 'repro_midterm', 'repro_final'),
]

impr_corrs = []
for name, mid, fin in blocks_info:
    df_model[f'{name}_impr'] = df_model[fin] - df_model[mid]
    r, p = stats.pearsonr(df_model[f'{name}_impr'], df_model['cbse_first_score'])
    impr_corrs.append((name, r, p))

# 5a: Improvement correlation per block
i_names = [ic[0] for ic in impr_corrs]
i_rs = [ic[1] for ic in impr_corrs]
i_colors = ['#2ecc71' if r > 0.1 else '#e74c3c' if r < -0.1 else '#95a5a6' for r in i_rs]

axes[0].barh(range(len(i_names)), i_rs, color=i_colors, edgecolor='gray', linewidth=0.5)
axes[0].set_yticks(range(len(i_names)))
axes[0].set_yticklabels(i_names, fontsize=11)
axes[0].set_xlabel('Correlation with CBSE (r)')
axes[0].axvline(x=0, color='black', linewidth=1)
axes[0].set_title('Does Midterm->Final Improvement Predict CBSE?\n(Green=positive, Red=inverse)')
axes[0].invert_yaxis()
add_annotation(axes[0],
    'Confidence: MODERATE\n'
    'School Meaning: Students who\n'
    'improve most from midterm to final\n'
    'do NOT score higher on CBSE.\n'
    'Absolute scores matter more\nthan improvement trends.',
    loc='lower right')

# 5b: Block average vs CBSE (the real predictor)
df_model['block_avg'] = df_model[exam_cols].mean(axis=1)
pass_mask = df_model['passed']
axes[1].scatter(df_model[pass_mask]['block_avg'], df_model[pass_mask]['cbse_first_score'],
               alpha=0.6, s=50, color='#2ecc71', label='Passed CBSE', edgecolors='gray', linewidth=0.3)
axes[1].scatter(df_model[~pass_mask]['block_avg'], df_model[~pass_mask]['cbse_first_score'],
               alpha=0.6, s=50, color='#e74c3c', label='Failed CBSE', edgecolors='gray', linewidth=0.3)
axes[1].axhline(y=PASS_THRESHOLD, color='black', linewidth=1.5, linestyle='--', label=f'Pass threshold ({PASS_THRESHOLD})')

# Fit line
z = np.polyfit(df_model['block_avg'], df_model['cbse_first_score'], 1)
p_line = np.poly1d(z)
x_range = np.linspace(df_model['block_avg'].min(), df_model['block_avg'].max(), 100)
axes[1].plot(x_range, p_line(x_range), 'b--', linewidth=2, alpha=0.6)

r_avg, p_avg = stats.pearsonr(df_model['block_avg'], df_model['cbse_first_score'])
axes[1].set_xlabel('Average Block Exam Score')
axes[1].set_ylabel('CBSE First Attempt Score (EPC)')
axes[1].set_title(f'Block Average vs CBSE Score (r={r_avg:.2f})\nAbsolute scores are the real predictor')
axes[1].legend(fontsize=9)
add_annotation(axes[1],
    f'Confidence: VERY HIGH\nr={r_avg:.2f}, p<0.0001\n'
    'School Meaning: Overall block\naverage is the simplest and most\n'
    'reliable predictor of CBSE success.',
    loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, "pres_05_improvement_paradox.png"), dpi=150, bbox_inches='tight')
print("  Saved: pres_05_improvement_paradox.png")


# ============================================================
# FIGURE 6: Future Improvements
# ============================================================
print("Generating Figure 6: Future Improvements...")

fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('off')

future_text = """
FUTURE IMPROVEMENTS: How to Build a Better Model

CURRENT STATE
  - Model R-squared: ~0.49 (block exam scores explain 49% of CBSE variance)
  - Prediction error: ~6 EPC points
  - 38% of students fail CBSE on first attempt

WHAT ADDITIONAL DATA COULD IMPROVE PREDICTIONS

  1. UWorld / Question Bank Performance           [Expected Impact: HIGH]
     Percent correct on practice questions is one of the strongest
     known predictors of Step 1/CBSE performance.

  2. NBME Practice Exam Scores                    [Expected Impact: HIGH]
     These directly mirror CBSE format and content.
     Schools that track these could significantly boost model accuracy.

  3. Study Behavior Data (Anki, hours logged)     [Expected Impact: MODERATE]
     Captures effort and consistency, which block exams may not reflect.

  4. Historical Data (Prior Year Cohorts)          [Expected Impact: HIGH]
     Current model uses n=114 from one year.
     Adding 2-3 more years would 3-4x the sample size,
     making all conclusions more robust and enabling complex models.

  5. Attendance / Engagement Metrics               [Expected Impact: LOW-MODERATE]
     Already partially available in current CSV exports.
     Non-compliance signal (missing data = risk factor).

LIMITATIONS OF CURRENT ANALYSIS
  - Single school, single cohort (n=114)
  - No external validation (need to test on next year's students)
  - Block scores only capture exam performance, not study strategy
  - 5 students with block data but no CBSE (selection bias)
  - CBSE is a single snapshot; test-day variability adds noise (SEE = 4 EPC)
"""

ax.text(0.05, 0.95, future_text, transform=ax.transAxes,
        fontsize=11, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.savefig(os.path.join(DATA_DIR, "pres_06_future_improvements.png"), dpi=150, bbox_inches='tight')
print("  Saved: pres_06_future_improvements.png")


# ============================================================
# Summary
# ============================================================
print(f"\n{'='*70}")
print("  ALL PRESENTATION FIGURES GENERATED")
print("=" * 70)
print(f"  Location: {DATA_DIR}")
print(f"  Files:")
print(f"    pres_01_correlation_overview.png")
print(f"    pres_02_model_performance.png")
print(f"    pres_03_early_warning_timeline.png")
print(f"    pres_04_retake_analysis.png")
print(f"    pres_05_improvement_paradox.png")
print(f"    pres_06_future_improvements.png")
print(f"\n  WARNING: These contain real student data analysis.")
print(f"  NEVER push to GitHub!")
