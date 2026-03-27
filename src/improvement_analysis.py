"""
Improvement Trend Analysis:
Does midterm->final improvement within blocks predict CBSE performance?
Students who improve more = better study ability = better CBSE?
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data", "real")

PASS_THRESHOLD = 64  # EPC

df = pd.read_csv(os.path.join(DATA_DIR, "merged_all_data.csv"))
df_model = df[df['cbse_first_score'].notna()].copy()

exam_cols = [c for c in df.columns if any(x in c for x in ['midterm', 'final'])
             and 'cbse' not in c and 'date' not in c]
df_clean = df_model.dropna(subset=exam_cols).copy()

# Calculate midterm->final improvement per block
blocks = [
    ('Foundation', 'foundation_midterm1', 'foundation_final'),
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

print("=" * 70)
print("  IMPROVEMENT TREND ANALYSIS: Midterm -> Final vs CBSE")
print("=" * 70)

# Calculate improvements
impr_cols = []
for name, mid, fin in blocks:
    col = f"{name}_improvement"
    df_clean[col] = df_clean[fin] - df_clean[mid]
    impr_cols.append(col)

# Overall average improvement
df_clean['avg_improvement'] = df_clean[impr_cols].mean(axis=1)

print("\n--- Per-block improvement correlation with CBSE ---")
print(f"  {'Block':<20} {'Avg Impr':>10} {'r vs CBSE':>10} {'p-value':>10} {'Meaning'}")
print("-" * 70)

impr_corrs = []
for name, mid, fin in blocks:
    col = f"{name}_improvement"
    avg_impr = df_clean[col].mean()
    r, p = stats.pearsonr(df_clean[col], df_clean['cbse_first_score'])
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

    if r > 0.1:
        meaning = "Improvers do better on CBSE"
    elif r < -0.1:
        meaning = "Improvers do WORSE on CBSE (!)"
    else:
        meaning = "No relationship"

    print(f"  {name:<20} {avg_impr:>+9.1f} {r:>+9.3f} {p:>9.4f} {sig:>3}  {meaning}")
    impr_corrs.append((name, r, p, avg_impr))

# Overall average improvement vs CBSE
r_avg, p_avg = stats.pearsonr(df_clean['avg_improvement'], df_clean['cbse_first_score'])
print(f"\n  Average improvement across all blocks:")
print(f"  r = {r_avg:+.3f}, p = {p_avg:.4f}")

if r_avg > 0:
    print(f"  -> Students who improve more from midterm to final tend to score higher on CBSE")
else:
    print(f"  -> Improvement does NOT predict higher CBSE (counterintuitive but important!)")

# ============================================================
# Pass/Fail analysis with threshold
# ============================================================
print(f"\n{'='*70}")
print(f"  PASS/FAIL ANALYSIS (threshold = {PASS_THRESHOLD} EPC)")
print("=" * 70)

df_clean['passed_cbse'] = df_clean['cbse_first_score'] >= PASS_THRESHOLD

n_pass = df_clean['passed_cbse'].sum()
n_fail = (~df_clean['passed_cbse']).sum()
print(f"\n  Passed (>= {PASS_THRESHOLD}): {n_pass} ({n_pass/len(df_clean)*100:.0f}%)")
print(f"  Failed (< {PASS_THRESHOLD}):  {n_fail} ({n_fail/len(df_clean)*100:.0f}%)")

# Average improvement: passers vs failers
pass_impr = df_clean[df_clean['passed_cbse']]['avg_improvement'].mean()
fail_impr = df_clean[~df_clean['passed_cbse']]['avg_improvement'].mean()
t, p = stats.ttest_ind(
    df_clean[df_clean['passed_cbse']]['avg_improvement'],
    df_clean[~df_clean['passed_cbse']]['avg_improvement']
)
print(f"\n  Avg improvement (passers):  {pass_impr:+.1f}")
print(f"  Avg improvement (failers):  {fail_impr:+.1f}")
print(f"  Difference: {pass_impr - fail_impr:+.1f}, p={p:.4f}")

# ============================================================
# Visualization
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Plot 1: Improvement correlation with CBSE per block
names = [ic[0] for ic in impr_corrs]
rs = [ic[1] for ic in impr_corrs]
colors = ['#2ecc71' if r > 0.1 else '#e74c3c' if r < -0.1 else '#95a5a6' for r in rs]
axes[0, 0].barh(range(len(names)), rs, color=colors)
axes[0, 0].set_yticks(range(len(names)))
axes[0, 0].set_yticklabels(names)
axes[0, 0].set_xlabel('Correlation (r) with CBSE')
axes[0, 0].set_title('Does Midterm->Final Improvement Predict CBSE?\n(Green=yes, Red=inverse, Gray=no)')
axes[0, 0].axvline(x=0, color='black', linewidth=1)
axes[0, 0].invert_yaxis()

# Plot 2: Average improvement vs CBSE scatter
axes[0, 1].scatter(df_clean['avg_improvement'], df_clean['cbse_first_score'], alpha=0.5, s=40)
z = np.polyfit(df_clean['avg_improvement'], df_clean['cbse_first_score'], 1)
p_line = np.poly1d(z)
x_range = np.linspace(df_clean['avg_improvement'].min(), df_clean['avg_improvement'].max(), 100)
axes[0, 1].plot(x_range, p_line(x_range), 'r--', label=f'r={r_avg:.3f}, p={p_avg:.4f}')
axes[0, 1].axhline(y=PASS_THRESHOLD, color='orange', linestyle=':', label=f'Pass threshold ({PASS_THRESHOLD})')
axes[0, 1].set_xlabel('Average Midterm->Final Improvement (across all blocks)')
axes[0, 1].set_ylabel('CBSE First Attempt Score (EPC)')
axes[0, 1].set_title('Overall Improvement Trend vs CBSE Score')
axes[0, 1].legend()

# Plot 3: CBSE score distribution with pass/fail
cbse_scores = df_clean['cbse_first_score']
axes[1, 0].hist(cbse_scores[df_clean['passed_cbse']], bins=15, alpha=0.6, color='#2ecc71', label=f'Passed (n={n_pass})')
axes[1, 0].hist(cbse_scores[~df_clean['passed_cbse']], bins=15, alpha=0.6, color='#e74c3c', label=f'Failed (n={n_fail})')
axes[1, 0].axvline(x=PASS_THRESHOLD, color='black', linewidth=2, linestyle='--', label=f'Pass line ({PASS_THRESHOLD})')
axes[1, 0].set_xlabel('CBSE First Attempt Score (EPC)')
axes[1, 0].set_ylabel('Count')
axes[1, 0].set_title(f'CBSE First Attempt Score Distribution\n{n_fail}/{len(df_clean)} ({n_fail/len(df_clean)*100:.0f}%) failed on first attempt')
axes[1, 0].legend()

# Plot 4: Block average vs CBSE with pass/fail coloring
df_clean['block_avg'] = df_clean[exam_cols].mean(axis=1)
pass_mask = df_clean['passed_cbse']
axes[1, 1].scatter(df_clean[pass_mask]['block_avg'], df_clean[pass_mask]['cbse_first_score'],
                   alpha=0.6, s=40, color='#2ecc71', label='Passed')
axes[1, 1].scatter(df_clean[~pass_mask]['block_avg'], df_clean[~pass_mask]['cbse_first_score'],
                   alpha=0.6, s=40, color='#e74c3c', label='Failed')
axes[1, 1].axhline(y=PASS_THRESHOLD, color='black', linewidth=1, linestyle='--')
axes[1, 1].set_xlabel('Average Block Exam Score')
axes[1, 1].set_ylabel('CBSE First Attempt Score (EPC)')
axes[1, 1].set_title('Block Average vs CBSE Score\n(Can block average identify at-risk students?)')
axes[1, 1].legend()

plt.tight_layout()
fig_path = os.path.join(DATA_DIR, "improvement_analysis.png")
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"\nSaved: {fig_path}")
