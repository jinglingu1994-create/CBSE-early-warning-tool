"""
Early Warning Timing Analysis:
How early can we predict CBSE performance?

Tests prediction accuracy after each block is completed,
to find the earliest point where intervention is useful.

Course order (sequential):
Foundation -> MSK -> GI -> Heme -> CV -> Pulm -> Renal -> Neuro -> Endo -> Behavioral Med -> Repro -> CBSE
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data", "real")

# Load data
df = pd.read_csv(os.path.join(DATA_DIR, "merged_all_data.csv"))
df_model = df[df['cbse_first_score'].notna()].copy()

# Course order (sequential, as confirmed by user)
COURSE_ORDER = [
    ("Foundation", ["foundation_midterm1", "foundation_midterm2", "foundation_final"]),
    ("MSK", ["msk_midterm", "msk_final"]),
    ("GI", ["gi_midterm", "gi_final"]),
    ("Heme", ["heme_midterm", "heme_final"]),
    ("CV", ["cv_midterm", "cv_final"]),
    ("Pulm", ["pulm_midterm", "pulm_final"]),
    ("Renal", ["renal_midterm", "renal_final"]),
    ("Neuro", ["neuro_midterm", "neuro_final"]),
    ("Endo", ["endo_midterm", "endo_final"]),
    ("Behav Med", ["behavioral_science_midterm", "behavioral_science_final"]),
    ("Repro", ["repro_midterm", "repro_final"]),
]

# Also test midterm-only checkpoints (earliest possible warning within a block)
MIDTERM_CHECKPOINTS = [
    ("Foundation Mid", ["foundation_midterm1"]),
    ("Foundation", ["foundation_midterm1", "foundation_midterm2", "foundation_final"]),
    ("+ MSK Mid", ["foundation_midterm1", "foundation_midterm2", "foundation_final", "msk_midterm"]),
    ("+ MSK", ["foundation_midterm1", "foundation_midterm2", "foundation_final", "msk_midterm", "msk_final"]),
]

y = df_model['cbse_first_score'].values

print("=" * 70)
print("  EARLY WARNING ANALYSIS: How early can we predict CBSE?")
print("=" * 70)

# ============================================================
# Analysis 1: Cumulative prediction after each block completes
# ============================================================
print("\n[Analysis 1] Prediction accuracy after completing each block:")
print("-" * 70)
print(f"  {'Courses Completed':<35} {'CV R2':>8} {'CV MAE':>8} {'Features':>8}")
print("-" * 70)

cumulative_cols = []
results = []

for course_name, exam_cols in COURSE_ORDER:
    cumulative_cols.extend(exam_cols)

    # Get clean data for current feature set
    df_clean = df_model.dropna(subset=cumulative_cols)
    X = df_clean[cumulative_cols].values
    y_curr = df_clean['cbse_first_score'].values

    if len(df_clean) < 20:
        continue

    # Use Ridge regression (handles multicollinearity better than plain LR)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    model = Ridge(alpha=1.0)
    cv_r2 = cross_val_score(model, X_s, y_curr, cv=5, scoring='r2')
    cv_mae = -cross_val_score(model, X_s, y_curr, cv=5, scoring='neg_mean_absolute_error')

    r2_mean = cv_r2.mean()
    mae_mean = cv_mae.mean()
    n_features = len(cumulative_cols)
    n_students = len(df_clean)

    results.append({
        'checkpoint': f"After {course_name}",
        'course_num': len(results) + 1,
        'cv_r2': r2_mean,
        'cv_mae': mae_mean,
        'n_features': n_features,
        'n_students': n_students,
    })

    indicator = ""
    if r2_mean > 0.4:
        indicator = " *** USEFUL"
    elif r2_mean > 0.25:
        indicator = " ** MODERATE"
    elif r2_mean > 0.1:
        indicator = " * EMERGING"

    print(f"  After {course_name:<30} {r2_mean:>7.3f} {mae_mean:>7.1f} {n_features:>8}{indicator}")

# ============================================================
# Analysis 2: Midterm-only early checkpoints
# ============================================================
print(f"\n{'='*70}")
print("[Analysis 2] Can we predict at MIDTERM checkpoints (even earlier)?")
print("-" * 70)
print(f"  {'Checkpoint':<35} {'CV R2':>8} {'CV MAE':>8}")
print("-" * 70)

# Build progressive midterm-only feature sets
all_mid_checkpoints = []
mid_results = []

cumul_mid = []
for course_name, exam_cols in COURSE_ORDER:
    mid_cols = [c for c in exam_cols if 'midterm' in c]
    if mid_cols:
        cumul_mid.extend(mid_cols)
        checkpoint_name = f"{course_name} midterm"

        df_clean = df_model.dropna(subset=cumul_mid)
        X = df_clean[cumul_mid].values
        y_curr = df_clean['cbse_first_score'].values

        if len(df_clean) < 20:
            continue

        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        model = Ridge(alpha=1.0)
        cv_r2 = cross_val_score(model, X_s, y_curr, cv=5, scoring='r2')
        cv_mae = -cross_val_score(model, X_s, y_curr, cv=5, scoring='neg_mean_absolute_error')

        r2_mean = cv_r2.mean()
        mae_mean = cv_mae.mean()

        mid_results.append({
            'checkpoint': checkpoint_name,
            'cv_r2': r2_mean,
            'cv_mae': mae_mean,
        })

        indicator = ""
        if r2_mean > 0.4:
            indicator = " *** USEFUL"
        elif r2_mean > 0.25:
            indicator = " ** MODERATE"
        elif r2_mean > 0.1:
            indicator = " * EMERGING"

        print(f"  Through {checkpoint_name:<30} {r2_mean:>7.3f} {mae_mean:>7.1f}{indicator}")

# ============================================================
# Visualization
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Plot 1: Cumulative R2 after each block
r2_vals = [r['cv_r2'] for r in results]
mae_vals = [r['cv_mae'] for r in results]
labels = [r['checkpoint'].replace('After ', '') for r in results]

ax1 = axes[0]
bars = ax1.bar(range(len(r2_vals)), r2_vals,
               color=['#e74c3c' if v < 0.1 else '#f39c12' if v < 0.25 else '#3498db' if v < 0.4 else '#2ecc71' for v in r2_vals])
ax1.set_xticks(range(len(labels)))
ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
ax1.set_ylabel('CV R-squared')
ax1.set_title('Prediction Accuracy After Each Block Completed\n(Higher = Better Prediction)')
ax1.axhline(y=0.25, color='orange', linestyle='--', alpha=0.5, label='Moderate threshold')
ax1.axhline(y=0.4, color='green', linestyle='--', alpha=0.5, label='Useful threshold')
ax1.legend(fontsize=8)

# Add MAE as text on bars
for i, (r2, mae) in enumerate(zip(r2_vals, mae_vals)):
    if r2 > 0:
        ax1.text(i, r2 + 0.01, f'MAE={mae:.1f}', ha='center', fontsize=7)

# Plot 2: Midterm-only checkpoints
mid_r2 = [r['cv_r2'] for r in mid_results]
mid_labels = [r['checkpoint'].replace(' midterm', '\nmid') for r in mid_results]

ax2 = axes[1]
ax2.plot(range(len(mid_r2)), mid_r2, 'o-', color='#e67e22', linewidth=2, markersize=8, label='Midterm-only')
ax2.fill_between(range(len(mid_r2)), mid_r2, alpha=0.2, color='#e67e22')
ax2.set_xticks(range(len(mid_labels)))
ax2.set_xticklabels(mid_labels, rotation=45, ha='right', fontsize=8)
ax2.set_ylabel('CV R-squared')
ax2.set_title('Prediction Using ONLY Midterm Scores\n(How early can we warn?)')
ax2.axhline(y=0.25, color='orange', linestyle='--', alpha=0.5, label='Moderate threshold')
ax2.axhline(y=0.4, color='green', linestyle='--', alpha=0.5, label='Useful threshold')
ax2.legend(fontsize=8)

plt.tight_layout()
fig_path = os.path.join(DATA_DIR, "early_warning_timing.png")
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"\nSaved: {fig_path}")

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*70}")
print("CONCLUSION: When Can We Start Warning?")
print("=" * 70)

# Find first block where R2 > 0.25 (moderate) and > 0.4 (useful)
for r in results:
    if r['cv_r2'] > 0.25:
        print(f"\n  MODERATE prediction possible: {r['checkpoint']} (R2={r['cv_r2']:.3f}, MAE={r['cv_mae']:.1f})")
        break

for r in results:
    if r['cv_r2'] > 0.4:
        print(f"  USEFUL prediction possible:   {r['checkpoint']} (R2={r['cv_r2']:.3f}, MAE={r['cv_mae']:.1f})")
        break

print(f"\n  Using ALL blocks: R2={results[-1]['cv_r2']:.3f}, MAE={results[-1]['cv_mae']:.1f}")
print(f"\n  Implication: The school doesn't need to wait until all courses are done.")
print(f"  Early intervention can start after the first few blocks show a pattern.")


if __name__ == "__main__":
    pass
