"""
Enhanced data extraction: adds Formative NBME scores and Remediation flags.
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data", "real")
SOURCE_DIR = r"C:\Users\lexie\Downloads\(No subject)"

# Formative exam columns (raw score columns, not the "Final Score" variants)
FORMATIVE_CONFIG = {
    "foundation_formative_nbme": {
        "file_pattern": "foundation",
        "col": "Formative NBME Exam  (86587)",
    },
    "msk_formative": {
        "file_pattern": "MSK",
        "col": "Formative Exam (92125)",
    },
    "heme_formative": {
        "file_pattern": "Heme",
        "col": "Hematology Formative Exam (91060)",
    },
}

# Remediation columns - presence of a score means student needed remediation
REMEDIATION_CONFIG = {
    "foundation": {"file_pattern": "foundation", "col": "Remediation (100071)"},
    "msk": {"file_pattern": "MSK", "col": "Remediation Final Summative Exam (94275)"},
    "heme": {"file_pattern": "Heme", "col": "Remediation Final Summative Exam (94274)"},
    "cv": {"file_pattern": "CV", "col": "Remediation (99463)"},
    "behavioral_science": {"file_pattern": "Behavior", "col": "Winter Remediation 1/2/26 (108330)"},
    "endo": {"file_pattern": "Endo", "col": "Winter Remediation 1/2/26 (108329)"},
}


def find_csv(source_dir, pattern):
    for f in os.listdir(source_dir):
        if f.endswith(".csv") and pattern in f:
            return os.path.join(source_dir, f)
    return None


def read_and_clean(csv_path):
    df = pd.read_csv(csv_path)
    df = df[~df["Student"].str.strip().str.startswith("Points Possible", na=False)]
    df = df.dropna(subset=["Student"])
    df["student_id"] = df["SIS Login ID"].astype(str).str.extract(r"(\d+)")[0]
    df["student_id"] = pd.to_numeric(df["student_id"], errors="coerce")
    df = df.dropna(subset=["student_id"])
    df["student_id"] = df["student_id"].astype(int)
    df = df[(df["student_id"] >= 10000) & (df["student_id"] <= 99999)]
    df = df[~df["Student"].str.contains("Test", case=False, na=False)]
    return df


def main():
    # Load existing merged data
    merged = pd.read_csv(os.path.join(DATA_DIR, "merged_all_data.csv"))
    print(f"Existing merged data: {len(merged)} students")

    # ============================================================
    # Extract Formative NBME scores
    # ============================================================
    print(f"\n{'='*60}")
    print("Extracting Formative Exam Scores...")
    print("=" * 60)

    for feat_name, config in FORMATIVE_CONFIG.items():
        csv_path = find_csv(SOURCE_DIR, config["file_pattern"])
        if csv_path is None:
            print(f"  WARNING: No file for {feat_name}")
            continue

        df = read_and_clean(csv_path)
        col = config["col"]
        if col in df.columns:
            df[feat_name] = pd.to_numeric(df[col], errors="coerce")
            # Merge into main dataset
            merge_cols = df[["student_id", feat_name]].dropna()
            n_before = merged[feat_name].notna().sum() if feat_name in merged.columns else 0
            merged = merged.merge(merge_cols, on="student_id", how="left", suffixes=("_old", ""))
            # Handle duplicate columns from re-runs
            if f"{feat_name}_old" in merged.columns:
                merged[feat_name] = merged[feat_name].fillna(merged[f"{feat_name}_old"])
                merged = merged.drop(columns=[f"{feat_name}_old"])
            n_after = merged[feat_name].notna().sum()
            print(f"  {feat_name}: {n_after} students have scores")
        else:
            print(f"  WARNING: Column '{col}' not found")

    # ============================================================
    # Extract Remediation flags
    # ============================================================
    print(f"\n{'='*60}")
    print("Extracting Remediation Flags...")
    print("=" * 60)

    for block, config in REMEDIATION_CONFIG.items():
        csv_path = find_csv(SOURCE_DIR, config["file_pattern"])
        if csv_path is None:
            continue

        df = read_and_clean(csv_path)
        col = config["col"]
        feat_name = f"{block}_remediation"

        if col in df.columns:
            # Read the raw column BEFORE pandas converts N/A to NaN
            # Re-read with keep_default_na=False to preserve "N/A" as string
            df_raw = pd.read_csv(csv_path, keep_default_na=False)
            df_raw = df_raw[~df_raw["Student"].str.strip().str.startswith("Points Possible", na=False)]
            df_raw = df_raw.dropna(subset=["Student"])
            df_raw["student_id"] = df_raw["SIS Login ID"].astype(str).str.extract(r"(\d+)")[0]
            df_raw["student_id"] = pd.to_numeric(df_raw["student_id"], errors="coerce")
            df_raw = df_raw.dropna(subset=["student_id"])
            df_raw["student_id"] = df_raw["student_id"].astype(int)
            df_raw = df_raw[(df_raw["student_id"] >= 10000) & (df_raw["student_id"] <= 99999)]

            # N/A or empty = did not need remediation, a number = needed remediation
            raw_vals = df_raw[col].astype(str).str.strip()
            df_raw[feat_name] = (~raw_vals.isin(["N/A", "nan", "", "0", "0.0"])).astype(int)
            df = df_raw  # use the raw-parsed version for remediation

            merge_cols = df[["student_id", feat_name]]
            merged = merged.merge(merge_cols, on="student_id", how="left", suffixes=("_old", ""))
            if f"{feat_name}_old" in merged.columns:
                merged[feat_name] = merged[feat_name].fillna(merged[f"{feat_name}_old"])
                merged = merged.drop(columns=[f"{feat_name}_old"])

            n_remediated = merged[feat_name].sum() if feat_name in merged.columns else 0
            print(f"  {block}: {int(n_remediated)} students needed remediation")
        else:
            print(f"  WARNING: Column '{col}' not found for {block}")

    # Add total remediation count
    rem_cols = [c for c in merged.columns if c.endswith("_remediation")]
    if rem_cols:
        merged["total_remediations"] = merged[rem_cols].sum(axis=1)
        print(f"\n  Total remediation columns: {len(rem_cols)}")
        print(f"  Students with 0 remediations: {(merged['total_remediations'] == 0).sum()}")
        print(f"  Students with 1+ remediations: {(merged['total_remediations'] >= 1).sum()}")
        print(f"  Students with 2+ remediations: {(merged['total_remediations'] >= 2).sum()}")

    # Save enhanced dataset
    out_path = os.path.join(DATA_DIR, "merged_enhanced.csv")
    merged.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(f"Total columns: {len(merged.columns)}")

    # ============================================================
    # Quick check: do new features correlate with CBSE?
    # ============================================================
    print(f"\n{'='*60}")
    print("New Feature Correlations with CBSE First Attempt")
    print("=" * 60)

    from scipy import stats

    df_cbse = merged[merged['cbse_first_score'].notna()]

    new_features = list(FORMATIVE_CONFIG.keys()) + rem_cols + ["total_remediations"]
    for feat in new_features:
        if feat in df_cbse.columns:
            valid = df_cbse[[feat, 'cbse_first_score']].dropna()
            if len(valid) > 10:
                r, p = stats.pearsonr(valid[feat], valid['cbse_first_score'])
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                print(f"  {feat:<35} r={r:+.3f}  p={p:.4f} {sig}")
            else:
                print(f"  {feat:<35} (too few data points: {len(valid)})")


if __name__ == "__main__":
    main()
