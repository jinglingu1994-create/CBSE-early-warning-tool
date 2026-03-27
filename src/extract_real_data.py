"""
Extract real CBSE and block score data from source files.

This script reads:
1. Block score CSVs (from Canvas grade exports)
2. CBSE roster data (hardcoded from PDF extraction)

And outputs a single merged CSV for analysis.

IMPORTANT: Output CSVs contain real student data (FERPA protected).
They are saved to data/real/ which is in .gitignore.
NEVER push real data to GitHub.
"""

import pandas as pd
import os
import sys

# ============================================================
# Configuration: Block CSV files and their exam column mappings
# ============================================================

# Each block maps to: (csv_filename_pattern, {friendly_col_name: actual_col_name_in_csv})
BLOCK_CONFIG = {
    "foundation": {
        "file_pattern": "foundation",
        "exams": {
            "foundation_midterm1": "Mid-Course Summative Exam #1 Final Score",
            "foundation_midterm2": "Mid-Course Summative Exam #2 Final Score",
            "foundation_final": "End of Course Final Comprehensive NBME Summative Exam Final Score",
        }
    },
    "msk": {
        "file_pattern": "MSK",
        "exams": {
            "msk_midterm": "Mid-Course Summative Exam Final Score",
            "msk_final": "End of Course Comprehensive Summative Exam Final Score",
        }
    },
    "gi": {
        "file_pattern": "GI",
        "exams": {
            "gi_midterm": "Mid-Course Summative Exam Final Score",
            "gi_final": "Final Comprehensive NBME Summative Exam Final Score",
        }
    },
    "heme": {
        "file_pattern": "Heme",
        "exams": {
            "heme_midterm": "Mid-Course NBME Summative Exam Final Score",
            "heme_final": "End-of-Course Final NBME Summative Exam Final Score",
        }
    },
    "cv": {
        "file_pattern": "CV",
        "exams": {
            "cv_midterm": "Midterm Summative Exam Final Score",
            "cv_final": "Final Summative Exam Final Score",
        }
    },
    "pulm": {
        "file_pattern": "Pul",
        "exams": {
            "pulm_midterm": "Mid-Course NBME Summative Exam Final Score",
            "pulm_final": "Final Comprehensive NBME Summative Exam Final Score",
        }
    },
    "renal": {
        "file_pattern": "Renal",
        "exams": {
            "renal_midterm": "Mid-Course NBME Summative Exam Final Score",
            "renal_final": "End-of-Course Final Comprehensive NBME Summative Exam Final Score",
        }
    },
    "neuro": {
        "file_pattern": "Neuro",
        "exams": {
            "neuro_midterm": "Mid-Course Summative Exam Final Score",
            "neuro_final": "Final Comprehensive Summative Exam Final Score",
        }
    },
    "behavioral_science": {
        "file_pattern": "Behavior",
        "exams": {
            "behavioral_science_midterm": "Mid-Course NBME Summative Exam Final Score",
            "behavioral_science_final": "Final NBME Summative Exam Final Score",
        }
    },
    "endo": {
        "file_pattern": "Endo",
        "exams": {
            "endo_midterm": "Mid-Term Summative Examination 11/3/25 Final Score",
            "endo_final": "Final Summative Examination 11/24/25 Final Score",
        }
    },
    "repro": {
        "file_pattern": "Repro",
        "exams": {
            "repro_midterm": "Midcourse Exam Final Score",
            "repro_final": "Final Exam Final Score",
        }
    },
}

# ============================================================
# CBSE Roster Data (extracted from PDF by reading the documents)
# ============================================================

# Attempt 1: 02/27/2026, 74 students
CBSE_ATTEMPT_1 = {
    "date": "2026-02-27",
    "roster": {
        16279: 68, 16092: 55, 16266: 69, 16411: 53, 16222: 76, 16254: 79,
        16078: 65, 16079: 77, 16384: 54, 14677: 40, 16270: 66, 16439: 59,
        16297: 63, 16371: 66, 16397: 74, 16372: 69, 14210: 64, 16132: 72,
        16197: 71, 16490: 72, 15774: 72, 16406: 67, 16307: 64, 16370: 70,
        16358: 61, 16268: 83, 16340: 69, 16066: 59, 16215: 74, 16157: 76,
        16067: 74, 16231: 75, 16305: 50, 16138: 79, 16245: 75, 16224: 89,
        16213: 60, 16241: 56, 16349: 74, 16390: 75, 16200: 74, 16303: 71,
        16298: 56, 16399: 58, 16319: 66, 16074: 85, 16218: 73, 16246: 67,
        16369: 68, 15380: 74, 16455: 56, 16068: 73, 16442: 75, 14775: 78,
        16287: 67, 16221: 61, 16353: 64, 16143: 60, 16440: 67, 16376: 68,
        16361: 68, 16417: 61, 16146: 63, 16056: 88, 16459: 53, 16300: 79,
        16267: 83, 16347: 55, 16394: 85, 16308: 62, 16070: 74, 16128: 81,
        16346: 63, 14100: 54,
    }
}

# Attempt 2: 03/06/2026, 60 students
CBSE_ATTEMPT_2 = {
    "date": "2026-03-06",
    "roster": {
        16274: 48, 16077: 73, 16079: 71, 16345: 81, 16384: 52, 16237: 53,
        14677: 50, 16297: 71, 16397: 83, 16372: 64, 16359: 56, 16391: 57,
        15625: 56, 15952: 52, 16383: 86, 16217: 82, 16061: 80, 16341: 77,
        16480: 85, 16429: 47, 16307: 68, 16291: 51, 16230: 53, 16066: 66,
        16443: 50, 16215: 74, 16233: 54, 16157: 76, 16067: 73, 16305: 55,
        16213: 63, 16062: 74, 16342: 57, 16200: 73, 16303: 60, 16310: 69,
        16175: 71, 16350: 71, 13009: 57, 16369: 70, 16455: 52, 16240: 64,
        16287: 65, 16299: 62, 16381: 68, 14156: 50, 16090: 66, 16271: 77,
        16361: 68, 16417: 58, 16225: 67, 16459: 45, 16248: 56, 16393: 77,
        16347: 60, 16441: 59, 16479: 57, 16346: 65, 16235: 78, 14100: 61,
    }
}

# Attempt 3: 03/13/2026, 49 students
CBSE_ATTEMPT_3 = {
    "date": "2026-03-13",
    "roster": {
        16274: 51, 16092: 64, 16411: 64, 16077: 76, 16078: 75, 16384: 55,
        16237: 53, 14677: 55, 16439: 63, 16392: 55, 16359: 64, 16391: 60,
        15625: 61, 16302: 54, 15952: 65, 16217: 84, 16219: 65, 16429: 52,
        16358: 68, 16291: 60, 16230: 61, 16443: 51, 16233: 63, 16305: 57,
        16213: 68, 16241: 67, 16342: 60, 16298: 65, 16399: 67, 16319: 67,
        16350: 71, 13009: 63, 16455: 60, 14775: 81, 16221: 48, 16299: 70,
        16143: 58, 14156: 51, 16417: 67, 16146: 69, 16459: 50, 16248: 73,
        16347: 66, 16441: 54, 16479: 59, 16394: 80, 16308: 69, 16125: 73,
        14100: 57,
    }
}


def find_csv_file(source_dir, pattern):
    """Find a CSV file in source_dir that contains the given pattern."""
    for f in os.listdir(source_dir):
        if f.endswith(".csv") and pattern in f:
            return os.path.join(source_dir, f)
    return None


def extract_block_scores(source_dir):
    """
    Extract summative exam Final Scores from all block CSVs.
    Returns a DataFrame with student_id as index and exam scores as columns.
    """
    all_blocks = {}

    for block_name, config in BLOCK_CONFIG.items():
        csv_path = find_csv_file(source_dir, config["file_pattern"])
        if csv_path is None:
            print(f"  WARNING: No CSV found for {block_name} (pattern: {config['file_pattern']})")
            continue

        # Read CSV, skip the "Points Possible" row (row index 1 after header)
        df = pd.read_csv(csv_path)
        # Remove the "Points Possible" row (first data row where Student starts with spaces)
        df = df[~df["Student"].str.strip().str.startswith("Points Possible", na=False)]
        # Also remove rows where Student is NaN or empty
        df = df.dropna(subset=["Student"])

        # Extract student_id (some IDs have notes like "16301 - LOA")
        df["student_id"] = df["SIS Login ID"].astype(str).str.extract(r"(\d+)")[0]
        df["student_id"] = pd.to_numeric(df["student_id"], errors="coerce")
        df = df.dropna(subset=["student_id"])
        df["student_id"] = df["student_id"].astype(int)

        # Filter out invalid IDs (test accounts, junk rows)
        # Real student IDs are 5 digits: 13000-16999
        df = df[(df["student_id"] >= 10000) & (df["student_id"] <= 99999)]

        # Filter out test accounts
        df = df[~df["Student"].str.contains("Test", case=False, na=False)]

        # Extract student name (clean up quotes)
        df["name"] = df["Student"].str.strip().str.strip('"')

        # Extract exam scores
        for friendly_name, actual_col in config["exams"].items():
            if actual_col in df.columns:
                df[friendly_name] = pd.to_numeric(df[actual_col], errors="coerce")
            else:
                print(f"  WARNING: Column '{actual_col}' not found in {block_name}")
                df[friendly_name] = None

        # Keep only relevant columns
        exam_cols = list(config["exams"].keys())
        block_df = df[["student_id", "name"] + exam_cols].copy()
        all_blocks[block_name] = block_df
        print(f"  {block_name}: {len(block_df)} students, {len(exam_cols)} exams")

    # Merge all blocks on student_id
    if not all_blocks:
        raise ValueError("No block data found!")

    merged = None
    for block_name, block_df in all_blocks.items():
        if merged is None:
            merged = block_df
        else:
            # Merge, keeping name from first block only
            exam_cols = [c for c in block_df.columns if c not in ("student_id", "name")]
            merged = merged.merge(
                block_df[["student_id"] + exam_cols],
                on="student_id",
                how="outer"
            )

    return merged


def extract_cbse_scores():
    """
    Build CBSE results DataFrame from the hardcoded roster data.
    Tracks multiple attempts per student.
    """
    # Collect all unique student IDs across all attempts
    all_ids = set()
    for attempt in [CBSE_ATTEMPT_1, CBSE_ATTEMPT_2, CBSE_ATTEMPT_3]:
        all_ids.update(attempt["roster"].keys())

    rows = []
    for sid in sorted(all_ids):
        row = {"student_id": sid}

        # Check each attempt
        scores = []
        for i, attempt in enumerate([CBSE_ATTEMPT_1, CBSE_ATTEMPT_2, CBSE_ATTEMPT_3], 1):
            if sid in attempt["roster"]:
                row[f"cbse_attempt{i}_date"] = attempt["date"]
                row[f"cbse_attempt{i}_score"] = attempt["roster"][sid]
                scores.append(attempt["roster"][sid])
            else:
                row[f"cbse_attempt{i}_date"] = None
                row[f"cbse_attempt{i}_score"] = None

        row["cbse_total_attempts"] = len(scores)
        row["cbse_best_score"] = max(scores) if scores else None
        row["cbse_latest_score"] = scores[-1] if scores else None
        row["cbse_first_score"] = scores[0] if scores else None

        # Score change from first to latest attempt
        if len(scores) >= 2:
            row["cbse_score_change"] = scores[-1] - scores[0]
        else:
            row["cbse_score_change"] = None

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    # Paths
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source_dir = r"C:\Users\lexie\Downloads\(No subject)"
    output_dir = os.path.join(project_dir, "data", "real")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  Extracting Real Student Data (LOCAL ONLY)")
    print("=" * 60)

    # Step 1: Extract block scores
    print("\n[Step 1] Extracting block scores from CSVs...")
    block_df = extract_block_scores(source_dir)
    print(f"\n  Total students in block data: {len(block_df)}")

    # Step 2: Extract CBSE scores
    print("\n[Step 2] Extracting CBSE scores from roster data...")
    cbse_df = extract_cbse_scores()
    print(f"  Total students in CBSE data: {len(cbse_df)}")
    print(f"  1 attempt: {(cbse_df['cbse_total_attempts'] == 1).sum()}")
    print(f"  2 attempts: {(cbse_df['cbse_total_attempts'] == 2).sum()}")
    print(f"  3 attempts: {(cbse_df['cbse_total_attempts'] == 3).sum()}")

    # Step 3: Merge block scores with CBSE scores
    print("\n[Step 3] Merging block scores with CBSE scores...")
    merged = block_df.merge(cbse_df, on="student_id", how="outer")
    print(f"  Total unique students: {len(merged)}")

    # Identify students in blocks but NOT in CBSE (potentially dismissed)
    block_only = merged[merged["cbse_total_attempts"].isna() | (merged["cbse_total_attempts"] == 0)]
    cbse_only = merged[merged["name"].isna()]
    both = merged[merged["name"].notna() & merged["cbse_total_attempts"].notna() & (merged["cbse_total_attempts"] > 0)]

    print(f"\n  Students with BOTH block scores and CBSE: {len(both)}")
    print(f"  Students with block scores but NO CBSE (possibly dismissed/deferred): {len(block_only)}")
    print(f"  Students with CBSE but NO block scores (different cohort?): {len(cbse_only)}")

    # Step 4: Save outputs
    print("\n[Step 4] Saving to data/real/ (LOCAL ONLY - never push to GitHub!)...")

    # Save full merged dataset
    merged_path = os.path.join(output_dir, "merged_all_data.csv")
    merged.to_csv(merged_path, index=False)
    print(f"  Saved: {merged_path}")

    # Save block scores only
    block_path = os.path.join(output_dir, "block_scores_real.csv")
    block_df.to_csv(block_path, index=False)
    print(f"  Saved: {block_path}")

    # Save CBSE scores only
    cbse_path = os.path.join(output_dir, "cbse_results_real.csv")
    cbse_df.to_csv(cbse_path, index=False)
    print(f"  Saved: {cbse_path}")

    # Save students with block scores but no CBSE
    if len(block_only) > 0:
        dismissed_path = os.path.join(output_dir, "students_no_cbse.csv")
        block_only[["student_id", "name"]].to_csv(dismissed_path, index=False)
        print(f"  Saved: {dismissed_path}")

    print("\n" + "=" * 60)
    print("  COMPLETE - All files saved to data/real/")
    print("  WARNING: These files contain real student data.")
    print("  NEVER push data/real/ to GitHub!")
    print("=" * 60)


if __name__ == "__main__":
    main()
