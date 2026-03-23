"""
Data Loader（数据加载器）
========================
目的：加载 block_scores.csv 和 cbse_results.csv，按 student_id 合并

学习点 (Learning Points):
- Data Merging / Join: 两个表通过共同的 key 连接
- Data Validation: 检查数据质量（缺失值、格式错误等）
"""

import pandas as pd
from pathlib import Path


def load_block_scores(filepath: str = "data/raw/block_scores.csv") -> pd.DataFrame:
    """Load block exam scores CSV."""
    df = pd.read_csv(filepath)

    # Validation: student_id must exist
    if "student_id" not in df.columns:
        raise ValueError("block_scores.csv must have a 'student_id' column")

    print(f"Loaded block scores: {len(df)} students, {len(df.columns)} columns")
    return df


def load_cbse_results(filepath: str = "data/raw/cbse_results.csv") -> pd.DataFrame:
    """Load CBSE results CSV."""
    df = pd.read_csv(filepath)

    if "student_id" not in df.columns:
        raise ValueError("cbse_results.csv must have a 'student_id' column")
    if "cbse_score" not in df.columns:
        raise ValueError("cbse_results.csv must have a 'cbse_score' column")

    print(f"Loaded CBSE results: {len(df)} students")
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Auto-detect feature columns.
    All numeric columns except student_id and cbse_score are treated as features.
    """
    exclude = {"student_id", "name", "cbse_score"}
    return [col for col in df.select_dtypes(include="number").columns
            if col not in exclude]


def merge_data(
    blocks_df: pd.DataFrame,
    cbse_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge block scores with CBSE results on student_id (inner join).

    Inner join: only keep students that appear in BOTH tables.
    Students in block_scores but NOT in cbse_results (current cohort) are excluded.
    """
    merged = pd.merge(blocks_df, cbse_df, on="student_id", how="inner")

    n_blocks = len(blocks_df)
    n_cbse = len(cbse_df)
    n_merged = len(merged)
    n_lost = n_cbse - n_merged

    print(f"Merged: {n_merged} students (from {n_blocks} block records + {n_cbse} CBSE records)")
    if n_lost > 0:
        print(f"  Warning: {n_lost} students in CBSE results had no matching block scores")

    return merged


def load_and_merge(
    blocks_path: str = "data/raw/block_scores.csv",
    cbse_path: str = "data/raw/cbse_results.csv",
    save_path: str = "data/processed/merged_data.csv",
) -> pd.DataFrame:
    """
    Full pipeline: load both CSVs, merge, save processed data.
    """
    blocks_df = load_block_scores(blocks_path)
    cbse_df = load_cbse_results(cbse_path)
    merged = merge_data(blocks_df, cbse_df)

    # Save merged data
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"Saved merged data to: {output_path}")

    return merged


def load_current_cohort(
    blocks_path: str = "data/raw/block_scores.csv",
    cbse_path: str = "data/raw/cbse_results.csv",
) -> pd.DataFrame:
    """
    Load students who have block scores but NO CBSE results yet.
    These are the current cohort we want to predict for.
    """
    blocks_df = load_block_scores(blocks_path)
    cbse_df = load_cbse_results(cbse_path)

    # Students in blocks but not in CBSE = current cohort
    cbse_ids = set(cbse_df["student_id"])
    current = blocks_df[~blocks_df["student_id"].isin(cbse_ids)].copy()

    print(f"Current cohort (no CBSE scores yet): {len(current)} students")
    return current


if __name__ == "__main__":
    load_and_merge()
