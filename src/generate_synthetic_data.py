"""
模拟数据生成器 (Synthetic Data Generator)
==========================================
目的：生成逼真的医学院学生 block 考试成绩和 NBME CBSE 分数
用于在真实数据到手前跑通整个 ML pipeline

学习点 (Learning Points):
- Normal Distribution: 大部分学生成绩集中在中间，极端值少
- Weighted Sum: CBSE score 是各 block 成绩的加权组合
- Random Noise: 模拟现实中不可预测的因素（Irreducible Error）
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# 配置：Block 名称和模拟参数
# ============================================================

# 你的 10 个 organ system blocks
BLOCK_NAMES = [
    "foundation",
    "heme",
    "msk",
    "neuro",
    "renal",
    "cv",
    "pulm",
    "behavioral_science",
    "endo",
    "repro",
]

# 每个 block 的成绩分布参数 (mean, std)
# 不同 block 难度不同，所以均值和标准差不一样
BLOCK_PARAMS = {
    "foundation":         (72, 12),  # Foundation 相对基础，均分偏高
    "heme":               (68, 13),
    "msk":                (65, 14),
    "neuro":              (62, 15),  # Neuro 公认较难，均分偏低
    "renal":              (64, 14),
    "cv":                 (66, 13),
    "pulm":               (67, 13),
    "behavioral_science": (74, 11),  # Behavioral Science 相对好拿分
    "endo":               (63, 14),
    "repro":              (69, 12),
}

# 每个 block 对 CBSE score 的真实权重 (weights)
# 这是我们人为设定的 "ground truth"，模型需要学习发现这些权重
# 设定 neuro 和 cv 权重最高——因为它们在 CBSE 中占比大
CBSE_WEIGHTS = {
    "foundation":         0.12,
    "heme":               0.08,
    "msk":                0.07,
    "neuro":              0.15,  # Neuro 对 CBSE 影响最大
    "renal":              0.10,
    "cv":                 0.14,  # CV 对 CBSE 影响第二大
    "pulm":               0.09,
    "behavioral_science": 0.08,
    "endo":               0.09,
    "repro":              0.08,
}


def generate_block_scores(n_students: int, seed: int = 42) -> pd.DataFrame:
    """
    生成学生的 block 考试成绩

    原理：
    - 每个学生有一个 "基础能力值"(base_ability)，用 Normal Distribution 生成
    - 每个 block 成绩 = base_ability 的影响 + 该 block 特有的随机波动
    - 这样自然产生了 block 之间的 correlation（因为共享 base_ability）
      但又不是完全相同（因为每个 block 有自己的 noise）
    """
    rng = np.random.default_rng(seed)

    # 每个学生的基础能力值：决定了整体水平
    # 80% 普通学生，15% 偏弱学生，5% 优秀学生
    student_type = rng.choice(
        ["normal", "weak", "strong"],
        size=n_students,
        p=[0.80, 0.15, 0.05],
    )

    base_ability = np.zeros(n_students)
    for i, stype in enumerate(student_type):
        if stype == "normal":
            base_ability[i] = rng.normal(68, 8)
        elif stype == "weak":
            base_ability[i] = rng.normal(50, 8)
        else:  # strong
            base_ability[i] = rng.normal(85, 5)

    # 生成每个 block 的成绩
    scores = {}
    for block in BLOCK_NAMES:
        mean, std = BLOCK_PARAMS[block]
        # block 成绩 = 50% 来自基础能力 + 50% 来自该 block 的独立表现
        block_specific = rng.normal(mean, std, size=n_students)
        raw_score = 0.5 * base_ability + 0.5 * block_specific
        # 限制在 0-100 范围内
        scores[block] = np.clip(raw_score, 0, 100).round(1)

    # 组装 DataFrame
    df = pd.DataFrame(scores)
    df.insert(0, "student_id", [f"STU{i+1:04d}" for i in range(n_students)])
    df.insert(1, "name", [f"Student_{i+1}" for i in range(n_students)])

    return df


def generate_cbse_scores(block_scores_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    根据 block 成绩生成 CBSE 分数

    原理：
    - CBSE score = Σ (block_score × weight) × scale_factor + noise
    - scale_factor 把 0-100 的加权平均映射到 CBSE 的 180-280 范围
    - noise 就是 Irreducible Error：考试当天状态、题目运气等

    为什么加 noise？
    如果不加 noise，模型可以达到 R² = 1.0（完美预测），这不现实。
    加了 noise 后，模型的 R² 会有一个 ceiling（比如 0.75-0.85），
    这更接近真实世界的情况。
    """
    rng = np.random.default_rng(seed)

    # 计算加权分数
    weighted_sum = np.zeros(len(block_scores_df))
    for block, weight in CBSE_WEIGHTS.items():
        weighted_sum += block_scores_df[block].values * weight

    # weighted_sum 现在大约在 50-85 范围（因为权重加起来 = 1.0）
    # 映射到 CBSE 分数范围：大约 180-280
    scale_factor = 1.8
    base_score = 80  # 基础分
    cbse_raw = weighted_sum * scale_factor + base_score

    # 加入 Random Noise（Irreducible Error）
    # std=12 意味着大约 68% 的学生实际分数在预测值 ±12 分以内
    noise = rng.normal(0, 12, size=len(block_scores_df))
    cbse_score = (cbse_raw + noise).round(0).astype(int)

    # 限制在合理范围
    cbse_score = np.clip(cbse_score, 150, 300)

    df = pd.DataFrame({
        "student_id": block_scores_df["student_id"],
        "cbse_score": cbse_score,
    })

    return df


def generate_and_save(output_dir: str = "data/raw", seed: int = 42):
    """
    生成完整数据集并保存到 CSV

    生成两组：
    1. 历史数据：500 名学生，有 block scores + CBSE score（用于训练模型）
    2. 当前学生：100 名学生，只有 block scores（用于预测，模拟真实场景）
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- Historical data (for training) ---
    print("Generating historical student data (500 students)...")
    historical_blocks = generate_block_scores(500, seed=seed)
    historical_cbse = generate_cbse_scores(historical_blocks, seed=seed)

    # --- Current cohort (for prediction) ---
    print("Generating current cohort data (100 students, no CBSE scores)...")
    current_blocks = generate_block_scores(100, seed=seed + 100)
    # 当前学生的 student_id 从 STU0501 开始，避免和历史数据重复
    current_blocks["student_id"] = [f"STU{i+501:04d}" for i in range(100)]
    current_blocks["name"] = [f"Student_{i+501}" for i in range(100)]

    # 合并所有 block scores（历史 + 当前）
    all_blocks = pd.concat([historical_blocks, current_blocks], ignore_index=True)

    # 保存
    all_blocks.to_csv(output_path / "block_scores.csv", index=False)
    historical_cbse.to_csv(output_path / "cbse_results.csv", index=False)

    print(f"\nDone!")
    print(f"  block_scores.csv : {len(all_blocks)} students (500 historical + 100 current)")
    print(f"  cbse_results.csv : {len(historical_cbse)} students (historical only)")
    print(f"  Saved to: {output_path.resolve()}")

    return all_blocks, historical_cbse


if __name__ == "__main__":
    generate_and_save()
