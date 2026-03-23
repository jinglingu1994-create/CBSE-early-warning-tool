# ML/DS Learning Notes — NBME CBSE Early Alarm Project

> Learning by doing, documenting knowledge at every step. Annotated with Healthcare AI PM importance ratings (⭐~⭐⭐⭐).

---

## Step 1: Environment Setup + Synthetic Data Generation

### 1.1 Synthetic Data ⭐⭐
**What it is**: When real data is unavailable, you use code to generate fake data according to predefined rules.
**Why it matters**: It lets you run through the entire ML pipeline without waiting for data, verifying that your code logic is correct.
**PM Perspective**: PMs don't need to write data generation code, but should know that teams commonly use synthetic data during the prototype phase for rapid idea validation (proof of concept).

### 1.2 Normal Distribution ⭐⭐
**What it is**: A data distribution pattern where most values cluster around the center, with extreme values being rare. The shape resembles a bell (bell curve).
**In our project**: Student scores typically follow a normal distribution — most students score between 60-80, with very few scoring 0 or 100.
**PM Perspective**: PMs need to be able to read data distribution charts and judge whether data looks "normal." If the distribution is abnormal (e.g., bimodal), it may indicate data problems or the existence of subgroups.

### 1.3 Weighted Sum ⭐⭐⭐
**What it is**: Different features contribute differently to the outcome. CBSE score = sum of (block_score x weight).
**In our project**: We set Neuro weight at 0.15 (highest) and MSK weight at 0.07 (lowest), meaning Neuro's influence on CBSE is more than double that of MSK.
**PM Perspective**: This is the core logic of all ML products. PMs must understand that "different inputs contribute differently to the output" — this directly affects product design (e.g., which metrics should be prioritized for user display).

### 1.4 Random Noise / Irreducible Error ⭐⭐⭐
**What it is**: Random fluctuations in data that cannot be explained by features. These come from unmeasured factors (test-day condition, question luck, etc.).

**Complete Knowledge (PM Must-Know)**:

**Total Error = Bias + Variance + Irreducible Error**
- Bias: Model too simple — solve by switching to a better model
- Variance: Model too complex (overfitting) — solve by adding data or simplifying the model
- Irreducible Error: Information not present in the data — **cannot be solved by optimizing the model**

**Key Insight**: Irreducible error is fixed *given* a feature set, but not fixed *across* feature sets. Changing the feature set (adding new data sources) can turn previously irreducible portions into reducible ones.

**How to Estimate Whether Irreducible Error is High or Low**:
1. **Human Baseline**: Have domain experts perform the same prediction task — human accuracy serves as a reference ceiling
2. **Learning Curve**: If test error stops decreasing after adding more data, most of what remains is irreducible
3. **Multiple Models**: If all models plateau at the same level, the bottleneck is the data, not the model
4. **Repeated Measurements**: The score difference between two exams taken by the same student directly reflects noise magnitude

**Questions PMs Should Ask the ML Team**:
- "How much of the model error is irreducible?"
- "What is the human baseline accuracy? How far are we from it?"
- "If we add data source X, how much error reduction can we expect?"
- "Are we already approaching the performance ceiling?"

**PM Decision Table for Real-World Scenarios**:
| Situation | Action |
|---|---|
| Large gap between model and human baseline | Push the ML team to keep optimizing |
| Model approaching human baseline | Focus on product experience, stop chasing accuracy |
| All models perform the same | Need a new data source, not a new model |
| Error stops decreasing after adding data | Stop collecting similar data, save budget |

### 1.5 Feature Set ⭐⭐⭐
**What it is**: The collection of all input variables fed to the model. In our project, the feature set = 10 block scores.
**PM Perspective**: One of the PM's greatest contributions is leveraging domain knowledge to optimize the feature set — deciding which features to add and which to remove. The ML team can only find statistical correlations from data, but PMs (especially those with clinical backgrounds) can judge whether a correlation has a causal relationship behind it.

**Example**: A Heart Failure model includes urine Na (no causal relationship — remove it) but lacks CAD history (has a pathophysiological link — add it). Only a domain expert can make this kind of judgment.

### 1.6 Feature Selection vs Feature Engineering ⭐⭐⭐
- **Feature Selection**: Choosing which existing features to include in the model and which to exclude
- **Feature Engineering**: Creating new features (e.g., computing "average" and "variance" from block scores)
- PMs don't need to write code for these, but need to guide the direction: telling the team "this variable has no clinical significance, don't include it" or "you should consider adding X data"

---

## Step 2: Data Loading + EDA

### 2.1 Data Merging / Join ⭐⭐
**What it is**: Connecting two tables into one through a shared key (such as student_id).
**PM Perspective**: Data integration is one of the most time-consuming parts of real-world ML projects (often taking 60-80% of project time). PMs should allocate sufficient time in the roadmap for data cleaning and integration — don't assume "once we get the data, we can use it immediately."

### 2.2 Correlation Heatmap ⭐⭐⭐
**What it is**: A visualization that uses color intensity to show the degree of correlation between any two variables (r value, ranging from -1 to +1).

**Quick Reference for r Values**:
| r Value | Meaning |
|---|---|
| > 0.7 | Strong correlation |
| 0.4-0.7 | Moderate |
| < 0.4 | Weak |
| Negative | Negative correlation (when A is high, B is low) |

**Limitations**:
- Can only detect linear relationships
- Non-linear relationships (e.g., U-shaped) will be missed (r near 0 but actually strongly related)
- Correlation does not equal Causation

**Clinical Non-linear Examples**:
- Body temperature and Sepsis: Both high fever and hypothermia increase risk (U-shaped)
- Serum potassium and arrhythmia: Both hypokalemia and hyperkalemia are dangerous (U-shaped)
- If a DS only looks at linear correlation, they would wrongly conclude these features are "useless" and remove them

**Questions PMs Should Ask in Review Meetings**:
- "Does this high correlation have a clinical basis?"
- "Could there be a confounding variable creating a spurious correlation?"
- "Beyond linear correlation, has anyone checked for non-linear relationships?"
- "Does this relationship hold across different subgroups?"

**How to Report to Stakeholders**:
- Don't say: "The correlation between Neuro and CBSE is 0.72"
- Do say: "Neuroscience has the strongest predictive power for CBSE scores — students with high Neuro scores have significantly higher CBSE pass rates. If we want to implement early intervention, we should prioritize monitoring Neuro performance."

### 2.3 Redundancy ⭐⭐⭐
**What it is**: Two features are highly correlated (r > 0.8), carrying nearly identical information.

**The criterion for whether to remove is not how high the r value is, but whether the two features provide different information**:
| Situation | Example | Action |
|---|---|---|
| High correlation + different causal pathways | DM and MI both cause HF, but through different mechanisms | Keep both |
| High correlation + different measurements of the same thing | Weight in kg and weight in lb | Remove one |
| High correlation + one is downstream of the other | HbA1c and fasting glucose | Depends on context |

**PMs should know**: If the team says "we added 50 features," ask "did you run a redundancy check?"

### 2.4 Confounding + Mediation ⭐⭐⭐
**What it is**: The DM to MI to HF example. DM affects the outcome through both a direct pathway (diabetic cardiomyopathy) and an indirect pathway (first causing MI, then MI causing HF).

**The most important judgment for PMs**: Is the product for prediction or intervention?
- **Prediction product**: Include both correlated features — the more accurate, the better
- **Intervention product**: Must clarify the causal chain, find the upstream cause (controlling DM is more cost-effective than treating MI)

### 2.5 PM and Data Team Division of Labor ⭐⭐⭐
| Task | Who Does It | What the PM Does |
|---|---|---|
| Write code, create charts | Data Analyst / DS | Not your job |
| Review heatmap to identify important features | DA produces the chart | You interpret it and ask the right questions |
| Decide which features to add/remove | **You** | Based on domain knowledge |
| Judge whether correlation implies causation | **You** | Using clinical knowledge |
| Data ID mismatches | Data Engineer fixes it | You give the team enough time |

**Data Analysts produce data and charts; you produce decisions.**

### 2.6 EDA Results (Our Project)
- All 10 blocks have moderate correlation with CBSE (r around 0.43-0.53)
- CV (r=0.53) and Renal (r=0.52) have the highest correlation
- 35.2% of students scored below 194 on CBSE (passing threshold)
- No missing values (synthetic data — real data almost never has zero missing values)

---

## Step 3: Preprocessing & Feature Engineering

### 3.1 Missing Value Handling ⭐⭐⭐
**The core question is not "what method to use for imputation," but "is the missingness random?"**

**Three Strategies and Their Use Cases**:
| Strategy | Use Case | Risk |
|---|---|---|
| Drop (delete) | Missingness <5%, large dataset, random missing | If missing is not random, introduces Selection Bias |
| Impute (fill, e.g., median) | Small dataset, every record is precious | Pretends missing patients are "normal," underestimates risk |
| Missing Indicator (missingness flag) | Preferred for healthcare scenarios | Safest approach — lets the model decide |

**Healthcare Specificity — Missingness is Informative**:
- Patients who miss follow-up appointments may have worsening conditions or be non-compliant
- Missing ICU lab results may mean the patient was too unstable for testing
- Missing medication data may indicate the patient is not taking medication on schedule

**Example (HF Medication Data Missing)**:
- Dropping these patients: Model trains only on compliant patients, becomes overly optimistic for non-compliant patients
- Median imputation: Assumes they are taking medication, equally overly optimistic
- **Correct approach**: Create `med_data_missing = 1/0` as a new feature, so the model learns that "data absence is itself a risk factor"

**Three Must-Ask Questions for PM Review**:
1. "What is the missingness rate? Is it random?"
2. "Has the imputation strategy introduced any bias?"
3. "Has missingness itself been used as a feature?"

### 3.2 Feature Engineering ⭐⭐⭐
**What it is**: Creating new predictive signals from existing data.

**In our project**:
- `block_average`: Mean of all block scores — reflects overall ability level
- `block_variance`: Variance across block scores — high = significant weakness in certain areas
- `lowest_block`: Lowest block score — the "weakest link" effect

**PM Perspective**: PMs don't write feature engineering code, but should guide the direction. For example, telling the team "the trend in student scores (improving or declining) might be more predictive than absolute scores" — this kind of insight comes from domain knowledge.

### 3.3 Standardization / Scaling ⭐
**What it is**: Rescaling all features to the same scale (mean=0, std=1).
**Why it matters**: Linear Regression is sensitive to scale — if one feature ranges from 0-100 and another from 0-1, the former will be overweighted.
**PM Perspective**: Just know it exists. If the DS mentions scaling, knowing what they are talking about is sufficient.

### 3.4 Selection Bias ⭐⭐⭐
**What it is**: When improper data collection methods cause the training data to not represent the real population.
**Clinical example**: Training a model only on compliant patients means the model fails for non-compliant patients.
**PM Perspective**: This is a key review item during FDA approval for healthcare AI products. PMs must ensure the training data population matches the product's target users.

---

## Step 4: Model Training & Evaluation

### 4.1 Train/Test Split ⭐⭐⭐
**What it is**: Randomly splitting data into two portions — 80% for training (model learning) and 20% for testing (simulating new data to check real performance).
**Why it matters**: Without splitting, the model tests itself on data it already learned from = using the exact same questions on an exam = inflated scores.
**PM Perspective**: If the team gives you a single performance number, your first response should be: "Is this on training or test data? Show me both."

### 4.2 Overfitting ⭐⭐⭐
**What it is**: The model "memorizes" noise in the training data, performing poorly on new data.

**How to detect — just look at one number: the Train vs Test gap**
| Situation | Judgment |
|---|---|
| Train 0.52, Test 0.37 (gap 0.15) | OK |
| Train 0.99, Test 0.22 (gap 0.77) | Overfitting — tell the team to fix it |

**Why complex models are more prone to overfitting**:
- Linear Regression: 11 parameters, forced to learn general patterns
- XGBoost with 100 trees at depth=5: 3200+ independent regions, capable of assigning each student a unique prediction — memorizing noise

**PMs don't need to manage the fix** (adjusting depth, reducing trees, adding regularization is the DS's job) — you just need to see a large gap and flag it.

### 4.3 Data Leakage ⭐⭐⭐
**What it is**: The model "peeks" at information during training that it shouldn't have access to.

**Three Common Types**:
1. **Target Leakage**: A feature contains information that is only available after the outcome is known
   - Example: Predicting ICU admission with "mechanical ventilation hours" as a feature (only available in the ICU)
2. **Train-Test Leakage**: Test data information leaks into the training process
3. **Temporal Leakage**: Using future data to predict the past

**How PMs Can Check**: Look at the feature list and ask yourself "At the time we need to make a prediction, does this information actually exist already?" Spending 15 minutes scanning the feature list may be the highest-ROI 15 minutes of the entire project.

**DS may not catch this** — they don't know that mechanical ventilation is only available in the ICU, or that discharge codes are entered only at discharge. Your clinical knowledge serves as a safety guardrail.

### 4.4 Model Complexity vs Data Size ⭐⭐⭐
| Data Size | Suitable Model | PM Action |
|---|---|---|
| < 500 | Simple models (Linear Regression) | Push the team to collect more data |
| 500-5000 | Can compare multiple models, strictly control overfitting | Balance optimization vs data acquisition |
| > 10000 | Complex models can be used freely | Let the DS team experiment freely |

**PM Core Judgment**: With small data, rather than spending 3 months tuning models, it's better to spend 3 months acquiring more data — this is a resource allocation decision.

### 4.5 Cross-Validation ⭐⭐
**What it is**: Splitting data into K folds, taking turns as the test set, and averaging the results. More reliable than a single 80/20 split.
**PM Perspective**: Knowing that "CV R-squared = 0.45" is more trustworthy than a single test R-squared is sufficient. Whether K=5 or K=10 is the DS's decision.

### 4.6 Results (Our Project)
- Linear Regression won (Test R-squared = 0.37), because of small data + linear relationships
- XGBoost severely overfit (Train 0.99 vs Test 0.22)
- CV R-squared values were all between 0.40-0.46, indicating true performance is at this level
- R-squared = 0.37-0.46 means block scores can only explain about 40% of CBSE variation; improvement requires more features

---

## Step 5: Feature Importance

### 5.1 Linear Regression Coefficients ⭐⭐⭐
**What it is**: Each coefficient represents "how much the CBSE score changes when that feature increases by 1 standard deviation."
**PM Perspective**: The easiest to explain to stakeholders. "For every 1-point increase in the Renal block, CBSE is expected to increase by X points."

### 5.2 SHAP Values ⭐⭐⭐
**What it is**: Breaks down each student's prediction into the positive/negative contribution of each feature.

**Why healthcare AI must have this**:
- **Clinical trust**: Clinicians won't accept a black box. "Why was this student flagged as high risk?" SHAP answers: "Primarily because Neuro was too low (-8 points)"
- **Actionable Insight**: Instead of just saying "will fail," it says "why they will fail" — enabling targeted intervention
- **FDA Regulatory**: An increasing number of healthcare AI products are required to provide model explainability

**SHAP Limitation**: It explains the model's logic, not real-world causation. A model may make a correct prediction for the wrong reason.

### 5.3 Permutation Importance ⭐⭐
**What it is**: Randomly shuffle one feature's values and see how much model performance drops. A large drop = that feature is important.
**PM Perspective**: More robust than coefficients because it doesn't depend on the model type (model-agnostic).

### 5.4 Results (Our Project)
- `lowest_block` had the greatest impact (weakest link effect)
- Renal and CV were the strongest predictors among the original blocks
- MSK and Behavioral Science had the least impact

---

## Step 6: Early Warning System

### 6.1 Model Deployment ⭐⭐⭐
**What it is**: Using a trained model to make predictions on new data (students never seen before).
**PM Perspective**: Training performance does not represent post-deployment performance. PMs need to ensure monitoring is in place (continuously tracking model performance after launch).

### 6.2 Threshold Setting ⭐⭐⭐
**What it is**: What score counts as "at risk"? We use 194 (the former USMLE Step 1 passing score).
**PM Perspective**: The threshold is not a technical decision — it's a product/business decision. Lowering the threshold = flagging fewer students = missing more truly at-risk students. Raising the threshold = flagging more = more false alarms. PMs should decide based on the clinical cost of false negatives vs false positives.

### 6.3 Data Privacy ⭐⭐⭐
**FERPA**: Student grades in the US are legally protected PII. Real student data must never appear in a portfolio.
**Correct approach**: Use synthetic data to demonstrate methodology, noting "designed to work with real data when deployed."

---

## Step 7: End-to-End Pipeline

### 7.1 Modular Code ⭐⭐
**What it is**: Each function lives in its own file (data_loader, preprocessing, models, etc.), with main.py connecting them all.
**PM Perspective**: PMs don't need to write modular code, but should know that good code structure = higher team collaboration efficiency + lower maintenance costs. If the team's code is all in one notebook, that's a red flag.

