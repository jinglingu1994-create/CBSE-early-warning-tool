# ML/DS Learning Notes — NBME CBSE Early Alarm Project

> 边做边学，每一步记录知识点。标注 Healthcare AI PM 重要性（⭐~⭐⭐⭐）。

---

## Step 1: Environment Setup + Synthetic Data Generation

### 1.1 Synthetic Data（模拟数据）⭐⭐
**是什么**：在没有真实数据时，用代码按照预设规则生成假数据。
**为什么**：让你不用等数据就能跑通整个 ML pipeline，验证代码逻辑是否正确。
**PM 视角**：PM 不需要会写生成代码，但要知道团队在 prototype 阶段常用模拟数据来快速验证想法（proof of concept）。

### 1.2 Normal Distribution（正态分布）⭐⭐
**是什么**：一种数据分布模式——大部分值集中在中间，极端值少。形状像钟（bell curve）。
**在项目中**：学生成绩通常符合正态分布，大部分人 60-80 分，很少有 0 分或 100 分。
**PM 视角**：PM 需要能看懂数据分布图，判断数据是否"正常"。如果分布异常（比如双峰），可能意味着数据有问题或存在 subgroup。

### 1.3 Weighted Sum（加权求和）⭐⭐⭐
**是什么**：不同 feature 对结果的贡献不同。CBSE score = Σ(block_score × weight)。
**在项目中**：我们设定 Neuro 权重 0.15（最大），MSK 权重 0.07（最小），意味着 Neuro 对 CBSE 的影响是 MSK 的两倍多。
**PM 视角**：这是所有 ML 产品的核心逻辑。PM 必须理解"不同 input 对 output 的贡献不同"，这直接影响产品设计（比如应该优先展示哪些指标给用户）。

### 1.4 Random Noise / Irreducible Error（随机噪声/不可约误差）⭐⭐⭐
**是什么**：数据中无法被 features 解释的随机波动。来源是你没有测量的因素（考试状态、题目运气等）。

**完整知识（PM 必须掌握）**：

**Total Error = Bias + Variance + Irreducible Error**
- Bias: 模型太简单 → 换更好的模型解决
- Variance: 模型太复杂（overfitting）→ 加数据/简化模型解决
- Irreducible Error: 数据里没有的信息 → **无法通过优化模型解决**

**关键理解**：Irreducible error is fixed *given* a feature set, but not fixed *across* feature sets. 换 feature set（加入新数据源）可以把原本 irreducible 的部分变成 reducible。

**怎么估算 Irreducible Error 高低**：
1. **Human Baseline**: 让 domain expert 做同样的预测任务，人类准确率就是 ceiling 的参考
2. **Learning Curve**: 加数据后 test error 不再下降 → 剩下的大部分是 irreducible
3. **Multiple Models**: 所有模型都卡在同一水平 → 瓶颈是数据，不是模型
4. **Repeated Measurements**: 同一学生两次考试的分差直接反映 noise 大小

**PM 需要会问 ML 团队的问题**：
- "Model error 里有多少是 irreducible 的？"
- "Human baseline accuracy 是多少？我们离它还有多远？"
- "加入 X 数据源，预计能降多少 error？"
- "我们是否已经接近 performance ceiling？"

**PM 实战决策表**：
| 情况 | 行动 |
|---|---|
| Model 和 human baseline 差距大 | Push ML 团队继续优化 |
| Model 接近 human baseline | Focus on 产品体验，不追 accuracy |
| 所有模型 performance 都一样 | 需要新 data source，不是新 model |
| 加数据后 error 不再下降 | 停止采集同类数据，省预算 |

### 1.5 Feature Set（特征集合）⭐⭐⭐
**是什么**：喂给模型的所有 input variables 的集合。在我们的项目中，feature set = 10 个 block scores。
**PM 视角**：PM 最大的价值之一是利用 domain knowledge 优化 feature set——判断该加什么 feature、该删什么 feature。ML 团队只能从数据里找 statistical correlation，但 PM（尤其是有临床背景的 PM）能判断 correlation 背后有没有 causal relationship。

**例子**：Heart Failure 模型包含 urine Na（无 causal relationship → 去掉），缺少 CAD history（有 pathophysiological link → 加上）。这种判断只有 domain expert 能做。

### 1.6 Feature Selection vs Feature Engineering ⭐⭐⭐
- **Feature Selection**: 从现有 features 中选哪些放进模型、哪些去掉
- **Feature Engineering**: 创造新的 feature（比如用 block scores 算出 "average" 和 "variance"）
- PM 不需要写代码做这些，但需要能指导方向：告诉团队"这个变量没有临床意义别放"或"你们应该考虑加入 X 数据"

---

## Interview Talking Points（面试话术）

### Q: What unique value do you bring as a clinician-turned-PM?
> "My role is to bridge clinical knowledge and ML engineering. For example, if an ML team builds a heart failure prediction model and hits a performance ceiling, I can evaluate the feature set from a clinical perspective. I might identify that urine sodium was included but has no direct causal relationship to HF — it's a spurious correlation that adds noise. Meanwhile, CAD history, which has a clear pathophysiological link, might be missing. By curating features based on domain expertise rather than pure statistical correlation, I can help the team break through performance plateaus that no amount of model tuning can fix."

### Q: How do you handle stakeholders asking "why isn't the model more accurate?"
> "I understand that model performance has a theoretical upper bound determined by irreducible error. As a PM, my job is to help the team assess whether the bottleneck is the model itself, the feature set, or the fundamental noise in the data. If we're close to the human baseline, I'd redirect the conversation toward product experience improvements rather than chasing marginal accuracy gains."

### Q: What's your approach when a model hits a performance plateau?
> "I'd first evaluate whether the current feature set is missing clinically meaningful variables. As a physician, I can identify which data sources have pathophysiological relevance versus spurious correlations. This is often the highest-ROI way to break through a plateau — not a better algorithm, but better input data guided by domain expertise."

---

## Step 2: Data Loading + EDA

### 2.1 Data Merging / Join（数据合并）⭐⭐
**是什么**：通过共同的 key（如 student_id）把两个表连接成一个。
**PM 视角**：数据整合是 real-world ML 项目中最耗时的环节之一（经常占 60-80% 项目时间）。PM 要在 roadmap 中给数据清洗和整合留足时间，不要以为"拿到数据就能直接用"。

### 2.2 Correlation Heatmap（相关性热力图）⭐⭐⭐
**是什么**：用颜色深浅展示任意两个变量之间的相关程度（r 值，范围 -1 到 +1）。

**r 值速查表**：
| r 值 | 含义 |
|---|---|
| > 0.7 | Strong correlation |
| 0.4-0.7 | Moderate |
| < 0.4 | Weak |
| 负值 | Negative correlation（A 高 B 低）|

**局限性**：
- 只能发现 linear relationship
- Non-linear relationship（如 U-shaped）会被漏掉（r ≈ 0 但实际有很强关系）
- Correlation ≠ Causation

**临床 Non-linear 例子**：
- 体温和 Sepsis：高热和低体温都增加风险（U-shaped）
- 血钾和心律失常：hypokalemia 和 hyperkalemia 都危险（U-shaped）
- 如果 DS 只看 linear correlation，会误认为这些 feature "没用"而删掉

**PM 在 review meeting 应该问的问题**：
- "这个 high correlation 有 clinical basis 吗？"
- "有没有 confounding variable 造成虚假相关？"
- "除了 linear correlation，有没有做 non-linear relationship 检查？"
- "这个关系在不同 subgroup 里都成立吗？"

**怎么向 stakeholder 汇报**：
- 不要说："Neuro 和 CBSE 的 correlation 是 0.72"
- 要说："Neuroscience 对 CBSE 成绩的预测力最强，Neuro 成绩高的学生 CBSE 通过率明显更高。如果要做 early intervention，应该优先关注 Neuro 表现。"

### 2.3 Redundancy（信息重复）⭐⭐⭐
**是什么**：两个 feature 高度 correlated（r > 0.8），携带几乎相同的信息。

**是否要删的判断标准——不是看 r 值多高，而是问"这两个 feature 是否提供不同的信息"**：
| 情况 | 例子 | 行动 |
|---|---|---|
| 高 correlation + 不同 causal pathway | DM 和 MI 都导致 HF，但机制不同 | 保留两个 |
| 高 correlation + 同一东西的不同测量 | 体重 kg 和 体重 lb | 删一个 |
| 高 correlation + 一个是另一个的下游 | HbA1c 和 fasting glucose | 看情况 |

**PM 需要知道**：如果团队说"我们加了 50 个 features"，要问"有没有做 redundancy check？"

### 2.4 Confounding + Mediation（混杂 + 中介效应）⭐⭐⭐
**是什么**：DM → MI → HF 的例子。DM 同时通过直接路径（糖尿病心肌病）和间接路径（先导致 MI，MI 再导致 HF）影响结果。

**对 PM 最重要的判断**：产品是 prediction 还是 intervention？
- **Prediction 产品**：两个 correlated features 都放，越准越好
- **Intervention 产品**：必须搞清因果链，找到上游原因（控制 DM 比治 MI 更 cost-effective）

### 2.5 PM 与 Data Team 的分工 ⭐⭐⭐
| 任务 | 谁做 | PM 做什么 |
|---|---|---|
| 写代码、画图 | Data Analyst / DS | 不用你做 |
| 看 heatmap 判断哪些 feature 重要 | DA 出图 | 你看懂、问对的问题 |
| 决定加/删哪些 feature | **你做** | 基于 domain knowledge |
| 判断 correlation 有没有 causation | **你做** | 临床知识判断 |
| 数据 ID 对不上 | Data Engineer 修 | 你给团队留足时间 |

**Data Analyst 产出数据和图表，你产出 decision。**

### 2.6 EDA 实际结果（我们的项目）
- 所有 10 个 blocks 和 CBSE 都是 moderate correlation（r ≈ 0.43-0.53）
- CV（r=0.53）和 Renal（r=0.52）correlation 最高
- 35.2% 的学生 CBSE score 低于 194（passing threshold）
- 无缺失值（模拟数据，真实数据几乎不可能没缺失值）

---

## Interview Talking Points（续）

### Q: How do you work with data science teams?
> "I see my role as providing the clinical context that pure data scientists don't have. For example, a DS might drop serum potassium from a cardiac risk model because it shows near-zero linear correlation with outcomes. But I know the relationship is U-shaped — both hypokalemia and hyperkalemia increase risk. I'd flag this and suggest transforming it to 'deviation from normal range' so the model can capture the non-linear signal. Data analysts produce the charts and numbers; I produce the clinical interpretation and decisions."

### Q: What's the difference between prediction and intervention products?
> "This distinction fundamentally changes the modeling approach. For prediction — like our CBSE early warning tool — we include all correlated features because the goal is accuracy. But for intervention products, we need causal inference. For example, DM and MI are correlated and both predict heart failure. For prediction, include both. But for intervention, you need to know that controlling DM is higher leverage because it reduces HF risk through both direct and MI-mediated pathways."

### Q: Why should we hire you over a regular PM?
> "A regular PM can manage sprints, roadmaps, and stakeholders. But when the ML team says 'model accuracy is stuck at 82%,' they can only ask 'can we make it better?' I can say: 'Add CAD history — it has a direct pathophysiological link. Remove urine Na — it's a spurious correlation.' I also know that body temperature and sepsis have a U-shaped relationship that standard correlation analysis would miss. This domain knowledge is systematic, not occasional — it applies to every feature in a healthcare AI product."

---

## Step 3: Preprocessing & Feature Engineering

### 3.1 Missing Value Handling（缺失值处理）⭐⭐⭐
**核心问题不是"用什么方法填"，而是"缺失是不是 random 的"。**

**三种策略及适用场景**：
| 策略 | 适用场景 | 风险 |
|---|---|---|
| Drop（删除） | 缺失 <5%，数据量大，random missing | 如果 missing not random，会引入 Selection Bias |
| Impute（填充，如 median） | 数据量小、每条珍贵 | 假装缺失的人是"正常的"，低估风险 |
| Missing Indicator（缺失标记） | Healthcare 场景首选 | 最安全，让模型自己判断 |

**Healthcare 特殊性——Missingness is informative**：
- 没来复查的病人 → 可能病情恶化或 non-compliant
- ICU 检查缺失 → 病人可能太 unstable 无法检查
- 用药数据缺失 → 可能不按时吃药

**例子（HF 用药数据缺失）**：
- 删掉这些人 → 模型只在 compliant patients 上训练 → 对 non-compliant patients 过于乐观
- Median 填充 → 假装他们在吃药 → 同样过于乐观
- **正确做法**：创建 `med_data_missing = 1/0` 作为新 feature → 模型学到"数据缺失本身是 risk factor"

**PM Review 三个必问问题**：
1. "缺失比例是多少？缺失是 random 的吗？"
2. "填充策略有没有引入 bias？"
3. "有没有把 missingness 本身作为 feature？"

### 3.2 Feature Engineering（特征工程）⭐⭐⭐
**是什么**：从现有数据创造新的 predictive signal。

**在我们的项目中**：
- `block_average`：所有 block 均分 → 反映整体 ability level
- `block_variance`：各 block 分数的方差 → 高 = 偏科严重
- `lowest_block`：最低 block 分数 → 木桶效应

**PM 视角**：PM 不写 feature engineering 的代码，但要能指导方向。比如告诉团队"学生的成绩趋势（是在进步还是退步）可能比绝对分数更有预测力"——这种 insight 来自 domain knowledge。

### 3.3 Standardization / Scaling（标准化）⭐
**是什么**：把所有 feature 缩放到同一 scale（mean=0, std=1）。
**为什么**：Linear Regression 对 scale 敏感——如果一个 feature 范围是 0-100，另一个是 0-1，前者会被高估。
**PM 视角**：知道它存在就行，不需要深入。如果 DS 提到 scaling，你知道他们在说什么就够了。

### 3.4 Selection Bias（选择偏差）⭐⭐⭐
**是什么**：因为数据收集方式不当，导致训练数据不能代表真实人群。
**临床例子**：只用 compliant patients 训练模型 → 模型对 non-compliant patients 失效。
**PM 视角**：这是 healthcare AI 产品 FDA 审批时的重点审查项。PM 必须确保训练数据的 population 和产品目标用户一致。

---

## Interview Talking Points（续）

### Q: How do you handle missing data in healthcare ML?
> "In healthcare, I treat missing data as a signal, not just a nuisance. Patients missing outpatient medication records may be non-compliant — and non-compliance is itself a strong predictor of poor outcomes. Simply dropping these patients biases the model toward compliant patients. Imputing with median values assumes they're average, which is equally misleading. My approach is to create missingness indicator features so the model can learn that data absence is clinically meaningful."

### Q: Tell me about a time you identified a data quality issue.
> (Use our project example) "During EDA, I noticed that if we simply dropped students with missing block scores, we'd lose the exact students who need the early warning system most — the ones who may have missed exams. Instead of dropping them, I recommended creating a 'missing exam' indicator feature. This way, the model learns that missing an exam is itself a risk factor for poor CBSE performance."

### Q: Your DS team says model accuracy is 78%. What do you do?
> "Before touching the model, I'd look at the data. What's the missingness rate — is it random? In healthcare, missing data is often informative. If we're dropping those patients or imputing with median values, we might be introducing bias and losing signal. Then I'd review the feature set — are there clinically meaningful variables we're missing, or spurious correlations we should remove? Model tuning is the last step, not the first."

---

## Step 4: Model Training & Evaluation

### 4.1 Train/Test Split（训练/测试集拆分）⭐⭐⭐
**是什么**：随机把数据拆成两份——80% 训练（模型学习用），20% 测试（模拟新数据，检验真实水平）。
**为什么**：如果不拆，模型在自己学过的数据上测自己 = 考试用原题 = 分数虚高。
**PM 视角**：如果团队只给你一个 performance 数字，第一反应问："这是 training 还是 test？两个都给我看。"

### 4.2 Overfitting（过拟合）⭐⭐⭐
**是什么**：模型把训练数据里的 noise 也"背"下来了，在新数据上表现很差。

**怎么发现——只看一个数字：Train vs Test gap**
| 情况 | 判断 |
|---|---|
| Train 0.52, Test 0.37 (gap 0.15) | OK |
| Train 0.99, Test 0.22 (gap 0.77) | Overfitting，告诉团队 fix it |

**为什么复杂模型更容易 overfit**：
- Linear Regression：11 个参数，被迫学 general pattern
- XGBoost 100 棵树 depth=5：3200+ 个独立区域，有能力给每个学生单独一个预测值 → 记住了 noise

**PM 不需要管怎么修**（调 depth、减树、加 regularization 是 DS 的活），只需要看到 gap 大 → flag it。

### 4.3 Data Leakage（数据泄漏）⭐⭐⭐
**是什么**：训练时模型"偷看"了不该看到的信息。

**三种常见类型**：
1. **Target Leakage**：feature 里包含了只有知道结果后才有的信息
   - 例：预测 ICU 转入 → feature 含 "mechanical ventilation hours"（只有 ICU 才有）
2. **Train-Test Leakage**：test data 信息泄漏到 training 过程
3. **Temporal Leakage**：用了未来的数据预测过去

**PM 的检查方式**：看 feature list，问自己"在需要做预测的时间点，这个信息真的已经存在了吗？" 花 15 分钟扫一遍 feature list，可能是整个项目 ROI 最高的 15 分钟。

**DS 不一定能发现**——他们不知道 mechanical ventilation 只有 ICU 才有，不知道 discharge code 是出院才补录的。你的临床知识是 safety guardrail。

### 4.4 模型复杂度 vs 数据量 ⭐⭐⭐
| 数据量 | 适合的模型 | PM 行动 |
|---|---|---|
| < 500 | 简单模型（Linear Regression） | 推动团队收集更多数据 |
| 500-5000 | 可以比对多种，严格控制 overfitting | 平衡优化 vs 数据获取 |
| > 10000 | 复杂模型可以放开用 | 让 DS 自由发挥 |

**PM 核心判断**：小数据时，与其花 3 个月调模型，不如花 3 个月获取更多数据——这是 resource allocation 决策。

### 4.5 Cross-Validation（交叉验证）⭐⭐
**是什么**：把数据拆成 K 份，轮流当 test set，平均结果。比单次 80/20 拆分更可靠。
**PM 视角**：知道"CV R² = 0.45"比单次 test R² 更可信就够了。K=5 还是 K=10 是 DS 的决定。

### 4.6 实际结果（我们的项目）
- Linear Regression 赢了（Test R² = 0.37），因为数据少 + 关系线性
- XGBoost 严重 overfit（Train 0.99 vs Test 0.22）
- CV R² 都在 0.40-0.46，说明真实 performance 在这个水平
- R² = 0.37-0.46 意味着 block scores 只能解释 ~40% 的 CBSE 变化，提升需要更多 features

---

## Interview Talking Points（续）

### Q: Your DS team says model accuracy is 78%. What do you do? (Complete answer)
> "First, I'd clarify what 78% means — accuracy, AUC, sensitivity? For clinical tools, raw accuracy can be misleading with imbalanced data. Then I'd benchmark against human baseline. Next, I'd investigate the data: missingness patterns, selection bias, and review the feature set with clinical expertise. I'd check for overfitting — show me train vs test performance. I'd ask for subgroup analysis to ensure the model works across patient demographics. Finally, I'd make a product decision: is 78% good enough to launch with human-in-the-loop, or do we need to improve first?"

### Q: How do you choose between different ML models?
> "I match model complexity to data size. In a project with 400 training samples, I saw XGBoost achieve 99% on training data but only 22% on test data — classic overfitting. Linear Regression, the simplest model, actually generalized best. As PM, when data is limited, I'd rather invest in getting more data than tuning a complex model. That's a higher-ROI resource allocation."

### Q: How do you catch overfitting or data leakage?
> "For overfitting, I always ask to see both training and test performance. A large gap is a red flag. For data leakage, I review the feature list with my clinical knowledge — I can spot variables that wouldn't be available at prediction time, like mechanical ventilation hours in an ICU admission prediction model. This 15-minute feature list review is often the highest-ROI activity in the entire project."

---

## Step 5: Feature Importance

### 5.1 Linear Regression Coefficients ⭐⭐⭐
**是什么**：每个 coefficient 表示"该 feature 增加 1 个标准差时，CBSE 分数变化多少"。
**PM 视角**：最容易向 stakeholder 解释。"Renal block 每提高 1 分，CBSE 预计提高 X 分。"

### 5.2 SHAP Values ⭐⭐⭐
**是什么**：把每个学生的预测拆解成每个 feature 的正/负贡献。

**为什么 healthcare AI 必须有**：
- **临床信任**：医生不接受黑箱。"为什么这个学生被标记为高风险？"→ SHAP 说"主要因为 Neuro 太低（-8 分）"
- **Actionable Insight**：不只说"会 fail"，还说"为什么会 fail"→ 针对性干预
- **FDA Regulatory**：越来越多 healthcare AI 产品被要求提供 model explainability

**SHAP 的局限**：解释的是模型的逻辑，不是现实的因果关系。模型可能因为错误的原因做出正确的预测。

### 5.3 Permutation Importance ⭐⭐
**是什么**：把一个 feature 的值随机打乱，看模型 performance 下降多少。下降多 = 该 feature 重要。
**PM 视角**：比 coefficients 更 robust，因为不依赖于模型类型（model-agnostic）。

### 5.4 实际结果（我们的项目）
- `lowest_block` 影响最大（木桶效应）
- Renal 和 CV 是原始 block 中最强的 predictor
- MSK 和 Behavioral Science 影响最小

---

## Step 6: Early Warning System

### 6.1 Model Deployment（模型部署）⭐⭐⭐
**是什么**：用训练好的模型预测新数据（从没见过的学生）。
**PM 视角**：训练时的 performance 不代表部署后的 performance。PM 需要确保 monitoring（上线后持续追踪模型表现）。

### 6.2 Threshold Setting（阈值设定）⭐⭐⭐
**是什么**：什么分数以下算"at risk"？我们用 194（USMLE Step 1 旧 passing score）。
**PM 视角**：阈值不是技术决定，是产品/业务决定。降低阈值 = 标记更少学生 = 漏掉更多真正 at-risk 的人。升高阈值 = 标记更多 = 误报更多。PM 要根据 false negative vs false positive 的 clinical cost 来决定。

### 6.3 Data Privacy（数据隐私）⭐⭐⭐
**FERPA**：美国学生成绩是受法律保护的 PII。Portfolio 中绝对不能用真实学生数据。
**正确做法**：用 synthetic data 展示方法论，注明"designed to work with real data when deployed"。

---

## Step 7: End-to-End Pipeline

### 7.1 Modular Code（模块化代码）⭐⭐
**是什么**：每个功能一个文件（data_loader, preprocessing, models, etc.），main.py 串联。
**PM 视角**：PM 不需要写模块化代码，但要知道好的代码结构 = 团队协作效率高 + 维护成本低。如果团队的代码全写在一个 notebook 里，是 red flag。

---

## Interview Talking Points（完整版）

### Q: What unique value do you bring as a clinician-turned-PM?
> "My role is to bridge clinical knowledge and ML engineering. For example, if an ML team builds a heart failure prediction model and hits a performance ceiling, I can evaluate the feature set from a clinical perspective. I might identify that urine sodium was included but has no direct causal relationship to HF — it's a spurious correlation that adds noise. Meanwhile, CAD history, which has a clear pathophysiological link, might be missing. By curating features based on domain expertise rather than pure statistical correlation, I can help the team break through performance plateaus that no amount of model tuning can fix."

### Q: How do you handle stakeholders asking "why isn't the model more accurate?"
> "I understand that model performance has a theoretical upper bound determined by irreducible error. As a PM, my job is to help the team assess whether the bottleneck is the model itself, the feature set, or the fundamental noise in the data. If we're close to the human baseline, I'd redirect the conversation toward product experience improvements rather than chasing marginal accuracy gains."

### Q: What's your approach when a model hits a performance plateau?
> "I'd first evaluate whether the current feature set is missing clinically meaningful variables. As a physician, I can identify which data sources have pathophysiological relevance versus spurious correlations. This is often the highest-ROI way to break through a plateau — not a better algorithm, but better input data guided by domain expertise."

### Q: How do you work with data science teams?
> "I see my role as providing the clinical context that pure data scientists don't have. For example, a DS might drop serum potassium from a cardiac risk model because it shows near-zero linear correlation with outcomes. But I know the relationship is U-shaped — both hypokalemia and hyperkalemia increase risk. I'd flag this and suggest transforming it to 'deviation from normal range' so the model can capture the non-linear signal. Data analysts produce the charts and numbers; I produce the clinical interpretation and decisions."

### Q: What's the difference between prediction and intervention products?
> "This distinction fundamentally changes the modeling approach. For prediction — like our CBSE early warning tool — we include all correlated features because the goal is accuracy. But for intervention products, we need causal inference. For example, DM and MI are correlated and both predict heart failure. For prediction, include both. But for intervention, you need to know that controlling DM is higher leverage because it reduces HF risk through both direct and MI-mediated pathways."

### Q: Why should we hire you over a regular PM?
> "A regular PM can manage sprints, roadmaps, and stakeholders. But when the ML team says 'model accuracy is stuck at 82%,' they can only ask 'can we make it better?' I can say: 'Add CAD history — it has a direct pathophysiological link. Remove urine Na — it's a spurious correlation.' I also know that body temperature and sepsis have a U-shaped relationship that standard correlation analysis would miss. This domain knowledge is systematic, not occasional — it applies to every feature in a healthcare AI product."

### Q: Your DS team says model accuracy is 78%. What do you do? (Complete 7-step answer)
> "First, I'd clarify what 78% means — accuracy, AUC, sensitivity? For clinical tools, raw accuracy can be misleading with imbalanced data. Then I'd benchmark against human baseline. Next, I'd investigate the data: missingness patterns, selection bias, and review the feature set with clinical expertise. I'd check for overfitting — show me train vs test performance. I'd ask for subgroup analysis to ensure the model works across patient demographics. Finally, I'd make a product decision: is 78% good enough to launch with human-in-the-loop, or do we need to improve first?"

### Q: How do you choose between different ML models?
> "I match model complexity to data size. In a project with 400 training samples, I saw XGBoost achieve 99% on training data but only 22% on test data — classic overfitting. Linear Regression, the simplest model, actually generalized best. As PM, when data is limited, I'd rather invest in getting more data than tuning a complex model. That's a higher-ROI resource allocation."

### Q: How do you catch overfitting or data leakage?
> "For overfitting, I always ask to see both training and test performance. A large gap is a red flag. For data leakage, I review the feature list with my clinical knowledge — I can spot variables that wouldn't be available at prediction time, like mechanical ventilation hours in an ICU admission prediction model. This 15-minute feature list review is often the highest-ROI activity in the entire project."

### Q: How do you handle missing data in healthcare ML?
> "In healthcare, I treat missing data as a signal, not just a nuisance. Patients missing outpatient medication records may be non-compliant — and non-compliance is itself a strong predictor of poor outcomes. Simply dropping these patients biases the model toward compliant patients. Imputing with median values assumes they're average, which is equally misleading. My approach is to create missingness indicator features so the model can learn that data absence is clinically meaningful."

### Q: How do you ensure model explainability?
> "I consider model explainability non-negotiable in healthcare AI. Clinicians won't adopt a tool they can't understand, and patients deserve to know why an AI flagged them as high risk. SHAP values let us provide per-patient explanations — not just 'you're at risk,' but 'your Neuroscience performance is the primary driver of this prediction.' This transforms a prediction tool into an actionable intervention tool. It's also increasingly required for FDA clearance."
