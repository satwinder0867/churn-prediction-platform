# Problem Framing: Customer Churn Prediction

## 1. Business problem

A telecom company is losing customers. Winning a new customer costs far more than keeping an existing one, so the company wants to **identify customers who are likely to leave early enough to act**, and spend its retention budget where it has the highest return.

**Goal:** predict, for each active customer, the probability that they will churn, and use that probability to decide **whether and how** to intervene.

## 2. Churn definition

- **Churn rate (business KPI):** the share of customers who stop doing business with the company during a given period (churned customers ÷ total customers).
- **Churn label (model target):** per customer, a binary label.

| Label | Rule |
|---|---|
| `1` (churned) | `Churn Value == 1`: the customer left the company within the last month |
| `0` (retained) | Otherwise |

This is **contractual churn**: customers explicitly cancel, so the label doesn't need to be inferred from inactivity.

## 3. Dataset

- **Source:** IBM Telco Customer Churn (extended version), `data/raw/Telco_customer_churn.xlsx`
- **Size:** 7,043 customers × 33 columns, one row per customer
- **Churn rate:** 1,869 / 7,043 ≈ **26.5%**, so the classes are imbalanced
- **Region:** California, USA. Monetary values are in **USD**.

### Column groups

| Group | Columns | Use |
|---|---|---|
| Identifier | `CustomerID` | Key only, **not a feature** |
| Constant | `Count`, `Country`, `State` | Single value, **drop** |
| Geography | `City`, `Zip Code`, `Lat Long`, `Latitude`, `Longitude` | Optional. High cardinality; `Lat Long` duplicates `Latitude` + `Longitude` |
| Demographics | `Gender`, `Senior Citizen`, `Partner`, `Dependents` | Features |
| Account | `Tenure Months`, `Contract`, `Paperless Billing`, `Payment Method`, `Monthly Charges`, `Total Charges` | Features |
| Services | `Phone Service`, `Multiple Lines`, `Internet Service`, `Online Security`, `Online Backup`, `Device Protection`, `Tech Support`, `Streaming TV`, `Streaming Movies` | Features |
| **Target** | `Churn Value` (0/1) | **Label** |
| **Leakage, never use as features** | `Churn Label`, `Churn Score`, `Churn Reason`, `CLTV` | See section 8 |

### Known data quality issues

- `Total Charges` has **11 blank values**, all for customers with `Tenure Months == 0` (new customers who haven't been billed yet). It must be converted to numeric; treat the blanks as `0`.

## 4. Observation and prediction windows

The dataset is a **static snapshot**: there are no event timestamps, only each customer's state at a single point in time.

| Concept | In this dataset | Target production design |
|---|---|---|
| Observation window | Implicit: the customer's whole history up to the snapshot (`Tenure Months`) | Last **6 months** of usage, billing and support data |
| Gap (lead time) | **None.** Features and label are from the same moment | **14 days**, enough for the email → wait → call workflow |
| Prediction window | **1 month** (left within the last month) | Next **30 days** after the gap |

```
|------ observation (6 months) ------|-- gap (14 days) --|---- prediction (30 days) ----|
        features computed here          intervention time      did the customer churn?
                                 ^
                           scoring date
```

**Limitation:** because this is a snapshot, the project shows the modeling and decision logic. A production system would need **point-in-time feature generation** so that no feature uses information from after the scoring date.

## 5. Intervention strategy

A two-tier retention campaign, escalating only when needed:

| Tier | Action | Who | Cost | Assumed save rate* |
|---|---|---|---|---|
| 0 | Do nothing | p < 0.013 | $0 | n/a |
| 1 | Retention **email** | 0.013 ≤ p < 0.21 | **$1** | 10% |
| 2 | Email, then if no response in **7 days**, a **call + discount coupon** (20% off for 3 months) | p ≥ 0.21 | **$50** ($10 agent time + $40 discount) | 30% |

\* Save rate = the share of true churners who stay because of the action. These are assumptions, to be replaced with A/B test results in production.

**Capacity constraint:** the call team can handle a limited number of calls per month (assumed **K = 200**). If more customers exceed the Tier 2 threshold than the team can call, rank them by `p` (or by expected value, section 6) and call the top K.

## 6. Cost of errors and decision thresholds

### Value of a saved customer

- Average `Monthly Charges` ≈ **$65**
- Assume a saved customer stays **12 more months**, so value ≈ 65 × 12 ≈ **$780**
- In evaluation, use the per-customer value: `Monthly Charges × 12`

`CLTV` (mean ≈ $4,400) is a **pre-computed IBM estimate of total lifetime value**, not of *remaining* value, and its calculation is unknown. It's used only for a **sensitivity analysis** (how results change if saved customers are worth more), never as a feature.

### Error costs

| Outcome | Meaning | Cost |
|---|---|---|
| True positive | Churner contacted | Contact cost; recovers `save_rate × value` on average |
| **False negative** | Churner missed | Lost customer, **≈ $780** |
| **False positive** | Non-churner contacted | Wasted contact: **$1** (email) or **$50** (call + discount given to someone who would have stayed) |
| True negative | Non-churner ignored | $0 |

A missed churner costs about **15× more** than a wasted call and **780× more** than a wasted email, so the system should lean towards **catching churners**.

### Threshold rule (expected value)

Contact a customer only if the expected benefit is greater than the cost:

```
p × save_rate × value  >  cost      ⇒      p* = cost / (save_rate × value)
```

| Tier | Calculation | Threshold p* |
|---|---|---|
| Email | 1 / (0.10 × 780) | **≈ 0.013** |
| Call + discount | 50 / (0.30 × 780) | **≈ 0.21** |

These thresholds assume the predicted probabilities are **well calibrated** (see section 7).

## 7. Success metrics

| Level | Metric | Why |
|---|---|---|
| Model selection (primary) | **PR-AUC** | Focuses on the minority churn class. Random baseline ≈ 0.265 (the churn rate) |
| Model selection (secondary) | ROC-AUC | Standard ranking quality. Random baseline = 0.5 |
| Probability quality | **Brier score**, calibration curve | Thresholds in section 6 only work if `p` means a real probability |
| Tier 2 (calls) | **Precision@K** (K = 200) | Share of called customers who were real churners |
| Tier 1 (emails) | **Recall** at p ≥ 0.013 | Share of churners reached |
| Business | **Expected net savings ($)** on the test set | Σ(save_rate × value for contacted churners) − Σ(contact costs) |
| Production impact | Churn rate of treated vs. **control group** (A/B test) | Proves the campaign itself reduces churn |

**Baselines to beat:** (a) contact nobody, (b) contact everyone, (c) a simple rule, e.g. "call all month-to-month customers."

**Not used:** accuracy. Predicting "no one churns" already scores ≈ 73.5%.

## 8. Data leakage risks

| Column | Why it leaks | Decision |
|---|---|---|
| `Churn Label` | Same as the target ("Yes"/"No") | Drop |
| `Churn Score` | IBM's own model output, computed with knowledge of churn. Every churner has a score ≥ 65 | Drop |
| `Churn Reason` | Only filled in for customers who already churned | Drop |
| `CLTV` | Pre-computed with unknown logic; may use churn information | Drop from features; use only for sensitivity analysis |
| `CustomerID` | Identifier, no predictive meaning | Drop |

**Preprocessing leakage:** split train and test **first**. Fit all encoders, scalers and resampling (e.g. SMOTE) **on training data only**, inside a scikit-learn `Pipeline`, so cross-validation folds never see validation or test statistics. Use a **stratified** split to keep the 26.5% churn rate in every fold.

## 9. Initial hypotheses (to check in EDA)

| Segment | Churn rate |
|---|---|
| Month-to-month contract | ~42.7% |
| One-year contract | ~11.3% |
| Two-year contract | ~2.8% |
| Fiber optic internet | ~41.9% |
| Electronic check payment | ~45.3% |

Other expected drivers: short tenure, high monthly charges, no tech support or online security.

## 10. Prediction vs. intervention (scope note)

The model predicts **who is at risk**, not **who will respond to an offer**. Customers fall into four groups:

- **Persuadables:** stay only if contacted. **The real target.**
- **Sure things:** stay anyway. A discount is wasted.
- **Lost causes:** leave anyway. Effort is wasted.
- **Sleeping dogs:** contact prompts them to leave. Avoid.

This dataset has no treatment history, so **uplift modeling is out of scope**. It's the natural next step after running an A/B test with a control group.

## 11. Assumptions to revisit

- Save rates (10% email, 30% call) and costs ($1, $50) are estimates.
- 12-month retention value for a saved customer.
- Call capacity K = 200 per month.
- The snapshot label (churned within the last month) stands in for a real 30-day prediction window.
