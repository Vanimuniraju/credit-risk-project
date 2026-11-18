# Cost-Optimized Credit Risk Assessment Pipeline

A machine learning project built with Scikit-Learn that predicts credit risk using the **UCI Statlog (German Credit Data)** dataset. This repository demonstrates end-to-end Machine Learning hygiene, including leak-free preprocessing pipelines, hyperparameter tuning via cross-validation, and financial cost-sensitive threshold optimization.

---

## Executive Summary & Business Context

In commercial lending, standard classification metrics (like accuracy) often fail because they assume all classification errors carry equal weight. In credit risk:
* **False Positives (FP):** Declining a good applicant leads to an **opportunity cost** (e.g., lost interest income).
* **False Negatives (FN):** Approving a defaulting applicant leads to a **direct financial loss** (e.g., unrecoverable principal debt).

A defaulted loan is typically **10x more costly** than a turned-away customer. This project addresses that asymmetry by optimizing decision thresholds based on real-world financial impact rather than arbitrary probability defaults ($0.5$).

---

## Dataset Overview

* **Source:** [UCI Machine Learning Repository - Statlog (German Credit Data)](https://archive.ics.uci.edu/ml/datasets/Statlog+(German+Credit+Data))
* **Instances:** 1,000 credit applicants
* **Features:** 20 categorical and numerical attributes (checking account status, credit history, purpose, savings, duration, age, etc.)
* **Target:** Binary classification (`1`: Good Risk, `0`: Bad Risk)

---

## Key Architecture & ML Hygiene

1. **Leak-Free Pipelines:** Preprocessing steps (numerical imputation with median + `StandardScaler`, categorical imputation + `OneHotEncoder`) are encapsulated inside a Scikit-Learn `Pipeline` and `ColumnTransformer`.
2. **Hyperparameter Tuning:** 5-fold cross-validation (`GridSearchCV`) tunes model hyperparameters (`n_estimators`, `max_depth`, `min_samples_split`) without exposing validation folds to preprocessing parameters.
3. **Imbalance-Aware Metrics:** Evaluated using ROC-AUC, Precision-Recall curves, and Confusion Matrices rather than baseline accuracy alone.

---

## Results & Business Impact

| Metric | Score / Outcome |
| :--- | :--- |
| **Model** | Random Forest Classifier |
| **Cross-Validation ROC-AUC** | ~0.78 |
| **Default Threshold ($0.50$)** | High financial risk due to unoptimized False Negatives |
| **Cost-Optimized Threshold** | **~0.32** (Minimizes overall institutional financial risk) |

### Optimal Underwriting Strategy
* **Auto-Approval ($\ge 0.65$):** Low-risk applicants undergo fast-track automated approval.
* **Manual Review ($0.32 - 0.65$):** Borderline applicants are flagged for human review or conditional terms (e.g., higher interest rates or collateral requirement).
* **Auto-Decline ($< 0.32$):** High probability of default; declined to protect institutional capital.

---

## Getting Started

### Prerequisites
* Python 3.8+
* Required packages: `pandas`, `numpy`, `scikit-learn`, `matplotlib`

### Installation & Execution
```bash
# Clone the repository
git clone https://github.com/Vanimuniraju/credit-risk-pipeline.git
cd credit-risk-pipeline

# Install dependencies
pip install -r requirements.txt

# Run the main pipeline script
python main.py
```

---

## Repository Structure

```
├── data/                  # Raw and processed datasets
├── notebooks/             # Exploratory Data Analysis & experiments
├── src/                   # Production Python scripts
│   ├── pipeline.py        # Scikit-Learn ColumnTransformer and Pipeline setup
│   ├── train.py           # GridSearchCV and model training
│   └── evaluate.py        # Cost optimization and performance plotting
├── README.md              # Project documentation
└── requirements.txt       # Dependencies
```

---

## Author
* Vanimuniraju – [GitHub Profile](https://github.com/Vanimuniraju) | [LinkedIn](https://www.linkedin.com/in/vanishree-m-a55b6a1a2)