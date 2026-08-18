# 📊 Analyzing and Predicting Adult Census Income

Machine Learning course project (Team 27) — CdLM in Data Science, Università degli Studi di Milano-Bicocca. Submitted February 2025.

**Authors:** Arnab Biswas · Francesca Negri · Any Das · Farabi Issa · Tahira Rezaie

---

## 📌 Overview

This project predicts whether a US adult earns more or less than $50K per year, using the 1994 US Census **Adult Income** dataset and a range of socio-economic and demographic features. The full pipeline — cleaning, exploratory analysis, feature engineering, model training, and evaluation — was built visually in **KNIME**, with the preprocessing steps cross-checked against an equivalent Python (pandas) implementation to confirm consistency.

## 📂 Dataset

- **Source:** [Adult Census Income Dataset](https://www.kaggle.com/datasets/uciml/adult-census-income) (Kaggle / UCI), originally extracted from the US Census Bureau, 1994
- **Size:** ~32,561 records after cleaning, 15 variables (categorical + numerical)
- **Target:** `income` — `>50K` or `≤50K` (binary, imbalanced: only ~7,508 records are `>50K`)

| Variable | Description |
|---|---|
| age | Age of the individual |
| workclass | Type of employment (Private, Self-emp, Government, ...) |
| fnlwgt | Final weight assigned to represent the population |
| education / education-num | Education level / years of education |
| marital-status | Marital status |
| occupation | Type of occupation |
| relationship | Relationship to head of household |
| race, sex | Demographic attributes |
| capital-gain / capital-loss | Capital gains/losses |
| hours-per-week | Hours worked per week |
| native-country | Country of origin |
| income | Target variable |

**Demographic snapshot:** ~61% of records are aged 16–41, ~68% are male, ~86% are of White ethnicity, and the `fnlwgt` feature is heavily concentrated (~99%) in the 12K–50K range.

## 🛠️ Tech Stack

- **KNIME Analytics Platform** — full visual workflow (cleaning, EDA, modeling, evaluation)
- **Python (pandas, scikit-learn's `LabelEncoder`)** — used in parallel to verify the KNIME preprocessing results matched

## 🧪 Methodology

**1. Data cleaning** — missing values were originally encoded as `"?"` rather than nulls. These were converted to `NaN`, found in `workclass` (1,836), `occupation` (1,843), and `native-country` (583) — 4,262 rows total, removed to leave a clean dataset of 30,000+ rows.

**2. Feature scaling & encoding** — numerical features standardized with **Z-score normalization**; categorical features label-encoded. Both were implemented in KNIME and cross-verified with an equivalent Python/pandas script.

**3. Exploratory analysis** — correlation matrix and distribution plots for numerical features, plus bar charts and heatmaps of income by gender and workclass.

**4. Model evaluation strategies:**
- **Holdout** — 80/20 train-test split with a fixed random seed
- **5-fold Cross Validation**
- **Feature Selection** — Correlation Feature Selection (CFS) applied after undersampling the majority class to address class imbalance

**5. Models trained:** J48 (C4.5 Decision Tree), Logistic Regression, Naïve Bayes, SMO (SVM via Sequential Minimal Optimization), Gradient Boosted Trees (GBT), Random Forest, and Grading (a meta-learner, cross-validation only).

**6. Evaluation metrics:** Accuracy, Precision, Recall, F1-measure, Cohen's Kappa, Specificity, AUC-ROC.

## 📊 Results

### Holdout (80/20 split)

| Model | Accuracy | Cohen's Kappa | ≤50K F1 | >50K F1 |
|---|---|---|---|---|
| **Random Forest** | **95.6%** | **0.878** | 97.1% | 90.7% |
| Gradient Boosting | 83.6% | 0.522 | 89.5% | 62.6% |
| SMO | 83.3% | 0.507 | 89.4% | 61.1% |
| J48 (Decision Tree) | 83.0% | 0.502 | 89.1% | 60.9% |
| Logistic Regression | 82.8% | 0.489 | 89.1% | 59.6% |
| Naïve Bayes | 80.8% | 0.481 | 87.3% | 60.9% |

### 5-Fold Cross Validation

| Model | Accuracy | Cohen's Kappa | ≤50K F1 | >50K F1 |
|---|---|---|---|---|
| **J48 (Decision Tree)** | **83.5%** | **0.524** | 89.4% | 62.9% |
| Logistic Regression | 83.2% | 0.506 | 89.3% | 61.1% |
| SMO | 81.9% | 0.485 | 88.3% | 60.1% |
| Naïve Bayes | 81.4% | 0.501 | 87.7% | 62.5% |
| Random Forest | 80.4% | 0.446 | 87.2% | 57.4% |
| Gradient Boosting | 75.9% | 0.000 | 86.3% | 0.0% |

### After Feature Selection + Undersampling

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 79.8% | 0.777 | 0.837 | 0.806 | **0.876** |
| Naïve Bayes | 79.2% | 0.768 | 0.837 | 0.801 | 0.876 |
| J48 | 79.5% | 0.781 | 0.819 | 0.800 | 0.856 |
| Multilayer Perceptron | 78.9% | 0.773 | 0.818 | 0.795 | 0.869 |
| SMO | 78.9% | 0.775 | 0.815 | 0.794 | 0.789 |
| Random Forest | 75.7% | 0.747 | 0.776 | 0.761 | 0.826 |

## 🔑 Key Findings

- **Random Forest topped the Holdout evaluation (95.6%) but dropped sharply under Cross-Validation (80.4%)** — a strong sign that the Holdout result was optimistic and overfit to a single train/test split.
- **J48 (Decision Tree) was the most consistently balanced model** across both Holdout and Cross-Validation, staying around 83% in both.
- **Gradient Boosted Trees struggled badly under Cross-Validation**, completely failing to identify the minority `>50K` class (0% recall) — a sign of high sensitivity to class imbalance.
- After addressing class imbalance through **undersampling + Correlation Feature Selection**, **Logistic Regression** achieved the best AUC (0.876), making it the preferred model once class imbalance is properly handled.

## 📁 Repository Structure

```
├── Adult_Census_Income_Report.pdf   # Full project report
├── Adult_Census_Knime.knwf          # KNIME workflow (cleaning → EDA → modeling → evaluation)
├── adult_census_income_dataset.csv  # adultcensusincome.csv
└── README.md
```

## ▶️ Running the Workflow

- Requires the free [KNIME Analytics Platform](https://www.knime.com/downloads).
- **Important:** the `CSV Reader` node in the workflow currently points to a local file path from one teammate's machine. After downloading the dataset (see below), open the workflow in KNIME and re-point the `CSV Reader` node to your local copy before running.

## 📥 About the Dataset File

The Adult Census Income dataset is public and freely available (Kaggle/UCI), so — unlike a project using restricted or proprietary data — there's no licensing issue with including it here. Since it's a small CSV (a few MB), adding it to a `data/` folder is a reasonable option for reproducibility; the alternative is linking to the Kaggle page and letting users download it themselves. Either way, remember to update the `CSV Reader` node's path afterward.

## 📚 References

1. [Adult Census Income Dataset](https://www.kaggle.com/datasets/uciml/adult-census-income) — Kaggle.
2. Kavyasrirelangi. "From Hold-Out to K-Fold: Understanding Cross-Validation Methods in Machine Learning." Medium.
3. Pranckevicius, T. & Marcinkevičius, V. Comparison of Naive Bayes, Random Forest, Decision Tree, Support Vector Machines, and Logistic Regression Classifiers for Text Reviews Classification. *Baltic Journal of Modern Computing* 5 (2017).
4. McGill Science Undergraduate Research Journal (MSURJ, Montreal, 2022).
5. Google. "Classification: Accuracy, recall, precision, and related metrics." Google Developers.
