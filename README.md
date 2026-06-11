# Machine Learning from Disaster — Titanic Survival Prediction

**Author:** Marcus Lui  
**Date:** July 2025  
**Competition:** [Kaggle — Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)

---

## Overview

This project tackles the classic Kaggle Titanic competition: given passenger data, build a model to predict who survived the 1912 sinking. The Titanic carried 2,224 passengers and crew; 1,502 (roughly 68%) did not survive, yielding a baseline survival rate of about 32%. The goal was to move well beyond that baseline using exploratory analysis and multiple classification models.

---

## Dataset

- **Source:** [kaggle.com/competitions/titanic](https://www.kaggle.com/competitions/titanic)
- **Training set:** 891 passengers with labels (40% of total known data)
- **Test set:** 418 passengers without labels (19%)
- **Features:** PassengerId, Pclass, Name, Sex, Age, SibSp (siblings/spouses), Parch (parents/children), Ticket, Fare, Cabin, Embarked

---

## Procedure

### 1. Data Cleaning
The datasets contained **1,279 total null values** across three columns:

- **Age** — filled with column mean
- **Embarked** — filled with column mode
- **Fare** (test set only) — filled with column mean
- **Cabin** — rather than dropping, null cabins were labeled `'U'` (unknown), then binarized into a new `Cabin_Cat` feature (0 = unknown, 1 = known cabin)

### 2. Feature Engineering
New features were derived to improve model signal:

- **Famsi** — total family size = SibSp + Parch + 1
- **Famsi_Grouped** — categorized family size into Alone (1), Small (2–4), Medium (5–6), Large (7–11)
- **Age_Group** — age bucketed in 10-year intervals
- **Fare_Group** — fare bucketed in £25 intervals
- **Cabin_Cat** — binary: known vs. unknown cabin assignment

### 3. Exploratory Data Analysis (EDA)
Visualizations (count plots, scatter plots, heatmaps) were produced for all major features against survival. Four hypotheses were tested and each was confirmed.

### 4. Encoding & Preprocessing
- **Sex** encoded as binary (0 = male, 1 = female)
- **Embarked** encoded ordinally (Southampton=1, Cherbourg=2, Queenstown=3)
- **Famsi_Grouped** one-hot encoded via `OneHotEncoder`
- Non-informative columns dropped before modeling: Name, Ticket, Cabin, PassengerId

### 5. Modeling — Round 1 (Baseline)
Six classifiers were trained and evaluated on train/test accuracy, R², MSE, precision, and recall:

| Model | Notes |
|---|---|
| Logistic Regression (LR) | Best test accuracy at ~81% |
| Random Forest (RF) | 98.45% train accuracy — overfit |
| Decision Tree (DT) | 98.45% train accuracy — overfit |
| K-Nearest Neighbors (KNN) | Underperformed initially |
| Support Vector Classification (SVC) | 68.1% train / 65.9% test |
| Gaussian Naive Bayes (GNB) | Moderate performance |

Decision Tree and Random Forest showed large train-test deltas, indicating overfitting. The target range for acceptable models was 70–90% test accuracy.

### 6. Modeling — Round 2 (GridSearchCV Tuning)
All six models were re-tuned using `GridSearchCV` with `StratifiedKFold` cross-validation. Results after tuning:

| Model | Test Accuracy |
|---|---|
| Logistic Regression | **77.511%** |
| Random Forest | **77.033%** |
| Decision Tree | **75.598%** |
| Support Vector Classification | **73.684%** |
| Gaussian Naive Bayes | **72.248%** |
| K-Nearest Neighbors | **66.028%** |
| **Average** | **73.684%** |

KNN and SVC each improved by approximately 5% and 13% respectively after tuning. Logistic Regression and Decision Tree showed the best combined R² and MSE metrics. A custom "stack score" (R² + Precision + Recall − MSE) was also computed for holistic model comparison.

### 7. Prediction & Submission
All six tuned models generated predictions on the test set, exported as individual CSV files (`sub1.csv` through `sub6.csv`). Submission rankings on Kaggle matched model rankings closely, with Logistic Regression placing first and KNN placing last in both.

### 8. Personal Survival Simulation
As a fun application, the author ran the trained models against a hypothetical input representing themselves:

- **As a 1st class passenger:** ~83% predicted survival
- **Otherwise:** ~25% predicted survival
- Ticket cost meaningfully increased survival odds; traveling as a couple was better than with a larger family; embarkation port had no effect on the simulation output.

---

## Hypotheses & Findings

### Hypothesis 1 — Women survive more than men ✅ Confirmed
Women were **55% more likely to survive** than men.
- Male survival rate: **19%**
- Female survival rate: **74%**

### Hypothesis 2 — Solo travelers survive more ❌ Partially Refuted
Solo travelers did *not* have the best odds. Small groups (2–4) outperformed everyone.
- Alone: **30%**
- Small (2–4): **58%**
- Medium (5–6): **16%**
- Large (7–11): **16%**

### Hypothesis 3 — 1st class passengers survive more ✅ Confirmed
1st class passengers were **33% more likely** to survive than 2nd class, and **159% more likely** than 3rd class.
- 1st class: **62.96%**
- 2nd class: **47.28%**
- 3rd class: **24.23%**

### Hypothesis 4 — Higher fare = higher survival ✅ Confirmed
Survival rate increased notably for fares above **£50**.

---

## Additional Findings

- **Embarkation port mattered:** Cherbourg passengers survived at **56%**, vs. Southampton at **34%** and Queenstown at **39%**
- **Age effects:** Children under 10 had a **61.29%** survival rate — the highest of any age group. Adults over 80 also had notably high survival, though the sample was very small.
- **Cabin assignment:** Having a known cabin number was positively correlated with survival, likely a proxy for higher class.

---

## Survivor Profile Summary

A passenger was more likely to survive if they were:
- **Female**
- **Traveling in a small group (2–4 people)**
- **First class** and/or paid a **higher fare**
- **Embarked from Cherbourg**
- A **child under 10**

---

## Tools & Libraries

- **Language:** Python 3 (Jupyter Notebook)
- **Data:** pandas, numpy
- **Visualization:** matplotlib, seaborn, plotly
- **Modeling:** scikit-learn (LogisticRegression, RandomForestClassifier, DecisionTreeClassifier, KNeighborsClassifier, SVC, GaussianNB)
- **Tuning:** GridSearchCV, StratifiedKFold
- **Metrics:** accuracy, R², MSE, precision, recall, confusion matrix, ROC/AUC

---

## File Structure

```
├── Data/
│   ├── titanic_train.csv
│   ├── titanic_test.csv
│   └── gender_submission.csv
├── Images/Titanic/          # All EDA and model visualizations
├── Titanic/                 # Submission CSVs (sub1–sub6)
└── Titanic_Survival.ipynb   # Main notebook
```
