# Breast Cancer Cytological Classification

A machine learning project built to accurately classify breast neoplasms as benign or malignant using quantitative cytological features from the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.

This project emphasizes **rigorous, leakage-free validation** to ensure that data preprocessing and feature selection do not artificially inflate model performance.

---

## 🚀 Key Features
* **Leakage-Free Validation:** All data scaling, correlation filtering, and feature selection are strictly contained within cross-validation folds.
* **Model Benchmarking:** Compares four different supervised learning models using R's `tidymodels` framework.
  * Extreme Gradient Boosting (XGBoost)
  * L1-Penalized Logistic Regression (Lasso)
  * Support Vector Machine (RBF Kernel)
  * Random Forest
* **High Performance:** The optimized XGBoost model achieved a top-tier **0.999 ROC-AUC** and **100% test sensitivity** on the hold-out test set.
* **Explainable AI (XAI):** Uses feature importance analysis to identify key clinical predictors, highlighting that *concave points (worst)* and *area (mean)* carry the highest predictive weight.

---

## 📊 Results Summary

| Model Family | ROC-AUC (CV) | Accuracy (CV) |
| :--- | :--- | :--- |
| **XGBoost** | 0.995 | 0.970 |
| **L1-Logistic** | 0.994 | 0.971 |
| **SVM (RBF)** | 0.993 | 0.971 |
| **Random Forest** | 0.990 | 0.946 |

---

## 🛠️ Tech Stack & Libraries
* **Language:** R (`tidymodels` framework) & Python
* **Algorithms:** XGBoost, Random Forest, SVM, Regularized Logistic Regression
* **Data Source:** Wisconsin Diagnostic Breast Cancer (WDBC) Dataset via UCI Machine Learning Repository

---

## 📂 Project Structure
```text
├── data/               # Dataset files (WDBC)
├── scripts/            # R and Python scripts for preprocessing, training, and evaluation
├── outputs/            # Generated figures, confusion matrices, and ROC curves
└── README.md
