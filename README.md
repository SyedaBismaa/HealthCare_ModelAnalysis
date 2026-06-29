# 🏥 Healthcare Test Result Prediction using Machine Learning

## 📌 Overview

This project presents an end-to-end Machine Learning pipeline for predicting patient **Test Results** using demographic, admission, medical condition, insurance, medication, and billing information. The project covers data preprocessing, exploratory data analysis (EDA), feature engineering, multicollinearity analysis (VIF), and multiclass classification using Logistic Regression and Decision Tree models.

---

# 🎯 Objective

Predict the patient's **Test Result** into one of the following categories:

* Abnormal
* Normal
* Inconclusive

This is a **multiclass classification** problem.

---

# 📂 Dataset

* **Records:** 55,500
* **Original Features:** 15
* **Target Variable:** `Test Results`

The dataset contains patient demographics, admission details, billing information, insurance details, medications, and medical conditions.

---

# 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Statsmodels

---

# 📊 Workflow

```text
Data Loading
      ↓
Data Cleaning
      ↓
Exploratory Data Analysis (EDA)
      ↓
Feature Engineering
      ↓
Variance Inflation Factor (VIF)
      ↓
Train-Test Split
      ↓
Logistic Regression
      ↓
Decision Tree
      ↓
GridSearchCV
      ↓
Model Evaluation
```

---

# ⚙️ Feature Engineering

New features created:

* Admission Year
* Admission Month
* Admission Day
* Discharge Year
* Discharge Month
* Discharge Day
* **Length of Stay** (Discharge Date − Admission Date)

Highly collinear features were removed after VIF analysis.

---

# 📈 Exploratory Data Analysis

Performed:

* Age Distribution
* Billing Amount Analysis
* Medical Condition Distribution
* Admission Type Analysis

---

# 📉 Feature Selection

Performed:

* Variance Inflation Factor (VIF)

**Note:** Weight of Evidence (WoE) and Information Value (IV) were **not applied** because the target variable contains three classes. These techniques are designed for binary classification problems.

---

# 🤖 Machine Learning Models

### Logistic Regression

* Multiclass Classification
* Evaluated using Accuracy, Precision, Recall, F1-Score and Confusion Matrix.

### Decision Tree

* Hyperparameter Tuning using GridSearchCV
* Decision Tree Visualization
* Multiclass Classification

---

# 📊 Model Comparison

| Model               | Accuracy  |
| ------------------- | --------- |
| Logistic Regression | **32.8%** |
| Decision Tree       | **34.1%** |

The **Decision Tree** model was selected as the final model due to its slightly better overall performance.

---

# 📌 Limitations

The relatively low accuracy is primarily due to the limited predictive information available in the dataset. The provided features mainly include demographic, administrative, and billing information, whereas medical test results are typically influenced by additional clinical factors such as laboratory values, vital signs, imaging reports, and patient history, which are not available in this dataset.

---

# 📚 Key Concepts

* Data Cleaning
* Exploratory Data Analysis (EDA)
* Feature Engineering
* Variance Inflation Factor (VIF)
* Logistic Regression
* Decision Tree
* GridSearchCV
* Multiclass Classification
* Model Evaluation

---

# ⭐ Conclusion

This project demonstrates a complete end-to-end machine learning workflow for multiclass healthcare prediction. It includes data preprocessing, feature engineering, feature selection using VIF, model training, hyperparameter tuning, and performance evaluation. While the predictive accuracy is limited by the available features, the project provides a practical implementation of a multiclass classification pipeline and highlights the importance of feature quality in healthcare machine learning applications.
