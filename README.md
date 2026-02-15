# Machine Learning Assignment 2  
## Adult Census Income Prediction – Classification Models Comparison  

**Program:** M.Tech (AIML/DSE) – BITS Pilani  
**Submission Date:** February 15, 2026  

---

## 📋 Problem Statement

The objective of this project is to predict whether an individual earns more than $50,000 annually based on demographic and employment-related attributes.

This is a binary classification problem where six different machine learning models are implemented and compared using multiple evaluation metrics.

Target classes:
- `<=50K` → 0  
- `>50K` → 1  

---

## 📊 Dataset Description

**Dataset:** Adult Census Income Dataset  
**Source:** UCI Machine Learning Repository  
**Total Instances:** 32,561  
**Total Features:** 14 (excluding target)  
**Target Variable:** `income`  

### Class Distribution
- Class 0 (<=50K): ~75%
- Class 1 (>50K): ~25%

The dataset is moderately imbalanced.

---

## 🔧 Data Preprocessing Steps

1. Replaced missing values represented as `?`
2. Removed rows containing missing values
3. Stripped extra whitespace from categorical columns
4. Encoded target variable (`<=50K` → 0, `>50K` → 1)
5. Applied one-hot encoding to categorical variables
6. Applied StandardScaler for scale-sensitive models
7. Train-Test Split: 80–20 with stratification  
   - **random_state = 123**

---

## 🤖 Models Implemented

The following six classification models were trained and evaluated:

1. Logistic Regression  
2. Decision Tree  
3. K-Nearest Neighbors (KNN)  
4. Gaussian Naive Bayes  
5. Random Forest  
6. XGBoost  

---

## 📈 Model Performance Comparison

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|-----------|----------|------|-----------|--------|------|------|
| Logistic Regression | 0.8548 | 0.9103 | 0.7600 | 0.6092 | 0.6763 | 0.5901 |
| Decision Tree | 0.8490 | 0.8957 | 0.7171 | 0.6498 | 0.6818 | 0.5843 |
| KNN | 0.8261 | 0.8597 | 0.6849 | 0.5586 | 0.6153 | 0.5089 |
| Naive Bayes | 0.4273 | 0.6775 | 0.2985 | 0.9627 | 0.4556 | 0.2311 |
| Random Forest | 0.8596 | 0.9159 | 0.8146 | 0.5646 | 0.6669 | 0.5973 |
| **XGBoost** | **0.8752** | **0.9295** | **0.7979** | **0.6678** | **0.7271** | **0.6513** |

---

## 🏆 Best Performing Model

**XGBoost** achieved the highest performance across all evaluation metrics:

- Accuracy: **87.52%**
- AUC: **0.9295**
- Precision: **0.7979**
- Recall: **0.6678**
- F1 Score: **0.7271**
- MCC: **0.6513**

XGBoost effectively captures complex feature interactions and provides balanced performance across precision and recall.

---

## 📌 Model Observations

### Logistic Regression
Performed well with strong accuracy and stable probability estimates. Serves as a reliable baseline model.

### Decision Tree
Captured non-linear relationships effectively. Performance is comparable to Logistic Regression.

### KNN
Moderate performance. Sensitive to feature scaling and dimensionality.

### Naive Bayes
Very high recall but low precision, resulting in many false positives. Independence assumption does not hold strongly for this dataset.

### Random Forest
Provided improved stability over Decision Tree with strong balanced metrics.

### XGBoost
Best performing model with highest accuracy and best overall metric balance. Selected as the final recommended model.

---

## ⚙️ Hyperparameters Used

| Model | Key Hyperparameters |
|-------|-------------------|
| Logistic Regression | max_iter=1500 |
| Decision Tree | max_depth=10, random_state=123 |
| KNN | n_neighbors=9 |
| Naive Bayes | Default |
| Random Forest | n_estimators=120, max_depth=12, random_state=123 |
| XGBoost | n_estimators=180, max_depth=5, learning_rate=0.08, random_state=123 |

---

## 🌐 Streamlit Application

The Streamlit web application allows:

- Uploading a CSV file for prediction
- Selecting any of the six trained models
- Displaying:
  - Accuracy
  - AUC
  - Precision
  - Recall
  - F1 Score
  - MCC
- Confusion Matrix visualization

---

## 📁 Repository Structure

```
Adult-Census-Income-Prediction/
│
├── app.py                              # Streamlit web application
├── requirements.txt                    # Python dependencies
├── README.md                           # This comprehensive documentation
├── test_data.csv                       # Test set (100 samples for demo)
│
└── model/                              # Saved trained models
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    └── xgboost.pkl
    ├── scaler.pkl                      # Fitted StandardScaler
    ├── train_models.py                 # Model training script  
    ├── model_results.csv               # Performance comparison table
    
	
---

## 🎯 Final Conclusion

Among all six models, **XGBoost provides the best performance** for Adult Income classification with 87.52% accuracy and the highest overall evaluation scores.

Therefore, XGBoost is recommended for deployment.

---

Machine Learning Assignment 2  
M.Tech (AIML/DSE) – BITS Pilani
