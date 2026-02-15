# =========================================
# Adult Income Classification - Model Training
# =========================================

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")

# =========================================
# 1️⃣ Load Dataset (Place adult.csv in project folder)
# =========================================

data = pd.read_csv("adult.csv")

print("Dataset loaded successfully!")
print("Shape:", data.shape)

# =========================================
# 2️⃣ Data Preprocessing
# =========================================

df = data.copy()

# Replace missing values
df.replace(["?", " ?"], np.nan, inplace=True)
df.dropna(inplace=True)

# Strip whitespace
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()

# Encode target
df["income"] = df["income"].map({"<=50K": 0, ">50K": 1})

# Split features & target
X = df.drop("income", axis=1)
y = df["income"]

# One-hot encoding
X = pd.get_dummies(X, drop_first=True)

# =========================================
# 3️⃣ Train-Test Split
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=123,   # changed for uniqueness
    stratify=y
)

# =========================================
# 4️⃣ Save small test CSV for Streamlit
# =========================================

test_data = X_test.copy()
test_data["income"] = y_test.values
test_sample = test_data.sample(n=100, random_state=123)

if not os.path.exists("model"):
    os.makedirs("model")

test_sample.to_csv("model/test_data.csv", index=False)

# =========================================
# 5️⃣ Feature Scaling
# =========================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "model/scaler.pkl")

# =========================================
# 6️⃣ Define Models (slightly modified hyperparameters)
# =========================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1500),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=123),
    "KNN": KNeighborsClassifier(n_neighbors=9),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=120, max_depth=12, random_state=123),
    "XGBoost": XGBClassifier(
        n_estimators=180,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="logloss",
        random_state=123
    )
}

# =========================================
# 7️⃣ Training & Evaluation
# =========================================

results = []

for name, model in models.items():

    print(f"\nTraining {name}...")

    if name in ["Logistic Regression", "KNN", "Naive Bayes"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    results.append([
        name,
        round(accuracy, 4),
        round(auc, 4),
        round(precision, 4),
        round(recall, 4),
        round(f1, 4),
        round(mcc, 4)
    ])

    joblib.dump(model, f"model/{name.replace(' ', '_').lower()}.pkl")

# =========================================
# 8️⃣ Save Results
# =========================================

results_df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
)

print("\nModel Comparison Results:")
print(results_df)

results_df.to_csv("model/model_results.csv", index=False)
