"""
train_models.py
================
End-to-end training pipeline for the Amazon Sales binary classification project.

Steps implemented (as per assignment guidelines):
  1. Preprocessing  : target creation, feature engineering, one-hot encoding,
                      leakage-safe train/test split, scaling (fit on train only)
  2. Model training : 5 classifiers instantiated in a dictionary and trained in a loop
  3. Evaluation     : Accuracy, AUC, Precision, Recall, F1, MCC on the test set
  4. Persistence    : models + scaler + column metadata saved to model/ via joblib,
                      unscaled encoded test split exported as test_data.csv

Run:  python train_models.py
"""

import os
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
TARGET_COL = "High_Value_Purchase"

# --------------------------------------------------------------------------
# Step 1a. Load raw data
# --------------------------------------------------------------------------
DATA_PATH = os.environ.get("AMAZON_CSV", "Amazon.csv")
df = pd.read_csv(DATA_PATH)
print(f"Loaded {DATA_PATH} -> {df.shape[0]:,} rows x {df.shape[1]} columns")

# --------------------------------------------------------------------------
# Step 1b. Target variable: High_Value_Purchase
#   1 if the order total exceeds the dataset median, else 0.
#   The median threshold yields a near-perfectly balanced binary target,
#   which keeps AUC and MCC well-behaved.
# --------------------------------------------------------------------------
THRESHOLD = df["TotalAmount"].median()
df[TARGET_COL] = (df["TotalAmount"] > THRESHOLD).astype(int)
print(f"Target threshold (median TotalAmount): {THRESHOLD:.2f}")
print(df[TARGET_COL].value_counts(normalize=True).rename("class balance"))

# --------------------------------------------------------------------------
# Step 1c. Feature engineering
#   - Date parts extracted from OrderDate (Year / Month / DayOfWeek)
#   - Identifier columns dropped (no predictive value, high cardinality)
#   - OrderStatus dropped (post-purchase outcome -> not known at order time)
#   - TotalAmount dropped from features (the target is derived from it ->
#     keeping it would be direct data leakage)
# --------------------------------------------------------------------------
df["OrderDate"] = pd.to_datetime(df["OrderDate"])
df["Order_Year"] = df["OrderDate"].dt.year
df["Order_Month"] = df["OrderDate"].dt.month
df["Order_DayOfWeek"] = df["OrderDate"].dt.dayofweek

DROP_COLS = [
    "OrderID", "OrderDate", "CustomerID", "CustomerName", "ProductID",
    "ProductName", "SellerID", "City", "State", "OrderStatus", "TotalAmount",
]
df = df.drop(columns=DROP_COLS)

NUMERIC_COLS = [
    "Quantity", "UnitPrice", "Discount", "Tax", "ShippingCost",
    "Order_Year", "Order_Month", "Order_DayOfWeek",
]
NOMINAL_COLS = ["Category", "Brand", "PaymentMethod", "Country"]

# One-Hot Encoding for nominal features. drop_first=True avoids the
# dummy-variable trap (multicollinearity hurts Logistic Regression).
df = pd.get_dummies(df, columns=NOMINAL_COLS, drop_first=True, dtype=int)

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]
FEATURE_COLS = X.columns.tolist()
print(f"Feature matrix: {X.shape[0]:,} rows x {X.shape[1]} features")

# --------------------------------------------------------------------------
# Step 1d. Split BEFORE scaling (prevents test-set leakage).
#   stratify=y keeps the class balance identical in both splits.
# --------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# Export the UNSCALED encoded test split + target for the Streamlit app.
# The app re-applies the saved scaler, mirroring the training pipeline.
test_df = X_test.copy()
test_df[TARGET_COL] = y_test.values
test_df.to_csv("test_data.csv", index=False)
print(f"test_data.csv written -> {test_df.shape[0]:,} rows")

# Scale numeric features: fit on TRAIN only, transform both splits.
scaler = StandardScaler()
X_train_s = X_train.copy()
X_test_s = X_test.copy()
X_train_s[NUMERIC_COLS] = scaler.fit_transform(X_train[NUMERIC_COLS])
X_test_s[NUMERIC_COLS] = scaler.transform(X_test[NUMERIC_COLS])

# --------------------------------------------------------------------------
# Step 2. The five required models, instantiated in a dictionary.
# --------------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),  # GaussianNB: features are continuous & scaled
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
}

os.makedirs("model", exist_ok=True)

# Persist the preprocessing artefacts alongside the models.
joblib.dump(scaler, "model/scaler.joblib")
joblib.dump(
    {"feature_columns": FEATURE_COLS, "numeric_columns": NUMERIC_COLS,
     "target_column": TARGET_COL, "threshold": float(THRESHOLD)},
    "model/columns.joblib",
)

# --------------------------------------------------------------------------
# Step 3. Train, save, evaluate (6 metrics) in a single loop.
#   AUC requires class probabilities, not hard predictions:
#   y_prob = model.predict_proba(X_test)[:, 1]
# --------------------------------------------------------------------------
results = []
for name, model in models.items():
    model.fit(X_train_s, y_train)

    fname = f"model/{name.replace(' ', '_')}.joblib"
    joblib.dump(model, fname, compress=3)

    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]

    results.append({
        "ML Model Name": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_test, y_prob), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1": round(f1_score(y_test, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 4),
    })
    print(f"Trained + saved: {fname}")

results_df = pd.DataFrame(results)
results_df.to_csv("model/metrics_summary.csv", index=False)
print("\n=== Test-set performance (20,000 held-out orders) ===")
print(results_df.to_string(index=False))
with open("model/metrics_summary.json", "w") as f:
    json.dump(results, f, indent=2)
