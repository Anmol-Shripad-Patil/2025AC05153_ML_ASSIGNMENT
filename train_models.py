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

AMAZON_RANDOM_STATE = 42
AMAZON_TARGET_COL = "High_Value_Purchase"

AMAZON_DATA_PATH = os.environ.get("AMAZON_CSV", "Amazon.csv")
amazon_df = pd.read_csv(AMAZON_DATA_PATH)
print(f"Loaded {AMAZON_DATA_PATH} -> {amazon_df.shape[0]:,} rows x {amazon_df.shape[1]} columns")

AMAZON_THRESHOLD = amazon_df["TotalAmount"].median()
amazon_df[AMAZON_TARGET_COL] = (amazon_df["TotalAmount"] > AMAZON_THRESHOLD).astype(int)
print(f"Target threshold (median TotalAmount): {AMAZON_THRESHOLD:.2f}")
print(amazon_df[AMAZON_TARGET_COL].value_counts(normalize=True).rename("class balance"))

amazon_df["OrderDate"] = pd.to_datetime(amazon_df["OrderDate"])
amazon_df["Order_Year"] = amazon_df["OrderDate"].dt.year
amazon_df["Order_Month"] = amazon_df["OrderDate"].dt.month
amazon_df["Order_DayOfWeek"] = amazon_df["OrderDate"].dt.dayofweek

AMAZON_DROP_COLS = [
    "OrderID", "OrderDate", "CustomerID", "CustomerName", "ProductID",
    "ProductName", "SellerID", "City", "State", "OrderStatus", "TotalAmount",
]
amazon_df = amazon_df.drop(columns=AMAZON_DROP_COLS)

AMAZON_NUMERIC_COLS = [
    "Quantity", "UnitPrice", "Discount", "Tax", "ShippingCost",
    "Order_Year", "Order_Month", "Order_DayOfWeek",
]
AMAZON_NOMINAL_COLS = ["Category", "Brand", "PaymentMethod", "Country"]

amazon_df = pd.get_dummies(amazon_df, columns=AMAZON_NOMINAL_COLS, drop_first=True, dtype=int)

amazon_X = amazon_df.drop(columns=[AMAZON_TARGET_COL])
amazon_y = amazon_df[AMAZON_TARGET_COL]
AMAZON_FEATURE_COLS = amazon_X.columns.tolist()
print(f"Feature matrix: {amazon_X.shape[0]:,} rows x {amazon_X.shape[1]} features")

amazon_X_train, amazon_X_test, amazon_y_train, amazon_y_test = train_test_split(
    amazon_X, amazon_y, test_size=0.2, stratify=amazon_y, random_state=AMAZON_RANDOM_STATE
)

amazon_test_df = amazon_X_test.copy()
amazon_test_df[AMAZON_TARGET_COL] = amazon_y_test.values
amazon_test_df.to_csv("test_data.csv", index=False)
print(f"test_data.csv written -> {amazon_test_df.shape[0]:,} rows")

amazon_scaler = StandardScaler()
amazon_X_train_s = amazon_X_train.copy()
amazon_X_test_s = amazon_X_test.copy()
amazon_X_train_s[AMAZON_NUMERIC_COLS] = amazon_scaler.fit_transform(amazon_X_train[AMAZON_NUMERIC_COLS])
amazon_X_test_s[AMAZON_NUMERIC_COLS] = amazon_scaler.transform(amazon_X_test[AMAZON_NUMERIC_COLS])

amazon_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=AMAZON_RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(random_state=AMAZON_RANDOM_STATE),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=AMAZON_RANDOM_STATE, n_jobs=-1),
}

os.makedirs("model", exist_ok=True)

joblib.dump(amazon_scaler, "model/scaler.joblib")
joblib.dump(
    {"feature_columns": AMAZON_FEATURE_COLS, "numeric_columns": AMAZON_NUMERIC_COLS,
     "target_column": AMAZON_TARGET_COL, "threshold": float(AMAZON_THRESHOLD)},
    "model/columns.joblib",
)

amazon_results = []
for amazon_model_name, amazon_model in amazon_models.items():
    amazon_model.fit(amazon_X_train_s, amazon_y_train)

    amazon_fname = f"model/{amazon_model_name.replace(' ', '_')}.joblib"
    joblib.dump(amazon_model, amazon_fname, compress=3)

    amazon_y_pred = amazon_model.predict(amazon_X_test_s)
    amazon_y_prob = amazon_model.predict_proba(amazon_X_test_s)[:, 1]

    amazon_results.append({
        "ML Model Name": amazon_model_name,
        "Accuracy": round(accuracy_score(amazon_y_test, amazon_y_pred), 4),
        "AUC": round(roc_auc_score(amazon_y_test, amazon_y_prob), 4),
        "Precision": round(precision_score(amazon_y_test, amazon_y_pred), 4),
        "Recall": round(recall_score(amazon_y_test, amazon_y_pred), 4),
        "F1": round(f1_score(amazon_y_test, amazon_y_pred), 4),
        "MCC": round(matthews_corrcoef(amazon_y_test, amazon_y_pred), 4),
    })
    print(f"Trained + saved: {amazon_fname}")

amazon_results_df = pd.DataFrame(amazon_results)
amazon_results_df.to_csv("model/metrics_summary.csv", index=False)
print("\n=== Test-set performance (20,000 held-out orders) ===")
print(amazon_results_df.to_string(index=False))
with open("model/metrics_summary.json", "w") as f:
    json.dump(amazon_results, f, indent=2)
