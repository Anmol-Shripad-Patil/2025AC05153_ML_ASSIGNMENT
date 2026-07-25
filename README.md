# 📦 Amazon Sales — High-Value Order Classification

An end-to-end machine-learning project that predicts whether an Amazon order is a high-value purchase, evaluated across five classification algorithms and served through an interactive Streamlit dashboard.

---

## 1. Problem Statement

Given an order's attributes at purchase time (quantity, unit price, discount, tax, shipping cost, product category, brand, payment method, country, and order-date features), predict the binary target:

> `High_Value_Purchase = 1` if the order's `TotalAmount` exceeds the dataset median (**714.32**), otherwise **0**.

Using the median as the threshold produces a balanced 50/50 target, keeping evaluation metrics mathematically well-behaved and directly comparable across models.

## 2. Dataset Description

`Amazon.csv` — 100,000 orders × 20 columns (order, customer, product, pricing, logistics, and status fields). No missing values. Minimum Feature Size (12) and Instance Size (500) requirements are met.

## 3. GitHub Repository Link

**Repository URL:** [https://github.com/Anmol-Shripad-Patil/Assignment_ML](https://github.com/Anmol-Shripad-Patil/Assignment_ML)

## 4. Preprocessing Pipeline (`train_models.py`)

| Step | What was done | Why |
| :--- | :--- | :--- |
| Target creation | `High_Value_Purchase = (TotalAmount > median)` | Clean binary formulation |
| Leakage removal | Dropped `TotalAmount` from features | Target is derived from it |
| Dropped columns | IDs (`OrderID`, `CustomerID`, `ProductID`, `SellerID`, names), `City`, `State`, `OrderStatus` | High-cardinality identifiers / post-purchase outcome |
| Feature engineering | `Order_Year`, `Order_Month`, `Order_DayOfWeek` extracted from `OrderDate` | Seasonality signal |
| One-Hot Encoding | `Category`, `Brand`, `PaymentMethod`, `Country` with `drop_first=True` | Avoids dummy-variable trap (multicollinearity) |
| Split before scale | `train_test_split(test_size=0.2, stratify=y, random_state=42)` | Prevents test-set leakage; preserves class balance |
| Scaling | `StandardScaler` fit on `X_train` only, applied to both splits | Required for distance/linear models |
| Export | Unscaled encoded `X_test + y` → `test_data.csv`; scaler & column metadata → `model/` | Re-applies exact training-time transforms |

Final feature matrix: **31 features**, 80,000 training / 20,000 test rows.

## 5. Models Implemented

All five classification models were trained and saved to the `model/` folder using `joblib`:

1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **K-Nearest Neighbors (k=5)**
4. **Gaussian Naive Bayes**
5. **Random Forest (100 trees, Ensemble)**

## 6. Results

### Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.9252 | 0.9850 | 0.9248 | 0.9257 | 0.9252 | 0.8504 |
| Decision Tree | 0.9956 | 0.9956 | 0.9950 | 0.9961 | 0.9956 | 0.9911 |
| KNN | 0.9245 | 0.9765 | 0.9417 | 0.9050 | 0.9230 | 0.8496 |
| Naive Bayes | 0.8924 | 0.9811 | 0.9501 | 0.8283 | 0.8850 | 0.7913 |
| Random Forest (Ensemble) | 0.9914 | 0.9997 | 0.9879 | 0.9950 | 0.9914 | 0.9828 |

### Performance Observations

| ML Model Name | Observation about model performance |
| :--- | :--- |
| Logistic Regression | A strong, well-calibrated baseline (AUC 0.985). However, order value is driven by the multiplicative interaction Quantity × UnitPrice × (1 − Discount), which a single linear boundary cannot fully express, capping accuracy at ~92.5%. |
| Decision Tree | Axis-aligned threshold splits capture the price–quantity interaction almost perfectly, giving high raw accuracy (0.9956). As a single deep tree it is variance-prone, yielding a lower AUC than the forest. |
| KNN | Competitive because features were standardized prior to distance calculations. Precision (0.94) exceeds recall (0.91), missing some borderline high-value orders. |
| Naive Bayes | Weakest accuracy and MCC due to feature correlation (Tax, UnitPrice, Quantity) violating independence assumptions, though order ranking remains strong (AUC 0.981). |
| **Overall Winner** | **Random Forest** — near-perfect AUC (0.9997), MCC 0.9828, and recall 0.995 on high-value orders. The 100-tree ensemble mitigates single-tree overfitting risk, making it the most robust model for unseen data. |

## 7. Streamlit App (`app.py`)

* **Sidebar:** `test_data.csv` uploader + model dropdown.
* **Main panel:** Dataset preview, all 6 evaluation metrics in a grid, a Seaborn confusion-matrix heatmap, and a full classification report.
* Uploaded data is re-scaled using the saved training scaler to mirror training-time performance.

## 8. Project Structure

```text
├── app.py                 # Streamlit dashboard
├── train_models.py        # Preprocessing, training, and evaluation script
├── test_data.csv          # Held-out test set (20,000 rows)
├── requirements.txt       # Project dependencies
├── README.md              # Documentation
└── model/
    ├── Logistic_Regression.joblib
    ├── Decision_Tree.joblib
    ├── KNN.joblib
    ├── Naive_Bayes.joblib
    ├── Random_Forest.joblib
    ├── scaler.joblib          # Fitted StandardScaler
    ├── columns.joblib         # Feature column names
    └── metrics_summary.csv    # Saved metric evaluation summary