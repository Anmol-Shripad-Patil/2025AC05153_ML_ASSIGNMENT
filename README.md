# 📦 Amazon Sales — High-Value Order Classification

An end-to-end machine-learning project that predicts whether an Amazon order is a high-value purchase, evaluated across five classification algorithms and served through an interactive Streamlit dashboard.

Live app: https://2025ac05153-ml-assignment.streamlit.app/

---

## a. Problem Statement

Given an order's attributes at purchase time (quantity, unit price, discount, tax, shipping cost, product category, brand, payment method, country, and order-date features), predict the binary target:

> `High_Value_Purchase = 1` if the order's `TotalAmount` exceeds the dataset median (**714.32**), otherwise **0**.

Using the median as the threshold produces a balanced 50/50 target, keeping evaluation metrics mathematically well-behaved and directly comparable across models.

## b. Dataset Description

`Amazon.csv` — 100,000 orders × 20 columns. No missing values.

**Source:** [PASTE YOUR KAGGLE / UCI DATASET URL HERE]

| Property | Value |
| :--- | :--- |
| Instances | 100,000 (minimum required: 500) |
| Features after encoding | 31 (minimum required: 12) |
| Class balance | 50% / 50% |

The 20 raw columns are `OrderID`, `OrderDate`, `CustomerID`, `CustomerName`, `ProductID`, `ProductName`, `SellerID`, `Category`, `Brand`, `Quantity`, `UnitPrice`, `Discount`, `Tax`, `ShippingCost`, `TotalAmount`, `PaymentMethod`, `City`, `State`, `Country` and `OrderStatus`.

After preprocessing the model uses 8 numeric features (`Quantity`, `UnitPrice`, `Discount`, `Tax`, `ShippingCost`, `Order_Year`, `Order_Month`, `Order_DayOfWeek`) and 23 one-hot dummy columns, giving 31 features in total.

`Amazon.csv` is not committed to the repository, since the assignment asks for test data only. The repo ships `test_data.csv` (20,000 encoded rows) instead.

## c. GitHub Repository Link

**Repository URL:** [https://github.com/Anmol-Shripad-Patil/2025AC05153_ML_ASSIGNMENT](https://github.com/Anmol-Shripad-Patil/2025AC05153_ML_ASSIGNMENT)

## d. Models Used

All five classification models were trained on the same dataset and saved to the `model/` folder using `joblib`:

1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **K-Nearest Neighbors (k=5)**
4. **Gaussian Naive Bayes**
5. **Random Forest (100 trees, Ensemble)**

### Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.9252 | 0.9850 | 0.9248 | 0.9257 | 0.9252 | 0.8504 |
| Decision Tree | 0.9956 | 0.9956 | 0.9950 | 0.9961 | 0.9956 | 0.9911 |
| kNN | 0.9245 | 0.9765 | 0.9417 | 0.9050 | 0.9230 | 0.8496 |
| Naive Bayes | 0.8924 | 0.9811 | 0.9501 | 0.8283 | 0.8850 | 0.7913 |
| Random Forest (Ensemble) | 0.9914 | 0.9997 | 0.9879 | 0.9950 | 0.9914 | 0.9828 |

All metrics were computed on the 20,000-row held-out test set. AUC was calculated using `predict_proba`.

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| Logistic Regression | A strong, well-calibrated baseline (AUC 0.985) with balanced precision and recall. Order value is driven by the multiplicative interaction Quantity × UnitPrice × (1 − Discount), which a single linear boundary cannot fully express, capping accuracy at around 92.5%. |
| Decision Tree | Axis-aligned threshold splits capture the price–quantity interaction almost perfectly, giving the highest raw accuracy (0.9956) and MCC (0.9911). Because the tree is grown unpruned its leaves are pure, so predicted probabilities are only 0 or 1. This is why its AUC equals its balanced accuracy: it can label an order but cannot rank orders by confidence. |
| kNN | Competitive because features were standardized before distance calculations. Precision (0.9417) exceeds recall (0.9050), so it misses some borderline high-value orders that sit close to the median threshold. |
| Naive Bayes | Weakest accuracy and MCC because feature correlation (Tax is derived from UnitPrice and Quantity) violates the independence assumption. It has the highest precision (0.9501) but the lowest recall (0.8283), and its strong AUC (0.9811) shows ranking is still good even though the decision threshold is poor. |
| **Overall Winner for this dataset** | **Random Forest** — highest AUC (0.9997), MCC 0.9828, and recall 0.9950 on high-value orders. Unlike the single tree it produces graded probabilities, and averaging 100 trees reduces variance, making it the most reliable model for unseen data. |

**Note on the high accuracy scores:** `TotalAmount` was correctly dropped, but the target is still an arithmetic function of features that remain (Quantity, UnitPrice, Discount, Tax, ShippingCost). Tree-based models can therefore reconstruct the relationship almost exactly. The gaps between models reflect each algorithm's ability to represent a multiplicative relationship rather than differences in predictive skill on noisy data.

---

## Preprocessing Pipeline (`train_models.py`)

| Step | What was done | Why |
| :--- | :--- | :--- |
| Target creation | `High_Value_Purchase = (TotalAmount > median)` | Clean binary formulation |
| Leakage removal | Dropped `TotalAmount` from features | Target is derived from it |
| Dropped columns | IDs (`OrderID`, `CustomerID`, `CustomerName`, `ProductID`, `ProductName`, `SellerID`), `City`, `State`, `OrderStatus`, and `OrderDate` after its date parts were extracted | High-cardinality identifiers / post-purchase outcome |
| Feature engineering | `Order_Year`, `Order_Month`, `Order_DayOfWeek` extracted from `OrderDate` | Seasonality signal |
| One-Hot Encoding | `Category`, `Brand`, `PaymentMethod`, `Country` with `drop_first=True` | Avoids dummy-variable trap (multicollinearity) |
| Split before scale | `train_test_split(test_size=0.2, stratify=y, random_state=42)` | Prevents test-set leakage; preserves class balance |
| Scaling | `StandardScaler` fit on `X_train` only, then applied to the numeric columns of both splits (dummies left unscaled) | Required for distance/linear models; dummies are already 0/1 |
| Export | Unscaled encoded `X_test + y` → `test_data.csv`; scaler & column metadata → `model/` | Re-applies exact training-time transforms |

Final feature matrix: **31 features**, 80,000 training / 20,000 test rows.

## Streamlit App (`app.py`)

Live app: https://2025ac05153-ml-assignment.streamlit.app/

* **Sidebar:** `test_data.csv` uploader + model dropdown.
* **Main panel:** Dataset preview, all 6 evaluation metrics in a grid, a Seaborn confusion-matrix heatmap, and a full classification report.
* Uploaded data is reindexed to the saved training column order and its numeric columns are transformed with the saved training scaler, so the metrics shown match training-time evaluation exactly.

## Project Structure

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
    ├── columns.joblib         # Feature order, numeric columns, target name, threshold
    ├── metrics_summary.csv    # Saved metric evaluation summary
    └── metrics_summary.json   # Same summary in JSON form
```