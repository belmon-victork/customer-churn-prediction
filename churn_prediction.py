"""
Customer Churn Prediction (Telco Customer Churn dataset)

Steps: load -> clean -> EDA -> preprocess -> train/compare models -> evaluate -> explain.
Charts are saved to the images/ folder so they can be shown in the README.
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------
DATA_PATH = Path("data/Telco-Customer-Churn.csv")  # relative path: works on any computer
IMAGE_DIR = Path("images")
SHOW_PLOTS = True  # set to False to only save charts without opening windows
RANDOM_STATE = 42

IMAGE_DIR.mkdir(exist_ok=True)
if not SHOW_PLOTS:
    matplotlib.use("Agg")


def finish_plot(filename):
    """Save the current chart to images/ and optionally display it."""
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / filename, dpi=150)
    if SHOW_PLOTS:
        plt.show()
    plt.close()


# ----------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

print(df.head())
print("\nShape:", df.shape)
print(df.info())
print("\nMissing values:")
print(df.isnull().sum())

# ----------------------------------------------------------------------
# 2. Clean data
# ----------------------------------------------------------------------
# TotalCharges is stored as text because a few rows contain blank spaces.
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

missing_total = df["TotalCharges"].isnull().sum()
print(f"\nRows with missing TotalCharges: {missing_total}")

df = df.dropna(subset=["TotalCharges"])

# customerID is only an identifier, so it is not useful for prediction.
df = df.drop(columns="customerID")

print("\nFinal dataset shape:", df.shape)
print("\nFinal data types:")
print(df.dtypes)

# ----------------------------------------------------------------------
# 3. Target variable
# ----------------------------------------------------------------------
print("\nChurn distribution:")
print(df["Churn"].value_counts())
print("\nChurn percentage:")
print(df["Churn"].value_counts(normalize=True) * 100)

churned = (df["Churn"] == "Yes").astype(int)  # 1 = churned, 0 = stayed

# ----------------------------------------------------------------------
# 4. Exploratory data analysis (EDA)
# ----------------------------------------------------------------------
sns.set_theme(style="whitegrid")

# 4.1 Class balance
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Churn")
plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
finish_plot("01_churn_distribution.png")

# 4.2 Churn RATE (%) by category. Rates are fairer than raw counts.
for column, filename in [
    ("Contract", "02_churn_by_contract.png"),
    ("InternetService", "03_churn_by_internet_service.png"),
    ("PaymentMethod", "04_churn_by_payment_method.png"),
]:
    rate = (churned.groupby(df[column]).mean() * 100).sort_values(ascending=False)
    print(f"\nChurn rate (%) by {column}:")
    print(rate.round(1))

    plt.figure(figsize=(8, 4))
    sns.barplot(x=rate.index, y=rate.values)
    plt.title(f"Churn Rate by {column}")
    plt.xlabel(column)
    plt.ylabel("Churn Rate (%)")
    plt.xticks(rotation=15)
    finish_plot(filename)

# 4.3 Numeric features vs churn
for column, filename in [
    ("tenure", "05_tenure_vs_churn.png"),
    ("MonthlyCharges", "06_monthly_charges_vs_churn.png"),
]:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x="Churn", y=column)
    plt.title(f"{column} by Churn")
    finish_plot(filename)

# ----------------------------------------------------------------------
# 5. Preprocessing
# ----------------------------------------------------------------------
X = df.drop(columns="Churn")
y = churned

numeric_cols = X.select_dtypes(include="number").columns
categorical_cols = X.select_dtypes(exclude="number").columns

preprocessor = ColumnTransformer(
    [
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

# ----------------------------------------------------------------------
# 6. Train and compare models
# ----------------------------------------------------------------------
# class_weight="balanced" helps the models pay attention to the smaller
# "churn" class (about a quarter of customers).
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE
    ),
}

results = {}
fitted_pipelines = {}

for name, model in models.items():
    # The pipeline keeps preprocessing inside cross-validation (no data leakage).
    pipe = Pipeline([("prep", preprocessor), ("model", model)])

    cv_auc = cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc")
    pipe.fit(X_train, y_train)

    test_proba = pipe.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_proba)
    predictions = pipe.predict(X_test)

    results[name] = {"cv_auc": cv_auc.mean(), "test_auc": test_auc}
    fitted_pipelines[name] = pipe

    print("\n" + "=" * 60)
    print(f"{name}: CV ROC-AUC {cv_auc.mean():.3f} | Test ROC-AUC {test_auc:.3f}")
    print(classification_report(y_test, predictions, target_names=["Stayed", "Churned"]))

    ConfusionMatrixDisplay.from_predictions(
        y_test, predictions, display_labels=["Stayed", "Churned"], cmap="Blues"
    )
    plt.title(f"Confusion Matrix: {name}")
    finish_plot(f"confusion_matrix_{name.lower().replace(' ', '_')}.png")

# ----------------------------------------------------------------------
# 7. Summary table and best model
# ----------------------------------------------------------------------
summary = pd.DataFrame(results).T.round(3)
summary.columns = ["CV ROC-AUC", "Test ROC-AUC"]
print("\nModel comparison:")
print(summary)

# Pick the best model using cross-validation (not the test set).
best_name = summary["CV ROC-AUC"].idxmax()
print(f"\nBest model by CV ROC-AUC: {best_name}")

# ----------------------------------------------------------------------
# 8. Explain the models: which features matter most?
# ----------------------------------------------------------------------
def top_features(pipe, importance_attr, n=10):
    names = pipe.named_steps["prep"].get_feature_names_out()
    model = pipe.named_steps["model"]
    values = getattr(model, importance_attr)
    values = values.ravel()
    series = pd.Series(values, index=names)
    return series.reindex(series.abs().sort_values(ascending=False).index).head(n)

# Logistic regression coefficients: positive = pushes toward churn.
logreg_top = top_features(fitted_pipelines["Logistic Regression"], "coef_")
print("\nTop 10 Logistic Regression coefficients (positive = higher churn risk):")
print(logreg_top.round(3))

plt.figure(figsize=(8, 5))
sns.barplot(x=logreg_top.values, y=logreg_top.index)
plt.title("Top 10 Features: Logistic Regression Coefficients")
plt.xlabel("Coefficient (positive = higher churn risk)")
finish_plot("07_logreg_top_features.png")

# Random forest feature importances.
rf_top = top_features(fitted_pipelines["Random Forest"], "feature_importances_")
print("\nTop 10 Random Forest feature importances:")
print(rf_top.round(3))

plt.figure(figsize=(8, 5))
sns.barplot(x=rf_top.values, y=rf_top.index)
plt.title("Top 10 Features: Random Forest Importance")
plt.xlabel("Importance")
finish_plot("08_rf_top_features.png")

print("\nDone. Charts saved in the images/ folder.")
