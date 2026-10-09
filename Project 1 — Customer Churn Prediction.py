import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("/Users/victor/Documents/Python/Telco-Customer-Churn.csv")

print(df.head())
print(df.shape)
print(df.info())
# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Check missing values again
print("\nMissing values after converting TotalCharges:")
print(df.isnull().sum())

# Remove rows with missing TotalCharges
df = df.dropna()

# Remove customer ID because it is not useful for prediction
df = df.drop("customerID", axis=1)

print("\nFinal dataset shape:")
print(df.shape)

print("\nFinal data types:")
print(df.dtypes)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df = df.dropna()
##Step 2: Check the target variable
print("\nChurn distribution:")
print(df["Churn"].value_counts())

print("\nChurn percentage:")
print(df["Churn"].value_counts(normalize=True) * 100)
###Step 3: Create our first EDA chart
plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Churn"
)

plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")

plt.show()
#Step 4: Analyze churn by contract
plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Contract",
    hue="Churn"
)

plt.title("Customer Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")

plt.show()
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, ConfusionMatrixDisplay

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
X = df.drop("Churn", axis=1)
y = df["Churn"]

num_cols = X.select_dtypes(include="number").columns
cat_cols = X.select_dtypes(exclude="number").columns

prep = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

for name, model in models.items():
    pipe = Pipeline([("prep", prep), ("model", model)])
    cv_auc = cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc")
    pipe.fit(X_train, y_train)
    test_auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
    print(f"{name}: CV AUC {cv_auc.mean():.3f}, Test AUC {test_auc:.3f}")
    print(classification_report(y_test, pipe.predict(X_test)))