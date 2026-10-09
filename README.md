# Customer Churn Prediction

Predicts whether a telecom customer will leave (churn) using customer account, service, and billing data. The project covers data cleaning, exploratory analysis, and a comparison of two classification models, so that a company could identify at-risk customers early and target retention offers.

## Dataset
- **Telco Customer Churn** dataset (available on Kaggle)
- About **[7,032]** customers after cleaning, with features such as contract type, tenure, monthly charges, internet service, and payment method
- Target: `Churn` (Yes/No), where roughly **[26]%** of customers churned

## Tools Used
- Python
- pandas, NumPy
- matplotlib, seaborn
- scikit-learn (Pipeline, ColumnTransformer, Logistic Regression, Random Forest)

## Approach
1. **Cleaning:** converted `TotalCharges` to numeric, removed **[11]** rows with missing values, and dropped `customerID`.
2. **EDA:** checked class balance and compared churn across contract types.
3. **Preprocessing:** scaled numeric features and one-hot encoded categorical features inside a scikit-learn `Pipeline` to avoid data leakage.
4. **Modeling:** trained Logistic Regression and Random Forest on an 80/20 stratified train/test split.
5. **Evaluation:** 5-fold cross-validation and ROC-AUC on the held-out test set.

## Results

| Model | CV ROC-AUC | Test ROC-AUC |
|---|---|---|
| Logistic Regression | [0.xxx] | [0.xxx] |
| Random Forest | [0.xxx] | [0.xxx] |

Best model: **[model name]**, with a recall of **[xx]%** on the churn class.

## Key Findings
- **[Finding 1]:** e.g., customers on month-to-month contracts churned at a much higher rate than those on one- or two-year contracts (check your contract chart for the exact percentages).
- **[Finding 2]:** e.g., how tenure relates to churn (look at your feature importances or coefficients).
- **[Finding 3]:** e.g., a business recommendation, such as offering longer-term contract incentives to new month-to-month customers.

## How to Run
1. Clone the repository:
```bash
   git clone https://github.com/YOUR-USERNAME/customer-churn-prediction.git
   cd customer-churn-prediction
```
2. Install the dependencies:
```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
```
3. Make sure the dataset is at `data/Telco-Customer-Churn.csv`.
4. Run the script:
```bash
   python "Project 1 — Customer Churn Prediction.py"
```

## Future Improvements
- Handle class imbalance (class weights or SMOTE)
- Hyperparameter tuning
- Try gradient boosting models such as XGBoost

## Author
**K. Belmon Victor** | [LinkedIn](https://linkedin.com/in/belmon-victor-k-81618b38a)
