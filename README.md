# Customer Churn Prediction

Predicts whether a telecom customer will leave (churn) using customer account, service, and billing data. The project covers data cleaning, exploratory analysis, and a comparison of two classification models, so that a company could identify at-risk customers early and target retention offers.

## Dataset
- **Telco Customer Churn** dataset (available on Kaggle)
- 7,043 customers and 21 columns originally; **7,032 customers** after cleaning
- Features include contract type, tenure, monthly charges, internet service, and payment method
- Target: `Churn` (Yes/No). **1,869 customers (26.6%) churned**, so the classes are imbalanced

## Tools Used
- Python
- pandas, NumPy
- matplotlib, seaborn
- scikit-learn (Pipeline, ColumnTransformer, Logistic Regression, Random Forest)

## Approach
1. **Cleaning:** converted `TotalCharges` to numeric, removed 11 rows with missing values, and dropped `customerID`.
2. **EDA:** checked class balance and compared churn rates across contract type, internet service, payment method, tenure, and monthly charges.
3. **Preprocessing:** scaled numeric features and one-hot encoded categorical features inside a scikit-learn `Pipeline` to avoid data leakage.
4. **Modeling:** trained Logistic Regression and Random Forest (with balanced class weights) on an 80/20 stratified train/test split.
5. **Evaluation:** 5-fold cross-validation, ROC-AUC, classification reports, and confusion matrices.
6. **Explainability:** reviewed logistic regression coefficients and random forest feature importances.

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
   python churn_prediction.py
```
   Charts are saved to the `images/` folder.

## Future Improvements
- Try SMOTE for class imbalance
- Hyperparameter tuning
- Try gradient boosting models such as XGBoost

## Author
**K. Belmon Victor** | [LinkedIn](https://linkedin.com/in/belmon-victor-k-81618b38a)
