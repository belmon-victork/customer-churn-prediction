# Customer Churn Prediction

Predicts whether a telecom customer will leave (churn) using customer account, service, and billing data. The project covers data cleaning, exploratory analysis, and a comparison of two classification models, so that a company could identify at-risk customers early and target retention offers.

## Dataset
- **Telco Customer Churn** dataset, available on [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- 7,043 customers and 21 columns originally; **7,032 customers** after cleaning
- Features include contract type, tenure, monthly charges, internet service, and payment method
- Target: `Churn` (Yes/No). **1,869 customers (26.6%) churned**, so the classes are imbalanced

**Note:** The dataset is not included in this repository (data files © original authors). Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from Kaggle and place it in a `data/` folder before running the script.

## Tools Used
- Python
- pandas, NumPy
- matplotlib, seaborn
- scikit-learn (Pipeline, ColumnTransformer, Logistic Regression, Random Forest)

## Approach
1. **Cleaning:** converted `TotalCharges` to numeric, removed 11 rows with missing values, and dropped `customerID`.
2. **EDA:** checked class balance and compared churn rates across contract type, internet service, payment method, tenure, and monthly charges.
3. **Preprocessing:** scaled numeric features and one-hot encoded categorical features inside a scikit-learn `Pipeline` to avoid data leakage.
4. **Modeling:** trained Logistic Regression and Random Forest (with balanced class weights) on an 80/20 stratified train/test split (5,625 train, 1,407 test).
5. **Evaluation:** 5-fold cross-validation, ROC-AUC, classification reports, and confusion matrices.
6. **Explainability:** reviewed logistic regression coefficients and random forest feature importances.

## Results

| Model | CV ROC-AUC | Test ROC-AUC | Churn recall | Churn precision |
|---|---|---|---|---|
| Logistic Regression | 0.846 | 0.835 | 0.80 | 0.49 |
| Random Forest | 0.821 | 0.812 | 0.47 | 0.62 |

**Best model by cross-validation: Logistic Regression.** It catches 80% of the customers who actually churn on the held-out test set. The trade-off is precision: about half of the customers it flags would have stayed anyway (overall accuracy 73%, versus 78% for Random Forest). For a retention campaign where missing a churner costs more than a wasted offer, the higher recall is the better fit.

## Key Findings
- **Contract type matters most:** month-to-month customers churned at **42.7%**, versus 11.3% on one-year and **2.8%** on two-year contracts.
- **Internet service and payment method:** fiber-optic customers churned at 41.9% (DSL 19.0%, no internet 7.4%), and electronic-check payers churned at 45.3% (other payment methods 15-19%).
- **Tenure is the strongest signal in the model:** it has the largest logistic regression coefficient (negative), so newer customers are at the highest risk. Tenure, total charges, and monthly charges also lead the Random Forest importances. Charges and tenure are correlated, so individual coefficients for those features should be read with care.
- **Business idea:** target new month-to-month, fiber-optic customers with incentives to move to longer contracts or automatic payment.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/belmon-victor/customer-churn-prediction.git
   cd customer-churn-prediction
   ```
2. Install the dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. Download the dataset from Kaggle and place it at `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`.
4. Run the script:
   ```bash
   python churn_prediction.py
   ```
   Charts are saved to the `images/` folder.

## Future Improvements
- Try SMOTE for class imbalance
- Hyperparameter tuning and threshold tuning to balance precision and recall
- Try gradient boosting models such as XGBoost

## Author
**K. Belmon Victor** | [LinkedIn](https://linkedin.com/in/belmon-victor-k-81618b38a) | [GitHub](https://github.com/belmon-victor)
