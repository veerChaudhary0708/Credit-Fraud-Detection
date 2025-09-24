# Machine Learning for Credit Fraud Detection

## Project Objective
To build and optimize a machine learning model to detect fraudulent credit card/loan applications from a large, real-world dataset. The goal is to help businesses reduce financial losses by accurately identifying high-risk applicants in a timely manner.

## Dataset
The project utilizes an anonymized financial dataset containing over 300,000 samples and 139 features, including transaction history, credit scores, and account information. The target variable is highly imbalanced, which is typical for fraud detection problems.

## Key Steps & Workflow
1.  **Data Cleaning & Preprocessing:** Handled missing values (up to 90% in some columns) using various strategies like median imputation and the creation of indicator variables. Encoded all categorical features for model compatibility.
2.  **Feature Scaling:** Standardized over 100 numerical features using `StandardScaler` to prepare the data for modeling.
3.  **Handling Class Imbalance:** Addressed the imbalanced target variable using the `scale_pos_weight` parameter in XGBoost to ensure the model paid more attention to the minority (fraud) class.
4.  **Modeling:**
    * Established a baseline using **Logistic Regression**.
    * Developed a high-performance **XGBoost** model.
5.  **Hyperparameter Tuning:** Optimized the XGBoost model using `RandomizedSearchCV` with 3-fold cross-validation, leveraging a GPU (`CUDA`) for acceleration.
6.  **Evaluation & Interpretation:** Evaluated models using Precision, Recall, and F1-Score. Performed feature importance analysis to identify the key drivers of fraud.

## Results
The final tuned XGBoost model demonstrated a significant improvement over the baseline, showcasing a strong balance between catching fraud and minimizing false alarms.

| Model | Precision (Fraud) | Recall (Fraud) | F1-Score (Fraud) |
| :--- | :---: | :---: | :---: |
| Logistic Regression | 0.26 | 0.77 | 0.39 |
| **Tuned XGBoost** | **0.43** | **0.82** | **0.56** |

## Key Insights
The feature importance analysis revealed that the strongest predictors of fraud are:
* **Product Type:** Personal Loans carried the highest risk.
* **Specific Risk Grades:** A customer's `LAST_1_YR_RG2` status was highly predictive.
* **Credit Bureau Scores:** `CRIFF` scores and a history of delinquency (`TIMES_IRAC_SLIP`) were major factors.
* **Recent Activity:** Transaction behavior in the last 1-2 months was more influential than older history.

## Technologies Used
* **Python**
* **Pandas & NumPy** for data manipulation.
* **Scikit-learn** for preprocessing, modeling, and evaluation.
* **XGBoost** for the primary classification model.
* **Matplotlib & Seaborn** for data visualization.
* **Google Colab** with GPU acceleration.
