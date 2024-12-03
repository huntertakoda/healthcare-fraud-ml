# Machine Learning Healthcare Fraud Detection

This project aims to detect potential healthcare fraud by analyzing patient reimbursement data using machine learning techniques. By leveraging a Random Forest model, the system identifies unusual patterns in inpatient and outpatient claims, flagging potential fraud cases for further investigation|

## Table of Contents

1. Features
2. Data Preprocessing
3. Models and Results
4. Folder Structure

## Features

Detection of fraudulent healthcare claims using reimbursement patterns.
Random Forest model fine-tuned with hyperparameter optimization.
Streamlit app for interactive fraud prediction.
Comprehensive data preprocessing pipeline

## Data Preprocessing

The raw dataset was cleaned and prepared using the following steps:
1. Removed irrelevant or missing data (e.g., `DOD` column).
2. Generated the `FraudFlag` feature based on the top 1% of reimbursement amounts.
3. Scaled numerical features for better model performance.
4. Addressed class imbalance with SMOTE (Synthetic Minority Oversampling Technique).

## Models and Results

- **Random Forest Classifier:**
  - Hyperparameter tuning resulted in a model with:
    - `n_estimators`: 300
    - `max_depth`: 10
    - `min_samples_split`: 5
    - `class_weight`: {0: 1, 1: 10}
  - **Final Results:**
    - Accuracy: 98.4%
    - Precision (Fraud): 63%
    - Recall (Fraud): 48%

- **Feature Importance:**
  - Key features contributing to fraud detection:
    1. Outpatient Annual Deductible Amount
    2. Inpatient Annual Deductible Amount
    3. Renal Disease Indicator
    4. Chronic Kidney Disease

## Folder Structure

- `gifs/`: Demonstrations of the Streamlit app.
- `outputs&images/`: Contains model outputs and visualizations.
- `scripts/`: Python scripts for data preprocessing, modeling, and app deployment.
