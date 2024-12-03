import pandas as pd

# loading 

file_path = r'Cleaned_Beneficiarydata.csv'
data = pd.read_csv(file_path)

# corr. analysis

target_cols = ['IPAnnualReimbursementAmt', 'OPAnnualReimbursementAmt']
correlations = data.corr()[target_cols]

sorted_correlations = correlations.abs().sort_values(by='IPAnnualReimbursementAmt', ascending=False)

print("Top Correlated Features with Inpatient Reimbursement Amount:")
print(sorted_correlations['IPAnnualReimbursementAmt'].head(10))

print("\nTop Correlated Features with Outpatient Reimbursement Amount:")
print(sorted_correlations['OPAnnualReimbursementAmt'].head(10))

selected_features = [
    'IPAnnualDeductibleAmt', 'RenalDiseaseIndicator', 'ChronicCond_KidneyDisease',
    'ChronicCond_ObstrPulmonary', 'ChronicCond_Heartfailure', 'ChronicCond_IschemicHeart',
    'ChronicCond_Diabetes', 'ChronicCond_stroke', 'Age'
]

selected_columns = selected_features + ['IPAnnualReimbursementAmt', 'OPAnnualReimbursementAmt']

data_subset = data[selected_columns]

print(data_subset.info())
print(data_subset.head())

ip_threshold = data_subset['IPAnnualReimbursementAmt'].quantile(0.99)
op_threshold = data_subset['OPAnnualReimbursementAmt'].quantile(0.99)

data_subset['FraudFlag'] = (
    (data_subset['IPAnnualReimbursementAmt'] > ip_threshold) |
    (data_subset['OPAnnualReimbursementAmt'] > op_threshold)
).astype(int)

print(data_subset['FraudFlag'].value_counts())
print(data_subset.head())

from sklearn.model_selection import train_test_split

X = data_subset.drop(columns=['FraudFlag', 'IPAnnualReimbursementAmt', 'OPAnnualReimbursementAmt'])
y = data_subset['FraudFlag']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print("Training set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)

# logistics regression

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
print("\nClassification Report:\n", classification_report(y_test, y_pred_lr))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred_lr))

# precision assurance

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

print("Class distribution after SMOTE:")
print(y_resampled.value_counts())

from sklearn.ensemble import RandomForestClassifier

rf_model_smote = RandomForestClassifier(random_state=42, class_weight='balanced')
rf_model_smote.fit(X_resampled, y_resampled)

y_pred_rf_smote = rf_model_smote.predict(X_test)

print("\nRandom Forest with SMOTE Performance")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf_smote))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf_smote))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred_rf_smote))

# random forest

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')  # Handles imbalance
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

print("\nRandom Forest Performance")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred_rf))

# adjusting the decision threshold

import numpy as np
from sklearn.metrics import precision_recall_curve

y_prob_rf = rf_model_smote.predict_proba(X_test)[:, 1]

threshold = 0.3  
y_pred_threshold = (y_prob_rf >= threshold).astype(int)

print(f"\nRandom Forest with Threshold {threshold}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_threshold))
print("\nClassification Report:\n", classification_report(y_test, y_pred_threshold))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred_threshold))

precision, recall, thresholds = precision_recall_curve(y_test, y_prob_rf)
print("\nPrecision-Recall Curve Analysis:")
print("Precision:", precision)
print("Recall:", recall)

# xg boost

from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)

print("\nXGBoost Performance")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))
print("\nClassification Report:\n", classification_report(y_test, y_pred_xgb))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred_xgb))

