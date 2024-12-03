import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# load cleaned dataset

file_path = r'Cleaned_Beneficiarydata.csv'
data = pd.read_csv(file_path)

# split features + dataset

X = data.drop(columns=['FraudFlag', 'IPAnnualReimbursementAmt', 'OPAnnualReimbursementAmt'])
y = data['FraudFlag']

# train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# setting up hyperparameters for grid search

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'class_weight': ['balanced', {0: 1, 1: 10}]
}

# random forest definition 

rf = RandomForestClassifier(random_state=42)

# grid search

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring='f1',       
    cv=3,              
    verbose=2,
    n_jobs=-1          
)

grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)

best_rf = grid_search.best_estimator_

y_pred_best_rf = best_rf.predict(X_test)

# model evaluation

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_best_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_best_rf))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred_best_rf))

import matplotlib.pyplot as plt
import pandas as pd

feature_importances = best_rf.feature_importances_
feature_names = X.columns

#  fearures and importance dataframe

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# top 10 features

print("Top Features Contributing to Fraud Detection:")
print(importance_df.head(10))

# feature importance plotting

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.gca().invert_yaxis()
plt.title("Feature Importance for Fraud Detection (Random Forest)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

