import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# loading

file_path = r'Cleaned_Beneficiarydata.csv'
data = pd.read_csv(file_path)

# feature selection

selected_features = [
    'OPAnnualDeductibleAmt', 'IPAnnualDeductibleAmt', 'RenalDiseaseIndicator',
    'ChronicCond_KidneyDisease', 'County', 'Age', 'State',
    'ChronicCond_Heartfailure', 'ChronicCond_Diabetes'
]
X = data[selected_features]
y = data['FraudFlag']

# splitting the dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# initialize the random forest model with tuned parameters

rf_model = RandomForestClassifier(
    n_estimators=300,          # number of trees
    max_depth=10,              # depth of each tree
    min_samples_split=5,       # minimum samples needed to split a node
    class_weight={0: 1, 1: 10}, # adjusting class weights for imbalance
    random_state=42
)

# training rf-model

rf_model.fit(X_train, y_train)

# predictions

y_pred = rf_model.predict(X_test)

# evaluating model performance
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nAccuracy Score:", accuracy_score(y_test, y_pred))

import joblib

# saving trained model

model_path = r'RandomForest_FraudModel.pkl'
joblib.dump(rf_model, model_path)

print(f"Model saved at: {model_path}")
