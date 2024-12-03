import pandas as pd

file_path = r'Test_Beneficiarydata.csv'

data = pd.read_csv(file_path)

# missing values check
print("Missing Values:\n", data.isnull().sum())

# numeric columns stat 
print("\nDescriptive Statistics:\n", data.describe())

# dod column too many missing values going to be dropping that column

data = data.drop(columns=['DOD'])

print(data.info())

# dropped

import matplotlib.pyplot as plt
import seaborn as sns

# ip / op reimbursement amounts

plt.figure(figsize=(10, 5))
sns.histplot(data['IPAnnualReimbursementAmt'], bins=50, kde=True, color='blue', label='Inpatient')
sns.histplot(data['OPAnnualReimbursementAmt'], bins=50, kde=True, color='orange', label='Outpatient')
plt.legend()
plt.title("Distribution of Annual Reimbursement Amounts")
plt.xlabel("Reimbursement Amount")
plt.ylabel("Frequency")
plt.show()

# chronic condition frequency rate

chronic_cols = [col for col in data.columns if 'ChronicCond' in col]
data[chronic_cols].sum().plot(kind='bar', figsize=(10, 5), color='purple')
plt.title("Frequency of Chronic Conditions")
plt.xlabel("Condition")
plt.ylabel("Count")
plt.show()

# correlation analysis // cc - ra

cc_and_ra_cols = [col for col in data.columns if 'ChronicCond' in col] + [
    'IPAnnualReimbursementAmt', 'OPAnnualReimbursementAmt'
]
correlation_matrix = data[cc_and_ra_cols].corr()

# correlation mtrx

print(correlation_matrix)

# heatmap plot

plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Between Chronic Conditions and Reimbursement Amounts")
plt.show()

# scatterplot testing

plt.figure(figsize=(8, 5))
sns.scatterplot(
    x=data['ChronicCond_Cancer'],
    y=data['IPAnnualReimbursementAmt'],
    alpha=0.5
)
plt.title("Reimbursement Amount vs ChronicCond_Cancer")
plt.xlabel("Cancer Condition (0 = No, 1 = Yes)")
plt.ylabel("Inpatient Reimbursement Amount")
plt.show()

# outlier thresold marking here

ip_outlier_threshold = data['IPAnnualReimbursementAmt'].quantile(0.99)  # Top 1% for inpatient
op_outlier_threshold = data['OPAnnualReimbursementAmt'].quantile(0.99)  # Top 1% for outpatient

# ip / op reimbursement filtering

ip_outliers = data[data['IPAnnualReimbursementAmt'] > ip_outlier_threshold]
op_outliers = data[data['OPAnnualReimbursementAmt'] > op_outlier_threshold]

print("Inpatient Outliers - Chronic Conditions:")
print(ip_outliers[[col for col in data.columns if 'ChronicCond' in col]].sum())

print("\nOutpatient Outliers - Chronic Conditions:")
print(op_outliers[[col for col in data.columns if 'ChronicCond' in col]].sum())

print("\nSample of Inpatient Reimbursement Outliers:")
print(ip_outliers.head())

print("\nSample of Outpatient Reimbursement Outliers:")
print(op_outliers.head())
 
# (outliers)
# cancer stroke osteoporosis highest reimbursement rates in both ip & op
# op sees a higher rate of rheumatoid arthritis and depression also

# data preprocessing

from datetime import datetime

data['DOB'] = pd.to_datetime(data['DOB'])
current_year = datetime.now().year
data['Age'] = current_year - data['DOB'].dt.year
data = data.drop(columns=['BeneID', 'DOB'])

print(data.info())

# encoding RenalDiseaseIndicator
data['RenalDiseaseIndicator'] = data['RenalDiseaseIndicator'].apply(lambda x: 1 if x == 'Y' else 0)

print(data[['Gender', 'RenalDiseaseIndicator']].head())

from sklearn.preprocessing import MinMaxScaler

# scaling columns
scaler = MinMaxScaler()
scale_cols = ['IPAnnualReimbursementAmt', 'IPAnnualDeductibleAmt', 
              'OPAnnualReimbursementAmt', 'OPAnnualDeductibleAmt']

data[scale_cols] = scaler.fit_transform(data[scale_cols])

print(data[scale_cols].describe())

# saving the preprocessed data

output_path = r'Cleaned_Beneficiarydata.csv'
data.to_csv(output_path, index=False)

print(f"dataset saved to {output_path}")
