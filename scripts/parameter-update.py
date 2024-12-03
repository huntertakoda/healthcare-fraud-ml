import pandas as pd

# loading

file_path = r'Cleaned_Beneficiarydata.csv'
data = pd.read_csv(file_path)

print(data.columns)

# fraud detection threshold defining

fraud_threshold_ip = data['IPAnnualReimbursementAmt'].quantile(0.99)  # Top 1% inpatient
fraud_threshold_op = data['OPAnnualReimbursementAmt'].quantile(0.99)  # Top 1% outpatient

# creating fraud detection flag column

data['FraudFlag'] = (
    (data['IPAnnualReimbursementAmt'] > fraud_threshold_ip) |
    (data['OPAnnualReimbursementAmt'] > fraud_threshold_op)
).astype(int)

# saving

data.to_csv(r'Cleaned_Beneficiarydata.csv', index=False)

print(data['FraudFlag'].value_counts())
