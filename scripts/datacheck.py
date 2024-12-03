import pandas as pd

file_path = r'Test_Beneficiarydata.csv'

data = pd.read_csv(file_path)

print(data.info())

print(data.head())

# 63968 entries, 25 columns
# ids, dates, categorical data
# high amount of missing data in the dod (date of death) category
