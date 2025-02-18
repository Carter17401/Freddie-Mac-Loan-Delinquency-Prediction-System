import pandas as pd

# Load transformed data and historical loan data
transformed_data = pd.pd.read_csv('data/transformed_loans_USER_PROVIDED.csv')
df_sample = pd.pd.read_csv('data/historical_loans_USER_PROVIDED.csv')

# Standardize column types for merging
transformed_data['LOAN_SEQUENCE_NUMBER'] = transformed_data['LOAN_SEQUENCE_NUMBER'].astype(str)
df_sample['Loan Sequence Number'] = df_sample['Loan Sequence Number'].astype(str)

# Merge the datasets
merged_data = pd.merge(transformed_data, df_sample, left_on='LOAN_SEQUENCE_NUMBER', right_on='Loan Sequence Number', how='inner')

# Drop unnecessary columns
columns_to_drop = ['First Payment Date', 'Maturity Date']
merged_data.drop(columns=columns_to_drop, errors='ignore', inplace=True)

# Save the merged data
merged_data_path = 'data/final_merged_loans_USER_PROVIDED.csv'
merged_data.pd.to_csv(merged_data_path, index=False)
print(f"Merged dataset saved to {merged_data_path}")
