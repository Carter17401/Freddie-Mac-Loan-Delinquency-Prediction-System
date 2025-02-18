import pandas as pd
import numpy as np

# Load merged data
file_path = 'data/merged_loans_USER_PROVIDED.csv'
merged_data = pd.pd.read_csv(file_path)

# Function to reclassify delinquency status
def reclassify_status(value):
    if pd.isna(value):  # Handle NaN values
        return value
    elif str(value).isdigit():
        return '3' if int(value) > 3 else str(int(value))
    return value

# Apply transformations
merged_data['CURRENT_LOAN_DELINQUENCY_STATUS'] = merged_data['CURRENT_LOAN_DELINQUENCY_STATUS'].astype(str)
merged_data['CURRENT_LOAN_DELINQUENCY_STATUS'] = merged_data['CURRENT_LOAN_DELINQUENCY_STATUS'].apply(reclassify_status)

# Drop unnecessary columns
columns_to_drop = ['DEFECT_SETTLEMENT_DATE', 'ESTIMATED_LOAN_TO_VALUE']
merged_data.drop(columns=columns_to_drop, errors='ignore', inplace=True)

# Save cleaned data
cleaned_path = 'data/cleaned_loans_USER_PROVIDED.csv'
merged_data.pd.to_csv(cleaned_path, index=False)
print(f"Data cleaned and saved to {cleaned_path}")
