import pandas as pd
import os

# Define the file path and the years to process
base_path = 'data/'  # Update path if needed
years = range(2000, 2019)  # Data from 2000 to 2018

# Initialize a list to hold the transformed DataFrames
all_data = []

# Define column headers (Modify based on actual dataset)
headers = [
    "LOAN_SEQUENCE_NUMBER", "MONTHLY_REPORTING_PERIOD", "CURRENT_ACTUAL_UPB", "CURRENT_LOAN_DELINQUENCY_STATUS",
    "LOAN_AGE", "REMAINING_MONTHS_TO_LEGAL_MATURITY", "DEFECT_SETTLEMENT_DATE", "MODIFICATION_FLAG",
    "ZERO_BALANCE_CODE", "ZERO_BALANCE_EFFECTIVE_DATE", "CURRENT_INTEREST_RATE", "CURRENT_NON-INTEREST_BEARING_UPB",
    "DUE_DATE_OF_LAST_PAID_INSTALLMENT", "MI_RECOVERIES", "NET_SALE_PROCEEDS", "NON_MI_RECOVERIES",
    "TOTAL_EXPENSES", "LEGAL_COSTS", "MAINTENANCE_AND_PRESERVATION_COSTS", "TAXES_AND_INSURANCE",
    "MISCELLANEOUS_EXPENSES", "ACTUAL_LOSS_CALCULATION", "CUMULATIVE_MODIFICATION_COST", "STEP_MODIFICATION_FLAG",
    "PAYMENT_DEFERRAL", "ESTIMATED_LOAN_TO_VALUE", "ZERO_BALANCE_REMOVAL_UPB", "DELINQUENT_ACCRUED_INTEREST",
    "DELINQUENCY_DUE_TO_DISASTER", "BORROWER_ASSISTANCE_STATUS_CODE", "CURRENT_MONTH_MODIFICATION_COST",
    "INTEREST_BEARING_UPB"
]

# Loop through each year, process the file, and transform the data
for year in years:
    file_path = os.path.join(base_path, f'sample_svcg_{year}.txt')
    
    if os.path.exists(file_path):
        df = pd.pd.read_csv(file_path, header=None, delimiter="|")
        df.columns = headers  # Assign column headers
        all_data.append(df)

# Concatenate all the transformed DataFrames
if all_data:
    merged_data = pd.concat(all_data, ignore_index=True)
    print("Data successfully loaded and merged.")
else:
    raise ValueError("No data files found. Check the 'data/' folder.")

# Save to CSV (optional)
merged_data.pd.to_csv(os.path.join(base_path, 'merged_loans_USER_PROVIDED.csv'), index=False)
