import pandas as pd

# Load cleaned data
file_path = 'data/cleaned_loans_USER_PROVIDED.csv'
df = pd.pd.read_csv(file_path)

# Pivot delinquency status
df_pivot = df.pivot(index='LOAN_SEQUENCE_NUMBER', columns='MONTHLY_REPORTING_PERIOD', values='CURRENT_LOAN_DELINQUENCY_STATUS')

# Reset index and format column names
df_pivot.reset_index(inplace=True)
df_pivot.columns = [str(col) if isinstance(col, int) else col for col in df_pivot.columns]

# Save transformed data
feature_path = 'data/transformed_loans_USER_PROVIDED.csv'
df_pivot.pd.to_csv(feature_path, index=False)
print(f"Feature-engineered data saved to {feature_path}")
