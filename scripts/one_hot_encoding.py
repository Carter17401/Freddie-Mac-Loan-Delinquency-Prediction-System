# Filter features with outlier percentage greater than 5% accroding to GPT suggestion
features_above_5_percent = outlier_df[outlier_df['Outlier Percentage'] > 5]['Feature'].tolist()

# Display the list
print(features_above_5_percent)

# Step 1: Drop features from merged_data with more than 5% outliers
merged_data = merged_data.drop(columns=features_above_5_percent)

# Step 2: Replace Credit Score = 9999 with the mean of the valid values
merged_data["Credit Score"] = merged_data["Credit Score"].replace(9999, np.nan)
credit_score_mean = merged_data["Credit Score"].mean()
merged_data["Credit Score"].fillna(credit_score_mean, inplace=True)

# Step 3: Replace Original Combined Loan-to-Value (CLTV) = 999 with the mean
cltv_col = "Original Combined Loan-to-Value (CLTV)"
merged_data[cltv_col] = merged_data[cltv_col].replace(999, np.nan)
cltv_mean = merged_data[cltv_col].mean()
merged_data[cltv_col].fillna(cltv_mean, inplace=True)

# Step 4: Drop the specified columns
merged_data = merged_data.drop(columns=['Postal Code'], errors='ignore')

# Step 5: Convert 'Property Valuation Method' to categorical type
merged_data['Property Valuation Method'] = merged_data['Property Valuation Method'].astype('category')
