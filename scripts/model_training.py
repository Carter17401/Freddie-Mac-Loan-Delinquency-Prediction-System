from sklearn.preprocessing import MinMaxScaler

# Columns to normalize
columns_to_normalize = [
    'Credit Score',
    'Number of Units',
    'Original Combined Loan-to-Value (CLTV)',
    'Original UPB',
    'Original Interest Rate',
    'Number of Borrowers',
    'First Payment Year',
    'First Payment Month',
    'Maturity Month'
]

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Normalize specified columns
merged_data[columns_to_normalize] = scaler.fit_transform(merged_data[columns_to_normalize])

# Perform one-hot encoding on the specified columns
merged_data = pd.get_dummies(
    merged_data,
    columns=['First Time Homebuyer Flag', 'Property Valuation Method', 'Part of MSA'],
    prefix=['FirstTimeBuyer', 'PropertyValMethod', 'PartOfMSA'],
    drop_first=True  # Avoid dummy variable trap
)

# Display the updated dataset with encoded columns
merged_data.info()

from sklearn.model_selection import train_test_split

# Ensure 'Month 13' is the target column
target_column = 'Month 13'

# Separate features (X) and target variable (y)
X = merged_data.drop(columns=[target_column, 'LOAN_SEQUENCE_NUMBER'])  # Drop target and ID columns
y = merged_data[target_column]  # Target variable

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Display the shape of the resulting splits
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")


from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Initialize the model
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)

# Train the model on the training set
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print classification report for detailed metrics
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Example: Make predictions on new data
# Uncomment the following lines if you have new data to predict
# new_data = ...  # Replace with your new data in the same format as X
# new_predictions = rf_model.predict(new_data)
# print("Predictions:", new_predictions)


# Ensure 'Month 13' is the target column
target_column = 'Month 13'

# Separate features (X) and target variable (y)
X = merged_data.drop(columns=[target_column, 'LOAN_SEQUENCE_NUMBER'])  # Drop target and ID columns
y = merged_data[target_column]  # Target variable

# Retrain model using the entire dataset
# Initialize the model
model = RandomForestClassifier(random_state=42, n_estimators=100)

# Train the model on the training set
model.fit(X, y)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print classification report for detailed metrics
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Count the occurrences of each predicted value
predicted_counts = pd.Series(y_pred).value_counts()

# Display the counts
print("Count of Predicted Values:")
print(predicted_counts)
