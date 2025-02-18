from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt

# Initialize models
logistic_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42, max_iter=1000)
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
gb_model = GradientBoostingClassifier(random_state=42, n_estimators=100)

# Set up 5-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize storage for results
results = {
    "Logistic Regression": {"accuracies": [], "classification_reports": []},
    "Random Forest": {"accuracies": [], "classification_reports": []},
    "Gradient Boosting": {"accuracies": [], "classification_reports": []},
}

# Perform cross-validation for all models
for model_name, model in [
    ("Logistic Regression", logistic_model),
    ("Random Forest", rf_model),
    ("Gradient Boosting", gb_model),
]:
    print(f"\n{model_name} - 5-Fold Cross-Validation\n")
    fold = 1
    for train_index, test_index in skf.split(X, y):
        # Split the data into train and test for this fold
        X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
        y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
        
        # Train the model
        model.fit(X_train_fold, y_train_fold)
        
        # Make predictions
        y_pred_fold = model.predict(X_test_fold)
        
        # Calculate accuracy
        accuracy_fold = accuracy_score(y_test_fold, y_pred_fold)
        results[model_name]["accuracies"].append(accuracy_fold)
        
        # Generate classification report
        report_fold = classification_report(y_test_fold, y_pred_fold, zero_division=0, output_dict=True)
        results[model_name]["classification_reports"].append(report_fold)
        
        print(f"Fold {fold} Accuracy: {accuracy_fold:.2f}")
        fold += 1
    
    # Calculate and print average accuracy
    average_accuracy = np.mean(results[model_name]["accuracies"])
    print(f"\nAverage Accuracy (5-Fold CV): {average_accuracy:.2f}")


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple

# Plotting the accuracy for each model across folds and average accuracy
models = list(results.keys())
accuracies = [results[model]["accuracies"] for model in models]
average_accuracies = [np.mean(results[model]["accuracies"]) for model in models]

# Set colors for the average accuracy lines
colors = ['b', 'orange', 'green']  # Blue, Orange, Green

# Plotting
plt.figure(figsize=(10, 6))

# Plot bar chart for accuracies in each fold
fold_handles = []
for i, model in enumerate(models):
    fold_handle = plt.bar(np.arange(5) + (i * 0.2), accuracies[i], width=0.2, label=f'{model} - Fold', alpha=0.7)
    fold_handles.append(fold_handle[0])

# Plot the average accuracy as a dashed line with unique colors
avg_handles = []
for i, (avg_acc, color) in enumerate(zip(average_accuracies, colors)):
    avg_handle = plt.axhline(y=avg_acc, color=color, linestyle='--', label=f'{models[i]} - Avg. Accuracy')
    avg_handles.append(avg_handle)

# Set up labels and title
plt.xticks(np.arange(5), [f'Fold {i+1}' for i in range(5)])
plt.xlabel('Folds')
plt.ylabel('Accuracy (%)')
plt.title('Model Accuracy Comparison (5-Fold CV)')

# Set y-axis to display as percentage
plt.yticks(np.arange(0, 1.1, 0.1), [f'{int(x*100)}%' for x in np.arange(0, 1.1, 0.1)])

# Remove ALL grid lines
plt.grid(False)

# Prepare legend handles and labels
legend_handles = []
legend_labels = []

for i, model in enumerate(models):
    # Create a tuple of handles for each model (fold handle + avg handle)
    legend_handles.append((fold_handles[i], avg_handles[i]))
    legend_labels.append(model)

# Create custom legend
plt.legend(
    legend_handles,
    legend_labels,
    loc='upper center', 
    bbox_to_anchor=(0.5, -0.15), 
    ncol=3,
    frameon=False,
    handler_map={tuple: HandlerTuple(ndivide=None)},
)

# Adjust layout to make room for the legend at the bottom
plt.tight_layout(rect=[0, 0.1, 1, 1])

# Show plot
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Function to plot feature importance side-by-side
def plot_feature_importances_side_by_side(importances_rf, importances_gb, feature_names, top_n=10):
    """
    Plot the top_n feature importances for Random Forest and Gradient Boosting side-by-side.

    Args:
        importances_rf: Array of feature importances from Random Forest.
        importances_gb: Array of feature importances from Gradient Boosting.
        feature_names: List of feature names.
        top_n: Number of top features to display.
    """
    # Prepare data for Random Forest
    importance_rf_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances_rf
    }).sort_values(by="Importance", ascending=False).head(top_n)

    # Prepare data for Gradient Boosting
    importance_gb_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances_gb
    }).sort_values(by="Importance", ascending=False).head(top_n)

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

    # Plot Random Forest feature importances
    sns.barplot(ax=axes[0], x="Importance", y="Feature", data=importance_rf_df, palette="viridis")
    axes[0].set_title("Top Feature Importances - Random Forest")
    axes[0].set_xlabel("Importance Score")
    axes[0].set_ylabel("Feature")

    # Plot Gradient Boosting feature importances
    sns.barplot(ax=axes[1], x="Importance", y="Feature", data=importance_gb_df, palette="viridis")
    axes[1].set_title("Top Feature Importances - Gradient Boosting")
    axes[1].set_xlabel("Importance Score")
    axes[1].set_ylabel("")

    # Adjust layout and display the plots
    plt.tight_layout()
    plt.show()

# Perform cross-validation with feature importance analysis
feature_importances_rf = []
feature_importances_gb = []

for model_name, model, feature_storage in [
    ("Random Forest", rf_model, feature_importances_rf),
    ("Gradient Boosting", gb_model, feature_importances_gb),
]:
    print(f"\n{model_name} - 5-Fold Cross-Validation\n")
    fold = 1

    for train_index, test_index in skf.split(X, y):
        # Split the data into train and test for this fold
        X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
        y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

        # Train the model
        model.fit(X_train_fold, y_train_fold)

        # Make predictions
        y_pred_fold = model.predict(X_test_fold)

        # Calculate accuracy
        accuracy_fold = accuracy_score(y_test_fold, y_pred_fold)
        results[model_name]["accuracies"].append(accuracy_fold)

        # Generate classification report
        report_fold = classification_report(y_test_fold, y_pred_fold, zero_division=0, output_dict=True)
        results[model_name]["classification_reports"].append(report_fold)

        # Collect feature importances
        feature_storage.append(model.feature_importances_)

        print(f"Fold {fold} Accuracy: {accuracy_fold:.2f}")
        fold += 1

    # Calculate and print average accuracy
    average_accuracy = np.mean(results[model_name]["accuracies"])
    print(f"\nAverage Accuracy (5-Fold CV): {average_accuracy:.2f}")

# Calculate average feature importances
avg_importances_rf = np.mean(feature_importances_rf, axis=0)
avg_importances_gb = np.mean(feature_importances_gb, axis=0)

# Plot feature importances side-by-side
plot_feature_importances_side_by_side(avg_importances_rf, avg_importances_gb, X.columns)

# Conclusion
print("The gradient boosting model uses almost one feature excessively. Overfitting- bias variance trade-off")
