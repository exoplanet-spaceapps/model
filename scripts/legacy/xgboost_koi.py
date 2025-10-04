"""
# XGBoost KOI Feature Training
Using sklearn package to train a XGBoost model for koi features

## Import Packages
"""

import numpy as np
import csv
import pandas as pd
import glob
import os
import re
import math
import matplotlib.pyplot as plt
import seaborn as sns

"""### Import Packages about XGBoost"""

from sklearn.utils import shuffle
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix, precision_score, recall_score, roc_curve, auc, ConfusionMatrixDisplay, roc_auc_score
from sklearn.model_selection import ShuffleSplit, cross_val_score, KFold, GridSearchCV


folder_path = os.getcwd() + 'Dataset\KOI_features_csv'

"""### List and Process Files"""

# List all CSV files in the folder
all_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
print(f"Found {len(all_files)} CSV files.")

# Combine all CSV files into a single DataFrame
combined_df = pd.DataFrame()

for file in all_files:
    file_path = os.path.join(folder_path, file)
    print(f"Processing file: {file}")
    # Use the first column as the index
    df = pd.read_csv(file_path, index_col=0)
    combined_df = pd.concat([combined_df, df])

# Sort the combined DataFrame by index
combined_df = combined_df.sort_index()

# Check for duplicate indices in combined_df
duplicate_indices_combined = combined_df.index[combined_df.index.duplicated(keep=False)]

if not duplicate_indices_combined.empty:
    print("Duplicate indices found in combined_df:")
    print(duplicate_indices_combined)
else:
    print("No duplicate indices found in combined_df.")

# Load the second CSV file with the first column as the index
comparison_file_path = 'Dataset\q1_q17_dr25_koi.csv'
comparison_df = pd.read_csv(comparison_file_path, index_col=1)
comparison_df = comparison_df.iloc[:1886]

# Find the missing indices
missing_indices = comparison_df.index.difference(combined_df.index)

# Display the missing indices
if missing_indices.empty:
    print("No missing indices found.")
else:
  print("Missing indices:")
  print(missing_indices)


"""## Data Preprocess
### Drop Columns that is useless and with NaN values
"""

# Identify columns with all the same values or entirely NaN
same_value_columns = []
same_value_data = {}  # Dictionary to store the value each same column holds
nan_columns = []

# Check for columns with a single unique value
for col in combined_df.columns:
    if combined_df[col].nunique(dropna=True) == 1:  # Only one unique value (ignoring NaNs)
        # Store the unique value for columns with the same value
        unique_value = combined_df[col].dropna().iloc[0]  # Get the non-NaN value
        same_value_columns.append(col)
        same_value_data[col] = unique_value

# Check for columns with all NaN values
nan_columns = combined_df.columns[combined_df.isna().all()].tolist()

# Print out the results
print("Columns with the same value:")
for col in same_value_columns:
    print(f"{col}: {same_value_data[col]}")

print("\nColumns with all NaN values:")
print(nan_columns)

# Combine columns to drop
columns_to_drop = list(set(same_value_columns + nan_columns))

# Remove duplicates in case of overlap
columns_to_drop = list(set(columns_to_drop))

# List the columns to drop
print("Columns to drop:", columns_to_drop)

"""### Drop the identified columns"""

koi_features = combined_df.drop(columns=columns_to_drop)

"""### Check for rows that contains 'inf' value"""

inf_columns = koi_features.columns[(koi_features == float('inf')).any() | (koi_features == float('-inf')).any()]

# Loop through each column with inf values and display the indices
for col in inf_columns:
    # Get the indices of rows where the column has inf values
    inf_indices = koi_features[koi_features[col].isin([float('inf'), float('-inf')])].index
    print(f"Indices with inf values in column '{col}':")
    print(inf_indices)

# Drop rows where any column has inf values
koi_features = koi_features[~koi_features.isin([float('inf'), float('-inf')]).any(axis=1)]
print("Shape of the cleaned data:", koi_features.shape)

"""### Check for rows that contains 'nan' value"""

# Find rows and columns with NaN values
nan_rows_cols = koi_features[koi_features.isna().any(axis=1)]

if not nan_rows_cols.empty:
    print("Rows and columns with NaN values:")
    for index, row in nan_rows_cols.iterrows():
        nan_cols = row.index[row.isna()]
        print(f"Row index: {index}, Columns with NaN: {list(nan_cols)}")
else:
    print("No NaN values found in the DataFrame.")

# Fill all the nan value with 0
koi_features = koi_features.fillna(0)

# Find rows and columns with NaN values
nan_rows_cols = koi_features[koi_features.isna().any(axis=1)]

if not nan_rows_cols.empty:
    print("Rows and columns with NaN values:")
    for index, row in nan_rows_cols.iterrows():
        nan_cols = row.index[row.isna()]
        print(f"Row index: {index}, Columns with NaN: {list(nan_cols)}")
else:
    print("No NaN values found in the DataFrame.")

"""### Code to Extract label Column and Separate It from koi_features"""

# Extract the 'label' column as the target for training
koi_labels = koi_features.pop('label')

"""## Model Training

### Train with 1-1600 Dataset, Test with the 1601-2000
"""

train_features = koi_features.loc[:'K01126.01']
train_labels = koi_labels.loc[:'K01126.01']

test_features = koi_features.loc['K01126.02':]
test_labels = koi_labels.loc['K01126.02':]

test_features

"""### Initialize the XGBoost model"""

# Ensure the data is in the correct format
X = train_features  # Features (training data)
y = train_labels   # Labels (target data)

# Compute class weights
scale_pos_weight = sum(y == 0) / len(y)

# Initialize the XGBoost model with custom hyperparameters
model = XGBClassifier(
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight,
    n_estimators=250,
    learning_rate=0.05,
    max_depth=5,
    objective='binary:logistic',
    gamma=1,
    min_child_weight=10,
    subsample=0.8,
    colsample_bytree=0.8
)

model.fit(train_features, train_labels)

# Predict on test data
predictions = model.predict(test_features)

# Calculate metrics
accuracy = accuracy_score(test_labels, predictions)
f1 = f1_score(test_labels, predictions)
precision = precision_score(test_labels, predictions)
recall = recall_score(test_labels, predictions)

# Confusion matrix
conf_matrix = confusion_matrix(test_labels, predictions)

# Display results
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(test_labels), yticklabels=np.unique(test_labels))
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

"""### K-Fold Cross-Validation"""

# Perform K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=25)  # shuffle=True ensures random shuffling

# List to store classification reports for each fold
reports = []

# Initialize lists to store precision, recall, and F1 for each fold
precisions = []
recalls = []
f1_scores = []

# Loop through each fold
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Generate classification report for this fold
    report = classification_report(y_test, y_pred, output_dict=True)
    reports.append(report)
    print(f"Classification Report for fold {len(reports)}:")
    print(classification_report(y_test, y_pred))

    # Extract precision, recall, and F1-score for each class
    precision_class_0 = report['0']['precision']
    recall_class_0 = report['0']['recall']
    f1_class_0 = report['0']['f1-score']

    precision_class_1 = report['1']['precision']
    recall_class_1 = report['1']['recall']
    f1_class_1 = report['1']['f1-score']

    # Append the metrics for each fold
    precisions.append([precision_class_0, precision_class_1])
    recalls.append([recall_class_0, recall_class_1])
    f1_scores.append([f1_class_0, f1_class_1])

# Calculate the average of precision, recall, and F1-score for each class across all folds
average_precision = np.mean(precisions, axis=0)
average_recall = np.mean(recalls, axis=0)
average_f1 = np.mean(f1_scores, axis=0)

# Print the average metrics
print("\nAverage Precision, Recall, and F1-Score across all folds:")
print(f"Class 0 - Precision: {average_precision[0]}, Recall: {average_recall[0]}, F1-Score: {average_f1[0]}")
print(f"Class 1 - Precision: {average_precision[1]}, Recall: {average_recall[1]}, F1-Score: {average_f1[1]}")

# Get the feature importance from the trained model
feature_importance = model.get_booster().get_score(importance_type='weight')

# Sort the features by importance (descending order)
sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

# Display the sorted features
print("Sorted Feature Importance:")
for feature, score in sorted_features[0:50]:
    print(f"{feature}: {score}")