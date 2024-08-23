# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 13:59:46 2024

@author: syazw
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import joblib
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('water_quality_with_WQI.csv')

# Drop 'Timestamp' column
data.drop('Timestamp', axis=1, inplace=True)

# Check for missing values
missing_values = data.isnull().sum()
print("Missing Values:")
print(missing_values)

# Define columns for numerical and categorical data
numerical_cols = ['Temperature (°C)', 'Rainfall (mm)', 'Turbidity (NTU)', 'Dissolved Oxygen (mg/L)', 'Conductivity (μS/cm)']
categorical_cols = ['Location']

# Preprocessing for categorical data (label encoding)
label_encoders = {}
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    data[col] = label_encoders[col].fit_transform(data[col])

# Statistical Feature Extraction
statistical_features = data.describe().transpose()

# Display statistical features
print("\nStatistical Features:")
print(statistical_features)

# Split the dataset into features (X) and target variable (y)
X = data.drop('WQI', axis=1)  # Features excluding WQI
y = data['WQI']  # Target variable WQI

# Split the data into train and test sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model to a file
joblib.dump(model, 'trained_model_WQI_LinearRegression.pkl')

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Visualize predictions vs actual values
plt.scatter(y_test, y_pred, color='blue', alpha=0.7)
plt.xlabel('Actual WQI')
plt.ylabel('Predicted WQI')
plt.title('Actual vs Predicted WQI')
plt.show()

# Get feature coefficients
feature_coeffs = pd.Series(model.coef_, index=X.columns)

# Sort feature coefficients by their absolute values in descending order
sorted_coeffs = feature_coeffs.abs().sort_values(ascending=False)

# Display feature importance
print("\nFeature Importance (Absolute Coefficients):")
print(sorted_coeffs)

# Visualize feature coefficients
plt.figure(figsize=(10, 6))
sorted_coeffs.plot(kind='bar')
plt.title('Feature Importance (Absolute Coefficients)')
plt.xlabel('Features')
plt.ylabel('Absolute Coefficients')
plt.xticks(rotation=45)
plt.show()
