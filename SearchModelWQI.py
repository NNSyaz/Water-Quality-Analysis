# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 13:05:03 2024

@author: syazw
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

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

# Split the data into features and target variable
X = data.drop('WQI', axis=1)  # Features
y = data['WQI']  # Target variable

# Split the data into train and test sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize various regression models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'Neural Network': MLPRegressor(random_state=42)
}

# Train and evaluate each model
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Calculate RMSE and R-squared
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    
    # Display model performance
    print(f"{name} - RMSE: {rmse}, R-squared: {r2}")
