# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 19:32:08 2023

@author: syazw
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import joblib

# Load the dataset
data = pd.read_csv('water_quality_dataset.csv')

# Convert 'Timestamp' column to datetime format
data['Timestamp'] = pd.to_datetime(data['Timestamp'], dayfirst=True)  # Specify dayfirst=True for DD/MM/YYYY format

# Check for missing values
#missing_values = data.isnull().sum()
#print("Missing Values:")
#print(missing_values)

# Data Pre-processing
# Convert categorical 'Location' column to numerical
data['Location'] = data['Location'].map({'Lake': 0, 'River': 1, 'Pond': 2})

# Separating numerical and categorical columns
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_cols = ['Location']  # Categorical column after transformation

# Preprocessing for numerical data (scaling)
numerical_transformer = StandardScaler()

# Preprocessing for categorical data (one-hot encoding)
categorical_transformer = OneHotEncoder(drop='first')

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Apply preprocessing pipeline
transformed_data = preprocessor.fit_transform(data)

# Get the feature names for the transformed DataFrame
transformed_num_cols = numerical_cols
transformed_cat_cols = preprocessor.named_transformers_['cat'].get_feature_names_out(input_features=categorical_cols)
transformed_cols = transformed_num_cols + list(transformed_cat_cols)

# Convert transformed_data (array) back to DataFrame
transformed_df = pd.DataFrame(transformed_data, columns=transformed_cols)

# Display the preprocessed data
print("\nPreprocessed Data:")
print(transformed_df.head())

# Split the data into features and target variable
X = transformed_df.drop('pH', axis=1)  # Features
y = transformed_df['pH']  # Target variable

# Split the data into train and test sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the best hyperparameters
best_params = {'activation': 'relu', 'alpha': 0.0001, 'hidden_layer_sizes': (50, 50)}

# Train the model with best hyperparameters
best_model = MLPRegressor(random_state=42, **best_params)
best_model.fit(X_train, y_train)

joblib.dump(best_model, 'water_quality_prediction_model.pkl')

# Predictions on test set
y_pred = best_model.predict(X_test)

# Calculate RMSE and R-squared for the best model
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
#print(f"\nBest Model - RMSE: {rmse}, R-squared: {r2}")

# Visualize predictions vs actual values
plt.scatter(y_test, y_pred, color='blue', alpha=0.7)
plt.xlabel('Actual pH')
plt.ylabel('Predicted pH')
plt.title('Actual vs Predicted pH')
plt.show()

# Feature Importance
feature_importance = pd.Series(best_model.coefs_[0].mean(axis=1), index=X.columns).sort_values(ascending=False)
print("\nFeature Importance:")
print(feature_importance)
