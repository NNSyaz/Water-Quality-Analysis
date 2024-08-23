# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 23:20:01 2023

@author: syazw
"""
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('trained_model2.pkl')

# Function to predict water quality based on user input
def predict_water_quality():
    try:
        # Retrieve user input
        temperature = float(temperature_entry.get())
        rainfall = float(rainfall_entry.get())
        location = location_combobox.get()  
        turbidity = float(turbidity_entry.get())
        oxygen = float(oxygen_entry.get())
        conductivity = float(conductivity_entry.get())

        
        # Make a DataFrame from user input
        user_input = pd.DataFrame({
            'Temperature (°C)': [temperature],
            'Rainfall (mm)': [rainfall],
            'Location': [location],
            'Turbidity (NTU)': [turbidity],
            'Dissolved Oxygen (mg/L)': [oxygen],
            'Conductivity (μS/cm)': [conductivity]

        })
        
        # Make prediction using the model
        prediction = model.predict(user_input)
        
        # Display the predicted pH value
        result_label.config(text=f"Predicted pH: {prediction[0]:.2f}")
        
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numerical values")

# Create the main window
root = tk.Tk()
root.title("Water Quality Prediction")

# Create labels and entry fields for user input
tk.Label(root, text="Temperature (°C):").pack()
temperature_entry = tk.Entry(root)
temperature_entry.pack()

tk.Label(root, text="Rainfall (mm):").pack()
rainfall_entry = tk.Entry(root)
rainfall_entry.pack()

# Label and input for 'Location'
tk.Label(root, text="Location:").pack()
location_combobox = ttk.Combobox(root, values=['0', '1', '2'])  # Change values accordingly
location_combobox.pack()

tk.Label(root, text="Turbidity (NTU):").pack()
turbidity_entry = tk.Entry(root)
turbidity_entry.pack()

tk.Label(root, text="Dissolved Oxygen (mg/L):").pack()
oxygen_entry = tk.Entry(root)
oxygen_entry.pack()

tk.Label(root, text="Conductivity (μS/cm):").pack()
conductivity_entry = tk.Entry(root)
conductivity_entry.pack()

# Button to trigger prediction
predict_button = tk.Button(root, text="Predict", command=predict_water_quality)
predict_button.pack()

# Display predicted result
result_label = tk.Label(root, text="")
result_label.pack()

# Run the main loop
root.mainloop()
