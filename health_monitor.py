import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import messagebox

# Generate synthetic data
np.random.seed(42)
n_samples = 1000

# Features
age = np.random.randint(18, 70, size=n_samples)
heart_rate = np.random.randint(60, 100, size=n_samples)
calories_burned = np.random.randint(150, 500, size=n_samples)
steps = np.random.randint(1000, 10000, size=n_samples)
sleep_hours = np.random.randint(4, 10, size=n_samples)
stress_level = np.random.randint(1, 10, size=n_samples)

# Target (Risk Level): 0 - Low, 1 - Medium, 2 - High
risk_level = np.random.choice([0, 1, 2], size=n_samples)

# Create a DataFrame
data = pd.DataFrame({
    'age': age,
    'heart_rate': heart_rate,
    'calories_burned': calories_burned,
    'steps': steps,
    'sleep_hours': sleep_hours,
    'stress_level': stress_level,
    'risk_level': risk_level
})

# Data Preprocessing
X = data.drop(columns=['risk_level'])
y = data['risk_level']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of the model: {accuracy * 100:.2f}%')

# Tkinter GUI Application
def make_prediction():
    """Handles prediction based on user inputs from the GUI."""
    try:
        # Get user inputs
        age_input = int(age_entry.get())
        heart_rate_input = int(heart_rate_entry.get())
        calories_input = int(calories_entry.get())
        steps_input = int(steps_entry.get())
        sleep_input = float(sleep_entry.get())
        stress_input = int(stress_entry.get())
        
        # Validate input ranges
        if not (18 <= age_input <= 70):
            raise ValueError("Age must be between 18 and 70.")
        if not (60 <= heart_rate_input <= 100):
            raise ValueError("Heart rate must be between 60 and 100.")
        if not (150 <= calories_input <= 500):
            raise ValueError("Calories burned must be between 150 and 500.")
        if not (1000 <= steps_input <= 10000):
            raise ValueError("Steps must be between 1000 and 10000.")
        if not (4 <= sleep_input <= 10):
            raise ValueError("Sleep hours must be between 4 and 10.")
        if not (1 <= stress_input <= 10):
            raise ValueError("Stress level must be between 1 and 10.")

        # Standardize the input data
        sample_data = np.array([[age_input, heart_rate_input, calories_input, steps_input, sleep_input, stress_input]])
        sample_scaled = scaler.transform(sample_data)

        # Predict the risk level
        predicted_risk = model.predict(sample_scaled)
        risk_levels = {0: 'Low', 1: 'Medium', 2: 'High'}
        result = risk_levels[predicted_risk[0]]

        # Show the prediction result
        messagebox.showinfo("Prediction Result", f"The predicted risk level is: {result}")

    except ValueError as e:
        # Show validation error messages
        messagebox.showerror("Input Error", str(e))
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {e}")

# GUI Setup
root = tk.Tk()
root.title("Health Risk Level Prediction")
root.geometry("400x500")
root.resizable(False, False)

# GUI Instructions
tk.Label(root, text="Enter the following details:", font=("Arial", 14)).pack(pady=10)

# Input Fields
fields = [
    ("Age (18-70):", "age_entry"),
    ("Heart Rate (60-100):", "heart_rate_entry"),
    ("Calories Burned (150-500):", "calories_entry"),
    ("Steps (1000-10000):", "steps_entry"),
    ("Sleep Hours (4-10):", "sleep_entry"),
    ("Stress Level (1-10):", "stress_entry")
]

entries = {}

# Dynamic Input Field Creation
for label_text, var_name in fields:
    frame = tk.Frame(root)
    frame.pack(pady=5, fill=tk.X, padx=10)
    tk.Label(frame, text=label_text, anchor="w", width=20).pack(side=tk.LEFT)
    entries[var_name] = tk.Entry(frame)
    entries[var_name].pack(side=tk.LEFT, expand=True, fill=tk.X)

age_entry = entries["age_entry"]
heart_rate_entry = entries["heart_rate_entry"]
calories_entry = entries["calories_entry"]
steps_entry = entries["steps_entry"]
sleep_entry = entries["sleep_entry"]
stress_entry = entries["stress_entry"]

# Predict Button
predict_button = tk.Button(root, text="Predict Risk Level", font=("Arial", 12), command=make_prediction)
predict_button.pack(pady=20)

# Run the GUI
root.mainloop()
