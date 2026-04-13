import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import date

if not os.path.exists('cardata.csv'):
    print("The file 'car data.csv' does not exist in current directory")
    exit()


# Load the data
data = pd.read_csv('cardata.csv')

print(f"Data loaded: {data.shape[0]} rows, {data.shape[1]} columns")

df = pd.DataFrame(data)
df = df.drop('Car_Name', axis = 1) # Drop the 'Id' column
print(f"DataFrame created. Columns: {list(df.columns)}")

# Filter and Remodel the Year Column
current_year = date.today().year
df['Car_Year_Usage'] = current_year- df['Year']
df = df.drop('Year', axis = 1)

# Encoding the target variable
df['Fuel_Type'] = df['Fuel_Type'].astype('category').cat.codes
df['Selling_type'] = df['Selling_type'].astype('category').cat.codes
df['Transmission'] = df['Transmission'].astype('category').cat.codes

X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test: {y_test.shape}")


# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict the Model
y_pred = model.predict(X_test)
print(f"Predictions made for {len(y_pred)} test samples.")

# Evaluate the model
m = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {m:.2f}')
print(f'R² Score: {r2:.2f}')
if r2 >= 0.9:
    print("Excellent Model.")
    print("Model Successfully Trained")
elif r2 >= 0.7:
    print("Good Model")
    print("Model Succesfully Trained")
elif r2 >= 0.5:
    print("Moderate Model, consider working on it")
else:
    print("Model accuracy is poor, consider working on it")
    

# Plot a chart to show the relativness
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color='Blue', alpha= 0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('Actual Price VS Selling Price')
plt.xlabel('Actual Price', fontweight='bold')
plt.ylabel('Selling Price', fontweight='bold')
plt.tight_layout()
plt.savefig('PRICE_COMPARISON_PLOT.jpeg', dpi = 300, bbox_inches = 'tight')
plt.show()

print("Scatter plot saved as 'PRICE_COMPARISON_PLOT.jpeg'.")


# Test the model with a new sample
new_sample = pd.DataFrame([[18.61, 40001, 0, 0, 0, 0, 13]])  # [Present_Price, Driven_kms, Fuel_Type, Selling_type, Transmission, Owner, Car_Year_Usage]
prediction = model.predict(new_sample)
print(f'Predicted Selling Price for the new sample: {prediction[0]:.2f}')

