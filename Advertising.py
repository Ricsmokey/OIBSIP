import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

if not os.path.exists('Advertising.csv'):
    print("The file 'Advertising.csv' does not exist in the current directory")
    exit()

# Load the data
data = pd.read_csv('Advertising.csv')
print(f"Data Loaded: {data.shape[0]} rows, {data.shape[1]} columns")

df = pd.DataFrame(data)
df = df.drop(columns=["Unnamed: 0"]) # Drop the 'Id' column
print(f"DataFrame created. Columns: {list(df.columns)}")

# Data Cleaning and EDA
print(f"\n Dataset Overview: {df.head()}")
print(f"\n Descriptive Statistics: {df.describe()}")
print(f"\n Missing Values: {df.isnull().sum().sum()}")


# Split the data into features and target variable
X = df.drop('Sales', axis=1)
y = df['Sales']


# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")


# Train the model
model = LinearRegression()
model.fit(X_train, y_train)


# Make Prediction
y_pred = model.predict(X_test)
print(f"Predictions made for {len(y_pred)} test samples.")
  

# Evaluate the model
m = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Absolute Error: {m:.2f}")
print(f"R² Score: {r2:.2f}")
if r2 >= 0.9:
    print("Excellent model fit.")
    print("Model Successfully Trained")
elif r2 >= 0.7:
    print("Good model fit.")
    print("Model Successfully Trained")
elif r2 >= 0.5:
    print("Moderate model fit — consider working on it.")
else:
    print("Poor model fit — consider working on it.")


# Correlation heatmap
plt.figure(figsize=(7, 5))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('Correlation_Heatmap.jpeg', dpi=300, bbox_inches='tight')
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
channels = ['TV', 'Radio', 'Newspaper']
for ax, channel in zip(axes, channels):
    ax.scatter(df[channel], df['Sales'], alpha=0.5, color='steelblue')
    ax.set_xlabel(f'{channel} Advertising Budget', fontweight='bold')
    ax.set_ylabel('Sales', fontweight='bold')
    ax.set_title(f'{channel} vs Sales')
plt.suptitle('Advertising Spend vs Sales by Channel', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('Advertising_vs_Sales.jpeg', dpi=300, bbox_inches='tight')
plt.show()


# Plot a chart to show the relativness of the predicted values to the actual values
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color='steelblue', alpha=0.6, label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Prediction')
plt.title('Actual Sales vs Predicted Sales')
plt.xlabel('Actual Sales', fontweight='bold')
plt.ylabel('Predicted Sales', fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig('Sales_Prediction.jpeg', dpi=300, bbox_inches='tight')
plt.show()


# Test the Model with a new sample
new_sample = pd.DataFrame([[230.1, 37.8, 69.2]], columns=['TV', 'Radio', 'Newspaper'])
prediction = model.predict(new_sample)
print(f"\nPredicted Sales for TV=230.1, Radio=37.8, Newspaper=69.2: {prediction[0]:.2f}")
