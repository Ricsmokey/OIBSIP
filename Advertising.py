import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import os

if  not os.path.exists('Advertising.csv'):
    print("The file 'Advertising.csv' does not exist in the current directory")
    exit()

#Load the data
data = pd.read_csv('Advertising.csv')
print(f"Date Loaded: {data.shape[0]}, rows {data.shape[1]}, columns")

df = pd.DataFrame(data)
df = df.drop(columns=["Unnamed: 0"]) # Drop the 'Id' column
print(f"DataFrame created. Columns: {list(df.columns)}")


# Split the data into features and target variable
X = df.drop('Sales', axis =1)
y= df['Sales']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test: {y_test.shape}")

# Train the model
model = KNeighborsRegressor(n_neighbors=3)
model.fit(X_train, y_train)

# Make Prediction
y_pred = model.predict(X_test)
print(f"Predictions made for {len(y_pred)} test samples.")

# Evaluate the model
m = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"m: {m:.2f}")
print(f"R² Score: {r2:.2f}")
if r2 >= 0.9:
    print("Excellent Model")
    print("Model Successfully Trained")
elif r2 >= 0.7:
    print("Good Model")
    print("Model Successfully Trained")
elif r2 >= 0.5:
    print("Moderate Model, consider working on it")
else:
    print("Model Accuracy is Low, consider working on it")

# Plot a chart to show the relativness
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('Actual Price VS Selling Price')
plt.xlabel('Actual Sales', fontweight='bold')
plt.ylabel('Predicted Sales', fontweight='bold')
plt.tight_layout()
plt.savefig('Sales_Prediction.jpeg', dpi = 300, bbox_inches = 'tight')
plt.show()


# Test the Model with a new sample
new_sample = pd.DataFrame([[230.1, 37.8, 69.2]], columns=['TV', 'Radio', 'Newspaper']) #[TV, Radio, Newspaper]
prediction = model.predict(new_sample)
print(f"Predicted Sales for the new sample: {prediction[0]:.2f}")




