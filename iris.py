import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

if not os.path.exists('iris.csv'):
    print("The file 'iris.csv' does not exist in the current directory.")
    exit()


# Load the data
data = pd.read_csv('iris.csv')
print(f"Data loaded: {data.shape[0]} rows, {data.shape[1]} columns")

df = pd.DataFrame(data)
df = df.drop('Id', axis=1)  # Drop the 'Id' column
print(f"DataFrame created. Columns: {list(df.columns)}")

# Data Cleaning and EDA
print(f"\n Dataset Overview: {data.head()}")
print(f"\n Descriptive Statistics: {data.describe()}")
print(f"\n Missing Values: {data.isnull().sum().sum()}")


# Encoding the target variable
Species_categories = df['Species'].astype('category').cat.categories
df['Species'] = df['Species'].astype('category').cat.codes
print(f"Species encoded. Categories: {list(Species_categories)}")


# Split the data into features and target variable
X = df.drop('Species', axis=1)
y = df['Species']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")


# Train the model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
print(f"Predictions made for {len(y_pred)} test samples.")

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
if accuracy >= 0.9:
    print("Good model.")
    print("Model succesfully trained")
else:
    print("Model accuracy is low, consider working on it.")
print('Classification Report:')
print(classification_report(y_test, y_pred))


# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{conf_matrix}")
sns.heatmap(conf_matrix, annot=True, fmt='d',
            xticklabels=data['Species'].unique(),
            yticklabels=data['Species'].unique())
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('Confusion_Matrix.jpeg', dpi=300, bbox_inches='tight')
plt.show()

print("Confusion matrix plot saved as 'Confusion_Matrix.jpeg'.")


# Test the model with a new sample
new_sample = pd.DataFrame([[5.4, 3.9, 1.3, 0.4]])  # [SepallenghtCm, SepalwidthCm, PetallengthCm, PetalwidthCm]
prediction = model.predict(new_sample)
prediction = Species_categories[prediction[0]]
print(f'Predicted species for the new sample: {prediction}')

