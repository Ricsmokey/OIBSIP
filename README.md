# OIBSIP — Machine Learning Internship Projects

Three machine learning projects completed during an Oasis Infobyte internship. Each project covers a different ML task — classification, regression, and sales forecasting using real-world datasets.

---

## Projects

### 1. Iris Flower Classification

**Goal:** Classify Iris flowers into three species (Setosa, Versicolor, Virginica) based on sepal and petal measurements.

**Model:** K-Nearest Neighbors (KNN) Classifier

**Key highlights:**
- Automated encoding of species categories
- Model evaluation with a confusion matrix heatmap
- Detailed classification report covering Precision, Recall, and F1-Score

**Files:** `iris.py`, `iris.csv`, `Iris_Confusion_Matrix.jpeg`

---

### 2. Car Price Prediction

**Goal:** Predict the selling price of used cars based on mileage, fuel type, transmission, and age.

**Model:** Linear Regression

**Key highlights:**
- Feature engineering — converted the `Year` column into a `Car_Year_Usage` column for better model relevance
- Categorical encoding for fuel type and transmission
- Scatter plot comparing actual vs predicted selling prices

**Files:** `Car Price.py`, `cardata.csv`, `Car_Price_Comparison.jpeg`

---

### 3. Sales Prediction from Advertising Spend

**Goal:** Forecast sales revenue based on advertising spend across TV, Radio, and Newspaper channels.

**Model:** Linear Regression

**Key highlights:**
- Correlation heatmap showing the impact of each media channel on sales
- Model evaluation using R² Score and Mean Absolute Error (MAE)
- Scatter plot of predicted vs actual sales figures

**Files:** `Advertising.py`, `Advertising.csv`, `Advertising_vs_Sales.jpeg`, `advertising_Correlation_Heatmap.jpeg`, `advertising_Sales_Prediction.jpeg`

---

## Tech Stack

• Language: Python
• Libraries: * Pandas & NumPy (Data Manipulation)
• Scikit-learn (Machine Learning & Evaluation)
• Matplotlib & Seaborn (Data Visualization)

---

## Setup & Run

```bash
git clone https://github.com/Ricsmokey/OIBSIP.git
cd OIBSIP
pip install pandas numpy scikit-learn matplotlib seaborn
```

Run each project individually:

```bash
python iris.py
python "Car Price.py"
python Advertising.py
```

---

## Author

**Akorede Kareem** — [github.com/Ricsmokey](https://github.com/Ricsmokey)
