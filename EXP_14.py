import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("House Price.csv")

# Handle missing values
df.fillna(df.mean(numeric_only=True), inplace=True)  # Replace NaN with mean

# Drop unnecessary column
df.drop(["Id"], axis=1, inplace=True)

# Encode categorical features
categorical_cols = ["MSZoning", "LotConfig", "BldgType", "Exterior1st"]
label_encoders = {col: LabelEncoder().fit(df[col]) for col in categorical_cols}

for col, encoder in label_encoders.items():
    df[col] = encoder.transform(df[col])

# Split dataset
X = df.drop("SalePrice", axis=1)  # Features
y = df["SalePrice"]  # Target

# Normalize numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Display results
print("\nMean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("\nMean Squared Error:", mean_squared_error(y_test, y_pred))
print("\nR² Score:", r2_score(y_test, y_pred))
print("\nModel Coefficients:", model.coef_)
print("\nIntercept:", model.intercept_)
