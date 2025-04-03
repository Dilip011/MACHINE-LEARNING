import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv("IRIS.csv")

# Split features and target variable
X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]  # Features
y = df["species"]  # Target

# Encode target labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Normalize numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predictions
y_pred = knn.predict(X_test)

# Display results
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
