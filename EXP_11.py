import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv("CREDIT SCORE.csv")

# Drop unnecessary columns
df.drop(["ID", "Customer_ID", "Month", "Name", "SSN", "Type_of_Loan"], axis=1, inplace=True)

# Encode categorical features
label_encoders = {}
categorical_cols = ["Occupation", "Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour", "Credit_Score"]

for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col])

# Handle missing values (optional)
df.fillna(df.median(), inplace=True)

# Split dataset
X = df.drop(["Credit_Score"], axis=1)  # Features
y = df["Credit_Score"]  # Target

# Normalize numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Display results
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
