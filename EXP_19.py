import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

np.random.seed(42)
data = {
    "Income": np.random.randint(20000, 150000, 100),
    "Credit_Score": np.random.randint(300, 850, 100),
    "Loan_Amount": np.random.randint(5000, 50000, 100),
    "Employment_Status": np.random.choice(["Employed", "Unemployed", "Self-Employed"], 100),
    "Loan_Term": np.random.choice([12, 24, 36, 48, 60], 100),
    "Loan_Purpose": np.random.choice(["Home", "Car", "Education", "Personal"], 100),
    "Loan_Status": np.random.choice(["Approved", "Rejected"], 100)
}

df = pd.DataFrame(data)

label_encoders = {}
for column in ["Employment_Status", "Loan_Purpose", "Loan_Status"]:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
