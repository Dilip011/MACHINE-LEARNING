import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

np.random.seed(42)

data = {
    'feature1': np.random.rand(100) * 10,
    'feature2': np.random.rand(100) * 10,
    'feature3': np.random.rand(100) * 10,
    'feature4': np.random.rand(100) * 10,
    'label': np.random.choice([0, 1], 100)
}

df = pd.DataFrame(data)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

class_report = classification_report(y_test, y_pred)
coefficients = model.coef_
intercept = model.intercept_

print("\nClassification Report:")
print(class_report)
print("\nModel Coefficients:", coefficients)
print("\nModel Intercept:", intercept)
