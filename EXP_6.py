import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

data = {
    'feature1': np.random.randint(0, 10, 100),
    'feature2': np.random.randint(0, 10, 100),
    'feature3': np.random.randint(0, 10, 100),
    'feature4': np.random.randint(0, 10, 100),
    'label': np.random.choice(['A', 'B', 'C'], 100)
}

df = pd.DataFrame(data)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)
