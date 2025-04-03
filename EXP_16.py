import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn import datasets

# 1️⃣ Load the IRIS Dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

# 2️⃣ Split into Train & Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 3️⃣ Add Noise to Features (to make classification harder)
X_train_noisy = X_train + np.random.normal(0, 0.1, X_train.shape)
X_test_noisy = X_test + np.random.normal(0, 0.1, X_test.shape)

# 4️⃣ Standardize Features (improves performance for some models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_noisy)
X_test_scaled = scaler.transform(X_test_noisy)

# 5️⃣ Define Classification Models
models = {
    "Logistic Regression": LogisticRegression(),
    "K-Nearest Neighbors (k=3)": KNeighborsClassifier(n_neighbors=3),
    "Naïve Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}

# 6️⃣ Train & Evaluate Models
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    print(f"\n{name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

# 7️⃣ Evaluate Models with Cross-Validation
print("\n🔄 Cross-Validation Results (5-Fold):")
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5)
    print(f"{name}: Mean Accuracy = {scores.mean():.4f}, Std Dev = {scores.std():.4f}")

# 8️⃣ Evaluate KNN with Different k values
print("\n🔍 KNN Performance with Different k Values:")
for k in [1, 3, 5, 10]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred_knn = knn.predict(X_test_scaled)
    print(f"KNN (k={k}) Accuracy: {accuracy_score(y_test, y_pred_knn):.4f}")
