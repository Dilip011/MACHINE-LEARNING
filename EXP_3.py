import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

data = {
    "Age": ["Young", "Young", "Middle-aged", "Senior", "Senior", "Senior", "Middle-aged", "Young", "Young", "Senior", "Young", "Middle-aged", "Middle-aged", "Senior"],
    "Income": ["High", "High", "High", "Medium", "Low", "Low", "Low", "Medium", "Low", "Medium", "Medium", "Medium", "High", "Medium"],
    "CreditScore": ["Fair", "Excellent", "Fair", "Fair", "Fair", "Excellent", "Excellent", "Fair", "Fair", "Fair", "Excellent", "Excellent", "Fair", "Excellent"],
    "BuysCar": ["No", "No", "Yes", "Yes", "Yes", "No", "Yes", "No", "Yes", "Yes", "Yes", "Yes", "Yes", "No"]
}

df = pd.DataFrame(data)

le = LabelEncoder()
for column in df.columns:
    df[column] = le.fit_transform(df[column])

X = df.drop(columns=["BuysCar"])
y = df["BuysCar"]

clf = DecisionTreeClassifier(criterion="entropy")
clf.fit(X, y)

print("Decision Tree Structure:")
for i, feature in enumerate(X.columns):
    print(f"Node {i+1}: Split on '{feature}'")

new_sample = pd.DataFrame({"Age": ["Middle-aged"], "Income": ["High"], "CreditScore": ["Fair"]})
for column in new_sample.columns:
    new_sample[column] = le.fit_transform(new_sample[column])

prediction = clf.predict(new_sample)
print(f"\nPrediction for new sample: {'Yes' if prediction[0] == 1 else 'No'}")
