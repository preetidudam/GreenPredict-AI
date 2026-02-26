import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1️⃣ Load Dataset
df = pd.read_csv("data/plant_data.csv")

print("Dataset Loaded Successfully!")
print("Shape:", df.shape)

# 2️⃣ Convert soil_type to numeric (One-Hot Encoding)
df = pd.get_dummies(df, columns=["soil_type"])

# 3️⃣ Define Features (X) and Target (y)
X = df.drop("plant", axis=1)
y = df["plant"]

# 4️⃣ Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5️⃣ Train Random Forest Model
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    random_state=42
)
model.fit(X_train, y_train)

# 6️⃣ Evaluate Model
y_pred = model.predict(X_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

accuracy = model.score(X_test, y_test)
print("\nModel Accuracy:", accuracy)

# 7️⃣ Save Model
joblib.dump(model, "model/random_forest.pkl")

print("\nModel Saved Successfully inside model folder!")

import matplotlib.pyplot as plt
import numpy as np

importances = model.feature_importances_
features = X.columns

indices = np.argsort(importances)

plt.figure(figsize=(10,6))
plt.title("Feature Importance")
plt.barh(range(len(indices)), importances[indices])
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.show()