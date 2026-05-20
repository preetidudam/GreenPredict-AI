import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ── Load the trained model ──────────────────────────────────────────────────
model = joblib.load("model/random_forest.pkl")

# ── Load dataset ────────────────────────────────────────────────────────────
df = pd.read_csv("data/plant_data.csv")
df = pd.get_dummies(df, columns=["soil_type"])

X = df.drop("plant", axis=1)
y = df["plant"]

# ── Same split used during training ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Align columns to model's expected features
X_test = X_test.reindex(columns=model.feature_names_in_, fill_value=0)

# ── Predict ──────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)

# ── Report ───────────────────────────────────────────────────────────────────
print("=" * 55)
print("       GreenPredict-AI — Model Accuracy Report")
print("=" * 55)
print(f"  Algorithm        : Random Forest Classifier")
print(f"  Trees            : 300  |  Max Depth : 15")
print(f"  Train/Test Split : 80% / 20%")
print(f"  Test Samples     : {len(y_test)}")
print("=" * 55)
print(f"  Overall Accuracy : {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("=" * 55)
print()
print("Per-Plant Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
