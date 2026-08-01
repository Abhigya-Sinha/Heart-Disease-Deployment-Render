"""
train_model.py
---------------
Covers Assignment Task 1 (Data Understanding & Preprocessing) and
Task 2 (Model Development).

Run this file once to train the model and produce model.pkl, which
app.py then loads to serve predictions.

Usage:
    python train_model.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ---------------------------------------------------------------
# TASK 1: Data Understanding and Preprocessing
# ---------------------------------------------------------------

# 1. Load the dataset using Pandas
df = pd.read_csv("heart.csv")

print("=" * 60)
print("TASK 1: DATA UNDERSTANDING AND PREPROCESSING")
print("=" * 60)

# 2. Display the first five records
print("\nFirst 5 records:")
print(df.head())

# 3. Identify numerical features and target variable
target_col = "target"
numerical_features = [c for c in df.columns if c != target_col]
print(f"\nNumerical features ({len(numerical_features)}): {numerical_features}")
print(f"Target variable: '{target_col}' (1 = Heart Disease Detected, 0 = No Heart Disease)")

# 4. Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# 5. Split the dataset into 80% training and 20% testing
X = df[numerical_features]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining samples: {len(X_train)}  |  Testing samples: {len(X_test)}")

# ---------------------------------------------------------------
# TASK 2: Model Development
# ---------------------------------------------------------------

print("\n" + "=" * 60)
print("TASK 2: MODEL DEVELOPMENT")
print("=" * 60)

# Feature scaling helps Logistic Regression converge reliably
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Logistic Regression classifier
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate using Accuracy Score
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel: Logistic Regression")
print(f"Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))

# Save the trained model AND the scaler AND the feature order together,
# since app.py needs all three to make consistent predictions.
bundle = {
    "model": model,
    "scaler": scaler,
    "feature_order": numerical_features,
    "accuracy": accuracy,
}
joblib.dump(bundle, "model.pkl")
print("\nSaved trained model, scaler, and feature order to model.pkl")
