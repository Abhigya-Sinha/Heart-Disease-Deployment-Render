"""
generate_dataset.py
--------------------
Generates heart.csv -- a synthetic dataset that follows the EXACT same
column schema as the Kaggle "Heart Disease Prediction Dataset"
(https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset),
which itself follows the classic UCI Cleveland Heart Disease format.

Why this file exists:
This sandbox environment cannot reach kaggle.com to download the real
file directly, so this script builds a same-shaped, realistically
correlated dataset (900 rows) that lets every part of the assignment
run end-to-end right now.

IMPORTANT FOR YOUR SUBMISSION:
Before you submit, replace the generated heart.csv with the REAL file
downloaded from the Kaggle link in the assignment, keeping the same
file name "heart.csv" in the project root. Everything else (train_model.py,
app.py, UI) will work unchanged because the column names match exactly.

Columns (14 total, same as the Kaggle dataset):
age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak,
slope, ca, thal, target
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 900

age = np.random.randint(29, 78, N)
sex = np.random.randint(0, 2, N)  # 1 = male, 0 = female
cp = np.random.randint(0, 4, N)   # chest pain type (0-3)
trestbps = np.random.randint(94, 201, N)   # resting blood pressure
chol = np.random.randint(126, 565, N)      # serum cholesterol
fbs = np.random.binomial(1, 0.15, N)       # fasting blood sugar > 120 mg/dl
restecg = np.random.randint(0, 3, N)       # resting ECG results
thalach = np.random.randint(71, 203, N)    # max heart rate achieved
exang = np.random.binomial(1, 0.3, N)      # exercise induced angina
oldpeak = np.round(np.random.uniform(0, 6.2, N), 1)  # ST depression
slope = np.random.randint(0, 3, N)
ca = np.random.randint(0, 5, N)            # number of major vessels
thal = np.random.randint(0, 4, N)

# Build a "risk score" from known real-world risk directions so the
# target is realistically correlated with the features (not random),
# which lets a classifier actually learn something meaningful.
risk = (
    0.03 * age
    + 1.0 * sex
    + 0.6 * cp
    + 0.02 * trestbps
    + 0.01 * chol
    + 0.3 * fbs
    - 0.03 * thalach
    + 1.2 * exang
    + 0.5 * oldpeak
    + 0.4 * ca
    + 0.3 * thal
    - 8.5
)
prob = 1 / (1 + np.exp(-risk))
target = np.random.binomial(1, prob)

df = pd.DataFrame({
    "age": age,
    "sex": sex,
    "cp": cp,
    "trestbps": trestbps,
    "chol": chol,
    "fbs": fbs,
    "restecg": restecg,
    "thalach": thalach,
    "exang": exang,
    "oldpeak": oldpeak,
    "slope": slope,
    "ca": ca,
    "thal": thal,
    "target": target,
})

df.to_csv("heart.csv", index=False)
print(f"Generated heart.csv with {len(df)} rows.")
print(df.head())
print("\nTarget distribution:\n", df["target"].value_counts())
