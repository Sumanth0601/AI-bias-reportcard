"""
Generate heart_disease_demo.csv from the UCI Cleveland Heart Disease dataset.
Run once: python generate_demo.py
"""
import urllib.request
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from pathlib import Path

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
           "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]

print("Downloading UCI Heart Disease dataset...")
raw_path = Path(__file__).parent / "cleveland.data"
urllib.request.urlretrieve(URL, raw_path)

df = pd.read_csv(raw_path, header=None, names=COLUMNS, na_values="?")
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# Binarize target
df["target"] = (df["target"] > 0).astype(int)

# Encode sex
df["sex"] = df["sex"].map({1: "male", 0: "female"})

# Age groups
def age_group(age):
    if age < 45:
        return "< 45"
    elif age < 55:
        return "45-54"
    elif age < 65:
        return "55-64"
    else:
        return "65+"

df["age_group"] = df["age"].apply(age_group)

# Features: exclude demographic and target columns
feature_cols = [c for c in COLUMNS if c not in ["target", "sex", "age"]]
X = df[feature_cols].copy()
y = df["target"]

# Train/test split for fitting only
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_all_s = scaler.transform(X)

model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_s, y_train)

# Predict on full dataset
df["prediction"] = model.predict(X_all_s)

# Save
out_path = Path(__file__).parent / "heart_disease_demo.csv"
df[["age", "age_group", "sex", "target", "prediction"]].to_csv(out_path, index=False)
print(f"Saved {len(df)} rows to {out_path}")

# Quick stats
from sklearn.metrics import accuracy_score
print(f"Overall accuracy: {accuracy_score(df['target'], df['prediction']):.3f}")
for grp in ["sex", "age_group"]:
    print(f"\n--- {grp} ---")
    for val, sub in df.groupby(grp):
        acc = accuracy_score(sub["target"], sub["prediction"])
        recall = (sub[(sub['target']==1) & (sub['prediction']==1)].shape[0] /
                  max(sub[sub['target']==1].shape[0], 1))
        print(f"  {val}: n={len(sub)}, acc={acc:.3f}, recall={recall:.3f}")

# Cleanup
raw_path.unlink()
