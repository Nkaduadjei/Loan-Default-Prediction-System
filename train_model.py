# train_model.py
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import os

# ADJUST path to your CSV
CSV_PATH = "data/loan_data_synthetic.csv"
OUT_DIR = "Serialized Trained Model"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# quick data sanity: columns must include features used by app
# We'll expect these columns (same names used in app):
expected = ['Term','NoEmp','NewExist','CreateJob','RetainedJob',
            'FranchiseCode','UrbanRural','LowDoc','ChgOffPrinGr','SBA_Appv','DaysforDisbursement','Default']
for c in expected:
    if c not in df.columns:
        raise SystemExit(f"Missing column in dataset: {c}")

# Rename target to 'Default' if needed (some earlier code used 'MIS_Status' or 'Default')
target_col = 'Default'  # binary 0/1, 1 = defaulter (match your dataset)
X = df.drop(columns=[target_col])
y = df[target_col]

# label encode categorical columns and save encoders
cats = ['NewExist','UrbanRural','LowDoc']
encoders = {}
for c in cats:
    le = LabelEncoder()
    X[c] = X[c].astype(str)
    X[c] = le.fit_transform(X[c])
    encoders[c] = le

# Ensure numeric columns are numeric
num_cols = ['Term','NoEmp','CreateJob','RetainedJob','FranchiseCode','ChgOffPrinGr','SBA_Appv','DaysforDisbursement']
for n in num_cols:
    X[n] = pd.to_numeric(X[n], errors='coerce').fillna(0)

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model
model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_jobs=-1)
model.fit(X_train, y_train)

# save model in native XGBoost JSON
model.save_model(os.path.join(OUT_DIR, "model.json"))

# save encoders and feature order
pickle.dump(encoders, open(os.path.join(OUT_DIR, "encoders.pkl"), "wb"))
pickle.dump(list(X.columns), open(os.path.join(OUT_DIR, "feature_columns.pkl"), "wb"))

print("Saved model.json, encoders.pkl, feature_columns.pkl into", OUT_DIR)
