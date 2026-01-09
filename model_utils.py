# model_utils.py
import os
import pickle
import pandas as pd
from xgboost import XGBClassifier

MODEL_DIR = "Serialized Trained Model"
MODEL_JSON = os.path.join(MODEL_DIR, "model.json")
ENCODERS_PATH = os.path.join(MODEL_DIR, "encoders.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_columns.pkl")

def load_model_and_meta():
    # Loads model, encoders, and feature list; raises clear errors if missing.
    if not os.path.exists(MODEL_JSON):
        raise FileNotFoundError(f"Model JSON not found. Create it by running train_model.py -> {MODEL_JSON}")
    model = XGBClassifier()
    model.load_model(MODEL_JSON)

    encoders = {}
    features = None
    if os.path.exists(ENCODERS_PATH):
        encoders = pickle.load(open(ENCODERS_PATH, "rb"))
    if os.path.exists(FEATURES_PATH):
        features = pickle.load(open(FEATURES_PATH, "rb"))
    return model, encoders, features

def build_input_dataframe(raw_dict, encoders, feature_columns):
    """
    raw_dict: mapping of column names -> values (strings / numbers)
    return: pd.DataFrame with columns ordered as feature_columns
    """
    import pandas as pd
    # numeric columns (as floats)
    numeric_cols = ['Term','NoEmp','CreateJob','RetainedJob','FranchiseCode','ChgOffPrinGr','SBA_Appv','DaysforDisbursement']
    # copy raw
    row = {}
    for k,v in raw_dict.items():
        row[k] = v

    # normalize names: user forms have 'DaysforDibursement' typo — accept both
    if 'DaysforDibursement' in row and 'DaysforDisbursement' not in row:
        row['DaysforDisbursement'] = row.pop('DaysforDibursement')

    # convert numerics
    for n in numeric_cols:
        if n in row:
            try:
                row[n] = float(row[n])
            except Exception:
                row[n] = 0.0
        else:
            row[n] = 0.0

    # encode categorical columns
    for cat in ['NewExist','UrbanRural','LowDoc']:
        if cat in row:
            val = str(row[cat])
            if cat in encoders:
                try:
                    row[cat] = int(encoders[cat].transform([val])[0])
                except Exception:
                    # fallback to mode/0
                    row[cat] = 0
            else:
                # if no encoder available, try to cast numeric
                try:
                    row[cat] = int(float(val))
                except Exception:
                    row[cat] = 0
        else:
            row[cat] = 0

    # enforce column order
    if feature_columns is None:
        # simple default order
        cols = ['Term','NoEmp','NewExist','CreateJob','RetainedJob','FranchiseCode',
                'UrbanRural','LowDoc','ChgOffPrinGr','SBA_Appv','DaysforDisbursement']
    else:
        cols = feature_columns

    ordered_row = [ row.get(c, 0) for c in cols ]
    return pd.DataFrame([ordered_row], columns=cols)
