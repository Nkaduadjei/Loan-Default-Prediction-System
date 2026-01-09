"""
Hybrid Weighted Ensemble Learning for Loan Default Prediction (HWELDP)
This module implements a novel ensemble approach combining multiple ML algorithms
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
import pickle
import os


class FeatureSelector:
    """
    Feature selection using multiple methods
    """
    def __init__(self, method='mutual_info'):
        self.method = method
        self.selected_features = None
        
    def select_features(self, X, y, n_features=10):
        """
        Select top n_features using specified method
        """
        if self.method == 'mutual_info':
            from sklearn.feature_selection import mutual_info_classif
            scores = mutual_info_classif(X, y)
            feature_indices = np.argsort(scores)[-n_features:]
            self.selected_features = X.columns[feature_indices].tolist()
        elif self.method == 'correlation':
            corr = X.corrwith(y).abs().sort_values(ascending=False)
            self.selected_features = corr.head(n_features).index.tolist()
        
        return self.selected_features


class HybridEnsembleModel:
    """
    Hybrid Weighted Ensemble Learning for Loan Default Prediction
    Combines XGBoost, Random Forest, LightGBM, and CatBoost with weighted voting
    """
    
    def __init__(self, use_feature_selection=True, n_features=10):
        self.models = {
            'xgboost': XGBClassifier(
                use_label_encoder=False,
                eval_metric='logloss',
                n_jobs=-1,
                random_state=42
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ),
            'catboost': CatBoostClassifier(
                iterations=100,
                random_state=42,
                verbose=False
            )
        }
        self.weights = {}
        self.feature_selector = FeatureSelector() if use_feature_selection else None
        self.n_features = n_features
        self.selected_features = None
        self.is_trained = False
        
    def calculate_ensemble_weights(self, X_val, y_val):
        """
        Calculate weights for each model based on validation performance
        Weights are proportional to AUC-ROC scores
        """
        weights = {}
        total_score = 0
        
        for name, model in self.models.items():
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred_proba)
            weights[name] = auc
            total_score += auc
        
        # Normalize weights
        if total_score > 0:
            for name in weights:
                weights[name] = weights[name] / total_score
        else:
            # Equal weights if all models fail
            for name in weights:
                weights[name] = 1.0 / len(weights)
        
        self.weights = weights
        return weights
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train all base models
        """
        # Feature selection
        if self.feature_selector:
            self.selected_features = self.feature_selector.select_features(
                X_train, y_train, n_features=self.n_features
            )
            X_train = X_train[self.selected_features]
            if X_val is not None:
                X_val = X_val[self.selected_features]
        
        # Train each model
        print("Training base models...")
        for name, model in self.models.items():
            print(f"  Training {name}...")
            model.fit(X_train, y_train)
        
        # Calculate weights if validation set provided
        if X_val is not None and y_val is not None:
            print("Calculating ensemble weights...")
            self.calculate_ensemble_weights(X_val, y_val)
        else:
            # Equal weights
            for name in self.models:
                self.weights[name] = 1.0 / len(self.models)
        
        self.is_trained = True
        print("Training completed!")
        print(f"Ensemble weights: {self.weights}")
        
    def predict(self, X):
        """
        Predict using weighted ensemble voting
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        if self.selected_features:
            X = X[self.selected_features]
        
        # Get predictions from each model
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict_proba(X)[:, 1]
        
        # Weighted average
        ensemble_pred = np.zeros(len(X))
        for name, weight in self.weights.items():
            ensemble_pred += weight * predictions[name]
        
        # Convert to binary predictions (threshold = 0.5)
        return (ensemble_pred >= 0.5).astype(int)
    
    def predict_proba(self, X):
        """
        Predict probabilities using weighted ensemble
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        if self.selected_features:
            X = X[self.selected_features]
        
        # Get predictions from each model
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict_proba(X)[:, 1]
        
        # Weighted average
        ensemble_pred = np.zeros(len(X))
        for name, weight in self.weights.items():
            ensemble_pred += weight * predictions[name]
        
        # Return probabilities for both classes
        proba_class_0 = 1 - ensemble_pred
        proba_class_1 = ensemble_pred
        return np.column_stack([proba_class_0, proba_class_1])
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        """
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y_test, y_pred_proba)
        }
        
        return metrics, y_pred, y_pred_proba
    
    def get_feature_importance(self):
        """
        Get weighted feature importance from all models
        """
        if not self.is_trained:
            return None
        
        importance = {}
        for name, model in self.models.items():
            weight = self.weights.get(name, 1.0 / len(self.models))
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                for i, feat in enumerate(self.selected_features if self.selected_features else range(len(importances))):
                    if feat not in importance:
                        importance[feat] = 0
                    importance[feat] += weight * importances[i]
        
        return importance
    
    def save_model(self, filepath):
        """
        Save the trained model
        """
        model_data = {
            'models': self.models,
            'weights': self.weights,
            'selected_features': self.selected_features,
            'n_features': self.n_features
        }
        pickle.dump(model_data, open(filepath, 'wb'))
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a saved model
        """
        model_data = pickle.load(open(filepath, 'rb'))
        self.models = model_data['models']
        self.weights = model_data['weights']
        self.selected_features = model_data['selected_features']
        self.n_features = model_data['n_features']
        self.is_trained = True
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # Example usage
    print("Hybrid Ensemble Model for Loan Default Prediction")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv("data/loan_data_synthetic.csv")
    target_col = 'Default'
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Preprocess (same as train_model.py)
    from sklearn.preprocessing import LabelEncoder
    cats = ['NewExist', 'UrbanRural', 'LowDoc']
    encoders = {}
    for c in cats:
        if c in X.columns:
            le = LabelEncoder()
            X[c] = X[c].astype(str)
            X[c] = le.fit_transform(X[c])
            encoders[c] = le
    
    num_cols = ['Term', 'NoEmp', 'CreateJob', 'RetainedJob', 'FranchiseCode', 
                'ChgOffPrinGr', 'SBA_Appv', 'DaysforDisbursement']
    for n in num_cols:
        if n in X.columns:
            X[n] = pd.to_numeric(X[n], errors='coerce').fillna(0)
    
    # Train-test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Further split for validation
    X_train_main, X_val, y_train_main, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Train ensemble model
    print("\nTraining Hybrid Ensemble Model...")
    ensemble = HybridEnsembleModel(use_feature_selection=True, n_features=10)
    ensemble.fit(X_train_main, y_train_main, X_val, y_val)
    
    # Evaluate
    print("\nEvaluating on test set...")
    metrics, y_pred, y_pred_proba = ensemble.evaluate(X_test, y_test)
    
    print("\n" + "=" * 60)
    print("Final Results:")
    print("=" * 60)
    for metric_name, value in metrics.items():
        print(f"{metric_name.upper()}: {value:.4f}")
    
    # Save model
    os.makedirs("Serialized Trained Model", exist_ok=True)
    ensemble.save_model("Serialized Trained Model/hybrid_ensemble_model.pkl")
    
    print("\nModel training and evaluation completed!")

