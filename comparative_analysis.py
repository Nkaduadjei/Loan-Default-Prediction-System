"""
Comparative Analysis Script
Compares multiple ML models for loan default prediction
"""

import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
from hybrid_ensemble_model import HybridEnsembleModel
import json


class ModelComparator:
    """
    Compare multiple ML models for loan default prediction
    """
    
    def __init__(self):
        self.models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            ),
            'Decision Tree': DecisionTreeClassifier(
                max_depth=10,
                random_state=42
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'XGBoost (Baseline)': XGBClassifier(
                use_label_encoder=False,
                eval_metric='logloss',
                n_jobs=-1,
                random_state=42
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ),
            'CatBoost': CatBoostClassifier(
                iterations=100,
                random_state=42,
                verbose=False
            ),
            'Proposed (HWELDP)': None  # Will be set separately
        }
        self.results = {}
        self.training_times = {}
        self.prediction_times = {}
        
    def preprocess_data(self, df):
        """
        Preprocess the loan dataset
        """
        target_col = 'Default'
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Encode categorical variables
        cats = ['NewExist', 'UrbanRural', 'LowDoc']
        encoders = {}
        for c in cats:
            if c in X.columns:
                le = LabelEncoder()
                X[c] = X[c].astype(str)
                X[c] = le.fit_transform(X[c])
                encoders[c] = le
        
        # Convert numeric columns
        num_cols = ['Term', 'NoEmp', 'CreateJob', 'RetainedJob', 'FranchiseCode',
                    'ChgOffPrinGr', 'SBA_Appv', 'DaysforDisbursement']
        for n in num_cols:
            if n in X.columns:
                X[n] = pd.to_numeric(X[n], errors='coerce').fillna(0)
        
        return X, y, encoders
    
    def train_and_evaluate(self, X_train, y_train, X_test, y_test, model_name, model):
        """
        Train a model and evaluate its performance
        """
        print(f"\nTraining {model_name}...")
        
        # Training time
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        self.training_times[model_name] = training_time
        
        # Prediction time
        start_time = time.time()
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        prediction_time = time.time() - start_time
        self.prediction_times[model_name] = prediction_time
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'training_time': training_time,
            'prediction_time': prediction_time
        }
        
        if y_pred_proba is not None:
            metrics['auc_roc'] = roc_auc_score(y_test, y_pred_proba)
            metrics['roc_curve'] = roc_curve(y_test, y_pred_proba)
        else:
            metrics['auc_roc'] = 0.0
            metrics['roc_curve'] = None
        
        metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred)
        metrics['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
        
        self.results[model_name] = {
            'metrics': metrics,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
        print(f"  Training Time: {training_time:.2f}s")
        
        return metrics
    
    def compare_all_models(self, df, test_size=0.2):
        """
        Compare all models on the dataset
        """
        print("=" * 60)
        print("COMPARATIVE ANALYSIS: Loan Default Prediction Models")
        print("=" * 60)
        
        # Preprocess data
        X, y, encoders = self.preprocess_data(df)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Further split for validation (for ensemble)
        X_train_main, X_val, y_train_main, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # Train and evaluate each model
        for model_name, model in self.models.items():
            if model is None:
                # Train ensemble model separately
                print(f"\nTraining {model_name}...")
                ensemble = HybridEnsembleModel(use_feature_selection=True, n_features=10)
                start_time = time.time()
                ensemble.fit(X_train_main, y_train_main, X_val, y_val)
                training_time = time.time() - start_time
                self.training_times[model_name] = training_time
                
                start_time = time.time()
                metrics, y_pred, y_pred_proba = ensemble.evaluate(X_test, y_test)
                prediction_time = time.time() - start_time
                self.prediction_times[model_name] = prediction_time
                
                metrics['training_time'] = training_time
                metrics['prediction_time'] = prediction_time
                metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred)
                metrics['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
                
                # Get ROC curve
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                metrics['roc_curve'] = (fpr, tpr, _)
                
                self.results[model_name] = {
                    'metrics': metrics,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                print(f"  F1-Score: {metrics['f1_score']:.4f}")
                print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
                print(f"  Training Time: {training_time:.2f}s")
            else:
                self.train_and_evaluate(X_train, y_train, X_test, y_test, model_name, model)
        
        return self.results
    
    def generate_comparison_table(self):
        """
        Generate a comparison table of all models
        """
        comparison_data = []
        
        for model_name, result in self.results.items():
            metrics = result['metrics']
            comparison_data.append({
                'Model': model_name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}",
                'AUC-ROC': f"{metrics['auc_roc']:.4f}",
                'Training Time (s)': f"{metrics['training_time']:.2f}",
                'Prediction Time (s)': f"{metrics['prediction_time']:.4f}"
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        return df_comparison
    
    def save_results(self, filepath='comparison_results.json'):
        """
        Save comparison results to JSON
        """
        # Convert results to JSON-serializable format
        json_results = {}
        for model_name, result in self.results.items():
            json_results[model_name] = {
                'metrics': {k: float(v) if isinstance(v, (np.float64, np.float32, float)) else str(v)
                           for k, v in result['metrics'].items() if k not in ['roc_curve', 'confusion_matrix', 'classification_report']},
                'training_time': float(self.training_times[model_name]),
                'prediction_time': float(self.prediction_times[model_name])
            }
        
        with open(filepath, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\nResults saved to {filepath}")


if __name__ == "__main__":
    # Load data
    print("Loading dataset...")
    df = pd.read_csv("data/loan_data_synthetic.csv")
    print(f"Dataset shape: {df.shape}")
    print(f"Default rate: {df['Default'].mean():.2%}")
    
    # Run comparative analysis
    comparator = ModelComparator()
    results = comparator.compare_all_models(df, test_size=0.2)
    
    # Generate comparison table
    print("\n" + "=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)
    comparison_table = comparator.generate_comparison_table()
    print(comparison_table.to_string(index=False))
    
    # Save results
    comparator.save_results('comparison_results.json')
    
    # Save comparison table
    comparison_table.to_csv('comparison_table.csv', index=False)
    print("\nComparison table saved to comparison_table.csv")
    
    print("\n" + "=" * 60)
    print("Comparative analysis completed!")
    print("=" * 60)

