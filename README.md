# Loan Default Prediction System (HWELDP)

**Hybrid Weighted Ensemble Machine Learning for Credit Risk Prediction**

## 📌 Problem Statement

Loan defaults are a major risk for financial institutions. Traditional single-model approaches often fail to generalize across borrower profiles and economic conditions.
This project builds a **robust, production-style credit risk prediction system** that estimates the probability of loan default using a **hybrid weighted ensemble of tree-based ML models**, exposed through a Flask web application.

## 🧠 Solution Overview

Instead of relying on one algorithm, this system combines multiple high-performing gradient boosting models and assigns **dynamic weights based on validation AUC**, improving stability, calibration, and real-world reliability.

The application provides:

- Binary default prediction
- Probability-based confidence score
- Real-time inference via web UI
- Transparent model comparison and evaluation

## ⚙️ Models Used

- **XGBoost**
- **LightGBM**
- **CatBoost**

Each model is trained independently.
Final prediction is computed using a **Weighted Soft Voting Ensemble**, where:

Final Probability = Σ (Model Probability × AUC-based Weight)

Weights are derived from validation AUC scores to prioritize better-generalizing models.

## 📊 Model Evaluation

The system evaluates models using:

- ROC-AUC (primary metric)
- Precision, Recall, F1-Score
- ROC Curves and Comparison Tables

Automated scripts generate:

- Model comparison tables (`CSV`, `JSON`)
- ROC and performance visualizations
- Ensemble vs individual model analysis

This ensures **reproducibility and auditability**, critical for financial ML systems.

## 🖥️ Web Application

A Flask-based interface allows users to:

- Enter borrower details manually
- Get instant default prediction
- View probability/confidence score
- Store predictions for later analysis

Designed to simulate a **real credit risk decision support tool**, not just a demo.

## 🏗️ Project Structure

Loan-Default-Prediction-System/
│
├── app.py # Flask application
├── train_model.py # Model training pipeline
├── hybrid_ensemble_model.py # Weighted ensemble logic
├── comparative_analysis.py # Model performance comparison
├── generate_all_visualizations.py # Automated plots
├── model_utils.py # Utility functions
│
├── data/ # Dataset (sample/processed)
├── static/ # CSS, JS
├── templates/ # HTML templates
├── visualizations/ # ROC curves & charts
│
├── requirements.txt
├── README.md
└── .gitignore

## 🚀 How to Run Locally

### 1️⃣ Install dependencies

pip install -r requirements.txt

### 2️⃣ Verify ML libraries

python -c "import xgboost, lightgbm, catboost; print('OK')"

### 3️⃣ Train models

python train_model.py

### 4️⃣ Run ensemble & evaluation

python hybrid_ensemble_model.py
python comparative_analysis.py
python generate_all_visualizations.py

### 5️⃣ Start web app

python app.py

## 🛠️ Tech Stack

- **Language:** Python
- **ML:** Scikit-learn, XGBoost, LightGBM, CatBoost
- **Backend:** Flask
- **Visualization:** Matplotlib, Seaborn
- **Data Handling:** Pandas, NumPy

## 📈 Key Takeaways

- Demonstrates **production-oriented ML thinking**
- Focuses on **model evaluation, not just accuracy**
- Shows understanding of **credit risk modeling**
- Bridges **machine learning + backend deployment**

## 👩‍💻 Author

**Shruthika T R**
B.Tech — Artificial Intelligence & Data Science
GitHub: [https://github.com/shruthika-tr](https://github.com/shruthika-tr)
