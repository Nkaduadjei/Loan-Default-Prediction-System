Loan Default Prediction System (HWELDP)

Hybrid Weighted Ensemble Machine Learning for Credit Risk Prediction

📌 Problem Statement

Loan defaults represent a significant financial risk for lending institutions. Single-model approaches often fail to generalize across diverse borrower profiles and changing economic conditions.

This project implements a production-oriented credit risk prediction system that estimates the probability of loan default using a hybrid weighted ensemble of tree-based machine learning models, exposed through a Flask web application.

🧠 Solution Overview

Instead of relying on a single algorithm, this system combines multiple high-performing gradient boosting models and assigns dynamic weights based on validation ROC-AUC, improving robustness, calibration, and real-world reliability.

Key capabilities:

Binary loan default prediction

Probability-based confidence scoring

Real-time inference via a web interface

Transparent model comparison and evaluation

⚙️ Models Used

XGBoost

LightGBM

CatBoost

Each model is trained independently.
Final predictions are generated using a Weighted Soft Voting Ensemble:

Final Probability = Σ (Model Probability × AUC-based Weight)

Weights are derived from validation ROC-AUC scores, prioritizing models that generalize better.

📊 Model Evaluation

Models are evaluated using:

ROC-AUC (primary metric)

Precision, Recall, F1-Score

ROC Curves and comparative analysis

Automated scripts generate:

Model comparison tables (CSV, JSON)

ROC curves and performance visualizations

Ensemble vs individual model benchmarking

This ensures reproducibility, transparency, and auditability, which are essential in financial ML systems.

🖥️ Web Application

A Flask-based web interface allows users to:

Enter borrower details manually

Receive instant default predictions

View probability/confidence scores

Store predictions for further analysis

The application is designed to simulate a real-world credit risk decision support tool, not a toy demo.

🏗️ Project Structure
Loan-Default-Prediction-System/
│
├── app.py # Flask application
├── train_model.py # Model training pipeline
├── hybrid_ensemble_model.py # Weighted ensemble logic
├── comparative_analysis.py # Model performance comparison
├── generate_all_visualizations.py # Automated plots and reports
├── model_utils.py # Shared utilities and helpers
│
├── data/ # Dataset (raw / processed samples)
├── static/ # Frontend assets (CSS, JavaScript)
├── templates/ # HTML templates (Jinja2)
├── visualizations/ # ROC curves and performance charts
│
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── .gitignore # Ignored files and directories

🚀 How to Run Locally
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Verify ML libraries
python -c "import xgboost, lightgbm, catboost; print('OK')"

3️⃣ Train models
python train_model.py

4️⃣ Run ensemble and evaluation
python hybrid_ensemble_model.py
python comparative_analysis.py
python generate_all_visualizations.py

5️⃣ Start the web application
python app.py

🛠️ Tech Stack

Language: Python

Machine Learning: Scikit-learn, XGBoost, LightGBM, CatBoost

Backend: Flask

Visualization: Matplotlib, Seaborn

Data Processing: Pandas, NumPy

📈 Key Takeaways

Demonstrates production-focused ML system design

Emphasizes evaluation and model reliability, not just accuracy

Shows practical understanding of credit risk modeling

Integrates machine learning with backend deployment

👩‍💻 Author

Shruthika T R
B.Tech — Artificial Intelligence & Data Science
GitHub: https://github.com/shruthika-tr
