# app.py
import os
import pickle
import json
import sqlite3
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_file, flash, redirect, url_for
from werkzeug.utils import secure_filename
from model_utils import load_model_and_meta, build_input_dataframe
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io
import logging
import webbrowser
from threading import Timer

# config
UPLOAD_FOLDER = "data"
ALLOWED_EXT = {"csv"}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret-key")
DB = "submissions.db"

# setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# try load model
try:
    model, encoders, feature_columns = load_model_and_meta()
    logger.info("Model loaded.")
except Exception as e:
    model = None
    encoders = {}
    feature_columns = None
    logger.warning("Model not loaded: %s", str(e))

# initialize sqlite
def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS submissions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  submitted_at TEXT,
                  input_json TEXT,
                  prediction INTEGER,
                  proba REAL)''')
    conn.commit()
    conn.close()

init_db()

@app.route("/")
def home():
    return render_template("index.html", prediction_text=None)

@app.route("/upload_dataset", methods=["POST"])
def upload_dataset():
    f = request.files.get("dataset")
    if not f:
        flash("No file selected", "warning")
        return redirect(url_for("home"))
    filename = secure_filename(f.filename)
    ext = filename.rsplit('.', 1)[-1].lower()
    if ext not in ALLOWED_EXT:
        flash("Only CSV allowed", "danger")
        return redirect(url_for("home"))
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f.save(save_path)
    flash("Dataset uploaded successfully", "success")
    return redirect(url_for("home"))

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        flash("Model not loaded. Run train_model.py or provide model.json in Serialized Trained Model.", "danger")
        return render_template("index.html", prediction_text=None)

    # gather input
    raw = {}
    for k in request.form:
        raw[k] = request.form.get(k)

    try:
        input_df = build_input_dataframe(raw, encoders, feature_columns)
        # predict_proba if available
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)[0][1]  # probability of class 1 (defaulter)
        else:
            # fallback: use predict only
            proba = None
        pred = int(model.predict(input_df)[0])
    except Exception as e:
        logger.exception("Prediction error")
        flash(f"Prediction failed: {str(e)}", "danger")
        return render_template("index.html", prediction_text=None)

    # Save submission
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("INSERT INTO submissions (submitted_at, input_json, prediction, proba) VALUES (?, ?, ?, ?)",
              (datetime.utcnow().isoformat(), json.dumps(raw), pred, float(proba) if proba is not None else None))
    conn.commit()
    conn.close()

    label = "Defaulter" if pred == 1 else "Not Defaulter"
    prob_label = f"{proba*100:.2f}%" if proba is not None else "N/A"
    prob_value = float(proba) if proba is not None else 0.0

    return render_template("result.html", prediction_text=label, prob=prob_label, prob_value=prob_value, raw=raw)

@app.route("/download_report/<int:sub_id>")
def download_report(sub_id):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT submitted_at, input_json, prediction, proba FROM submissions WHERE id=?", (sub_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        flash("Submission not found", "warning")
        return redirect(url_for("home"))

    submitted_at, input_json, prediction, proba = row
    data = json.loads(input_json)

    # create PDF in-memory
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    p.setFont("Helvetica-Bold", 14)
    p.drawString(40, 750, "Loan Default Prediction Report")
    p.setFont("Helvetica", 10)
    p.drawString(40, 730, f"Submitted: {submitted_at}")
    p.drawString(40, 710, f"Prediction: {'Defaulter' if prediction==1 else 'Not Defaulter'}")
    p.drawString(40, 695, f"Probability (class 1): {proba if proba is not None else 'N/A'}")

    y = 670
    p.setFont("Helvetica-Bold", 11)
    p.drawString(40, y, "Inputs:")
    p.setFont("Helvetica", 10)
    y -= 18
    for k, v in data.items():
        p.drawString(50, y, f"{k}: {v}")
        y -= 14
        if y < 60:
            p.showPage()
            y = 750

    p.showPage()
    p.save()
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name="loan_report.pdf", mimetype='application/pdf')

@app.route("/history")
def history():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT id, submitted_at, prediction, proba FROM submissions ORDER BY id DESC LIMIT 50")
    rows = c.fetchall()
    conn.close()
    return render_template("history.html", rows=rows)

def open_browser():
    """Open the browser after a short delay to allow the server to start."""
    url = "http://127.0.0.1:5000"
    webbrowser.open(url)

if __name__ == "__main__":
    # Open browser automatically after 1.5 seconds
    Timer(1.5, open_browser).start()
    print("\n" + "="*50)
    print("Flask app starting...")
    print("The app will open automatically in your default browser")
    print("If it doesn't, manually visit: http://127.0.0.1:5000")
    print("="*50 + "\n")
    app.run(debug=True, host='127.0.0.1', port=5000)
