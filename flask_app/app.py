"""Flask UI for both models.
Run:  python app.py      then open http://127.0.0.1:5000
"""
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, abort, render_template, request

app = Flask(__name__)

HERE = Path(__file__).resolve().parent


def load_model(name):
    """Look for the .pkl next to this file first, then in the project folder above it."""
    for folder in (HERE, HERE.parent):
        path = folder / name
        if path.exists():
            return joblib.load(path)
    raise FileNotFoundError(f"{name} not found. Run the notebooks first (they create it).")


LEVELS = ["Low", "Medium", "High"]

# Each dataset is described once; the same form.html renders both.
CONFIG = {
    "student": {
        "title": "Student Performance Predictor",
        "model": load_model("student_model.pkl"),
        "positive": "PASS (score >= 70)",
        "negative": "FAIL (score < 70)",
        "fields": [
            {"name": "Hours_Studied", "label": "Hours Studied (per week)", "default": 20},
            {"name": "Attendance", "label": "Attendance (%)", "default": 80},
            {"name": "Previous_Scores", "label": "Previous Scores", "default": 70},
            {"name": "Sleep_Hours", "label": "Sleep Hours", "default": 7},
            {"name": "Tutoring_Sessions", "label": "Tutoring Sessions", "default": 1},
            {"name": "Physical_Activity", "label": "Physical Activity (hrs/week)", "default": 3},
            {"name": "Parental_Involvement", "label": "Parental Involvement", "options": LEVELS},
            {"name": "Motivation_Level", "label": "Motivation Level", "options": LEVELS},
        ],
    },
    "heart": {
        "title": "Heart Failure Predictor",
        "model": load_model("heart_model.pkl"),
        "positive": "HEART DISEASE LIKELY",
        "negative": "HEALTHY (low risk)",
        "fields": [
            {"name": "Age", "label": "Age", "default": 50},
            {"name": "RestingBP", "label": "Resting BP (mm Hg)", "default": 130},
            {"name": "Cholesterol", "label": "Cholesterol (mg/dl)", "default": 220},
            {"name": "FastingBS", "label": "Fasting Blood Sugar > 120 (1 = Yes, 0 = No)", "options": ["0", "1"]},
            {"name": "MaxHR", "label": "Max Heart Rate", "default": 140},
            {"name": "Oldpeak", "label": "Oldpeak (ST depression)", "default": 1.0},
            {"name": "Sex", "label": "Sex", "options": ["M", "F"]},
            {"name": "ChestPainType", "label": "Chest Pain Type", "options": ["ATA", "NAP", "ASY", "TA"]},
            {"name": "ExerciseAngina", "label": "Exercise Induced Angina", "options": ["Y", "N"]},
            {"name": "ST_Slope", "label": "ST Slope", "options": ["Up", "Flat", "Down"]},
        ],
    },
}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/<name>", methods=["GET", "POST"])
def predict(name):
    cfg = CONFIG.get(name)
    if cfg is None:
        abort(404)

    result, prob, values = None, None, {}
    if request.method == "POST":
        row = {}
        for f in cfg["fields"]:
            raw = request.form[f["name"]]
            values[f["name"]] = raw
            if f.get("options"):
                row[f["name"]] = int(raw) if raw.isdigit() else raw   # dropdown value
            else:
                row[f["name"]] = float(raw)                           # number input
        df = pd.DataFrame([row])
        pred = cfg["model"].predict(df)[0]
        prob = cfg["model"].predict_proba(df)[0][1]
        result = cfg["positive"] if pred == 1 else cfg["negative"]

    return render_template("form.html", cfg=cfg, result=result, prob=prob, values=values)


if __name__ == "__main__":
    app.run(debug=True)
