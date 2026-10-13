"""Gradio UI: one tab per dataset (Student Performance, Heart Failure).
Run:  python gradio_app.py
"""
from pathlib import Path

import gradio as gr
import joblib
import pandas as pd

HERE = Path(__file__).resolve().parent


def load_model(name):
  path = HERE / name
  if not path.exists():
    raise FileNotFoundError(
        f"{name} not found. Run the notebooks first (they create it)."
    )
  model = joblib.load(path)

  # --- Compatibility Patch for Scikit-Learn Version Mismatch ---
  # Injects missing attributes so older pickled pipelines run smoothly on newer sklearn versions
  if hasattr(model, "named_steps"):
    for _, step in model.named_steps.items():
      if hasattr(step, "transformers_"):
        for _, trans, _ in step.transformers_:
          if hasattr(trans, "categories_") and not hasattr(
              trans, "_infrequent_enabled"
          ):
            trans._infrequent_enabled = False
  return model


student_model = load_model("student_model.pkl")
heart_model = load_model("heart_model.pkl")


def predict_student(
    hours, attendance, prev, sleep, tutoring, activity, parental, motivation
):
  df = pd.DataFrame([{
      "Hours_Studied": hours,
      "Attendance": attendance,
      "Previous_Scores": prev,
      "Sleep_Hours": sleep,
      "Tutoring_Sessions": tutoring,
      "Physical_Activity": activity,
      "Parental_Involvement": parental,
      "Motivation_Level": motivation,
  }])
  pred = student_model.predict(df)[0]
  prob = student_model.predict_proba(df)[0][1]
  label = "PASS (score >= 70)" if pred == 1 else "FAIL (score < 70)"
  return f"{label}\nProbability of passing: {prob:.1%}"


def predict_heart(age, bp, chol, fbs, maxhr, oldpeak, sex, cp, angina, slope):
  df = pd.DataFrame([{
      "Age": age,
      "RestingBP": bp,
      "Cholesterol": chol,
      "FastingBS": int(fbs),
      "MaxHR": maxhr,
      "Oldpeak": oldpeak,
      "Sex": sex,
      "ChestPainType": cp,
      "ExerciseAngina": angina,
      "ST_Slope": slope,
  }])
  pred = heart_model.predict(df)[0]
  prob = heart_model.predict_proba(df)[0][1]
  label = "HEART DISEASE LIKELY" if pred == 1 else "HEALTHY (low risk)"
  return f"{label}\nProbability of heart disease: {prob:.1%}"


student_ui = gr.Interface(
    fn=predict_student,
    inputs=[
        gr.Slider(1, 44, value=20, label="Hours Studied (per week)"),
        gr.Slider(60, 100, value=80, label="Attendance (%)"),
        gr.Slider(50, 100, value=70, label="Previous Scores"),
        gr.Slider(4, 10, value=7, step=1, label="Sleep Hours"),
        gr.Slider(0, 8, value=1, step=1, label="Tutoring Sessions"),
        gr.Slider(0, 6, value=3, step=1, label="Physical Activity (hrs/week)"),
        gr.Dropdown(
            ["Low", "Medium", "High"],
            value="Medium",
            label="Parental Involvement",
        ),
        gr.Dropdown(
            ["Low", "Medium", "High"], value="Medium", label="Motivation Level"
        ),
    ],
    outputs=gr.Textbox(label="Result"),
    title="Student Performance Predictor",
    description="Logistic Regression: predicts whether a student will pass.",
)

heart_ui = gr.Interface(
    fn=predict_heart,
    inputs=[
        gr.Number(value=50, label="Age"),
        gr.Number(value=130, label="Resting BP (mm Hg)"),
        gr.Number(value=220, label="Cholesterol (mg/dl)"),
        gr.Radio(
            [0, 1],
            value=0,
            label="Fasting Blood Sugar > 120 (1 = Yes, 0 = No)",
        ),
        gr.Number(value=140, label="Max Heart Rate"),
        gr.Number(value=1.0, label="Oldpeak (ST depression)"),
        gr.Dropdown(["M", "F"], value="M", label="Sex"),
        gr.Dropdown(
            ["ATA", "NAP", "ASY", "TA"], value="ASY", label="Chest Pain Type"
        ),
        gr.Dropdown(["Y", "N"], value="N", label="Exercise Induced Angina"),
        gr.Dropdown(["Up", "Flat", "Down"], value="Flat", label="ST Slope"),
    ],
    outputs=gr.Textbox(label="Result"),
    title="Heart Failure Predictor",
    description="Logistic Regression: predicts heart disease risk.",
)

demo = gr.TabbedInterface(
    [student_ui, heart_ui], ["Student Performance", "Heart Failure"]
)

if __name__ == "__main__":
  demo.launch()