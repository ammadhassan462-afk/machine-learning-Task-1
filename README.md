# Dual-Domain Predictive Analytics Platform (Logistic Regression)

An end-to-end machine learning web application built with Python and Scikit-Learn that solves two distinct classification problems: predicting **Student Academic Performance** and assessing **Heart Failure Risk**. The project features dual interactive frontends powered by **Gradio** and **Flask**.

---

## 🚀 Key Features

* **Student Performance Predictor**: Evaluates weekly study hours, attendance, previous scores, sleep hours, tutoring sessions, and motivational levels to predict academic success.
* **Heart Disease Predictor**: Analyzes clinical markers like age, cholesterol, resting blood pressure, max heart rate, and chest pain types to estimate cardiovascular risk.
* **Dual Frontend Interfaces**:
  * **Gradio Dashboard**: A clean, multi-tab interface (`gr.TabbedInterface`) allowing instant testing of both models.
  * **Flask Web App**: A lightweight, server-rendered web application for deployment and custom UI rendering.

---

## 🛠️ Tech Stack

* **Machine Learning**: Python, Scikit-Learn (`Pipeline`, `ColumnTransformer`, `StandardScaler`, `OneHotEncoder`, `LogisticRegression`), Pandas, NumPy.
* **Model Persistence**: Joblib.
* **Web Frameworks**: Gradio, Flask.

---

## 📂 Project Structure

```text
ML_Task1_LogisticRegression/
│
├── student_model.pkl        # Serialized student performance pipeline
├── heart_model.pkl          # Serialized heart disease risk pipeline
├── gradio_app.py            # Multi-tab Gradio interactive app
├── requirements.txt         # Project dependencies
└── flask_app/               # Flask backend deployment folder
    ├── app.py               # Flask application routes
    └── templates/           # HTML templates for Flask UI
