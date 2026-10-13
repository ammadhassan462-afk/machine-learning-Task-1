Project Title: Dual-Domain Predictive Analytics Platform
Overview
This project is an end-to-end machine learning web application designed to solve two distinct classification problems: predicting Student Academic Performance and assessing Heart Disease Risk. Built using modular Python pipelines and scikit-learn, the application provides interactive user interfaces through both Gradio (for rapid ML prototyping) and Flask (for a custom web deployment).

Key Features & Datasets
Student Performance Predictor:

Analyzes behavioral and academic metrics (such as weekly study hours, attendance percentage, previous exam scores, sleep hours, tutoring sessions, and motivation levels) to predict whether a student is likely to pass or fail.

Heart Failure Predictor:

Evaluates clinical and physiological indicators (including age, resting blood pressure, cholesterol levels, fasting blood sugar, maximum heart rate, and chest pain types) to estimate cardiovascular health risks.

Dual Frontend Interfaces:

Gradio UI: Features a clean tabbed layout (gr.TabbedInterface) allowing users to switch seamlessly between the student and medical prediction tools in a single interactive dashboard.

Flask Web Application: Serves a lightweight, traditional web server routing inputs to the trained scikit-learn pipelines for production-ready deployment.

Tech Stack & Architecture
Core Machine Learning: Python, Scikit-Learn (Pipeline, ColumnTransformer, StandardScaler, OneHotEncoder, LogisticRegression), Pandas, NumPy.

Model Persistence: Joblib for optimized pipeline serialization and loading.

User Interfaces:

Gradio for interactive component-based UI and immediate inference testing.

Flask & Jinja for custom server-side web routing and HTML rendering.

Project Structure
Plaintext
ML_Task1_LogisticRegression/
│
├── student_model.pkl        # Serialized student performance pipeline
├── heart_model.pkl          # Serialized heart disease risk pipeline
├── gradio_app.py            # Multi-tab Gradio interactive dashboard
├── requirements.txt         # Project dependencies
└── flask_app/               # Flask backend deployment folder
    ├── app.py               # Flask application routes
    └── templates/           # HTML frontend templates
