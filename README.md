# ML Task 1: LinearRegression on 2 Datasets

Datasets: Student Performance Factors (Pass/Fail) and Heart Failure Prediction.

## Folder
```
ML_Task1_LogisticRegression/
├── student_performance.ipynb   <- notebook 1
├── heart_failure.ipynb         <- notebook 2
├── gradio_app.py               <- Gradio UI (both models, one tab each)
├── flask_app/
│   ├── app.py                  <- Flask UI
│   └── templates/ (index.html, form.html)
├── student_model.pkl, heart_model.pkl
├── StudentPerformanceFactors.csv   <- ADD THIS (your dataset)
└── heart.csv                       <- ADD THIS (your dataset)
```

## Steps
1. Copy your two CSV files into this folder (names must match the ones above).
2. Open the folder in VS Code and select the same Python environment you used for your notebooks.
3. Install packages once:  `pip install -r requirements.txt`
4. Open each notebook and click **Run All**. Screenshot the outputs. This also re-creates the two `.pkl` files.
5. Gradio:  `python gradio_app.py`  then open http://127.0.0.1:7860
6. Flask:   `cd flask_app`  then  `python app.py`  then open http://127.0.0.1:5000

## If you see an error about OneHotEncoder / `_infrequent_enabled`
The `.pkl` files were made with a different scikit-learn version than the one running the apps.
Run both notebooks again (Run All) in the same environment you use for the apps.
