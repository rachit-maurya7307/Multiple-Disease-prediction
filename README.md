# Multiple Disease Prediction System — B.Tech Minor Project

## Overview
An educational ML application with separate modules for diabetes, heart disease and Parkinson's disease. The UI is built with Streamlit; model training uses scikit-learn pipelines. It is a learning prototype, not a medical device.

## Requirements
- Python 3.10+ recommended
- pip
- VS Code or another editor

## Setup (Windows)
Open Command Prompt/PowerShell in this project folder:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Download the datasets
Download the datasets from these sources and place them in `datasets/` using the filenames below. A dataset can be a plain CSV or a ZIP archive containing its CSV; the training script reads either format without deleting or extracting your original file.

1. **Diabetes (`datasets/diabetes.csv`)**: use a Pima Indians Diabetes dataset CSV that has numeric feature columns and a target column named `Outcome` (or `target`). A Kaggle search page is provided in the project notes: https://www.kaggle.com/search?q=Pima+Indians+Diabetes+Dataset
2. **Heart disease (`datasets/heart.csv`)**: use the processed UCI Cleveland Heart Disease data or a compatible CSV. Target should be named `target`, `num`, `output`, or `HeartDisease`. UCI dataset: https://archive.ics.uci.edu/dataset/45/heart+disease
3. **Parkinson's (`datasets/parkinsons.csv`)**: use the UCI Parkinsons CSV with numeric voice feature columns and a `status` target column. UCI dataset: https://archive.ics.uci.edu/dataset/174/parkinsons

Dataset formats can vary by source. If the target column or feature layout differs, adjust `train_models.py` to match your downloaded file. Keep the original dataset attribution/license requirements and do not upload personal or private medical records.

## Train the models
After placing all three CSVs in `datasets/`, run:

```powershell
python train_models.py
```

The script saves model bundles in `models/` and evaluation JSON files in `reports/`. If it reports a missing target column, inspect the CSV headers and update the target candidate list in `train_models.py`.

## Launch the web app
```powershell
streamlit run app.py
```
Streamlit will open the application in your browser. If the command is not recognized, try:
```powershell
python -m streamlit run app.py
```

## Important interpretation limits
- This is a classroom demonstration, not a diagnostic tool.
- Test metrics depend on dataset selection and split; do not present them as clinical accuracy.
- The standard Parkinson's dataset contains multiple voice recordings per participant. A random row split can leak participant-specific information between train and test sets. For a research-quality evaluation, use participant-grouped validation.
- Never make healthcare decisions based on this app. Consult a qualified clinician.

## Suggested demo
1. Show the home page and disclaimer.
2. Select each disease module.
3. Explain the feature inputs and target labels.
4. Explain train/test split, preprocessing pipeline, Random Forest and metrics.
5. Show the report/PPT and discuss limitations.
