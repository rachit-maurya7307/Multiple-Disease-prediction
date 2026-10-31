"""Train separate teaching models from CSV datasets placed in ./datasets.

Expected files:
  diabetes.csv   - Pima-style dataset with Outcome target
  heart.csv      - Cleveland heart dataset with binary target column
  parkinsons.csv - UCI Parkinsons dataset with status target
"""
from pathlib import Path
import zipfile
import json
import urllib.request
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "datasets"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)
DATASET_SOURCES = {
    "diabetes": {
        "file": "diabetes.csv",
        "url": "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv",
    },
    "heart": {
        "file": "heart.csv",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
        "columns": [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
            "exang", "oldpeak", "slope", "ca", "thal", "target",
        ],
    },
    "parkinsons": {
        "file": "parkinsons.csv",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data",
    },
}


def read_dataset_file(name, path):
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            csv_members = [
                member for member in archive.infolist()
                if not member.is_dir() and Path(member.filename).suffix.lower() == ".csv"
            ]
            expected_name = DATASET_SOURCES[name]["file"].lower()
            matching_members = [
                member for member in csv_members
                if Path(member.filename).name.lower() == expected_name
            ]
            if matching_members:
                csv_members = matching_members
            if len(csv_members) != 1:
                names = [member.filename for member in csv_members]
                raise ValueError(
                    f"{path} is a ZIP archive and must contain one matching CSV file. "
                    f"CSV files found: {names}"
                )
            with archive.open(csv_members[0]) as csv_file:
                return _read_csv(name, csv_file)
    return _read_csv(name, path)


def _read_csv(name, source):
    if name == "heart":
        return pd.read_csv(
            source, header=None, names=DATASET_SOURCES[name]["columns"], na_values=["?"]
        )
    return pd.read_csv(source, encoding="latin", on_bad_lines="skip")


def load_csv(name):
    source = DATASET_SOURCES[name]
    path = DATA_DIR / source["file"]
    aliases = [path]
    if name == "heart":
        aliases.append(DATA_DIR / "heartdisease.csv")

    for candidate in aliases:
        if candidate.exists():
            try:
                return read_dataset_file(name, candidate)
            except (OSError, UnicodeDecodeError, ValueError, zipfile.BadZipFile) as exc:
                raise ValueError(
                    f"Could not read {name} dataset at {candidate}: {exc}"
                ) from exc

    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {name} dataset from {source['url']} ...")
    with urllib.request.urlopen(source["url"], timeout=30) as response:
        data = response.read()
    path.write_bytes(data)

    try:
        return read_dataset_file(name, path)
    except (OSError, UnicodeDecodeError, ValueError, zipfile.BadZipFile) as exc:
        raise ValueError(
            f"The downloaded {name} dataset could not be read as a CSV or ZIP containing a CSV: {exc}"
        ) from exc


def choose_target(df, candidates):
    lower = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    raise ValueError(f"Could not find target column among {candidates}. Columns found: {list(df.columns)}")


def train_one(name, df, target_candidates, drop_columns=None, label_map=None):
    drop_columns = drop_columns or []
    target = choose_target(df, target_candidates)
    clean = df.copy()
    clean.columns = [str(c).strip() for c in clean.columns]
    target = next(c for c in clean.columns if c.lower() == str(target).lower())
    clean = clean.drop(columns=[c for c in drop_columns if c in clean.columns], errors="ignore")
    y = pd.to_numeric(clean[target], errors="coerce")
    if name == "heart" and y.max() > 1:
        y = (y > 0).astype(int)
    X = clean.drop(columns=[target])
    X = X.select_dtypes(include=[np.number]).copy()
    valid = y.notna()
    X, y = X.loc[valid], y.loc[valid].astype(int)
    if X.empty or y.nunique() < 2:
        raise ValueError(f"{name}: need numeric features and at least two target classes.")
    X = X.dropna(axis=1, how="all")
    features = list(X.columns)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=250, class_weight="balanced", random_state=42
        ))
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "classification_report": classification_report(y_test, pred, zero_division=0)
    }
    labels = label_map or {str(int(v)): f"Dataset class {int(v)}" for v in sorted(y.unique())}
    bundle = {
        "model": model,
        "features": features,
        "metrics": metrics,
        "labels": labels,
        "defaults": {f: float(X_train[f].median()) for f in features}
    }
    out = MODEL_DIR / f"{name}_model.joblib"
    joblib.dump(bundle, out)
    with open(REPORT_DIR / f"{name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n{name}: saved {out}")
    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))
    print(metrics["classification_report"])
    return bundle


def main():
    diabetes = load_csv("diabetes")
    train_one("diabetes", diabetes, ["Outcome", "target", "class"],
              drop_columns=["PatientID", "id"],
              label_map={"0": "Negative dataset class", "1": "Positive dataset class"})

    heart = load_csv("heart")
    train_one("heart", heart, ["target", "HeartDisease", "Heart Disease", "num", "output"],
              drop_columns=["id", "name", "patient_id"],
              label_map={"0": "No disease label in dataset", "1": "Disease label in dataset"})

    parkinsons = load_csv("parkinsons")
    train_one("parkinsons", parkinsons, ["status", "target", "class"],
              drop_columns=["name", "id", "subject#"],
              label_map={"0": "Healthy label in dataset", "1": "Parkinson's label in dataset"})

    print("\nAll models trained. Open the app with: streamlit run app.py")
    print("IMPORTANT: Metrics are for a simple holdout split and are not clinical validation.")


if __name__ == "__main__":
    main()
