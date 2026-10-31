from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "models"

st.set_page_config(page_title="Multiple Disease Prediction", page_icon="🩺", layout="wide")
st.title("🩺 Multiple Disease Prediction System")
st.caption("B.Tech Minor Project • Machine Learning • Educational demonstration only")

st.warning(
   "This application is not a medical device and does not diagnose disease. "
   "Do not use its output to make healthcare decisions. Consult a qualified clinician."
)

st.markdown("""
This prototype provides separate machine-learning demonstrations for:
- **Diabetes** — tabular health measurements
- **Heart disease** — clinical attributes
- **Parkinson's disease** — voice-derived measurements

Models must first be trained using the public datasets listed in `README.md`.
""")

MODEL_FILES = {
    "Diabetes": "diabetes_model.joblib",
    "Heart Disease": "heart_model.joblib",
    "Parkinson's Disease": "parkinsons_model.joblib",
}

def render_module(name, bundle_path):
    if not bundle_path.exists():
        st.error(f"The {name} model is not available yet.")
        st.info("Follow README.md: download the matching dataset into `datasets/`, then run `python train_models.py`.")
        return

    bundle = joblib.load(bundle_path)
    features = bundle["features"]
    model = bundle["model"]
    metrics = bundle.get("metrics", {})

    st.subheader(f"{name} prediction demo")
    if metrics:
        c1, c2, c3 = st.columns(3)
        c1.metric("Test accuracy", f"{metrics.get('accuracy', 0):.1%}")
        c2.metric("Test precision", f"{metrics.get('precision', 0):.1%}")
        c3.metric("Test recall", f"{metrics.get('recall', 0):.1%}")
        st.caption("These are dataset-specific holdout metrics, not evidence of clinical validity.")

    labels = bundle.get("labels", {"0": "Class 0", "1": "Class 1"})
    with st.form(f"prediction_form_{name}"):
        st.write("Enter values matching the dataset's feature definitions.")
        inputs = {}
        cols = st.columns(2)
        for i, feature in enumerate(features):
            with cols[i % 2]:
                default = float(bundle.get("defaults", {}).get(feature, 0))
                # Feature values are collected as numeric because these teaching datasets encode categories numerically.
                inputs[feature] = st.number_input(feature, value=default, format="%.4f")
        submitted = st.form_submit_button("Run model demonstration")

    if submitted:
        row = pd.DataFrame([[inputs[f] for f in features]], columns=features)
        pred = int(model.predict(row)[0])
        label = labels.get(str(pred), f"Class {pred}")
        st.subheader("Model output")
        st.info(f"Predicted dataset class: **{label}**")
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(row)[0]
            st.write("Model class scores (not calibrated clinical probabilities):")
            st.bar_chart(pd.DataFrame(
                {"score": probabilities},
                index=[labels.get(str(c), str(c)) for c in model.classes_],
            ))
        st.caption("This result is for academic demonstration only. It is not a diagnosis or personal risk estimate.")


tabs = st.tabs(list(MODEL_FILES.keys()))
for tab, (name, filename) in zip(tabs, MODEL_FILES.items()):
    with tab:
        render_module(name, MODEL_DIR / filename)
