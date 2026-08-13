from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parent
METADATA_PATH = ROOT / "model" / "model_metadata.json"
DEFAULT_TEST_DATA = ROOT / "test_data.csv"


st.set_page_config(
    page_title="Breast Cancer Classifier",
    layout="wide",
)


@st.cache_resource
def load_metadata() -> dict:
    with METADATA_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_resource
def load_models(model_paths: dict[str, str]) -> dict[str, object]:
    models = {}
    for model_name, relative_path in model_paths.items():
        models[model_name] = joblib.load(ROOT / relative_path)
    return models


def normalize_target(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int)

    cleaned = series.astype(str).str.strip().str.lower()
    mapping = {
        "benign": 0,
        "b": 0,
        "0": 0,
        "malignant": 1,
        "m": 1,
        "1": 1,
    }
    return cleaned.map(mapping)


def find_target_column(dataframe: pd.DataFrame) -> str | None:
    for candidate in ["diagnosis", "target", "label", "class"]:
        if candidate in dataframe.columns:
            return candidate
    return None


def score_positive_class(model: object, features: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(features)
        return (scores - scores.min()) / (scores.max() - scores.min())
    return model.predict(features)


def evaluate_predictions(model: object, features: pd.DataFrame, target: pd.Series) -> dict:
    predictions = model.predict(features)
    scores = score_positive_class(model, features)
    return {
        "Accuracy": accuracy_score(target, predictions),
        "AUC": roc_auc_score(target, scores),
        "Precision": precision_score(target, predictions, zero_division=0),
        "Recall": recall_score(target, predictions, zero_division=0),
        "F1": f1_score(target, predictions, zero_division=0),
        "MCC": matthews_corrcoef(target, predictions),
        "Confusion Matrix": confusion_matrix(target, predictions, labels=[0, 1]),
        "Classification Report": classification_report(
            target,
            predictions,
            target_names=["Benign", "Malignant"],
            digits=4,
            zero_division=0,
            output_dict=True,
        ),
        "Predictions": predictions,
        "Scores": scores,
    }


def prepare_uploaded_data(dataframe: pd.DataFrame, feature_names: list[str]) -> tuple[pd.DataFrame, pd.Series | None]:
    missing_features = [name for name in feature_names if name not in dataframe.columns]
    if missing_features:
        st.error("The uploaded CSV is missing required feature columns.")
        st.dataframe(pd.DataFrame({"Missing feature": missing_features}), use_container_width=True)
        st.stop()

    features = dataframe[feature_names].copy()
    target_column = find_target_column(dataframe)
    target = None
    if target_column:
        target = normalize_target(dataframe[target_column])
        if target.isna().any():
            st.error(
                f"The target column `{target_column}` contains values that could not be mapped to Benign/Malignant."
            )
            st.stop()

    return features, target


def show_metric_cards(metrics: dict) -> None:
    columns = st.columns(6)
    ordered_metrics = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    for column, metric_name in zip(columns, ordered_metrics):
        column.metric(metric_name, f"{metrics[metric_name]:.4f}")


def show_confusion_matrix(matrix: np.ndarray) -> None:
    confusion = pd.DataFrame(
        matrix,
        index=["Actual Benign", "Actual Malignant"],
        columns=["Predicted Benign", "Predicted Malignant"],
    )
    st.dataframe(confusion, use_container_width=True)


def main() -> None:
    metadata = load_metadata()
    models = load_models(metadata["models"])
    feature_names = metadata["feature_names"]

    st.title("Breast Cancer Classification")
    st.caption("Interactive evaluation app for Logistic Regression, Decision Tree, kNN, Naive Bayes, Random Forest, and SVM.")

    with st.sidebar:
        st.header("Dataset")
        uploaded_file = st.file_uploader("Upload test CSV", type=["csv"])
        selected_model = st.selectbox("Select model", list(models.keys()))
        threshold = st.slider(
            "Malignant probability threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.50,
            step=0.05,
        )

    if uploaded_file is not None:
        raw_data = pd.read_csv(uploaded_file)
    else:
        raw_data = pd.read_csv(DEFAULT_TEST_DATA)

    features, target = prepare_uploaded_data(raw_data, feature_names)

    overview_tab, evaluate_tab, predict_tab = st.tabs(
        ["Model Comparison", "Uploaded Test Data", "Single Prediction"]
    )

    with overview_tab:
        st.subheader("Dataset summary")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Rows in current test data", f"{len(features):,}")
        col_b.metric("Input features", f"{len(feature_names):,}")
        col_c.metric("Positive class", "Malignant")

        st.subheader("Training-time model comparison")
        metric_rows = []
        for model_name, values in metadata["metrics"].items():
            metric_rows.append(
                {
                    "ML Model Name": model_name,
                    "Accuracy": values["Accuracy"],
                    "AUC": values["AUC"],
                    "Precision": values["Precision"],
                    "Recall": values["Recall"],
                    "F1": values["F1"],
                    "MCC": values["MCC"],
                }
            )
        st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)
        st.info(f"Overall winner on the held-out split: {metadata['winner']}")

    with evaluate_tab:
        st.subheader(selected_model)
        model = models[selected_model]
        scores = score_positive_class(model, features)
        predictions = (scores >= threshold).astype(int)
        result_frame = raw_data.copy()
        result_frame["predicted_diagnosis"] = np.where(predictions == 1, "Malignant", "Benign")
        result_frame["malignant_probability"] = scores

        if target is not None:
            metrics = {
                "Accuracy": accuracy_score(target, predictions),
                "AUC": roc_auc_score(target, scores),
                "Precision": precision_score(target, predictions, zero_division=0),
                "Recall": recall_score(target, predictions, zero_division=0),
                "F1": f1_score(target, predictions, zero_division=0),
                "MCC": matthews_corrcoef(target, predictions),
                "Confusion Matrix": confusion_matrix(target, predictions, labels=[0, 1]),
                "Classification Report": classification_report(
                    target,
                    predictions,
                    target_names=["Benign", "Malignant"],
                    digits=4,
                    zero_division=0,
                    output_dict=True,
                ),
            }
            show_metric_cards(metrics)
            left, right = st.columns([1, 2])
            with left:
                st.subheader("Confusion matrix")
                show_confusion_matrix(metrics["Confusion Matrix"])
            with right:
                st.subheader("Classification report")
                report = pd.DataFrame(metrics["Classification Report"]).transpose()
                st.dataframe(report, use_container_width=True)
        else:
            st.warning("No target column was found, so the app is showing predictions only.")

        st.subheader("Prediction preview")
        preview_columns = ["predicted_diagnosis", "malignant_probability"] + feature_names[:6]
        st.dataframe(result_frame[preview_columns].head(25), use_container_width=True)

    with predict_tab:
        st.subheader("Predict one record")
        selected_single_model = st.selectbox("Prediction model", list(models.keys()), key="single_model")
        inputs = {}
        columns = st.columns(3)
        for index, feature_name in enumerate(feature_names):
            values = features[feature_name]
            inputs[feature_name] = columns[index % 3].number_input(
                feature_name,
                min_value=float(values.min()),
                max_value=float(values.max()),
                value=float(values.median()),
                step=max(float(values.std() / 10), 0.001),
            )

        single_features = pd.DataFrame([inputs], columns=feature_names)
        single_model = models[selected_single_model]
        probability = float(score_positive_class(single_model, single_features)[0])
        diagnosis = "Malignant" if probability >= threshold else "Benign"
        st.metric("Predicted diagnosis", diagnosis)
        st.metric("Malignant probability", f"{probability:.4f}")


if __name__ == "__main__":
    main()
