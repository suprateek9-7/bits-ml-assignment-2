from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
RANDOM_STATE = 42


def load_assignment_dataset() -> tuple[pd.DataFrame, pd.Series, dict]:
    dataset = load_breast_cancer()
    features = pd.DataFrame(dataset.data, columns=dataset.feature_names)

    # scikit-learn stores malignant as 0 and benign as 1. For the classifier
    # app, 1 means the positive clinical class: malignant.
    target = pd.Series(np.where(dataset.target == 0, 1, 0), name="diagnosis")

    metadata = {
        "name": "Breast Cancer Wisconsin Diagnostic",
        "source": "UCI Machine Learning Repository, available through scikit-learn",
        "samples": int(features.shape[0]),
        "features": int(features.shape[1]),
        "problem_type": "Binary classification",
        "target_mapping": {"0": "Benign", "1": "Malignant"},
        "positive_class": "Malignant",
        "feature_names": list(features.columns),
    }
    return features, target, metadata


def build_models() -> dict[str, object]:
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=5000,
                        solver="liblinear",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=4,
            random_state=RANDOM_STATE,
        ),
        "kNN": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=7)),
            ]
        ),
        "Naive Bayes": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "Support Vector Machine": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    CalibratedClassifierCV(
                        estimator=SVC(
                            kernel="rbf",
                            C=2.0,
                            gamma="scale",
                            random_state=RANDOM_STATE,
                        ),
                        method="sigmoid",
                        cv=5,
                        ensemble=False,
                    ),
                ),
            ]
        ),
    }


def positive_class_scores(model: object, X_test: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]
    if hasattr(model, "decision_function"):
        raw_scores = model.decision_function(X_test)
        return (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())
    return model.predict(X_test)


def evaluate_model(model: object, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    scores = positive_class_scores(model, X_test)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    return {
        "Accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "AUC": round(float(roc_auc_score(y_test, scores)), 4),
        "Precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "F1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "MCC": round(float(matthews_corrcoef(y_test, y_pred)), 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=["Benign", "Malignant"],
            digits=4,
            zero_division=0,
            output_dict=True,
        ),
    }


def slugify_model_name(name: str) -> str:
    return (
        name.lower()
        .replace("(", "")
        .replace(")", "")
        .replace(" ", "_")
        .replace("-", "_")
    )


def write_metrics_csv(metrics: dict[str, dict]) -> None:
    rows = []
    for model_name, values in metrics.items():
        rows.append(
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
    pd.DataFrame(rows).to_csv(ROOT / "model_metrics.csv", index=False)


def write_readme(metrics: dict[str, dict], metadata: dict, winner: str) -> None:
    metric_rows = "\n".join(
        "| {model} | {Accuracy:.4f} | {AUC:.4f} | {Precision:.4f} | {Recall:.4f} | {F1:.4f} | {MCC:.4f} |".format(
            model=model_name,
            **values,
        )
        for model_name, values in metrics.items()
    )

    observations = {
        "Logistic Regression": "Strong baseline with high AUC and balanced precision/recall, showing that the standardized feature space is close to linearly separable.",
        "Decision Tree": "Useful and interpretable, but slightly weaker than the best models because a single tree is more sensitive to split choices.",
        "kNN": "Performed well after scaling because similar biopsy profiles tend to share the same diagnosis, though it is less transparent than tree/logistic models.",
        "Naive Bayes": "Competitive despite its strong feature-independence assumption, indicating that the dataset has highly informative individual measurements.",
        "Random Forest (Ensemble)": "Most stable tree-based model because averaging many trees reduced overfitting and improved generalization.",
        "Support Vector Machine": "Best performer on this split, with perfect or near-perfect discrimination after feature scaling.",
    }

    observation_rows = "\n".join(
        f"| {model_name} | {observations[model_name]} |" for model_name in metrics
    )

    readme = f"""# Breast Cancer Classification Streamlit App

## Problem statement

The goal of this project is to build an end-to-end machine learning classification application that predicts whether a breast tumor sample is benign or malignant from numeric diagnostic measurements. The project trains multiple classifiers on the same dataset, compares them with standard evaluation metrics, and exposes the results through an interactive Streamlit application.

## Dataset description

- Dataset: {metadata["name"]}
- Source: {metadata["source"]}
- Problem type: {metadata["problem_type"]}
- Number of instances: {metadata["samples"]}
- Number of input features: {metadata["features"]}
- Target classes: 0 = Benign, 1 = Malignant
- Positive class used for precision, recall, F1, AUC, and MCC: Malignant

The dataset satisfies the assignment constraints because it has more than 500 records and more than 12 features. The bundled `test_data.csv` file contains the held-out test split used for app demonstration and evaluation.

## GitHub repository link

Add your GitHub repository link here after uploading this folder:

`https://github.com/<your-user>/<your-repo>`

## Live Streamlit app link

Add your deployed Streamlit Community Cloud link here after deployment:

`https://<your-app-name>.streamlit.app`

## Models used

The assignment PDF text says six ML models are required, while the visible list/table names five. This solution implements the five listed models and adds Support Vector Machine as the sixth standard classifier.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
{metric_rows}

## Observations about model performance

| ML Model Name | Observation about model performance |
|---|---|
{observation_rows}
| Overall Winner | {winner} achieved the strongest aggregate result on the held-out test set, especially when considering accuracy, F1, and MCC together. |

## How to run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python model/train_models.py
streamlit run app.py
```

## Streamlit app features

- CSV upload option for test data
- Model selection dropdown for individual model evaluation
- Comparison table for all models
- Accuracy, AUC, precision, recall, F1 score, and MCC
- Confusion matrix and classification report
- Single-record prediction form

## Repository structure

```text
project-folder/
|-- app.py
|-- requirements.txt
|-- README.md
|-- test_data.csv
|-- model/
|   |-- train_models.py
|   |-- model_metadata.json
|   |-- *.joblib
|-- model_metrics.csv
```
"""
    (ROOT / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)

    X, y, metadata = load_assignment_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    test_data = X_test.copy()
    test_data["diagnosis"] = y_test.map({0: "Benign", 1: "Malignant"}).values
    test_data.to_csv(ROOT / "test_data.csv", index=False)

    models = build_models()
    metrics = {}
    artifact_map = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        metrics[model_name] = evaluate_model(model, X_test, y_test)
        artifact_name = f"{slugify_model_name(model_name)}.joblib"
        artifact_map[model_name] = f"model/{artifact_name}"
        joblib.dump(model, MODEL_DIR / artifact_name)

    metric_priority = lambda item: np.mean(
        [
            item[1]["Accuracy"],
            item[1]["AUC"],
            item[1]["Precision"],
            item[1]["Recall"],
            item[1]["F1"],
            item[1]["MCC"],
        ]
    )
    winner = max(metrics.items(), key=metric_priority)[0]

    metadata.update(
        {
            "random_state": RANDOM_STATE,
            "test_size": 0.2,
            "models": artifact_map,
            "metrics": metrics,
            "winner": winner,
        }
    )
    (MODEL_DIR / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    write_metrics_csv(metrics)
    write_readme(metrics, metadata, winner)

    print("Training complete.")
    print(f"Winner: {winner}")
    print(pd.read_csv(ROOT / "model_metrics.csv").to_string(index=False))


if __name__ == "__main__":
    main()
