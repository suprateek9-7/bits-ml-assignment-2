# Breast Cancer Classification Streamlit App

## Problem statement

The goal of this project is to build an end-to-end machine learning classification application that predicts whether a breast tumor sample is benign or malignant from numeric diagnostic measurements. The project trains multiple classifiers on the same dataset, compares them with standard evaluation metrics, and exposes the results through an interactive Streamlit application.

## Dataset description

- Dataset: Breast Cancer Wisconsin Diagnostic
- Source: UCI Machine Learning Repository, available through scikit-learn
- Problem type: Binary classification
- Number of instances: 569
- Number of input features: 30
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
| Logistic Regression | 0.9737 | 0.9960 | 0.9756 | 0.9524 | 0.9639 | 0.9433 |
| Decision Tree | 0.8772 | 0.9654 | 0.9118 | 0.7381 | 0.8158 | 0.7343 |
| kNN | 0.9561 | 0.9825 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Random Forest (Ensemble) | 0.9737 | 0.9970 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |
| Support Vector Machine | 0.9825 | 0.9947 | 1.0000 | 0.9524 | 0.9756 | 0.9626 |

## Observations about model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong baseline with high AUC and balanced precision/recall, showing that the standardized feature space is close to linearly separable. |
| Decision Tree | Useful and interpretable, but slightly weaker than the best models because a single tree is more sensitive to split choices. |
| kNN | Performed well after scaling because similar biopsy profiles tend to share the same diagnosis, though it is less transparent than tree/logistic models. |
| Naive Bayes | Competitive despite its strong feature-independence assumption, indicating that the dataset has highly informative individual measurements. |
| Random Forest (Ensemble) | Most stable tree-based model because averaging many trees reduced overfitting and improved generalization. |
| Support Vector Machine | Best performer on this split, with perfect or near-perfect discrimination after feature scaling. |
| Overall Winner | Support Vector Machine achieved the strongest aggregate result on the held-out test set, especially when considering accuracy, F1, and MCC together. |

## How to run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python model/train_models.py
streamlit run app.py
```

After running Streamlit, open:

```text
http://localhost:8501
```

## Streamlit app features

- CSV upload option for test data
- Model selection dropdown for individual model evaluation
- Comparison table for all models
- Accuracy, AUC, precision, recall, F1 score, and MCC
- Confusion matrix and classification report
- Single-record prediction form

```
