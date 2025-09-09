"""
Bank term-deposit classification using scikit-learn Pipelines.

This script loads the UCI bank marketing dataset, builds preprocessing + model
pipelines, evaluates a few classifiers, and saves predictions and a trained
Random Forest model. Cleaned up from an earlier exploratory script with added
type hints and documentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from joblib import dump


# -----------------------------
# Configuration constants
# -----------------------------
DATA_URL: str = "https://raw.githubusercontent.com/rafiag/DTI2020/main/data/bank.csv"
RANDOM_STATE: int = 1
PREDICTION_CSV: str = "deposit_prediction.csv"
MODEL_PATH: str = "bank_deposit_classification.joblib"


@dataclass
class Evaluation:
    """Holds common classification metrics and artifacts for plotting.

    Attributes:
        acc: Accuracy.
        prec: Precision (binary, positive class=1).
        rec: Recall (binary, positive class=1).
        f1: F1-score.
        kappa: Cohen's kappa score.
        auc: ROC AUC using predicted probabilities.
        fpr: False positive rates for ROC curve.
        tpr: True positive rates for ROC curve.
        cm: Confusion matrix (2x2 array).
    """

    acc: float
    prec: float
    rec: float
    f1: float
    kappa: float
    auc: float
    fpr: np.ndarray
    tpr: np.ndarray
    cm: np.ndarray


def load_data(url: str = DATA_URL) -> pd.DataFrame:
    """Load the bank marketing dataset and drop non-predictive columns.

    Args:
        url: CSV URL to load.

    Returns:
        A DataFrame with the original columns except for "duration".
    """
    df = pd.read_csv(url)
    # The original notebook drops 'duration' as it's not known before the call.
    return df.drop(columns=["duration"]) if "duration" in df.columns else df


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split raw DataFrame into features X and target y.

    The target column is 'deposit'. y is encoded to 1 (yes) and 0 (no).
    """
    if "deposit" not in df.columns:
        raise ValueError("Expected 'deposit' column in dataframe")
    y = df["deposit"].map({"yes": 1, "no": 0})
    X = df.drop(columns=["deposit"])  # keep raw columns; preprocessing handled by Pipeline
    return X, y


def build_preprocessor(numeric: Iterable[str], categorical: Iterable[str]) -> ColumnTransformer:
    """Create a ColumnTransformer for numeric scaling and categorical one-hot encoding."""
    numeric = list(numeric)
    categorical = list(categorical)
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ],
        remainder="drop",
    )


def build_pipelines(preprocessor: ColumnTransformer) -> Mapping[str, Pipeline]:
    """Build model pipelines keyed by a short model name."""
    models: Mapping[str, ClassifierMixin] = {
        "dt": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "rf": RandomForestClassifier(random_state=RANDOM_STATE),
        "nb": GaussianNB(),
        "knn": KNeighborsClassifier(n_neighbors=5),
    }
    return {
        name: Pipeline(steps=[("pre", preprocessor), ("model", est)]) for name, est in models.items()
    }


def evaluate(estimator: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Evaluation:
    """Compute common classification metrics for a trained estimator.

    Note: Requires estimator to expose predict_proba for ROC/AUC.
    """
    y_pred = estimator.predict(X_test)

    acc = metrics.accuracy_score(y_test, y_pred)
    prec = metrics.precision_score(y_test, y_pred)
    rec = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    kappa = metrics.cohen_kappa_score(y_test, y_pred)

    # Predict probabilities for positive class 1 if available
    if hasattr(estimator, "predict_proba"):
        y_proba = estimator.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = metrics.roc_curve(y_test, y_proba)
        auc = metrics.roc_auc_score(y_test, y_proba)
    else:
        # Fallback using decision function if available; else use scores as ranks
        if hasattr(estimator, "decision_function"):
            scores = estimator.decision_function(X_test)
        else:
            scores = y_pred  # not ideal; enables ROC shape at least
        fpr, tpr, _ = metrics.roc_curve(y_test, scores)
        auc = metrics.roc_auc_score(y_test, scores)

    cm = metrics.confusion_matrix(y_test, y_pred)
    return Evaluation(acc, prec, rec, f1, kappa, auc, fpr, tpr, cm)


def plot_comparison(evals: Mapping[str, Evaluation]) -> None:
    """Plot bar metrics and ROC curves for multiple evaluations."""
    if not evals:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle("Model Comparison", fontsize=16, fontweight="bold")

    labels = ["Accuracy", "Precision", "Recall", "F1", "Kappa"]
    bar_width = 0.18

    # Metrics bars
    positions = np.arange(len(labels))
    for i, (name, ev) in enumerate(evals.items()):
        scores = [ev.acc, ev.prec, ev.rec, ev.f1, ev.kappa]
        ax1.bar(positions + i * bar_width, scores, width=bar_width, label=name.upper())
    ax1.set_xticks(positions + bar_width * (len(evals) - 1) / 2)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0, 1)
    ax1.set_title("Evaluation Metrics", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Score", fontweight="bold")
    ax1.legend()

    # ROC curves
    for name, ev in evals.items():
        ax2.plot(ev.fpr, ev.tpr, label=f"{name.upper()}, AUC={ev.auc:.4f}")
    ax2.set_xlabel("False Positive Rate", fontweight="bold")
    ax2.set_ylabel("True Positive Rate", fontweight="bold")
    ax2.set_title("ROC Curve", fontsize=14, fontweight="bold")
    ax2.legend(loc=4)
    fig.tight_layout()
    plt.show()


def save_predictions_csv(model: Pipeline, raw_df: pd.DataFrame, out_path: str = PREDICTION_CSV) -> pd.DataFrame:
    """Predict deposit on the full dataset and save as a CSV next to raw columns.

    Returns the dataframe that was written for convenience.
    """
    X_all = raw_df.drop(columns=["deposit"]) if "deposit" in raw_df.columns else raw_df.copy()
    preds = model.predict(X_all)
    # Map numeric predictions back to yes/no: 1 -> yes, 0 -> no
    pred_labels = np.where(preds == 1, "yes", "no")
    out_df = raw_df.copy()
    out_df["deposit_prediction"] = pred_labels
    out_df.to_csv(out_path, index=False)
    return out_df


def main() -> None:
    """Train models on the bank dataset, evaluate, save RF model and predictions."""
    df = load_data(DATA_URL)

    # Define schema
    numeric_cols: List[str] = ["age", "balance", "day", "campaign", "pdays", "previous"]
    categorical_cols: List[str] = [
        "job",
        "marital",
        "education",
        "default",
        "housing",
        "loan",
        "contact",
        "month",
        "poutcome",
    ]

    # Features/target
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=RANDOM_STATE
    )

    pre = build_preprocessor(numeric_cols, categorical_cols)
    pipes = build_pipelines(pre)

    evaluations: Dict[str, Evaluation] = {}
    for name, pipe in pipes.items():
        pipe.fit(X_train, y_train)
        evaluations[name] = evaluate(pipe, X_test, y_test)

    plot_comparison(evaluations)

    rf_model = pipes["rf"]
    dump(rf_model, MODEL_PATH)
    preview_df = save_predictions_csv(rf_model, df, PREDICTION_CSV)

    rf_eval = evaluations["rf"]
    print(
        f"Random Forest — Acc: {rf_eval.acc:.3f}, Prec: {rf_eval.prec:.3f}, Rec: {rf_eval.rec:.3f}, "
        f"F1: {rf_eval.f1:.3f}, AUC: {rf_eval.auc:.3f}"
    )
    print(preview_df.head(10))


if __name__ == "__main__":
    main()
