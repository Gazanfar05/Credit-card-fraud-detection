from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
DATA_FILE = DATA_DIR / "fraud_transactions.csv"
MODEL_FILE = ARTIFACT_DIR / "fraud_model.joblib"
SUMMARY_FILE = ARTIFACT_DIR / "model_summary.json"
RANDOM_STATE = 42
FEATURES = ["amount", "time", "f1", "f2", "f3", "f4", "f5", "f6"]


def generate_dataset(path: Path) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    x, y = make_classification(
        n_samples=6000,
        n_features=6,
        n_informative=4,
        n_redundant=1,
        n_repeated=0,
        weights=[0.985, 0.015],
        class_sep=1.85,
        flip_y=0.01,
        random_state=RANDOM_STATE,
    )

    frame = pd.DataFrame(x, columns=[f"f{i}" for i in range(1, 7)])
    frame["amount"] = np.clip(
        np.abs(rng.normal(loc=85 + y * 130 + frame["f1"] * 18, scale=35)),
        0.25,
        2500,
    )
    frame["time"] = rng.uniform(0, 172800, size=len(frame)).round(2)
    frame = frame[["amount", "time", "f1", "f2", "f3", "f4", "f5", "f6"]]
    frame["is_fraud"] = y.astype(int)

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def load_or_create_dataset() -> pd.DataFrame:
    if DATA_FILE.exists():
        return pd.read_csv(DATA_FILE)
    return generate_dataset(DATA_FILE)


def make_classifier() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=260,
        max_depth=None,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def evaluate_model(model, x_test, y_test, label: str) -> dict:
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    roc_auc = roc_auc_score(y_test, probabilities)
    precision_curve, recall_curve, thresholds = precision_recall_curve(y_test, probabilities)
    f1_scores = 2 * precision_curve * recall_curve / np.clip(precision_curve + recall_curve, 1e-9, None)
    best_index = int(np.nanargmax(f1_scores[:-1])) if len(thresholds) else 0
    threshold = float(thresholds[best_index]) if len(thresholds) else 0.5
    threshold_predictions = (probabilities >= threshold).astype(int)
    confusion = confusion_matrix(y_test, threshold_predictions).tolist()

    return {
        "strategy": label,
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "roc_auc": round(float(roc_auc), 4),
        "threshold": round(threshold, 4),
        "confusion_matrix": confusion,
    }


def train_models(frame: pd.DataFrame) -> tuple[dict, dict]:
    x = frame[FEATURES]
    y = frame["is_fraud"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    strategies: dict[str, tuple[pd.DataFrame, pd.Series, dict]] = {
        "baseline": (x_train, y_train, {"class_weight": "balanced_subsample"}),
        "undersample": RandomUnderSampler(random_state=RANDOM_STATE).fit_resample(x_train, y_train),
        "oversample": RandomOverSampler(random_state=RANDOM_STATE).fit_resample(x_train, y_train),
        "smote": SMOTE(random_state=RANDOM_STATE, k_neighbors=3).fit_resample(x_train, y_train),
    }

    summary: dict[str, dict] = {}
    fitted_models: dict[str, RandomForestClassifier] = {}

    for name, payload in strategies.items():
        if name == "baseline":
            x_fit, y_fit, params = payload
        else:
            x_fit, y_fit = payload
            params = {}

        model = make_classifier()
        if params:
            model.set_params(**params)
        model.fit(x_fit, y_fit)
        summary[name] = evaluate_model(model, x_test, y_test, name)
        fitted_models[name] = model

    best_strategy = max(summary.values(), key=lambda item: (item["roc_auc"], item["recall"]))["strategy"]
    best_model = fitted_models[best_strategy]
    best_summary = summary[best_strategy]

    feature_importances = sorted(
        zip(FEATURES, best_model.feature_importances_),
        key=lambda item: item[1],
        reverse=True,
    )[:5]

    bundle = {
        "model": best_model,
        "features": FEATURES,
        "strategy": best_strategy,
        "threshold": best_summary["threshold"],
        "summary": summary,
    }

    overview = {
        "dataset": {
            "rows": int(len(frame)),
            "fraud_count": int(frame["is_fraud"].sum()),
            "fraud_rate": round(float(frame["is_fraud"].mean()), 4),
            "feature_count": len(FEATURES),
        },
        "best_strategy": best_strategy,
        "best_metrics": best_summary,
        "all_metrics": summary,
        "feature_importance": [{"feature": name, "score": round(float(score), 4)} for name, score in feature_importances],
    }

    return bundle, overview


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_or_create_dataset()
    bundle, overview = train_models(frame)
    joblib.dump(bundle, MODEL_FILE)
    with SUMMARY_FILE.open("w", encoding="utf-8") as handle:
        json.dump(overview, handle, indent=2)
    print(json.dumps(overview, indent=2))


if __name__ == "__main__":
    main()
