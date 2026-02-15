from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

from data.schemas import NormalizedComplaint


@dataclass(frozen=True)
class BaselineEvaluation:
    baseline_name: str
    test_metrics: dict[str, Any]
    hard_test_metrics: dict[str, Any]
    test_predictions: list[str]
    hard_test_predictions: list[str]


def _prepare_xy(records: list[NormalizedComplaint]) -> tuple[list[str], list[str]]:
    x = [record.complaint_text_clean for record in records]
    y = [record.normalized_category for record in records]
    return x, y


def _compute_ece(probabilities: np.ndarray, y_true: list[str], label_order: list[str], bins: int = 10) -> float:
    if probabilities.size == 0:
        return 0.0
    label_to_index = {label: index for index, label in enumerate(label_order)}
    true_indices = np.array([label_to_index[label] for label in y_true])
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correctness = (predictions == true_indices).astype(float)

    boundaries = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        mask = (confidences >= start) & (confidences < end if end < 1.0 else confidences <= end)
        if not np.any(mask):
            continue
        bucket_conf = confidences[mask].mean()
        bucket_acc = correctness[mask].mean()
        ece += float(mask.mean()) * abs(bucket_acc - bucket_conf)
    return round(float(ece), 6)


def _extract_top_confusions(matrix: np.ndarray, labels: list[str], top_n: int = 10) -> list[dict[str, Any]]:
    pairs: list[tuple[int, str, str]] = []
    for row_idx, actual in enumerate(labels):
        for col_idx, predicted in enumerate(labels):
            if row_idx == col_idx:
                continue
            count = int(matrix[row_idx, col_idx])
            if count > 0:
                pairs.append((count, actual, predicted))
    pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [
        {"count": count, "actual": actual, "predicted": predicted}
        for count, actual, predicted in pairs[:top_n]
    ]


def _evaluate_predictions(
    y_true: list[str],
    y_pred: list[str],
    label_order: list[str],
    y_prob: np.ndarray | None = None,
) -> dict[str, Any]:
    if not y_true:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "per_class": {},
            "confusion_matrix": [],
            "labels": label_order,
            "top_confusions": [],
            "ece": None,
        }

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=label_order, average="macro", zero_division=0)
    precision, recall, f1_values, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=label_order,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=label_order)
    per_class = {
        label: {
            "precision": round(float(precision[index]), 6),
            "recall": round(float(recall[index]), 6),
            "f1": round(float(f1_values[index]), 6),
            "support": int(support[index]),
        }
        for index, label in enumerate(label_order)
    }

    return {
        "accuracy": round(float(accuracy), 6),
        "macro_f1": round(float(macro_f1), 6),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "labels": label_order,
        "top_confusions": _extract_top_confusions(matrix=matrix, labels=label_order, top_n=12),
        "ece": _compute_ece(y_prob, y_true, label_order) if y_prob is not None else None,
    }


def _majority_baseline_predictions(train_labels: list[str], target_len: int) -> list[str]:
    majority = Counter(train_labels).most_common(1)[0][0]
    return [majority for _ in range(target_len)]


def run_baselines(
    *,
    train_records: list[NormalizedComplaint],
    test_records: list[NormalizedComplaint],
    hard_test_records: list[NormalizedComplaint],
    config: dict[str, Any],
    mode: str,
    seed: int,
) -> dict[str, BaselineEvaluation]:
    train_x, train_y = _prepare_xy(train_records)
    test_x, test_y = _prepare_xy(test_records)
    hard_x, hard_y = _prepare_xy(hard_test_records)
    label_order = sorted(set(train_y) | set(test_y) | set(hard_y))

    if len(set(train_y)) < 2:
        fallback_test_pred = _majority_baseline_predictions(train_y, len(test_y))
        fallback_hard_pred = _majority_baseline_predictions(train_y, len(hard_y))
        fallback_eval = BaselineEvaluation(
            baseline_name="baseline_tfidf_linear",
            test_metrics=_evaluate_predictions(test_y, fallback_test_pred, label_order=label_order),
            hard_test_metrics=_evaluate_predictions(hard_y, fallback_hard_pred, label_order=label_order),
            test_predictions=fallback_test_pred,
            hard_test_predictions=fallback_hard_pred,
        )
        return {"baseline_tfidf_linear": fallback_eval}

    baseline_results: dict[str, BaselineEvaluation] = {}

    tfidf_linear = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=int(config["baseline1"]["min_df"]),
                    max_features=int(config["baseline1"]["max_features"]),
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=int(config["baseline1"]["max_iter"]),
                    random_state=seed,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    tfidf_linear.fit(train_x, train_y)
    test_pred = tfidf_linear.predict(test_x).tolist()
    hard_pred = tfidf_linear.predict(hard_x).tolist() if hard_x else []
    test_prob = tfidf_linear.predict_proba(test_x) if test_x else None
    hard_prob = tfidf_linear.predict_proba(hard_x) if hard_x else None

    baseline_results["baseline_tfidf_linear"] = BaselineEvaluation(
        baseline_name="baseline_tfidf_linear",
        test_metrics=_evaluate_predictions(test_y, test_pred, label_order=label_order, y_prob=test_prob),
        hard_test_metrics=_evaluate_predictions(hard_y, hard_pred, label_order=label_order, y_prob=hard_prob),
        test_predictions=test_pred,
        hard_test_predictions=hard_pred,
    )

    run_baseline2 = bool(config["run_baseline2_full"]) if mode == "FULL" else bool(config["run_baseline2_smoke"])
    if run_baseline2:
        n_components = int(config["baseline2"]["svd_components"])
        n_components = max(2, min(n_components, int(config["baseline2"]["max_features"]) - 1))
        embedding_linear = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        ngram_range=(1, 2),
                        min_df=int(config["baseline2"]["min_df"]),
                        max_features=int(config["baseline2"]["max_features"]),
                        sublinear_tf=True,
                    ),
                ),
                (
                    "svd",
                    TruncatedSVD(
                        n_components=n_components,
                        random_state=seed,
                    ),
                ),
                ("norm", Normalizer(copy=False)),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=int(config["baseline2"]["max_iter"]),
                        random_state=seed,
                        class_weight="balanced",
                    ),
                ),
            ]
        )
        embedding_linear.fit(train_x, train_y)
        test_pred_2 = embedding_linear.predict(test_x).tolist()
        hard_pred_2 = embedding_linear.predict(hard_x).tolist() if hard_x else []
        test_prob_2 = embedding_linear.predict_proba(test_x) if test_x else None
        hard_prob_2 = embedding_linear.predict_proba(hard_x) if hard_x else None

        baseline_results["baseline_sentence_embedding_linear"] = BaselineEvaluation(
            baseline_name="baseline_sentence_embedding_linear",
            test_metrics=_evaluate_predictions(test_y, test_pred_2, label_order=label_order, y_prob=test_prob_2),
            hard_test_metrics=_evaluate_predictions(
                hard_y, hard_pred_2, label_order=label_order, y_prob=hard_prob_2
            ),
            test_predictions=test_pred_2,
            hard_test_predictions=hard_pred_2,
        )

    return baseline_results
