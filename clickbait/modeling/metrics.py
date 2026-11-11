"""낚시성(1)을 양성으로 둔 이진 분류 지표."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from clickbait.schema import CLICKBAIT, NORMAL

# schema.Prediction.label 과 같은 기준: 0.5 이상이면 낚시성.
THRESHOLD = 0.5


def predict_labels(prob_clickbait: np.ndarray) -> np.ndarray:
    return (prob_clickbait >= THRESHOLD).astype(int)


def compute_metrics(y_true: np.ndarray, prob_clickbait: np.ndarray) -> dict[str, object]:
    y_pred = predict_labels(prob_clickbait)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, pos_label=CLICKBAIT, average="binary", zero_division=0
    )
    return {
        "n": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        # 행: 실제 [일반, 낚시성], 열: 예측 [일반, 낚시성]
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[NORMAL, CLICKBAIT]).tolist(),
    }
