"""기준선(joblib)과 BERT 모델 폴더를 같은 경로로 평가하고 보고서를 쓴다."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from clickbait.config import MODEL_META_FILENAME  # noqa: E402
from clickbait.modeling.baseline import BASELINE_FILENAME, BaselineModel  # noqa: E402
from clickbait.modeling.metrics import compute_metrics, predict_labels  # noqa: E402
from clickbait.schema import CLICKBAIT, LABEL_NAMES  # noqa: E402

METRICS_FILENAME = "metrics.json"


class Scorer(Protocol):
    name: str
    accepts_body: bool

    def predict_proba(self, titles: Sequence[str], bodies: Sequence[str] | None = None) -> np.ndarray: ...


def load_scorer(model_dir: Path, device: str = "cpu") -> Scorer:
    if (model_dir / MODEL_META_FILENAME).is_file():
        from clickbait.modeling.inference import TransformerPredictor

        return TransformerPredictor.load(model_dir, device=device)
    if (model_dir / BASELINE_FILENAME).is_file():
        return BaselineModel.load(model_dir)
    raise FileNotFoundError(f"{model_dir} 에 {MODEL_META_FILENAME} 도 {BASELINE_FILENAME} 도 없다.")


@dataclass(frozen=True)
class EvalResult:
    metrics: dict[str, object]
    # title, true, pred, prob_clickbait, category, source, error. 가장 확신한 오답이 먼저 온다.
    misclassified: pd.DataFrame


def evaluate(scorer: Scorer, df: pd.DataFrame, bodies: dict[str, str] | None = None) -> EvalResult:
    pair = [bodies.get(news_id, "") for news_id in df["news_id"]] if scorer.accepts_body and bodies is not None else None
    prob = scorer.predict_proba(df["title"].tolist(), pair)
    y_true = df["label"].astype(int).to_numpy()
    y_pred = predict_labels(prob)
    wrong = y_true != y_pred
    misclassified = pd.DataFrame(
        {
            "title": df["title"][wrong].to_numpy(),
            "true": [LABEL_NAMES[y] for y in y_true[wrong]],
            "pred": [LABEL_NAMES[y] for y in y_pred[wrong]],
            "prob_clickbait": np.round(prob[wrong], 4),
            "category": df["category"][wrong].to_numpy(),
            "source": df["source"][wrong].to_numpy(),
            "error": np.where(y_pred[wrong] == CLICKBAIT, "FP", "FN"),
            # 정답 반대쪽으로 얼마나 확신했는가.
            "confidence": np.abs(prob[wrong] - 0.5),
        }
    )
    misclassified = (
        misclassified.sort_values("confidence", ascending=False, kind="stable").drop(columns="confidence").reset_index(drop=True)
    )
    metrics = {**compute_metrics(y_true, prob), "by_source": _by_source(df["source"], y_true == y_pred, prob)}
    return EvalResult(metrics=metrics, misclassified=misclassified)


def _by_source(source: pd.Series, correct: np.ndarray, prob: np.ndarray) -> dict[str, dict[str, object]]:
    """원본 하위 묶음(Clickbait_Auto, Clickbait_Direct, NonClickbait_Auto)별 정답률.

    한 묶음은 라벨이 하나뿐이라, 낚시성 묶음의 정답률은 재현율이고 일반 묶음의 정답률은 특이도다.
    어느 묶음에서 틀리는지가 제목만으로 풀 수 있는 문제인지를 알려 준다.
    """
    frame = pd.DataFrame({"source": source.to_numpy(), "correct": correct, "prob": prob})
    return {
        str(name): {
            "n": int(len(group)),
            "accuracy": round(float(group["correct"].mean()), 4),
            "mean_prob_clickbait": round(float(group["prob"].mean()), 4),
        }
        for name, group in frame.groupby("source", sort=True)
    }


def _plot_confusion(matrix: list[list[int]], title: str, path: Path) -> None:
    # 한글 글꼴이 없는 환경(Colab)에서도 깨지지 않게 축 이름은 영어로 둔다.
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    data = np.asarray(matrix)
    ax.imshow(data, cmap="Blues")
    ax.set_xticks([0, 1], labels=list(LABEL_NAMES))
    ax.set_yticks([0, 1], labels=list(LABEL_NAMES))
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(data[i, j]), ha="center", va="center", color="white" if data[i, j] > data.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(result: EvalResult, out_dir: Path, info: dict[str, object]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {**info, **result.metrics}
    (out_dir / METRICS_FILENAME).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _plot_confusion(result.metrics["confusion_matrix"], str(info.get("name", "")), out_dir / "confusion_matrix.png")  # type: ignore[arg-type]
    result.misclassified.to_csv(out_dir / "misclassified.csv", index=False, encoding="utf-8-sig")


def comparison_table(reports_dir: Path) -> str:
    """reports/*/metrics.json 을 모아 마크다운 표를 만든다. F1 이 높은 순."""
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(reports_dir.glob(f"*/{METRICS_FILENAME}"))]
    rows.sort(key=lambda r: float(r["f1"]), reverse=True)
    lines = [
        "| 보고서 | 모델 | 입력 | 평가 데이터 | n | 정확도 | 정밀도 | 재현율 | F1 |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['model']} | {r['input_mode']} | {r['eval_csv']} | {r['n']} | "
            f"{r['accuracy']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} |"
        )
    return "\n".join(lines) + "\n"
