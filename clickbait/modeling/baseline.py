"""비교 기준선: 단어 TF-IDF(1~2그램) + 로지스틱 회귀. 제목만 본다."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from clickbait.config import SEED
from clickbait.schema import CLICKBAIT

BASELINE_FILENAME = "baseline.joblib"
# 기본 토큰 규칙은 한 글자 단어와 문장부호를 버린다. "?", "!", "..." 는 낚시성 단서라 따로 토큰으로 남긴다.
_TOKEN_PATTERN = r"\w+|[^\w\s]+"


class BaselineModel:
    name = "baseline"
    accepts_body = False

    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline

    @classmethod
    def train(cls, titles: Sequence[str], labels: Sequence[int], *, min_df: int = 2, class_weights: bool = False) -> BaselineModel:
        pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        analyzer="word",
                        token_pattern=_TOKEN_PATTERN,
                        ngram_range=(1, 2),
                        sublinear_tf=True,
                        min_df=min_df,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced" if class_weights else None,
                        random_state=SEED,
                    ),
                ),
            ]
        )
        pipeline.fit(list(titles), list(labels))
        return cls(pipeline)

    def predict_proba(self, titles: Sequence[str], bodies: Sequence[str] | None = None) -> np.ndarray:
        column = list(self.pipeline.classes_).index(CLICKBAIT)
        return self.pipeline.predict_proba(list(titles))[:, column]

    def save(self, model_dir: Path) -> Path:
        model_dir.mkdir(parents=True, exist_ok=True)
        path = model_dir / BASELINE_FILENAME
        joblib.dump(self.pipeline, path)
        return path

    @classmethod
    def load(cls, model_dir: Path) -> BaselineModel:
        return cls(joblib.load(model_dir / BASELINE_FILENAME))
