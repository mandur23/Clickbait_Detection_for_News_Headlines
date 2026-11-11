"""서버가 쓰는 판별기 인터페이스. 학습된 모델이 없으면 가짜 판별기로 화면을 먼저 만든다."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from clickbait.config import MODEL_META_FILENAME
from clickbait.schema import Prediction


class Predictor(Protocol):
    name: str
    is_mock: bool
    accepts_body: bool

    def predict(self, title: str, body: str | None = None) -> Prediction:
        """정리가 끝난 제목(과 본문)을 받아 판별 결과를 돌려준다."""
        ...


class MockPredictor:
    """모델이 나오기 전 화면 개발용. 문장부호 몇 개로 확률을 흉내 낼 뿐 실제 판별이 아니다."""

    name = "mock"
    is_mock = True
    accepts_body = False

    _CUES = ("?", "!", "...", "충격", "경악", "결국", "알고 보니", "알고보니", "이유는", "정체")

    def predict(self, title: str, body: str | None = None) -> Prediction:
        hits = sum(1 for cue in self._CUES if cue in title)
        return Prediction(prob_clickbait=min(0.2 + 0.25 * hits, 0.95))


def has_trained_model(model_dir: Path) -> bool:
    return (model_dir / MODEL_META_FILENAME).is_file()


def load_predictor(model_dir: Path) -> Predictor:
    """저장된 모델이 있으면 실제 판별기를, 없으면 가짜 판별기를 돌려준다."""
    if not has_trained_model(model_dir):
        return MockPredictor()
    # torch·transformers는 무겁다. 가짜 판별기만 쓸 때는 불러오지 않는다.
    from clickbait.modeling.inference import TransformerPredictor

    return TransformerPredictor.load(model_dir)
