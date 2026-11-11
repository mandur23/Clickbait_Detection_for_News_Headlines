from __future__ import annotations

from dataclasses import dataclass, field

from clickbait.schema import Prediction


@dataclass
class FakePredictor:
    """받은 입력을 기록하고 정해 둔 확률을 돌려준다."""

    prob_clickbait: float = 0.73
    accepts_body: bool = False
    fail: bool = False
    name: str = "fake-model"
    is_mock: bool = False
    calls: list[tuple[str, str | None]] = field(default_factory=list)

    def predict(self, title: str, body: str | None = None) -> Prediction:
        self.calls.append((title, body))
        if self.fail:
            raise RuntimeError("모델 내부 오류")
        return Prediction(prob_clickbait=self.prob_clickbait)
