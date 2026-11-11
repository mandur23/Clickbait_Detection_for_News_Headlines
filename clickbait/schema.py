"""프로젝트 전체가 공유하는 라벨 규칙과 판별 결과 형식."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# AI Hub 원본은 clickbaitClass 0이 낚시성이다. 이 프로젝트 안에서는 낚시성을
# 양성(1)으로 두어 정밀도·재현율·F1이 낚시성 기준으로 계산되게 한다.
# 원본 값은 clickbait/data/aihub.py 에서만 변환한다.
NORMAL = 0
CLICKBAIT = 1
LABEL_NAMES: tuple[str, str] = ("normal", "clickbait")
LABEL_NAMES_KO: tuple[str, str] = ("일반", "낚시성")

LabelName = Literal["normal", "clickbait"]

# 정제된 CSV(data/processed/*.csv)의 열 순서.
HEADLINE_COLUMNS: tuple[str, ...] = (
    "news_id",
    "title",
    "label",
    "category",
    "part",
    "source",
)

InputMode = Literal["title", "title_body"]


@dataclass(frozen=True)
class Prediction:
    """낚시성 확률 하나만 저장한다. 나머지 값은 여기서 계산하므로 서로 어긋날 수 없다."""

    prob_clickbait: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.prob_clickbait <= 1.0:
            raise ValueError(f"확률은 0~1이어야 한다: {self.prob_clickbait}")

    @property
    def prob_normal(self) -> float:
        return 1.0 - self.prob_clickbait

    @property
    def label(self) -> LabelName:
        return "clickbait" if self.prob_clickbait >= 0.5 else "normal"

    def to_json(self) -> dict[str, object]:
        return {
            "label": self.label,
            "prob_clickbait": round(self.prob_clickbait, 4),
            "prob_normal": round(self.prob_normal, 4),
        }
