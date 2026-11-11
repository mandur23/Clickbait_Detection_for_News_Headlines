"""경로, 입력 제한, 학습 초기값. 계획서 5.2절의 초기 설정을 그대로 둔다."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from clickbait.schema import InputMode

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SAMPLE_RAW_DIR = DATA_DIR / "sample_aihub"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

DEFAULT_MODEL_DIR = MODELS_DIR / "klue-bert"
MODEL_META_FILENAME = "clickbait_meta.json"

# 정제 단계에서 이 범위를 벗어난 제목은 버린다(글자 수, 정리 후 기준).
MIN_TITLE_CHARS = 5
MAX_TITLE_CHARS = 120

# 서버 입력 제한. 넘으면 모델에 보내지 않고 안내 메시지를 돌려준다.
MAX_INPUT_TITLE_CHARS = 200
MAX_INPUT_BODY_CHARS = 20_000

SEED = 42


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hub_id: str
    display_name: str


MODEL_SPECS: dict[str, ModelSpec] = {
    "klue-bert": ModelSpec("klue-bert", "klue/bert-base", "KLUE-BERT"),
    "kobert": ModelSpec("kobert", "skt/kobert-base-v1", "KoBERT"),
}


@dataclass(frozen=True)
class TrainConfig:
    model_key: str = "klue-bert"
    input_mode: InputMode = "title"
    max_length: int = 64
    batch_size: int = 32
    learning_rate: float = 2e-5
    epochs: int = 3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    class_weights: bool = False
    seed: int = SEED


# 제목+본문 모드는 본문이 들어가므로 토큰 길이를 늘린다.
TITLE_BODY_MAX_LENGTH = 256
