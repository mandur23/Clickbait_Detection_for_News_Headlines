"""모델 종류별 토크나이저·분류 모델 불러오기, 그리고 학습과 추론이 함께 쓰는 입력 인코딩.

모델 키(config.MODEL_SPECS)마다 불러오는 방법을 FAMILIES 표 한 곳에 둔다. 허브 이름과
저장한 로컬 폴더 모두 같은 함수로 불러온다.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from clickbait.modeling.kobert_tokenizer import KoBERTTokenizer

NUM_LABELS = 2


@dataclass(frozen=True)
class ModelFamily:
    load_tokenizer: Callable[[str], PreTrainedTokenizerBase]


def _sequence_classifier(source: str) -> PreTrainedModel:
    return AutoModelForSequenceClassification.from_pretrained(source, num_labels=NUM_LABELS)


FAMILIES: dict[str, ModelFamily] = {
    "klue-bert": ModelFamily(load_tokenizer=AutoTokenizer.from_pretrained),
    # KoBERT 는 허브의 토크나이저 설정이 XLNet 형식이라 BERT 순서([CLS] A [SEP] B [SEP])로
    # 감싸는 전용 토크나이저를 쓴다. 모델 본체는 평범한 BERT 다.
    "kobert": ModelFamily(load_tokenizer=KoBERTTokenizer.from_pretrained),
}


def load_pretrained(model_key: str, source: str) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """source 는 허브 이름이거나 save_pretrained 로 저장한 폴더다."""
    family = FAMILIES[model_key]
    return family.load_tokenizer(source), _sequence_classifier(source)


def encode(
    tokenizer: PreTrainedTokenizerBase,
    titles: Sequence[str],
    bodies: Sequence[str] | None,
    max_length: int,
) -> list[BatchEncoding]:
    """패딩 없이 한 건씩 인코딩한다. 본문이 있으면 제목을 앞 문장, 본문을 뒷 문장으로 넣고 본문만 자른다."""
    if bodies is None:
        encoded = tokenizer(list(titles), truncation=True, max_length=max_length)
    else:
        encoded = tokenizer(list(titles), list(bodies), truncation="only_second", max_length=max_length)
    keys = list(encoded.keys())
    return [BatchEncoding({k: encoded[k][i] for k in keys}) for i in range(len(titles))]


@torch.inference_mode()
def predict_proba_batched(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    titles: Sequence[str],
    bodies: Sequence[str] | None,
    max_length: int,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """낚시성 확률을 돌려준다. 호출하는 쪽이 model.eval() 을 해 둔다."""
    features = encode(tokenizer, titles, bodies, max_length)
    out = []
    for start in range(0, len(features), batch_size):
        batch = tokenizer.pad(features[start : start + batch_size], return_tensors="pt").to(device)
        logits = model(**batch).logits.float()
        out.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)
