"""저장된 BERT 계열 모델로 판별한다. 서버(노트북 CPU)와 평가가 함께 쓴다."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from clickbait.config import MODEL_META_FILENAME
from clickbait.modeling.models import load_pretrained, predict_proba_batched
from clickbait.schema import Prediction


class TransformerPredictor:
    is_mock = False

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        meta: dict[str, object],
        device: torch.device,
    ) -> None:
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.meta = meta
        self.device = device
        self.name = str(meta.get("display_name") or meta["model_key"])
        self.accepts_body = meta["input_mode"] == "title_body"
        self.max_length = int(meta["max_length"])

    @classmethod
    def load(cls, model_dir: Path, device: str = "cpu") -> TransformerPredictor:
        meta = json.loads((model_dir / MODEL_META_FILENAME).read_text(encoding="utf-8"))
        tokenizer, model = load_pretrained(str(meta["model_key"]), str(model_dir))
        return cls(model, tokenizer, meta, torch.device(device))

    def predict_proba(self, titles: Sequence[str], bodies: Sequence[str | None] | None = None) -> np.ndarray:
        """낚시성 확률 배열. 본문을 받지 않는 모델이면 bodies 는 무시한다."""
        pair = None
        if self.accepts_body:
            pair = [body or "" for body in bodies] if bodies is not None else [""] * len(titles)
        return predict_proba_batched(self.model, self.tokenizer, titles, pair, self.max_length, self.device)

    def predict(self, title: str, body: str | None = None) -> Prediction:
        prob = float(self.predict_proba([title], [body])[0])
        # float32 softmax 반올림으로 1을 아주 조금 넘을 수 있다.
        return Prediction(prob_clickbait=min(max(prob, 0.0), 1.0))
