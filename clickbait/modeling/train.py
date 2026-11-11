"""BERT 계열 미세조정. 순수 PyTorch 학습 루프.

에폭마다 검증하고, 이어 학습용 체크포인트를 원자적으로 쓰고, 검증 F1(낚시성 양성)이
가장 좋으면 출력 폴더에 모델·토크나이저·메타를 저장한다. Colab 세션이 끊겨도
--resume 으로 이어 가면 끊기지 않은 실행과 같은 결과(에폭 단위)에 도달한다.
"""

from __future__ import annotations

import json
import os
import random
import shutil
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import CrossEntropyLoss
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase, get_linear_schedule_with_warmup

from clickbait.config import MODEL_META_FILENAME, MODEL_SPECS, TrainConfig
from clickbait.modeling.metrics import compute_metrics
from clickbait.modeling.models import encode, load_pretrained, predict_proba_batched

CHECKPOINT_FILENAME = "checkpoint.pt"
TRAIN_LOG_FILENAME = "train_log.json"


class CheckpointMismatch(ValueError):
    """이어 학습하려는 체크포인트가 다른 설정으로 만들어졌다."""


@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    train_loss: float
    val: dict[str, object]
    is_best: bool


@dataclass(frozen=True)
class TrainData:
    titles: list[str]
    labels: list[int]
    # title_body 모드에서만 쓴다. 제목과 같은 순서.
    bodies: list[str] | None


def frame_to_data(df: pd.DataFrame, bodies: dict[str, str] | None) -> TrainData:
    return TrainData(
        titles=df["title"].tolist(),
        labels=df["label"].astype(int).tolist(),
        bodies=[bodies.get(news_id, "") for news_id in df["news_id"]] if bodies is not None else None,
    )


def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA 를 쓸 수 없다. --device cpu 또는 auto 로 실행한다.")
    return torch.device(choice)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class _Batches:
    """미리 토큰화한 입력을 에폭별로 정해진 순서로 섞어 배치로 내준다."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, data: TrainData, max_length: int, batch_size: int) -> None:
        self.tokenizer = tokenizer
        self.features = encode(tokenizer, data.titles, data.bodies, max_length)
        self.labels = data.labels
        self.batch_size = batch_size

    def __len__(self) -> int:
        return (len(self.labels) + self.batch_size - 1) // self.batch_size

    def epoch(self, seed: int, epoch: int) -> list[dict[str, torch.Tensor]]:
        # 에폭 번호로 순서를 정하므로 이어 학습해도 같은 순서가 나온다.
        generator = torch.Generator().manual_seed(seed * 1000 + epoch)
        order = torch.randperm(len(self.labels), generator=generator).tolist()
        batches = []
        for start in range(0, len(order), self.batch_size):
            idx = order[start : start + self.batch_size]
            batch = self.tokenizer.pad([self.features[i] for i in idx], return_tensors="pt")
            batch["labels"] = torch.tensor([self.labels[i] for i in idx])
            batches.append(dict(batch))
        return batches


def _class_weights(labels: Sequence[int]) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels), minlength=2)
    return torch.tensor(len(labels) / (2 * np.maximum(counts, 1)), dtype=torch.float)


def _save_atomic(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _save_best(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    meta: dict[str, object],
    output_dir: Path,
) -> None:
    # 옆 폴더에 다 쓴 뒤 바꿔 끼워, 중간에 끊겨도 반쯤 쓴 모델 폴더가 남지 않게 한다.
    staging = output_dir.with_name(output_dir.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    model.save_pretrained(staging)
    tokenizer.save_pretrained(staging)
    _write_json(staging / MODEL_META_FILENAME, meta)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    staging.rename(output_dir)


def train(
    config: TrainConfig,
    train_data: TrainData,
    val_data: TrainData,
    *,
    output_dir: Path,
    checkpoint_dir: Path,
    resume: bool = False,
    device: torch.device | None = None,
    pretrained: str | None = None,
    extra_meta: dict[str, object] | None = None,
    on_epoch_end: Callable[[EpochRecord], None] | None = None,
    progress: bool = False,
) -> list[EpochRecord]:
    """학습을 끝까지 돌리고 에폭 기록을 돌려준다.

    pretrained 는 허브 이름 대신 쓸 로컬 폴더다(테스트의 작은 모델, 미리 받아 둔 모델).
    resume 이면 체크포인트가 있을 때 이어 가고, 없으면 처음부터 시작한다.
    """
    device = device or torch.device("cpu")
    spec = MODEL_SPECS[config.model_key]
    _seed_everything(config.seed)
    tokenizer, model = load_pretrained(config.model_key, pretrained or spec.hub_id)
    model.to(device)

    train_batches = _Batches(tokenizer, train_data, config.max_length, config.batch_size)
    total_steps = len(train_batches) * config.epochs
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    params = [
        {"params": [p for n, p in model.named_parameters() if not n.endswith(no_decay)], "weight_decay": config.weight_decay},
        {"params": [p for n, p in model.named_parameters() if n.endswith(no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(params, lr=config.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * config.warmup_ratio), total_steps)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = CrossEntropyLoss(weight=_class_weights(train_data.labels).to(device) if config.class_weights else None)

    start_epoch = 1
    best_f1 = -1.0
    best_epoch = 0
    history: list[EpochRecord] = []
    checkpoint_path = checkpoint_dir / CHECKPOINT_FILENAME
    if resume and checkpoint_path.is_file():
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if state["config"] != asdict(config):
            raise CheckpointMismatch(
                f"체크포인트의 학습 설정이 지금 설정과 다르다. 같은 설정으로 이어 가거나 --resume 없이 새로 시작한다.\n"
                f"체크포인트: {state['config']}\n지금: {asdict(config)}"
            )
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"] + 1
        best_f1 = state["best_f1"]
        best_epoch = state["best_epoch"]
        history = [EpochRecord(**record) for record in state["history"]]

    for epoch in range(start_epoch, config.epochs + 1):
        torch.manual_seed(config.seed * 1000 + epoch)
        model.train()
        losses = []
        batches = train_batches.epoch(config.seed, epoch)
        for batch in tqdm(batches, desc=f"에폭 {epoch}/{config.epochs}", disable=not progress):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(**batch).logits
            loss = loss_fn(logits.float(), labels)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(loss.item())

        model.eval()
        prob = predict_proba_batched(model, tokenizer, val_data.titles, val_data.bodies, config.max_length, device, config.batch_size)
        val_metrics = compute_metrics(np.asarray(val_data.labels), prob)
        is_best = float(val_metrics["f1"]) > best_f1
        record = EpochRecord(epoch=epoch, train_loss=round(float(np.mean(losses)), 6), val=val_metrics, is_best=is_best)
        history.append(record)
        if is_best:
            best_f1 = float(val_metrics["f1"])
            best_epoch = epoch
            meta = {
                "model_key": config.model_key,
                "hub_id": spec.hub_id,
                "display_name": spec.display_name,
                "input_mode": config.input_mode,
                "max_length": config.max_length,
                "best_epoch": epoch,
                "val_metrics": val_metrics,
                "train_config": asdict(config),
                **(extra_meta or {}),
            }
            _save_best(model, tokenizer, meta, output_dir)
        _save_atomic(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best_f1": best_f1,
                "best_epoch": best_epoch,
                "history": [asdict(r) for r in history],
                "config": asdict(config),
            },
            checkpoint_path,
        )
        _write_json(output_dir / TRAIN_LOG_FILENAME, {"best_epoch": best_epoch, "epochs": [asdict(r) for r in history]})
        if on_epoch_end is not None:
            on_epoch_end(record)
    return history
