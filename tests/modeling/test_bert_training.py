import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from clickbait.config import MODEL_META_FILENAME, MODEL_SPECS, TrainConfig
from clickbait.data.prepare import read_bodies, read_split, stratified_sample
from clickbait.modeling.models import FAMILIES
from clickbait.modeling.train import CHECKPOINT_FILENAME, TRAIN_LOG_FILENAME, CheckpointMismatch, EpochRecord, frame_to_data, train
from clickbait.predictor import load_predictor
from clickbait.schema import Prediction

CONFIG = TrainConfig(model_key="klue-bert", max_length=48, batch_size=16, learning_rate=1e-3, epochs=2, seed=7)


class Interrupted(Exception):
    pass


def _data(processed_dir: Path, bodies: dict[str, str] | None = None):
    train_df = stratified_sample(read_split(processed_dir / "train.csv"), 96, seed=0)
    val_df = read_split(processed_dir / "val.csv")
    return frame_to_data(train_df, bodies), frame_to_data(val_df, bodies)


def _run(tiny_bert_dir: Path, processed_dir: Path, out: Path, config: TrainConfig = CONFIG, **kwargs):
    bodies = read_bodies(processed_dir / "bodies.jsonl") if config.input_mode == "title_body" else None
    train_data, val_data = _data(processed_dir, bodies)
    return train(
        config,
        train_data,
        val_data,
        output_dir=out / "model",
        checkpoint_dir=out / "ckpt",
        pretrained=str(tiny_bert_dir),
        **kwargs,
    )


def test_every_model_key_has_a_loader() -> None:
    assert set(FAMILIES) == set(MODEL_SPECS)


def test_training_saves_best_model_that_the_server_can_load(tiny_bert_dir: Path, processed_dir: Path, tmp_path: Path) -> None:
    history = _run(tiny_bert_dir, processed_dir, tmp_path)
    model_dir = tmp_path / "model"

    assert [r.epoch for r in history] == [1, 2]
    meta = json.loads((model_dir / MODEL_META_FILENAME).read_text(encoding="utf-8"))
    best = max(history, key=lambda r: r.val["f1"])
    assert meta["model_key"] == "klue-bert" and meta["hub_id"] == "klue/bert-base"
    assert meta["input_mode"] == "title" and meta["max_length"] == 48
    assert meta["best_epoch"] == best.epoch and meta["val_metrics"] == best.val
    log = json.loads((model_dir / TRAIN_LOG_FILENAME).read_text(encoding="utf-8"))
    assert [e["epoch"] for e in log["epochs"]] == [1, 2]

    predictor = load_predictor(model_dir)
    assert predictor.is_mock is False and predictor.accepts_body is False
    prediction = predictor.predict("결국 터졌다…배우 정민의 충격 근황")
    assert isinstance(prediction, Prediction)

    titles = ["정부, 예산안 발표", "경악! 정체 드러나", "한국은행 금리 동결"]
    batch = predictor.predict_proba(titles)
    single = [predictor.predict(t).prob_clickbait for t in titles]
    assert batch == pytest.approx(single, abs=1e-5)


def test_resume_after_interruption_matches_uninterrupted_run(tiny_bert_dir: Path, processed_dir: Path, tmp_path: Path) -> None:
    straight = _run(tiny_bert_dir, processed_dir, tmp_path / "straight")

    def crash_after_first(record: EpochRecord) -> None:
        if record.epoch == 1:
            raise Interrupted

    with pytest.raises(Interrupted):
        _run(tiny_bert_dir, processed_dir, tmp_path / "resumed", on_epoch_end=crash_after_first)
    resumed = _run(tiny_bert_dir, processed_dir, tmp_path / "resumed", resume=True)

    assert resumed == straight
    a = torch.load(tmp_path / "straight/ckpt" / CHECKPOINT_FILENAME, weights_only=False)["model"]
    b = torch.load(tmp_path / "resumed/ckpt" / CHECKPOINT_FILENAME, weights_only=False)["model"]
    assert all(torch.equal(a[k], b[k]) for k in a)

    again = _run(tiny_bert_dir, processed_dir, tmp_path / "resumed", resume=True)
    assert again == straight


def test_resume_refuses_a_checkpoint_from_other_settings(tiny_bert_dir: Path, processed_dir: Path, tmp_path: Path) -> None:
    _run(tiny_bert_dir, processed_dir, tmp_path, config=replace(CONFIG, epochs=1))
    with pytest.raises(CheckpointMismatch):
        _run(tiny_bert_dir, processed_dir, tmp_path, config=replace(CONFIG, epochs=1, learning_rate=5e-4), resume=True)


def test_title_body_model_uses_the_body(tiny_bert_dir: Path, processed_dir: Path, tmp_path: Path) -> None:
    config = replace(CONFIG, input_mode="title_body", max_length=96, epochs=1)
    _run(tiny_bert_dir, processed_dir, tmp_path, config=config)
    predictor = load_predictor(tmp_path / "model")
    assert predictor.accepts_body is True
    title = "정부, 예산안 발표"
    with_body = predictor.predict(title, "관계자는 단계적으로 진행될 것이라고 설명했다.").prob_clickbait
    assert with_body != predictor.predict(title).prob_clickbait
