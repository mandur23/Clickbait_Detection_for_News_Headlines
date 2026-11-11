"""KLUE-BERT·KoBERT 를 미세조정한다. 에폭마다 체크포인트를 쓰므로 --resume 으로 이어 갈 수 있다."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clickbait.config import MODEL_SPECS, MODELS_DIR, PROCESSED_DIR, TITLE_BODY_MAX_LENGTH, TrainConfig  # noqa: E402
from clickbait.data.prepare import BODIES_FILENAME, read_bodies, read_split, stratified_sample  # noqa: E402

DEFAULTS = TrainConfig()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BERT 계열 모델로 낚시성 제목 판별기를 미세조정한다.")
    parser.add_argument("--model-key", choices=sorted(MODEL_SPECS), default=DEFAULTS.model_key)
    parser.add_argument("--input-mode", choices=["title", "title_body"], default=DEFAULTS.input_mode, help="title_body 는 본문을 뒷 문장으로 함께 넣는다")
    parser.add_argument("--max-length", type=int, default=None, help=f"토큰 길이(기본: title {DEFAULTS.max_length}, title_body {TITLE_BODY_MAX_LENGTH})")
    parser.add_argument("--batch-size", type=int, default=DEFAULTS.batch_size)
    parser.add_argument("--learning-rate", type=float, default=DEFAULTS.learning_rate)
    parser.add_argument("--epochs", type=int, default=DEFAULTS.epochs)
    parser.add_argument("--weight-decay", type=float, default=DEFAULTS.weight_decay)
    parser.add_argument("--warmup-ratio", type=float, default=DEFAULTS.warmup_ratio)
    parser.add_argument("--class-weights", action="store_true", help="클래스 불균형을 손실 가중치로 보정한다")
    parser.add_argument("--seed", type=int, default=DEFAULTS.seed)
    parser.add_argument("--train-csv", type=Path, default=PROCESSED_DIR / "train.csv")
    parser.add_argument("--val-csv", type=Path, default=PROCESSED_DIR / "val.csv")
    parser.add_argument("--bodies", type=Path, default=PROCESSED_DIR / BODIES_FILENAME, help="title_body 모드의 본문 파일")
    parser.add_argument("--output-dir", type=Path, default=None, help="가장 좋은 모델을 저장할 폴더(기본 models/<model-key>)")
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="이어 학습용 체크포인트 폴더(Colab 에서는 Drive 경로, 기본 models/<model-key>-checkpoint)")
    parser.add_argument("--resume", action="store_true", help="체크포인트가 있으면 이어서 학습한다(없으면 처음부터)")
    parser.add_argument("--limit-train", type=int, default=None, help="학습 데이터를 라벨 비율대로 N 건만 쓴다(먼저 작게 돌려 볼 때)")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.train_csv, args.val_csv) + ((args.bodies,) if args.input_mode == "title_body" else ()):
        if not path.is_file():
            sys.exit(f"파일이 없다: {path} (먼저 scripts/prepare_data.py 를 실행한다)")

    # torch·transformers 는 인자 오류를 먼저 알려 준 다음에 불러온다.
    from clickbait.modeling.train import CheckpointMismatch, EpochRecord, frame_to_data, resolve_device, train

    max_length = args.max_length or (TITLE_BODY_MAX_LENGTH if args.input_mode == "title_body" else DEFAULTS.max_length)
    config = TrainConfig(
        model_key=args.model_key,
        input_mode=args.input_mode,
        max_length=max_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        class_weights=args.class_weights,
        seed=args.seed,
    )
    output_dir = args.output_dir or MODELS_DIR / args.model_key
    checkpoint_dir = args.checkpoint_dir or MODELS_DIR / f"{args.model_key}-checkpoint"
    try:
        device = resolve_device(args.device)
    except ValueError as error:
        sys.exit(str(error))

    train_df = read_split(args.train_csv)
    if args.limit_train is not None:
        train_df = stratified_sample(train_df, args.limit_train, args.seed)
    val_df = read_split(args.val_csv)
    bodies = read_bodies(args.bodies) if args.input_mode == "title_body" else None

    def report(record: EpochRecord) -> None:
        mark = " (최고, 저장함)" if record.is_best else ""
        val = record.val
        print(
            f"[{record.epoch}/{config.epochs}] 학습 손실 {record.train_loss:.4f} | 검증 정확도 {val['accuracy']:.4f} "
            f"정밀도 {val['precision']:.4f} 재현율 {val['recall']:.4f} F1 {val['f1']:.4f}{mark}",
            flush=True,
        )

    print(f"{MODEL_SPECS[args.model_key].display_name} 학습 시작: 학습 {len(train_df)}건, 검증 {len(val_df)}건, 장치 {device}")
    print(f"설정: {asdict(config)}")
    try:
        history = train(
            config,
            frame_to_data(train_df, bodies),
            frame_to_data(val_df, bodies),
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
            resume=args.resume,
            device=device,
            extra_meta={"limit_train": args.limit_train, "train_size": len(train_df), "train_csv": str(args.train_csv)},
            on_epoch_end=report,
            progress=True,
        )
    except CheckpointMismatch as error:
        sys.exit(str(error))
    best = max(history, key=lambda r: float(r.val["f1"]))
    print(f"끝. 최고 검증 F1 {best.val['f1']:.4f} (에폭 {best.epoch}). 모델: {output_dir}, 체크포인트: {checkpoint_dir}")


if __name__ == "__main__":
    main()
