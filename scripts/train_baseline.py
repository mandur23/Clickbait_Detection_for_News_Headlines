"""TF-IDF + 로지스틱 회귀 기준선을 학습해 models/baseline/ 에 저장한다."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clickbait.config import MODELS_DIR, PROCESSED_DIR  # noqa: E402
from clickbait.data.prepare import read_split  # noqa: E402
from clickbait.modeling.baseline import BaselineModel  # noqa: E402
from clickbait.modeling.metrics import compute_metrics  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="TF-IDF(단어 1~2그램) + 로지스틱 회귀 기준선을 학습한다.")
    parser.add_argument("--train-csv", type=Path, default=PROCESSED_DIR / "train.csv")
    parser.add_argument("--val-csv", type=Path, default=PROCESSED_DIR / "val.csv")
    parser.add_argument("--output-dir", type=Path, default=MODELS_DIR / "baseline")
    parser.add_argument("--min-df", type=int, default=2, help="이보다 적은 문서에 나온 단어는 버린다")
    parser.add_argument("--class-weights", action="store_true", help="클래스 불균형을 손실 가중치로 보정한다")
    args = parser.parse_args()

    for path in (args.train_csv, args.val_csv):
        if not path.is_file():
            parser.error(f"파일이 없다: {path} (먼저 scripts/prepare_data.py 를 실행한다)")
    train = read_split(args.train_csv)
    model = BaselineModel.train(train["title"], train["label"], min_df=args.min_df, class_weights=args.class_weights)
    path = model.save(args.output_dir)
    val = read_split(args.val_csv)
    metrics = compute_metrics(val["label"].to_numpy(), model.predict_proba(val["title"].tolist()))
    print(f"학습 {len(train)}건. 검증 정확도 {metrics['accuracy']:.4f}, F1 {metrics['f1']:.4f}")
    print(f"모델을 {path} 에 저장했다.")


if __name__ == "__main__":
    main()
