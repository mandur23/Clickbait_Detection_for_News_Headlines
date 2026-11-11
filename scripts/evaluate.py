"""저장된 모델(기준선 또는 BERT 폴더)을 평가해 reports/<이름>/ 에 지표·혼동행렬·오분류 목록을 쓴다."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clickbait.config import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR  # noqa: E402
from clickbait.data.prepare import BODIES_FILENAME, read_bodies, read_split  # noqa: E402
from clickbait.modeling.evaluate import evaluate, load_scorer, write_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="학습한 모델을 평가 데이터로 평가한다(낚시성이 양성).")
    parser.add_argument("--model-dir", type=Path, default=MODELS_DIR / "baseline", help="models/baseline 또는 BERT 모델 폴더")
    parser.add_argument("--eval-csv", type=Path, default=PROCESSED_DIR / "test.csv")
    parser.add_argument("--bodies", type=Path, default=PROCESSED_DIR / BODIES_FILENAME, help="본문을 쓰는 모델에만 필요")
    parser.add_argument("--name", default=None, help="보고서 이름(기본: 모델 폴더 이름)")
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--device", default="cpu", help="BERT 모델을 돌릴 장치(cpu, cuda)")
    args = parser.parse_args()

    if not args.eval_csv.is_file():
        parser.error(f"평가 파일이 없다: {args.eval_csv}")
    try:
        scorer = load_scorer(args.model_dir, device=args.device)
    except FileNotFoundError as error:
        parser.error(str(error))
    bodies = None
    if scorer.accepts_body:
        if not args.bodies.is_file():
            parser.error(f"이 모델은 본문을 쓰는데 본문 파일이 없다: {args.bodies}")
        bodies = read_bodies(args.bodies)

    name = args.name or args.model_dir.name
    result = evaluate(scorer, read_split(args.eval_csv), bodies)
    info = {
        "name": name,
        "model": scorer.name,
        "model_dir": str(args.model_dir),
        "input_mode": "title_body" if scorer.accepts_body else "title",
        "eval_csv": args.eval_csv.name,
    }
    out_dir = args.reports_dir / name
    write_report(result, out_dir, info)
    m = result.metrics
    print(f"[{name}] n={m['n']} 정확도 {m['accuracy']:.4f} 정밀도 {m['precision']:.4f} 재현율 {m['recall']:.4f} F1 {m['f1']:.4f}")
    print(f"혼동행렬(행: 실제 일반/낚시성, 열: 예측 일반/낚시성): {m['confusion_matrix']}")
    for source, row in m["by_source"].items():
        print(f"  {source or '(출처 없음)'}: n={row['n']} 정답률 {row['accuracy']:.4f} 평균 낚시성 확률 {row['mean_prob_clickbait']:.4f}")
    print(f"오분류 {len(result.misclassified)}건. 보고서를 {out_dir} 에 썼다.")


if __name__ == "__main__":
    main()
