"""AI Hub 원본을 읽어 정제하고 train/val/test CSV, 본문 파일, 보고서를 만든다."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clickbait.config import PROCESSED_DIR, SEED  # noqa: E402
from clickbait.data.aihub import ReadStats, iter_articles  # noqa: E402
from clickbait.data.prepare import prepare, write_prepared  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Hub 낚시성 기사 데이터를 학습용 CSV 로 정제한다.")
    parser.add_argument("--raw-dir", type=Path, required=True, help="JSON 파일이나 ZIP 묶음이 있는 폴더(하위 폴더 포함)")
    parser.add_argument("--out-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--parts", type=int, nargs="+", choices=[1, 2], default=[1], help="쓸 Part(기본 1). Part2 는 제목이 원본이라 제목 판별에는 잡음이다")
    parser.add_argument("--balance", action="store_true", help="많은 쪽 클래스를 줄여 1:1 로 맞춘다")
    parser.add_argument("--sample", type=int, default=None, help="라벨 비율을 유지한 채 N 건만 쓴다")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if not args.raw_dir.is_dir():
        parser.error(f"원본 폴더가 없다: {args.raw_dir}")
    stats = ReadStats()
    articles = list(iter_articles(args.raw_dir, stats))
    try:
        data = prepare(articles, parts=args.parts, balance=args.balance, sample=args.sample, seed=args.seed)
    except ValueError as error:
        print(f"읽기 결과: {json.dumps(stats.to_json(), ensure_ascii=False)}", file=sys.stderr)
        sys.exit(str(error))
    data.report = {"raw_dir": str(args.raw_dir), "read": stats.to_json(), **data.report}
    write_prepared(data, args.out_dir)

    report = data.report
    print(f"파일 {stats.files}개, 기사 {stats.articles}건 읽음. 건너뜀: {dict(stats.skipped)}, 경로·필드 라벨 불일치 {stats.label_path_mismatch}건")
    print(f"단계별 건수: {report['stages']}")
    print(f"중복 정리: {report['dedupe']}")
    print(f"클래스 비율: {report['class_ratio']}")
    print(f"분할 크기: {report['split_sizes']}")
    print(f"결과를 {args.out_dir} 에 썼다.")


if __name__ == "__main__":
    main()
