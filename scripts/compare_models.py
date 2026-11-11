"""reports/*/metrics.json 을 모아 reports/comparison.md 비교표를 만든다."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clickbait.config import REPORTS_DIR  # noqa: E402
from clickbait.modeling.evaluate import METRICS_FILENAME, comparison_table  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="평가 보고서를 모아 모델 비교표를 만든다.")
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    args = parser.parse_args()
    if not any(args.reports_dir.glob(f"*/{METRICS_FILENAME}")):
        sys.exit(f"{args.reports_dir} 아래에 {METRICS_FILENAME} 가 없다. 먼저 scripts/evaluate.py 를 실행한다.")
    table = "# 모델 비교\n\n낚시성을 양성으로 둔 지표다. F1 이 높은 순.\n\n" + comparison_table(args.reports_dir)
    out = args.reports_dir / "comparison.md"
    out.write_text(table, encoding="utf-8")
    print(table)
    print(f"{out} 에 썼다.")


if __name__ == "__main__":
    main()
