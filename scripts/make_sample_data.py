"""AI Hub 형식의 합성 데이터를 data/sample_aihub/ 에 만든다.

실제 데이터가 승인되기 전에 코드가 끝까지 도는지만 확인하는 용도다.
이 데이터로 얻은 지표는 아무 의미가 없다. 자세한 내용은 clickbait/data/sample.py 참고.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clickbait.config import SAMPLE_RAW_DIR  # noqa: E402
from clickbait.data.sample import SAMPLE_SEED, write_sample  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Hub 형식의 합성 샘플 데이터를 만든다(지표는 의미 없음).")
    parser.add_argument("--out-dir", type=Path, default=SAMPLE_RAW_DIR, help="출력 폴더(기존 내용은 지운다)")
    parser.add_argument("--n", type=int, default=600, help="정상 기사 수(경계 사례 파일은 따로 더한다)")
    parser.add_argument("--seed", type=int, default=SAMPLE_SEED)
    args = parser.parse_args()
    count = write_sample(args.out_dir, args.n, args.seed)
    print(f"합성 문서 {count}건을 {args.out_dir} 에 썼다. 이 데이터의 지표는 의미가 없다.")


if __name__ == "__main__":
    main()
