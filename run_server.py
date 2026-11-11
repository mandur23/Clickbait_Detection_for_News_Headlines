"""시연용 서버 실행. 판별기는 여기서 한 번만 불러온다.

환경 변수:
    CLICKBAIT_MODEL_DIR   학습된 모델 폴더 (기본: models/klue-bert)
    CLICKBAIT_HISTORY_DB  판별 기록 SQLite 파일 경로, off 면 기록 끔 (기본: instance/history.db)
    HOST, PORT            기본 127.0.0.1:5000
"""

from __future__ import annotations

import os
from pathlib import Path

from clickbait.config import DEFAULT_MODEL_DIR, MODEL_META_FILENAME, ROOT
from clickbait.predictor import load_predictor
from server.app import create_app
from server.history import HistoryStore

DEFAULT_HISTORY_DB = ROOT / "instance" / "history.db"


def main() -> None:
    model_dir = Path(os.environ.get("CLICKBAIT_MODEL_DIR", DEFAULT_MODEL_DIR))
    history_setting = os.environ.get("CLICKBAIT_HISTORY_DB", str(DEFAULT_HISTORY_DB))
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))

    predictor = load_predictor(model_dir)
    if predictor.is_mock:
        print("[데모 모드] 학습된 모델이 없어 가짜 판별기를 씁니다. 화면의 확률은 실제 모델 결과가 아닙니다.")
        print(f"  찾은 위치: {model_dir}")
        print(f"  실제 모델을 쓰려면 학습한 모델을 이 폴더에 저장해 {MODEL_META_FILENAME} 이 생기게 하거나,")
        print("  CLICKBAIT_MODEL_DIR 환경 변수로 모델 폴더를 지정한 뒤 서버를 다시 시작하세요.")
    else:
        print(f"모델: {predictor.name} ({model_dir})")
        print(f"본문 입력: {'사용' if predictor.accepts_body else '사용 안 함 (제목만)'}")

    history = None if history_setting.strip().lower() == "off" else HistoryStore(Path(history_setting))
    print(f"판별 기록: {history.path if history else '끔'}")
    print(f"주소: http://{host}:{port}")

    create_app(predictor, history).run(host=host, port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
