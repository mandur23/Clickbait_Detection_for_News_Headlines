"""Flask 앱. 판별기는 밖에서 한 번 만들어 넣으므로 이 모듈은 모델 구현을 모른다."""

from __future__ import annotations

import sqlite3

from flask import Flask, Response, jsonify, render_template, request

from clickbait.config import MAX_INPUT_BODY_CHARS, MAX_INPUT_TITLE_CHARS
from clickbait.predictor import Predictor
from server.history import HistoryStore
from server.validation import InputError, parse_history_limit, parse_predict_payload


def _error(status: int, code: str, message: str) -> tuple[Response, int]:
    return jsonify({"error": {"code": code, "message": message}}), status


def create_app(predictor: Predictor, history: HistoryStore | None = None) -> Flask:
    """판별기와 (선택) 기록 저장소를 받아 앱을 만든다. history가 None이면 기록을 끈다."""
    app = Flask(__name__)
    app.json.ensure_ascii = False  # type: ignore[attr-defined]

    @app.get("/")
    def index() -> str:
        return render_template("index.html")

    @app.post("/predict")
    def predict() -> Response | tuple[Response, int]:
        parsed = parse_predict_payload(request.get_json(silent=True))
        if isinstance(parsed, InputError):
            return _error(400, parsed.code, parsed.message)

        body = parsed.body if predictor.accepts_body else None
        try:
            prediction = predictor.predict(parsed.title, body)
        except Exception:
            app.logger.exception("판별 중 오류: %r", parsed.title)
            return _error(500, "prediction_failed", "판별 중 문제가 생겼습니다. 잠시 후 다시 시도해 주세요.")

        if history is not None:
            try:
                history.add(parsed.title, prediction, predictor.name)
            except sqlite3.Error:
                app.logger.exception("판별 기록 저장 실패")
        return jsonify(prediction.to_json())

    @app.get("/api/info")
    def info() -> Response:
        return jsonify(
            {
                "model": predictor.name,
                "is_mock": predictor.is_mock,
                "accepts_body": predictor.accepts_body,
                "max_title_chars": MAX_INPUT_TITLE_CHARS,
                "max_body_chars": MAX_INPUT_BODY_CHARS,
                "history_enabled": history is not None,
            }
        )

    @app.get("/api/history")
    def recent_history() -> Response:
        if history is None:
            return jsonify([])
        limit = parse_history_limit(request.args.get("limit"))
        return jsonify([record.to_json() for record in history.recent(limit)])

    return app
