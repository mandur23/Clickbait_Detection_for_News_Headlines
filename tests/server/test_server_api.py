from __future__ import annotations

import pytest
from flask.testing import FlaskClient
from server_fakes import FakePredictor
from werkzeug.test import TestResponse

from clickbait.config import MAX_INPUT_BODY_CHARS, MAX_INPUT_TITLE_CHARS
from server.app import create_app
from server.history import HistoryStore


def assert_error(response: TestResponse, status: int, code: str) -> None:
    assert response.status_code == status
    body = response.get_json()
    assert set(body) == {"error"}
    assert set(body["error"]) == {"code", "message"}
    assert body["error"]["code"] == code
    assert body["error"]["message"]


def test_index_page(client: FlaskClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "판별하기" in response.get_data(as_text=True)


def test_static_assets_are_served(client: FlaskClient) -> None:
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200


def test_predict_valid_title(client: FlaskClient, predictor: FakePredictor) -> None:
    response = client.post("/predict", json={"title": "충격! 결국 밝혀진 이유는"})
    assert response.status_code == 200
    body = response.get_json()
    assert set(body) == {"label", "prob_clickbait", "prob_normal"}
    assert body["label"] == "clickbait"
    assert body["prob_clickbait"] == pytest.approx(0.73)
    assert body["prob_clickbait"] + body["prob_normal"] == pytest.approx(1.0)
    assert predictor.calls == [("충격! 결국 밝혀진 이유는", None)]


def test_predict_normal_label() -> None:
    client = create_app(FakePredictor(prob_clickbait=0.1)).test_client()
    body = client.post("/predict", json={"title": "기준금리 동결"}).get_json()
    assert body["label"] == "normal"
    assert body["prob_normal"] == pytest.approx(0.9)


@pytest.mark.parametrize(
    ("title", "cleaned"),
    [
        ("특수문자 #@$%^&*() 제목?!", "특수문자 #@$%^&*() 제목?!"),
        ("이모지 😱 들어간 제목", "이모지 😱 들어간 제목"),
        ("<script>alert(1)</script> 기사 제목", "alert(1) 기사 제목"),
        ("따옴표 “인용” &quot;제목&quot;", '따옴표 "인용" "제목"'),
        ("대박!!!!!!! 이게 뭐야", "대박!!! 이게 뭐야"),
    ],
)
def test_unusual_titles_reach_predictor_cleaned(
    client: FlaskClient, predictor: FakePredictor, title: str, cleaned: str
) -> None:
    response = client.post("/predict", json={"title": title})
    assert response.status_code == 200
    assert predictor.calls == [(cleaned, None)]


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"json": {"title": ""}}, "title_empty"),
        ({"json": {"title": "    "}}, "title_empty"),
        ({"json": {"title": "가" * (MAX_INPUT_TITLE_CHARS + 1)}}, "title_too_long"),
        ({"json": {"title": 42}}, "title_not_string"),
        ({"json": {}}, "title_required"),
        ({"json": ["제목"]}, "invalid_request"),
        ({"data": "title=제목", "content_type": "application/x-www-form-urlencoded"}, "invalid_request"),
        ({"data": "{not json", "content_type": "application/json"}, "invalid_request"),
        ({"json": {"title": "제목", "body": 3}}, "body_not_string"),
        ({"json": {"title": "제목", "body": "가" * (MAX_INPUT_BODY_CHARS + 1)}}, "body_too_long"),
    ],
)
def test_invalid_input_is_rejected_before_predictor(
    client: FlaskClient, predictor: FakePredictor, kwargs: dict[str, object], code: str
) -> None:
    assert_error(client.post("/predict", **kwargs), 400, code)
    assert predictor.calls == []


def test_body_ignored_when_predictor_takes_title_only(client: FlaskClient, predictor: FakePredictor) -> None:
    response = client.post("/predict", json={"title": "제목", "body": "본문 내용"})
    assert response.status_code == 200
    assert predictor.calls == [("제목", None)]


def test_body_passed_when_predictor_accepts_it() -> None:
    predictor = FakePredictor(accepts_body=True)
    client = create_app(predictor).test_client()
    client.post("/predict", json={"title": "제목", "body": "  본문   내용 "})
    client.post("/predict", json={"title": "제목만"})
    assert predictor.calls == [("제목", "본문 내용"), ("제목만", None)]


def test_predictor_exception_returns_500() -> None:
    predictor = FakePredictor(fail=True)
    client = create_app(predictor).test_client()
    response = client.post("/predict", json={"title": "제목"})
    assert_error(response, 500, "prediction_failed")
    assert "모델 내부 오류" not in response.get_data(as_text=True)


def test_info(client: FlaskClient) -> None:
    assert client.get("/api/info").get_json() == {
        "model": "fake-model",
        "is_mock": False,
        "accepts_body": False,
        "max_title_chars": MAX_INPUT_TITLE_CHARS,
        "max_body_chars": MAX_INPUT_BODY_CHARS,
        "history_enabled": False,
    }


def test_info_reports_mock_body_and_history(history: HistoryStore) -> None:
    predictor = FakePredictor(name="mock", is_mock=True, accepts_body=True)
    info = create_app(predictor, history).test_client().get("/api/info").get_json()
    assert info["is_mock"] is True
    assert info["accepts_body"] is True
    assert info["history_enabled"] is True


def test_history_off_returns_empty_list(client: FlaskClient) -> None:
    client.post("/predict", json={"title": "제목"})
    response = client.get("/api/history")
    assert response.status_code == 200
    assert response.get_json() == []


def test_history_records_newest_first(history: HistoryStore) -> None:
    client = create_app(FakePredictor(prob_clickbait=0.2), history).test_client()
    for title in ["첫 번째 제목", "두 번째 제목", "  세 번째   제목 "]:
        client.post("/predict", json={"title": title})
    client.post("/predict", json={"title": ""})

    records = client.get("/api/history").get_json()
    assert [r["title"] for r in records] == ["세 번째 제목", "두 번째 제목", "첫 번째 제목"]
    first = records[0]
    assert set(first) == {"id", "created_at", "title", "label", "prob_clickbait", "model"}
    assert first["label"] == "normal"
    assert first["prob_clickbait"] == pytest.approx(0.2)
    assert first["model"] == "fake-model"
    assert first["created_at"].endswith("+00:00")

    assert len(client.get("/api/history?limit=2").get_json()) == 2


def test_history_failure_does_not_fail_prediction(history: HistoryStore) -> None:
    history.path.unlink()
    history.path.mkdir()
    client = create_app(FakePredictor(), history).test_client()
    response = client.post("/predict", json={"title": "제목"})
    assert response.status_code == 200
    assert set(response.get_json()) == {"label", "prob_clickbait", "prob_normal"}
