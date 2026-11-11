from __future__ import annotations

import pytest

from clickbait.config import MAX_INPUT_BODY_CHARS, MAX_INPUT_TITLE_CHARS
from server.validation import (
    DEFAULT_HISTORY_LIMIT,
    InputError,
    PredictInput,
    parse_history_limit,
    parse_predict_payload,
)


def error_code(payload: object) -> str:
    result = parse_predict_payload(payload)
    assert isinstance(result, InputError), result
    return result.code


def test_valid_title_is_cleaned() -> None:
    result = parse_predict_payload({"title": "  <b>속보</b>   정부 발표&amp;해명  "})
    assert result == PredictInput(title="속보 정부 발표&해명", body=None)


def test_body_is_cleaned_and_blank_body_becomes_none() -> None:
    assert parse_predict_payload({"title": "제목", "body": " 본문\n\n내용 "}) == PredictInput("제목", "본문 내용")
    assert parse_predict_payload({"title": "제목", "body": "   "}) == PredictInput("제목", None)


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        (None, "invalid_request"),
        (["제목"], "invalid_request"),
        ("제목", "invalid_request"),
        ({}, "title_required"),
        ({"title": 123}, "title_not_string"),
        ({"title": ["제목"]}, "title_not_string"),
        ({"title": ""}, "title_empty"),
        ({"title": " \t\n "}, "title_empty"),
        ({"title": "<br><p></p>"}, "title_empty"),
        ({"title": "가" * (MAX_INPUT_TITLE_CHARS + 1)}, "title_too_long"),
        ({"title": "제목", "body": 1}, "body_not_string"),
        ({"title": "제목", "body": "가" * (MAX_INPUT_BODY_CHARS + 1)}, "body_too_long"),
    ],
)
def test_invalid_payloads(payload: object, code: str) -> None:
    assert error_code(payload) == code


def test_title_at_limit_is_accepted() -> None:
    result = parse_predict_payload({"title": "가" * MAX_INPUT_TITLE_CHARS})
    assert isinstance(result, PredictInput)


def test_error_messages_are_korean_guidance() -> None:
    result = parse_predict_payload({"title": "가" * (MAX_INPUT_TITLE_CHARS + 5)})
    assert isinstance(result, InputError)
    assert f"{MAX_INPUT_TITLE_CHARS}자 이하" in result.message


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, DEFAULT_HISTORY_LIMIT), ("5", 5), ("abc", DEFAULT_HISTORY_LIMIT), ("0", 1), ("-3", 1), ("9999", 100)],
)
def test_history_limit(raw: str | None, expected: int) -> None:
    assert parse_history_limit(raw) == expected
