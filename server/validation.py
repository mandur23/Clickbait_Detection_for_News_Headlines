"""요청 입력 검사. 서버로 들어온 값은 모두 여기를 거친 뒤에만 판별기로 간다."""

from __future__ import annotations

from dataclasses import dataclass

from clickbait.config import MAX_INPUT_BODY_CHARS, MAX_INPUT_TITLE_CHARS
from clickbait.text import clean_text, clean_title

DEFAULT_HISTORY_LIMIT = 20
MAX_HISTORY_LIMIT = 100


@dataclass(frozen=True)
class PredictInput:
    """정리가 끝난 제목과, 있으면 정리된 본문."""

    title: str
    body: str | None


@dataclass(frozen=True)
class InputError:
    """사용자에게 그대로 보여 줄 수 있는 입력 오류."""

    code: str
    message: str


def parse_predict_payload(payload: object) -> PredictInput | InputError:
    """JSON으로 해석된 요청 본문을 검사해 판별 입력이나 오류 하나로 바꾼다.

    길이 제한은 사용자가 입력한 그대로의 글자 수로 센다. 화면의 글자 수 표시와 같은 기준이다.
    """
    if not isinstance(payload, dict):
        return InputError("invalid_request", "요청 형식이 올바르지 않습니다. 제목을 JSON 객체로 보내 주세요.")

    title = payload.get("title")
    if title is None:
        return InputError("title_required", "제목을 입력해 주세요.")
    if not isinstance(title, str):
        return InputError("title_not_string", "제목은 문자열이어야 합니다.")
    if len(title) > MAX_INPUT_TITLE_CHARS:
        return InputError(
            "title_too_long",
            f"제목은 {MAX_INPUT_TITLE_CHARS}자 이하로 입력해 주세요. (현재 {len(title)}자)",
        )
    cleaned_title = clean_title(title)
    if not cleaned_title:
        return InputError("title_empty", "제목을 입력해 주세요.")

    body = payload.get("body")
    if body is None:
        return PredictInput(cleaned_title, None)
    if not isinstance(body, str):
        return InputError("body_not_string", "본문은 문자열이어야 합니다.")
    if len(body) > MAX_INPUT_BODY_CHARS:
        return InputError(
            "body_too_long",
            f"본문은 {MAX_INPUT_BODY_CHARS:,}자 이하로 입력해 주세요. (현재 {len(body):,}자)",
        )
    return PredictInput(cleaned_title, clean_text(body) or None)


def parse_history_limit(raw: str | None) -> int:
    """기록 개수 요청값. 숫자가 아니면 기본값, 범위를 벗어나면 1~100 안으로 맞춘다."""
    try:
        limit = int(raw) if raw is not None else DEFAULT_HISTORY_LIMIT
    except ValueError:
        return DEFAULT_HISTORY_LIMIT
    return max(1, min(limit, MAX_HISTORY_LIMIT))
