"""제목·본문 정리. 학습 데이터와 서버 입력이 같은 함수를 거쳐야 모델이 본 형태와 같아진다."""

from __future__ import annotations

import html
import re
import unicodedata

_TAG = re.compile(r"<[^>]+>")
_CONTROL = re.compile(r"[\u0000-\u0008\u000b-\u001f\u007f​-‏﻿]")
_SPACE = re.compile(r"\s+")
_REPEATED_MARK = re.compile(r"([!?.,~])\1{2,}")
_QUOTES = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "„": '"',
        "‘": "'",
        "’": "'",
        "`": "'",
        "…": "...",
        "‧": "·",
        "ㆍ": "·",
    }
)


def clean_text(text: str) -> str:
    """HTML 흔적, 제어 문자, 중복 공백을 없앤다. 물음표·느낌표 같은 표현 단서는 남긴다."""
    text = html.unescape(text)
    text = _TAG.sub(" ", text)
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_QUOTES)
    text = _CONTROL.sub("", text)
    text = _REPEATED_MARK.sub(r"\1\1\1", text)
    return _SPACE.sub(" ", text).strip()


def clean_title(title: str) -> str:
    return clean_text(title)


def dedup_key(title: str) -> str:
    """중복 제목 판정용 키. 공백과 대소문자 차이는 같은 제목으로 본다."""
    return _SPACE.sub("", clean_title(title)).casefold()
