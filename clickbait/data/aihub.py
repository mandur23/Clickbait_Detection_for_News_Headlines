"""AI Hub "낚시성 기사 탐지 데이터" 원본(JSON 파일, ZIP 묶음)을 읽어 Article 로 바꾼다.

원본 형식을 아는 곳은 이 파일뿐이다. 원본 라벨(clickbaitClass 0 = 낚시성)은 여기서만
프로젝트 라벨(schema.CLICKBAIT = 1)로 바꾼다. 읽을 수 없는 파일은 이유별로 세고 건너뛴다.
"""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from clickbait.schema import CLICKBAIT, NORMAL

# 원본 clickbaitClass 값 → 프로젝트 라벨.
_FIELD_LABELS: dict[int, int] = {0: CLICKBAIT, 1: NORMAL}

CATEGORY_NAMES: dict[str, str] = {
    "EC": "경제",
    "ET": "연예",
    "GB": "세계",
    "IS": "IT과학",
    "LC": "생활문화",
    "PO": "정치",
    "SO": "사회",
}

_PART = re.compile(r"Part([12])", re.IGNORECASE)
_SOURCE = re.compile(r"(NonClickbait|Clickbait)_(Auto|Direct)", re.IGNORECASE)
_CATEGORY = re.compile(r"(?:^|[\\/_])(" + "|".join(CATEGORY_NAMES) + r")(?=$|[\\/_.])")


@dataclass(frozen=True)
class Article:
    news_id: str
    title: str
    label: int
    category: str
    part: int | None
    source: str
    body: str
    # 원본 안에서의 위치(루트 기준 상대 경로, ZIP 이면 "묶음.zip!멤버").
    origin: str


@dataclass
class ReadStats:
    files: int = 0
    articles: int = 0
    skipped: Counter[str] = field(default_factory=Counter)
    # 경로가 말하는 라벨과 clickbaitClass 가 다른 수. 필드를 믿는다.
    label_path_mismatch: int = 0

    def to_json(self) -> dict[str, object]:
        return {
            "files": self.files,
            "articles": self.articles,
            "skipped": dict(sorted(self.skipped.items())),
            "label_path_mismatch": self.label_path_mismatch,
        }


@dataclass(frozen=True)
class _PathInfo:
    label: int | None
    part: int | None
    source: str
    category: str


def _path_info(origin: str) -> _PathInfo:
    source_match = _SOURCE.search(origin)
    label = None
    source = ""
    if source_match:
        kind, method = source_match.groups()
        label = NORMAL if kind.lower() == "nonclickbait" else CLICKBAIT
        source = f"{'NonClickbait' if label == NORMAL else 'Clickbait'}_{method.capitalize()}"
    part_match = _PART.search(origin)
    category_match = _CATEGORY.search(origin)
    return _PathInfo(
        label=label,
        part=int(part_match.group(1)) if part_match else None,
        source=source,
        category=CATEGORY_NAMES[category_match.group(1)] if category_match else "",
    )


def _parse_field_label(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return _FIELD_LABELS.get(value)
    if isinstance(value, str) and value.strip().isdigit():
        return _FIELD_LABELS.get(int(value.strip()))
    return None


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _parse_part(value: object) -> int | None:
    if value is None:
        return None
    match = re.fullmatch(r"(?:Part|P)?\s*0?([12])", str(value).strip(), re.IGNORECASE)
    return int(match.group(1)) if match else None


def parse_article(raw: bytes, origin: str, stats: ReadStats) -> Article | None:
    """JSON 한 건을 Article 로 바꾼다. 쓸 수 없으면 이유를 세고 None."""
    try:
        doc = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        stats.skipped["malformed_json"] += 1
        return None
    if not isinstance(doc, dict):
        stats.skipped["malformed_json"] += 1
        return None

    labeled = doc.get("labeledDataInfo")
    source_info = doc.get("sourceDataInfo")
    labeled = labeled if isinstance(labeled, dict) else {}
    source_info = source_info if isinstance(source_info, dict) else {}
    path = _path_info(origin)

    title = _text(labeled.get("newTitle")) or _text(source_info.get("newsTitle"))
    if not title:
        stats.skipped["missing_title"] += 1
        return None

    if "clickbaitClass" in labeled:
        label = _parse_field_label(labeled["clickbaitClass"])
        if label is None:
            stats.skipped["bad_label"] += 1
            return None
        if path.label is not None and path.label != label:
            stats.label_path_mismatch += 1
    elif path.label is not None:
        label = path.label
    else:
        stats.skipped["no_label"] += 1
        return None

    news_id = str(source_info.get("newsID") or "").strip()
    if not news_id:
        news_id = "path-" + hashlib.sha1(origin.encode("utf-8")).hexdigest()[:16]

    stats.articles += 1
    return Article(
        news_id=news_id,
        title=title,
        label=label,
        category=path.category or _text(source_info.get("newsCategory")),
        part=path.part if path.part is not None else _parse_part(source_info.get("partNum")),
        source=path.source or _text(source_info.get("processType")),
        body=_text(source_info.get("newsContent")),
        origin=origin,
    )


def _iter_raw(root: Path) -> Iterator[tuple[bytes, str]]:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        suffix = path.suffix.lower()
        if suffix == ".json":
            yield path.read_bytes(), rel
        elif suffix == ".zip":
            with zipfile.ZipFile(path) as archive:
                for name in sorted(archive.namelist()):
                    if name.lower().endswith(".json"):
                        yield archive.read(name), f"{rel}!{name}"


def iter_articles(root: Path, stats: ReadStats) -> Iterator[Article]:
    """root 아래의 JSON 과 ZIP 속 JSON 을 경로 순서대로 읽는다. 순서가 정해져 있어 결과가 매번 같다."""
    for raw, origin in _iter_raw(root):
        stats.files += 1
        article = parse_article(raw, origin, stats)
        if article is not None:
            yield article
