"""판별 기록을 SQLite 파일에 남긴다. 켜고 끄는 것은 run_server.py 설정이 정한다."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from clickbait.schema import Prediction

_CREATE = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    title TEXT NOT NULL,
    label TEXT NOT NULL,
    prob_clickbait REAL NOT NULL,
    model TEXT NOT NULL
)
"""


@dataclass(frozen=True)
class HistoryRecord:
    id: int
    created_at: str
    title: str
    label: str
    prob_clickbait: float
    model: str

    def to_json(self) -> dict[str, object]:
        return asdict(self)


class HistoryStore:
    """작업마다 새 연결을 연다. 시연용 서버라 동시 요청이 적어 이 정도로 충분하다."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn, conn:
            conn.execute(_CREATE)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def add(self, title: str, prediction: Prediction, model: str) -> None:
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT INTO predictions (created_at, title, label, prob_clickbait, model)"
                " VALUES (?, ?, ?, ?, ?)",
                (created_at, title, prediction.label, round(prediction.prob_clickbait, 4), model),
            )

    def recent(self, limit: int) -> list[HistoryRecord]:
        """최근 기록부터 돌려준다."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT id, created_at, title, label, prob_clickbait, model"
                " FROM predictions ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [HistoryRecord(*row) for row in rows]
