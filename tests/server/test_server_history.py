from __future__ import annotations

from pathlib import Path

from clickbait.schema import Prediction
from server.history import HistoryStore


def test_reopening_keeps_records(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "history.db"
    HistoryStore(path).add("제목 하나", Prediction(prob_clickbait=0.9), "m")
    reopened = HistoryStore(path)
    reopened.add("제목 둘", Prediction(prob_clickbait=0.1), "m")

    records = reopened.recent(10)
    assert [(r.title, r.label) for r in records] == [("제목 둘", "normal"), ("제목 하나", "clickbait")]


def test_recent_respects_limit(tmp_path: Path) -> None:
    store = HistoryStore(tmp_path / "history.db")
    for i in range(5):
        store.add(f"제목 {i}", Prediction(prob_clickbait=0.5), "m")
    assert [r.title for r in store.recent(2)] == ["제목 4", "제목 3"]
