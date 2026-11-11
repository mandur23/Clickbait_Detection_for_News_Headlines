import json
from pathlib import Path

import pandas as pd
import pytest

from clickbait.config import MAX_TITLE_CHARS, MIN_TITLE_CHARS
from clickbait.data.aihub import Article, ReadStats, iter_articles
from clickbait.data.prepare import prepare, read_bodies, read_split, write_prepared
from clickbait.data.sample import write_sample
from clickbait.schema import CLICKBAIT, HEADLINE_COLUMNS, NORMAL
from clickbait.text import dedup_key


def _article(i: int, title: str, label: int, part: int | None = 1) -> Article:
    return Article(
        news_id=f"N{i:04d}",
        title=title,
        label=label,
        category="경제",
        part=part,
        source="Clickbait_Auto" if label == CLICKBAIT else "NonClickbait_Auto",
        body=f"<p>본문 {i}</p>",
        origin=f"x/{i}.json",
    )


def _corpus(n_normal: int = 120, n_clickbait: int = 80) -> list[Article]:
    normal = [_article(i, f"정부 정책 발표 제{i}호", NORMAL) for i in range(n_normal)]
    clickbait = [_article(1000 + i, f"결국 터졌다 충격 {i}번째", CLICKBAIT) for i in range(n_clickbait)]
    return normal + clickbait


def _all_rows(data) -> pd.DataFrame:
    return pd.concat(data.splits.values(), ignore_index=True)


def test_titles_are_cleaned_and_length_limits_applied() -> None:
    articles = _corpus() + [
        _article(5000, "<b>정부</b>   예산안&amp;세제 개편", NORMAL),
        _article(5001, "가" * (MIN_TITLE_CHARS - 1), NORMAL),
        _article(5002, "나" * (MAX_TITLE_CHARS + 1), NORMAL),
        _article(5003, "다" * MAX_TITLE_CHARS, NORMAL),
    ]
    data = prepare(articles)
    rows = _all_rows(data)
    assert "정부 예산안&세제 개편" in set(rows["title"])
    assert "다" * MAX_TITLE_CHARS in set(rows["title"])
    assert rows["title"].str.len().between(MIN_TITLE_CHARS, MAX_TITLE_CHARS).all()
    assert data.report["stages"]["after_length"] == data.report["stages"]["after_parts"] - 2


def test_duplicates_keep_one_and_conflicting_labels_drop_all() -> None:
    articles = _corpus() + [
        _article(6000, "같은 제목 기사입니다", NORMAL),
        _article(6001, "같은  제목 기사입니다", NORMAL),
        _article(6002, "엇갈린 라벨 제목", NORMAL),
        _article(6003, "엇갈린 라벨 제목", CLICKBAIT),
    ]
    data = prepare(articles)
    keys = _all_rows(data)["title"].map(dedup_key)
    assert (keys == dedup_key("같은 제목 기사입니다")).sum() == 1
    assert (keys == dedup_key("엇갈린 라벨 제목")).sum() == 0
    assert data.report["dedupe"] == {"duplicates_removed": 1, "conflict_keys": 1, "conflict_rows": 2}


def test_only_requested_parts_are_kept() -> None:
    articles = _corpus() + [_article(7000, "파트2 기사 제목", CLICKBAIT, part=2), _article(7001, "파트 모름 제목", NORMAL, part=None)]
    titles = set(_all_rows(prepare(articles))["title"])
    assert "파트2 기사 제목" not in titles and "파트 모름 제목" not in titles
    assert "파트2 기사 제목" in set(_all_rows(prepare(articles, parts=[1, 2]))["title"])


def test_split_is_8_1_1_stratified_and_titles_never_cross_splits() -> None:
    data = prepare(_corpus(600, 400))
    sizes = {name: len(df) for name, df in data.splits.items()}
    assert sizes == {"train": 800, "val": 100, "test": 100}
    for df in data.splits.values():
        assert df["label"].mean() == pytest.approx(0.4, abs=0.02)
    keys = [set(df["title"].map(dedup_key)) for df in data.splits.values()]
    assert not (keys[0] & keys[1]) and not (keys[0] & keys[2]) and not (keys[1] & keys[2])


def test_same_seed_gives_same_split_and_other_seed_differs() -> None:
    a = prepare(_corpus(), seed=1)
    b = prepare(_corpus(), seed=1)
    c = prepare(_corpus(), seed=2)
    for name in ("train", "val", "test"):
        pd.testing.assert_frame_equal(a.splits[name], b.splits[name])
    assert list(a.splits["test"]["news_id"]) != list(c.splits["test"]["news_id"])


def test_balance_downsamples_majority_to_minority() -> None:
    data = prepare(_corpus(150, 50), balance=True)
    counts = _all_rows(data)["label"].value_counts()
    assert counts[NORMAL] == counts[CLICKBAIT] == 50
    assert data.report["class_ratio_before_balance"]["normal"]["count"] == 150


def test_sample_keeps_class_ratio() -> None:
    rows = _all_rows(prepare(_corpus(600, 400), sample=200))
    assert len(rows) == 200
    assert rows["label"].mean() == pytest.approx(0.4, abs=0.01)


def test_too_little_data_is_reported_not_split() -> None:
    with pytest.raises(ValueError):
        prepare(_corpus(30, 3))


def test_written_files_have_contract_columns_and_rerun_is_identical(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    write_sample(raw, n_articles=200)

    def run(out: Path) -> dict[str, bytes]:
        stats = ReadStats()
        write_prepared(prepare(iter_articles(raw, stats)), out)
        return {p.name: p.read_bytes() for p in sorted(out.iterdir())}

    first = run(tmp_path / "out")
    second = run(tmp_path / "out")
    assert first == second
    assert set(first) == {"train.csv", "val.csv", "test.csv", "bodies.jsonl", "data_report.json"}

    out = tmp_path / "out"
    train = read_split(out / "train.csv")
    assert tuple(train.columns) == HEADLINE_COLUMNS
    bodies = read_bodies(out / "bodies.jsonl")
    all_ids = pd.concat([read_split(out / f"{n}.csv") for n in ("train", "val", "test")])["news_id"]
    assert set(all_ids) == set(bodies)
    assert all("\n" not in body for body in bodies.values())
    report = json.loads((out / "data_report.json").read_text(encoding="utf-8"))
    assert report["split_sizes"]["train"] == len(train)
