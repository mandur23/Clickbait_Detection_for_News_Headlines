import hashlib
from pathlib import Path

from clickbait.data.aihub import ReadStats, iter_articles
from clickbait.data.prepare import prepare
from clickbait.data.sample import write_sample


def _digest(root: Path) -> str:
    h = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            h.update(path.relative_to(root).as_posix().encode())
            h.update(path.read_bytes())
    return h.hexdigest()


def test_sample_generation_is_deterministic(tmp_path: Path) -> None:
    write_sample(tmp_path / "a", n_articles=50)
    write_sample(tmp_path / "b", n_articles=50)
    assert _digest(tmp_path / "a") == _digest(tmp_path / "b")


def test_sample_exercises_every_boundary_branch(tmp_path: Path) -> None:
    write_sample(tmp_path, n_articles=600)
    stats = ReadStats()
    articles = list(iter_articles(tmp_path, stats))
    assert stats.skipped == {"malformed_json": 1, "missing_title": 1, "bad_label": 1}
    assert stats.label_path_mismatch == 1
    assert any(".zip!" in a.origin for a in articles)
    assert any(a.part == 2 for a in articles)
    assert any(a.news_id.startswith("path-") for a in articles)

    report = prepare(articles).report
    assert report["dedupe"]["conflict_keys"] == 1
    assert report["dedupe"]["duplicates_removed"] >= 5
    stages = report["stages"]
    assert stages["after_parts"] == stages["read"] - 1
    assert stages["after_length"] == stages["after_parts"] - 1
    ratio = report["class_ratio_before_balance"]["clickbait"]["ratio"]
    assert 0.35 < ratio < 0.55
