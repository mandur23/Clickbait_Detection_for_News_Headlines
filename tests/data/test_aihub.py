import json
import zipfile
from pathlib import Path

from clickbait.data.aihub import Article, ReadStats, iter_articles
from clickbait.schema import CLICKBAIT, NORMAL


def _doc(title: str | None = "정부, 예산안 발표", cls: object = 1, news_id: str | None = "N1", **source: object) -> str:
    labeled: dict[str, object] = {}
    if title is not None:
        labeled["newTitle"] = title
    if cls is not None:
        labeled["clickbaitClass"] = cls
    source_info: dict[str, object] = {"newsContent": "첫 문장.\n둘째 문장.", **source}
    if news_id is not None:
        source_info["newsID"] = news_id
    return json.dumps({"sourceDataInfo": source_info, "labeledDataInfo": labeled}, ensure_ascii=False)


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _read(root: Path) -> tuple[list[Article], ReadStats]:
    stats = ReadStats()
    return list(iter_articles(root, stats)), stats


def test_clickbait_class_zero_becomes_internal_clickbait(tmp_path: Path) -> None:
    _write(tmp_path, "Part1/train/Clickbait_Auto/EC/a.json", _doc(cls=0, news_id="A"))
    _write(tmp_path, "Part1/train/NonClickbait_Auto/EC/b.json", _doc(cls=1, news_id="B"))
    articles, _ = _read(tmp_path)
    assert {a.news_id: a.label for a in articles} == {"A": CLICKBAIT, "B": NORMAL}


def test_string_label_values_are_accepted(tmp_path: Path) -> None:
    _write(tmp_path, "Part1/train/Clickbait_Auto/EC/a.json", _doc(cls="0"))
    articles, _ = _read(tmp_path)
    assert [a.label for a in articles] == [CLICKBAIT]


def test_path_metadata_fills_part_category_and_source(tmp_path: Path) -> None:
    _write(tmp_path, "Part1/validation/Clickbait_Direct/PO/a.json", _doc(cls=0))
    [article], _ = _read(tmp_path)
    assert (article.part, article.category, article.source) == (1, "정치", "Clickbait_Direct")
    assert article.body == "첫 문장.\n둘째 문장."


def test_label_falls_back_to_path_when_field_is_absent(tmp_path: Path) -> None:
    _write(tmp_path, "Part1/train/NonClickbait_Auto/SO/a.json", _doc(cls=None, news_id="A"))
    _write(tmp_path, "Part1/train/Clickbait_Auto/SO/b.json", _doc(cls=None, news_id="B"))
    articles, _ = _read(tmp_path)
    assert {a.news_id: a.label for a in articles} == {"A": NORMAL, "B": CLICKBAIT}


def test_field_wins_over_path_and_disagreement_is_counted(tmp_path: Path) -> None:
    _write(tmp_path, "Part1/train/NonClickbait_Auto/SO/a.json", _doc(cls=0))
    [article], stats = _read(tmp_path)
    assert article.label == CLICKBAIT
    assert stats.label_path_mismatch == 1


def test_unusable_files_are_skipped_and_counted_by_reason(tmp_path: Path) -> None:
    _write(tmp_path, "Part1/train/Clickbait_Auto/SO/ok.json", _doc(cls=0))
    _write(tmp_path, "Part1/train/Clickbait_Auto/SO/broken.json", '{"sourceDataInfo": ')
    _write(tmp_path, "Part1/train/Clickbait_Auto/SO/list.json", "[1, 2]")
    _write(tmp_path, "Part1/train/Clickbait_Auto/SO/no_title.json", _doc(title="  "))
    _write(tmp_path, "Part1/train/Clickbait_Auto/SO/bad.json", _doc(cls=5))
    _write(tmp_path, "Part1/train/Clickbait_Auto/SO/bool.json", _doc(cls=True))
    _write(tmp_path, "loose/no_label.json", _doc(cls=None))
    articles, stats = _read(tmp_path)
    assert len(articles) == 1
    assert stats.files == 7
    assert stats.skipped == {"malformed_json": 2, "missing_title": 1, "bad_label": 2, "no_label": 1}


def test_original_title_is_used_when_new_title_is_missing(tmp_path: Path) -> None:
    _write(tmp_path, "Part1/train/NonClickbait_Auto/SO/a.json", _doc(title=None, newsTitle="원래 제목입니다"))
    [article], _ = _read(tmp_path)
    assert article.title == "원래 제목입니다"


def test_missing_news_id_gets_a_stable_id_from_the_path(tmp_path: Path) -> None:
    _write(tmp_path, "Part1/train/NonClickbait_Auto/SO/a.json", _doc(news_id=None))
    _write(tmp_path, "Part1/train/NonClickbait_Auto/SO/b.json", _doc(news_id=None))
    first, _ = _read(tmp_path)
    again, _ = _read(tmp_path)
    assert [a.news_id for a in first] == [a.news_id for a in again]
    assert len({a.news_id for a in first}) == 2


def test_json_members_inside_zip_are_read_with_metadata_from_zip_name(tmp_path: Path) -> None:
    with zipfile.ZipFile(tmp_path / "TL_Part1_Clickbait_Auto_EC.zip", "w") as archive:
        archive.writestr("x1.json", _doc(cls=0, news_id="Z1"))
        archive.writestr("x2.json", "not json")
        archive.writestr("readme.txt", "무시")
    articles, stats = _read(tmp_path)
    assert [(a.news_id, a.label, a.part, a.category, a.source) for a in articles] == [
        ("Z1", CLICKBAIT, 1, "경제", "Clickbait_Auto")
    ]
    assert stats.skipped == {"malformed_json": 1}


def test_part_comes_from_part_num_when_path_has_none(tmp_path: Path) -> None:
    _write(tmp_path, "loose/a.json", _doc(cls=1, news_id="A", partNum="2"))
    _write(tmp_path, "loose/b.json", _doc(cls=1, news_id="B", partNum="Part1"))
    articles, _ = _read(tmp_path)
    assert {a.news_id: a.part for a in articles} == {"A": 2, "B": 1}


def test_utf8_bom_is_tolerated(tmp_path: Path) -> None:
    path = tmp_path / "Part1/train/Clickbait_Auto/SO/a.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(_doc(cls=0).encode("utf-8-sig"))
    articles, _ = _read(tmp_path)
    assert len(articles) == 1
