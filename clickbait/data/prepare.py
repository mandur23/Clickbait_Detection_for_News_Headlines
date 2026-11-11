"""읽어 온 기사를 학습용 표로 정제하고 8:1:1 로 나눈다.

모든 단계가 DataFrame 을 받아 DataFrame 을 돌려주는 순수 함수다. 같은 입력과 seed 면 결과가 같다.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from clickbait.config import MAX_TITLE_CHARS, MIN_TITLE_CHARS, SEED
from clickbait.data.aihub import Article
from clickbait.schema import HEADLINE_COLUMNS, LABEL_NAMES
from clickbait.text import clean_text, clean_title, dedup_key

SPLIT_NAMES: tuple[str, str, str] = ("train", "val", "test")
BODIES_FILENAME = "bodies.jsonl"
REPORT_FILENAME = "data_report.json"


@dataclass
class PreparedData:
    splits: dict[str, pd.DataFrame]
    # news_id, body. 학습 표에 남은 기사만 들어 있다.
    bodies: pd.DataFrame
    report: dict[str, object] = field(default_factory=dict)


def articles_to_frame(articles: Iterable[Article]) -> pd.DataFrame:
    rows = [
        {
            "news_id": a.news_id,
            "title": a.title,
            "label": a.label,
            "category": a.category,
            "part": a.part,
            "source": a.source,
            "body": a.body,
        }
        for a in articles
    ]
    return pd.DataFrame(rows, columns=[*HEADLINE_COLUMNS, "body"])


def select_parts(df: pd.DataFrame, parts: Iterable[int]) -> pd.DataFrame:
    return df[df["part"].isin(list(parts))].reset_index(drop=True)


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["title"] = out["title"].map(clean_title)
    out["body"] = out["body"].map(clean_text)
    out["part"] = out["part"].astype(int)
    return out


def filter_title_length(df: pd.DataFrame) -> pd.DataFrame:
    length = df["title"].str.len()
    return df[(length >= MIN_TITLE_CHARS) & (length <= MAX_TITLE_CHARS)].reset_index(drop=True)


@dataclass(frozen=True)
class DedupeResult:
    frame: pd.DataFrame
    duplicates_removed: int
    conflict_keys: int
    conflict_rows: int


def dedupe_titles(df: pd.DataFrame) -> DedupeResult:
    """같은 제목은 처음 것만 남긴다. 같은 제목에 라벨이 갈리면 정답을 알 수 없으므로 모두 버린다."""
    keys = df["title"].map(dedup_key)
    labels_per_key = df.groupby(keys)["label"].nunique()
    conflicted = set(labels_per_key[labels_per_key > 1].index)
    in_conflict = keys.isin(conflicted)
    clean = df[~in_conflict]
    deduped = clean[~keys[~in_conflict].duplicated(keep="first")]
    return DedupeResult(
        frame=deduped.reset_index(drop=True),
        duplicates_removed=len(clean) - len(deduped),
        conflict_keys=len(conflicted),
        conflict_rows=int(in_conflict.sum()),
    )


def balance_classes(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """많은 쪽 클래스를 적은 쪽 수만큼 무작위로 줄인다. 원래 순서는 유지한다."""
    n = int(df["label"].value_counts().min())
    picked = df.groupby("label", group_keys=False).sample(n=n, random_state=seed)
    return df.loc[sorted(picked.index)].reset_index(drop=True)


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """라벨 비율을 유지한 채 n 건을 뽑는다. n 이 전체 이상이면 그대로 돌려준다."""
    if n >= len(df):
        return df.reset_index(drop=True)
    picked, _ = train_test_split(df, train_size=n, stratify=df["label"], random_state=seed)
    return df.loc[sorted(picked.index)].reset_index(drop=True)


def split_frame(df: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    """라벨 비율을 유지한 8:1:1 분할."""
    train, rest = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=seed)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest["label"], random_state=seed)
    return {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def class_ratio(df: pd.DataFrame) -> dict[str, object]:
    counts = df["label"].value_counts()
    total = int(counts.sum())
    return {
        name: {"count": int(counts.get(label, 0)), "ratio": round(int(counts.get(label, 0)) / total, 4) if total else 0.0}
        for label, name in enumerate(LABEL_NAMES)
    }


def title_length_stats(df: pd.DataFrame) -> dict[str, float]:
    length = df["title"].str.len()
    return {
        "min": int(length.min()),
        "max": int(length.max()),
        "mean": round(float(length.mean()), 2),
        "median": float(length.median()),
        "p95": float(length.quantile(0.95)),
    }


def prepare(
    articles: Iterable[Article],
    *,
    parts: Iterable[int] = (1,),
    balance: bool = False,
    sample: int | None = None,
    seed: int = SEED,
) -> PreparedData:
    stages: dict[str, int] = {}
    df = articles_to_frame(articles)
    stages["read"] = len(df)
    df = clean_frame(select_parts(df, parts))
    stages["after_parts"] = len(df)
    df = filter_title_length(df)
    stages["after_length"] = len(df)
    dedupe = dedupe_titles(df)
    df = dedupe.frame
    stages["after_dedupe"] = len(df)
    ratio_before_balance = class_ratio(df)
    if balance:
        df = balance_classes(df, seed)
    stages["after_balance"] = len(df)
    if sample is not None:
        df = stratified_sample(df, sample, seed)
    stages["after_sample"] = len(df)
    if df["label"].value_counts().min() < 10 or df["label"].nunique() < 2:
        raise ValueError(f"정제 후 남은 기사가 너무 적어 나눌 수 없다(단계별 건수: {stages})")

    splits = split_frame(df, seed)
    bodies = df[["news_id", "body"]].drop_duplicates("news_id", keep="first").reset_index(drop=True)
    report: dict[str, object] = {
        "stages": stages,
        "dedupe": {
            "duplicates_removed": dedupe.duplicates_removed,
            "conflict_keys": dedupe.conflict_keys,
            "conflict_rows": dedupe.conflict_rows,
        },
        # 같은 news_id 에 서로 다른 제목이 남은 수. bodies.jsonl 은 첫 본문만 쓴다.
        "duplicate_news_ids": int(df["news_id"].duplicated().sum()),
        "class_ratio_before_balance": ratio_before_balance,
        "class_ratio": class_ratio(df),
        "title_length": title_length_stats(df),
        "split_sizes": {name: len(frame) for name, frame in splits.items()},
        "split_class_ratio": {name: class_ratio(frame) for name, frame in splits.items()},
        "options": {"parts": sorted(parts), "balance": balance, "sample": sample, "seed": seed},
    }
    return PreparedData(splits=splits, bodies=bodies, report=report)


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def write_prepared(data: PreparedData, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in SPLIT_NAMES:
        csv = data.splits[name][list(HEADLINE_COLUMNS)].to_csv(index=False, lineterminator="\n")
        _write_atomic(out_dir / f"{name}.csv", csv)
    lines = (
        json.dumps({"news_id": row.news_id, "body": row.body}, ensure_ascii=False)
        for row in data.bodies.itertuples(index=False)
    )
    _write_atomic(out_dir / BODIES_FILENAME, "".join(line + "\n" for line in lines))
    _write_atomic(out_dir / REPORT_FILENAME, json.dumps(data.report, ensure_ascii=False, indent=2) + "\n")


def read_split(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"news_id": str, "title": str, "category": str, "source": str}, keep_default_na=False)


def read_bodies(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8") as f:
        rows = (json.loads(line) for line in f if line.strip())
        return {row["news_id"]: row["body"] for row in rows}
