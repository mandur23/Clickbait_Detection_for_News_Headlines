"""AI Hub 형식을 흉내 낸 합성 데이터 생성기.

실제 데이터는 AI Hub 승인 전이라 이 저장소에 없다. 이 데이터는 읽기·정제·학습·평가 코드가
끝까지 도는지 확인하려고 템플릿으로 만든 가짜 기사다. 여기서 나온 정확도·F1 같은 지표는
아무 의미가 없다. 경계 처리 분기(중복 제목, 라벨이 갈린 중복, 너무 짧은 제목, 깨진 JSON,
Part2 파일, ZIP 묶음, 라벨 필드 누락, 경로·필드 라벨 불일치 등)를 모두 한 번씩 지나가게 만든다.
"""

from __future__ import annotations

import json
import random
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

CATEGORIES: tuple[str, ...] = ("EC", "ET", "GB", "IS", "LC", "PO", "SO")
SAMPLE_SEED = 20240601
_ZIP_TIME = (2024, 1, 1, 0, 0, 0)

_SUBJECTS = {
    "EC": ["한국은행", "기획재정부", "코스피", "삼성전자", "현대차", "부동산 시장", "수출", "가계부채"],
    "ET": ["배우 김하늘", "가수 이준", "아이돌 그룹 루미", "예능 프로그램", "드라마 '봄날'", "영화 '바다'", "배우 정민"],
    "GB": ["미국 연준", "일본 정부", "유럽연합", "중국 당국", "유엔", "영국 총리", "프랑스 의회"],
    "IS": ["인공지능", "스마트폰 신제품", "반도체 업계", "우주항공청", "전기차 배터리", "클라우드 기업", "보안 업체"],
    "LC": ["여름 휴가철", "주말 날씨", "건강보험", "국립중앙박물관", "지역 축제", "커피 가격", "도서관"],
    "PO": ["국회", "여당", "야당", "대통령실", "선거관리위원회", "국무총리", "지방자치단체"],
    "SO": ["교육부", "경찰청", "서울시", "소방당국", "법원", "보건복지부", "고용노동부"],
}
_TOPICS = ["금리", "예산안", "신규 정책", "실적", "공급 대책", "안전 점검", "지원 사업", "개정안", "협약", "조사 결과"]
_ACTIONS = ["발표", "확정", "시행", "추진", "논의", "공개", "검토", "합의"]
_NUMBERS = ["3%", "10년 만에", "1조 원", "2배", "역대 최대", "5개월 연속", "20만 명"]

_NORMAL_TEMPLATES = [
    "{s}, {t} {a}",
    "{s} {t} {a}…내년부터 적용",
    "{s}, {t} {n} 증가",
    "{s} {t} 관련 {a} 예정",
    "{s}, {t} 두고 {a} 착수",
    "{s} \"{t} {a}할 것\"",
    "{s} {t}, 전문가 의견 엇갈려",
]
_CLICKBAIT_TEMPLATES = [
    "결국 터졌다…{s}의 충격 {t}",
    "{s}, 이것 모르면 손해…{t}의 비밀",
    "\"설마 했는데\" {s} {t} 알고 보니",
    "{s}의 {t}, 그 이유는?",
    "경악! {s} {t} 정체 드러나",
    "{s} {t}…네티즌 '난리났다'",
    "아무도 몰랐던 {s}의 {t}, 지금 확인하세요",
    "{s}가 숨겨온 {t}?! 충격 근황",
]
_SENTENCES = [
    "{s}은(는) 이날 {t}에 대한 입장을 밝혔다.",
    "관계자는 \"{t}은(는) 단계적으로 진행될 것\"이라고 설명했다.",
    "업계에서는 이번 {t}이(가) 시장에 미칠 영향에 주목하고 있다.",
    "{s}은(는) 다음 달 추가 내용을 공개할 계획이다.",
    "전문가들은 {t}의 효과를 지켜봐야 한다고 말했다.",
    "이번 결정은 지난해 논의된 내용을 바탕으로 한다.",
]


@dataclass(frozen=True)
class _Doc:
    relpath: str
    payload: str


def _article_json(news_id: str | None, title: str, cls: int | str | None, body: str, category: str) -> str:
    labeled: dict[str, object] = {"newTitle": title, "referSentenceInfo": []}
    if cls is not None:
        labeled["clickbaitClass"] = cls
    source: dict[str, object] = {
        "newsCategory": category,
        "newsContent": body,
        "sentenceInfo": [],
    }
    if news_id is not None:
        source["newsID"] = news_id
    return json.dumps({"sourceDataInfo": source, "labeledDataInfo": labeled}, ensure_ascii=False, indent=1)


def _title(rng: random.Random, category: str, clickbait: bool) -> str:
    template = rng.choice(_CLICKBAIT_TEMPLATES if clickbait else _NORMAL_TEMPLATES)
    return template.format(
        s=rng.choice(_SUBJECTS[category]),
        t=rng.choice(_TOPICS),
        a=rng.choice(_ACTIONS),
        n=rng.choice(_NUMBERS),
    )


def _body(rng: random.Random, category: str) -> str:
    s = rng.choice(_SUBJECTS[category])
    t = rng.choice(_TOPICS)
    return "\n".join(line.format(s=s, t=t) for line in rng.sample(_SENTENCES, k=rng.randint(3, 5)))


def _docs(n_articles: int, seed: int) -> list[_Doc]:
    rng = random.Random(seed)
    docs: list[_Doc] = []
    normal_titles: list[str] = []
    for i in range(n_articles):
        category = CATEGORIES[i % len(CATEGORIES)]
        clickbait = rng.random() < 0.45
        source = rng.choice(["Clickbait_Auto", "Clickbait_Direct"]) if clickbait else "NonClickbait_Auto"
        split = "validation" if rng.random() < 0.15 else "train"
        # 두 템플릿 묶음은 겹치지 않으므로 우연한 중복은 같은 라벨끼리만 생긴다. 일부는 반대쪽
        # 템플릿에 꼬리표를 붙여 일부러 헷갈리게 한다(오분류 목록이 비지 않게). 꼬리표 덕에 라벨 충돌은 없다.
        if rng.random() < 0.06:
            title = _title(rng, category, not clickbait) + " (종합)"
        else:
            title = _title(rng, category, clickbait)
        if not clickbait:
            normal_titles.append(title)
        cls = 0 if clickbait else 1
        news_id = f"SYN{i:05d}"
        relpath = f"Part1/{split}/{source}/{category}/{news_id}.json"
        docs.append(_Doc(relpath, _article_json(news_id, title, cls, _body(rng, category), category)))

    def extra(relpath: str, news_id: str | None, title: str, cls: int | str | None, category: str = "SO") -> None:
        docs.append(_Doc(relpath, _article_json(news_id, title, cls, _body(rng, category), category)))

    # 같은 라벨의 중복 제목 다섯 건(공백만 다른 것 포함).
    for k, title in enumerate(normal_titles[:5]):
        variant = title.replace(" ", "  ", 1) if k % 2 else title
        extra(f"Part1/train/NonClickbait_Auto/SO/DUP{k:03d}.json", f"DUP{k:03d}", variant, 1)
    # 라벨이 갈린 중복: 같은 제목이 낚시성 폴더와 비낚시성 폴더에 하나씩.
    conflict = "정부, 내년 예산안 국회 제출…총지출 소폭 증가"
    extra("Part1/train/NonClickbait_Auto/PO/CONFLICT_A.json", "CONFLICT_A", conflict, 1, "PO")
    extra("Part1/train/Clickbait_Auto/PO/CONFLICT_B.json", "CONFLICT_B", conflict, 0, "PO")
    extra("Part1/train/NonClickbait_Auto/EC/SHORT.json", "SHORT", "속보", 1, "EC")
    extra("Part1/train/NonClickbait_Auto/EC/NO_ID.json", None, "기획재정부, 하반기 경제정책 방향 발표", 1, "EC")
    # 라벨 필드가 없으면 경로(NonClickbait)로 정한다.
    extra("Part1/train/NonClickbait_Auto/LC/NO_FIELD.json", "NO_FIELD", "국립중앙박물관, 여름 특별전 개최", None, "LC")
    # 경로는 비낚시성인데 필드는 낚시성: 필드를 믿고 불일치로 센다.
    extra("Part1/train/NonClickbait_Auto/ET/MISMATCH.json", "MISMATCH", "충격! 배우 정민 결국 은퇴?", 0, "ET")
    extra("Part1/train/Clickbait_Auto/IS/BAD_LABEL.json", "BAD_LABEL", "반도체 업계 실적 공개", 7, "IS")
    extra("Part1/train/Clickbait_Auto/IS/NO_TITLE.json", "NO_TITLE", "", 0, "IS")
    extra("Part2/train/Clickbait_Auto/EC/PART2.json", "PART2", "한국은행, 기준금리 동결 결정", 0, "EC")
    docs.append(_Doc("Part1/train/Clickbait_Auto/GB/BROKEN.json", '{"sourceDataInfo": {"newsID": "BROKEN", '))
    return docs


def write_sample(out_dir: Path, n_articles: int = 600, seed: int = SAMPLE_SEED) -> int:
    """out_dir 를 지우고 다시 만든다. 같은 인자면 바이트 단위로 같은 결과가 나온다. 쓴 문서 수를 돌려준다."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    docs = _docs(n_articles, seed)
    # 실제 내려받은 형태처럼 일부(Clickbait_Direct 의 IS)는 폴더 대신 ZIP 묶음으로 둔다.
    zipped_prefix = "Part1/train/Clickbait_Direct/IS/"
    zip_path = out_dir / "TL_Part1_Clickbait_Direct_IS.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for doc in docs:
            if doc.relpath.startswith(zipped_prefix):
                info = zipfile.ZipInfo(doc.relpath.removeprefix(zipped_prefix), date_time=_ZIP_TIME)
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, doc.payload.encode("utf-8"))
    for doc in docs:
        if doc.relpath.startswith(zipped_prefix):
            continue
        path = out_dir / doc.relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(doc.payload.encode("utf-8"))
    return len(docs)
