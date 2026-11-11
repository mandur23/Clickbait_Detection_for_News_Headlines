# 뉴스 제목 낚시성 탐지 시스템

뉴스 제목을 입력하면 낚시성 제목인지 판별하고 낚시성/일반 확률을 함께 보여 주는 웹 프로그램이다. 딥러닝 1조 개발 계획서(2026. 11. 11)의 기본 기능과 선택 기능(본문 비교, SQLite 기록)을 구현했다.

## 지금 상태

- 코드는 전 구간이 동작한다. 정제 → 기준 모델 → KLUE-BERT/KoBERT 학습 → 평가 → 서버 → 화면.
- **실제 AI Hub 데이터로는 아직 학습하지 않았다.** 데이터 이용 승인이 필요해서다. 저장소에 들어 있는 것은 코드 확인용 합성 표본 생성기이고, 그 표본으로 낸 지표는 의미가 없다.
- 학습된 모델이 `models/klue-bert` 에 없으면 서버는 **데모 모드**(가짜 판별기)로 뜨고 화면에 그 사실을 표시한다.

## 설치

Python 3.10 이상.

```bash
python -m venv .venv
```

```bash
.venv\Scripts\python -m pip install -r requirements.txt
```

로컬 GPU로 학습하려면 CUDA용 PyTorch를 따로 설치한다(기본으로는 CPU 빌드가 들어간다). 서버 실행과 표본 학습은 CPU 빌드로 충분하다.

```bash
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
```

## 바로 돌려 보기 (합성 표본)

AI Hub 형식을 흉내 낸 합성 기사 약 600건으로 전체 흐름을 확인한다.

```bash
.venv\Scripts\python scripts/make_sample_data.py
```

```bash
.venv\Scripts\python scripts/prepare_data.py --raw-dir data/sample_aihub
```

```bash
.venv\Scripts\python scripts/train_baseline.py
```

```bash
.venv\Scripts\python scripts/train_bert.py --epochs 2 --output-dir models/klue-bert-sample
```

```bash
.venv\Scripts\python scripts/evaluate.py --model-dir models/klue-bert-sample
```

```bash
.venv\Scripts\python scripts/compare_models.py
```

## 실제 데이터로 학습 (Colab)

1. AI Hub에서 "낚시성 기사 탐지 데이터"를 신청해 받는다.
2. 라벨링 데이터 ZIP 중 Part1(`TL_Part1_*.zip`, `VL_Part1_*.zip`)을 Google Drive의 `clickbait/aihub_raw/` 에 올린다. 압축은 풀지 않아도 된다.
3. 이 저장소를 GitHub에 올리고 `notebooks/train_colab.ipynb` 를 Colab에서 연다. `REPO_URL` 을 채우고 위에서부터 실행한다.
4. 노트북이 정제, 기준 모델, KLUE-BERT 표본 학습, KLUE-BERT·KoBERT 전체 학습, 평가, 비교표까지 돌리고 가장 좋은 모델을 Drive의 `clickbait/best_model/` 에 복사한다. 세션이 끊기면 같은 셀을 다시 실행하면 체크포인트에서 이어 간다.
5. `best_model` 폴더를 내려받아 이 저장소의 `models/klue-bert/` 로 둔다. 서버를 다시 시작하면 실제 모델로 판별한다.

첫 실행 때 `data/processed/data_report.json` 을 꼭 확인한다. 건너뛴 파일 수, 경로와 라벨이 어긋난 수, 클래스 비율, 제목 길이 분포가 들어 있다. 클래스 비율 차이가 크면 `prepare_data.py --balance` 나 `train_bert.py --class-weights` 를 쓴다.

Part2는 본문을 조작한 데이터라 제목만 보는 모델에는 라벨이 잡음이다. 그래서 기본값은 Part1만 쓴다(`--parts 1 2` 로 바꿀 수 있다).

## 서버 실행

```bash
.venv\Scripts\python run_server.py
```

브라우저에서 http://127.0.0.1:5000 을 연다.

| 환경 변수 | 뜻 | 기본값 |
|---|---|---|
| `CLICKBAIT_MODEL_DIR` | 학습된 모델 폴더 | `models/klue-bert` |
| `CLICKBAIT_HISTORY_DB` | 판별 기록 SQLite 파일. `off` 면 기록 안 함 | `instance/history.db` |
| `HOST`, `PORT` | 주소 | `127.0.0.1`, `5000` |

### API

`POST /predict` 에 `{"title": "..."}` 를 보내면 다음을 돌려준다.

```json
{"label": "clickbait", "prob_clickbait": 0.93, "prob_normal": 0.07}
```

`label` 은 `clickbait` 또는 `normal` 이다. 빈 제목, 200자를 넘는 제목, JSON이 아닌 요청은 모델에 보내지 않고 400과 `{"error": {"code", "message"}}` 를 돌려준다. `GET /api/info` 는 사용 중인 모델 정보를, `GET /api/history` 는 최근 판별 기록을 준다.

## 선택 기능

- **본문 비교**: `train_bert.py --input-mode title_body` 로 학습한 모델을 서버에 연결하면 화면에 본문 입력칸이 나타나고, 제목과 본문을 함께 보고 판별한다.
- **판별 기록**: 기본으로 켜져 있다. `CLICKBAIT_HISTORY_DB=off` 로 끈다.

## 폴더 구조

```
clickbait/
  schema.py        라벨 규칙, 판별 결과 형식
  config.py        경로, 입력 제한, 학습 초기값(최대 토큰 64, 배치 32, 학습률 2e-5, 에폭 3)
  text.py          제목 정리(학습과 서버가 같은 함수를 쓴다)
  predictor.py     판별기 인터페이스, 데모용 가짜 판별기
  data/            AI Hub 원본 읽기, 정제, 8:1:1 층화 분할
  modeling/        기준 모델, BERT 학습, 평가, 추론, KoBERT 토크나이저
scripts/           위 기능을 실행하는 명령들
server/            Flask 앱, 입력 검사, 기록, 화면(HTML/CSS/JS)
notebooks/         Colab 학습 노트북
tests/             pytest
```

라벨은 프로젝트 안에서 `0 = 일반`, `1 = 낚시성` 이다. AI Hub 원본(`clickbaitClass` 0 = 낚시성)과 반대이며, 변환은 `clickbait/data/aihub.py` 한 곳에서만 한다.

## 산출물 위치

| 계획서 산출물 | 위치 |
|---|---|
| 전처리 코드, 정제 데이터 | `clickbait/data/`, `data/processed/*.csv` |
| 학습·평가 코드 | `clickbait/modeling/`, `notebooks/train_colab.ipynb` |
| 평가 결과, 혼동 행렬, 오분류 목록 | `reports/<모델>/metrics.json`, `confusion_matrix.png`, `misclassified.csv`, `reports/comparison.md` |
| 학습된 모델 | `models/<모델>/` (Transformers 저장 형식) |
| 웹 화면, 서버 | `server/`, `run_server.py` |

## 테스트

```bash
.venv\Scripts\python -m pytest -q
```

## 알아 둘 점

- AI Hub 원본에서 직접 확인한 키는 `labeledDataInfo.newTitle`, `clickbaitClass`, `sourceDataInfo.newsContent` 다(공개 베이스라인 코드 기준). `newsID`, `newsCategory`, `partNum` 은 있으면 쓰고 없어도 동작하게 했다. 실제 데이터를 처음 읽을 때 `data_report.json` 으로 확인한다.
- KoBERT(`skt/kobert-base-v1`)는 허브 설정 그대로 불러오면 한글이 대부분 `[UNK]` 가 된다. `clickbait/modeling/kobert_tokenizer.py` 가 이를 바로잡는다. `scripts/check_kobert.py` 로 확인할 수 있다.
- 본문까지 읽어 표로 만들기 때문에 전체 데이터 정제는 메모리를 많이 쓴다. Colab에서 부족하면 `--sample` 로 줄여서 시작한다.
