import json
from pathlib import Path

import pandas as pd

from clickbait.data.prepare import read_split
from clickbait.modeling.baseline import BaselineModel
from clickbait.modeling.evaluate import comparison_table, evaluate, load_scorer, write_report


def test_baseline_trains_saves_reloads_and_evaluates(processed_dir: Path, tmp_path: Path) -> None:
    train = read_split(processed_dir / "train.csv")
    test = read_split(processed_dir / "test.csv")
    model = BaselineModel.train(train["title"], train["label"], min_df=2)
    model.save(tmp_path / "baseline")

    scorer = load_scorer(tmp_path / "baseline")
    prob = scorer.predict_proba(test["title"].tolist())
    assert prob.shape == (len(test),)
    assert ((prob >= 0) & (prob <= 1)).all()

    result = evaluate(scorer, test)
    # 합성 데이터는 일부러 섞은 6% 를 빼면 템플릿으로 갈린다. 학습이 된다는 확인일 뿐 실제 성능과는 무관하다.
    assert result.metrics["f1"] > 0.75
    assert sum(sum(row) for row in result.metrics["confusion_matrix"]) == len(test)


def test_report_files_and_misclassified_ordering(tmp_path: Path) -> None:
    class Fixed:
        name = "fixed"
        accepts_body = False

        def predict_proba(self, titles, bodies=None):
            return pd.Series([0.9, 0.6, 0.2, 0.05]).to_numpy()

    df = pd.DataFrame(
        {
            "news_id": ["a", "b", "c", "d"],
            "title": ["확신한 오탐", "덜 확신한 오탐", "놓친 낚시", "맞힌 일반"],
            "label": [0, 0, 1, 0],
            "category": ["경제", "정치", "사회", "연예"],
        }
    )
    result = evaluate(Fixed(), df)
    assert result.metrics["confusion_matrix"] == [[1, 2], [1, 0]]
    wrong = result.misclassified
    assert list(wrong["title"]) == ["확신한 오탐", "놓친 낚시", "덜 확신한 오탐"]
    assert list(wrong["error"]) == ["FP", "FN", "FP"]

    out = tmp_path / "reports" / "fixed"
    write_report(result, out, {"name": "fixed", "model": "fixed", "input_mode": "title", "eval_csv": "test.csv"})
    assert {p.name for p in out.iterdir()} == {"metrics.json", "confusion_matrix.png", "misclassified.csv"}
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["f1"] == 0.0 and metrics["accuracy"] == 0.25
    assert list(pd.read_csv(out / "misclassified.csv", encoding="utf-8-sig").columns) == [
        "title", "true", "pred", "prob_clickbait", "category", "error"
    ]
    assert "| fixed | fixed | title | test.csv | 4 |" in comparison_table(tmp_path / "reports")
