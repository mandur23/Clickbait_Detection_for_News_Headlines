from pathlib import Path

import pytest

from clickbait.data.aihub import ReadStats, iter_articles
from clickbait.data.prepare import prepare, read_split, write_prepared
from clickbait.data.sample import write_sample


@pytest.fixture(scope="session")
def processed_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """합성 샘플을 정제한 data/processed 와 같은 모양의 폴더."""
    root = tmp_path_factory.mktemp("sample")
    write_sample(root / "raw", n_articles=600)
    write_prepared(prepare(iter_articles(root / "raw", ReadStats())), root / "processed")
    return root / "processed"


@pytest.fixture(scope="session")
def tiny_bert_dir(tmp_path_factory: pytest.TempPathFactory, processed_dir: Path) -> Path:
    """네트워크 없이 만든 아주 작은 무작위 BERT. 어휘는 샘플 데이터의 글자다."""
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

    root = tmp_path_factory.mktemp("tiny-bert")
    text = "".join(read_split(processed_dir / f"{n}.csv")["title"].str.cat() for n in ("train", "val", "test"))
    text += (processed_dir / "bodies.jsonl").read_text(encoding="utf-8")
    chars = sorted({c for c in text if not c.isspace()})
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *chars, *(f"##{c}" for c in chars)]
    vocab_file = root / "vocab.txt"
    vocab_file.write_text("\n".join(vocab) + "\n", encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(vocab_file), do_lower_case=False)
    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=128,
        num_labels=2,
    )
    model_dir = root / "model"
    BertForSequenceClassification(config).save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    return model_dir
