from pathlib import Path

import sentencepiece as spm

from clickbait.data.prepare import read_split
from clickbait.modeling.models import FAMILIES


def _tiny_spiece(processed_dir: Path, out: Path) -> Path:
    titles = read_split(processed_dir / "train.csv")["title"].tolist()
    # 실제 KoBERT 의 spiece.model 처럼 [UNK]=0, [PAD]=1, [CLS]=2, [SEP]=3, [MASK]=4 로 둔다.
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(titles),
        model_prefix=str(out / "spiece"),
        vocab_size=300,
        unk_id=0,
        unk_piece="[UNK]",
        pad_id=1,
        pad_piece="[PAD]",
        bos_id=-1,
        eos_id=-1,
        control_symbols=["[CLS]", "[SEP]", "[MASK]"],
        character_coverage=1.0,
    )
    return out / "spiece.model"


def test_kobert_tokenizer_uses_bert_order_and_reloads_from_saved_dir(processed_dir: Path, tmp_path: Path) -> None:
    hub_like = tmp_path / "hub"
    hub_like.mkdir()
    _tiny_spiece(processed_dir, hub_like)
    load = FAMILIES["kobert"].load_tokenizer
    tokenizer = load(str(hub_like))

    single = tokenizer("결국 터졌다 충격 근황")
    tokens = tokenizer.convert_ids_to_tokens(single["input_ids"])
    assert tokens[0] == "[CLS]" and tokens[-1] == "[SEP]"
    assert single["input_ids"][0] == 2 and single["input_ids"][-1] == 3
    assert tokenizer.unk_token_id not in single["input_ids"]

    pair = tokenizer("정부 예산안 발표", "관계자는 설명했다")
    pair_tokens = tokenizer.convert_ids_to_tokens(pair["input_ids"])
    first_sep = pair_tokens.index("[SEP]")
    assert pair_tokens[0] == "[CLS]" and pair_tokens[-1] == "[SEP]" and pair_tokens.count("[SEP]") == 2
    assert pair["token_type_ids"] == [0] * (first_sep + 1) + [1] * (len(pair_tokens) - first_sep - 1)
    assert tokenizer.pad_token_id == 1
    assert len(tokenizer) == 300

    saved = tmp_path / "saved"
    tokenizer.save_pretrained(saved)
    reloaded = load(str(saved))
    assert reloaded("정부 예산안 발표", "관계자는 설명했다")["input_ids"] == pair["input_ids"]
