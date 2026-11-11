"""KoBERT 토크나이저·모델이 이 저장소의 불러오기 경로로 제대로 동작하는지 확인한다(허브에서 내려받는다).

확인하는 것:
1. 한국어 문장이 [UNK] 투성이가 아닌 조각으로 나뉜다.
2. 특수 토큰이 BERT 순서([CLS] A [SEP] B [SEP])로 붙고 token_type_ids 가 0/1 로 나뉜다.
3. 어휘 크기가 모델 임베딩 크기와 같다.
4. save_pretrained 로 저장한 폴더를 같은 불러오기 함수로 다시 읽으면 토큰과 출력이 같다.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from clickbait.config import MODEL_SPECS  # noqa: E402
from clickbait.modeling.models import encode, load_pretrained  # noqa: E402

SENTENCE = "한국어 모델을 공유합니다."
PAIR = ("결국 터졌다…배우의 충격 고백", "소속사는 이날 공식 입장을 내고 사실이 아니라고 밝혔다.")


def check(ok: bool, message: str) -> None:
    print(("통과: " if ok else "실패: ") + message)
    if not ok:
        sys.exit(1)


def main() -> None:
    hub_id = MODEL_SPECS["kobert"].hub_id
    tokenizer, model = load_pretrained("kobert", hub_id)
    model.eval()

    ids = tokenizer(SENTENCE)["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(ids)
    print(f"{SENTENCE!r} -> {tokens} {ids}")
    body = ids[1:-1]
    unk_ratio = sum(i == tokenizer.unk_token_id for i in body) / len(body)
    check(unk_ratio < 0.1, f"[UNK] 비율 {unk_ratio:.2f}")
    check(tokens[0] == "[CLS]" and tokens[-1] == "[SEP]", "한 문장: [CLS] ... [SEP]")

    pair = tokenizer(*PAIR)
    pair_tokens = tokenizer.convert_ids_to_tokens(pair["input_ids"])
    print(f"문장 쌍 -> {pair_tokens}")
    seps = [i for i, t in enumerate(pair_tokens) if t == "[SEP]"]
    check(pair_tokens[0] == "[CLS]" and len(seps) == 2 and seps[-1] == len(pair_tokens) - 1, "문장 쌍: [CLS] A [SEP] B [SEP]")
    types = pair["token_type_ids"]
    check(set(types[: seps[0] + 1]) == {0} and set(types[seps[0] + 1 :]) == {1}, "token_type_ids: A 는 0, B 는 1")
    special = {t: tokenizer.convert_tokens_to_ids(t) for t in ("[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]")}
    print(f"특수 토큰 번호: {special}, pad_token_id={tokenizer.pad_token_id}")
    check(len(tokenizer) == model.config.vocab_size, f"어휘 크기 {len(tokenizer)} = 임베딩 크기 {model.config.vocab_size}")

    with tempfile.TemporaryDirectory() as tmp:
        tokenizer.save_pretrained(tmp)
        model.save_pretrained(tmp)
        print(f"저장한 파일: {sorted(p.name for p in Path(tmp).iterdir())}")
        tokenizer2, model2 = load_pretrained("kobert", tmp)
        model2.eval()
        check(tokenizer2(*PAIR)["input_ids"] == pair["input_ids"], "저장 후 다시 읽은 토크나이저가 같은 번호를 낸다")
        with torch.inference_mode():
            batch = tokenizer.pad(encode(tokenizer, [PAIR[0]], [PAIR[1]], 64), return_tensors="pt")
            same = torch.allclose(model(**batch).logits, model2(**batch).logits)
        check(same, "저장 후 다시 읽은 모델이 같은 출력을 낸다")
    print("KoBERT 확인 끝.")


if __name__ == "__main__":
    main()
