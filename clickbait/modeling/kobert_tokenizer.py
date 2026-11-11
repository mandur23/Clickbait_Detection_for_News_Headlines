"""KoBERT(skt/kobert-base-v1) 토크나이저.

허브의 tokenizer_config 는 XLNetTokenizer 를 가리킨다. AutoTokenizer 로 그대로 읽으면(transformers 5.22 확인)
한글이 자모로 쪼개져 대부분 [UNK] 가 되고 특수 토큰도 XLNet 순서(A [SEP] [CLS])로 붙는다. SKT 의 kobert_tokenizer 패키지(git 의존성)가 하던 일을 여기서 직접 한다. SentencePiece
조각 번호를 그대로 어휘 번호로 쓰고 BERT 순서([CLS] A [SEP] B [SEP])로 감싼다.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import sentencepiece as spm
from transformers import PreTrainedTokenizer

# 허브 설정에 남아 있는 XLNet 전용 인자. 이 토크나이저에는 해당하지 않는다.
_XLNET_ONLY = ("bos_token", "eos_token", "do_lower_case", "remove_space", "keep_accents", "sp_model_kwargs")


class KoBERTTokenizer(PreTrainedTokenizer):
    vocab_files_names = {"vocab_file": "spiece.model"}
    model_input_names = ["input_ids", "token_type_ids", "attention_mask"]
    padding_side = "right"

    def __init__(
        self,
        vocab_file: str,
        unk_token: str = "[UNK]",
        sep_token: str = "[SEP]",
        pad_token: str = "[PAD]",
        cls_token: str = "[CLS]",
        mask_token: str = "[MASK]",
        **kwargs: object,
    ) -> None:
        for key in _XLNET_ONLY:
            kwargs.pop(key, None)
        self.vocab_file = vocab_file
        self.sp_model = spm.SentencePieceProcessor(model_file=vocab_file)
        super().__init__(
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return self.sp_model.get_piece_size()

    def get_vocab(self) -> dict[str, int]:
        vocab = {self.sp_model.id_to_piece(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text: str) -> list[str]:
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token: str) -> int:
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index: int) -> str:
        return self.sp_model.id_to_piece(index)

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return "".join(tokens).replace("▁", " ").strip()

    def build_inputs_with_special_tokens(self, token_ids_0: list[int], token_ids_1: list[int] | None = None) -> list[int]:
        cls, sep = [self.cls_token_id], [self.sep_token_id]
        if token_ids_1 is None:
            return cls + token_ids_0 + sep
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = None, already_has_special_tokens: bool = False
    ) -> list[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(token_ids_0, token_ids_1, already_has_special_tokens=True)
        if token_ids_1 is None:
            return [1] + [0] * len(token_ids_0) + [1]
        return [1] + [0] * len(token_ids_0) + [1] + [0] * len(token_ids_1) + [1]

    def create_token_type_ids_from_sequences(self, token_ids_0: list[int], token_ids_1: list[int] | None = None) -> list[int]:
        first = [0] * (len(token_ids_0) + 2)
        if token_ids_1 is None:
            return first
        return first + [1] * (len(token_ids_1) + 1)

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None) -> tuple[str]:
        target = Path(save_directory) / ((filename_prefix + "-" if filename_prefix else "") + "spiece.model")
        if Path(self.vocab_file).resolve() != target.resolve():
            shutil.copyfile(self.vocab_file, target)
        return (str(target),)
