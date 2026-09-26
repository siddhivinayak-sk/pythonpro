"""Word level tokenizer with the four special tokens a seq2seq model needs."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional

PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN = "<pad>", "<sos>", "<eos>", "<unk>"
PAD_ID, SOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3
SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[?!.,;:]")
_RAW_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?|[?!.,;:]")
_NO_SPACE_BEFORE = {"?", "!", ".", ",", ";", ":", "'s", "'t", "'re", "'m", "'ll", "'ve"}
_SENTENCE_END = {".", "!", "?"}


def tokenize(text: str) -> List[str]:
    return _TOKEN_PATTERN.findall(text.lower())


def detokenize(tokens: Iterable[str]) -> str:
    """Join tokens back into a sentence, restoring spacing and sentence capitalisation."""
    out = ""
    capitalize_next = True
    for token in tokens:
        if capitalize_next and token[:1].isalpha():
            token = token[:1].upper() + token[1:]
            capitalize_next = False
        elif token in _SENTENCE_END:
            capitalize_next = True
        if out and token not in _NO_SPACE_BEFORE:
            out += " "
        out += token
    return out


class WordTokenizer:
    """Builds a vocabulary from the corpus and converts text <-> id sequences."""

    def __init__(self, min_frequency: int = 1, max_vocab_size: Optional[int] = None) -> None:
        self.min_frequency = min_frequency
        self.max_vocab_size = max_vocab_size
        self.token_to_id: Dict[str, int] = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
        self.id_to_token: List[str] = list(SPECIAL_TOKENS)
        self.display_form: Dict[str, str] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def fit(self, texts: Iterable[str], casing_texts: Optional[Iterable[str]] = None) -> "WordTokenizer":
        """Build the vocabulary from `texts`; learn casing from `casing_texts` (the replies)."""
        texts = list(texts)
        counter: Counter = Counter()
        for text in texts:
            counter.update(tokenize(text))

        casings: Dict[str, Counter] = {}
        for text in list(casing_texts) if casing_texts is not None else texts:
            for raw in _RAW_TOKEN_PATTERN.findall(text):
                casings.setdefault(raw.lower(), Counter())[raw] += 1

        self.token_to_id = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
        self.id_to_token = list(SPECIAL_TOKENS)
        # Remember how each word is normally written so generated replies keep their casing.
        # Only words that never appear in lowercase are treated as proper nouns.
        self.display_form = {"i": "I"}
        for token, variants in casings.items():
            if token not in variants:
                self.display_form[token] = variants.most_common(1)[0][0]

        budget = None if self.max_vocab_size is None else self.max_vocab_size - len(SPECIAL_TOKENS)
        for token, count in counter.most_common(budget):
            if count < self.min_frequency:
                break
            self.token_to_id[token] = len(self.id_to_token)
            self.id_to_token.append(token)
        return self

    def encode(
        self,
        text: str,
        add_sos: bool = False,
        add_eos: bool = False,
        max_length: Optional[int] = None,
    ) -> List[int]:
        ids = [self.token_to_id.get(token, UNK_ID) for token in tokenize(text)]
        if max_length is not None:
            reserved = int(add_sos) + int(add_eos)
            ids = ids[: max(0, max_length - reserved)]
        if add_sos:
            ids = [SOS_ID] + ids
        if add_eos:
            ids = ids + [EOS_ID]
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        tokens = []
        for token_id in ids:
            token_id = int(token_id)
            if skip_special:
                if token_id in (PAD_ID, SOS_ID):
                    continue
                if token_id == EOS_ID:
                    break
            tokens.append(self.id_to_token[token_id] if token_id < self.vocab_size else UNK_TOKEN)
        return detokenize(self.display_form.get(token, token) for token in tokens)

    # ------------------------------------------------------------- persistence
    def save(self, path: str | Path) -> Path:
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {
                    "min_frequency": self.min_frequency,
                    "max_vocab_size": self.max_vocab_size,
                    "vocabulary": self.id_to_token,
                    "display_form": self.display_form,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> "WordTokenizer":
        payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        tokenizer = cls(payload.get("min_frequency", 1), payload.get("max_vocab_size"))
        tokenizer.id_to_token = list(payload["vocabulary"])
        tokenizer.token_to_id = {token: index for index, token in enumerate(tokenizer.id_to_token)}
        tokenizer.display_form = dict(payload.get("display_form", {}))
        return tokenizer
