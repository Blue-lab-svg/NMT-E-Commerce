"""
OBPE Tokenizer for South African e-commerce corpora.

"""

import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, Sequence, Dict, List
import pandas as pd
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DATA_DIR = Path("src/data/processed")
VOCAB_FILE = Path("src/tokenizer/obpe_vocab.json")


class OBPETokenizer:
    def __init__(self, min_frequency: int = 2, preload_common: int = 100) -> None:
        self.vocab: Dict[str, int] = {
            "<pad>": 0,
            "<unk>": 1,
            "<bos>": 2,
            "<eos>": 3,
            "<eng>": 4,
            "<zu>": 5,
            "<s>": 6,
            "<afr>": 7,
        }
        self.inverse_vocab: Dict[int, str] = {v: k for k, v in self.vocab.items()}
        self.min_frequency = min_frequency
        self.preload_common = preload_common

    @staticmethod
    def _normalize_text(text: Any) -> str:
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"\s+", " ", text)
        return text.lower().strip()

    def _morphological_tokenize(self, text: str) -> List[str]:
        tokens: List[str] = []
        for word in text.split():
            if word.endswith("ile") and len(word) > 3:  # Zulu suffix
                tokens.append(word[:-3])
                tokens.append("-ile")
            elif word.endswith("ng") and len(word) > 2:  # Sesotho suffix
                tokens.append(word[:-2])
                tokens.append("-ng")
            elif "-" in word:  # Afrikaans hyphenation
                parts = word.split("-")
                for i, part in enumerate(parts):
                    if part:
                        tokens.append(part)
                    if i < len(parts) - 1:
                        tokens.append("-")
            else:
                tokens.append(word)
        return tokens

    def build_vocab(self, files: Sequence[str]) -> None:
        token_counts: Counter[str] = Counter()

        for fname in files:
            fpath = DATA_DIR / fname
            if not fpath.exists():
                logging.warning("File not found: %s", fpath)
                continue
            df = pd.read_csv(fpath)
            for col in df.columns:
                for text_item in df[col].dropna().tolist():
                    normalized = self._normalize_text(text_item)
                    tokens = self._morphological_tokenize(normalized)
                    token_counts.update(tokens)

       
        common_words = [tok for tok, _ in token_counts.most_common(self.preload_common)]
        token_id = max(self.vocab.values()) + 1
        for tok in common_words:
            if tok not in self.vocab:
                self.vocab[tok] = token_id
                token_id += 1

        for tok, freq in token_counts.items():
            if freq >= self.min_frequency and tok not in self.vocab:
                self.vocab[tok] = token_id
                token_id += 1

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        logging.info("Vocabulary built successfully (%d items).", len(self.vocab))

    def encode(self, text: str, lang_marker: str = "<eng>") -> List[int]:
        normalized = self._normalize_text(text)
        tokens = [lang_marker] + self._morphological_tokenize(normalized)
        unk_id = self.vocab.get("<unk>", 1)
        return [self.vocab.get(tok, unk_id) for tok in tokens]

    def decode(self, token_ids: Sequence[int]) -> str:
        tokens = [self.inverse_vocab.get(tid, "<unk>") for tid in token_ids]
        filtered_tokens = [
            tok for tok in tokens if not (tok.startswith("<") and tok.endswith(">"))
        ]
        reconstructed: List[str] = []
        for tok in filtered_tokens:
            if tok.startswith("-") and reconstructed:
                if tok == "-":
                    reconstructed[-1] = f"{reconstructed[-1]}-"
                else:
                    reconstructed[-1] = f"{reconstructed[-1]}{tok[1:]}"
            elif reconstructed and reconstructed[-1].endswith("-"):
                reconstructed[-1] = f"{reconstructed[-1]}{tok}"
            else:
                reconstructed.append(tok)
        return " ".join(reconstructed)

    def save(self, filepath: Path = VOCAB_FILE) -> None:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        logging.info("Saved vocabulary to %s", filepath)

    def load(self, filepath: Path = VOCAB_FILE) -> None:
        if not filepath.exists():
            logging.error("Vocab file not found: %s", filepath)
            return
        with open(filepath, "r", encoding="utf-8") as f:
            raw_data: Dict[str, Any] = json.load(f)
            loaded_vocab: Dict[str, int] = {}
            for key, value in raw_data.items():
                token = str(key)
                if isinstance(value, bool):
                    token_id = int(value)
                elif isinstance(value, (int, float, str)):
                    token_id = int(value)
                else:
                    logging.warning(
                        "Skipping invalid vocabulary entry for token %s with value %r",
                        token,
                        value,
                    )
                    continue
                loaded_vocab[token] = token_id
            self.vocab = loaded_vocab
            self.inverse_vocab = {v: k for k, v in self.vocab.items()}
            logging.info("Loaded vocabulary from %s", filepath)


def main() -> None:
    tokenizer = OBPETokenizer(min_frequency=2, preload_common=50)
    tokenizer.build_vocab(["clean_parallel_corpora.csv", "clean_term_pairs.csv"])
    tokenizer.save()

    sample_texts = [
        ("Add milk to cart", "<eng>"),
        ("Faka ubisi enqoleni", "<zu>"),
        ("Eketsa lebese ka setokong", "<s>"),
        ("Voeg melk by mandjie", "<afr>"),
    ]

    for text, marker in sample_texts:
        encoded = tokenizer.encode(text, lang_marker=marker)
        decoded = tokenizer.decode(encoded)
        print(f"Text:    {text}")
        print(f"Encoded: {encoded}")
        print(f"Decoded: {decoded}")
        print("-" * 30)


if __name__ == "__main__":
    main()
