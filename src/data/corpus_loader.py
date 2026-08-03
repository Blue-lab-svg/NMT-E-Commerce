"""
Corpus Loader for South African multilingual corpora.

Acts as the bridge between preprocessing and tokenizer/model stages
"""

import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DATA_DIR: Path = Path("src/data/processed")


class CorpusLoader:

    def __init__(self, data_dir: Path = DATA_DIR) -> None:
        self.data_dir: Path = data_dir

    @staticmethod
    def _normalize_text(text: Any) -> str:
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s\-']", "", text)  # keep words, hyphens, apostrophes
        return text.lower().strip()

    def load_parallel_corpus(self, filename: str = "clean_parallel_corpora.csv") -> pd.DataFrame:
        file_path: Path = self.data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Parallel corpus file not found: {file_path}")

        df = pd.read_csv(file_path)
        logging.info("Loaded parallel corpus with columns: %s", list(df.columns))

        required_cols = {"source_text", "target_text"}
        if not required_cols.issubset(set(df.columns)):
            raise ValueError("Parallel corpus missing expected columns")

        df = df.dropna(subset=["source_text", "target_text"]).copy()
        df["source_text"] = df["source_text"].astype(str).apply(self._normalize_text)
        df["target_text"] = df["target_text"].astype(str).apply(self._normalize_text)

        return df[["source_text", "target_text"]]

    def load_term_pairs(self, filename: str = "clean_term_pairs.csv") -> pd.DataFrame:
        """Load integrated term pairs for domain-specific vocabulary."""
        file_path: Path = self.data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Term pairs file not found: {file_path}")

        df = pd.read_csv(file_path)
        logging.info("Loaded term pairs with columns: %s", list(df.columns))

        # Normalize all language columns
        for col in df.columns:
            df[col] = df[col].astype(str).apply(self._normalize_text)

        return df

    def load_all(self) -> Dict[str, pd.DataFrame]:
        """Convenience method to load both parallel corpora and term pairs."""
        parallel = self.load_parallel_corpus()
        terms = self.load_term_pairs()
        return {"parallel": parallel, "terms": terms}


if __name__ == "__main__":
    loader = CorpusLoader()
    corpora = loader.load_all()

    print("Parallel corpus sample:")
    print(corpora["parallel"].head())

    print("\nTerm pairs sample:")
    print(corpora["terms"].head())
