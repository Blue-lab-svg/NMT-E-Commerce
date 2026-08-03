"""Data cleaning pipeline for parallel text corpora and term pairs."""

import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, Mapping
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DATA_DIR: Path = Path("src/data/raw")
OUTPUT_DIR: Path = Path("src/data/processed")

def normalize_text(text: Any) -> str:
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()

def remap_columns(df: pd.DataFrame, mapping: Mapping[str, str]) -> pd.DataFrame:
    df.columns = df.columns.str.strip()  # strip whitespace from headers
    columns_to_rename = {old: new for old, new in mapping.items() if old in df.columns}
    if columns_to_rename:
        df = df.rename(columns=columns_to_rename)
    return df

def clean_csv(input_file: str, col_mapping: Mapping[str, str], output_file: str) -> None:
    file_path: Path = DATA_DIR / input_file
    if not file_path.exists():
        logging.warning("File not found: %s", file_path)
        return

    df: pd.DataFrame = pd.read_csv(file_path, encoding="latin1")
    df.columns = df.columns.str.strip()

    df = remap_columns(df, col_mapping)

    expected_cols: set[str] = set(col_mapping.values())
    present_cols: list[str] = [col for col in expected_cols if col in df.columns]

    if not present_cols:
        logging.warning("Missing expected columns in %s", input_file)
        logging.info("Available columns: %s", list(df.columns))
        return

    df = df.dropna(subset=present_cols)

    for col in present_cols:
        df[col] = df[col].apply(normalize_text)

    df = df.drop_duplicates()

    for col in present_cols:
        df = df[df[col].astype(str).str.len() > 0]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path: Path = OUTPUT_DIR / output_file
    df.to_csv(output_path, index=False)
    logging.info("Cleaned file saved: %s", output_path)

def main() -> None:
    clean_csv(
        input_file="checkers_parallel_corpora.csv",
        col_mapping={"source_text": "source_text", "target_text": "target_text"},
        output_file="clean_parallel_corpora.csv",
    )

    clean_csv(
        input_file="checkers_term_pairs.csv",
        col_mapping={
            "English": "english",
            "English ": "english",
            "Zulu": "zulu",
            "Zulu ": "zulu",
            "Afrikaans": "afrikaans",
            "Afrikaans ": "afrikaans",
            "Sesotho": "sesotho",
            "Sotho": "sesotho",
        },
        output_file="clean_term_pairs.csv",
    )

if __name__ == "__main__":
    main()
