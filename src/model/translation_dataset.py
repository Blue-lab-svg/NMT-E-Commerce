"""
Translation Dataset for multilingual NMT training.

"""

import torch
from torch.utils.data import Dataset
from typing import Dict, Any, Optional
from data.corpus_loader import CorpusLoader
from tokenizer.obpe_tokenizer import OBPETokenizer


class TranslationDataset(Dataset[Dict[str, Any]]):
    def __init__(
        self,
        corpus_file: str = "clean_parallel_corpora.csv",
        tokenizer: Optional[OBPETokenizer] = None,
        source_lang: str = "<eng>",
        target_lang: str = "<zu>",
    ):
        """
        Args:
            corpus_file: Path to processed parallel corpus CSV.
            tokenizer: OBPE tokenizer instance.
            source_lang: Language marker for source side (<eng>, <zu>, <s>, <afr>).
            target_lang: Language marker for target side (<eng>, <zu>, <s>, <afr>).
        """
        loader = CorpusLoader()
        df = loader.load_parallel_corpus(corpus_file)
        self.source_texts = df["source_text"].tolist()
        self.target_texts = df["target_text"].tolist()
        self.tokenizer = tokenizer or OBPETokenizer()
        self.source_lang = source_lang
        self.target_lang = target_lang

    def __len__(self) -> int:
        return len(self.source_texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        src = self.source_texts[idx]
        tgt = self.target_texts[idx]

        input_ids = self.tokenizer.encode(src, lang_marker=self.source_lang)
        labels = self.tokenizer.encode(tgt, lang_marker=self.target_lang)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_fn(batch: list[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Pad sequences in a batch to the same length."""
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]

    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=0
    )
    labels_padded = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100  # ignore index for loss
    )

    return {"input_ids": input_ids_padded, "labels": labels_padded}


if __name__ == "__main__":
    tokenizer = OBPETokenizer(min_frequency=2, preload_common=50)
    tokenizer.build_vocab(["clean_parallel_corpora.csv", "clean_term_pairs.csv"])

    dataset = TranslationDataset(
        corpus_file="clean_parallel_corpora.csv",
        tokenizer=tokenizer,
        source_lang="<eng>",
        target_lang="<zu>",
    )

    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

    for batch in dataloader:
        print("Batch input_ids:", batch["input_ids"].shape)
        print("Batch labels:", batch["labels"].shape)
        break
