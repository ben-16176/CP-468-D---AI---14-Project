"""
PyTorch Dataset + collate function for the English->Spanish translation task.
Handles padding/masking. No leakage: vocab must be built on train split only
and passed in when constructing val/test datasets.
"""
import csv
import torch
from torch.utils.data import Dataset

from .vocab import tokenize, PAD_IDX


def read_tsv_pairs(path):
    """Reads a tab-separated file with columns: src \\t tgt"""
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            src, tgt = row[0].strip(), row[1].strip()
            if src and tgt:
                pairs.append((src, tgt))
    return pairs


class TranslationDataset(Dataset):
    """
    src_lang: source language sentences (English)
    tgt_lang: target language sentences (Spanish)
    """

    def __init__(self, pairs, src_vocab, tgt_vocab, max_len=50):
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_text, tgt_text = self.pairs[idx]
        src_tokens = tokenize(src_text)[: self.max_len - 2]
        tgt_tokens = tokenize(tgt_text)[: self.max_len - 2]
        src_ids = self.src_vocab.encode(src_tokens, add_sos_eos=True)
        tgt_ids = self.tgt_vocab.encode(tgt_tokens, add_sos_eos=True)
        return {
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "tgt_ids": torch.tensor(tgt_ids, dtype=torch.long),
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


def collate_fn(batch):
    """Pad src/tgt sequences to the max length within the batch."""
    batch = sorted(batch, key=lambda x: len(x["src_ids"]), reverse=True)

    src_seqs = [item["src_ids"] for item in batch]
    tgt_seqs = [item["tgt_ids"] for item in batch]
    src_lens = torch.tensor([len(s) for s in src_seqs], dtype=torch.long)
    tgt_lens = torch.tensor([len(s) for s in tgt_seqs], dtype=torch.long)

    max_src = int(src_lens.max())
    max_tgt = int(tgt_lens.max())

    src_padded = torch.full((len(batch), max_src), PAD_IDX, dtype=torch.long)
    tgt_padded = torch.full((len(batch), max_tgt), PAD_IDX, dtype=torch.long)
    for i, (s, t) in enumerate(zip(src_seqs, tgt_seqs)):
        src_padded[i, : len(s)] = s
        tgt_padded[i, : len(t)] = t

    return {
        "src": src_padded,          # [batch, max_src]
        "src_lens": src_lens,       # [batch]
        "tgt": tgt_padded,          # [batch, max_tgt]
        "tgt_lens": tgt_lens,       # [batch]
        "src_text": [b["src_text"] for b in batch],
        "tgt_text": [b["tgt_text"] for b in batch],
    }
