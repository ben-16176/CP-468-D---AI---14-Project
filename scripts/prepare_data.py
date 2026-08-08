"""
Cleans the raw English-Spanish sentence pairs (Tatoeba, via the "Anki
sentence pairs" export at manythings.org/anki) and splits into
train/val/test with NO leakage (splits happen before any vocab/model work).

Input format (raw spa.txt from manythings.org/anki):
    English sentence \t Spanish sentence \t attribution info

Usage:
    python scripts/prepare_data.py --raw data/raw/spa.txt --out_dir data \
        --max_len 40 --val_frac 0.1 --test_frac 0.1 --seed 42
"""
import argparse
import csv
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.vocab import tokenize  # noqa: E402


def load_raw(path):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            en, es = parts[0].strip(), parts[1].strip()
            if en and es:
                pairs.append((en, es))
    return pairs


def filter_pairs(pairs, max_len=40, min_len=1):
    filtered = []
    seen = set()
    for en, es in pairs:
        key = (en.lower(), es.lower())
        if key in seen:
            continue  # de-duplicate
        seen.add(key)
        en_len = len(tokenize(en))
        es_len = len(tokenize(es))
        if min_len <= en_len <= max_len and min_len <= es_len <= max_len:
            filtered.append((en, es))
    return filtered


def split_pairs(pairs, val_frac, test_frac, seed):
    rng = random.Random(seed)
    pairs = pairs[:]
    rng.shuffle(pairs)
    n = len(pairs)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)
    val = pairs[:n_val]
    test = pairs[n_val:n_val + n_test]
    train = pairs[n_val + n_test:]
    return train, val, test


def write_tsv(pairs, path):
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        for en, es in pairs:
            writer.writerow([en, es])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=str, required=True, help="path to raw spa.txt")
    parser.add_argument("--out_dir", type=str, default="data")
    parser.add_argument("--max_len", type=int, default=40)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--test_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_pairs = load_raw(args.raw)
    print(f"Loaded {len(raw_pairs)} raw pairs")

    filtered = filter_pairs(raw_pairs, max_len=args.max_len)
    print(f"After filtering/dedup: {len(filtered)} pairs")

    train, val, test = split_pairs(filtered, args.val_frac, args.test_frac, args.seed)
    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    write_tsv(train, os.path.join(args.out_dir, "train.tsv"))
    write_tsv(val, os.path.join(args.out_dir, "val.tsv"))
    write_tsv(test, os.path.join(args.out_dir, "test.tsv"))
    print(f"Wrote train/val/test .tsv files to {args.out_dir}")


if __name__ == "__main__":
    main()
