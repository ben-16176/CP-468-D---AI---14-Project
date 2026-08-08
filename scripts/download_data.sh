#!/usr/bin/env bash
# Downloads the English-Spanish sentence pairs (Tatoeba, via manythings.org/anki)
# and prepares train/val/test splits.
#
# Dataset: "spa-eng.zip" -- English/Spanish sentence pairs derived from the
# Tatoeba Project (https://tatoeba.org), redistributed by manythings.org/anki.
# License: sentences are released under CC-BY 2.0 FR by the Tatoeba community.
# Cite: Tatoeba.org, and http://www.manythings.org/anki/ for this specific export.
#
# Usage: bash scripts/download_data.sh

set -e

mkdir -p data/raw
cd data/raw

echo "Downloading spa-eng.zip from manythings.org/anki ..."
curl -L -o spa-eng.zip http://www.manythings.org/anki/spa-eng.zip
unzip -o spa-eng.zip
mv -f spa.txt ../raw_spa.txt || true

cd ../..
echo "Preparing train/val/test splits ..."
python scripts/prepare_data.py --raw data/raw_spa.txt --out_dir data --max_len 40 --val_frac 0.1 --test_frac 0.1 --seed 42

echo "Done. See data/train.tsv, data/val.tsv, data/test.tsv"
