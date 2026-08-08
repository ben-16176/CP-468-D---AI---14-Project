# Dataset

**Source:** English-Spanish sentence pairs (`spa-eng.zip`) from the Tatoeba
Project (https://tatoeba.org), redistributed as a ready-to-use file by
manythings.org/anki: http://www.manythings.org/anki/

**License:** Sentences are contributed by the Tatoeba community and released
under **CC-BY 2.0 FR**. See https://tatoeba.org/en/terms_of_use for details.
Cite both Tatoeba.org and the manythings.org/anki export in the report.

**Format:** Each line of the raw file is
`English sentence <TAB> Spanish sentence <TAB> attribution`.

## Reproducing the splits

```bash
bash scripts/download_data.sh
```

This downloads the raw file, then runs `scripts/prepare_data.py`, which:
1. Deduplicates pairs.
2. Filters to sentences with 1-40 tokens (both languages).
3. Shuffles (fixed seed) and splits into `train.tsv` (80%), `val.tsv` (10%),
   `test.tsv` (10%) -- **splitting happens before any vocabulary or model
   work**, so there is no train/val/test leakage.

Resulting files (`train.tsv`, `val.tsv`, `test.tsv`) land directly in this
`data/` directory and are read by `src/train.py`, `src/evaluate.py`, and
`src/llm_baseline.py`.

If you can't reach manythings.org from your environment, download
`spa-eng.zip` manually from the URL above and run:

```bash
python scripts/prepare_data.py --raw path/to/spa.txt --out_dir data
```
