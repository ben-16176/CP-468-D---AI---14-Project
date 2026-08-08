# English → Spanish Machine Translation: LSTM Seq2Seq vs. LLM

Course project (CP468): a from-scratch LSTM encoder-decoder with Bahdanau
attention, trained on English-Spanish sentence pairs, compared against an
LLM baseline (zero-shot and few-shot) on the same test set.

## 1. Setup

Use PowerShell from the project root.

```powershell
python -m venv venv
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\venv\bin\Activate.ps1
pip install -r requirements.txt
```

If pip cannot find a PyTorch wheel for your interpreter, install it manually from the official PyTorch index for your Python version and OS, then rerun the command above.


If you'll run the LLM baseline, set your API key:

```powershell
$env:ANTHROPIC_API_KEY = "sk-..."
```

## 2. Get the data

For Windows PowerShell, download and prepare the data directly:

```powershell
New-Item -ItemType Directory -Force -Path data/raw | Out-Null
Invoke-WebRequest -Uri "http://www.manythings.org/anki/spa-eng.zip" -OutFile "data/raw/spa-eng.zip"
Expand-Archive -Path "data/raw/spa-eng.zip" -DestinationPath "data/raw" -Force
Move-Item -Force "data/raw/spa.txt" "data/raw_spa.txt" -ErrorAction SilentlyContinue
python scripts/prepare_data.py --raw data/raw_spa.txt --out_dir data --max_len 40 --val_frac 0.1 --test_frac 0.1 --seed 42
```

See `data/README.md` for dataset details, license, and manual-download
instructions if the script can't reach the source in your environment.
This produces `data/train.tsv`, `data/val.tsv`, and `data/test.tsv` (fixed seed,
no leakage between splits).

## 3. Train the LSTM seq2seq model

```powershell
python -m src.train --config configs/config.yaml
```

- Architecture: `Embedding -> Bidirectional LSTM Encoder -> Bahdanau Attention
  -> LSTM Decoder -> Output projection` (see `src/model.py`). Built only from
  `nn.LSTM` / `nn.Embedding` / `nn.Linear` primitives -- no prebuilt seq2seq
  pipeline.
- All hyperparameters live in `configs/config.yaml`.
- Fixed random seed (`seed: 42` by default) via `src/utils.set_seed`.
- Saves the best checkpoint (by validation loss) to `checkpoints/best_model.pt`,
  plus `checkpoints/training_report.json` containing parameter count,
  per-epoch timing, total training time, and hardware info (for the report's
  "model size / training time / hardware" requirement).

## 4. Evaluate the LSTM model on the test set

```powershell
python -m src.evaluate --config configs/config.yaml --checkpoint checkpoints/best_model.pt --split test
```

Computes corpus BLEU (`sacrebleu`) and ROUGE-1/2/L (`rouge-score`), and
writes:
- `outputs/lstm_test_translations.tsv` (source / reference / hypothesis, for
  qualitative analysis and for merging with the LLM outputs)
- `outputs/lstm_test_metrics.json`

## 5. Run the LLM baseline

```powershell
python -m src.llm_baseline `
    --config configs/config.yaml --split test `
    --model claude-sonnet-5 --n_examples 200 `
    --k_shot 0 --prompt_variant A

python -m src.llm_baseline `
    --config configs/config.yaml --split test `
    --model claude-sonnet-5 --n_examples 200 `
    --k_shot 4 --prompt_variant B
```

- Run at least the two prompt variants (`A`, `B` in `src/llm_baseline.py`)
  in both zero-shot (`--k_shot 0`) and few-shot (`--k_shot 3` to `5`)
  settings, as required.
- `--n_examples` subsamples the test set to control API cost; use the full
  test set if your budget allows.
- Each run writes an `outputs/llm_test_<tag>.tsv` translations file, an
  `outputs/llm_test_<tag>_stats.json` with token counts and an **estimated
  USD cost**, and an `outputs/llm_test_<tag>_prompts.json` sidecar containing
  the exact prompts used for each example (update
  `PRICING_USD_PER_1M_TOKENS` in `src/llm_baseline.py` with current pricing
  before reporting final numbers).
- To use a free local open-weights model instead of an API, swap out the
  `call_llm` function for a local `transformers` pipeline; the rest of the
  script (prompt building, cost/token accounting stub, output format) stays
  the same.

## 6. Build the qualitative comparison table

```powershell
python -m src.qualitative_analysis `
    --lstm_tsv outputs/lstm_test_translations.tsv `
    --llm_tsv outputs/llm_test_claude-sonnet-5_A_k0.tsv `
    --n 10 --out outputs/qualitative_examples.md
```

Samples across short/medium/long sentence buckets so the 10 examples show a
*range* of behavior, not just easy cases. The "Error Category" column is left
blank for manual annotation (under-translation, repetition, hallucination,
OOV failure, fluency vs. adequacy, etc.) -- this requires human judgment and
is intentionally not auto-filled.

## Project structure

```
configs/config.yaml         hyperparameters, paths
data/README.md              dataset source, license, download instructions
scripts/download_data.sh    downloads + prepares train/val/test splits
scripts/prepare_data.py     cleans, dedupes, filters, splits raw pairs
src/vocab.py                tokenization + vocabulary construction
src/dataset.py              PyTorch Dataset + padding/masking collate_fn
src/model.py                Encoder / Attention / Decoder / Seq2Seq (from scratch)
src/train.py                training loop, checkpointing, logging
src/evaluate.py             greedy decoding, BLEU/ROUGE, translation dump
src/llm_baseline.py         LLM zero-/few-shot baseline, cost estimate
src/qualitative_analysis.py side-by-side comparison table builder
src/utils.py                seeding, parameter counting, checkpoint I/O
requirements.txt            pinned dependencies
```

## Notes on fairness / reproducibility

- Vocabulary is built **only** on `train.tsv`, after the train/val/test split,
  to avoid leakage.
- All random seeds (Python `random`, NumPy, PyTorch, CUDA) are fixed via
  `src/utils.set_seed`.
- The LSTM model's parameter count, training time, and hardware are logged
  automatically to `checkpoints/training_report.json` for the report.
- The LLM baseline logs the exact prompts used (`PROMPT_VARIANT_A/B` in
  `src/llm_baseline.py`) and estimated cost, per the assignment requirements.
