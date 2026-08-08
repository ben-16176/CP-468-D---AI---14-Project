"""
Build the side-by-side qualitative comparison table required by the report
(source, reference, LSTM output, LLM output) for >=10 test examples chosen
to show a *range* of behaviors (short/long sentences, rare words), not just
cherry-picked best cases.

Usage:
    python -m src.qualitative_analysis \
        --lstm_tsv outputs/lstm_test_translations.tsv \
        --llm_tsv outputs/llm_test_claude-sonnet-5_A_k0.tsv \
        --n 10 --out outputs/qualitative_examples.md

The error-category column is left BLANK for your team to fill in by hand
after reading each example (under-translation, repetition, hallucination,
OOV failure, fluency vs. adequacy, etc. -- see the assignment's suggested
error taxonomy). Do not auto-fill these; error categorization requires
human judgment.
"""
import argparse
import csv
import random


def load_tsv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
    return rows


def merge_by_source(lstm_rows, llm_rows):
    llm_by_src = {r["source"]: r["llm_hypothesis"] for r in llm_rows}
    merged = []
    for r in lstm_rows:
        src = r["source"]
        if src in llm_by_src:
            merged.append({
                "source": src,
                "reference": r["reference"],
                "lstm_hypothesis": r["lstm_hypothesis"],
                "llm_hypothesis": llm_by_src[src],
            })
    return merged


def select_diverse_examples(rows, n=10, seed=42):
    """Sample across length buckets (short/medium/long) so the examples
    illustrate distinct behaviors rather than only easy short sentences."""
    random.seed(seed)
    buckets = {"short": [], "medium": [], "long": []}
    for r in rows:
        length = len(r["source"].split())
        if length <= 5:
            buckets["short"].append(r)
        elif length <= 12:
            buckets["medium"].append(r)
        else:
            buckets["long"].append(r)

    per_bucket = max(1, n // 3)
    selected = []
    for key in ["short", "medium", "long"]:
        pool = buckets[key]
        random.shuffle(pool)
        selected.extend(pool[:per_bucket])

    # top up if short of n (e.g. a bucket was empty)
    remaining = [r for r in rows if r not in selected]
    random.shuffle(remaining)
    while len(selected) < n and remaining:
        selected.append(remaining.pop())

    return selected[:n]


def to_markdown_table(rows):
    header = (
        "| # | Source (EN) | Reference (ES) | LSTM Output | LLM Output | Error Category (fill in) |\n"
        "|---|---|---|---|---|---|\n"
    )
    lines = [header]
    for i, r in enumerate(rows, 1):
        lines.append(
            f"| {i} | {r['source']} | {r['reference']} | {r['lstm_hypothesis']} | "
            f"{r['llm_hypothesis']} |  |\n"
        )
    return "".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lstm_tsv", type=str, required=True)
    parser.add_argument("--llm_tsv", type=str, required=True)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--out", type=str, default="outputs/qualitative_examples.md")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    lstm_rows = load_tsv(args.lstm_tsv)
    llm_rows = load_tsv(args.llm_tsv)
    merged = merge_by_source(lstm_rows, llm_rows)
    if not merged:
        raise ValueError(
            "No overlapping `source` rows between the LSTM and LLM output files. "
            "Make sure both were run on the same test subset."
        )
    selected = select_diverse_examples(merged, n=args.n, seed=args.seed)
    table = to_markdown_table(selected)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# Qualitative Comparison: LSTM vs. LLM\n\n")
        f.write(
            "Fill in the 'Error Category' column by hand after reading each example. "
            "Suggested categories: under-translation, repetition, hallucination, "
            "out-of-vocabulary failure, fluency error, adequacy error, format violation.\n\n"
        )
        f.write(table)

    print(f"Wrote {len(selected)} examples to {args.out}")


if __name__ == "__main__":
    main()
