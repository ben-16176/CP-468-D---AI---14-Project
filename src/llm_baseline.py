"""
LLM baseline for English->Spanish translation, evaluated on the SAME test
set as the LSTM model (see evaluate.py). Implements:
  - zero-shot and few-shot (k=3-5) prompting
  - two distinct prompt variants (so the comparison isn't an artifact of one weak prompt)
  - a rough USD cost estimate based on token usage

Requires an Anthropic API key in the environment variable ANTHROPIC_API_KEY.
(If your team prefers a free local open-weights model instead, swap out the
`call_llm` function for a local HF `transformers` pipeline -- everything
else in this script stays the same.)

Example:
    export ANTHROPIC_API_KEY=sk-...
    python -m src.llm_baseline --config configs/config.yaml --split test \
        --model claude-sonnet-5 --n_examples 200 --k_shot 4
"""
import argparse
import csv
import os
import random
import time

import yaml

from .dataset import read_tsv_pairs

# ---- Pricing table (USD per 1M tokens). These values should be checked
# against Anthropic's latest pricing page before reporting final cost numbers.
PRICING_USD_PER_1M_TOKENS = {
    "claude-sonnet-5": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    "claude-3-5-sonnet-latest": {"input": 3.00, "output": 15.00},
}

PROMPT_VARIANT_A = (
    "Translate the following English sentence into Spanish. "
    "Respond with ONLY the Spanish translation, no explanation, no quotes.\n\n"
    "English: {src}\nSpanish:"
)

PROMPT_VARIANT_B = (
    "You are a professional English-to-Spanish translator. Produce a natural, "
    "fluent, and accurate Spanish translation of the sentence below. "
    "Output must contain the translation ONLY -- no preamble, no notes, no quotation marks.\n\n"
    "Sentence: {src}\nTranslation:"
)

PROMPT_VARIANTS = {"A": PROMPT_VARIANT_A, "B": PROMPT_VARIANT_B}


def rough_token_count(text: str) -> int:
    """Cheap approximation (~4 chars/token for English/Spanish); replace with
    a real tokenizer count if you want a precise cost estimate."""
    return max(1, len(text) // 4)


def build_prompt(variant_template, src_sentence, few_shot_examples=None):
    prompt = ""
    if few_shot_examples:
        for ex_src, ex_tgt in few_shot_examples:
            prompt += variant_template.format(src=ex_src) + f" {ex_tgt}\n\n"
    prompt += variant_template.format(src=src_sentence)
    return prompt


def call_llm(client, model_name, prompt, max_tokens=128):
    """Calls the Anthropic API. Returns (text, input_tokens, output_tokens)."""
    response = client.messages.create(
        model=model_name,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in response.content if block.type == "text").strip()
    # Strip stray quotes some models add
    text = text.strip('"').strip("'")
    in_tok = getattr(response.usage, "input_tokens", rough_token_count(prompt))
    out_tok = getattr(response.usage, "output_tokens", rough_token_count(text))
    return text, in_tok, out_tok


def run_llm_baseline(test_pairs, model_name, prompt_key, k_shot, few_shot_pool, max_tokens, sleep_s=0.0):
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    template = PROMPT_VARIANTS[prompt_key]

    few_shot_examples = None
    if k_shot > 0:
        few_shot_examples = random.sample(few_shot_pool, k=min(k_shot, len(few_shot_pool)))

    results = []
    total_in_tok, total_out_tok = 0, 0
    for src, ref in test_pairs:
        prompt = build_prompt(template, src, few_shot_examples)
        text, in_tok, out_tok = call_llm(client, model_name, prompt, max_tokens=max_tokens)
        total_in_tok += in_tok
        total_out_tok += out_tok
        results.append({
            "source": src,
            "reference": ref,
            "llm_hypothesis": text,
            "prompt": prompt,
        })
        if sleep_s:
            time.sleep(sleep_s)

    pricing = PRICING_USD_PER_1M_TOKENS.get(model_name)
    cost_usd = None
    if pricing:
        cost_usd = (total_in_tok / 1e6) * pricing["input"] + (total_out_tok / 1e6) * pricing["output"]

    stats = {
        "model": model_name,
        "prompt_variant": prompt_key,
        "k_shot": k_shot,
        "n_examples": len(test_pairs),
        "total_input_tokens": total_in_tok,
        "total_output_tokens": total_out_tok,
        "estimated_cost_usd": cost_usd,
    }
    return results, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--model", type=str, default="claude-sonnet-5")
    parser.add_argument("--n_examples", type=int, default=200, help="subsample test set for cost control")
    parser.add_argument("--k_shot", type=int, default=0, help="0 = zero-shot; 3-5 = few-shot")
    parser.add_argument("--prompt_variant", type=str, default="A", choices=["A", "B"])
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    random.seed(args.seed)
    test_pairs = read_tsv_pairs(os.path.join(config["data_dir"], f"{args.split}.tsv"))
    train_pairs = read_tsv_pairs(os.path.join(config["data_dir"], "train.tsv"))

    if args.n_examples and args.n_examples < len(test_pairs):
        test_pairs = random.sample(test_pairs, args.n_examples)

    results, stats = run_llm_baseline(
        test_pairs,
        model_name=args.model,
        prompt_key=args.prompt_variant,
        k_shot=args.k_shot,
        few_shot_pool=train_pairs,
        max_tokens=args.max_tokens,
    )

    out_dir = config.get("output_dir", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{args.model}_{args.prompt_variant}_k{args.k_shot}"
    out_path = os.path.join(out_dir, f"llm_{args.split}_{tag}.tsv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["source", "reference", "llm_hypothesis"])
        for r in results:
            writer.writerow([r["source"], r["reference"], r["llm_hypothesis"]])

    from .utils import save_json
    save_json(stats, os.path.join(out_dir, f"llm_{args.split}_{tag}_stats.json"))
    prompt_out_path = os.path.join(out_dir, f"llm_{args.split}_{tag}_prompts.json")
    save_json(
        {
            "model": args.model,
            "prompt_variant": args.prompt_variant,
            "k_shot": args.k_shot,
            "examples": [
                {
                    "source": r["source"],
                    "reference": r["reference"],
                    "llm_hypothesis": r["llm_hypothesis"],
                    "prompt": r["prompt"],
                }
                for r in results
            ],
        },
        prompt_out_path,
    )
    print(f"Saved {len(results)} LLM translations to {out_path}")
    print(f"Saved exact prompts to {prompt_out_path}")
    print(stats)


if __name__ == "__main__":
    main()
