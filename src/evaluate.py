"""
Evaluate a trained LSTM seq2seq checkpoint on the test set:
  - Generates translations via greedy decoding
  - Computes corpus BLEU (sacrebleu) and ROUGE-1/2/L (rouge-score)
  - Saves per-example outputs to a TSV for qualitative analysis / LLM comparison

Example:
    python -m src.evaluate --config configs/config.yaml --checkpoint checkpoints/best_model.pt --split test
"""
import argparse
import csv
import os

import torch
import yaml
from torch.utils.data import DataLoader

import sacrebleu
from rouge_score import rouge_scorer

from .dataset import TranslationDataset, collate_fn, read_tsv_pairs
from .model import build_model
from .utils import get_device, save_json
from .vocab import Vocab


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    src_vocab = Vocab.from_dict(ckpt["src_vocab"])
    tgt_vocab = Vocab.from_dict(ckpt["tgt_vocab"])
    config = ckpt["config"]
    model = build_model(len(src_vocab), len(tgt_vocab), config, device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, src_vocab, tgt_vocab, config


def translate_dataset(model, loader, tgt_vocab, device, max_len):
    hypotheses, references, sources = [], [], []
    for batch in loader:
        src = batch["src"].to(device)
        src_lens = batch["src_lens"]
        sequences = model.greedy_decode(src, src_lens, max_len=max_len)
        for i, seq in enumerate(sequences):
            hyp = tgt_vocab.decode(seq, strip_special=True)
            hypotheses.append(hyp)
            references.append(batch["tgt_text"][i])
            sources.append(batch["src_text"][i])
    return sources, references, hypotheses


def compute_metrics(references, hypotheses):
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rl = [], [], []
    for ref, hyp in zip(references, hypotheses):
        scores = scorer.score(ref, hyp)
        r1.append(scores["rouge1"].fmeasure)
        r2.append(scores["rouge2"].fmeasure)
        rl.append(scores["rougeL"].fmeasure)
    n = max(len(r1), 1)
    return {
        "bleu": bleu.score,
        "bleu_signature": str(bleu),
        "rouge1_f": sum(r1) / n,
        "rouge2_f": sum(r2) / n,
        "rougeL_f": sum(rl) / n,
        "n_examples": len(references),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    device = get_device()

    ckpt_path = args.checkpoint or os.path.join(config["checkpoint_dir"], "best_model.pt")
    model, src_vocab, tgt_vocab, model_config = load_checkpoint(ckpt_path, device)
    model.to(device)

    data_path = os.path.join(config["data_dir"], f"{args.split}.tsv")
    pairs = read_tsv_pairs(data_path)
    ds = TranslationDataset(pairs, src_vocab, tgt_vocab, max_len=config["max_len"])
    loader = DataLoader(ds, batch_size=config["eval_batch_size"], shuffle=False, collate_fn=collate_fn)

    sources, references, hypotheses = translate_dataset(model, loader, tgt_vocab, device, config["max_len"])
    metrics = compute_metrics(references, hypotheses)
    print(f"[{args.split}] BLEU={metrics['bleu']:.2f} | ROUGE-1={metrics['rouge1_f']:.3f} "
          f"| ROUGE-2={metrics['rouge2_f']:.3f} | ROUGE-L={metrics['rougeL_f']:.3f} | N={metrics['n_examples']}")

    out_dir = config.get("output_dir", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = args.output or os.path.join(out_dir, f"lstm_{args.split}_translations.tsv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["source", "reference", "lstm_hypothesis"])
        for s, r, h in zip(sources, references, hypotheses):
            writer.writerow([s, r, h])
    print(f"Saved translations to {out_path}")

    save_json(metrics, os.path.join(out_dir, f"lstm_{args.split}_metrics.json"))


if __name__ == "__main__":
    main()
