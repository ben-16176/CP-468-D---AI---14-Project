"""
Train the LSTM seq2seq (with attention) model on the English->Spanish
translation task.

Example:
    python -m src.train --config configs/config.yaml
"""
import argparse
import os
import time

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from .dataset import TranslationDataset, collate_fn, read_tsv_pairs
from .model import build_model
from .utils import set_seed, get_device, count_parameters, hardware_info, save_checkpoint, save_json
from .vocab import Vocab, tokenize, PAD_IDX


def build_vocabs(train_pairs, config):
    src_sentences = [tokenize(s) for s, _ in train_pairs]
    tgt_sentences = [tokenize(t) for _, t in train_pairs]
    src_vocab = Vocab(min_freq=config["min_freq"], max_size=config["max_vocab_size"]).build(src_sentences)
    tgt_vocab = Vocab(min_freq=config["min_freq"], max_size=config["max_vocab_size"]).build(tgt_sentences)
    return src_vocab, tgt_vocab


def run_epoch(model, loader, optimizer, criterion, device, teacher_forcing_ratio, clip, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            src = batch["src"].to(device)
            src_lens = batch["src_lens"]
            tgt = batch["tgt"].to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(src, src_lens, tgt=tgt, teacher_forcing_ratio=teacher_forcing_ratio if train else 0.0)
            # outputs: [batch, tgt_len, vocab] ; skip t=0 (<sos>) for loss
            output_dim = outputs.size(-1)
            outputs_flat = outputs[:, 1:, :].reshape(-1, output_dim)
            tgt_flat = tgt[:, 1:].reshape(-1)

            loss = criterion(outputs_flat, tgt_flat)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()

            total_loss += loss.item()
    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--data_dir", type=str, default=None, help="override config data_dir")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    if args.data_dir:
        config["data_dir"] = args.data_dir

    set_seed(config["seed"])
    device = get_device()
    print(f"Using device: {device}")

    data_dir = config["data_dir"]
    train_pairs = read_tsv_pairs(os.path.join(data_dir, "train.tsv"))
    val_pairs = read_tsv_pairs(os.path.join(data_dir, "val.tsv"))
    print(f"Train pairs: {len(train_pairs)} | Val pairs: {len(val_pairs)}")

    src_vocab, tgt_vocab = build_vocabs(train_pairs, config)
    print(f"Src vocab size: {len(src_vocab)} | Tgt vocab size: {len(tgt_vocab)}")

    train_ds = TranslationDataset(train_pairs, src_vocab, tgt_vocab, max_len=config["max_len"])
    val_ds = TranslationDataset(val_pairs, src_vocab, tgt_vocab, max_len=config["max_len"])

    train_loader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.get("eval_batch_size", config["batch_size"]), shuffle=False, collate_fn=collate_fn
    )

    model = build_model(len(src_vocab), len(tgt_vocab), config, device)
    n_params = count_parameters(model)
    print(f"Trainable parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    best_val_loss = float("inf")
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    best_ckpt_path = os.path.join(config["checkpoint_dir"], "best_model.pt")

    training_log = []
    start_time = time.time()

    for epoch in range(1, config["epochs"] + 1):
        epoch_start = time.time()
        train_loss = run_epoch(
            model, train_loader, optimizer, criterion, device,
            teacher_forcing_ratio=config["teacher_forcing_ratio"], clip=config["clip"], train=True,
        )
        val_loss = run_epoch(
            model, val_loader, optimizer, criterion, device,
            teacher_forcing_ratio=0.0, clip=config["clip"], train=False,
        )
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | {epoch_time:.1f}s")
        training_log.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "seconds": epoch_time})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(best_ckpt_path, model, optimizer, epoch, src_vocab, tgt_vocab, config)
            print(f"  -> saved new best checkpoint (val_loss={val_loss:.4f})")

    total_time = time.time() - start_time
    report = {
        "trainable_parameters": n_params,
        "total_training_seconds": total_time,
        "epochs": config["epochs"],
        "best_val_loss": best_val_loss,
        "hardware": hardware_info(),
        "training_log": training_log,
    }
    save_json(report, os.path.join(config["checkpoint_dir"], "training_report.json"))
    print(f"Done. Total training time: {total_time/60:.1f} min. Report saved to checkpoint_dir.")


if __name__ == "__main__":
    main()
