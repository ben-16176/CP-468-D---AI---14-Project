"""
Shared utilities: reproducibility, timing, checkpointing, parameter counting.
"""
import os
import json
import random
import time
import platform
from contextlib import contextmanager

import numpy as np
import torch


def set_seed(seed: int = 42):
    """Fix all relevant random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make cuDNN deterministic (slower, but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@contextmanager
def timer():
    """Usage: with timer() as t: ... ; print(t['seconds'])"""
    result = {}
    start = time.time()
    yield result
    result["seconds"] = time.time() - start


def hardware_info():
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
    return info


def save_checkpoint(path, model, optimizer, epoch, src_vocab, tgt_vocab, config, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "src_vocab": src_vocab.to_dict(),
        "tgt_vocab": tgt_vocab.to_dict(),
        "config": config,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
