"""Shared helper used by the public RACE pipeline and matched baselines."""
import argparse
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from maft.data import CUBImages, cub_vocabulary


def text_features(model, tokenizer, prompts, device, batch_size=128):
    chunks = []
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            tokens = tokenizer(prompts[start:start + batch_size]).to(device)
            chunks.append(F.normalize(model.encode_text(tokens).float(), dim=-1).cpu())
    return torch.cat(chunks)
