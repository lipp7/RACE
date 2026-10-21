"""Shared seeded development split."""
import torch

def stratified_split(labels, seed, val_fraction=0.1):
    generator = torch.Generator().manual_seed(seed)
    train, val = [], []
    for cls in labels.unique(sorted=True):
        ids = (labels == cls).nonzero(as_tuple=True)[0]
        ids = ids[torch.randperm(len(ids), generator=generator)]
        n_val = max(1, round(len(ids) * val_fraction))
        val.extend(ids[:n_val].tolist())
        train.extend(ids[n_val:].tolist())
    return torch.tensor(train), torch.tensor(val)
