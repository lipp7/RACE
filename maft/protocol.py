"""Nested per-class few-shot samples from training IDs; validation stays fixed."""
import torch

def fewshot_ids(labels, train_ids, shots, seed):
    if not shots: return train_ids
    generator=torch.Generator().manual_seed(202600+seed)
    selected=[]
    for category in labels.unique(sorted=True):
        ids=train_ids[labels[train_ids]==category]
        if len(ids)<shots:raise ValueError(f'Class {category}: only {len(ids)} training examples for {shots}-shot')
        selected.append(ids[torch.randperm(len(ids),generator=generator)[:shots]])
    return torch.cat(selected)
