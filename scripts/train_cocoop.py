"""Shared helper used by the public RACE pipeline and matched baselines."""
import argparse,json,math,random,time
from pathlib import Path
import open_clip
import torch
from torch.nn import functional as F
from maft.cocoop import CoCoOp
from scripts.train import stratified_split
from scripts.train_reference import reference_features
from maft.protocol import fewshot_ids


@torch.no_grad()
def predictions(model,features):
    output=[]
    for batch in features.split(4):
        with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(batch)
        output.append(logits.argmax(-1).cpu())
    return torch.cat(output)
