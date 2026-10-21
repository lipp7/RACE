"""Project-relative inputs, portable run records, and shared evaluation."""
import hashlib
import json
import os
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("cub", "pets", "flowers", "cars", "aircraft")


def project_root():
    os.chdir(PROJECT_ROOT)


def relative_path(value):
    path = Path(value)
    if path.is_absolute():
        raise ValueError("Use a path relative to the project root: " + str(value))
    if not path.resolve().is_relative_to(PROJECT_ROOT):
        raise ValueError("Keep inputs and outputs inside the project directory.")
    return path


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def device_from_arg(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def load_cache(folder, split):
    cache = torch.load(Path(folder) / (split + ".pt"), weights_only=True, map_location="cpu")
    if cache["views"].ndim != 3 or cache["views"].shape[1] != 10:
        raise ValueError("Expected [images, 10, embedding_dim] view features.")
    if len(cache["views"]) != len(cache["label"]):
        raise ValueError("Feature and label counts differ.")
    return cache


@torch.no_grad()
def predict(model, views, batch_size=64):
    model.eval()
    device = next(model.parameters()).device
    return torch.cat([model(v.to(device))["logits"].argmax(-1).cpu()
                      for v in views.split(batch_size)])
