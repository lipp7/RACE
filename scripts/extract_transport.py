"""Shared helper used by the public RACE pipeline and matched baselines."""
import argparse
import io
import json
from pathlib import Path
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from scripts.extract_cub import text_features


class TransportViews(Dataset):
    def __init__(self, root, dataset, split, preprocess):
        self.kind = dataset
        self.preprocess = preprocess
        if dataset == "aircraft":
            from torchvision.datasets import FGVCAircraft
            source = FGVCAircraft(root, split=split, annotation_level="variant", download=False)
            self.classes = source.classes
            self.items = list(zip(source._image_files, source._labels))
        else:
            import pyarrow.parquet as pq
            files = sorted(Path(root).glob(f"{split}-*.parquet"))
            if not files:
                raise FileNotFoundError(f"No {split} parquet files in {root}")
            self.items = []
            for path in files:
                table = pq.read_table(path)
                info = json.loads(table.schema.metadata[b"huggingface"])
                names = info["info"]["features"]["label"]["names"]
                if hasattr(self, "classes"):
                    assert self.classes == names
                self.classes = names
                self.items.extend((row["image"]["bytes"], row["label"]) for row in table.to_pylist())
            assert len(self.classes) == 196
            assert len(self.items) == (8144 if split == "train" else 8041)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        source, label = self.items[index]
        with Image.open(io.BytesIO(source) if isinstance(source, bytes) else source) as opened:
            image = opened.convert("RGB")
        if self.kind == "aircraft":
            image = image.crop((0, 0, image.width, image.height - 20))
        width, height = image.size
        cw, ch = width // 2, height // 2
        views = [image]
        for row in range(3):
            top = round(row * (height - ch) / 2)
            for col in range(3):
                left = round(col * (width - cw) / 2)
                views.append(image.crop((left, top, left + cw, top + ch)))
        return torch.stack([self.preprocess(view) for view in views]), label
