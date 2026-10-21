"""Shared helper used by the public RACE pipeline and matched baselines."""
import argparse
from pathlib import Path
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import OxfordIIITPet
from scripts.extract_cub import text_features


class PetViews(Dataset):
    def __init__(self, root, split, preprocess):
        dataset = OxfordIIITPet(root, split=split, download=False)
        self.items = list(zip(dataset._images, dataset._labels))
        self.classes = dataset.classes
        self.preprocess = preprocess

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, label = self.items[index]
        with Image.open(path) as source:
            image = source.convert("RGB")
            width, height = image.size
            crop_width, crop_height = width // 2, height // 2
            views = [image.copy()]
            for row in range(3):
                top = round(row * (height - crop_height) / 2)
                for col in range(3):
                    left = round(col * (width - crop_width) / 2)
                    views.append(image.crop((left, top, left + crop_width, top + crop_height)))
            return torch.stack([self.preprocess(view) for view in views]), label
