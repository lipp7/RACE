from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


def read_index(path):
    return {int(line.split(maxsplit=1)[0]): line.strip().split(maxsplit=1)[1]
            for line in Path(path).read_text().splitlines()}


class CUBImages(Dataset):
    def __init__(self, root, split, preprocess, view_grid=2):
        self.root = Path(root)
        self.preprocess = preprocess
        self.view_grid = view_grid
        if view_grid not in (2, 3):
            raise ValueError("view_grid must be 2 or 3")
        names = read_index(self.root / "images.txt")
        labels = read_index(self.root / "image_class_labels.txt")
        flags = read_index(self.root / "train_test_split.txt")
        wanted = "1" if split == "train" else "0"
        self.items = [(self.root / "images" / names[i], int(labels[i]) - 1)
                      for i in sorted(names) if flags[i] == wanted]
        if not self.items:
            raise ValueError(f"No CUB images found for {split} in {root}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        import torch
        path, label = self.items[index]
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            # One global view plus an overlapping 2x2 or 3x3 local grid.
            crop_w = (w * 2) // 3 if self.view_grid == 2 else w // 2
            crop_h = (h * 2) // 3 if self.view_grid == 2 else h // 2
            views = [im.copy()]
            for row in range(self.view_grid):
                top = round(row * (h - crop_h) / (self.view_grid - 1))
                for col in range(self.view_grid):
                    left = round(col * (w - crop_w) / (self.view_grid - 1))
                    views.append(im.crop((left, top, left + crop_w, top + crop_h)))
            return torch.stack([self.preprocess(v) for v in views]), label


def cub_vocabulary(root):
    root = Path(root)
    attribute_file = root / "attributes" / "attributes.txt"
    if not attribute_file.exists():
        # The official Caltech archive stores this file beside CUB_200_2011/.
        attribute_file = root.parent / "attributes.txt"
    classes = [x.split(maxsplit=1)[1].replace("_", " ").split(".", 1)[-1]
               for x in (root / "classes.txt").read_text().splitlines()]
    attributes = [x.split(maxsplit=1)[1].replace("::", " ").replace("_", " ")
                  for x in attribute_file.read_text().splitlines()]
    if len(classes) != 200 or len(attributes) != 312:
        raise ValueError("Expected official CUB-200-2011 metadata (200 classes, 312 attributes).")
    return classes, attributes
