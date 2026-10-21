"""Re-encode all ten views after CUB pixel deletion/retention interventions."""
import argparse
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from PIL import Image

from maft.data import CUBImages, read_index
from . import pixel_masks
from .evidence import load_run, crop_rectangles
from .io import (project_root, relative_path, read_json, write_json,
                 load_cache, device_from_arg)


@torch.no_grad()
def encode(encoder, x, device):
    z = torch.cat([encoder.encode_image(t.to(device)).float()
                   for t in x.flatten(0, -4).split(128)])
    return F.normalize(z, dim=-1).half()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", default="runs/pixel_analysis")
    parser.add_argument("--cub-root", default="data/CUB_200_2011")
    parser.add_argument("--fractions", default="0.1,0.2,0.3")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true", help="Only the first two fixed images")
    args = parser.parse_args()
    project_root()
    import open_clip
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = device_from_arg(args.device)
    pixel_masks.FRACTIONS = tuple(float(x) for x in args.fractions.split(","))
    if not all(0 < x < 1 for x in pixel_masks.FRACTIONS):
        parser.error("fractions must lie strictly between zero and one")
    output, cub = relative_path(args.output), relative_path(args.cub_root)
    output.mkdir(parents=True, exist_ok=True)
    encoder, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32-quickgelu", pretrained="openai", cache_dir="cache/open_clip")
    encoder = encoder.eval().to(device)
    source = CUBImages(cub, "test", preprocess, view_grid=3)
    paths, labels = zip(*source.items)
    labels = np.array(labels)
    image_ids = [i for i, flag in sorted(read_index(cub / "train_test_split.txt").items()) if flag == "0"]
    bbox = {int(v.split()[0]): list(map(float, v.split()[1:]))
            for v in (cub / "bounding_boxes.txt").read_text().splitlines()}
    boxes, rectangles = [bbox[i] for i in image_ids], []
    for path in paths:
        with Image.open(path) as image:
            rectangles.append(crop_rectangles(*image.size))
    rng = np.random.default_rng(20261015)
    ids = [int(rng.choice(np.flatnonzero(labels == c))) for c in np.unique(labels)]
    if args.smoke:
        ids = ids[:2]
    write_json(output / "protocol.json", dict(image_indices=ids, image_ids=[image_ids[i] for i in ids],
               runs=[str(relative_path(run)) for run in args.runs],
               fractions=pixel_masks.FRACTIONS, policies=pixel_masks.POLICIES,
               selection="one fixed-RNG image per class; no prediction-based selection",
               margin="true class versus original-image base-hardest rival; rival fixed for measurement"))
    all_records, all_baselines = [], []
    for name in args.runs:
        folder = relative_path(name)
        model, features = load_run(folder, device)
        cached = load_cache(features, "test")
        with np.load(folder / "evidence/weights.npz") as data:
            assert np.array_equal(data["labels"], labels)
            weights = {k: data[k].copy() for k in ("private", "shared")}
        original_set = pixel_masks.PixelDataset(ids, paths, rectangles, weights, boxes, preprocess, True)
        rivals, margins, baselines, records = {}, {}, [], []
        with torch.no_grad():
            for x, index, _, _ in DataLoader(original_set, batch_size=8, num_workers=args.workers):
                z = encode(encoder, x, device).reshape(len(index), 10, -1)
                old = cached["views"][index].to(device).float()
                if float((z.float() - old).abs().max()) >= .002 or float(F.cosine_similarity(z.float(), old, dim=-1).min()) <= .99999:
                    raise ValueError("Original-image encoding does not match the supplied feature cache.")
                out = model(z)
                y = torch.tensor(labels[index.numpy()], device=device)
                rows = torch.arange(len(y), device=device)
                masked = out["base_logits"].detach().clone()
                masked[rows, y] = -torch.inf
                other = masked.argmax(-1)
                for row, i in enumerate(index.tolist()):
                    rivals[i] = int(other[row])
                    margins[i] = float(out["logits"][row, y[row]] - out["logits"][row, other[row]])
                    baselines.append(dict(run=name, index=i, label=int(y[row]), rival=rivals[i],
                                          margin=margins[i], prediction=int(out["logits"][row].argmax())))
            interventions = pixel_masks.PixelDataset(ids, paths, rectangles, weights, boxes, preprocess)
            for x, index, policy, fraction in DataLoader(interventions, batch_size=8, num_workers=args.workers):
                z = encode(encoder, x, device).reshape(len(index), 2, 10, -1)
                logits = model(z.flatten(0, 1))["logits"].reshape(len(index), 2, -1)
                for row, i in enumerate(index.tolist()):
                    y, j = int(labels[i]), rivals[i]
                    for k, kind in enumerate(("delete", "retain")):
                        values = logits[row, k]
                        margin = float(values[y] - values[j])
                        records.append(dict(run=name, index=i, label=y, rival=j, policy=policy[row],
                            fraction=float(fraction[row]), intervention=kind, prediction=int(values.argmax()),
                            margin=margin, margin_drop=margins[i] - margin))
        write_json(output / ("seed" + str(read_json(folder / "result.json")["config"]["seed"]) + ".json"),
                   dict(records=records, baseline=baselines))
        all_records.extend(records)
        all_baselines.extend(baselines)
    groups = []
    for intervention in ("delete", "retain"):
        for policy in pixel_masks.POLICIES:
            for fraction in pixel_masks.FRACTIONS:
                per_run = []
                for run in args.runs:
                    selected = [r for r in all_records if r["run"] == run and r["intervention"] == intervention
                                and r["policy"] == policy and abs(r["fraction"] - fraction) < 1e-9]
                    per_run.append(dict(run=run, accuracy=100 * np.mean([r["prediction"] == r["label"] for r in selected]),
                                        margin_drop=float(np.mean([r["margin_drop"] for r in selected]))))
                groups.append(dict(intervention=intervention, policy=policy, fraction=fraction, per_run=per_run))
    write_json(output / "summary.json", dict(images=len(ids), groups=groups, baseline=all_baselines))


if __name__ == "__main__":
    main()
