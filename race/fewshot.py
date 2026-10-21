"""Prepare the paper's released 16-shot/4-validation index selections."""
import argparse
import gzip
import hashlib
import json
import torch
from .io import DATASETS, project_root, relative_path, load_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--features", help="Defaults to features/DATASET_b16_views_grid3")
    parser.add_argument("--output", help="Defaults to features/spl_coop16_DATASET")
    args = parser.parse_args()
    project_root()
    source = relative_path(args.features or f"features/{args.dataset}_b16_views_grid3")
    output = relative_path(args.output or f"features/spl_coop16_{args.dataset}")
    plan = json.loads(gzip.decompress(relative_path(f"configs/fewshot/{args.dataset}.json.gz").read_bytes()))
    train, test = load_cache(source, "train"), load_cache(source, "test")
    labels = torch.cat([train["label"], test["label"]])
    if hashlib.sha256(labels.numpy().tobytes()).hexdigest() != plan["label_sha256"]:
        raise ValueError("Source cache label order differs from the released few-shot pools.")
    output.mkdir(parents=True, exist_ok=True)
    meta = torch.load(source / "metadata.pt", map_location="cpu", weights_only=False)
    torch.save(meta, output / "metadata.pt")
    views = torch.cat([train["views"], test["views"]])
    for split, key in [("train", "development_original_ids"), ("test", "test_original_ids")]:
        ids = torch.tensor(plan[key])
        torch.save(dict(views=views[ids], image=views[ids, 0].float(), label=labels[ids]), output / (split + ".pt"))
    for seed, split in plan["seeds"].items():
        torch.save({k: torch.tensor(v) for k, v in split.items()}, output / ("fewshot_s" + seed + ".pt"))


if __name__ == "__main__":
    main()
