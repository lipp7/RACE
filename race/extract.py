"""Encode deterministic ten-view inputs and optional matched baseline caches."""
import argparse
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .io import DATASETS, project_root, relative_path, read_json, write_json


def make_dataset(name, split, transform, root=None):
    from maft.data import CUBImages
    from scripts.extract_pets import PetViews
    from scripts.extract_flowers import FlowerViews
    from scripts.extract_transport import TransportViews
    if name == "cub":
        return CUBImages(root or "data/CUB_200_2011", split, transform, view_grid=3)
    if name == "pets":
        return PetViews(root or "data", "trainval" if split == "train" else "test", transform)
    if name == "flowers":
        return FlowerViews(root or "data", ("train", "val") if split == "train" else ("test",), transform)
    return TransportViews(root or ("data/stanford-cars-parquet" if name == "cars" else "data"),
                          name, "trainval" if name == "aircraft" and split == "train" else split,
                          transform)


def prompts(name, vocabulary):
    noun = dict(cub="bird", pets="", flowers="flower", cars="car", aircraft="aircraft")[name]
    classes = [f"a photo of a {c}" + (" " + noun if noun else "") for c in vocabulary["classes"]]
    concepts = [f"a {'pet' if name == 'pets' else noun} with {c}" for c in vocabulary["concepts"]]
    return classes, concepts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--root", help="Relative dataset directory; standard layout is documented in README")
    parser.add_argument("--output", help="Relative cache directory")
    parser.add_argument("--model", default="ViT-B-32-quickgelu",
                        choices=("ViT-B-32-quickgelu", "ViT-B-16-quickgelu"))
    parser.add_argument("--profile", choices=("fresh", "ablation"), default="fresh")
    parser.add_argument("--baseline-caches", action="store_true",
                        help="Also save real preprojection means (ProLIP) and raw means (LaBo)")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--encode-batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    project_root()
    from .io import device_from_arg
    from scripts.extract_cub import text_features
    import open_clip
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = device_from_arg(args.device)
    if args.precision == "bf16" and device.type != "cuda":
        parser.error("BF16 extraction requires CUDA; use --precision fp32 for CPU.")
    if args.baseline_caches and (args.model != "ViT-B-32-quickgelu" or args.profile != "fresh" or args.precision != "fp32"):
        parser.error("Matched baseline caches require fresh ViT-B/32 FP32 extraction.")
    root = str(relative_path(args.root)) if args.root else None
    default_output = (f"features/spl_fresh32_{args.dataset}" if args.profile == "fresh"
                      else f"features/{args.dataset}_views_grid3")
    if args.model == "ViT-B-16-quickgelu":
        default_output = f"features/{args.dataset}_b16_views_grid3"
    output = relative_path(args.output or default_output)
    extra = [Path(f"features/spl_preprojection10_{args.dataset}"),
             Path(f"features/spl_labo_fresh10_{args.dataset}")] if args.baseline_caches else []
    for directory in [output] + extra:
        if not args.overwrite and any((directory / (s + ".pt")).exists() for s in ("train", "test")):
            raise FileExistsError("Cache already exists: " + str(directory))
        directory.mkdir(parents=True, exist_ok=True)
    vocabulary = read_json(f"configs/vocabularies/{args.dataset}.json")
    model, _, transform = open_clip.create_model_and_transforms(
        args.model, pretrained="openai", cache_dir="cache/open_clip")
    model = model.eval().to(device)
    tokenizer = open_clip.get_tokenizer(args.model)
    class_prompts, concept_prompts = prompts(args.dataset, vocabulary)
    metadata = dict(vocabulary, class_text=text_features(model, tokenizer, class_prompts, device),
                    concept_text=text_features(model, tokenizer, concept_prompts, device),
                    model=args.model, pretrained="openai", view_grid=3,
                    image_precision=args.precision)
    torch.save(metadata, output / "metadata.pt")
    projection = None
    if args.profile == "fresh":
        projection = model.visual.proj.detach().clone()
        model.visual.proj = None
    if args.baseline_caches:
        torch.save(dict(projection=projection.cpu(), class_text=metadata["class_text"],
                        classes=vocabulary["classes"], model=args.model, pretrained="openai"),
                   extra[0] / "metadata.pt")
        torch.save(metadata, extra[1] / "metadata.pt")
    for split in ("train", "test"):
        dataset = make_dataset(args.dataset, split, transform, root)
        if hasattr(dataset, "classes") and dataset.classes != vocabulary["classes"]:
            raise ValueError("Dataset class order differs from the released vocabulary.")
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=device.type == "cuda")
        values, labels, pre_means, raw_means = [], [], [], []
        with torch.no_grad():
            for views, y in loader:
                images = views.flatten(0, 1)
                context = (torch.autocast("cuda", dtype=torch.bfloat16)
                           if args.precision == "bf16" else nullcontext())
                with context:
                    encoded = torch.cat([model.encode_image(x.to(device)).float()
                                         for x in images.split(args.encode_batch_size)])
                if projection is not None:
                    before = encoded.reshape(len(y), 10, -1)
                    projected = before @ projection
                else:
                    projected = encoded.reshape(len(y), 10, -1)
                normalized = F.normalize(projected, dim=-1)
                values.append(normalized.cpu().half())
                labels.append(y)
                if args.baseline_caches:
                    pre_means.append((before / projected.norm(dim=-1, keepdim=True).clamp_min(1e-12)).mean(1).cpu())
                    raw_means.append(projected.mean(1).cpu())
                print(f"{split}: {sum(len(t) for t in labels)}/{len(dataset)}", flush=True)
        features, targets = torch.cat(values), torch.cat(labels)
        torch.save(dict(views=features, image=features[:, 0].float(), label=targets), output / (split + ".pt"))
        if args.baseline_caches:
            torch.save(dict(preprojection=torch.cat(pre_means), label=targets), extra[0] / (split + ".pt"))
            torch.save(dict(image=torch.cat(raw_means), label=targets), extra[1] / (split + ".pt"))
    write_json(output / "extraction.json", dict(dataset=args.dataset, root=root or "standard relative layout",
               model=args.model, precision=args.precision, profile=args.profile,
               view_count=10, class_prompts=class_prompts, concept_prompts=concept_prompts,
               baseline_caches=args.baseline_caches))


if __name__ == "__main__":
    main()
