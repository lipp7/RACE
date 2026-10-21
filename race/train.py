"""Train on cached views; checkpoint selection never reads the test cache."""
import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from maft.semantic_preservation import banks, calibrate, preservation_metrics
from scripts.train import stratified_split
from scripts.train_attack_defense_combination import make_model
from .io import (DATASETS, project_root, relative_path, read_json, write_json,
                 load_cache, predict, device_from_arg, sha256)
from .model import build_model, forward_features, objective


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--features", help="Relative feature directory; defaults to features/spl_fresh32_DATASET")
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/paper.json")
    parser.add_argument("--variant", default="full")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--split-file", help="Optional relative path to explicit train_ids/val_ids")
    parser.add_argument("--rotation-seed", type=int)
    args = parser.parse_args()
    project_root()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = device_from_arg(args.device)
    folder = relative_path(args.output)
    features = relative_path(args.features or f"features/spl_fresh32_{args.dataset}")
    config = read_json(relative_path(args.config))
    variants = read_json("configs/variants.json")
    if args.variant not in variants:
        parser.error("Unknown variant. Choose from: " + ", ".join(variants))
    config.update(variants[args.variant])
    for argument, key in [(args.epochs, "epochs"), (args.batch_size, "batch_size"),
                          (args.lr, "lr"), (args.rotation_seed, "rotation_seed")]:
        if argument is not None:
            config[key] = argument
    if (folder / "result.json").exists():
        raise FileExistsError("This run is complete. Select a new --output directory.")
    folder.mkdir(parents=True, exist_ok=True)
    metadata = torch.load(features / "metadata.pt", weights_only=False, map_location="cpu")
    source = load_cache(features, "train")
    if args.split_file:
        split = torch.load(relative_path(args.split_file), weights_only=True, map_location="cpu")
        train, val = split["train_ids"], split["val_ids"]
    else:
        train, val = stratified_split(source["label"], args.seed)
    train, val = train.long(), val.long()
    if not len(train) or not len(val) or set(train.tolist()) & set(val.tolist()):
        raise ValueError("Training and validation indices must be nonempty and disjoint.")
    if min(int(train.min()), int(val.min())) < 0 or max(int(train.max()), int(val.max())) >= len(source["label"]):
        raise ValueError("Split indices exceed the development cache.")
    torch.save(dict(train_ids=train, val_ids=val), folder / "split.pt")
    # Match the paper's head initialization and minibatch RNG across variants.
    torch.manual_seed(args.seed)
    reference = make_model(metadata, config.get("mode", "smooth"), 0)
    minibatch_rng = torch.get_rng_state()
    reference_head = reference.head.weight.detach().clone()
    del reference
    torch.manual_seed(args.seed)
    model = build_model(metadata, config).to(device)
    torch.set_rng_state(minibatch_rng)
    assert torch.equal(reference_head, model.head.weight.detach().cpu())
    statistics = calibrate(source["views"], train, banks(metadata["concept_text"]), device=device)
    measurement = config.get("preservation", "concept")
    if measurement == "rotated":
        generator = torch.Generator().manual_seed(config.get("rotation_seed", 0))
        dim = metadata["concept_text"].shape[1]
        rotation = torch.linalg.qr(torch.randn(dim, dim, generator=generator)).Q
        original = statistics["concept"]
        statistics["rotated"] = dict(bank=original["bank"] @ rotation,
                                     std=original["std"], mean=original["mean"])
    torch.save(statistics, folder / "statistics.pt")
    statistics = {name: {key: value.to(device) for key, value in values.items()}
                  for name, values in statistics.items()}
    data, labels = source["views"].to(device), source["label"].to(device)
    loader = DataLoader(TensorDataset(train), batch_size=config["batch_size"], shuffle=True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=config["lr"], weight_decay=config["weight_decay"])
    run_config = dict(dataset=args.dataset, seed=args.seed, variant=args.variant,
                      features=str(features), parameters=config, device=str(device),
                      source_sha256={str(p): sha256(p) for directory in ["maft", "race"]
                                     for p in sorted(Path(directory).glob("*.py"))},
                      metadata_sha256=sha256(features / "metadata.pt"),
                      selection="earliest validation-best checkpoint, including epoch 0")
    write_json(folder / "config.json", run_config)
    initial = float((predict(model, source["views"][val]) == source["label"][val]).double().mean())
    best, best_epoch, history, gradient_scale = initial, 0, [], 1.0
    torch.save(model.state_dict(), folder / "best.pt")
    started = time.monotonic()
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        accumulated, batches = {}, 0
        for batch_number, (indices,) in enumerate(loader):
            out, original, adapted = forward_features(model, data[indices])
            loss, parts = objective(out, labels[indices], config)
            metrics = preservation_metrics(original, adapted, statistics)
            keep = metrics[measurement]
            if config.get("gradient_match") and epoch == 1 and batch_number == 1:
                parameters = list(model.visual_adapter.parameters())
                gc = torch.autograd.grad(metrics["concept"], parameters, retain_graph=True)
                gp = torch.autograd.grad(keep, parameters, retain_graph=True)
                nc = sum(g.square().sum() for g in gc).sqrt()
                np = sum(g.square().sum() for g in gp).sqrt()
                gradient_scale = float(nc / np.clamp_min(1e-20))
            total = loss + config.get("preserve_weight", 1.0) * gradient_scale * keep
            if not torch.isfinite(total):
                raise FloatingPointError("Nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            values = dict(total=float(total.detach()), preservation=float(keep.detach()),
                          **{k: float(v) for k, v in parts.items()})
            for key, value in values.items():
                accumulated[key] = accumulated.get(key, 0.0) + value
            batches += 1
        score = float((predict(model, source["views"][val]) == source["label"][val]).double().mean())
        row = dict(epoch=epoch, val_accuracy=score,
                   metrics={k: v / batches for k, v in accumulated.items()})
        history.append(row)
        print(row, flush=True)
        if score > best:
            best, best_epoch = score, epoch
            torch.save(model.state_dict(), folder / "best.pt")
    write_json(folder / "result.json", dict(config=run_config, train_count=len(train),
               val_count=len(val), initial_validation_accuracy=initial, val_accuracy=best,
               best_epoch=best_epoch, history=history, seconds=time.monotonic() - started,
               gradient_matching_scale=gradient_scale,
               trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)))


if __name__ == "__main__":
    main()
