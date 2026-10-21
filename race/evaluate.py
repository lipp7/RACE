"""Evaluate the validation-selected checkpoint on the complete test cache."""
import argparse
import torch

from .io import (project_root, relative_path, read_json, write_json,
                 load_cache, predict, device_from_arg, sha256)
from .model import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    project_root()
    torch.set_num_threads(2)
    folder = relative_path(args.run)
    result = read_json(folder / "result.json")
    config = result["config"]
    features = relative_path(config["features"])
    metadata = torch.load(features / "metadata.pt", weights_only=False, map_location="cpu")
    model = build_model(metadata, config["parameters"]).to(device_from_arg(args.device))
    model.load_state_dict(torch.load(folder / "best.pt", weights_only=True, map_location="cpu"))
    split = torch.load(folder / "split.pt", weights_only=True, map_location="cpu")
    source = load_cache(features, "train")
    ids = split["val_ids"]
    validation = float((predict(model, source["views"][ids]) == source["label"][ids]).double().mean())
    if abs(validation - result["val_accuracy"]) > 1e-7:
        raise ValueError("Checkpoint validation accuracy does not match the training record.")
    test = load_cache(features, "test")
    predictions = predict(model, test["views"])
    record = dict(accuracy=float((predictions == test["label"]).double().mean()),
                  n=len(predictions), predictions=predictions.tolist(), labels=test["label"].tolist(),
                  checkpoint_validation_accuracy=validation,
                  checkpoint_sha256=sha256(folder / "best.pt"),
                  result_sha256=sha256(folder / "result.json"))
    write_json(folder / "evaluation_full.json", record)
    print("Test accuracy: {:.4f}% ({:d} images)".format(100 * record["accuracy"], record["n"]))


if __name__ == "__main__":
    main()
