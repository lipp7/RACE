"""Export true-class private/shared crop weights for CUB mechanism analysis."""
import argparse
import numpy as np
import torch

from .io import project_root, relative_path, read_json, write_json, load_cache, device_from_arg
from .model import build_model


def load_run(folder, device):
    config = read_json(folder / "result.json")["config"]
    features = relative_path(config["features"])
    meta = torch.load(features / "metadata.pt", weights_only=False, map_location="cpu")
    model = build_model(meta, config["parameters"]).to(device)
    model.load_state_dict(torch.load(folder / "best.pt", weights_only=True, map_location="cpu"))
    return model.eval(), features


def crop_rectangles(width, height):
    """Actual footprint after CLIP resize/center crop inside each local view."""
    cw, ch = width // 2, height // 2
    rectangles = []
    for row in range(3):
        top = round(row * (height - ch) / 2)
        for col in range(3):
            left = round(col * (width - cw) / 2)
            rw, rh = (224, int(224 * ch / cw)) if cw <= ch else (int(224 * cw / ch), 224)
            ox, oy = round((rw - 224) / 2), round((rh - 224) / 2)
            rectangles.append((left + ox * cw / rw, top + oy * ch / rh,
                               left + (ox + 224) * cw / rw, top + (oy + 224) * ch / rh))
    return rectangles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    project_root()
    torch.set_num_threads(2)
    model, features = load_run(relative_path(args.run), device_from_arg(args.device))
    data = load_cache(features, "test")
    output = relative_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    arrays = {key: [] for key in ("private", "shared", "attention", "rivals", "predictions")}
    device = next(model.parameters()).device
    with torch.no_grad():
        for start in range(0, len(data["label"]), 64):
            y = data["label"][start:start + 64].to(device)
            out = model(data["views"][start:start + 64].to(device))
            rows = torch.arange(len(y), device=device)
            masked = out["base_logits"].detach().clone()
            masked[rows, y] = -torch.inf
            rivals = masked.argmax(-1)
            own, other = out["gate"][rows, y], out["gate"][rows, rivals]
            values = dict(private=((own - other) / model.evidence_temperature).softmax(-1),
                          shared=(torch.minimum(own, other) / model.evidence_temperature).softmax(-1),
                          attention=out["attention"][rows, y], rivals=rivals,
                          predictions=out["logits"].argmax(-1))
            for key, value in values.items():
                arrays[key].append(value.cpu().numpy())
    np.savez_compressed(output / "weights.npz", labels=data["label"].numpy(),
                        **{key: np.concatenate(value) for key, value in arrays.items()})
    write_json(output / "protocol.json", dict(run=str(relative_path(args.run)),
               features=str(features), labels_used_for="explanation branch selection only",
               view_order="global, then nine half-size crops in row-major order"))


if __name__ == "__main__":
    main()
