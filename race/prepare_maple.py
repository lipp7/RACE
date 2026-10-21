"""Download original OpenAI ViT-B/32 weights into the relative cache directory."""
import argparse
import sys
from pathlib import Path
import torch
from .io import PROJECT_ROOT, project_root, relative_path


def prepare_state(output):
    sys.path.insert(0, str(PROJECT_ROOT / "third_party/PromptSRC"))
    from clip import clip
    checkpoint = clip._download(clip._MODELS["ViT-B/32"], root="cache/clip")
    try:
        state = torch.jit.load(checkpoint, map_location="cpu").state_dict()
    except RuntimeError:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    output = relative_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="cache/openai_b32_state.pt")
    args = parser.parse_args()
    project_root()
    prepare_state(Path(args.output))


if __name__ == "__main__":
    main()
