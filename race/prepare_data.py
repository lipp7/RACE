"""Download supported torchvision datasets into the project data directory."""
import argparse
from .io import project_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("pets", "flowers", "aircraft"), required=True)
    args = parser.parse_args()
    project_root()
    from torchvision.datasets import OxfordIIITPet, Flowers102, FGVCAircraft
    if args.dataset == "pets":
        OxfordIIITPet("data", split="trainval", download=True)
    elif args.dataset == "flowers":
        Flowers102("data", split="train", download=True)
    else:
        FGVCAircraft("data", split="trainval", annotation_level="variant", download=True)


if __name__ == "__main__":
    main()
