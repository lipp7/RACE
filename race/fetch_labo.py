"""Fetch the pinned official LaBo concept bank when reproducing this baseline."""
import argparse
import subprocess
from pathlib import Path
from .io import project_root, read_json


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    project_root()
    reference = read_json("third_party/versions.json")["LaBo"]
    target = Path("third_party/LaBo")
    if target.exists():
        raise FileExistsError("third_party/LaBo already exists; its contents were left unchanged.")
    subprocess.run(["git", "clone", reference["url"], str(target)], check=True)
    subprocess.run(["git", "-C", str(target), "checkout", reference["commit"]], check=True)


if __name__ == "__main__":
    main()
