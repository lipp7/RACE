"""Check source syntax, local imports, relative paths, and packaged artifacts."""
import ast
import gzip
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    failures = []
    files = [p for p in ROOT.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    forbidden = re.compile(r"/(?:mnt|home|root|tmp|opt|usr|private|Users|data|workspace|scratch)/|[A-Za-z]:\\\\")
    for path in files:
        rel = path.relative_to(ROOT)
        if any(part in {".git", "runs", "cache", "features", "data"} for part in rel.parts):
            failures.append("Unexpected generated/private directory: " + str(rel))
        if path.suffix == ".gz" and "fewshot" in rel.parts:
            text = gzip.decompress(path.read_bytes()).decode()
            json.loads(text)
        elif path.suffix in {".py", ".json", ".toml", ".md", ".txt", ".yml", ".cff", ".sh"}:
            text = path.read_text()
        else:
            continue
        path_text = text.removeprefix("#!/usr/bin/env bash\n")
        if path.name != "check_release.py" and forbidden.search(path_text):
            failures.append("Absolute filesystem path in " + str(rel))
        if path.suffix != ".py":
            continue
        tree = ast.parse(text, filename=str(rel))
        if "third_party" in rel.parts:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] in {"race", "maft", "scripts"}:
                module = ROOT / node.module.replace(".", "/")
                if not module.with_suffix(".py").exists() and not (module / "__init__.py").exists():
                    failures.append("Missing import in " + str(rel) + ": " + node.module)
    if failures:
        print("\n".join(failures))
        return 1
    print("Release checks passed: source syntax, local imports, paths, and file selection.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
