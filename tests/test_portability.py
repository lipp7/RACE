import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import torch


class PortabilityTests(unittest.TestCase):
    def test_train_move_evaluate(self):
        source = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="race-portability-") as temporary:
            root = Path(temporary) / "checkout"
            shutil.copytree(source, root, ignore=shutil.ignore_patterns(
                "__pycache__", "*.pyc", "*.egg-info", "data", "features", "runs", "cache", ".git"))
            feature = root / "features/synthetic"
            feature.mkdir(parents=True)
            torch.manual_seed(5)
            torch.save(dict(class_text=torch.randn(5, 32), concept_text=torch.randn(12, 32)), feature / "metadata.pt")
            torch.save(dict(views=torch.randn(60, 10, 32).half(), label=torch.arange(60) % 5), feature / "train.pt")
            # No test cache exists at training time.
            command = [sys.executable, "-m", "race.train", "--dataset", "cub", "--features", "features/synthetic",
                       "--output", "runs/smoke", "--device", "cpu", "--epochs", "2", "--batch-size", "16"]
            subprocess.run(command, cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            torch.save(dict(views=torch.randn(15, 10, 32).half(), label=torch.arange(15) % 5), feature / "test.pt")
            evaluate = [sys.executable, "-m", "race.evaluate", "--run", "runs/smoke", "--device", "cpu"]
            subprocess.run(evaluate, cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            before = json.loads((root / "runs/smoke/evaluation_full.json").read_text())
            moved = Path(temporary) / "moved_project"
            root.rename(moved)
            subprocess.run(evaluate, cwd=moved, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            after = json.loads((moved / "runs/smoke/evaluation_full.json").read_text())
            self.assertEqual(before, after)
            self.assertEqual(after["n"], 15)
            config = json.loads((moved / "runs/smoke/config.json").read_text())
            self.assertEqual(config["features"], "features/synthetic")
            self.assertNotIn(str(Path(temporary)), json.dumps(config))


if __name__ == "__main__":
    unittest.main()
