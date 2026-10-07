import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.ml.train import train_and_save

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="models")
    parser.add_argument("--data", default=None)
    args = parser.parse_args()
    meta = train_and_save(Path(args.output), args.data)
    print(f"Trained {meta['model_name']} v{meta['version']}")
    for k, v in meta["metrics"].items():
        print(f"{k}: {v:.4f}")
