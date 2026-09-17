"""Convert a compatible actor-bound observer checkpoint without JAX imports."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rin.inference.native.observer_checkpoint import export_observer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    manifest = export_observer(args.source, args.config, args.model_dir, expected_sha256=args.expected_sha256)
    print(manifest["observer_sha256"], manifest["parameter_count"])


if __name__ == "__main__":
    main()
