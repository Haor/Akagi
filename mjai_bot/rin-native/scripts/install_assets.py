"""Install user-supplied runtime/model bundles without downloading dependencies."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def copy_new(source, destination, check):
    source, destination = source.resolve(), destination.resolve()
    if not source.is_dir():
        raise ValueError(f"Bundle directory does not exist: {source}")
    if destination.exists():
        raise FileExistsError(f"Destination already exists; choose a new directory: {destination}")
    if destination == source or source in destination.parents:
        raise ValueError("Destination must not be inside the source bundle")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".rin-install-", dir=destination.parent) as temporary:
        stage = Path(temporary) / destination.name
        shutil.copytree(source, stage)
        check(stage)
        stage.rename(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--runtime", type=Path, help="Directory containing a self-contained native Python")
    parser.add_argument("--model-bundle", type=Path, help="Directory containing the converted model bundle")
    args = parser.parse_args()
    root = args.root.resolve()
    if not args.runtime and not args.model_bundle:
        parser.error("Supply --runtime and/or --model-bundle")
    if not (root / "src/rin/inference/native/checkpoint.py").is_file():
        parser.error("--root must be an installed RIN Native plugin directory")
    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = str(root / "src")
    executable = "python.exe" if os.name == "nt" else "bin/python3"
    if args.runtime:
        def check_runtime(stage):
            code = ("import torch,numpy,safetensors,filelock,msgpack,libriichi; "
                    "assert callable(getattr(libriichi.state.PlayerState(0),'public_shanten',None)), "
                    "'Libriichi public_shanten is required'")
            subprocess.run([str(stage / executable), "-c", code], check=True, env=env)
        copy_new(args.runtime, root / "runtime/python", check_runtime)
    if args.model_bundle:
        import json
        import sys
        sys.path.insert(0, str(root / "src"))
        from rin.inference.native.checkpoint import validate_model_id
        manifest = json.loads((args.model_bundle / "conversion-manifest.json").read_text())
        name = validate_model_id(manifest["model"])
        python = Path(os.environ.get("RIN_NATIVE_PYTHON", root / "runtime/python" / executable))
        def check_model(stage):
            code = ("import sys; from pathlib import Path; "
                    "from rin.inference.native.checkpoint import load_actor_arrays; "
                    "from rin.inference.native.observer_checkpoint import FILES,load_observer_arrays; "
                    "p=Path(sys.argv[1]); _,c,m=load_actor_arrays(p); "
                    "load_observer_arrays(p,c,m) if any((p/name).exists() for name in FILES) else None")
            subprocess.run([str(python), "-c", code, str(stage)], check=True, env=env)
        copy_new(args.model_bundle, root / "models" / name, check_model)


if __name__ == "__main__":
    main()
