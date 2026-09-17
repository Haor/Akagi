"""Compare a local observer bundle with an isolated, supplied JAX reference.

Run this verification tool with the native runtime. JAX stays in the reference
subprocess and is never imported by the deployed bot.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


def flatten(values, prefix=""):
    result = {}
    for name, value in values.items():
        key = prefix + name
        if isinstance(value, dict):
            result.update(flatten(value, key + "/"))
        else:
            result[key] = np.asarray(value)
    return result


def nested(values):
    result = {}
    for name, value in values.items():
        target = result
        parts = name.split("/")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return result


def reference(args):
    sys.path.insert(0, str(Path(args.reference_source).resolve()))
    import jax
    import jax.numpy as jnp
    from rin.model import RINConfig
    from rin.semantic_frontend import SemanticRINActor
    from rin.observer_readout import ActorRepresentationObserver, ObserverReadoutConfig, observer_readout_probabilities
    stage = Path(args.stage)
    config = json.loads((stage / "configuration.json").read_text())
    with np.load(stage / "actor-weights.npz") as arrays:
        actor_params = nested(dict(arrays))
    with np.load(stage / "observer-weights.npz") as arrays:
        observer_params = nested(dict(arrays))
    with np.load(stage / "native.npz") as arrays:
        native = nested(dict(arrays))
    inputs = jax.tree.map(jnp.asarray, native["inputs"])
    actor = SemanticRINActor(config=RINConfig(**config["actor"]["model"]), dtype=jnp.float32)
    observer = ActorRepresentationObserver(ObserverReadoutConfig.from_record(config["observer"]["readout_contract"]), dtype=jnp.float32)
    logits, representations = jax.jit(lambda x: actor.apply({"params": actor_params}, **x, return_observer_inputs=True))(inputs)
    apply = jax.jit(lambda r: observer.apply({"params": observer_params}, r, inputs["candidates"], inputs["candidate_mask"]))
    raw = apply(jax.tree.map(jnp.asarray, native["representations"]))
    complete = apply(representations)
    result = {"logits": logits, "representations": representations, "raw": raw,
              "probabilities": observer_readout_probabilities(raw),
              "end_to_end": observer_readout_probabilities(complete)}
    np.savez_compressed(stage / "reference.npz", **flatten(result))


def main(args):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import torch
    from rin.inference.native.actor_torch import Predictor, tensor_inputs
    from rin.inference.native.checkpoint import load_actor_arrays
    from rin.inference.native.observer_checkpoint import load_observer_arrays
    model = Path(args.model_dir).resolve()
    predictor = Predictor(model, attention="sdpa", precision="fp32", device=args.device)
    if predictor.observer_status != "ready":
        raise RuntimeError("The supplied observer bundle must load successfully")
    with np.load(args.fixtures, allow_pickle=False) as arrays:
        indices = np.linspace(0, len(arrays["candidate_mask"]) - 1, args.limit, dtype=int)
        inputs = nested({name: value[indices] for name, value in arrays.items()})
    tensors = tensor_inputs(inputs, predictor.device)
    with torch.inference_mode():
        baseline = predictor.actor(**tensors)
        logits, representations = predictor.actor(**tensors, return_observer_inputs=True)
        if not torch.equal(baseline, logits):
            raise AssertionError("Exposing representations changed Actor logits")
        raw = predictor.observer(representations, tensors["candidates"], tensors["candidate_mask"])
        probabilities = predictor.observer.probabilities(raw)
    numpy = lambda value: value.detach().cpu().numpy()
    output = {"inputs": inputs, "logits": numpy(logits),
              "representations": {name: numpy(value) for name, value in representations.items()},
              "raw": {name: numpy(value) for name, value in raw.items()},
              "probabilities": {name: numpy(value) for name, value in probabilities.items()}}
    actor_arrays, actor_config, manifest = load_actor_arrays(model)
    observer_arrays, observer_config, observer_manifest = load_observer_arrays(model, actor_config, manifest)
    with tempfile.TemporaryDirectory(prefix=".observer-parity-", dir=model) as temporary:
        stage = Path(temporary)
        (stage / "configuration.json").write_text(json.dumps({"actor": actor_config, "observer": observer_config}))
        np.savez(stage / "actor-weights.npz", **actor_arrays)
        np.savez(stage / "observer-weights.npz", **observer_arrays)
        np.savez_compressed(stage / "native.npz", **flatten(output))
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env["JAX_PLATFORMS"] = "cpu"
        env["OMP_NUM_THREADS"] = "4"
        env["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"
        command = [args.reference_python, str(Path(__file__).resolve()), "--reference", "--stage", str(stage),
                   "--reference-source", str(Path(args.reference_source).resolve())]
        subprocess.run(command, check=True, env=env)
        with np.load(stage / "reference.npz") as arrays:
            expected = nested(dict(arrays))
    legal = inputs["candidate_mask"]
    result = {"states": len(indices), "actor_logits_unchanged": True,
              "actor_legal_max_abs_error": float(np.max(np.abs(output["logits"][legal] - expected["logits"][legal]))),
              "actor_sha256": manifest["actor_sha256"], "observer_sha256": observer_manifest["observer_sha256"],
              "device": str(predictor.device), "precision": "fp32", "attention": "sdpa", "comparisons": {}}
    for group in ("representations", "raw", "probabilities", "end_to_end"):
        actual = output["probabilities"] if group == "end_to_end" else output[group]
        metrics = {}
        for name, values in expected[group].items():
            observed = actual[name]
            if group == "representations" and name == "history_tokens":
                # Padded query rows are ignored by both readouts before projection.
                # All-masked attention rows are not part of the visible contract.
                observed, values = observed[inputs["history_mask"]], values[inputs["history_mask"]]
            error = float(np.max(np.abs(observed.astype(np.float64) - values.astype(np.float64))))
            metrics[name] = error
            tolerance = 0.25 if "points" in name else 5e-5 if group in ("probabilities", "end_to_end") else 3e-4
            if error > tolerance:
                raise AssertionError(f"{group}/{name}: absolute error {error} exceeds {tolerance}")
        result["comparisons"][group] = metrics
    if result["actor_legal_max_abs_error"] > 3e-4:
        raise AssertionError("Actor differs from the reference")
    result["passed"] = True
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir")
    parser.add_argument("--fixtures")
    parser.add_argument("--reference-source", required=True)
    parser.add_argument("--reference-python")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--output")
    parser.add_argument("--reference", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--stage", help=argparse.SUPPRESS)
    args = parser.parse_args()
    reference(args) if args.reference else main(args)
