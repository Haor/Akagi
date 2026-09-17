"""Versioned, actor-bound safetensors packaging for the public readout head."""
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile

from .checkpoint import flatten, load_actor_arrays, sha256

FORMAT = "rin.observer.safetensors.v1"
ARCHITECTURE = "rin.observer-readout.v1"
KIND = "actor-token-cross-attention"
FILES = ("observer.safetensors", "observer-config.json", "observer-manifest.json")
PAYMENT_BUCKET_EDGES = (0.5, 1000., 2000., 4000., 8000., 12000., 16000., 24000., 32000., 48000., 96000.)


def validate_config(config, actor_config, actor_manifest):
    if (config.get("schema_version") != "rin.observer-readout-run.v1"
            or config.get("input_boundary") != "actor_representations_stop_gradient"
            or config.get("representation_timing") != "post_ppo_actor_update"):
        raise ValueError("Unsupported observer representation boundary or timing")
    source = config.get("representation_actor_sha256", "")
    if not re.fullmatch(r"[0-9a-f]{64}", source) or source != actor_manifest.get("original_sha256"):
        raise ValueError("Observer is not bound to this source actor")
    c = config["observer"]
    known = {"state_width", "tile_width", "history_width", "global_width", "width", "heads",
             "mlp_width", "tile_count", "opponent_count", "loss_point_scale", "payment_distribution"}
    if set(c) != known:
        raise ValueError("Unsupported observer configuration fields")
    for key in known - {"loss_point_scale", "payment_distribution"}:
        if type(c[key]) is not int or not 1 <= c[key] <= 16384:
            raise ValueError(f"Invalid observer dimension: {key}")
    if c["tile_count"] != 34 or c["opponent_count"] != 3 or c["width"] % c["heads"]:
        raise ValueError("Unsupported observer tile/opponent/attention contract")
    if type(c["payment_distribution"]) is not bool or not math.isfinite(c["loss_point_scale"]) or c["loss_point_scale"] <= 0:
        raise ValueError("Invalid observer output contract")
    for key, actor_key in (("state_width", "fusion_width"), ("tile_width", "tile_width"),
                           ("history_width", "history_width"), ("global_width", "global_width")):
        if c[key] != actor_config["model"][actor_key]:
            raise ValueError(f"Observer representation width differs from actor: {key}")
    expected = {"schema_version": ARCHITECTURE, "kind": KIND, "readout": c}
    if c["payment_distribution"]:
        expected["payment_distribution"] = {
            "edges_points": list(PAYMENT_BUCKET_EDGES),
            "interval": "left_closed_right_open_with_final_open_tail",
            "condition": "one_or_more_eligible_ron_declarations_including_zero_settlement",
        }
    if config.get("readout_contract") != expected:
        raise ValueError("Observer readout schema or payment buckets differ")
    return c


def parameter_shapes(c):
    w, heads = c["width"], c["heads"]
    shapes = {}
    def dense(name, before, after):
        shapes[name + "/kernel"] = (before, after)
        shapes[name + "/bias"] = (after,)
    for name, key in (("state", "state_width"), ("tile", "tile_width"),
                      ("history", "history_width"), ("global", "global_width")):
        dense(name + "_projection", c[key], w)
    shapes.update(memory_type_embedding=(4, w), opponent_query=(3, w), wait_tile_query=(34, w))
    for name, size in (("tile", 38), ("kind", 12), ("from_draw", 3), ("red_mask", 17), ("phase", 5)):
        shapes[f"candidate_{name}/embedding"] = (size, w)
    for name in ("query_norm", "memory_norm", "mlp_norm", "output_norm"):
        shapes[name + "/scale"] = (w,)
    for name in ("query", "key", "value"):
        shapes[f"readout_attention/{name}/kernel"] = (w, heads, w // heads)
        shapes[f"readout_attention/{name}/bias"] = (heads, w // heads)
    shapes["readout_attention/out/kernel"] = (heads, w // heads, w)
    shapes["readout_attention/out/bias"] = (w,)
    dense("mlp_in", w, 2 * c["mlp_width"])
    dense("mlp_out", c["mlp_width"], w)
    for name, size in (("tenpai_head", 1), ("conditional_wait_head", 1), ("ron_joint_head", 8), ("conditional_loss_head", 1)):
        dense(name, w, size)
    if c["payment_distribution"]:
        dense("conditional_payment_distribution_head", w, len(PAYMENT_BUCKET_EDGES) + 1)
    return shapes


def validate_arrays(arrays, config):
    import numpy as np
    shapes = parameter_shapes(config)
    if arrays.keys() != shapes.keys():
        raise ValueError("Observer parameter inventory differs from the versioned architecture")
    for name, value in arrays.items():
        if value.shape != shapes[name] or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f"Invalid observer tensor: {name}")
    return sum(value.size for value in arrays.values())


def export_observer(source, config_path, model_dir, *, expected_sha256=None):
    import msgpack
    import numpy as np
    from safetensors.numpy import save_file
    source, config_path, model_dir = Path(source), Path(config_path), Path(model_dir)
    if any((model_dir / name).exists() for name in FILES):
        raise FileExistsError("Observer files already exist; install a new versioned model bundle")
    _, actor_config, actor_manifest = load_actor_arrays(model_dir)
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    dimensions = validate_config(config, actor_config, actor_manifest)
    source_hash = sha256(source)
    if expected_sha256 and source_hash != expected_sha256.lower():
        raise ValueError("Observer source checkpoint hash mismatch")
    def extension(code, data):
        if code == 1:
            shape, dtype, buffer = msgpack.unpackb(data, raw=False)
            return np.frombuffer(buffer, dtype=np.dtype(dtype)).reshape(shape).copy()
        if code == 3:
            dtype, buffer = msgpack.unpackb(data, raw=False)
            return np.frombuffer(buffer, dtype=np.dtype(dtype))[0]
        if code == 2:
            return complex(*msgpack.unpackb(data))
        raise ValueError(f"Unsupported MessagePack extension: {code}")
    checkpoint = msgpack.unpackb(source.read_bytes(), ext_hook=extension, raw=False, strict_map_key=False)
    arrays = {key: np.ascontiguousarray(value) for key, value in flatten(checkpoint["params"]).items()}
    count = validate_arrays(arrays, dimensions)
    # Keep only the explicit deployment contract, never arbitrary source paths.
    portable = {key: config[key] for key in ("schema_version", "input_boundary", "observer", "readout_contract",
                                            "representation_actor_sha256", "representation_timing")}
    with tempfile.TemporaryDirectory(prefix=".observer-export-", dir=model_dir) as temporary:
        stage = Path(temporary)
        save_file(arrays, stage / FILES[0], metadata={"format": FORMAT, "architecture": ARCHITECTURE, "layout": "flax"})
        (stage / FILES[1]).write_text(json.dumps(portable, indent=2) + "\n", encoding="utf-8")
        manifest = {"format": FORMAT, "architecture": ARCHITECTURE, "kind": KIND,
                    "source_sha256": source_hash, "source_config_sha256": sha256(config_path),
                    "observer_sha256": sha256(stage / FILES[0]), "config_sha256": sha256(stage / FILES[1]),
                    "actor_sha256": actor_manifest["actor_sha256"], "actor_config_sha256": actor_manifest["config_sha256"],
                    "representation_actor_sha256": actor_manifest["original_sha256"],
                    "converter_sha256": sha256(__file__), "optimizer_state_exported": False,
                    "parameter_count": count, "tensors": {key: {"shape": list(value.shape), "dtype": str(value.dtype),
                    "sha256": hashlib.sha256(value.tobytes()).hexdigest()} for key, value in arrays.items()}}
        (stage / FILES[2]).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        loaded, _, _ = load_observer_arrays(stage, actor_config, actor_manifest)
        if any(not np.array_equal(value, loaded[key]) for key, value in arrays.items()):
            raise ValueError("Observer export round trip changed weights")
        # Publish the manifest last; partial installation cannot be treated as ready.
        for name in FILES:
            (stage / name).rename(model_dir / name)
    return manifest


def load_observer_arrays(path, actor_config, actor_manifest):
    from safetensors import safe_open
    from safetensors.numpy import load_file
    path = Path(path)
    manifest = json.loads((path / FILES[2]).read_text(encoding="utf-8"))
    if (manifest.get("format") != FORMAT or manifest.get("architecture") != ARCHITECTURE
            or manifest.get("kind") != KIND):
        raise ValueError("Unsupported observer bundle architecture")
    for observer_key, actor_key in (("actor_sha256", "actor_sha256"), ("actor_config_sha256", "config_sha256"),
                                    ("representation_actor_sha256", "original_sha256")):
        if manifest.get(observer_key) != actor_manifest.get(actor_key):
            raise ValueError(f"Observer/actor identity mismatch: {observer_key}")
    for filename, key in ((FILES[0], "observer_sha256"), (FILES[1], "config_sha256")):
        if sha256(path / filename) != manifest.get(key):
            raise ValueError(f"Observer bundle hash mismatch: {filename}")
    config = json.loads((path / FILES[1]).read_text(encoding="utf-8"))
    dimensions = validate_config(config, actor_config, actor_manifest)
    with safe_open(path / FILES[0], framework="numpy") as stream:
        if stream.metadata() != {"format": FORMAT, "architecture": ARCHITECTURE, "layout": "flax"}:
            raise ValueError("Observer safetensors metadata mismatch")
    arrays = load_file(path / FILES[0])
    if validate_arrays(arrays, dimensions) != manifest.get("parameter_count") or arrays.keys() != manifest["tensors"].keys():
        raise ValueError("Observer manifest inventory mismatch")
    for name, value in arrays.items():
        expected = manifest["tensors"][name]
        if (list(value.shape) != expected["shape"] or str(value.dtype) != expected["dtype"]
                or hashlib.sha256(value.tobytes()).hexdigest() != expected["sha256"]):
            raise ValueError(f"Observer tensor metadata mismatch: {name}")
    return arrays, config, manifest
