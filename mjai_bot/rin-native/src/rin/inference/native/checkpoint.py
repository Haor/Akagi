"""Non-pickle actor export and strict, shape-checked loading."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import tempfile

FORMAT = "rin.actor.safetensors.v1"
ARCHITECTURE = "rin.semantic-dynamic.v1"
SOURCE_SCHEMA = "rin.semantic-actor-inheritance.v1"

def validate_model_id(name):
    if not isinstance(name,str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,79}",name):
        raise ValueError("Model ID must use 1-80 lowercase letters, digits, hyphens or underscores")
    if name in {"con","prn","aux","nul",*(f"com{i}" for i in range(10)),*(f"lpt{i}" for i in range(10))}:
        raise ValueError("Reserved Windows model ID")
    return name

def validate_config(config, architecture=ARCHITECTURE):
    if architecture != ARCHITECTURE:
        raise ValueError(f"Unsupported architecture: {architecture}; add a versioned adapter")
    if config.get("schema_version") != SOURCE_SCHEMA:
        raise ValueError("Unsupported source schema; an explicit architecture adapter is required")
    if config.get("action_head_variant") != "dynamic":
        raise ValueError("Only the visible dynamic action contract is supported")
    contract=config.get("resume_contract",{})
    if any(contract.get(k) is not False for k in ("same_shanten_required_tiles","best_current_required_tiles","oracle_ev_curve_input")):
        raise ValueError("Visible input contract must explicitly exclude oracle features")
    c=config["model"]
    known={"action_field_width","action_heads","action_layers","action_mlp_width","action_width","fusion_layers","fusion_mlp_width","fusion_width","global_feature_count","global_width","history_convolution_kernel","history_heads","history_layers","history_mlp_width","history_recent_heads","history_recent_window","history_width","max_candidates","max_history","padded_visible_channels","tile_count","tile_heads","tile_layers","tile_mlp_width","tile_width","visible_channels"}
    if set(c)-known: raise ValueError(f"Unknown architecture fields: {sorted(set(c)-known)}")
    for key in ("tile_width","tile_heads","tile_layers","tile_mlp_width","history_width","history_heads","history_layers","history_mlp_width","action_width","action_field_width","action_layers","action_mlp_width","fusion_width","fusion_layers","fusion_mlp_width","global_width"):
        if type(c.get(key)) is not int or not 1<=c[key]<=16384:
            raise ValueError(f"Invalid architecture dimension: {key}")
    for key,value in (("tile_count",34),("global_feature_count",64),("max_history",200),("max_candidates",64),("visible_channels",1012)):
        if type(c.get(key)) is not int or c[key]!=value:
            raise ValueError(f"Unsupported input contract: {key} must be {value}")
    if c["tile_width"]%c["tile_heads"] or c["history_width"]%c["history_heads"]:
        raise ValueError("Attention width must be divisible by heads")
    if c["action_width"]!=c["fusion_width"]:
        raise ValueError("Action and fusion widths must match for the interaction scorer")
    return c

def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def parameter_shapes(c):
    shapes = {}
    def dense(p, a, b):
        shapes[p + "/kernel"] = (a, b)
        shapes[p + "/bias"] = (b,)
    def norm(p, n): shapes[p + "/scale"] = (n,)
    def swiglu(p, w, m):
        dense(p + "/in", w, 2*m)
        dense(p + "/out", m, w)
    def residual(p, w, m):
        norm(p + "/norm", w)
        swiglu(p + "/swiglu", w, m)
    def transformer(p, w, h, m):
        norm(p + "/attention_norm", w)
        norm(p + "/mlp_norm", w)
        for q in ("query", "key", "value"):
            shapes[f"{p}/attention/{q}/kernel"] = (w, h, w//h)
            shapes[f"{p}/attention/{q}/bias"] = (h, w//h)
        shapes[p + "/attention/out/kernel"] = (h, w//h, w)
        shapes[p + "/attention/out/bias"] = (w,)
        swiglu(p + "/swiglu", w, m)
    tw, hw, aw, fw = [c[k] for k in ("tile_width", "history_width", "action_width", "fusion_width")]
    dense("semantic_projection", 27, tw)
    for name, n in (("tile",34),("suit",4),("rank",10),("terminal_honor",3)):
        shapes[f"tile_encoder/{name}_embedding/embedding"] = (n,tw)
    shapes["tile_encoder/state_token"] = (1,1,tw)
    for i in range(c["tile_layers"]): transformer(f"tile_encoder/block_{i}",tw,c["tile_heads"],c["tile_mlp_width"])
    norm("tile_encoder/output_norm",tw)
    for name, n in (("event_type",97),("actor",6),("target",6),("tile",38),("tsumogiri",4),("meld_orientation",6),("round_position",258),("recency",c["max_history"]+2),("consumed_tile",38)):
        shapes[f"history_encoder/{name}_embedding/embedding"] = (n,hw)
    shapes["history_encoder/history_token"] = (1,1,hw)
    for i in range(c["history_layers"]): transformer(f"history_encoder/block_{i}",hw,c["history_heads"],c["history_mlp_width"])
    norm("history_encoder/output_norm",hw)
    dense("global_encoder/hidden",c["global_feature_count"],256)
    dense("global_encoder/output",256,c["global_width"])
    dense("fusion_projection",tw+hw+c["global_width"],fw)
    for i in range(c["fusion_layers"]): residual(f"fusion_block_{i}",fw,c["fusion_mlp_width"])
    norm("state_norm",fw)
    af=c["action_field_width"]
    for name,n in (("kind",12),("tile",38),("called_tile",38),("target",6),("from_draw",3),("red_mask",17),("phase",5),("consumed_tile",38)):
        shapes[f"action_encoder/{name}_embedding/embedding"]=(n,af)
    shapes["action_encoder/rule_flag_embedding"]=(16,2,af)
    dense("action_encoder/input_projection",9*af,aw)
    for i in range(c["action_layers"]): residual(f"action_encoder/block_{i}",aw,c["action_mlp_width"])
    norm("action_encoder/output_norm",aw)
    for name in ("state","action","interaction"): dense(f"action_scorer/{name}",aw,aw)
    dense("action_scorer/output",aw,1)
    shapes["action_scorer/kind_bias"]=(11,)
    return shapes

def flatten(tree, prefix=""):
    import numpy as np
    result={}
    for k,v in tree.items():
        name=f"{prefix}/{k}".lstrip("/")
        if isinstance(v,dict): result.update(flatten(v,name))
        else: result[name]=np.asarray(v)
    return result

def validate(arrays, config):
    import numpy as np
    expected=parameter_shapes(validate_config(config))
    if arrays.keys()!=expected.keys(): raise ValueError(f"Parameter keys differ: {arrays.keys() ^ expected.keys()}")
    for k,v in arrays.items():
        if v.shape!=expected[k] or v.dtype!=np.float32 or not np.isfinite(v).all():
            raise ValueError(f"Invalid parameter {k}: {v.shape}, {v.dtype}")
    return sum(v.size for v in arrays.values())

def export_actor(source, output, *, model_id=None, config_path=None, params_key="params", architecture=ARCHITECTURE, expected_sha256=None):
    import numpy as np
    import msgpack
    from safetensors.numpy import save_file
    source,output=Path(source),Path(output)
    checkpoint=source/"latest.msgpack" if source.is_dir() else source
    config_path=Path(config_path) if config_path else checkpoint.parent/"config.json"
    name=validate_model_id(model_id or output.name)
    if output.name!=name: raise ValueError("Output directory name must match model ID")
    if output.exists(): raise FileExistsError(f"Output already exists; use a new model/version directory: {output}")
    origin=sha256(checkpoint)
    if expected_sha256 and origin!=expected_sha256.lower(): raise ValueError("Source checkpoint hash mismatch")
    config=json.loads(config_path.read_text(encoding="utf-8-sig"))
    validate_config(config,architecture)
    def ext(code, data):
        if code==1:
            shape,dtype,buffer=msgpack.unpackb(data,raw=False)
            return np.frombuffer(buffer,dtype=np.dtype(dtype)).reshape(shape).copy()
        if code==3:
            dtype,buffer=msgpack.unpackb(data,raw=False)
            return np.frombuffer(buffer,dtype=np.dtype(dtype))[0]
        if code==2: return complex(*msgpack.unpackb(data))
        raise ValueError(f"Unsupported MessagePack extension {code}")
    raw=msgpack.unpackb(checkpoint.read_bytes(),ext_hook=ext,raw=False,strict_map_key=False)
    tree=raw
    for key in filter(None,params_key.split("/")):
        if not isinstance(tree,dict) or key not in tree: raise ValueError(f"Actor parameter subtree not found: {params_key}")
        tree=tree[key]
    if not isinstance(tree,dict): raise ValueError("Actor parameters must be a named tensor tree")
    arrays={k:np.ascontiguousarray(v) for k,v in flatten(tree).items()}
    count=validate(arrays,config)
    output.parent.mkdir(parents=True,exist_ok=True)
    # Publish only a fully written and independently reloaded model directory.
    with tempfile.TemporaryDirectory(prefix=".rin-export-",dir=output.parent) as temp:
        stage=Path(temp)/name; stage.mkdir()
        save_file(arrays,stage/"actor.safetensors",metadata={"format":FORMAT,"architecture":architecture,"layout":"flax","model":name})
        (stage/"config.json").write_bytes(config_path.read_bytes())
        manifest={"format":FORMAT,"architecture":architecture,"model":name,"source_format":"flax-msgpack","params_key":params_key,"original_sha256":origin,"actor_sha256":sha256(stage/"actor.safetensors"),"config_sha256":sha256(stage/"config.json"),"converter_sha256":sha256(__file__),"parameter_count":count,"layout":"Flax named axes; explicit DenseGeneral flattening by native loader","optimizer_state_exported":False,"tensors":{k:{"shape":list(v.shape),"dtype":str(v.dtype),"sha256":hashlib.sha256(v.tobytes()).hexdigest()} for k,v in arrays.items()}}
        (stage/"conversion-manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
        restored,_,_=load_actor_arrays(stage)
        if any(not np.array_equal(v,restored[k]) for k,v in arrays.items()): raise ValueError("Export round-trip changed weights")
        stage.rename(output)
    return manifest

def load_actor_arrays(path):
    from safetensors import safe_open
    from safetensors.numpy import load_file
    path=Path(path)
    m=json.loads((path/"conversion-manifest.json").read_text())
    validate_model_id(m["model"])
    if m["format"]!=FORMAT or m["architecture"]!=ARCHITECTURE: raise ValueError("Unsupported model format or architecture")
    if path.name!=m["model"]: raise ValueError("Model directory and manifest identity differ")
    for filename,key in (("actor.safetensors","actor_sha256"),("config.json","config_sha256")):
        if sha256(path/filename)!=m[key]: raise ValueError(f"Hash mismatch: {filename}")
    with safe_open(path/"actor.safetensors",framework="numpy") as f:
        if f.metadata()!={"format":FORMAT,"architecture":ARCHITECTURE,"layout":"flax","model":m["model"]}: raise ValueError("Safetensors metadata mismatch")
    arrays=load_file(path/"actor.safetensors")
    config=json.loads((path/"config.json").read_text(encoding="utf-8-sig"))
    if validate(arrays,config)!=m["parameter_count"] or arrays.keys()!=m["tensors"].keys(): raise ValueError("Manifest tensor inventory mismatch")
    for k,v in arrays.items():
        if list(v.shape)!=m["tensors"][k]["shape"] or str(v.dtype)!=m["tensors"][k]["dtype"]: raise ValueError(f"Tensor metadata mismatch: {k}")
        if hashlib.sha256(v.tobytes()).hexdigest()!=m["tensors"][k]["sha256"]: raise ValueError(f"Tensor hash mismatch: {k}")
    return arrays,config,m
