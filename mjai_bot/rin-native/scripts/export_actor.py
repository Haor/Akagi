"""Convert any supported RIN actor checkpoint to a versioned safetensors bundle."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from rin.inference.native.checkpoint import ARCHITECTURE,export_actor
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--model-id",help="Defaults to the output directory name")
    p.add_argument("--config",help="Defaults to config.json beside the checkpoint")
    p.add_argument("--params-key",default="params",help="Slash-separated actor subtree; empty for a params-only root")
    p.add_argument("--architecture",default=ARCHITECTURE)
    p.add_argument("--expected-sha256",help="Optional independently supplied source checkpoint digest")
    a=p.parse_args()
    m=export_actor(a.source,a.output,model_id=a.model_id,config_path=a.config,params_key=a.params_key,architecture=a.architecture,expected_sha256=a.expected_sha256)
    print(m["model"],m["parameter_count"],m["actor_sha256"])
