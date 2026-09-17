"""Extract the NumPy deployment boundary from a compatible RIN source tree.

The input is a source directory containing rin/inference/mjai_engine.py.
Actor loading and training-only dependencies are deliberately excluded.
"""
import argparse
import ast
from pathlib import Path
import shutil


BASE_INIT = '''
def __init__(self, *, predictor, player_state_type, name="RIN Native", mortal_version=3,
             rule_profile=RuleProfile(), history_contract=COMPLETE_ROUND_HISTORY):
    from types import SimpleNamespace
    self.name = name
    self.predictor = predictor
    self.config = SimpleNamespace(**predictor.config["model"])
    self.player_state_type = player_state_type
    self.mortal_version = mortal_version
    self.rule_profile = rule_profile
    if history_contract != COMPLETE_ROUND_HISTORY:
        raise ValueError("Native deployment requires complete-round public history")
    self.history_contract = history_contract
    self.player_ids = ()
    self.memories = {}
    self.action_counts = Counter()
    self.decision_count = 0
    self.riichi_continuation_count = 0
    self.maximum_candidate_count = 0
    self.params = None
    self._apply = lambda unused, inputs: self.predictor.predict(inputs)
'''
SEMANTIC_INIT = '''
def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.semantic_memories = {}
    self.shanten_contract = PUBLIC_SHANTEN_CONTRACT
'''


def extract(source, destination):
    tree = ast.parse(source.read_text(encoding="utf-8"))
    blocked = {"flax", "jax", "rin.model", "rin.semantic_frontend", "rin.ukeire"}
    kept = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module in blocked:
            continue
        if isinstance(node, ast.Import) and any(a.name in blocked for a in node.names):
            continue
        if isinstance(node, ast.FunctionDef) and node.name.startswith("load_"):
            continue
        if isinstance(node, ast.ClassDef) and node.name in ("RINMjaiEngine", "SemanticRINMjaiEngine"):
            replacement = BASE_INIT if node.name == "RINMjaiEngine" else SEMANTIC_INIT
            node.body = [ast.parse(replacement).body[0] if isinstance(n, ast.FunctionDef)
                         and n.name == "__init__" else n for n in node.body]
        kept.append(node)
    tree.body = kept
    text = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
    for required in ("end_kyoku_with_log", "native_public_shanten", "PUBLIC_SHANTEN_CONTRACT"):
        if required not in text:
            raise ValueError(f"Source adapter lacks required contract: {required}")
    destination.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    args = parser.parse_args()
    source = args.source_root / "rin"
    target = Path(__file__).resolve().parents[1] / "src" / "rin"
    for directory in (target, target / "data", target / "inference", target / "inference/native"):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "__init__.py").touch()
    for relative in ("actions.py", "data/atomic_adapter.py", "data/public_state.py",
                     "data/semantic_state.py", "data/input_contract.py"):
        shutil.copyfile(source / relative, target / relative)
    policy = ast.parse((source / "policy.py").read_text(encoding="utf-8"))
    policy.body = [n for n in policy.body if not isinstance(n, ast.FunctionDef)
                   or n.name in {"protocol_class_ids", "greedy_protocol_candidate"}]
    (target / "policy.py").write_text(ast.unparse(policy) + "\n", encoding="utf-8")
    replay = ast.parse((source / "data/libriichi_replay.py").read_text(encoding="utf-8"))
    replay.body = [n for n in replay.body if isinstance(n, (ast.Import, ast.ImportFrom))
                   or isinstance(n, ast.FunctionDef) and n.name in {"_mortal_cans", "snapshot_player_state"}]
    (target / "data/libriichi_replay.py").write_text(ast.unparse(replay) + "\n", encoding="utf-8")
    extract(source / "inference/mjai_engine.py", target / "inference/native/engine.py")


if __name__ == "__main__":
    main()
