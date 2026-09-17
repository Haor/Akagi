"""Synthetic checks for binding, public masks, and optional-head isolation."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import msgpack
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rin.actions import ActionKind
from rin.inference.native import checkpoint, observer_checkpoint
from rin.inference.native.actor_torch import Predictor, tensor_inputs
from rin.inference.native.observer_torch import ObserverReadoutTorch, observer_metadata


def configurations():
    actor = {"tile_width": 8, "tile_heads": 2, "tile_layers": 1, "tile_mlp_width": 16,
             "history_width": 8, "history_heads": 2, "history_layers": 1, "history_mlp_width": 16,
             "action_width": 8, "action_field_width": 2, "action_layers": 1, "action_mlp_width": 16,
             "fusion_width": 8, "fusion_layers": 1, "fusion_mlp_width": 16,
             "global_width": 4, "global_feature_count": 64, "max_history": 200, "max_candidates": 64,
             "tile_count": 34, "visible_channels": 1012}
    readout = {"state_width": 8, "tile_width": 8, "history_width": 8, "global_width": 4,
               "width": 8, "heads": 2, "mlp_width": 16, "tile_count": 34, "opponent_count": 3,
               "loss_point_scale": 10000.0, "payment_distribution": True}
    contract = {"schema_version": observer_checkpoint.ARCHITECTURE, "kind": observer_checkpoint.KIND,
                "readout": readout, "payment_distribution": {
                    "edges_points": list(observer_checkpoint.PAYMENT_BUCKET_EDGES),
                    "interval": "left_closed_right_open_with_final_open_tail",
                    "condition": "one_or_more_eligible_ron_declarations_including_zero_settlement"}}
    observer = {"schema_version": "rin.observer-readout-run.v1", "input_boundary": "actor_representations_stop_gradient",
                "representation_timing": "post_ppo_actor_update", "representation_actor_sha256": "1" * 64,
                "observer": readout, "readout_contract": contract}
    return {"model": actor}, observer


def weights(shapes):
    rng = np.random.default_rng(12)
    return {name: np.ones(shape, dtype=np.float32) if name.endswith("/scale")
            else (rng.standard_normal(shape) * 0.02).astype(np.float32) for name, shape in shapes.items()}


def inputs():
    history = {name: np.zeros((1, 200), dtype=np.int32) for name in
               ("event_type", "actor", "target", "tile", "tsumogiri", "meld_orientation", "round_position", "recency")}
    history["consumed_tiles"] = np.full((1, 200, 4), -1, dtype=np.int32)
    candidates = {name: np.full((1, 64), -1, dtype=np.int32) for name in
                  ("kind", "tile", "called_tile", "relative_target", "from_draw", "red_tile_mask", "decision_phase", "rule_flags")}
    for name in ("from_draw", "red_tile_mask", "rule_flags"):
        candidates[name].fill(0)
    candidates["kind"][0, :3] = [int(ActionKind.DISCARD), int(ActionKind.RIICHI_DISCARD), int(ActionKind.PASS)]
    candidates["tile"][0, :2] = [4, 34]
    candidates["consumed_tiles"] = np.full((1, 64, 4), -1, dtype=np.int32)
    return {"tile_features": np.zeros((1, 34, 27), dtype=np.float32), "history": history,
            "history_mask": np.arange(200)[None] < 3, "global_features": np.zeros((1, 64), dtype=np.float32),
            "candidates": candidates, "candidate_mask": np.arange(64)[None] < 3}


class ObserverTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root / "model"
        self.model.mkdir()
        self.actor_config, self.config = configurations()
        self.manifest = {"model": "test", "architecture": checkpoint.ARCHITECTURE,
                         "original_sha256": "1" * 64, "config_sha256": "2" * 64, "actor_sha256": "3" * 64}
        self.actor_arrays = weights(checkpoint.parameter_shapes(self.actor_config["model"]))
        self.arrays = weights(observer_checkpoint.parameter_shapes(self.config["observer"]))
        def encode(value):
            if isinstance(value, np.ndarray):
                return msgpack.ExtType(1, msgpack.packb((value.shape, str(value.dtype), value.tobytes()), use_bin_type=True))
            raise TypeError(type(value))
        tree = {}
        for name, value in self.arrays.items():
            target = tree
            parts = name.split("/")
            for part in parts[:-1]:
                target = target.setdefault(part, {})
            target[parts[-1]] = value
        self.source = self.root / "source.msgpack"
        self.source.write_bytes(msgpack.packb({"params": tree, "optimizer_state": {"private": "omitted"}}, default=encode, use_bin_type=True))
        self.config_path = self.root / "config.json"
        self.config_path.write_text(json.dumps(self.config))
        with patch.object(observer_checkpoint, "load_actor_arrays", return_value=(self.actor_arrays, self.actor_config, self.manifest)):
            self.converted = observer_checkpoint.export_observer(self.source, self.config_path, self.model)

    def predictor(self):
        with patch.object(checkpoint, "load_actor_arrays", return_value=(self.actor_arrays, self.actor_config, self.manifest)):
            return Predictor(self.model, device="cpu", attention="sdpa", cpu_threads=1)

    def test_generic_conversion_roundtrip_and_exact_actor_binding(self):
        loaded, _, manifest = observer_checkpoint.load_observer_arrays(self.model, self.actor_config, self.manifest)
        for name in loaded:
            np.testing.assert_array_equal(loaded[name], self.arrays[name])
        self.assertFalse(manifest["optimizer_state_exported"])
        for field in ("actor_sha256", "original_sha256", "config_sha256"):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "identity mismatch"):
                observer_checkpoint.load_observer_arrays(self.model, self.actor_config, {**self.manifest, field: "9" * 64})

    def test_corrupt_head_is_rejected_without_disabling_actor(self):
        with (self.model / "observer.safetensors").open("ab") as stream:
            stream.write(b"corrupt")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            observer_checkpoint.load_observer_arrays(self.model, self.actor_config, self.manifest)
        predictor = self.predictor()
        self.assertEqual(predictor.observer_status, "error")
        self.assertTrue(np.isfinite(predictor.predict(inputs())).all())
        self.assertEqual(observer_metadata(predictor, seat=0)["status"], "error")

    def test_one_actor_forward_and_identical_policy_with_observer_enabled(self):
        predictor = self.predictor()
        batch = inputs()
        with torch.inference_mode():
            expected = predictor.actor(**tensor_inputs(batch, predictor.device)).numpy()
        with patch.object(predictor.actor, "forward", wraps=predictor.actor.forward) as forward:
            actual = predictor.predict(batch)
        self.assertEqual(forward.call_count, 1)
        np.testing.assert_array_equal(expected, actual)
        ready = observer_metadata(predictor, seat=2)
        self.assertEqual(ready["opponent_order"], [3, 0, 1])
        self.assertEqual([c["candidate_index"] for c in ready["candidates"]], [0, 1])
        self.assertEqual(ready["observer_identity"]["representation_actor_sha256"], "1" * 64)
        self.assertEqual(observer_metadata(predictor, seat=2, continuation=True)["status"], "not_evaluated")

    def test_padded_history_cannot_change_readout_and_joint_probabilities_agree(self):
        predictor = self.predictor()
        batch = tensor_inputs(inputs(), predictor.device)
        with torch.inference_mode():
            _, representations = predictor.actor(**batch, return_observer_inputs=True)
            original = predictor.observer(representations, batch["candidates"], batch["candidate_mask"])
            changed = {name: value.clone() for name, value in representations.items()}
            changed["history_tokens"][~changed["history_mask"]] = 1e20
            changed_output = predictor.observer(changed, batch["candidates"], batch["candidate_mask"])
            for name in original:
                self.assertTrue(torch.equal(original[name], changed_output[name]), name)
            probabilities = predictor.observer.probabilities(original)
        active = probabilities["candidate_applicable"]
        joint = probabilities["ron_joint_probability"][active]
        torch.testing.assert_close(joint.sum(-1), torch.ones(2))
        torch.testing.assert_close(probabilities["any_ron_probability"][active], 1 - joint[:, 0])
        self.assertTrue((probabilities["ron_probability"][active] <= probabilities["any_ron_probability"][active, None] + 1e-6).all())
        self.assertTrue((probabilities["conditional_loss_points"] >= 0).all())

    def test_readout_failure_preserves_actor_result(self):
        predictor = self.predictor()
        batch = inputs()
        with torch.inference_mode():
            expected = predictor.actor(**tensor_inputs(batch, predictor.device)).numpy()
        with patch.object(predictor.observer, "forward", side_effect=ValueError("invalid head")):
            actual = predictor.predict(batch)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(predictor.observer_status, "error")
        self.assertIsNone(predictor.last_observer_outputs)
        self.assertEqual(observer_metadata(predictor, seat=0)["reason"], "observer_inference_failed")


if __name__ == "__main__":
    unittest.main()
