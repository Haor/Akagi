"""Behavior tests using the installed, qualified Libriichi extension."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rin.actions import ActionKind
from rin.inference.native.session import Session, public_event


class Predictor:
    config = {"model": {"visible_channels": 1012, "tile_count": 34,
                        "max_history": 200, "max_candidates": 64}}
    manifest = {"model": "test", "actor_sha256": "test", "architecture": "rin.semantic-dynamic.v1"}

    def __init__(self, preferred=ActionKind.DISCARD):
        self.preferred = preferred
        self.inputs = []

    def predict(self, inputs):
        self.inputs.append(deepcopy(inputs))
        return np.where(inputs["candidates"]["kind"] == int(self.preferred), 20.0, 0.0)


def opening():
    hand = ["1m", "2m", "3m", "1p", "2p", "3p", "1s", "2s", "3s", "E", "E", "4p", "5p"]
    return [{"type": "start_game", "id": 0},
            {"type": "start_kyoku", "bakaze": "E", "kyoku": 1, "honba": 0,
             "kyotaku": 0, "oya": 0, "dora_marker": "9m", "scores": [25000] * 4,
             "tehais": [hand, ["?"] * 13, ["?"] * 13, ["?"] * 13]},
            {"type": "tsumo", "actor": 0, "pai": "9s"}]


def assert_arrays_equal(test, left, right):
    test.assertEqual(left.keys(), right.keys())
    for key in left:
        if isinstance(left[key], dict):
            assert_arrays_equal(test, left[key], right[key])
        else:
            np.testing.assert_array_equal(left[key], right[key], err_msg=key)


class SessionTests(unittest.TestCase):
    def test_hidden_hands_and_opponent_draws_do_not_change_inputs(self):
        first = opening()
        second = deepcopy(first)
        for seat in (1, 2, 3):
            second[1]["tehais"][seat] = ["C"] * 13
        first_predictor, second_predictor = Predictor(), Predictor()
        a, b = Session(first_predictor, 0), Session(second_predictor, 0)
        a.react(first)
        b.react(second)
        suffix = [{"type": "dahai", "actor": 0, "pai": "9s", "tsumogiri": True},
                  {"type": "tsumo", "actor": 1, "pai": "?"},
                  {"type": "dahai", "actor": 1, "pai": "9p", "tsumogiri": True},
                  {"type": "tsumo", "actor": 2, "pai": "?"},
                  {"type": "dahai", "actor": 2, "pai": "N", "tsumogiri": True},
                  {"type": "tsumo", "actor": 3, "pai": "?"},
                  {"type": "dahai", "actor": 3, "pai": "C", "tsumogiri": True},
                  {"type": "tsumo", "actor": 0, "pai": "8s"}]
        revealed = deepcopy(suffix)
        for event in revealed:
            if event["type"] == "tsumo" and event["actor"] != 0:
                event["pai"] = "5mr"
        a.react(suffix)
        b.react(revealed)
        for left, right in zip(first_predictor.inputs, second_predictor.inputs):
            assert_arrays_equal(self, left, right)
        self.assertEqual(a.engine.memories[0].full_events, b.engine.memories[0].full_events)

    def test_complete_round_retains_unobserved_tail(self):
        session = Session(Predictor(), 0)
        session.react(opening())
        tail = [{"type": "dahai", "actor": 0, "pai": "9s", "tsumogiri": True},
                {"type": "tsumo", "actor": 1, "pai": "?"},
                {"type": "dahai", "actor": 1, "pai": "C", "tsumogiri": True},
                {"type": "ryukyoku", "deltas": [0] * 4}, {"type": "end_kyoku"}]
        session.react(tail)
        memory = session.engine.memories[0]
        self.assertEqual(memory.full_events[-len(tail):], tail)
        self.assertEqual(memory.current_kyoku_events, [])
        self.assertEqual(sum(e["type"] == "end_kyoku" for e in memory.full_events), 1)
        next_round = deepcopy(opening()[1])
        next_round["kyoku"], next_round["oya"] = 2, 1
        session.react([next_round])
        self.assertIn(tail[-3], memory.full_events)

    def test_unplayed_riichi_suggestion_is_not_applied(self):
        predictor = Predictor(ActionKind.RIICHI_DISCARD)
        session = Session(predictor, 0)
        reaction = session.react(opening())
        self.assertEqual(reaction["type"], "reach")
        self.assertIsNotNone(session.engine.memories[0].pending_riichi)
        session.react([{"type": "dahai", "actor": 0, "pai": "9s", "tsumogiri": True}])
        self.assertIsNone(session.engine.memories[0].pending_riichi)
        self.assertFalse(any(e["type"] == "reach" for e in session.engine.memories[0].full_events))

    def test_actual_riichi_produces_marked_continuation(self):
        session = Session(Predictor(ActionKind.RIICHI_DISCARD), 0)
        first = session.react(opening())
        candidate = next(c for c in first["meta"]["candidates"] if c["selected"])
        response = session.react([{"type": "reach", "actor": 0}])
        self.assertEqual({k: v for k, v in response.items() if k != "meta"}, candidate["continuation"])
        self.assertTrue(response["meta"]["continuation"])

    def test_pass_is_a_decision_with_candidate_probabilities(self):
        session = Session(Predictor(ActionKind.PASS), 0)
        session.react(opening())
        response = session.react([
            {"type": "dahai", "actor": 0, "pai": "9s", "tsumogiri": True},
            {"type": "tsumo", "actor": 1, "pai": "?"},
            {"type": "dahai", "actor": 1, "pai": "E", "tsumogiri": True}])
        self.assertEqual(response["type"], "none")
        self.assertTrue(response["meta"]["decision"])
        self.assertAlmostEqual(sum(c["probability"] for c in response["meta"]["candidates"]), 1.0)
        self.assertEqual(sum(c["selected"] for c in response["meta"]["candidates"]), 1)
        self.assertTrue(response["meta"]["show"]["items"])

    def test_private_annotations_are_removed(self):
        event = public_event({"type": "tsumo", "actor": 1, "pai": "5mr",
                              "meta": {"hidden": True}, "wall": ["C"], "ura_markers": ["P"]}, 0)
        self.assertEqual(event, {"type": "tsumo", "actor": 1, "pai": "?"})

    def test_actor_only_bundle_reports_missing_observer_with_current_identity(self):
        predictor = Predictor()
        predictor.manifest = {**predictor.manifest, "actor_sha256": "a" * 64}
        session = Session(predictor, 0)
        events = opening()
        self.assertEqual(session.react(events[:1]), {"type": "none"})
        response = session.react(events[1:])
        self.assertTrue(response["meta"]["decision"])
        self.assertEqual(response["meta"]["observer"], {
            "status": "unavailable", "reason": "no_compatible_observer",
            "actor_identity": predictor.manifest["actor_sha256"], "observer_identity": None,
        })


if __name__ == "__main__":
    unittest.main()
