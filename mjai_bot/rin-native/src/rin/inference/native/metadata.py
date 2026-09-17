"""Public legal candidates for Akagi's HUD and local replay analysis."""
import json

import numpy as np

from rin.actions import ActionKind
from rin.policy import protocol_class_ids
from .engine import SemanticRINMjaiEngine, canonical_action_to_mjai


def model_identity(predictor, engine):
    manifest = predictor.manifest
    return {
        "model_id": manifest["model"],
        "actor_sha256": manifest["actor_sha256"],
        "original_sha256": manifest.get("original_sha256"),
        "architecture": manifest["architecture"],
        "history_contract": engine.history_contract,
        "shanten_contract": engine.shanten_contract,
    }


def show_candidates(candidates, model):
    labels = {"dahai": "Discard", "reach": "Riichi", "none": "Pass",
              "chi": "Chi", "pon": "Pon", "daiminkan": "Open kan",
              "ankan": "Closed kan", "kakan": "Added kan", "hora": "Win",
              "ryukyoku": "Abortive draw"}
    items = []
    # Display one representative of each protocol class, in policy order.
    seen = set()
    for candidate in sorted(candidates, key=lambda c: (c["protocol_probability"],
                                                       c["probability"]), reverse=True):
        group = candidate["protocol_class"]
        if group in seen:
            continue
        seen.add(group)
        action = candidate["action"]
        tile_action = candidate.get("continuation", action)
        pais = ([tile_action["pai"]] if "pai" in tile_action else []) + action.get("consumed", [])
        item = {"label": labels.get(action["type"], action["type"]), "pais": pais,
                "value": f'{candidate["protocol_probability"]:.2%}'}
        if candidate["selected"]:
            item["color"] = "#00b894"
            item["note"] = "Recommended"
        items.append(item)
        if len(items) == 8:
            break
    return {"title": f"RIN {model}", "items": items}


class ReviewableEngine(SemanticRINMjaiEngine):
    def _record_atomic_decision(self, *, game_idx, decision, logits, chosen):
        values = np.asarray(logits, dtype=np.float64)
        probabilities = np.exp(values - values.max())
        probabilities /= probabilities.sum()
        classes = protocol_class_ids([binding.action for binding in decision.bindings])
        totals = np.bincount(classes, weights=probabilities)
        memory = self.memories[game_idx]
        candidates = []
        for index, binding in enumerate(decision.bindings):
            action = canonical_action_to_mjai(binding.action, seat=memory.seat, events=memory.full_events)
            candidate = {"action": action, "probability": float(probabilities[index]),
                         "protocol_probability": float(totals[classes[index]]),
                         "protocol_class": int(classes[index]), "selected": index == chosen}
            if binding.action.kind == ActionKind.RIICHI_DISCARD:
                candidate["continuation"] = action
                candidate["action"] = {"type": "reach", "actor": memory.seat}
            candidates.append(candidate)
        self._decision_metadata[game_idx] = self._metadata(candidates)

    def _metadata(self, candidates, *, continuation=False):
        identity = model_identity(self.predictor, self)
        return {"decision": True, "continuation": continuation, "model_identity": identity,
                "candidates": candidates, "show": show_candidates(candidates, identity["model_id"])}

    def react_batch(self, game_states):
        self._decision_metadata = {}
        responses = super().react_batch(game_states)
        result = []
        for scene, response in zip(game_states, responses):
            event = json.loads(response)
            metadata = self._decision_metadata.get(int(scene.game_index))
            if metadata is None:
                # Atomic riichi is emitted as reach followed by its forced discard.
                metadata = self._metadata([{"action": dict(event), "probability": 1.0,
                                           "protocol_probability": 1.0, "protocol_class": 0,
                                           "selected": True}], continuation=True)
            event["meta"] = metadata
            result.append(json.dumps(event, separators=(",", ":"), allow_nan=False))
        return result
