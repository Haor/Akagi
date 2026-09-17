"""Shared policy semantics for distillation, deployment, and online RL."""
from __future__ import annotations
from collections.abc import Sequence
import numpy as np
from rin.actions import CanonicalAction

def protocol_class_ids(actions: Sequence[CanonicalAction]) -> np.ndarray:
    """Assign stable first-seen ids to Mortal-equivalent atomic actions."""
    if not actions:
        raise ValueError('at least one legal action is required')
    ids = np.empty(len(actions), dtype=np.int32)
    classes: dict[tuple[int, ...], int] = {}
    for index, action in enumerate(actions):
        key = action.protocol_equivalence_key
        class_id = classes.setdefault(key, len(classes))
        ids[index] = class_id
    return ids

def greedy_protocol_candidate(logits: np.ndarray, actions: Sequence[CanonicalAction]) -> int:
    """Choose the best equivalence class, then its best atomic member.

    Distillation supervises the marginal probability of a Mortal-equivalence
    class. Deployment must therefore select the class with the largest summed
    atomic probability, not the largest individual alias. Within the winning
    class, the largest atomic logit is the deterministic representative.
    Stochastic RL may still sample the atomic softmax directly; its behavior
    log-probability is then the ordinary atomic log-softmax.
    """
    values = np.asarray(logits, dtype=np.float64)
    if values.shape != (len(actions),):
        raise ValueError('logits must contain exactly one value per legal action')
    if not np.isfinite(values).all():
        raise ValueError('legal action logits must be finite')
    class_ids = protocol_class_ids(actions)
    weights = np.exp(values - values.max())
    class_weights = np.zeros(int(class_ids.max()) + 1, dtype=np.float64)
    np.add.at(class_weights, class_ids, weights)
    winning_class = int(np.argmax(class_weights))
    members = np.flatnonzero(class_ids == winning_class)
    return int(members[np.argmax(values[members])])
