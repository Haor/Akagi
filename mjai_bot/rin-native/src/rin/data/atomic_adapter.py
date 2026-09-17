"""Mortal-v4 to CanonicalAction v1 adapter primitives.

The functions in this module are independent of Libriichi's Python extension.
Integration code snapshots the visible PlayerState fields and passes them here,
which keeps protocol semantics unit-testable without a Mahjong runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import Iterable, Mapping, Sequence

import numpy as np

from rin.actions import (
    ActionKind,
    CanonicalAction,
    DecisionPhase,
    NO_TILE,
    RelativeTarget,
    assert_unique_candidates,
)


MORTAL_ACTION_COUNT = 46
MORTAL_DISCARD_COUNT = 37
MORTAL_RIICHI = 37
MORTAL_CHI_LOW = 38
MORTAL_CHI_MID = 39
MORTAL_CHI_HIGH = 40
MORTAL_PON = 41
MORTAL_KAN = 42
MORTAL_AGARI = 43
MORTAL_KYUUSHU = 44
MORTAL_PASS = 45

RED_TO_BASE = {34: 4, 35: 13, 36: 22}
BASE_TO_RED = {base: red for red, base in RED_TO_BASE.items()}


@dataclass(frozen=True, slots=True)
class MortalCans:
    can_discard: bool = False
    can_chi_low: bool = False
    can_chi_mid: bool = False
    can_chi_high: bool = False
    can_pon: bool = False
    can_daiminkan: bool = False
    can_kakan: bool = False
    can_ankan: bool = False
    can_riichi: bool = False
    can_tsumo_agari: bool = False
    can_ron_agari: bool = False
    can_ryukyoku: bool = False
    target_actor: int = 0

    @property
    def can_pass(self) -> bool:
        return any(
            (
                self.can_chi_low,
                self.can_chi_mid,
                self.can_chi_high,
                self.can_pon,
                self.can_daiminkan,
                self.can_ron_agari,
            )
        )


@dataclass(frozen=True, slots=True)
class PlayerSnapshot:
    seat: int
    hand_counts: tuple[int, ...]
    reds_in_hand: tuple[bool, bool, bool]
    last_draw: int = NO_TILE
    last_discard: int = NO_TILE
    ankan_candidates: tuple[int, ...] = ()
    kakan_candidates: tuple[int, ...] = ()
    cans: MortalCans = MortalCans()

    def __post_init__(self) -> None:
        if self.seat not in range(4):
            raise ValueError("seat must be in [0, 3]")
        if len(self.hand_counts) != 34 or any(count not in range(5) for count in self.hand_counts):
            raise ValueError("hand_counts must contain 34 values in [0, 4]")
        if len(self.reds_in_hand) != 3:
            raise ValueError("reds_in_hand must contain three flags")


@dataclass(frozen=True, slots=True)
class CandidateBinding:
    action: CanonicalAction
    primary_action: int
    continuation_action: int = -1

    @property
    def teacher_class_key(self) -> tuple[int, int]:
        return self.primary_action, self.continuation_action


@dataclass(frozen=True, slots=True)
class AtomicTeacherTarget:
    class_keys: tuple[tuple[int, int], ...]
    candidate_class_ids: np.ndarray
    class_probabilities: np.ndarray
    hard_class_id: int


def deaka(tile: int) -> int:
    return RED_TO_BASE.get(tile, tile)


def relative_target(seat: int, target: int) -> RelativeTarget:
    delta = (target - seat) % 4
    return RelativeTarget(delta)


def red_mask(consumed: Sequence[int]) -> int:
    mask = 0
    for index, tile in enumerate(consumed):
        if tile in RED_TO_BASE:
            mask |= 1 << index
    return mask


def padded_consumed(consumed: Iterable[int]) -> tuple[int, int, int, int]:
    values = tuple(sorted((int(tile) for tile in consumed), key=lambda tile: (deaka(tile), tile)))
    if len(values) > 4:
        raise ValueError("an action cannot consume more than four tiles")
    return (*values, *((NO_TILE,) * (4 - len(values))))


def _exact_hand_tiles(snapshot: PlayerSnapshot, base: int) -> tuple[int, ...]:
    count = snapshot.hand_counts[base]
    red = BASE_TO_RED.get(base)
    has_red = red is not None and snapshot.reds_in_hand[(base - 4) // 9]
    black_count = count - int(has_red)
    if black_count < 0:
        raise ValueError("red hand flag is inconsistent with hand_counts")
    return (base,) * black_count + ((red,) if has_red else ())


def _unique_choices(snapshot: PlayerSnapshot, base: int, count: int) -> tuple[tuple[int, ...], ...]:
    tiles = _exact_hand_tiles(snapshot, base)
    if len(tiles) < count:
        return ()
    return tuple(
        sorted(
            {padded_consumed(choice)[:count] for choice in combinations(tiles, count)},
            key=lambda choice: tuple((deaka(tile), tile) for tile in choice),
        )
    )


def _discard_variants(snapshot: PlayerSnapshot, tile: int) -> tuple[bool, ...]:
    if tile != snapshot.last_draw:
        return (False,)
    exact_count = _exact_hand_tiles(snapshot, deaka(tile)).count(tile)
    if exact_count < 1:
        raise ValueError("last_draw is absent from the hand snapshot")
    return (True, False) if exact_count >= 2 else (True,)


def _append_consuming_action(
    bindings: list[CandidateBinding],
    *,
    kind: ActionKind,
    tile: int,
    called_tile: int,
    consumed: Sequence[int],
    target: RelativeTarget,
    phase: DecisionPhase,
    primary_action: int,
    continuation_action: int = -1,
) -> None:
    padded = padded_consumed(consumed)
    bindings.append(
        CandidateBinding(
            action=CanonicalAction(
                kind=kind,
                tile=tile,
                called_tile=called_tile,
                consumed_tiles=padded,
                relative_target=target,
                red_tile_mask=red_mask(padded),
                decision_phase=phase,
            ),
            primary_action=primary_action,
            continuation_action=continuation_action,
        )
    )


def _chi_consumed_bases(called: int, mortal_action: int) -> tuple[int, int]:
    base = deaka(called)
    if mortal_action == MORTAL_CHI_LOW:
        return base + 1, base + 2
    if mortal_action == MORTAL_CHI_MID:
        return base - 1, base + 1
    if mortal_action == MORTAL_CHI_HIGH:
        return base - 2, base - 1
    raise ValueError(f"not a chi action: {mortal_action}")


def enumerate_mortal_atomic_candidates(
    snapshot: PlayerSnapshot,
    primary_mask: np.ndarray,
    *,
    riichi_continuation_mask: np.ndarray | None = None,
    kan_continuation_mask: np.ndarray | None = None,
) -> list[CandidateBinding]:
    """Enumerate every atomic action represented by a Mortal decision state.

    Multiple canonical candidates may share one teacher class when Mortal
    cannot express tsumogiri-vs-tedashi or red consumption choices.
    """

    primary_mask = np.asarray(primary_mask)
    if primary_mask.shape != (MORTAL_ACTION_COUNT,) or primary_mask.dtype != np.bool_:
        raise ValueError("primary_mask must be bool[46]")
    if not primary_mask.any():
        raise ValueError("a decision state must have at least one legal Mortal action")
    phase = DecisionPhase.AFTER_DRAW if snapshot.cans.can_discard else DecisionPhase.REACTION
    bindings: list[CandidateBinding] = []

    if primary_mask[MORTAL_RIICHI] != snapshot.cans.can_riichi:
        raise ValueError("riichi cans and Mortal mask disagree")
    if primary_mask[MORTAL_KAN] != any(
        (snapshot.cans.can_daiminkan, snapshot.cans.can_ankan, snapshot.cans.can_kakan)
    ):
        raise ValueError("kan cans and Mortal mask disagree")
    if primary_mask[MORTAL_AGARI] != any(
        (snapshot.cans.can_tsumo_agari, snapshot.cans.can_ron_agari)
    ):
        raise ValueError("agari cans and Mortal mask disagree")
    if primary_mask[MORTAL_KYUUSHU] != snapshot.cans.can_ryukyoku:
        raise ValueError("kyuushu cans and Mortal mask disagree")
    if primary_mask[MORTAL_PASS] != snapshot.cans.can_pass:
        raise ValueError("pass cans and Mortal mask disagree")

    for tile in np.flatnonzero(primary_mask[:MORTAL_DISCARD_COUNT]):
        for from_draw in _discard_variants(snapshot, int(tile)):
            bindings.append(
                CandidateBinding(
                    CanonicalAction(
                        kind=ActionKind.DISCARD,
                        tile=int(tile),
                        relative_target=RelativeTarget.NONE,
                        from_draw=from_draw,
                        decision_phase=phase,
                    ),
                    primary_action=int(tile),
                )
            )

    if primary_mask[MORTAL_RIICHI]:
        if riichi_continuation_mask is None:
            raise ValueError("riichi requires its discard continuation mask")
        continuation = np.asarray(riichi_continuation_mask)
        if continuation.shape != (MORTAL_ACTION_COUNT,) or continuation.dtype != np.bool_:
            raise ValueError("riichi continuation mask must be bool[46]")
        if continuation[MORTAL_DISCARD_COUNT:].any():
            raise ValueError("riichi continuation may only contain discard actions")
        for tile in np.flatnonzero(continuation[:MORTAL_DISCARD_COUNT]):
            for from_draw in _discard_variants(snapshot, int(tile)):
                bindings.append(
                    CandidateBinding(
                        CanonicalAction(
                            kind=ActionKind.RIICHI_DISCARD,
                            tile=int(tile),
                            relative_target=RelativeTarget.NONE,
                            from_draw=from_draw,
                            decision_phase=DecisionPhase.AFTER_DRAW,
                        ),
                        primary_action=MORTAL_RIICHI,
                        continuation_action=int(tile),
                    )
                )

    called = snapshot.last_discard
    target = relative_target(snapshot.seat, snapshot.cans.target_actor)
    for mortal_action, enabled in (
        (MORTAL_CHI_LOW, snapshot.cans.can_chi_low),
        (MORTAL_CHI_MID, snapshot.cans.can_chi_mid),
        (MORTAL_CHI_HIGH, snapshot.cans.can_chi_high),
    ):
        if primary_mask[mortal_action] != enabled:
            raise ValueError("chi cans and Mortal mask disagree")
        if not enabled:
            continue
        if called == NO_TILE:
            raise ValueError("chi requires the last discarded tile")
        bases = _chi_consumed_bases(called, mortal_action)
        variants = [_unique_choices(snapshot, base, 1) for base in bases]
        for left, right in product(*variants):
            consumed = (left[0], right[0])
            _append_consuming_action(
                bindings,
                kind=ActionKind.CHI,
                tile=called,
                called_tile=called,
                consumed=consumed,
                target=target,
                phase=DecisionPhase.REACTION,
                primary_action=mortal_action,
            )

    if primary_mask[MORTAL_PON] != snapshot.cans.can_pon:
        raise ValueError("pon cans and Mortal mask disagree")
    if snapshot.cans.can_pon:
        for consumed in _unique_choices(snapshot, deaka(called), 2):
            _append_consuming_action(
                bindings,
                kind=ActionKind.PON,
                tile=called,
                called_tile=called,
                consumed=consumed,
                target=target,
                phase=DecisionPhase.REACTION,
                primary_action=MORTAL_PON,
            )

    if primary_mask[MORTAL_KAN]:
        if kan_continuation_mask is None:
            raise ValueError("kan requires its tile continuation mask")
        continuation = np.asarray(kan_continuation_mask)
        if continuation.shape != (MORTAL_ACTION_COUNT,) or continuation.dtype != np.bool_:
            raise ValueError("kan continuation mask must be bool[46]")
        if continuation[34:].any():
            raise ValueError("kan continuation must use de-red 34-tile indices")

        if snapshot.cans.can_daiminkan:
            base = deaka(called)
            if not continuation[base]:
                raise ValueError("daiminkan continuation omits its called tile")
            for consumed in _unique_choices(snapshot, base, 3):
                _append_consuming_action(
                    bindings,
                    kind=ActionKind.DAIMINKAN,
                    tile=called,
                    called_tile=called,
                    consumed=consumed,
                    target=target,
                    phase=DecisionPhase.REACTION,
                    primary_action=MORTAL_KAN,
                    continuation_action=base,
                )
        for base in snapshot.ankan_candidates:
            if not continuation[base]:
                raise ValueError("ankan continuation omits a candidate tile")
            for consumed in _unique_choices(snapshot, base, 4):
                _append_consuming_action(
                    bindings,
                    kind=ActionKind.ANKAN,
                    tile=base,
                    called_tile=NO_TILE,
                    consumed=consumed,
                    target=RelativeTarget.SELF,
                    phase=DecisionPhase.KAN_SELECTION,
                    primary_action=MORTAL_KAN,
                    continuation_action=base,
                )
        for base in snapshot.kakan_candidates:
            if not continuation[base]:
                raise ValueError("kakan continuation omits a candidate tile")
            for added in _unique_choices(snapshot, base, 1):
                _append_consuming_action(
                    bindings,
                    kind=ActionKind.KAKAN,
                    tile=added[0],
                    called_tile=NO_TILE,
                    consumed=added,
                    target=RelativeTarget.SELF,
                    phase=DecisionPhase.KAN_SELECTION,
                    primary_action=MORTAL_KAN,
                    continuation_action=base,
                )

    if primary_mask[MORTAL_AGARI]:
        if snapshot.cans.can_tsumo_agari:
            bindings.append(
                CandidateBinding(
                    CanonicalAction(
                        kind=ActionKind.TSUMO,
                        tile=snapshot.last_draw,
                        relative_target=RelativeTarget.SELF,
                        from_draw=True,
                        decision_phase=phase,
                    ),
                    MORTAL_AGARI,
                )
            )
        elif snapshot.cans.can_ron_agari:
            bindings.append(
                CandidateBinding(
                    CanonicalAction(
                        kind=ActionKind.RON,
                        tile=called,
                        called_tile=called,
                        relative_target=target,
                        decision_phase=DecisionPhase.REACTION,
                    ),
                    MORTAL_AGARI,
                )
            )
        else:
            raise ValueError("agari mask has no corresponding can flag")

    if primary_mask[MORTAL_KYUUSHU]:
        if not snapshot.cans.can_ryukyoku:
            raise ValueError("kyuushu mask has no corresponding can flag")
        bindings.append(
            CandidateBinding(
                CanonicalAction(
                    kind=ActionKind.KYUUSHU_RYUKYOKU,
                    relative_target=RelativeTarget.SELF,
                    decision_phase=phase,
                ),
                MORTAL_KYUUSHU,
            )
        )

    if primary_mask[MORTAL_PASS]:
        if not snapshot.cans.can_pass:
            raise ValueError("pass mask has no corresponding reaction")
        bindings.append(
            CandidateBinding(
                CanonicalAction(
                    kind=ActionKind.PASS,
                    relative_target=RelativeTarget.NONE,
                    decision_phase=DecisionPhase.REACTION,
                ),
                MORTAL_PASS,
            )
        )

    assert_unique_candidates(binding.action for binding in bindings)
    represented_primary = {binding.primary_action for binding in bindings}
    legal_primary = set(map(int, np.flatnonzero(primary_mask)))
    if represented_primary != legal_primary:
        raise ValueError(
            f"atomic adapter primary mismatch: missing={legal_primary - represented_primary}, "
            f"extra={represented_primary - legal_primary}"
        )
    for primary_action, continuation in (
        (MORTAL_RIICHI, riichi_continuation_mask),
        (MORTAL_KAN, kan_continuation_mask),
    ):
        if continuation is None:
            continue
        legal_secondary = set(map(int, np.flatnonzero(continuation)))
        represented_secondary = {
            binding.continuation_action
            for binding in bindings
            if binding.primary_action == primary_action and binding.continuation_action >= 0
        }
        if represented_secondary != legal_secondary:
            raise ValueError(
                f"atomic adapter continuation mismatch for primary={primary_action}: "
                f"missing={legal_secondary - represented_secondary}, "
                f"extra={represented_secondary - legal_secondary}"
            )
    if not 1 <= len(bindings) <= 64:
        raise ValueError(f"candidate count {len(bindings)} is outside Kmax=64")
    return bindings


def _masked_softmax(q_values: np.ndarray, mask: np.ndarray, temperature: float) -> np.ndarray:
    q_values = np.asarray(q_values, dtype=np.float64)
    mask = np.asarray(mask)
    if q_values.shape != (MORTAL_ACTION_COUNT,) or mask.shape != q_values.shape:
        raise ValueError("Mortal Q and mask must have shape [46]")
    if mask.dtype != np.bool_ or not mask.any() or temperature <= 0:
        raise ValueError("invalid teacher mask or temperature")
    if not np.isfinite(q_values[mask]).all():
        raise FloatingPointError("legal teacher Q values must be finite")
    scaled = q_values[mask] / temperature
    scaled -= scaled.max()
    exp = np.exp(scaled)
    probabilities = np.zeros_like(q_values, dtype=np.float64)
    probabilities[mask] = exp / exp.sum()
    return probabilities


def map_mortal_teacher_to_atomic_classes(
    bindings: Sequence[CandidateBinding],
    primary_q: np.ndarray,
    primary_mask: np.ndarray,
    *,
    continuation_q: Mapping[int, np.ndarray] | None = None,
    continuation_mask: Mapping[int, np.ndarray] | None = None,
    temperature: float = 1.0,
) -> AtomicTeacherTarget:
    """Compose two-stage Mortal probabilities without inventing class-internal preferences."""

    continuation_q = continuation_q or {}
    continuation_mask = continuation_mask or {}
    primary_prob = _masked_softmax(primary_q, primary_mask, temperature)
    keys = tuple(dict.fromkeys(binding.teacher_class_key for binding in bindings))
    key_to_id = {key: index for index, key in enumerate(keys)}
    candidate_class_ids = np.asarray(
        [key_to_id[binding.teacher_class_key] for binding in bindings],
        dtype=np.uint8,
    )
    class_probabilities = np.zeros(len(keys), dtype=np.float64)
    cached_continuation: dict[int, np.ndarray] = {}
    for class_id, (primary_action, secondary_action) in enumerate(keys):
        probability = primary_prob[primary_action]
        if secondary_action >= 0:
            if primary_action not in continuation_q or primary_action not in continuation_mask:
                raise ValueError(f"missing continuation teacher output for action {primary_action}")
            branch_prob = cached_continuation.setdefault(
                primary_action,
                _masked_softmax(
                    continuation_q[primary_action],
                    continuation_mask[primary_action],
                    temperature,
                ),
            )
            probability *= branch_prob[secondary_action]
        class_probabilities[class_id] = probability

    if not np.isclose(class_probabilities.sum(), 1.0, atol=1e-7):
        raise ValueError(
            f"atomic teacher classes sum to {class_probabilities.sum():.12f}, expected 1"
        )
    hard_class_id = int(class_probabilities.argmax())
    return AtomicTeacherTarget(
        class_keys=keys,
        candidate_class_ids=candidate_class_ids,
        class_probabilities=class_probabilities.astype(np.float32),
        hard_class_id=hard_class_id,
    )
