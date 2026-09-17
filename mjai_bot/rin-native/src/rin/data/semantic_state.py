"""Privacy-safe semantic visible state reconstructed from public MJAI events."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from rin.data.public_state import tile_id


MAX_DISCARDS = 24
MAX_MELDS = 4
MAX_DORA_INDICATORS = 5
NO_TILE = -1


@dataclass(frozen=True, slots=True)
class SemanticVisibleSnapshot:
    hand_counts_red37: np.ndarray
    last_draw: np.int8
    dora_indicators: np.ndarray
    discards: np.ndarray
    discard_tsumogiri: np.ndarray
    discard_riichi: np.ndarray
    discard_called: np.ndarray
    meld_tiles: np.ndarray


@dataclass(frozen=True, slots=True)
class SemanticOracleSnapshot:
    """Exact four-hand state in acting-seat order for the independent critic."""

    hand_counts_red37: np.ndarray


def _remove_tiles(hand: np.ndarray, tiles: list[int]) -> None:
    for tile in tiles:
        if not 0 <= tile < 37 or hand[tile] <= 0:
            raise ValueError(f"visible hand does not contain tile {tile}")
        hand[tile] -= 1


@dataclass(slots=True)
class SemanticPublicState:
    """Track exactly the state available to one acting-seat observer."""

    seat: int
    hand_counts_red37: np.ndarray = field(
        default_factory=lambda: np.zeros(37, dtype=np.int8)
    )
    last_draw: int = NO_TILE
    dora_indicators: np.ndarray = field(
        default_factory=lambda: np.full(MAX_DORA_INDICATORS, NO_TILE, dtype=np.int8)
    )
    discards: np.ndarray = field(
        default_factory=lambda: np.full((4, MAX_DISCARDS), NO_TILE, dtype=np.int8)
    )
    discard_tsumogiri: np.ndarray = field(
        default_factory=lambda: np.zeros((4, MAX_DISCARDS), dtype=np.bool_)
    )
    discard_riichi: np.ndarray = field(
        default_factory=lambda: np.zeros((4, MAX_DISCARDS), dtype=np.bool_)
    )
    discard_called: np.ndarray = field(
        default_factory=lambda: np.zeros((4, MAX_DISCARDS), dtype=np.bool_)
    )
    discard_counts: np.ndarray = field(
        default_factory=lambda: np.zeros(4, dtype=np.int8)
    )
    meld_tiles: np.ndarray = field(
        default_factory=lambda: np.full((4, MAX_MELDS, 4), NO_TILE, dtype=np.int8)
    )
    meld_counts: np.ndarray = field(
        default_factory=lambda: np.zeros(4, dtype=np.int8)
    )
    pending_riichi: np.ndarray = field(
        default_factory=lambda: np.zeros(4, dtype=np.bool_)
    )

    def __post_init__(self) -> None:
        if self.seat not in range(4):
            raise ValueError("seat must be in [0, 3]")

    def _reset_kyoku(self, event: Mapping[str, Any]) -> None:
        tehais = event.get("tehais")
        if not isinstance(tehais, list) or len(tehais) != 4:
            raise ValueError("start_kyoku must contain four initial hands")
        own_hand = tehais[self.seat]
        if not isinstance(own_hand, list):
            raise ValueError("the acting seat's initial hand is malformed")
        self.hand_counts_red37.fill(0)
        for value in own_hand:
            encoded = tile_id(value)
            self.hand_counts_red37[encoded] += 1
        self.last_draw = NO_TILE
        self.dora_indicators.fill(NO_TILE)
        self.dora_indicators[0] = tile_id(event["dora_marker"])
        self.discards.fill(NO_TILE)
        self.discard_tsumogiri.fill(False)
        self.discard_riichi.fill(False)
        self.discard_called.fill(False)
        self.discard_counts.fill(0)
        self.meld_tiles.fill(NO_TILE)
        self.meld_counts.fill(0)
        self.pending_riichi.fill(False)

    def _append_discard(self, event: Mapping[str, Any]) -> None:
        actor = int(event["actor"])
        index = int(self.discard_counts[actor])
        if index >= MAX_DISCARDS:
            raise ValueError("discard count exceeds the semantic schema")
        encoded = tile_id(event["pai"])
        self.discards[actor, index] = encoded
        self.discard_tsumogiri[actor, index] = bool(event["tsumogiri"])
        self.discard_riichi[actor, index] = bool(self.pending_riichi[actor])
        self.pending_riichi[actor] = False
        self.discard_counts[actor] += 1
        if actor == self.seat:
            _remove_tiles(self.hand_counts_red37, [encoded])
            self.last_draw = NO_TILE

    def _mark_called_discard(self, target: int) -> None:
        index = int(self.discard_counts[target]) - 1
        if index < 0 or self.discards[target, index] < 0:
            raise ValueError("meld target has no preceding discard")
        self.discard_called[target, index] = True

    def _append_meld(self, actor: int, tiles: list[int]) -> None:
        index = int(self.meld_counts[actor])
        if index >= MAX_MELDS or not 1 <= len(tiles) <= 4:
            raise ValueError("meld exceeds the semantic schema")
        self.meld_tiles[actor, index].fill(NO_TILE)
        self.meld_tiles[actor, index, : len(tiles)] = tiles
        self.meld_counts[actor] += 1

    def _upgrade_kakan(self, actor: int, event: Mapping[str, Any]) -> None:
        added = tile_id(event["pai"])
        base_added = added if added < 34 else (4, 13, 22)[added - 34]
        for index in range(int(self.meld_counts[actor])):
            current = self.meld_tiles[actor, index]
            valid = current[current >= 0]
            bases = valid.copy()
            bases[valid == 34] = 4
            bases[valid == 35] = 13
            bases[valid == 36] = 22
            if valid.size == 3 and np.all(bases == base_added):
                empty = int(np.flatnonzero(current < 0)[0])
                current[empty] = added
                return
        raise ValueError("kakan has no matching visible pon")

    def update(self, event: Mapping[str, Any]) -> None:
        """Apply one event without reading an opponent's private draw or hand."""

        event_type = str(event.get("type", ""))
        if event_type == "start_kyoku":
            self._reset_kyoku(event)
            return
        if event_type == "tsumo":
            actor = int(event["actor"])
            if actor == self.seat:
                encoded = tile_id(event["pai"])
                self.hand_counts_red37[encoded] += 1
                self.last_draw = encoded
            return
        if event_type == "dahai":
            self._append_discard(event)
            return
        if event_type == "reach":
            self.pending_riichi[int(event["actor"])] = True
            return
        if event_type == "dora":
            empty = np.flatnonzero(self.dora_indicators < 0)
            if empty.size == 0:
                raise ValueError("dora indicator count exceeds the semantic schema")
            self.dora_indicators[int(empty[0])] = tile_id(event["dora_marker"])
            return
        if event_type in {"chi", "pon", "daiminkan"}:
            actor = int(event["actor"])
            target = int(event["target"])
            consumed = [tile_id(value) for value in event["consumed"]]
            called = tile_id(event["pai"])
            self._mark_called_discard(target)
            self._append_meld(actor, consumed + [called])
            if actor == self.seat:
                _remove_tiles(self.hand_counts_red37, consumed)
                self.last_draw = NO_TILE
            return
        if event_type == "ankan":
            actor = int(event["actor"])
            consumed = [tile_id(value) for value in event["consumed"]]
            self._append_meld(actor, consumed)
            if actor == self.seat:
                _remove_tiles(self.hand_counts_red37, consumed)
                self.last_draw = NO_TILE
            return
        if event_type == "kakan":
            actor = int(event["actor"])
            added = tile_id(event["pai"])
            self._upgrade_kakan(actor, event)
            if actor == self.seat:
                _remove_tiles(self.hand_counts_red37, [added])
                self.last_draw = NO_TILE
            return

    def snapshot(self) -> SemanticVisibleSnapshot:
        """Return a copy with player axes rotated to self/right/across/left."""

        order = np.asarray([(self.seat + offset) % 4 for offset in range(4)])
        return SemanticVisibleSnapshot(
            hand_counts_red37=self.hand_counts_red37.copy(),
            last_draw=np.int8(self.last_draw),
            dora_indicators=self.dora_indicators.copy(),
            discards=self.discards[order].copy(),
            discard_tsumogiri=self.discard_tsumogiri[order].copy(),
            discard_riichi=self.discard_riichi[order].copy(),
            discard_called=self.discard_called[order].copy(),
            meld_tiles=self.meld_tiles[order].copy(),
        )


@dataclass(slots=True)
class SemanticOracleState:
    """Track private hands from a complete referee MJAI log.

    This state must never be passed to the actor. It exists only to materialize
    the oracle critic input that MahJax can reproduce from its resident state.
    """

    hand_counts_red37: np.ndarray = field(
        default_factory=lambda: np.zeros((4, 37), dtype=np.int8)
    )

    def _remove(self, actor: int, tiles: list[int]) -> None:
        for tile in tiles:
            if not 0 <= actor < 4 or not 0 <= tile < 37:
                raise ValueError("oracle hand update is outside its valid range")
            if self.hand_counts_red37[actor, tile] <= 0:
                raise ValueError(f"oracle hand {actor} does not contain tile {tile}")
            self.hand_counts_red37[actor, tile] -= 1

    def update(self, event: Mapping[str, Any]) -> None:
        event_type = str(event.get("type", ""))
        if event_type == "start_kyoku":
            tehais = event.get("tehais")
            if not isinstance(tehais, list) or len(tehais) != 4:
                raise ValueError("oracle start_kyoku must contain four hands")
            self.hand_counts_red37.fill(0)
            for actor, hand in enumerate(tehais):
                if not isinstance(hand, list):
                    raise ValueError("oracle initial hand is malformed")
                for value in hand:
                    self.hand_counts_red37[actor, tile_id(value)] += 1
            return
        if event_type == "tsumo":
            actor = int(event["actor"])
            tile = tile_id(event["pai"])
            self.hand_counts_red37[actor, tile] += 1
            return
        if event_type == "dahai":
            self._remove(int(event["actor"]), [tile_id(event["pai"])])
            return
        if event_type in {"chi", "pon", "daiminkan", "ankan"}:
            self._remove(
                int(event["actor"]),
                [tile_id(value) for value in event["consumed"]],
            )
            return
        if event_type == "kakan":
            self._remove(int(event["actor"]), [tile_id(event["pai"])])

    def snapshot(self, seat: int) -> SemanticOracleSnapshot:
        if seat not in range(4):
            raise ValueError("seat must be in [0, 3]")
        order = np.asarray([(seat + offset) % 4 for offset in range(4)])
        return SemanticOracleSnapshot(
            hand_counts_red37=self.hand_counts_red37[order].copy()
        )
