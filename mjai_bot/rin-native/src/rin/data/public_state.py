"""Versioned visible history and global-state materialization for MJAI logs."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Mapping, Sequence

import numpy as np

from rin.actions import DecisionPhase, NO_TILE
from rin.data.atomic_adapter import deaka, padded_consumed


PUBLIC_EVENT_SCHEMA_VERSION = "rin.public-event.v1"
GLOBAL_FEATURE_SCHEMA_VERSION = "rin.global-features.v1"
GLOBAL_FEATURE_COUNT = 64


class PublicEventKind(IntEnum):
    START_GAME = 0
    START_KYOKU = 1
    TSUMO = 2
    DAHAI = 3
    CHI = 4
    PON = 5
    DAIMINKAN = 6
    KAKAN = 7
    ANKAN = 8
    DORA = 9
    REACH = 10
    REACH_ACCEPTED = 11
    HORA = 12
    RYUKYOKU = 13
    END_KYOKU = 14
    END_GAME = 15


_EVENT_KIND = {kind.name.lower(): kind for kind in PublicEventKind}
_HONOR_TILES = {"E": 27, "S": 28, "W": 29, "N": 30, "P": 31, "F": 32, "C": 33}


def tile_id(tile: str) -> int:
    """Map an MJAI tile string to the stable red-aware 0..36 encoding."""

    if tile in _HONOR_TILES:
        return _HONOR_TILES[tile]
    if not isinstance(tile, str) or len(tile) not in (2, 3):
        raise ValueError(f"invalid MJAI tile: {tile!r}")
    rank = tile[0]
    suit = tile[1]
    if rank not in "123456789" or suit not in "mps":
        raise ValueError(f"invalid MJAI tile: {tile!r}")
    if len(tile) == 3:
        if tile[2] != "r" or rank != "5":
            raise ValueError(f"invalid red MJAI tile: {tile!r}")
        return {"m": 34, "p": 35, "s": 36}[suit]
    return "mps".index(suit) * 9 + int(rank) - 1


def _relative(seat: int, actor: int | None) -> int:
    if actor is None:
        return -1
    if actor not in range(4):
        raise ValueError(f"actor must be in [0, 3], got {actor}")
    return int((actor - seat) % 4)


@dataclass(frozen=True, slots=True)
class PublicEvent:
    kind: PublicEventKind
    actor: int = -1
    target: int = -1
    tile: int = NO_TILE
    consumed_tiles: tuple[int, int, int, int] = (NO_TILE,) * 4
    tsumogiri: int = -1
    meld_orientation: int = -1
    round_position: int = 0

    def __post_init__(self) -> None:
        if self.actor not in range(-1, 4) or self.target not in range(-1, 4):
            raise ValueError("relative actor/target must be -1 or in [0, 3]")
        if self.tile not in range(-1, 37):
            raise ValueError("tile must be -1 or in [0, 36]")
        if len(self.consumed_tiles) != 4:
            raise ValueError("consumed_tiles must have four slots")
        if self.tsumogiri not in (-1, 0, 1):
            raise ValueError("tsumogiri must be -1, 0, or 1")
        if self.meld_orientation not in range(-1, 4):
            raise ValueError("meld_orientation must be -1 or in [0, 3]")
        if not 0 <= self.round_position <= 257:
            raise ValueError("round_position must fit the history embedding range")

    def as_model_fields(self) -> dict[str, int | tuple[int, int, int, int]]:
        return {
            "event_type": int(self.kind),
            "actor": self.actor,
            "target": self.target,
            "tile": self.tile,
            "consumed_tiles": self.consumed_tiles,
            "tsumogiri": self.tsumogiri,
            "meld_orientation": self.meld_orientation,
            "round_position": self.round_position,
        }


def public_event_from_mjai(
    event: Mapping[str, Any],
    *,
    seat: int,
    round_position: int,
) -> PublicEvent:
    """Convert one MJAI event without exposing an opponent's private draw."""

    if seat not in range(4):
        raise ValueError("seat must be in [0, 3]")
    event_type = str(event.get("type", ""))
    try:
        kind = _EVENT_KIND[event_type]
    except KeyError as exc:
        raise ValueError(f"unsupported public MJAI event type: {event_type!r}") from exc

    actor_abs = event.get("actor")
    target_abs = event.get("target")
    actor = _relative(seat, actor_abs)
    target = _relative(seat, target_abs)

    encoded_tile = NO_TILE
    if event_type == "tsumo":
        if actor_abs == seat:
            encoded_tile = tile_id(event["pai"])
    elif event_type in {"dahai", "chi", "pon", "daiminkan", "kakan"}:
        encoded_tile = tile_id(event["pai"])
    elif event_type in {"start_kyoku", "dora"}:
        encoded_tile = tile_id(event["dora_marker"])
    elif event_type == "ankan":
        encoded_tile = deaka(tile_id(event["consumed"][0]))

    consumed = padded_consumed(tile_id(tile) for tile in event.get("consumed", ()))
    is_discard = event_type == "dahai"
    is_called_meld = event_type in {"chi", "pon", "daiminkan"}
    return PublicEvent(
        kind=kind,
        actor=actor,
        target=target,
        tile=encoded_tile,
        consumed_tiles=consumed,
        tsumogiri=int(bool(event["tsumogiri"])) if is_discard else -1,
        meld_orientation=target if is_called_meld else -1,
        round_position=min(round_position, 257),
    )


def materialize_history(
    events: Sequence[PublicEvent],
    *,
    max_history: int = 200,
) -> dict[str, np.ndarray]:
    """Materialize a prefix into model fields, retaining its most recent events."""

    if max_history < 1:
        raise ValueError("max_history must be positive")
    kept = events[-max_history:]
    fields: dict[str, list[Any]] = {
        "event_type": [],
        "actor": [],
        "target": [],
        "tile": [],
        "consumed_tiles": [],
        "tsumogiri": [],
        "meld_orientation": [],
        "round_position": [],
    }
    for event in kept:
        for name, value in event.as_model_fields().items():
            fields[name].append(value)
    result = {
        name: np.asarray(values, dtype=np.int8 if name != "round_position" else np.int16)
        for name, values in fields.items()
    }
    result["recency"] = np.arange(len(kept) - 1, -1, -1, dtype=np.int16)
    return result


@dataclass(frozen=True, slots=True)
class RuleProfile:
    aka: bool = True
    kuitan: bool = True
    hanchan: bool = True
    sanma: bool = False


@dataclass(slots=True)
class GlobalState:
    scores: list[int] = field(default_factory=lambda: [25_000] * 4)
    bakaze: int = 27
    kyoku: int = 1
    honba: int = 0
    kyotaku: int = 0
    oya: int = 0
    tiles_left: int = 70
    dora_markers: list[int] = field(default_factory=list)
    riichi_declared: list[bool] = field(default_factory=lambda: [False] * 4)
    riichi_accepted: list[bool] = field(default_factory=lambda: [False] * 4)
    meld_counts: list[int] = field(default_factory=lambda: [0] * 4)
    turn_count: int = 0
    in_kyoku: bool = False

    def update(self, event: Mapping[str, Any]) -> None:
        event_type = str(event.get("type", ""))
        if event_type == "start_kyoku":
            scores = [int(score) for score in event["scores"]]
            if len(scores) != 4:
                raise ValueError("start_kyoku scores must contain four values")
            self.scores = scores
            self.bakaze = deaka(tile_id(event["bakaze"]))
            self.kyoku = int(event["kyoku"])
            self.honba = int(event["honba"])
            self.kyotaku = int(event["kyotaku"])
            self.oya = int(event["oya"])
            self.tiles_left = 70
            self.dora_markers = [tile_id(event["dora_marker"])]
            self.riichi_declared = [False] * 4
            self.riichi_accepted = [False] * 4
            self.meld_counts = [0] * 4
            self.turn_count = 0
            self.in_kyoku = True
        elif event_type == "tsumo":
            self.tiles_left = max(0, self.tiles_left - 1)
            self.turn_count += 1
        elif event_type in {"chi", "pon", "daiminkan", "ankan"}:
            self.meld_counts[int(event["actor"])] += 1
        elif event_type == "reach":
            self.riichi_declared[int(event["actor"])] = True
        elif event_type == "reach_accepted":
            actor = int(event["actor"])
            if not self.riichi_accepted[actor]:
                self.scores[actor] -= 1_000
                self.kyotaku += 1
                self.riichi_accepted[actor] = True
        elif event_type == "dora":
            self.dora_markers.append(tile_id(event["dora_marker"]))
        elif event_type in {"hora", "ryukyoku"} and event.get("deltas") is not None:
            deltas = [int(delta) for delta in event["deltas"]]
            if len(deltas) != 4:
                raise ValueError("terminal deltas must contain four values")
            self.scores = [score + delta for score, delta in zip(self.scores, deltas, strict=True)]
        elif event_type == "end_kyoku":
            self.in_kyoku = False

        if any(not -200_000 <= score <= 500_000 for score in self.scores):
            raise ValueError(f"implausible score state: {self.scores}")
        if not 0 <= self.tiles_left <= 70:
            raise ValueError(f"invalid remaining tile count: {self.tiles_left}")

    def vector_for(
        self,
        seat: int,
        *,
        shanten: int,
        furiten: bool,
        decision_phase: DecisionPhase,
        rule_profile: RuleProfile = RuleProfile(),
    ) -> np.ndarray:
        """Return the fixed global-features-v1 normalized 64-vector."""

        if seat not in range(4):
            raise ValueError("seat must be in [0, 3]")
        if not -1 <= shanten <= 6:
            raise ValueError(f"shanten must be in [-1, 6], got {shanten}")
        relative_players = [(seat + offset) % 4 for offset in range(4)]
        relative_scores = np.asarray([self.scores[player] for player in relative_players])
        order = sorted(range(4), key=lambda player: (-self.scores[player], player))
        ranks = {player: rank for rank, player in enumerate(order)}
        vector = np.zeros(GLOBAL_FEATURE_COUNT, dtype=np.float32)
        vector[0:4] = relative_scores / 50_000.0
        vector[4:8] = [ranks[player] / 3.0 for player in relative_players]
        vector[8:12] = (relative_scores - self.scores[seat]) / 50_000.0
        vector[12 + ((self.oya - seat) % 4)] = 1.0
        if self.bakaze not in range(27, 31):
            raise ValueError(f"invalid round wind tile id: {self.bakaze}")
        vector[16 + self.bakaze - 27] = 1.0
        if self.kyoku not in range(1, 5):
            raise ValueError(f"kyoku must be in [1, 4], got {self.kyoku}")
        vector[20 + self.kyoku - 1] = 1.0
        vector[24] = self.honba / 10.0
        vector[25] = self.kyotaku / 10.0
        vector[26] = self.tiles_left / 70.0
        for offset, marker in enumerate(self.dora_markers[:5]):
            vector[27 + offset] = (marker + 1) / 37.0
        vector[32:36] = [float(self.riichi_declared[player]) for player in relative_players]
        vector[36:40] = [self.meld_counts[player] / 4.0 for player in relative_players]
        vector[40] = (shanten + 1) / 7.0
        vector[41] = float(furiten)
        vector[42 + int(decision_phase)] = 1.0
        vector[46 + ((seat - self.oya) % 4)] = 1.0
        vector[50:54] = [
            float(rule_profile.aka),
            float(rule_profile.kuitan),
            float(rule_profile.hanchan),
            float(rule_profile.sanma),
        ]
        vector[54] = min(self.turn_count, 70) / 70.0
        vector[55] = float(seat == self.oya)
        opponents = relative_players[1:]
        vector[56] = float(any(self.riichi_declared[player] for player in opponents))
        vector[57] = sum(self.riichi_declared) / 4.0
        vector[58] = sum(self.meld_counts) / 16.0
        vector[59] = len(self.dora_markers) / 5.0
        vector[60] = float(self.in_kyoku)
        vector[61] = self.kyoku / 4.0
        vector[62] = (self.bakaze - 27) / 3.0
        vector[63] = 1.0
        if not np.isfinite(vector).all():
            raise FloatingPointError("global feature vector contains non-finite values")
        return vector
