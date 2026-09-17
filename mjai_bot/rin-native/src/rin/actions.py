"""Canonical, protocol-independent atomic action contract for RIN."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Final, Iterable


MIN_TILE: Final = 0
MAX_TILE: Final = 36
NO_TILE: Final = -1
CONSUMED_TILE_SLOTS: Final = 4
MAX_RULE_FLAGS: Final = (1 << 16) - 1
RED_TILES: Final = frozenset({34, 35, 36})


class ActionKind(IntEnum):
    DISCARD = 0
    RIICHI_DISCARD = 1
    CHI = 2
    PON = 3
    DAIMINKAN = 4
    ANKAN = 5
    KAKAN = 6
    TSUMO = 7
    RON = 8
    PASS = 9
    KYUUSHU_RYUKYOKU = 10


class RelativeTarget(IntEnum):
    SELF = 0
    RIGHT = 1
    ACROSS = 2
    LEFT = 3
    NONE = 4


class DecisionPhase(IntEnum):
    AFTER_DRAW = 0
    REACTION = 1
    KAN_SELECTION = 2
    TERMINAL = 3


_TILE_REQUIRED = frozenset(
    {
        ActionKind.DISCARD,
        ActionKind.RIICHI_DISCARD,
        ActionKind.CHI,
        ActionKind.PON,
        ActionKind.DAIMINKAN,
        ActionKind.ANKAN,
        ActionKind.KAKAN,
        ActionKind.TSUMO,
        ActionKind.RON,
    }
)

_CALLED_TILE_REQUIRED = frozenset(
    {
        ActionKind.CHI,
        ActionKind.PON,
        ActionKind.DAIMINKAN,
        ActionKind.RON,
    }
)

_REACTION_KINDS = frozenset(
    {
        ActionKind.CHI,
        ActionKind.PON,
        ActionKind.DAIMINKAN,
        ActionKind.RON,
        ActionKind.PASS,
    }
)


def _validate_tile(value: int, *, allow_none: bool, field: str) -> None:
    lower = NO_TILE if allow_none else MIN_TILE
    if not lower <= value <= MAX_TILE:
        raise ValueError(f"{field} must be in [{lower}, {MAX_TILE}], got {value}")


@dataclass(frozen=True, slots=True)
class CanonicalAction:
    """One legal atomic action presented to the RIN policy.

    The descriptor is deliberately independent of Mortal, MJAI, and MahJax
    action indices. Protocol adapters may attach provenance separately, but
    two descriptors compare equal only when every semantic field agrees.
    """

    kind: ActionKind
    tile: int = NO_TILE
    called_tile: int = NO_TILE
    consumed_tiles: tuple[int, int, int, int] = (NO_TILE,) * CONSUMED_TILE_SLOTS
    relative_target: RelativeTarget = RelativeTarget.NONE
    from_draw: bool = False
    red_tile_mask: int = 0
    decision_phase: DecisionPhase = DecisionPhase.TERMINAL
    rule_flags: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ActionKind):
            raise TypeError("kind must be an ActionKind")
        if not isinstance(self.relative_target, RelativeTarget):
            raise TypeError("relative_target must be a RelativeTarget")
        if not isinstance(self.decision_phase, DecisionPhase):
            raise TypeError("decision_phase must be a DecisionPhase")
        if not isinstance(self.from_draw, bool):
            raise TypeError("from_draw must be bool")
        if len(self.consumed_tiles) != CONSUMED_TILE_SLOTS:
            raise ValueError(f"consumed_tiles must have {CONSUMED_TILE_SLOTS} slots")

        _validate_tile(self.tile, allow_none=True, field="tile")
        _validate_tile(self.called_tile, allow_none=True, field="called_tile")
        for index, tile in enumerate(self.consumed_tiles):
            _validate_tile(tile, allow_none=True, field=f"consumed_tiles[{index}]")

        if not 0 <= self.red_tile_mask < (1 << CONSUMED_TILE_SLOTS):
            raise ValueError("red_tile_mask may only use the four consumed-tile bits")
        if not 0 <= self.rule_flags <= MAX_RULE_FLAGS:
            raise ValueError(f"rule_flags must fit uint16, got {self.rule_flags}")

        occupied = [tile != NO_TILE for tile in self.consumed_tiles]
        if occupied != sorted(occupied, reverse=True):
            raise ValueError("consumed_tiles padding must be trailing")
        for index, tile in enumerate(self.consumed_tiles):
            mask_is_red = bool(self.red_tile_mask & (1 << index))
            if mask_is_red and tile not in RED_TILES:
                raise ValueError("red_tile_mask must point to a red tile id")
            if tile in RED_TILES and not mask_is_red:
                raise ValueError("red consumed tiles must be represented in red_tile_mask")

        if self.kind in _TILE_REQUIRED and self.tile == NO_TILE:
            raise ValueError(f"{self.kind.name} requires tile")
        if self.kind not in _TILE_REQUIRED and self.tile != NO_TILE:
            raise ValueError(f"{self.kind.name} must not carry tile")
        if self.kind in _CALLED_TILE_REQUIRED and self.called_tile == NO_TILE:
            raise ValueError(f"{self.kind.name} requires called_tile")
        if self.kind not in _CALLED_TILE_REQUIRED and self.called_tile != NO_TILE:
            raise ValueError(f"{self.kind.name} must not carry called_tile")

        consumed_count = sum(occupied)
        required_consumed = {
            ActionKind.CHI: 2,
            ActionKind.PON: 2,
            ActionKind.DAIMINKAN: 3,
            ActionKind.ANKAN: 4,
            ActionKind.KAKAN: 1,
        }.get(self.kind, 0)
        if consumed_count != required_consumed:
            raise ValueError(
                f"{self.kind.name} requires {required_consumed} consumed tiles, got {consumed_count}"
            )

        if self.kind in _REACTION_KINDS:
            if self.decision_phase != DecisionPhase.REACTION:
                raise ValueError(f"{self.kind.name} must use REACTION phase")
        elif self.kind in {ActionKind.ANKAN, ActionKind.KAKAN}:
            if self.decision_phase != DecisionPhase.KAN_SELECTION:
                raise ValueError(f"{self.kind.name} must use KAN_SELECTION phase")

        if self.kind in {ActionKind.CHI, ActionKind.PON, ActionKind.DAIMINKAN, ActionKind.RON}:
            if self.relative_target in {RelativeTarget.SELF, RelativeTarget.NONE}:
                raise ValueError(f"{self.kind.name} requires an opponent target")
        elif self.kind in {ActionKind.ANKAN, ActionKind.KAKAN, ActionKind.TSUMO, ActionKind.KYUUSHU_RYUKYOKU}:
            if self.relative_target != RelativeTarget.SELF:
                raise ValueError(f"{self.kind.name} requires SELF target")
        elif self.kind in {ActionKind.DISCARD, ActionKind.RIICHI_DISCARD, ActionKind.PASS}:
            if self.relative_target != RelativeTarget.NONE:
                raise ValueError(f"{self.kind.name} requires NONE target")

    @property
    def protocol_equivalence_key(self) -> tuple[int, ...]:
        """Return the key used when a teacher cannot distinguish red usage.

        Red tile ids are normalized to their 34-type black-five ids, while
        the fully atomic dataclass equality remains red-aware.
        """

        def normalize(tile: int) -> int:
            return {34: 4, 35: 13, 36: 22}.get(tile, tile)

        return (
            int(self.kind),
            normalize(self.tile),
            normalize(self.called_tile),
            *(normalize(tile) for tile in self.consumed_tiles),
            int(self.relative_target),
            int(self.from_draw),
            int(self.decision_phase),
            self.rule_flags,
        )

    def as_int_tuple(self) -> tuple[int, ...]:
        """Serialize to the stable CanonicalAction v1 field order."""

        return (
            int(self.kind),
            self.tile,
            self.called_tile,
            *self.consumed_tiles,
            int(self.relative_target),
            int(self.from_draw),
            self.red_tile_mask,
            int(self.decision_phase),
            self.rule_flags,
        )

    @classmethod
    def from_int_tuple(cls, fields: Iterable[int]) -> "CanonicalAction":
        values = tuple(int(value) for value in fields)
        if len(values) != 12:
            raise ValueError(f"CanonicalAction v1 requires 12 fields, got {len(values)}")
        if values[8] not in (0, 1):
            raise ValueError(f"from_draw must serialize as 0 or 1, got {values[8]}")
        return cls(
            kind=ActionKind(values[0]),
            tile=values[1],
            called_tile=values[2],
            consumed_tiles=(values[3], values[4], values[5], values[6]),
            relative_target=RelativeTarget(values[7]),
            from_draw=bool(values[8]),
            red_tile_mask=values[9],
            decision_phase=DecisionPhase(values[10]),
            rule_flags=values[11],
        )


def assert_unique_candidates(candidates: Iterable[CanonicalAction]) -> None:
    """Reject duplicate semantic actions before padding a candidate batch."""

    seen: set[CanonicalAction] = set()
    for candidate in candidates:
        if candidate in seen:
            raise ValueError(f"duplicate canonical candidate: {candidate}")
        seen.add(candidate)
