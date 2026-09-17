from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence
import numpy as np
from rin.actions import ActionKind, CanonicalAction, DecisionPhase
from rin.data.atomic_adapter import MORTAL_ACTION_COUNT, MORTAL_AGARI, MORTAL_CHI_HIGH, MORTAL_CHI_LOW, MORTAL_CHI_MID, MORTAL_KAN, MORTAL_KYUUSHU, MORTAL_PASS, MORTAL_PON, MORTAL_RIICHI, CandidateBinding, MortalCans, PlayerSnapshot, deaka, enumerate_mortal_atomic_candidates, padded_consumed
from rin.data.public_state import GlobalState, PublicEvent, RuleProfile, public_event_from_mjai, tile_id
from rin.data.input_contract import LEGACY_SHANTEN_CONTRACT, native_public_shanten

def _mortal_cans(cans: Any) -> MortalCans:
    return MortalCans(can_discard=bool(cans.can_discard), can_chi_low=bool(cans.can_chi_low), can_chi_mid=bool(cans.can_chi_mid), can_chi_high=bool(cans.can_chi_high), can_pon=bool(cans.can_pon), can_daiminkan=bool(cans.can_daiminkan), can_kakan=bool(cans.can_kakan), can_ankan=bool(cans.can_ankan), can_riichi=bool(cans.can_riichi), can_tsumo_agari=bool(cans.can_tsumo_agari), can_ron_agari=bool(cans.can_ron_agari), can_ryukyoku=bool(cans.can_ryukyoku), target_actor=int(cans.target_actor))

def snapshot_player_state(state: Any, cans: Any) -> PlayerSnapshot:
    last_draw = state.last_self_tsumo()
    last_discard = state.last_kawa_tile()
    return PlayerSnapshot(seat=int(state.player_id), hand_counts=tuple(map(int, state.tehai)), reds_in_hand=tuple(map(bool, state.akas_in_hand)), last_draw=tile_id(last_draw) if last_draw is not None else -1, last_discard=tile_id(last_discard) if last_discard is not None else -1, ankan_candidates=tuple((deaka(tile_id(tile)) for tile in state.ankan_candidates())), kakan_candidates=tuple((deaka(tile_id(tile)) for tile in state.kakan_candidates())), cans=_mortal_cans(cans))
