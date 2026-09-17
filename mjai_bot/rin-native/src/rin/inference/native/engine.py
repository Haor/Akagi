"""Libriichi ``mjai-log`` arena engine for a distilled RIN actor."""
from __future__ import annotations
from dataclasses import dataclass, field, fields
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
from rin.actions import ActionKind, CanonicalAction, NO_TILE
from rin.data.atomic_adapter import MORTAL_ACTION_COUNT, MORTAL_KAN, MORTAL_RIICHI, CandidateBinding, deaka, enumerate_mortal_atomic_candidates
from rin.data.libriichi_replay import snapshot_player_state
from rin.data.input_contract import native_public_shanten, PUBLIC_SHANTEN_CONTRACT
from rin.data.semantic_state import SemanticPublicState, SemanticVisibleSnapshot
from rin.data.public_state import GlobalState, PublicEvent, RuleProfile, materialize_history, public_event_from_mjai, tile_id
from rin.policy import greedy_protocol_candidate
COMPLETE_ROUND_HISTORY = 'rin.public-history.complete-round.v1'
LEGACY_PREFIX_HISTORY = 'rin.public-history.actor-prefix.v1'
HISTORY_FIELDS = ('event_type', 'actor', 'target', 'tile', 'consumed_tiles', 'tsumogiri', 'meld_orientation', 'round_position', 'recency')
CANDIDATE_FIELDS = ('kind', 'tile', 'called_tile', 'consumed_tiles', 'relative_target', 'from_draw', 'red_tile_mask', 'decision_phase', 'rule_flags')

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        while (chunk := source.read(8 * 1024 * 1024)):
            digest.update(chunk)
    return digest.hexdigest()

def tile_to_mjai(tile: int) -> str:
    """Convert the stable red-aware 0..36 tile id back to MJAI text."""
    if tile in (34, 35, 36):
        return f"5{'mps'[tile - 34]}r"
    if 0 <= tile < 27:
        return f"{tile % 9 + 1}{'mps'[tile // 9]}"
    if 27 <= tile < 34:
        return ('E', 'S', 'W', 'N', 'P', 'F', 'C')[tile - 27]
    raise ValueError(f'tile id must be in [0, 36], got {tile}')

def _active_pon_consumed(events: Sequence[Mapping[str, Any]], *, seat: int, base_tile: int) -> list[str]:
    active: list[tuple[int, list[str]]] = []
    for event in events:
        event_type = event.get('type')
        if event_type in {'start_kyoku', 'end_kyoku'}:
            active.clear()
        elif event_type == 'pon' and int(event.get('actor', -1)) == seat:
            called = str(event['pai'])
            consumed = [str(tile) for tile in event['consumed']]
            active.append((deaka(tile_id(called)), [called, *consumed]))
        elif event_type == 'kakan' and int(event.get('actor', -1)) == seat:
            upgraded = deaka(tile_id(str(event['pai'])))
            for index in range(len(active) - 1, -1, -1):
                if active[index][0] == upgraded:
                    active.pop(index)
                    break
    matches = [consumed for base, consumed in active if base == base_tile]
    if len(matches) != 1:
        raise ValueError(f'kakan base tile {base_tile} matched {len(matches)} active pon events')
    return matches[0]

def canonical_action_to_mjai(action: CanonicalAction, *, seat: int, events: Sequence[Mapping[str, Any]]=()) -> dict[str, Any]:
    """Serialize one canonical action to an arena-valid MJAI reaction."""
    kind = action.kind
    if kind in (ActionKind.DISCARD, ActionKind.RIICHI_DISCARD):
        return {'type': 'dahai', 'actor': seat, 'pai': tile_to_mjai(action.tile), 'tsumogiri': bool(action.from_draw)}
    if kind in (ActionKind.CHI, ActionKind.PON, ActionKind.DAIMINKAN):
        return {'type': kind.name.lower(), 'actor': seat, 'target': (seat + int(action.relative_target)) % 4, 'pai': tile_to_mjai(action.called_tile), 'consumed': [tile_to_mjai(tile) for tile in action.consumed_tiles if tile != NO_TILE]}
    if kind == ActionKind.ANKAN:
        return {'type': 'ankan', 'actor': seat, 'consumed': [tile_to_mjai(tile) for tile in action.consumed_tiles if tile != NO_TILE]}
    if kind == ActionKind.KAKAN:
        return {'type': 'kakan', 'actor': seat, 'pai': tile_to_mjai(action.tile), 'consumed': _active_pon_consumed(events, seat=seat, base_tile=deaka(action.tile))}
    if kind in (ActionKind.TSUMO, ActionKind.RON):
        target = seat if kind == ActionKind.TSUMO else (seat + int(action.relative_target)) % 4
        return {'type': 'hora', 'actor': seat, 'target': target}
    if kind == ActionKind.PASS:
        return {'type': 'none'}
    if kind == ActionKind.KYUUSHU_RYUKYOKU:
        return {'type': 'ryukyoku'}
    raise ValueError(f'unsupported canonical action kind: {kind}')

@dataclass(slots=True)
class PublicGameMemory:
    """Public-only cross-kyoku state for one arena game."""
    seat: int
    rule_profile: RuleProfile = RuleProfile()
    public_events: list[PublicEvent] = field(default_factory=list)
    full_events: list[dict[str, Any]] = field(default_factory=list)
    current_kyoku_events: list[dict[str, Any]] = field(default_factory=list)
    global_state: GlobalState = field(default_factory=GlobalState)
    kyoku_position: int = 0
    pending_riichi: CanonicalAction | None = None

    def __post_init__(self) -> None:
        if self.seat not in range(4):
            raise ValueError('seat must be in [0, 3]')
        self._append_event({'type': 'start_game', 'names': ['RIN'] * 4})

    def _append_event(self, event: Mapping[str, Any]) -> None:
        copied = dict(event)
        event_type = str(copied.get('type', ''))
        if event_type == 'start_kyoku':
            self.kyoku_position = 0
        elif self.global_state.in_kyoku:
            self.kyoku_position += 1
        self.global_state.update(copied)
        self.public_events.append(public_event_from_mjai(copied, seat=self.seat, round_position=self.kyoku_position))
        self.full_events.append(copied)

    def close_kyoku(self) -> None:
        if self.global_state.in_kyoku:
            self._append_event({'type': 'end_kyoku'})
        self.current_kyoku_events.clear()
        self.pending_riichi = None

    def close_game(self) -> None:
        self.close_kyoku()
        self._append_event({'type': 'end_game'})

    def sync(self, events_json: str) -> tuple[dict[str, Any], ...]:
        parsed = json.loads(events_json)
        if not isinstance(parsed, list) or not parsed:
            raise ValueError('arena events_json must contain a nonempty event list')
        events = [dict(event) for event in parsed]
        if events[0].get('type') != 'start_kyoku':
            raise ValueError('arena event list must begin with start_kyoku')
        if not self.current_kyoku_events:
            suffix = events
        elif len(events) >= len(self.current_kyoku_events) and events[:len(self.current_kyoku_events)] == self.current_kyoku_events:
            suffix = events[len(self.current_kyoku_events):]
        elif events[0] != self.current_kyoku_events[0]:
            self.close_kyoku()
            suffix = events
        else:
            raise ValueError('arena event list is not an append-only kyoku prefix')
        for event in suffix:
            self._append_event(event)
        self.current_kyoku_events = events
        return tuple(suffix)

def pad_history(events: Sequence[PublicEvent], *, max_history: int) -> dict[str, np.ndarray]:
    materialized = materialize_history(events, max_history=max_history)
    length = int(materialized['event_type'].shape[0])
    result: dict[str, np.ndarray] = {}
    for name, values in materialized.items():
        output = np.full((max_history,) + values.shape[1:], -1, dtype=values.dtype)
        output[:length] = values
        result[name] = output
    mask = np.zeros(max_history, dtype=np.bool_)
    mask[:length] = True
    result['mask'] = mask
    return result

def pad_candidates(bindings: Sequence[CandidateBinding], *, max_candidates: int) -> dict[str, np.ndarray]:
    if not bindings or len(bindings) > max_candidates:
        raise ValueError(f'candidate count must be in [1, {max_candidates}], got {len(bindings)}')
    result = {'kind': np.full(max_candidates, -1, dtype=np.int8), 'tile': np.full(max_candidates, -1, dtype=np.int8), 'called_tile': np.full(max_candidates, -1, dtype=np.int8), 'consumed_tiles': np.full((max_candidates, 4), -1, dtype=np.int8), 'relative_target': np.full(max_candidates, -1, dtype=np.int8), 'from_draw': np.zeros(max_candidates, dtype=np.uint8), 'red_tile_mask': np.zeros(max_candidates, dtype=np.uint8), 'decision_phase': np.full(max_candidates, -1, dtype=np.int8), 'rule_flags': np.zeros(max_candidates, dtype=np.uint16), 'mask': np.zeros(max_candidates, dtype=np.bool_)}
    for index, binding in enumerate(bindings):
        fields = binding.action.as_int_tuple()
        result['kind'][index] = fields[0]
        result['tile'][index] = fields[1]
        result['called_tile'][index] = fields[2]
        result['consumed_tiles'][index] = fields[3:7]
        result['relative_target'][index] = fields[7]
        result['from_draw'][index] = fields[8]
        result['red_tile_mask'][index] = fields[9]
        result['decision_phase'][index] = fields[10]
        result['rule_flags'][index] = fields[11]
    result['mask'][:len(bindings)] = True
    return result

@dataclass(frozen=True, slots=True)
class PreparedDecision:
    visible_planes: np.ndarray
    history: Mapping[str, np.ndarray]
    history_mask: np.ndarray
    global_features: np.ndarray
    candidates: Mapping[str, np.ndarray]
    candidate_mask: np.ndarray
    bindings: tuple[CandidateBinding, ...]

@dataclass(frozen=True, slots=True)
class PreparedSemanticDecision:
    semantic: SemanticVisibleSnapshot
    history: Mapping[str, np.ndarray]
    history_mask: np.ndarray
    global_features: np.ndarray
    candidates: Mapping[str, np.ndarray]
    candidate_mask: np.ndarray
    bindings: tuple[CandidateBinding, ...]

def stack_prepared_decisions(decisions: Sequence[PreparedDecision], *, fixed_batch_size: int) -> dict[str, Any]:
    if not decisions or len(decisions) > fixed_batch_size:
        raise ValueError('decision batch is empty or exceeds fixed_batch_size')
    padded = list(decisions) + [decisions[0]] * (fixed_batch_size - len(decisions))
    return {'visible_planes': np.stack([row.visible_planes for row in padded]), 'history': {name: np.stack([row.history[name] for row in padded]) for name in HISTORY_FIELDS}, 'history_mask': np.stack([row.history_mask for row in padded]), 'global_features': np.stack([row.global_features for row in padded]), 'candidates': {name: np.stack([row.candidates[name] for row in padded]) for name in CANDIDATE_FIELDS}, 'candidate_mask': np.stack([row.candidate_mask for row in padded])}

def stack_prepared_semantic_decisions(decisions: Sequence[PreparedSemanticDecision], *, fixed_batch_size: int) -> dict[str, Any]:
    if not decisions or len(decisions) > fixed_batch_size:
        raise ValueError('decision batch is empty or exceeds fixed_batch_size')
    padded = list(decisions) + [decisions[0]] * (fixed_batch_size - len(decisions))
    semantic_fields = ('hand_counts_red37', 'last_draw', 'dora_indicators', 'discards', 'discard_tsumogiri', 'discard_riichi', 'discard_called', 'meld_tiles')
    return {'semantic': {name: np.stack([getattr(row.semantic, name) for row in padded]) for name in semantic_fields}, 'history': {name: np.stack([row.history[name] for row in padded]) for name in HISTORY_FIELDS}, 'history_mask': np.stack([row.history_mask for row in padded]), 'global_features': np.stack([row.global_features for row in padded]), 'candidates': {name: np.stack([row.candidates[name] for row in padded]) for name in CANDIDATE_FIELDS}, 'candidate_mask': np.stack([row.candidate_mask for row in padded])}

def _load_independent_human_actor_run(run: Path, config_record: Mapping[str, Any]) -> tuple[SemanticRINActor, Mapping[str, Any], RINConfig, dict[str, Any]]:
    """Load a self-contained human actor with an explicit export identity."""
    schema = 'rin.independent-human-actor.v1'
    if config_record.get('anchor_run') is not None:
        raise ValueError('independent human actors cannot reference an anchor_run')
    model = config_record.get('model')
    model_fields = {item.name for item in fields(RINConfig)}
    if not isinstance(model, dict) or set(model) != model_fields:
        raise ValueError('independent human actors require the complete RINConfig model')
    config = RINConfig(**model)
    actor_options = config_record.get('actor')
    boolean_flags = {'ev_auxiliary', 'same_shanten_required_tiles', 'best_current_required_tiles', 'oracle_ev_curve_input'}
    required_options = boolean_flags | {'dtype', 'action_head_variant'}
    if not isinstance(actor_options, dict) or set(actor_options) != required_options:
        raise ValueError('independent human actors require all explicit actor options')
    if any((type(actor_options[name]) is not bool for name in boolean_flags)):
        raise ValueError('independent human actor flags must be booleans')
    dtype = actor_options['dtype']
    if dtype not in {'bfloat16', 'float32'}:
        raise ValueError(f'unsupported independent human actor dtype: {dtype}')
    if actor_options['oracle_ev_curve_input']:
        raise ValueError('oracle EV curve input cannot enter arena inference')
    if any((actor_options[name] for name in boolean_flags)):
        raise ValueError('independent human arena actors require all auxiliary flags false')
    if actor_options['action_head_variant'] != 'dynamic':
        raise ValueError('independent human arena actors require the dynamic action head')
    config_path = run / 'config.json'
    checkpoint_path = run / 'latest.msgpack'
    checkpoint_hash = sha256_file(checkpoint_path)
    expected_hash = config_record.get('checkpoint_sha256')
    if expected_hash is not None and expected_hash != checkpoint_hash:
        raise ValueError('independent human checkpoint SHA-256 does not match config')
    restored = serialization.msgpack_restore(checkpoint_path.read_bytes())
    if not isinstance(restored, dict):
        raise ValueError('independent human checkpoint must contain a parameter mapping')
    if 'params' in restored:
        if restored.get('schema_version') != schema:
            raise ValueError('independent human checkpoint requires its own schema marker')
        params = restored['params']
        bookkeeping = restored
    else:
        if expected_hash is None or 'schema_version' in restored:
            raise ValueError('bare independent human parameters require checkpoint_sha256')
        params = restored
        bookkeeping = {}
    if not isinstance(params, dict) or not params:
        raise ValueError('independent human checkpoint has no parameter tree')
    metadata = {'run': str(run.resolve()), 'schema_version': schema, 'architecture': 'mahjax-native-semantic', 'config_sha256': sha256_file(config_path), 'checkpoint_sha256': checkpoint_hash, 'completed_epoch': int(bookkeeping.get('completed_epoch', config_record.get('completed_epoch', 0))), 'global_step': int(bookkeeping.get('global_step', config_record.get('global_step', 0))), **actor_options}
    return (SemanticRINActor(config=config, **{**actor_options, 'dtype': getattr(jax.numpy, dtype)}), params, config, metadata)

class RINMjaiEngine:
    """Batched greedy RIN engine implementing Libriichi's mjai-log contract."""
    engine_type = 'mjai-log'

    def __init__(self, *, predictor, player_state_type, name='RIN Native', mortal_version=3, rule_profile=RuleProfile(), history_contract=COMPLETE_ROUND_HISTORY):
        from types import SimpleNamespace
        self.name = name
        self.predictor = predictor
        self.config = SimpleNamespace(**predictor.config['model'])
        self.player_state_type = player_state_type
        self.mortal_version = mortal_version
        self.rule_profile = rule_profile
        if history_contract != COMPLETE_ROUND_HISTORY:
            raise ValueError('Native deployment requires complete-round public history')
        self.history_contract = history_contract
        self.player_ids = ()
        self.memories = {}
        self.action_counts = Counter()
        self.decision_count = 0
        self.riichi_continuation_count = 0
        self.maximum_candidate_count = 0
        self.params = None
        self._apply = lambda unused, inputs: self.predictor.predict(inputs)

    def set_player_ids(self, player_ids: Sequence[int]) -> None:
        values = tuple((int(value) for value in player_ids))
        if not values or any((value not in range(4) for value in values)):
            raise ValueError('player_ids must be a nonempty sequence of seats')
        self.player_ids = values

    def start_game(self, game_idx: int) -> None:
        if not self.player_ids:
            raise RuntimeError('set_player_ids must run before start_game')
        self.memories[game_idx] = PublicGameMemory(seat=self.player_ids[game_idx], rule_profile=self.rule_profile)

    def end_kyoku(self, game_idx: int) -> None:
        if self.history_contract == COMPLETE_ROUND_HISTORY:
            raise RuntimeError('Complete public history requires Libriichi end_kyoku_with_log; load the qualified native extension before collecting or evaluating.')
        self.memories[game_idx].close_kyoku()

    def end_kyoku_with_log(self, game_idx: int, events_json: str) -> None:
        if self.history_contract == LEGACY_PREFIX_HISTORY:
            self.end_kyoku(game_idx)
            return
        events = json.loads(events_json)
        if not isinstance(events, list) or not events or events[0].get('type') != 'start_kyoku' or (events[-1].get('type') != 'end_kyoku'):
            raise ValueError('complete round callback must contain START_KYOKU through END_KYOKU')
        self._sync_memory(game_idx, events_json)
        self.memories[game_idx].close_kyoku()

    def end_game(self, game_idx: int, scores: Sequence[int]) -> None:
        del scores
        self.memories[game_idx].close_game()

    def _riichi_continuation_mask(self, memory: PublicGameMemory) -> np.ndarray:
        branch = self.player_state_type(memory.seat)
        for event in memory.full_events:
            branch.update(json.dumps(event, separators=(',', ':')))
        branch.update(json.dumps({'type': 'reach', 'actor': memory.seat}, separators=(',', ':')))
        _, mask = branch.encode_obs(self.mortal_version, False)
        mask = np.asarray(mask, dtype=np.bool_)
        if mask.shape != (MORTAL_ACTION_COUNT,) or mask[37:].any() or (not mask[:37].any()):
            raise ValueError('invalid post-riichi continuation mask')
        return mask

    def _prepare(self, game_idx: int, game_state: Any, memory: PublicGameMemory) -> PreparedDecision:
        del game_idx
        state = game_state.state
        observation, primary_mask = state.encode_obs(self.mortal_version, False)
        observation = np.asarray(observation, dtype=np.float16)
        primary_mask = np.asarray(primary_mask, dtype=np.bool_)
        if observation.shape != (self.config.visible_channels, self.config.tile_count):
            raise ValueError('arena visible observation shape differs from the actor contract')
        if primary_mask.shape != (MORTAL_ACTION_COUNT,):
            raise ValueError('arena Mortal mask shape differs from the action adapter contract')
        snapshot = snapshot_player_state(state, state.last_cans)
        riichi_mask = self._riichi_continuation_mask(memory) if primary_mask[MORTAL_RIICHI] else None
        kan_mask = None
        if primary_mask[MORTAL_KAN]:
            _, encoded = state.encode_obs(self.mortal_version, True)
            kan_mask = np.asarray(encoded, dtype=np.bool_)
        bindings = tuple(enumerate_mortal_atomic_candidates(snapshot, primary_mask, riichi_continuation_mask=riichi_mask, kan_continuation_mask=kan_mask))
        padded_history = pad_history(memory.public_events, max_history=self.config.max_history)
        padded_candidates = pad_candidates(bindings, max_candidates=self.config.max_candidates)
        phase = bindings[0].action.decision_phase
        global_features = memory.global_state.vector_for(memory.seat, shanten=int(state.shanten), furiten=bool(state.at_furiten), decision_phase=phase, rule_profile=self.rule_profile)
        return PreparedDecision(visible_planes=observation, history={name: padded_history[name] for name in HISTORY_FIELDS}, history_mask=padded_history['mask'], global_features=global_features, candidates={name: padded_candidates[name] for name in CANDIDATE_FIELDS}, candidate_mask=padded_candidates['mask'], bindings=bindings)

    def _sync_memory(self, game_idx: int, events_json: str) -> None:
        self.memories[game_idx].sync(events_json)

    def _stack_prepared(self, decisions: Sequence[Any]) -> dict[str, Any]:
        return stack_prepared_decisions(decisions, fixed_batch_size=len(self.player_ids))

    def _choose_candidate(self, *, game_idx: int, logits: np.ndarray, decision: PreparedDecision | PreparedSemanticDecision) -> int:
        """Select one legal atomic candidate.

        Deployment remains deterministic. The stochastic Libriichi collector
        overrides this hook but shares the same preparation and action codec.
        """
        del game_idx
        return greedy_protocol_candidate(logits, [binding.action for binding in decision.bindings])

    def _record_atomic_decision(self, *, game_idx: int, decision: PreparedDecision | PreparedSemanticDecision, logits: np.ndarray, chosen: int) -> None:
        """Collector hook; greedy Arena intentionally keeps no PPO sidecar."""
        del game_idx, decision, logits, chosen

    def react_batch(self, game_states: Sequence[Any]) -> list[str]:
        if not game_states:
            return []
        reactions: list[str | None] = [None] * len(game_states)
        normal_positions: list[int] = []
        prepared: list[PreparedDecision] = []
        for position, game_state in enumerate(game_states):
            game_idx = int(game_state.game_index)
            memory = self.memories[game_idx]
            self._sync_memory(game_idx, game_state.events_json)
            if memory.pending_riichi is not None:
                action = memory.pending_riichi
                _, mask = game_state.state.encode_obs(self.mortal_version, False)
                mask = np.asarray(mask, dtype=np.bool_)
                if not mask[action.tile]:
                    raise ValueError('stored atomic riichi discard is no longer legal')
                memory.pending_riichi = None
                self.riichi_continuation_count += 1
                reactions[position] = json.dumps(canonical_action_to_mjai(action, seat=memory.seat, events=memory.full_events), separators=(',', ':'))
                continue
            normal_positions.append(position)
            prepared.append(self._prepare(game_idx, game_state, memory))
        if prepared:
            if not self.player_ids:
                raise RuntimeError('player ids are unavailable')
            inputs = self._stack_prepared(prepared)
            logits = np.asarray(self._apply(self.params, inputs))[:len(prepared)]
            for row, position in enumerate(normal_positions):
                decision = prepared[row]
                game_idx = int(game_states[position].game_index)
                legal_logits = logits[row, :len(decision.bindings)]
                chosen = self._choose_candidate(game_idx=game_idx, logits=legal_logits, decision=decision)
                if chosen >= len(decision.bindings):
                    raise ValueError('actor selected a padded candidate')
                self._record_atomic_decision(game_idx=game_idx, decision=decision, logits=legal_logits, chosen=chosen)
                binding = decision.bindings[chosen]
                self.decision_count += 1
                self.action_counts[binding.action.kind.name] += 1
                self.maximum_candidate_count = max(self.maximum_candidate_count, len(decision.bindings))
                memory = self.memories[game_idx]
                if binding.action.kind == ActionKind.RIICHI_DISCARD:
                    memory.pending_riichi = binding.action
                    event = {'type': 'reach', 'actor': memory.seat}
                else:
                    event = canonical_action_to_mjai(binding.action, seat=memory.seat, events=memory.full_events)
                reactions[position] = json.dumps(event, separators=(',', ':'))
        if any((reaction is None for reaction in reactions)):
            raise RuntimeError('failed to produce one reaction per arena state')
        return [str(reaction) for reaction in reactions]

class SemanticRINMjaiEngine(RINMjaiEngine):
    """Libriichi arena adapter for the MahJax-native semantic RIN actor.

    Libriichi owns the reference arena rules and exact legal-action mask here.
    The adapter currently obtains that mask through ``encode_obs`` for strict
    protocol parity, but discards the returned planes. The actor observation
    itself is reconstructed only from public MJAI events.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.semantic_memories = {}
        self.shanten_contract = PUBLIC_SHANTEN_CONTRACT

    def start_game(self, game_idx: int) -> None:
        super().start_game(game_idx)
        self.semantic_memories[game_idx] = SemanticPublicState(seat=self.player_ids[game_idx])

    def _sync_memory(self, game_idx: int, events_json: str) -> None:
        suffix = self.memories[game_idx].sync(events_json)
        semantic = self.semantic_memories[game_idx]
        for event in suffix:
            semantic.update(event)

    def _prepare(self, game_idx: int, game_state: Any, memory: PublicGameMemory) -> PreparedSemanticDecision:
        state = game_state.state
        _, primary_mask = state.encode_obs(self.mortal_version, False)
        primary_mask = np.asarray(primary_mask, dtype=np.bool_)
        if primary_mask.shape != (MORTAL_ACTION_COUNT,):
            raise ValueError('arena Mortal mask differs from the action adapter contract')
        snapshot = snapshot_player_state(state, state.last_cans)
        riichi_mask = self._riichi_continuation_mask(memory) if primary_mask[MORTAL_RIICHI] else None
        kan_mask = None
        if primary_mask[MORTAL_KAN]:
            _, encoded = state.encode_obs(self.mortal_version, True)
            kan_mask = np.asarray(encoded, dtype=np.bool_)
        bindings = tuple(enumerate_mortal_atomic_candidates(snapshot, primary_mask, riichi_continuation_mask=riichi_mask, kan_continuation_mask=kan_mask))
        padded_history = pad_history(memory.public_events, max_history=self.config.max_history)
        padded_candidates = pad_candidates(bindings, max_candidates=self.config.max_candidates)
        phase = bindings[0].action.decision_phase
        global_features = memory.global_state.vector_for(memory.seat, shanten=native_public_shanten(state, contract=self.shanten_contract), furiten=bool(state.at_furiten), decision_phase=phase, rule_profile=self.rule_profile)
        return PreparedSemanticDecision(semantic=self.semantic_memories[game_idx].snapshot(), history={name: padded_history[name] for name in HISTORY_FIELDS}, history_mask=padded_history['mask'], global_features=global_features, candidates={name: padded_candidates[name] for name in CANDIDATE_FIELDS}, candidate_mask=padded_candidates['mask'], bindings=bindings)

    def _stack_prepared(self, decisions: Sequence[Any]) -> dict[str, Any]:
        return stack_prepared_semantic_decisions(decisions, fixed_batch_size=len(self.player_ids))
