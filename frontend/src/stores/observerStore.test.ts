import { beforeEach, describe, expect, it } from 'vitest'
import { observerMessage, parseObserver, useObserverStore } from './observerStore'
import type { BotResponse, GameStateSnapshot, MjaiEvent } from '@/types'

const actor = 'a'.repeat(64)
const meta = {
  observer: { status: 'unavailable', reason: 'no_compatible_observer', actor_identity: actor, observer_identity: null },
  model_identity: { actor_sha256: actor },
}
const response: BotResponse = { type: 'none', meta }
const start = { type: 'start_game', names: ['a', 'b', 'c', 'd'], id: 0 } as const
const ready = { state: 'ready', bot: 'rin-native', actor_id: 0 } as const
const snapshot = { is_done: false } as GameStateSnapshot

function readyMeta() {
  return {
    decision: true, continuation: false,
    model_identity: { actor_sha256: actor, original_sha256: 'b'.repeat(64) },
    candidates: [
      { action: { type: 'none' }, probability: 0.1, selected: false },
      { action: { type: 'dahai', actor: 0, pai: '3p' }, probability: 0.5, selected: true },
      { action: { type: 'reach', actor: 0 }, continuation: { type: 'dahai', actor: 0, pai: '5sr' }, probability: 0.4, selected: false },
    ],
    observer: {
      status: 'ready', schema_version: 'rin.observer-readout.v1', kind: 'actor-token-cross-attention', actor_identity: actor,
      observer_identity: { source_sha256: 'c'.repeat(64), safetensors_sha256: 'd'.repeat(64), config_sha256: 'e'.repeat(64), representation_actor_sha256: 'b'.repeat(64) },
      opponent_order: [1, 2, 3],
      opponents: [1, 2, 3].map((seat) => ({ seat, relative_seat: seat, tenpai_probability: 0.5,
        conditional_wait_probability: Array(34).fill(0.7) as number[], unconditional_wait_probability: Array(34).fill(0.35) as number[] })),
      candidates: [2, 1].map((candidate_index) => ({ candidate_index, ron_probability: [0.1, 0.2, 0.3], any_ron_probability: 0.5,
        multiple_ron_probability: 0.1, conditional_loss_points: 8000, expected_loss_points: 4000,
        ron_joint_probability: [0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0], ron_wait_inclusion_violation: [0, 0, 0],
        conditional_payment_probability: Array(12).fill(1 / 12) as number[], conditional_payment_ge_8000: 0.5, payment_ge_8000: 0.25,
      })),
      payment_bucket_edges: [0.5, 1000, 2000, 4000, 8000, 12000, 16000, 24000, 32000, 48000, 96000],
    },
  }
}

beforeEach(() => {
  useObserverStore.setState({ gameActive: false, lifecycleSeen: false, runnerReady: false, botName: null, seat: null, observer: null })
})

describe('observation-head contract', () => {
  it('accepts the verified missing-weight status and rejects incompatible or unknown outputs', () => {
    expect(parseObserver(meta)).toEqual(meta.observer)
    expect(parseObserver({ ...meta, model_identity: { actor_sha256: 'b'.repeat(64) } })).toBeNull()
    expect(parseObserver({ observer: { ...meta.observer, observer_identity: 'old-rl300-head' } })).toBeNull()
    expect(parseObserver({ observer: { status: 'ready', probabilities: [0.5, 0.5] } })).toBeNull()
    expect(parseObserver({ observer: { ...meta.observer, actor_identity: 'unknown' } })).toBeNull()
    expect(parseObserver(undefined)).toBeNull()
  })

  it('waits outside a game and requires a ready current RIN runner', () => {
    const store = useObserverStore.getState()
    store.onResponse(response)
    expect(observerMessage(useObserverStore.getState())).toBe('observer.waiting')
    store.onGameEvent({ ...start, names: [...start.names] })
    store.onResponse(response)
    expect(observerMessage(useObserverStore.getState())).toBe('observer.not_provided')
    store.onBotStatus(ready)
    store.onResponse(response)
    expect(observerMessage(useObserverStore.getState())).toBe('observer.unavailable')
    store.onBotStatus({ ...ready, bot: 'other-bot' })
    store.onResponse(response)
    expect(observerMessage(useObserverStore.getState())).toBe('observer.not_provided')
  })

  it('clears previous-game output and ignores late responses before the next runner is ready', () => {
    const store = useObserverStore.getState()
    store.onGameEvent({ ...start, names: [...start.names] })
    store.onBotStatus(ready)
    store.onResponse(response)
    store.onGameEvent({ type: 'end_game' })
    store.onResponse(response)
    expect(useObserverStore.getState().observer).toBeNull()
    expect(observerMessage(useObserverStore.getState())).toBe('observer.waiting')
    store.onGameEvent({ ...start, names: [...start.names] })
    store.onResponse(response)
    expect(useObserverStore.getState().observer).toBeNull()
    store.hydrateBot(ready)
    store.hydrateGame(snapshot)
    store.onResponse(response)
    expect(useObserverStore.getState().observer).toBeNull()
    store.onBotStatus(ready)
    store.onResponse(response)
    expect(observerMessage(useObserverStore.getState())).toBe('observer.unavailable')
    store.onGameEvent({ type: 'end_kyoku' })
    expect(useObserverStore.getState().observer).toBeNull()
  })

  it('hydrates a running game without replaying any previously cached response', () => {
    const store = useObserverStore.getState()
    store.hydrateBot(ready)
    store.hydrateGame(snapshot)
    expect(observerMessage(useObserverStore.getState())).toBe('observer.not_provided')
    store.onResponse(response)
    expect(observerMessage(useObserverStore.getState())).toBe('observer.unavailable')
    store.onResponse({ type: 'none' })
    expect(observerMessage(useObserverStore.getState())).toBe('observer.not_provided')
  })

  it('maps readouts by candidate index and preserves non-normalized per-tile wait probabilities', () => {
    const parsed = parseObserver(readyMeta(), 0)
    expect(parsed?.status).toBe('ready')
    if (parsed?.status !== 'ready') throw new Error('expected ready readout')
    expect(parsed.opponents.map((opponent) => opponent.seat)).toEqual([1, 2, 3])
    expect(parsed.opponents[0].conditional_wait_probability.reduce((a, b) => a + b)).toBeGreaterThan(1)
    expect(parsed.candidates[0]).toMatchObject({ candidate_index: 2, action_type: 'reach', tile: '5sr', selected: false })
    expect(parsed.candidates[1]).toMatchObject({ candidate_index: 1, action_type: 'dahai', tile: '3p', selected: true })
    const noDiscard = readyMeta()
    noDiscard.observer.candidates = []
    expect(parseObserver(noDiscard)?.status).toBe('ready')
  })

  it('rejects actor and seat mismatches, bad candidate mappings, non-decisions and forced continuations', () => {
    const cases: ((payload: ReturnType<typeof readyMeta>) => void)[] = [
      (payload) => { payload.model_identity.actor_sha256 = 'f'.repeat(64) },
      (payload) => { payload.observer.observer_identity.representation_actor_sha256 = 'f'.repeat(64) },
      (payload) => { payload.observer.opponent_order = [1, 3, 2] },
      (payload) => { payload.observer.opponents[0].seat = 2 },
      (payload) => { payload.observer.opponents[0].relative_seat = 2 },
      (payload) => { payload.observer.candidates[0].candidate_index = 99 },
      (payload) => { payload.observer.candidates[0].candidate_index = 0 },
      (payload) => { payload.observer.candidates[0].candidate_index = 1 },
      (payload) => { payload.candidates[2].continuation!.actor = 1 },
      (payload) => { payload.decision = false },
      (payload) => { payload.continuation = true },
    ]
    for (const mutate of cases) {
      const payload = readyMeta()
      mutate(payload)
      expect(parseObserver(payload, 0)).toBeNull()
    }
    expect(parseObserver(readyMeta(), 2)).toBeNull()
  })

  it('rejects nonfinite or out-of-range probabilities and inconsistent shapes', () => {
    const cases: ((payload: ReturnType<typeof readyMeta>) => void)[] = [
      (payload) => { payload.observer.opponents[0].tenpai_probability = NaN },
      (payload) => { payload.observer.opponents[1].conditional_wait_probability[0] = 1.1 },
      (payload) => { payload.observer.opponents[2].unconditional_wait_probability.pop() },
      (payload) => { payload.observer.candidates[0].ron_probability = [0.1, 0.2] },
      (payload) => { payload.observer.candidates[0].any_ron_probability = Infinity },
      (payload) => { payload.observer.candidates[0].multiple_ron_probability = -0.1 },
      (payload) => { payload.observer.candidates[0].ron_joint_probability.pop() },
      (payload) => { payload.observer.candidates[0].ron_wait_inclusion_violation[0] = -0.01 },
      (payload) => { payload.observer.candidates[0].conditional_loss_points = -1 },
      (payload) => { payload.observer.candidates[0].expected_loss_points = Infinity },
      (payload) => { payload.observer.candidates[0].conditional_payment_probability.pop() },
      (payload) => { payload.observer.candidates[0].payment_ge_8000 = 2 },
      (payload) => { payload.observer.payment_bucket_edges.reverse() },
      (payload) => { payload.observer.payment_bucket_edges.push(192000) },
    ]
    for (const mutate of cases) {
      const payload = readyMeta()
      mutate(payload)
      expect(parseObserver(payload)).toBeNull()
    }
  })

  it('holds the latest estimate across other turns and forced continuations, then replaces it', () => {
    const store = useObserverStore.getState()
    store.onGameEvent({ ...start, names: [...start.names] })
    store.onBotStatus(ready)
    store.onResponse({ type: 'none', meta: readyMeta() })
    expect(observerMessage(useObserverStore.getState())).toBe('observer.estimates')
    const previous = useObserverStore.getState().observer
    const events: MjaiEvent[] = [
      { type: 'dahai', actor: 0, pai: '3p', tsumogiri: false },
      { type: 'tsumo', actor: 1, pai: '?' },
      { type: 'dahai', actor: 1, pai: '2s', tsumogiri: true },
      { type: 'pon', actor: 2, target: 1, pai: '2s', consumed: ['2s', '2s'] },
      { type: 'tsumo', actor: 0, pai: '1m' },
    ]
    for (const event of events) {
      store.onGameEvent(event)
      store.onResponse({ type: 'none' })
      store.onResponse({ type: 'none', meta: { decision: false } })
      expect(useObserverStore.getState().observer).toBe(previous)
    }
    store.onResponse({ type: 'none', meta: { ...readyMeta(), observer: { status: 'not_evaluated', reason: 'riichi_continuation', actor_identity: actor } } })
    expect(useObserverStore.getState().observer).toBe(previous)
    const next = readyMeta()
    next.observer.opponents[0].tenpai_probability = 0.8
    store.onResponse({ type: 'none', meta: next })
    expect(useObserverStore.getState().observer).toEqual(parseObserver(next, 0))
    expect(useObserverStore.getState().observer).not.toEqual(previous)
    store.onGameEvent({ type: 'end_kyoku' })
    expect(useObserverStore.getState().observer).toBeNull()
  })

  it('clears held estimates on a new round, bot change, or explicit observer error', () => {
    const store = useObserverStore.getState()
    store.onGameEvent({ ...start, names: [...start.names] })
    store.onBotStatus(ready)
    store.onResponse({ type: 'none', meta: readyMeta() })
    store.onGameEvent({ type: 'start_kyoku', bakaze: 'E', kyoku: 2, honba: 0, kyotaku: 0, oya: 1,
      dora_marker: '1m', scores: [25000, 25000, 25000, 25000], tehais: Array.from({ length: 4 }, () => Array<string>(13).fill('?')) })
    expect(useObserverStore.getState().observer).toBeNull()
    store.onResponse({ type: 'none', meta: readyMeta() })
    store.onBotStatus({ ...ready, bot: 'other-bot' })
    expect(useObserverStore.getState().observer).toBeNull()
    store.onBotStatus(ready)
    store.onResponse({ type: 'none', meta: readyMeta() })
    store.onResponse({ type: 'none', meta: { ...readyMeta(), observer: { status: 'error', reason: 'invalid_weights', actor_identity: actor } } })
    expect(observerMessage(useObserverStore.getState())).toBe('observer.error')
    store.onResponse({ type: 'none', meta: { ...readyMeta(), decision: false } })
    expect(useObserverStore.getState().observer).toBeNull()
  })
})
