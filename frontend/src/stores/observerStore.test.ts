import { beforeEach, describe, expect, it } from 'vitest'
import { observerMessage, parseObserver, useObserverStore } from './observerStore'
import type { BotResponse, GameStateSnapshot } from '@/types'

const actor = 'a'.repeat(64)
const meta = {
  observer: { status: 'unavailable', reason: 'no_compatible_observer', actor_identity: actor, observer_identity: null },
  model_identity: { actor_sha256: actor },
}
const response: BotResponse = { type: 'none', meta }
const start = { type: 'start_game', names: ['a', 'b', 'c', 'd'], id: 0 } as const
const ready = { state: 'ready', bot: 'rin-native', actor_id: 0 } as const
const snapshot = { is_done: false } as GameStateSnapshot

beforeEach(() => {
  useObserverStore.setState({ gameActive: false, lifecycleSeen: false, runnerReady: false, botName: null, observer: null })
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
})
