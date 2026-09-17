import { create } from 'zustand'
import type { BotResponse, BotStatus, GameStateSnapshot, MjaiEvent } from '@/types'

export type UnavailableObserver = {
  status: 'unavailable'
  reason: 'no_compatible_observer'
  actor_identity: string
  observer_identity: null
}

function object(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null
}

// Only the currently verified unavailable contract is supported. Numeric
// output and weights for a different actor must not be interpreted here.
export function parseObserver(meta: unknown): UnavailableObserver | null {
  const root = object(meta)
  const observer = object(root?.observer)
  if (!observer || observer.status !== 'unavailable'
    || observer.reason !== 'no_compatible_observer'
    || typeof observer.actor_identity !== 'string'
    || !/^[a-f\d]{64}$/i.test(observer.actor_identity)
    || observer.observer_identity !== null) return null
  const model = object(root?.model_identity)
  if (model?.actor_sha256 !== undefined && model.actor_sha256 !== observer.actor_identity) return null
  return {
    status: 'unavailable', reason: 'no_compatible_observer',
    actor_identity: observer.actor_identity, observer_identity: null,
  }
}

type ObserverStore = {
  gameActive: boolean
  lifecycleSeen: boolean
  runnerReady: boolean
  botName: string | null
  observer: UnavailableObserver | null
  hydrateBot: (status: BotStatus) => void
  hydrateGame: (snapshot: GameStateSnapshot | null) => void
  onGameEvent: (event: MjaiEvent) => void
  onBotStatus: (status: BotStatus) => void
  onResponse: (response: BotResponse) => void
}

export const useObserverStore = create<ObserverStore>((set) => ({
  gameActive: false, lifecycleSeen: false, runnerReady: false, botName: null, observer: null,
  hydrateBot: (status) => set((state) => state.lifecycleSeen ? state : {
    botName: 'bot' in status ? status.bot : null,
    runnerReady: status.state === 'ready', observer: null,
  }),
  hydrateGame: (snapshot) => set((state) => state.lifecycleSeen ? state : {
    gameActive: snapshot !== null && !snapshot.is_done,
    observer: null,
  }),
  onGameEvent: (event) => {
    if (event.type === 'start_game') {
      // Wait for the new runner's ready signal before accepting responses;
      // a delayed reply from the previous runner must not fill this tile.
      set({ gameActive: true, lifecycleSeen: true, runnerReady: false, observer: null })
    } else if (event.type === 'end_game') {
      set({ gameActive: false, lifecycleSeen: true, runnerReady: false, observer: null })
    } else if (event.type === 'start_kyoku') {
      set({ gameActive: true, observer: null })
    } else if (event.type === 'end_kyoku') {
      set({ observer: null })
    }
  },
  onBotStatus: (status) => set({
    botName: 'bot' in status ? status.bot : null,
    runnerReady: status.state === 'ready',
    observer: null,
  }),
  onResponse: (response) => set((state) => ({
    observer: state.gameActive && state.runnerReady && state.botName === 'rin-native'
      ? parseObserver(response.meta)
      : null,
  })),
}))

export function observerMessage(state: Pick<ObserverStore, 'gameActive' | 'observer'>) {
  if (!state.gameActive) return 'observer.waiting'
  return state.observer ? 'observer.unavailable' : 'observer.not_provided'
}
