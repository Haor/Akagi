import { create } from 'zustand'
import type { BotResponse, BotStatus, GameStateSnapshot, MjaiEvent } from '@/types'

export type UnavailableObserver = {
  status: 'unavailable'
  reason: 'no_compatible_observer'
  actor_identity: string
  observer_identity: null
}

export type ObserverOpponent = {
  seat: number
  relative_seat: 1 | 2 | 3
  tenpai_probability: number
  conditional_wait_probability: number[]
  unconditional_wait_probability: number[]
}

export type ObserverCandidate = {
  candidate_index: number
  action_type: 'dahai' | 'reach'
  tile: string
  selected: boolean
  ron_probability: number[]
  any_ron_probability: number
  multiple_ron_probability: number
  conditional_loss_points: number
  expected_loss_points: number
  ron_joint_probability: number[]
  ron_wait_inclusion_violation: number[]
  conditional_payment_probability?: number[]
  conditional_payment_ge_8000?: number
  payment_ge_8000?: number
  conditional_payment_ge_12000?: number
  payment_ge_12000?: number
  conditional_payment_ge_24000?: number
  payment_ge_24000?: number
}

export type ReadyObserver = {
  status: 'ready'
  schema_version: 'rin.observer-readout.v1'
  kind: 'actor-token-cross-attention'
  actor_identity: string
  observer_identity: {
    source_sha256: string
    safetensors_sha256: string
    config_sha256: string
    representation_actor_sha256: string
  }
  opponent_order: number[]
  opponents: ObserverOpponent[]
  candidates: ObserverCandidate[]
  payment_bucket_edges?: number[]
}

export type ObserverReadout = UnavailableObserver | ReadyObserver
  | { status: 'not_evaluated'; reason: 'riichi_continuation'; actor_identity: string }
  | { status: 'error'; reason: string; actor_identity: string }

function object(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null
}

const sha = (value: unknown): value is string => typeof value === 'string' && /^[a-f\d]{64}$/i.test(value)
const probability = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1
const points = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value) && value >= 0
const probabilities = (value: unknown, length: number): value is number[] => Array.isArray(value) && value.length === length && value.every(probability)
const seatNumber = (value: unknown): value is number => Number.isInteger(value) && Number(value) >= 0 && Number(value) < 4
const tile = (value: unknown): value is string => typeof value === 'string' && /^(?:[1-9][mps]|5[mps]r|[ESWNPFC])$/.test(value)

const PAYMENT_TAILS = [
  'conditional_payment_ge_8000', 'payment_ge_8000',
  'conditional_payment_ge_12000', 'payment_ge_12000',
  'conditional_payment_ge_24000', 'payment_ge_24000',
] as const

// Parse the versioned readout at the boundary. Conditional wait values are
// per-tile probabilities, not a distribution over mutually exclusive tiles.
export function parseObserver(meta: unknown, expectedSeat?: number | null): ObserverReadout | null {
  const root = object(meta)
  const observer = object(root?.observer)
  if (!root || !observer || !sha(observer.actor_identity)) return null
  const model = object(root?.model_identity)
  if (model?.actor_sha256 !== undefined && model.actor_sha256 !== observer.actor_identity) return null
  if (observer.status === 'not_evaluated' && observer.reason === 'riichi_continuation') {
    return { status: 'not_evaluated', reason: 'riichi_continuation', actor_identity: observer.actor_identity }
  }
  if (observer.status === 'error' && typeof observer.reason === 'string') {
    return { status: 'error', reason: observer.reason, actor_identity: observer.actor_identity }
  }
  if (observer.status === 'unavailable') {
    if (observer.reason !== 'no_compatible_observer' || observer.observer_identity !== null) return null
    return { status: 'unavailable', reason: 'no_compatible_observer', actor_identity: observer.actor_identity, observer_identity: null }
  }
  if (observer.status !== 'ready' || observer.schema_version !== 'rin.observer-readout.v1'
    || observer.kind !== 'actor-token-cross-attention' || root.decision !== true || root.continuation === true
    || model?.actor_sha256 !== observer.actor_identity || !sha(model?.original_sha256)) return null
  const identity = object(observer.observer_identity)
  if (!identity || !sha(identity.source_sha256) || !sha(identity.safetensors_sha256)
    || !sha(identity.config_sha256) || identity.representation_actor_sha256 !== model.original_sha256) return null
  const order = observer.opponent_order
  if (!Array.isArray(order) || order.length !== 3 || !order.every(seatNumber)) return null
  const ownSeat = (order[0] + 3) % 4
  if ((expectedSeat != null && ownSeat !== expectedSeat)
    || !order.every((seat, i) => seat === (ownSeat + i + 1) % 4)) return null
  if (!Array.isArray(observer.opponents) || observer.opponents.length !== 3) return null
  const opponents: ObserverOpponent[] = []
  for (const [i, value] of observer.opponents.entries()) {
    const opponent = object(value)
    if (!opponent || opponent.seat !== order[i] || opponent.relative_seat !== i + 1
      || !probability(opponent.tenpai_probability)
      || !probabilities(opponent.conditional_wait_probability, 34)
      || !probabilities(opponent.unconditional_wait_probability, 34)) return null
    opponents.push({ seat: order[i], relative_seat: (i + 1) as 1 | 2 | 3,
      tenpai_probability: opponent.tenpai_probability,
      conditional_wait_probability: opponent.conditional_wait_probability,
      unconditional_wait_probability: opponent.unconditional_wait_probability,
    })
  }
  if (!Array.isArray(observer.candidates) || !Array.isArray(root.candidates)) return null
  const candidates: ObserverCandidate[] = []
  const seen = new Set<number>()
  for (const value of observer.candidates) {
    const candidate = object(value)
    const index = candidate?.candidate_index
    if (!candidate || typeof index !== 'number' || !Number.isInteger(index) || index < 0
      || index >= root.candidates.length || seen.has(index)) return null
    const source = object(root.candidates[index])
    const action = object(source?.action)
    const continuation = object(source?.continuation)
    const discard = action?.type === 'reach' ? continuation : action
    if (!source || !action || !['dahai', 'reach'].includes(String(action.type)) || action.actor !== ownSeat
      || discard?.type !== 'dahai' || discard.actor !== ownSeat || !tile(discard.pai)
      || (source.probability !== undefined && !probability(source.probability))
      || !probabilities(candidate.ron_probability, 3)
      || !probability(candidate.any_ron_probability) || !probability(candidate.multiple_ron_probability)
      || !points(candidate.conditional_loss_points) || !points(candidate.expected_loss_points)
      || !probabilities(candidate.ron_joint_probability, 8)
      || !probabilities(candidate.ron_wait_inclusion_violation, 3)
      || (candidate.conditional_payment_probability !== undefined && !probabilities(candidate.conditional_payment_probability, 12))
      || PAYMENT_TAILS.some((key) => candidate[key] !== undefined && !probability(candidate[key]))) return null
    seen.add(index)
    const parsed: ObserverCandidate = {
      candidate_index: index, action_type: action.type as 'dahai' | 'reach', tile: discard.pai,
      selected: source.selected === true, ron_probability: candidate.ron_probability,
      any_ron_probability: candidate.any_ron_probability, multiple_ron_probability: candidate.multiple_ron_probability,
      conditional_loss_points: candidate.conditional_loss_points, expected_loss_points: candidate.expected_loss_points,
      ron_joint_probability: candidate.ron_joint_probability, ron_wait_inclusion_violation: candidate.ron_wait_inclusion_violation,
    }
    if (candidate.conditional_payment_probability !== undefined) parsed.conditional_payment_probability = candidate.conditional_payment_probability as number[]
    for (const key of PAYMENT_TAILS) if (candidate[key] !== undefined) parsed[key] = candidate[key] as number
    candidates.push(parsed)
  }
  const edges = observer.payment_bucket_edges
  if (edges !== undefined && (!Array.isArray(edges) || edges.length !== 11
    || !edges.every((value, i) => points(value) && (i === 0 || value > edges[i - 1])))) return null
  return {
    status: 'ready', schema_version: 'rin.observer-readout.v1', kind: 'actor-token-cross-attention',
    actor_identity: observer.actor_identity,
    observer_identity: { source_sha256: identity.source_sha256, safetensors_sha256: identity.safetensors_sha256,
      config_sha256: identity.config_sha256, representation_actor_sha256: model.original_sha256 },
    opponent_order: order, opponents, candidates,
    ...(edges === undefined ? {} : { payment_bucket_edges: edges as number[] }),
  }
}

type ObserverStore = {
  gameActive: boolean
  lifecycleSeen: boolean
  runnerReady: boolean
  botName: string | null
  seat: number | null
  observer: ObserverReadout | null
  hydrateBot: (status: BotStatus) => void
  hydrateGame: (snapshot: GameStateSnapshot | null) => void
  onGameEvent: (event: MjaiEvent) => void
  onBotStatus: (status: BotStatus) => void
  onResponse: (response: BotResponse) => void
}

export const useObserverStore = create<ObserverStore>((set) => ({
  gameActive: false, lifecycleSeen: false, runnerReady: false, botName: null, seat: null, observer: null,
  hydrateBot: (status) => set((state) => state.lifecycleSeen ? state : {
    botName: 'bot' in status ? status.bot : null,
    runnerReady: status.state === 'ready', seat: status.state === 'ready' ? status.actor_id : null, observer: null,
  }),
  hydrateGame: (snapshot) => set((state) => state.lifecycleSeen ? state : {
    gameActive: snapshot !== null && !snapshot.is_done,
    observer: null,
  }),
  onGameEvent: (event) => {
    if (event.type === 'start_game') {
      // Wait for the new runner's ready signal before accepting responses;
      // a delayed reply from the previous runner must not fill this tile.
      set({ gameActive: true, lifecycleSeen: true, runnerReady: false, seat: event.id ?? null, observer: null })
    } else if (event.type === 'end_game') {
      set({ gameActive: false, lifecycleSeen: true, runnerReady: false, seat: null, observer: null })
    } else if (event.type === 'start_kyoku') {
      set({ gameActive: true, observer: null })
    } else if (event.type === 'end_kyoku') {
      set({ observer: null })
    } else {
      // A numeric estimate belongs only to the decision that produced it.
      set((state) => state.observer?.status === 'ready' ? { observer: null } : state)
    }
  },
  onBotStatus: (status) => set({
    botName: 'bot' in status ? status.bot : null,
    runnerReady: status.state === 'ready',
    seat: status.state === 'ready' ? status.actor_id : null,
    observer: null,
  }),
  onResponse: (response) => set((state) => ({
    observer: state.gameActive && state.runnerReady && state.botName === 'rin-native'
      ? parseObserver(response.meta, state.seat)
      : null,
  })),
}))

export function observerMessage(state: Pick<ObserverStore, 'gameActive' | 'observer'>) {
  if (!state.gameActive) return 'observer.waiting'
  switch (state.observer?.status) {
    case 'ready': return 'observer.estimates'
    case 'unavailable': return 'observer.unavailable'
    case 'not_evaluated': return 'observer.riichi_continuation'
    case 'error': return 'observer.error'
    default: return 'observer.not_provided'
  }
}
