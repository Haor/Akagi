import { create } from 'zustand'
import { invoke, listen } from '@/lib/tauri'

export type ReviewAction = {
  type: string
  actor?: number
  target?: number
  pai?: string
  consumed?: string[]
  scores?: number[]
  deltas?: number[]
  dora_marker?: string
  bakaze?: string
  kyoku?: number
  honba?: number
}
export type ObserverTruth = {
  schema_version: 'akagi.observer-truth.v1'
  event_index: number
  opponents: { seat: number; tenpai: boolean | null; waits: string[] | null; furiten: boolean | null; reason: string | null }[]
  candidates: {
    candidate_index: number
    // Each ron entry belongs to the corresponding truth.opponents seat.
    ron: (boolean | null)[]
    any_ron: boolean | null
    actual_discard: boolean
    deal_in_points: number | null
  }[]
}
export type LocalDecision = {
  event_index: number
  round: string
  turn: number
  hand: string[]
  trigger: ReviewAction
  actual: ReviewAction | null
  recommended: ReviewAction
  matches: boolean | null
  observer_truth?: ObserverTruth | null
  meta: {
    decision?: boolean
    continuation?: boolean
    observer?: unknown
    model_identity?: { model_id?: string; actor_sha256?: string; original_sha256?: string }
    candidates?: { action: ReviewAction; continuation?: ReviewAction; probability: number; selected: boolean }[]
  } | null
}
export type LocalReviewResult = {
  history_id: string
  bot: string
  created_at: string
  seat: number
  event_count: number
  compared: number
  matched: number
  events: ReviewAction[]
  decisions: LocalDecision[]
}
type Progress = { history_id: string; processed: number; total: number }
type LocalReviewStore = {
  selectedId: string | null
  runningId: string | null
  result: LocalReviewResult | null
  loading: boolean
  progress: Progress | null
  error: string | null
  open: (id: string) => Promise<void>
  start: (id: string) => Promise<void>
  importTruth: (id: string, content: string) => Promise<boolean>
}

export const useLocalReviewStore = create<LocalReviewStore>((set, get) => ({
  selectedId: null, runningId: null, result: null, loading: false, progress: null, error: null,
  importTruth: async (id, content) => {
    if (get().selectedId !== id) return false
    const previous = get().result
    const result = await invoke<LocalReviewResult>('import_local_review_truth', { id, content })
    if (get().selectedId !== id || get().result !== previous) return false
    set({ result })
    return true
  },
  open: async (id) => {
    if (get().selectedId === id) return
    set({ selectedId: id, result: null, error: null, loading: true })
    try {
      const result = await invoke<LocalReviewResult | null>('get_local_review', { id })
      if (get().selectedId === id) set({ result })
    } catch (error) {
      if (get().selectedId === id) set({ error: String(error) })
    } finally {
      if (get().selectedId === id) set({ loading: false })
    }
  },
  start: async (id) => {
    if (get().runningId) return
    set({ runningId: id, selectedId: id, progress: null, error: null })
    let unlisten: (() => void) | undefined
    try {
      unlisten = await listen<Progress>('local-review-progress', (progress) => {
        if (progress.history_id === get().runningId) set({ progress })
      })
      const result = await invoke<LocalReviewResult>('local_review_history_game', { id })
      if (get().selectedId === id) set({ result })
    } catch (error) {
      set({ error: String(error) })
    } finally {
      unlisten?.()
      set({ runningId: null, progress: null })
    }
  },
}))
