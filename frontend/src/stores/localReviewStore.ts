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
export type LocalDecision = {
  event_index: number
  round: string
  turn: number
  hand: string[]
  trigger: ReviewAction
  actual: ReviewAction | null
  recommended: ReviewAction
  matches: boolean | null
  meta: {
    model_identity?: { model_id?: string; actor_sha256?: string }
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
}

export const useLocalReviewStore = create<LocalReviewStore>((set, get) => ({
  selectedId: null, runningId: null, result: null, loading: false, progress: null, error: null,
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
