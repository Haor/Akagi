import { beforeEach, describe, expect, it, vi } from 'vitest'
import { useLocalReviewStore, type LocalReviewResult } from './localReviewStore'

const mocks = vi.hoisted(() => ({ invoke: vi.fn(), unlisten: vi.fn(), listen: vi.fn() }))
vi.mock('@/lib/tauri', () => ({ invoke: mocks.invoke, listen: mocks.listen }))

const result = { history_id: 'game', bot: 'rin-native', decisions: [], events: [], compared: 0, matched: 0 } as unknown as LocalReviewResult

beforeEach(() => {
  vi.clearAllMocks()
  mocks.listen.mockResolvedValue(mocks.unlisten)
  useLocalReviewStore.setState({ selectedId: null, runningId: null, result: null, loading: false, progress: null, error: null })
})

describe('offline review', () => {
  it('runs without config or API credentials and calls only the local backend', async () => {
    mocks.invoke.mockResolvedValue(result)
    await useLocalReviewStore.getState().start('game')
    expect(mocks.invoke).toHaveBeenCalledExactlyOnceWith('local_review_history_game', { id: 'game' })
    expect(useLocalReviewStore.getState().result).toEqual(result)
    expect(useLocalReviewStore.getState().runningId).toBeNull()
    expect(mocks.unlisten).toHaveBeenCalledOnce()
  })

  it('prevents overlapping reviews and clears running state after an error', async () => {
    let reject!: (error: Error) => void
    mocks.invoke.mockImplementation(() => new Promise((_, no) => { reject = no }))
    const first = useLocalReviewStore.getState().start('game')
    await Promise.resolve()
    await useLocalReviewStore.getState().start('second')
    expect(mocks.invoke).toHaveBeenCalledTimes(1)
    reject(new Error('model missing'))
    await first
    expect(useLocalReviewStore.getState().error).toContain('model missing')
    expect(useLocalReviewStore.getState().runningId).toBeNull()
    expect(mocks.unlisten).toHaveBeenCalledOnce()
  })

  it('loads saved results and ignores a late cache response for another game', async () => {
    let resolve!: (value: LocalReviewResult) => void
    mocks.invoke.mockImplementationOnce(() => new Promise((yes) => { resolve = yes })).mockResolvedValueOnce(null)
    const first = useLocalReviewStore.getState().open('game')
    await useLocalReviewStore.getState().open('second')
    resolve(result)
    await first
    expect(useLocalReviewStore.getState().selectedId).toBe('second')
    expect(useLocalReviewStore.getState().result).toBeNull()
    expect(mocks.invoke.mock.calls.map(([command]) => command)).toEqual(['get_local_review', 'get_local_review'])
  })
})
