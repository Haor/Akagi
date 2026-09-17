import { beforeEach, describe, expect, it, vi } from 'vitest'
import { useLocalReviewStore, type LocalReviewResult, type TruthFetchResult } from './localReviewStore'

const mocks = vi.hoisted(() => ({ invoke: vi.fn(), unlisten: vi.fn(), listen: vi.fn() }))
vi.mock('@/lib/tauri', () => ({ invoke: mocks.invoke, listen: mocks.listen }))

const result = { history_id: 'game', bot: 'rin-native', decisions: [], events: [], compared: 0, matched: 0 } as unknown as LocalReviewResult

beforeEach(() => {
  vi.resetAllMocks()
  mocks.listen.mockResolvedValue(mocks.unlisten)
  mocks.invoke.mockImplementation(async (command: string) => command === 'fetch_local_review_truth'
    ? { status: 'unavailable', reason: 'browser_unavailable', review: null } : result)
  useLocalReviewStore.setState({ selectedId: null, runningId: null, result: null, loading: false, progress: null, error: null, truthFetch: null })
})

describe('offline review', () => {
  it('runs without config or API credentials and calls only the local backend', async () => {
    await useLocalReviewStore.getState().start('game')
    expect(mocks.invoke.mock.calls).toEqual([
      ['local_review_history_game', { id: 'game' }], ['fetch_local_review_truth', { id: 'game' }],
    ])
    expect(useLocalReviewStore.getState().result).toEqual(result)
    expect(useLocalReviewStore.getState().runningId).toBeNull()
    expect(mocks.unlisten).toHaveBeenCalledOnce()
  })

  it('displays saved native results while truth loads and applies the refreshed review when ready', async () => {
    let resolve!: (value: TruthFetchResult) => void
    mocks.invoke.mockImplementation((command: string) => command === 'fetch_local_review_truth'
      ? new Promise<TruthFetchResult>((done) => { resolve = done }) : Promise.resolve(result))
    await useLocalReviewStore.getState().open('game')
    expect(useLocalReviewStore.getState()).toMatchObject({ result, loading: false, truthFetch: { id: 'game', status: 'loading' } })
    await useLocalReviewStore.getState().open('game')
    expect(mocks.invoke.mock.calls.filter(([command]) => command === 'fetch_local_review_truth')).toHaveLength(1)
    const refreshed = { ...result }
    resolve({ status: 'ready', review: refreshed })
    await vi.waitFor(() => expect(useLocalReviewStore.getState().truthFetch?.status).toBe('ready'))
    expect(useLocalReviewStore.getState().result).toBe(refreshed)
  })

  it('preserves native results on failed retrieval, exposes the reason, and retries when reopened', async () => {
    useLocalReviewStore.setState({ selectedId: 'game', result })
    mocks.invoke.mockRejectedValueOnce(new Error('private download diagnostics'))
    await useLocalReviewStore.getState().fetchTruth('game')
    expect(useLocalReviewStore.getState()).toMatchObject({ result, error: null, truthFetch: { status: 'unavailable', reason: 'download_failed' } })
    mocks.invoke.mockResolvedValueOnce({ status: 'unavailable', reason: 'login_required', review: null })
    await useLocalReviewStore.getState().fetchTruth('game')
    expect(useLocalReviewStore.getState()).toMatchObject({ result, truthFetch: { status: 'unavailable', reason: 'login_required' } })
    mocks.invoke.mockResolvedValueOnce({ status: 'ready', review: result })
    await useLocalReviewStore.getState().open('game')
    await vi.waitFor(() => expect(useLocalReviewStore.getState().truthFetch?.status).toBe('ready'))
    expect(useLocalReviewStore.getState().result).toBe(result)
  })

  it.each(['selection', 'result'])('ignores truth arriving after the %s changes', async (change) => {
    let resolve!: (value: TruthFetchResult) => void
    mocks.invoke.mockImplementation(() => new Promise<TruthFetchResult>((done) => { resolve = done }))
    useLocalReviewStore.setState({ selectedId: 'game', result })
    const pending = useLocalReviewStore.getState().fetchTruth('game')
    const newer = { ...result, history_id: change === 'selection' ? 'other' : 'game' }
    useLocalReviewStore.setState({ selectedId: newer.history_id, result: newer })
    resolve({ status: 'ready', review: { ...result } })
    await pending
    expect(useLocalReviewStore.getState().result).toBe(newer)
    expect(useLocalReviewStore.getState().truthFetch).toBeNull()
  })

  it('keeps manual truth when an earlier automatic retrieval finishes later', async () => {
    let resolve!: (value: TruthFetchResult) => void
    const imported = { ...result }
    mocks.invoke.mockImplementation((command: string) => command === 'fetch_local_review_truth'
      ? new Promise<TruthFetchResult>((done) => { resolve = done }) : Promise.resolve(imported))
    useLocalReviewStore.setState({ selectedId: 'game', result })
    const pending = useLocalReviewStore.getState().fetchTruth('game')
    expect(await useLocalReviewStore.getState().importTruth('game', 'complete record')).toBe(true)
    resolve({ status: 'ready', review: { ...result } })
    await pending
    expect(useLocalReviewStore.getState()).toMatchObject({ truthFetch: { id: 'game', status: 'ready' } })
    expect(useLocalReviewStore.getState().result).toBe(imported)
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
