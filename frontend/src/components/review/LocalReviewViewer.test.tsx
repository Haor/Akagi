import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { LocalReviewViewer } from './LocalReviewViewer'
import { useLocalReviewStore, type LocalReviewResult } from '@/stores/localReviewStore'
import type { GameStateSnapshot, MahgenView } from '@/types'
import chinese from '@/i18n/resources/zh-CN.json'

const mocks = vi.hoisted(() => ({ invoke: vi.fn() }))
vi.mock('@/lib/tauri', () => ({ invoke: mocks.invoke }))
vi.mock('react-i18next', () => ({ useTranslation: () => ({
  i18n: { language: 'zh-CN' },
  t: (key: string) => key.split('.').reduce<unknown>((value, part) => (value as Record<string, unknown>)?.[part], chinese) ?? key,
}) }))
vi.mock('@/components/GameBoard', () => ({ GameBoard: ({ game }: { game: GameStateSnapshot | null }) => <span data-testid="board-hand">{game?.players[0].tehai.join(',') ?? 'empty'}</span> }))

const result: LocalReviewResult = {
  history_id: 'record', bot: 'rin-native', created_at: '2026-09-17', seat: 0,
  event_count: 6, compared: 1, matched: 0,
  events: [
    { type: 'start_game' }, { type: 'start_kyoku', bakaze: 'E', kyoku: 1, honba: 0 },
    { type: 'tsumo', actor: 0, pai: '2s' }, { type: 'dahai', actor: 0, pai: '3p' },
    { type: 'end_kyoku' }, { type: 'end_game' },
  ],
  decisions: [{ event_index: 2, round: 'E1 · 0', turn: 1, hand: ['3p', '2s'],
    trigger: { type: 'tsumo', actor: 0, pai: '2s' }, actual: { type: 'dahai', actor: 0, pai: '3p' },
    recommended: { type: 'dahai', actor: 0, pai: '2s' }, matches: false,
    meta: { candidates: [{ action: { type: 'dahai', actor: 0, pai: '2s' }, probability: 0.8, selected: true }] },
  }],
}
function frame(hand: string[]) {
  const game: GameStateSnapshot = {
    bakaze: 'E', kyoku: 1, honba: 0, kyotaku: 0, oya: 0, current_player: 0,
    turn_count: 1, phase: 'wait_act', is_done: false, num_players: 4, our_seat: 0,
    dora_markers: ['E'], players: Array.from({ length: 4 }, (_, seat) => ({
      seat, tehai: seat === 0 ? hand : Array(13).fill('?'), melds: [], river: [], score: 25000,
      riichi_declared: false, riichi_stage: false, double_riichi: false,
      riichi_declaration_index: null, kita_tiles: [],
    })),
  }
  const view: MahgenView = { num_players: 4, dora_indicators: '1z', players: [] }
  return { game, view }
}
beforeEach(() => {
  mocks.invoke.mockResolvedValue([null, frame(['3p']), frame(['3p', '2s']), frame(['2s']), null, null])
  useLocalReviewStore.setState({ selectedId: 'record', result, error: null })
})
afterEach(() => { vi.useRealTimers(); vi.clearAllMocks() })

it('synchronizes recorded table frames and decision selection with Chinese tile names', async () => {
  render(<LocalReviewViewer result={result} />)
  await waitFor(() => expect(screen.getByTestId('board-hand').textContent).toBe('3p,2s'))
  expect(mocks.invoke).toHaveBeenCalledWith('get_local_review_frames', { id: 'record' })
  expect(screen.getByText('实际操作: 打牌 三筒')).toBeTruthy()
  expect(screen.getByText('RIN 建议: 打牌 二索')).toBeTruthy()
  expect(screen.getByText('80.00%')).toBeTruthy()
  fireEvent.click(screen.getByRole('button', { name: '下一步' }))
  expect(screen.getByTestId('board-hand').textContent).toBe('2s')
  expect(screen.queryByText('80.00%')).toBeNull()
  const decisions = screen.getByRole('generic', { name: '决策列表' })
  fireEvent.click(within(decisions).getByRole('button'))
  expect(screen.getByTestId('board-hand').textContent).toBe('3p,2s')
  fireEvent.change(screen.getByRole('slider'), { target: { value: '0' } })
  expect(screen.getByTestId('board-hand').textContent).toBe('3p')
})

it('plays recorded events, skips empty frames, and stops at the end', async () => {
  render(<LocalReviewViewer result={result} />)
  await waitFor(() => expect(screen.getByTestId('board-hand').textContent).toBe('3p,2s'))
  vi.useFakeTimers()
  fireEvent.click(screen.getByRole('button', { name: '播放' }))
  await act(async () => { vi.advanceTimersByTime(650) })
  expect(screen.getByTestId('board-hand').textContent).toBe('2s')
  await act(async () => { vi.advanceTimersByTime(1) })
  expect(screen.getByRole('button', { name: '播放' })).toBeTruthy()
  fireEvent.click(screen.getByRole('button', { name: '播放' }))
  expect(screen.getByTestId('board-hand').textContent).toBe('3p')
  fireEvent.click(screen.getByRole('button', { name: '暂停' }))
  await act(async () => { vi.advanceTimersByTime(2000) })
  expect(screen.getByTestId('board-hand').textContent).toBe('3p')
})

it('reports a local frame loading error without starting playback', async () => {
  mocks.invoke.mockRejectedValueOnce(new Error('record missing'))
  render(<LocalReviewViewer result={result} />)
  expect((await screen.findByRole('alert')).textContent).toContain('record missing')
  expect((screen.getByRole('button', { name: '播放' }) as HTMLButtonElement).disabled).toBe(true)
})

function StoredViewer() {
  const current = useLocalReviewStore((state) => state.result)
  return current ? <LocalReviewViewer key={current.history_id} result={current} /> : null
}

function chooseTruthFile(content: string) {
  const file = new File([content], 'complete.mjai', { type: 'application/json' })
  Object.defineProperty(file, 'text', { value: async () => content })
  fireEvent.change(screen.getByLabelText(chinese.observer.truth_import), { target: { files: [file] } })
}

it('imports local MJAI text and refreshes the selected review through the store', async () => {
  const refreshed: LocalReviewResult = { ...result, decisions: [{ ...result.decisions[0], observer_truth: {
    schema_version: 'akagi.observer-truth.v1', event_index: 2, opponents: [], candidates: [],
  } }] }
  mocks.invoke.mockImplementation(async (command: string) => command === 'import_local_review_truth' ? refreshed : [null, frame(['3p']), frame(['3p', '2s']), frame(['2s']), null, null])
  render(<StoredViewer />)
  chooseTruthFile('{"type":"start_game"}\n')
  expect(await screen.findByText(chinese.observer.truth_import_success)).toBeTruthy()
  expect(mocks.invoke).toHaveBeenCalledWith('import_local_review_truth', { id: 'record', content: '{"type":"start_game"}\n' })
  expect(useLocalReviewStore.getState().result).toEqual(refreshed)
  expect(screen.getByTestId('board-hand').textContent).toBe('3p,2s')
})

it('preserves the review on rejected import and does not reveal internal error details', async () => {
  mocks.invoke.mockImplementation(async (command: string) => {
    if (command === 'import_local_review_truth') throw new Error('private/internal/path: wrong public events')
    return [null, frame(['3p']), frame(['3p', '2s']), frame(['2s']), null, null]
  })
  render(<StoredViewer />)
  chooseTruthFile('invalid')
  expect(await screen.findByText(chinese.observer.truth_import_error)).toBeTruthy()
  expect(screen.queryByText('private/internal/path', { exact: false })).toBeNull()
  expect(useLocalReviewStore.getState().result).toBe(result)
  expect((screen.getByLabelText(chinese.observer.truth_import) as HTMLInputElement).disabled).toBe(false)
})

it('ignores a late import response after switching to another review', async () => {
  let resolve!: (value: LocalReviewResult) => void
  mocks.invoke.mockImplementation((command: string) => command === 'import_local_review_truth'
    ? new Promise<LocalReviewResult>((done) => { resolve = done })
    : Promise.resolve([null, frame(['3p']), frame(['3p', '2s']), frame(['2s']), null, null]))
  render(<StoredViewer />)
  chooseTruthFile('full record')
  await waitFor(() => expect(mocks.invoke).toHaveBeenCalledWith('import_local_review_truth', { id: 'record', content: 'full record' }))
  const other = { ...result, history_id: 'other' }
  act(() => useLocalReviewStore.setState({ selectedId: 'other', result: other }))
  await act(async () => resolve({ ...result }))
  expect(useLocalReviewStore.getState().result).toBe(other)
  expect(screen.queryByText(chinese.observer.truth_import_success)).toBeNull()
  expect(screen.queryByText(chinese.observer.truth_import_error)).toBeNull()
})
