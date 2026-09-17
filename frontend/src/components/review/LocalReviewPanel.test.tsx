import { beforeEach, expect, it, vi } from 'vitest'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { Review } from '@/routes/Review'
import { useHistoryStore } from '@/stores/historyStore'
import { useLocalReviewStore } from '@/stores/localReviewStore'
import type { GameRecord } from '@/types'

const mocks = vi.hoisted(() => ({ invoke: vi.fn() }))
vi.mock('@/lib/tauri', () => ({ invoke: mocks.invoke, listen: async () => () => {} }))
vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key, i18n: { language: 'zh-CN' } }) }))
vi.mock('@/components/Mahgen', () => ({ Mahgen: ({ seq }: { seq: string }) => <span>{seq}</span> }))

beforeEach(() => {
  vi.clearAllMocks()
  useLocalReviewStore.setState({ selectedId: null, runningId: null, result: null, loading: false, progress: null, error: null })
  useHistoryStore.setState({ records: [{ id: 'record', num_players: 4, our_seat: 0, started_at: '2026-09-01T00:00:00Z' }] as GameRecord[] })
  mocks.invoke.mockResolvedValue(null)
})

it('opens local review directly with no API key and does not mount the cloud gate', async () => {
  render(<MemoryRouter initialEntries={['/review?game=record']}><Review /></MemoryRouter>)
  await waitFor(() => expect(mocks.invoke).toHaveBeenCalledWith('get_local_review', { id: 'record' }))
  expect(screen.queryByText('review.need_key_title')).toBeNull()
  expect(screen.getByText('review.local_description')).toBeTruthy()
  const button = screen.getByRole('button', { name: 'review.start_review' })
  await waitFor(() => expect((button as HTMLButtonElement).disabled).toBe(false))
  fireEvent.click(button)
  await waitFor(() => expect(mocks.invoke).toHaveBeenCalledWith('local_review_history_game', { id: 'record' }))
  expect(mocks.invoke.mock.calls.every(([command]) => !command.startsWith('native_api_'))).toBe(true)
})
