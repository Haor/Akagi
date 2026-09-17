import { expect, it, vi } from 'vitest'
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { ObserverDetails } from './ObserverDetails'
import { LocalReviewViewer } from './review/LocalReviewViewer'
import type { ReadyObserver } from '@/stores/observerStore'
import type { LocalReviewResult, ObserverTruth } from '@/stores/localReviewStore'
import chinese from '@/i18n/resources/zh-CN.json'

const mocks = vi.hoisted(() => ({ invoke: vi.fn() }))
vi.mock('@/lib/tauri', () => ({ invoke: mocks.invoke }))
vi.mock('@/components/GameBoard', () => ({ GameBoard: () => <div /> }))
vi.mock('react-i18next', () => ({ useTranslation: () => ({
  i18n: { language: 'zh-CN' },
  t: (key: string, values: Record<string, unknown> = {}) => {
    const text = key.split('.').reduce<unknown>((value, part) => (value as Record<string, unknown>)?.[part], chinese)
    return String(text ?? key).replace(/\{\{(\w+)\}\}/g, (_, name: string) => String(values[name] ?? name))
  },
}) }))

function readout(): ReadyObserver {
  const waits = Array(34).fill(0.01) as number[]
  for (const [index, value] of [[33, 0.7], [27, 0.6], [22, 0.5], [17, 0.4], [0, 0.3]]) waits[index] = value
  return {
    status: 'ready', schema_version: 'rin.observer-readout.v1', kind: 'actor-token-cross-attention', actor_identity: 'a'.repeat(64),
    observer_identity: { source_sha256: 'c'.repeat(64), safetensors_sha256: 'd'.repeat(64), config_sha256: 'e'.repeat(64), representation_actor_sha256: 'b'.repeat(64) },
    opponent_order: [3, 0, 1],
    opponents: [3, 0, 1].map((seat, i) => ({ seat, relative_seat: (i + 1) as 1 | 2 | 3, tenpai_probability: (i + 1) / 10,
      conditional_wait_probability: waits, unconditional_wait_probability: waits.map((value) => value / 10) })),
    candidates: [{ candidate_index: 2, action_type: 'reach', tile: '5sr', selected: true,
      ron_probability: [0.1, 0.2, 0.3], any_ron_probability: 0.5, multiple_ron_probability: 0.1,
      conditional_loss_points: 8000, expected_loss_points: 4000,
      ron_joint_probability: [0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0], ron_wait_inclusion_violation: [0, 0, 0] }],
  }
}

it('renders rotated seats and candidate probabilities in matching columns with Chinese wait names', () => {
  const { container } = render(<ObserverDetails observer={readout()} />)
  const right = screen.getByRole('article', { name: '下家 · 座位 4' })
  expect(within(right).getByText('10.00%')).toBeTruthy()
  expect(within(screen.getByRole('article', { name: '对家 · 座位 1' })).getByText('20.00%')).toBeTruthy()
  expect(within(screen.getByRole('article', { name: '上家 · 座位 2' })).getByText('30.00%', { selector: 'strong' })).toBeTruthy()
  for (const name of ['红中', '东风', '五索', '九筒', '一万']) expect(within(right).getByText(name, { exact: false })).toBeTruthy()
  expect(within(right).queryByText('二万', { exact: false })).toBeNull()
  expect(screen.getByText(chinese.observer.wait_hint)).toBeTruthy()
  const row = container.querySelector('[data-candidate-index="2"]') as HTMLTableRowElement
  expect([...row.cells].map((cell) => cell.textContent)).toEqual(['✓ 立直 红五索', '10.00%', '20.00%', '30.00%', '50.00%', '4,000'])
  expect(screen.getByText(chinese.observer.ron_hint)).toBeTruthy()
  expect(screen.queryByText(chinese.observer.truth_hint)).toBeNull()
  expect(screen.queryByLabelText(chinese.observer.truth_comparison)).toBeNull()
})

it('pairs truth by seat and candidate index, preserving full waits and false, unknown and zero outcomes', () => {
  const observer = readout()
  observer.candidates.push({ ...observer.candidates[0], candidate_index: 1, selected: false, action_type: 'dahai', tile: '3p' })
  const truth: ObserverTruth = {
    schema_version: 'akagi.observer-truth.v1', event_index: 1,
    opponents: [
      { seat: 1, tenpai: null, waits: null, furiten: null, reason: 'missing_hand' },
      { seat: 3, tenpai: true, waits: ['2m', '6m', '5s'], furiten: false, reason: null },
      { seat: 0, tenpai: false, waits: [], furiten: false, reason: null },
    ],
    candidates: [
      { candidate_index: 1, ron: [null, false, false], any_ron: null, actual_discard: false, deal_in_points: null },
      { candidate_index: 2, ron: [null, true, false], any_ron: true, actual_discard: true, deal_in_points: 0 },
    ],
  }
  const { container, rerender } = render(<ObserverDetails observer={observer} truth={truth} />)
  const right = within(screen.getByRole('article', { name: '下家 · 座位 4' }))
  expect(right.getByText('实际听牌:', { exact: false }).textContent).toBe('实际听牌: 是')
  expect(right.getByText('实际待牌（全部）:', { exact: false }).textContent).toBe('实际待牌（全部）: 二万、六万、五索')
  expect(within(screen.getByRole('article', { name: '对家 · 座位 1' })).getByText('实际听牌:', { exact: false }).textContent).toBe('实际听牌: 否')
  expect(within(screen.getByRole('article', { name: '上家 · 座位 2' })).getByText('实际听牌:', { exact: false }).textContent).toBe('实际听牌: 未知')
  expect(screen.getByText(chinese.observer.truth_missing_hand)).toBeTruthy()
  const played = container.querySelector('[data-candidate-index="2"]') as HTMLTableRowElement
  expect([...played.cells].slice(1).map((cell) => cell.textContent)).toEqual(['10.00%真值：是', '20.00%真值：否', '30.00%真值：未知', '50.00%真值：是', '4,000实打损失：0'])
  const unplayed = container.querySelector('[data-candidate-index="1"]') as HTMLTableRowElement
  expect(unplayed.cells[5].textContent).toBe('4,000实打损失：未实打')
  const missingOutcome = { ...truth, candidates: truth.candidates.map((candidate) => ({ ...candidate, deal_in_points: null })) }
  rerender(<ObserverDetails observer={observer} truth={missingOutcome} />)
  expect(played.cells[5].textContent).toBe('4,000实打损失：未知')
})

it('shows unknown truth explicitly for a local review without adding it to live estimates', () => {
  const { rerender } = render(<ObserverDetails observer={readout()} truth={null} />)
  expect(screen.getByText(chinese.observer.truth_unavailable)).toBeTruthy()
  expect(screen.getAllByText('实际听牌:', { exact: false }).every((item) => item.textContent === '实际听牌: 未知')).toBe(true)
  rerender(<ObserverDetails observer={readout()} />)
  expect(screen.queryByText(chinese.observer.truth_hint)).toBeNull()
  expect(screen.queryByText('实际听牌:', { exact: false })).toBeNull()
})

it('distinguishes tiny estimates from zero and an inapplicable candidate from zero risk', () => {
  const observer = readout()
  observer.candidates[0].ron_probability = [0.000001, 0.2, 0.3]
  observer.candidates[0].expected_loss_points = 0.2
  const { rerender } = render(<ObserverDetails observer={observer} />)
  expect(screen.getByText('<0.01%')).toBeTruthy()
  expect(screen.getByText('<1')).toBeTruthy()
  rerender(<ObserverDetails observer={{ ...observer, candidates: [] }} />)
  expect(screen.queryByRole('table')).toBeNull()
  expect(screen.getByText(chinese.observer.no_applicable_candidates)).toBeTruthy()
})

it('shows the same readout only at its exact review decision and hides it after seeking', async () => {
  const observer = readout()
  const meta = { decision: true, observer,
    model_identity: { actor_sha256: observer.actor_identity, original_sha256: observer.observer_identity.representation_actor_sha256 },
    candidates: [
      { action: { type: 'none' }, probability: 0.1, selected: false },
      { action: { type: 'dahai', actor: 2, pai: '3p' }, probability: 0.2, selected: false },
      { action: { type: 'reach', actor: 2 }, continuation: { type: 'dahai', actor: 2, pai: '5sr' }, probability: 0.7, selected: true },
    ],
  }
  const result: LocalReviewResult = { history_id: 'synthetic-observer', bot: 'rin-native', created_at: '2026-09-17', seat: 2,
    event_count: 3, compared: 1, matched: 0,
    events: [{ type: 'start_kyoku', bakaze: 'E', kyoku: 1, honba: 0 }, { type: 'tsumo', actor: 2, pai: '5sr' }, { type: 'dahai', actor: 2, pai: '3p' }],
    decisions: [{ event_index: 1, round: 'E1', turn: 1, hand: ['3p', '5sr'], trigger: { type: 'tsumo', actor: 2, pai: '5sr' },
      actual: { type: 'dahai', actor: 2, pai: '3p' }, recommended: { type: 'reach', actor: 2 }, matches: false, meta }],
  }
  mocks.invoke.mockResolvedValue(Array.from({ length: 3 }, () => ({ game: {}, view: {} })))
  result.decisions[0].observer_truth = { schema_version: 'akagi.observer-truth.v1', event_index: 999,
    opponents: observer.opponents.map((item) => ({ seat: item.seat, tenpai: true, waits: ['2m'], furiten: false, reason: null })), candidates: [] }
  const { rerender } = render(<LocalReviewViewer result={result} />)
  await waitFor(() => expect((screen.getByRole('button', { name: '下一步' }) as HTMLButtonElement).disabled).toBe(false))
  expect(screen.getByRole('region', { name: chinese.observer.estimates })).toBeTruthy()
  expect(screen.getByText(chinese.observer.truth_unavailable)).toBeTruthy()
  expect(screen.queryByText('实际待牌（全部）: 二万')).toBeNull()
  rerender(<LocalReviewViewer result={{ ...result, decisions: [{ ...result.decisions[0], observer_truth: { ...result.decisions[0].observer_truth!, event_index: 1 } }] }} />)
  expect(screen.getAllByText('实际待牌（全部）: 二万')).toHaveLength(3)
  fireEvent.click(screen.getByRole('button', { name: '下一步' }))
  expect(screen.queryByRole('region', { name: chinese.observer.estimates })).toBeNull()
  fireEvent.click(screen.getByRole('button', { name: '上一步' }))
  expect(screen.getByRole('region', { name: chinese.observer.estimates })).toBeTruthy()
})
