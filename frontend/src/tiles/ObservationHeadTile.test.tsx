import { beforeEach, expect, it, vi } from 'vitest'
import { fireEvent, render, screen } from '@testing-library/react'
import { ObservationHeadTile } from './ObservationHeadTile'
import { useObserverStore } from '@/stores/observerStore'
import { useLayoutStore } from '@/stores/layoutStore'

vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }))
beforeEach(() => {
  useObserverStore.setState({ gameActive: false, observer: null })
})

it('uses the normal draggable tile frame, renders only state text, and can be hidden', () => {
  const { container, rerender } = render(<ObservationHeadTile bp="lg" />)
  expect(container.querySelector('.tile-drag-handle')).not.toBeNull()
  expect(screen.getByRole('status').textContent).toBe('observer.waiting')
  useObserverStore.setState({ gameActive: true, observer: { status: 'unavailable', reason: 'no_compatible_observer', actor_identity: 'a'.repeat(64), observer_identity: null } })
  rerender(<ObservationHeadTile bp="lg" />)
  expect(screen.getByRole('status').textContent).toBe('observer.unavailable')
  expect(screen.getByRole('status').textContent).not.toMatch(/[0-9%]/)
  fireEvent.click(screen.getByRole('button', { name: 'tile.hide_x' }))
  expect(useLayoutStore.getState().hidden.lg).toContain('observation-head')
})
