import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { DEFAULT_LAYOUTS, DEFAULT_LAYOUTS_3P, type Breakpoint } from '@/tiles/defaults'

const points: Breakpoint[] = ['lg', 'md', 'sm', 'xs']
let backing: Map<string, string>
beforeEach(() => {
  vi.resetModules()
  backing = new Map()
  vi.stubGlobal('localStorage', {
    getItem: (key: string) => backing.get(key) ?? null,
    setItem: (key: string, value: string) => void backing.set(key, value),
  })
})
afterEach(() => vi.unstubAllGlobals())

it('adds a visible observation tile without moving saved four-player or three-player layouts', async () => {
  for (const [mode, defaults] of [['4p', DEFAULT_LAYOUTS], ['3p', DEFAULT_LAYOUTS_3P]] as const) {
    const saved = Object.fromEntries(points.map((bp) => [bp, defaults[bp]
      .filter((item) => item.i !== 'observation-head')
      .map((item) => ({ ...item, y: item.y + 100 }))]))
    backing.set(`akagi.dashboard.layouts.${mode}`, JSON.stringify({ schema: 1, data: saved }))
  }
  const { useLayoutStore, visibleTilesFor } = await import('./layoutStore')
  for (const [mode, defaults] of [['4p', DEFAULT_LAYOUTS], ['3p', DEFAULT_LAYOUTS_3P]] as const) {
    useLayoutStore.getState().setMode(mode)
    for (const bp of points) {
      const { layouts, hidden } = useLayoutStore.getState()
      for (const item of defaults[bp].filter((item) => item.i !== 'observation-head')) {
        expect(layouts[bp].find((saved) => saved.i === item.i)).toMatchObject({ x: item.x, y: item.y + 100, w: item.w, h: item.h })
      }
      const tile = layouts[bp].find((item) => item.i === 'observation-head')!
      expect(tile.y).toBeGreaterThanOrEqual(Math.max(...layouts[bp].filter((item) => item.i !== tile.i).map((item) => item.y + item.h)))
      expect(visibleTilesFor(bp, hidden, mode)).toContain('observation-head')
    }
  }
})

it('keeps hide/show and moved positions persistent across reloads', async () => {
  let { useLayoutStore } = await import('./layoutStore')
  const state = useLayoutStore.getState()
  state.setLayouts({ ...state.layouts, lg: state.layouts.lg.map((item) => item.i === 'observation-head' ? { ...item, x: 6, y: 80, w: 5 } : item) })
  vi.resetModules()
  ;({ useLayoutStore } = await import('./layoutStore'))
  expect(useLayoutStore.getState().layouts.lg.find((item) => item.i === 'observation-head')).toMatchObject({ x: 6, y: 80, w: 5 })
  useLayoutStore.getState().hide('observation-head', 'lg')
  vi.resetModules()
  ;({ useLayoutStore } = await import('./layoutStore'))
  expect(useLayoutStore.getState().hidden.lg).toContain('observation-head')
  useLayoutStore.getState().show('observation-head', 'lg')
  expect(useLayoutStore.getState().hidden.lg).not.toContain('observation-head')
  expect(useLayoutStore.getState().layouts.lg.some((item) => item.i === 'observation-head')).toBe(true)
})
