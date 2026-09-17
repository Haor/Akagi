import { expect, it } from 'vitest'
import { tileName } from './tileName'

it('names suited, honor, red, and unknown tiles without altering other locales', () => {
  expect(['2s', '3p', '9m', '5pr', '0s', 'E', 'P', 'F', 'C', '?'].map((tile) => tileName(tile)))
    .toEqual(['二索', '三筒', '九万', '红五筒', '红五索', '东风', '白板', '发财', '红中', '未知牌'])
  expect(['1m', '5mr', 'F', 'C'].map((tile) => tileName(tile, 'zh-TW')))
    .toEqual(['一萬', '紅五萬', '發財', '紅中'])
  expect(tileName('3p', 'en')).toBe('3p')
  expect(tileName('bad')).toBe('bad')
})
