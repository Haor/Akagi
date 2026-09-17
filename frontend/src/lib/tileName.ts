/** Human-readable tile names; MJAI protocol values remain unchanged. */
export function tileName(tile: string, language = 'zh-CN'): string {
  if (!language.toLowerCase().startsWith('zh')) return tile
  const traditional = /(?:tw|hk|hant)/i.test(language)
  const honors: Record<string, string> = traditional
    ? { E: '東風', S: '南風', W: '西風', N: '北風', P: '白板', F: '發財', C: '紅中', '?': '未知牌' }
    : { E: '东风', S: '南风', W: '西风', N: '北风', P: '白板', F: '发财', C: '红中', '?': '未知牌' }
  if (honors[tile]) return honors[tile]
  const match = /^([0-9])([mps])(r?)$/.exec(tile)
  if (!match || (match[3] && match[1] !== '5')) return tile
  const red = match[1] === '0' || match[3] === 'r'
  const number = red ? 5 : Number(match[1])
  const suit = { m: traditional ? '萬' : '万', p: '筒', s: '索' }[match[2]]
  return `${red ? traditional ? '紅' : '红' : ''}${'一二三四五六七八九'[number - 1]}${suit}`
}
