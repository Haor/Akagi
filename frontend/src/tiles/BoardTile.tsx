import { useTranslation } from 'react-i18next'
import { GameBoard } from '@/components/GameBoard'
import { TileFrame } from '@/components/TileFrame'
import { useGameStore } from '@/stores/gameStore'
import type { Breakpoint } from '@/tiles/defaults'

export function BoardTile({ bp }: { bp: Breakpoint }) {
  const { t } = useTranslation()
  const game = useGameStore((s) => s.game)
  const view = useGameStore((s) => s.view)
  return <TileFrame id="board" title={t('tile.board')} bp={bp} contentClassName="p-0">
    <GameBoard game={game} view={view} />
  </TileFrame>
}
