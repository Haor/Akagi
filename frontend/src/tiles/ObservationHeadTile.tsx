import { useTranslation } from 'react-i18next'
import { EyeOff } from 'lucide-react'
import { TileFrame } from '@/components/TileFrame'
import { observerMessage, useObserverStore } from '@/stores/observerStore'
import type { Breakpoint } from './defaults'

export function ObservationHeadTile({ bp }: { bp: Breakpoint }) {
  const { t } = useTranslation()
  const message = useObserverStore(observerMessage)
  return <TileFrame id="observation-head" title={t('tile.observation_head')} bp={bp}>
    <div className="flex h-full items-center gap-3 text-sm text-muted-foreground" role="status">
      <EyeOff className="h-5 w-5 shrink-0" aria-hidden="true" />
      <p>{t(message)}</p>
    </div>
  </TileFrame>
}
