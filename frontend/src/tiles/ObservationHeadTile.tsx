import { useTranslation } from 'react-i18next'
import { EyeOff } from 'lucide-react'
import { TileFrame } from '@/components/TileFrame'
import { ObserverDetails } from '@/components/ObserverDetails'
import { observerMessage, useObserverStore } from '@/stores/observerStore'
import type { Breakpoint } from './defaults'

export function ObservationHeadTile({ bp }: { bp: Breakpoint }) {
  const { t } = useTranslation()
  const message = useObserverStore(observerMessage)
  const observer = useObserverStore((state) => state.gameActive ? state.observer : null)
  return <TileFrame id="observation-head" title={t('tile.observation_head')} bp={bp}>
    {observer?.status === 'ready' ? <div className="space-y-3">
      <p className="text-xs text-muted-foreground">{t('observer.latest_estimate')}</p>
      <ObserverDetails observer={observer} />
    </div> : <div className="flex h-full items-center gap-3 text-sm text-muted-foreground" role="status">
      <EyeOff className="h-5 w-5 shrink-0" aria-hidden="true" />
      <p>{t(message)}</p>
    </div>}
  </TileFrame>
}
