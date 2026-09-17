import { useTranslation } from 'react-i18next'
import { tileName } from '@/lib/tileName'
import type { ReadyObserver } from '@/stores/observerStore'

const RELATIVE_NAMES = ['mahjong.shimocha', 'mahjong.toimen', 'mahjong.kamicha'] as const
const TILES_34 = [
  ...Array.from({ length: 9 }, (_, i) => `${i + 1}m`),
  ...Array.from({ length: 9 }, (_, i) => `${i + 1}p`),
  ...Array.from({ length: 9 }, (_, i) => `${i + 1}s`),
  'E', 'S', 'W', 'N', 'P', 'F', 'C',
]

function percent(value: number) {
  return value > 0 && value < 0.0001 ? '<0.01%' : `${(value * 100).toFixed(2)}%`
}

export function ObserverDetails({ observer }: { observer: ReadyObserver }) {
  const { t, i18n } = useTranslation()
  const language = i18n.resolvedLanguage ?? i18n.language
  const formatPoints = (value: number) => value > 0 && value < 1
    ? '<1' : new Intl.NumberFormat(language, { maximumFractionDigits: 0 }).format(value)
  return <section className="space-y-4 text-sm" aria-label={t('observer.estimates')}>
    <div className="space-y-1">
      <h4 className="font-medium">{t('observer.estimates')}</h4>
      <p className="text-xs text-muted-foreground">{t('observer.estimate_hint')}</p>
    </div>
    <div className="space-y-3">
      {observer.opponents.map((opponent, i) => {
        const waits = opponent.conditional_wait_probability.map((probability, tile) => ({ probability, tile }))
          .sort((a, b) => b.probability - a.probability).slice(0, 5)
        return <article key={opponent.seat} className="space-y-2 rounded border p-3" aria-label={`${t(RELATIVE_NAMES[i])} · ${t('observer.seat', { seat: opponent.seat + 1 })}`}>
          <div className="flex flex-wrap justify-between gap-2">
            <span className="font-medium">{t(RELATIVE_NAMES[i])} · {t('observer.seat', { seat: opponent.seat + 1 })}</span>
            <span>{t('observer.tenpai')} <strong className="font-mono">{percent(opponent.tenpai_probability)}</strong></span>
          </div>
          <p className="text-xs text-muted-foreground">{t('observer.conditional_waits')}</p>
          <div className="flex flex-wrap gap-x-3 gap-y-1 text-xs">
            {waits.map(({ probability, tile }) => <span key={tile}>{tileName(TILES_34[tile], language)} <span className="font-mono">{percent(probability)}</span></span>)}
          </div>
        </article>
      })}
    </div>
    <p className="text-xs text-muted-foreground">{t('observer.wait_hint')}</p>
    <div className="space-y-2">
      <h5 className="font-medium">{t('observer.discard_estimates')}</h5>
      {observer.candidates.length === 0 ? <p className="text-muted-foreground">{t('observer.no_applicable_candidates')}</p> : <>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead><tr className="border-b text-left text-muted-foreground">
              <th className="py-2 pr-3 font-medium whitespace-nowrap">{t('observer.action')}</th>
              {observer.opponent_order.map((seat, i) => <th key={seat} className="p-2 text-right font-medium whitespace-nowrap">{t(RELATIVE_NAMES[i])}</th>)}
              <th className="p-2 text-right font-medium whitespace-nowrap">{t('observer.any_ron')}</th>
              <th className="py-2 pl-2 text-right font-medium whitespace-nowrap">{t('observer.expected_loss')}</th>
            </tr></thead>
            <tbody>{observer.candidates.map((candidate) => <tr key={candidate.candidate_index} data-candidate-index={candidate.candidate_index} className={`border-b last:border-0 ${candidate.selected ? 'bg-muted/60' : ''}`}>
              <td className="py-2 pr-3 whitespace-nowrap">
                {candidate.selected && <span title={t('observer.actor_choice')} aria-label={t('observer.actor_choice')}>✓ </span>}
                {t(candidate.action_type === 'reach' ? 'mahjong.riichi' : 'observer.discard')} {tileName(candidate.tile, language)}
              </td>
              {candidate.ron_probability.map((probability, i) => <td key={observer.opponent_order[i]} className="p-2 text-right font-mono whitespace-nowrap">{percent(probability)}</td>)}
              <td className="p-2 text-right font-mono whitespace-nowrap">{percent(candidate.any_ron_probability)}</td>
              <td className="py-2 pl-2 text-right font-mono whitespace-nowrap">{formatPoints(candidate.expected_loss_points)}</td>
            </tr>)}</tbody>
          </table>
        </div>
        <p className="text-xs text-muted-foreground">{t('observer.ron_hint')}</p>
      </>}
    </div>
  </section>
}
