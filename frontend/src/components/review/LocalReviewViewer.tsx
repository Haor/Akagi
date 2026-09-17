import { useEffect, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { ChevronLeft, ChevronRight, Loader2, Pause, Play } from 'lucide-react'
import { GameBoard } from '@/components/GameBoard'
import { ObserverDetails } from '@/components/ObserverDetails'
import { Button } from '@/components/ui/button'
import { invoke } from '@/lib/tauri'
import { kyokuLabel } from '@/lib/format'
import { tileName } from '@/lib/tileName'
import type { GameStateSnapshot, MahgenView } from '@/types'
import type { LocalReviewResult } from '@/stores/localReviewStore'
import { parseObserver } from '@/stores/observerStore'
import { actionLabel } from './actionLabel'

type ReviewFrame = { game: GameStateSnapshot; view: MahgenView }

export function LocalReviewViewer({ result }: { result: LocalReviewResult }) {
  const { t, i18n } = useTranslation()
  const language = i18n.resolvedLanguage ?? i18n.language
  const [frames, setFrames] = useState<(ReviewFrame | null)[]>([])
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [index, setIndex] = useState(result.decisions[0]?.event_index ?? 1)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  const [differencesOnly, setDifferencesOnly] = useState(false)
  useEffect(() => {
    let cancelled = false
    invoke<(ReviewFrame | null)[]>('get_local_review_frames', { id: result.history_id })
      .then((value) => { if (!cancelled) setFrames(value) })
      .catch((reason) => { if (!cancelled) setError(String(reason)) })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [result.history_id])

  const steps = useMemo(() => frames.flatMap((frame, i) => frame ? [i] : []), [frames])
  const position = Math.max(0, steps.indexOf(index))
  const currentIndex = steps[position] ?? index
  const frame = frames[currentIndex]
  const event = result.events[currentIndex]
  const decision = result.decisions.find((item) => item.event_index === currentIndex)
  const observer = useMemo(() => parseObserver(decision?.meta, result.seat), [decision?.meta, result.seat])
  const decisions = result.decisions.filter((item) => !differencesOnly || item.matches === false)
  const rounds = result.events.flatMap((item, i) => item.type === 'start_kyoku'
    ? [{ index: i, label: `${kyokuLabel(item.bakaze ?? 'E', item.kyoku ?? 1)} · ${t('review.replay_honba', { count: item.honba ?? 0 })}` }]
    : [])
  const currentRound = [...rounds].reverse().find((round) => round.index <= currentIndex)
  const previousDecision = [...decisions].reverse().find((item) => item.event_index < currentIndex)
  const nextDecision = decisions.find((item) => item.event_index > currentIndex)

  useEffect(() => {
    if (!playing) return
    if (position >= steps.length - 1) {
      // A terminal timeout stops playback without scheduling another frame.
      const stop = window.setTimeout(() => setPlaying(false), 0)
      return () => window.clearTimeout(stop)
    }
    const timer = window.setTimeout(() => setIndex(steps[position + 1]), 650 / speed)
    return () => window.clearTimeout(timer)
  }, [playing, position, speed, steps])

  function seek(next: number) {
    setPlaying(false)
    setIndex(next)
  }

  return <section className="space-y-4" aria-label={t('review.replay_title')}>
    <div className="flex flex-wrap items-center gap-3">
      <h3 className="font-medium">{t('review.replay_title')}</h3>
      <select aria-label={t('review.replay_round')} className="rounded-md border bg-background p-2 text-sm" value={currentRound?.index ?? ''} disabled={loading || !steps.length} onChange={(e) => seek(Number(e.target.value))}>
        {rounds.map((round) => <option key={round.index} value={round.index}>{round.label}</option>)}
      </select>
      <span className="text-xs text-muted-foreground">{t('review.replay_visibility')}</span>
    </div>
    {loading && <div role="status" className="flex items-center gap-2 text-sm"><Loader2 className="h-4 w-4 animate-spin" />{t('review.replay_loading')}</div>}
    {error && <p role="alert" className="text-sm text-destructive break-words">{error}</p>}
    <div className="grid items-start gap-5 xl:grid-cols-[minmax(0,1.4fr)_minmax(280px,1fr)]">
      <div className="min-w-0 space-y-3">
        <div className="mx-auto aspect-square w-full max-w-[640px]" data-testid="review-board">
          <GameBoard game={frame?.game ?? null} view={frame?.view ?? null} />
        </div>
        <div className="flex items-center justify-between gap-3 text-sm">
          <span>{event?.actor != null ? `${t('review.local_seat', { seat: event.actor + 1 })} · ` : ''}{event ? actionLabel(event, t, language) : '—'}</span>
          <span className="shrink-0 font-mono text-muted-foreground">{steps.length ? position + 1 : 0} / {steps.length}</span>
        </div>
        <input type="range" className="w-full accent-primary" aria-label={t('review.replay_progress')} min={0} max={Math.max(0, steps.length - 1)} value={position} disabled={!steps.length} onChange={(e) => seek(steps[Number(e.target.value)])} />
        <div className="flex flex-wrap items-center justify-center gap-2">
          <Button variant="outline" size="sm" disabled={!previousDecision || !steps.length} onClick={() => previousDecision && seek(previousDecision.event_index)}>{t('review.replay_previous_decision')}</Button>
          <Button variant="outline" size="icon" aria-label={t('review.replay_previous')} disabled={position <= 0 || !steps.length} onClick={() => seek(steps[position - 1])}><ChevronLeft className="h-4 w-4" /></Button>
          <Button size="icon" aria-label={t(playing ? 'review.replay_pause' : 'review.replay_play')} disabled={!steps.length} onClick={() => { if (position >= steps.length - 1) setIndex(steps[0]); setPlaying(!playing) }}>{playing ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}</Button>
          <Button variant="outline" size="icon" aria-label={t('review.replay_next')} disabled={position >= steps.length - 1 || !steps.length} onClick={() => seek(steps[position + 1])}><ChevronRight className="h-4 w-4" /></Button>
          <Button variant="outline" size="sm" disabled={!nextDecision || !steps.length} onClick={() => nextDecision && seek(nextDecision.event_index)}>{t('review.replay_next_decision')}</Button>
          <select aria-label={t('review.replay_speed')} className="rounded-md border bg-background p-2 text-sm" value={speed} onChange={(e) => setSpeed(Number(e.target.value))}>{[0.5, 1, 2, 4].map((value) => <option key={value} value={value}>{value}×</option>)}</select>
        </div>
      </div>
      <div className="space-y-4 rounded-md border p-4 text-sm">
        <h4 className="font-medium">{t('review.local_candidates')}</h4>
        {decision ? <>
          <p>{t('review.local_turn', { turn: decision.turn })} · <span className={decision.matches === false ? 'text-amber-600' : 'text-muted-foreground'}>{t(`review.${decision.matches === null ? 'local_unknown' : decision.matches ? 'local_match' : 'local_different'}`)}</span></p>
          <div className="space-y-2 rounded bg-muted/50 p-3">
            <p>{t('review.replay_actual')}: {actionLabel(decision.actual, t, language)}</p>
            <p>{t('review.replay_recommended')}: {actionLabel(decision.recommended, t, language)}</p>
          </div>
          {[...(decision.meta?.candidates ?? [])].sort((a, b) => b.probability - a.probability).slice(0, 8).map((candidate, i) => <div className="flex justify-between gap-3" key={i}>
            <span>{candidate.selected ? '✓ ' : ''}{actionLabel(candidate.action, t, language)}{candidate.continuation?.pai ? ` → ${tileName(candidate.continuation.pai, language)}` : ''}</span>
            <span className="shrink-0 font-mono">{(candidate.probability * 100).toFixed(2)}%</span>
          </div>)}
          {observer?.status === 'ready' && <div className="border-t pt-4"><ObserverDetails observer={observer} /></div>}
        </> : <p className="text-muted-foreground">{t('review.replay_no_decision')}</p>}
      </div>
    </div>
    <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={differencesOnly} onChange={(e) => setDifferencesOnly(e.target.checked)} />{t('review.local_differences')}</label>
    <div className="max-h-80 space-y-1 overflow-y-auto rounded-md border p-2" aria-label={t('review.replay_decisions')}>
      {decisions.map((item) => <button key={item.event_index} type="button" aria-pressed={item.event_index === currentIndex} disabled={!steps.length} className={`flex w-full flex-wrap items-center justify-between gap-2 rounded px-3 py-2 text-left text-sm hover:bg-muted ${item.event_index === currentIndex ? 'bg-muted ring-1 ring-inset ring-primary/40' : ''}`} onClick={() => seek(item.event_index)}>
        <span>{item.round.replace(/^([ESWN])(\d+)/, (_, wind: string, round: string) => kyokuLabel(wind, Number(round)))} · {t('review.local_turn', { turn: item.turn })}</span>
        <span>{actionLabel(item.actual, t, language)} → {actionLabel(item.recommended, t, language)}</span>
        <span className={item.matches === false ? 'text-amber-600' : 'text-muted-foreground'}>{t(`review.${item.matches === null ? 'local_unknown' : item.matches ? 'local_match' : 'local_different'}`)}</span>
      </button>)}
      {!decisions.length && <p className="p-3 text-sm text-muted-foreground">{t('review.replay_no_differences')}</p>}
    </div>
  </section>
}
