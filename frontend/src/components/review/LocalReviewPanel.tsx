import { useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { Loader2, SearchCheck } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Mahgen } from '@/components/Mahgen'
import { useHistoryStore } from '@/stores/historyStore'
import { useLocalReviewStore, type LocalDecision, type ReviewAction, type LocalReviewResult } from '@/stores/localReviewStore'

function tileSeq(tiles: string[]) {
  const honors: Record<string, string> = { E: '1z', S: '2z', W: '3z', N: '4z', P: '5z', F: '6z', C: '7z' }
  return tiles.map((tile) => honors[tile] ?? tile.replace(/^5([mps])r$/, '0$1')).join('')
}

function Tiles({ tiles }: { tiles: string[] }) {
  return tiles.length ? <div className="max-w-xl"><Mahgen seq={tileSeq(tiles)} kind="hand" /></div> : <span>—</span>
}

function actionLabel(action: ReviewAction | null, t: (key: string) => string): string {
  if (!action) return t('review.local_unknown')
  const label = t(`review.local_actions.${action.type}`)
  return [label, action.pai, action.consumed?.join(' ')].filter(Boolean).join(' ')
}

function publicTable(events: ReviewAction[], index: number) {
  let scores = [25000, 25000, 25000, 25000]
  let rivers: string[][] = [[], [], [], []]
  let melds: ReviewAction[][] = [[], [], [], []]
  let dora: string[] = []
  for (const event of events.slice(0, index + 1)) {
    if (event.type === 'start_kyoku') {
      scores = [...(event.scores ?? scores)]
      rivers = [[], [], [], []]
      melds = [[], [], [], []]
      dora = event.dora_marker ? [event.dora_marker] : []
    }
    if (event.type === 'dora' && event.dora_marker) dora.push(event.dora_marker)
    if (event.type === 'dahai' && event.actor != null && event.pai) rivers[event.actor].push(event.pai)
    if (['chi', 'pon', 'daiminkan', 'ankan', 'kakan'].includes(event.type) && event.actor != null) {
      if (event.type === 'kakan') {
        const previous = melds[event.actor].findIndex((meld) => meld.type === 'pon' && meld.pai?.replace('r', '') === event.pai?.replace('r', ''))
        if (previous >= 0) melds[event.actor].splice(previous, 1)
      }
      melds[event.actor].push(event)
    }
    if (event.type === 'reach_accepted' && event.actor != null) scores[event.actor] -= 1000
    if (event.deltas) scores = scores.map((score, i) => score + (event.deltas?.[i] ?? 0))
  }
  return { scores, rivers, melds, dora }
}

function Decision({ decision, result }: { decision: LocalDecision; result: LocalReviewResult }) {
  const { t } = useTranslation()
  const [expanded, setExpanded] = useState(false)
  const status = decision.matches === null ? 'local_unknown' : decision.matches ? 'local_match' : 'local_different'
  const table = expanded ? publicTable(result.events, decision.event_index) : null
  return (
    <details className="rounded-md border p-3" onToggle={(event) => setExpanded(event.currentTarget.open)}>
      <summary className="cursor-pointer flex flex-wrap items-center gap-x-5 gap-y-2 text-sm">
        <span className="font-mono">{decision.round} · {t('review.local_turn', { turn: decision.turn })}</span>
        <span>{actionLabel(decision.actual, t)} → {actionLabel(decision.recommended, t)}</span>
        <span className={decision.matches === false ? 'text-amber-600' : 'text-muted-foreground'}>{t(`review.${status}`)}</span>
      </summary>
      {expanded && table && <div className="pt-4 space-y-4 text-sm">
        <p>{t('review.local_trigger')}: {decision.trigger.actor != null ? `${decision.trigger.actor + 1} · ` : ''}{actionLabel(decision.trigger, t)}</p>
        <div><p className="mb-2 text-muted-foreground">{t('tile.self_hand')}</p><Tiles tiles={decision.hand} /></div>
        <div><p className="mb-2 text-muted-foreground">{t('tile.dora')}</p><Tiles tiles={table.dora} /></div>
        <div className="grid gap-3 md:grid-cols-2">
          {table.rivers.map((river, seat) => <div className="rounded border p-3 space-y-2" key={seat}>
            <p>{t('review.local_seat', { seat: seat + 1 })} · {table.scores[seat]}{seat === result.seat ? ` · ${t('tile.self_hand')}` : ''}</p>
            <Tiles tiles={river} />
            {table.melds[seat].map((meld, i) => <p key={i}>{actionLabel(meld, t)}</p>)}
          </div>)}
        </div>
        <p className="text-xs text-muted-foreground">{t('review.local_river_hint')}</p>
        <div className="space-y-2">
          <p className="font-medium">{t('review.local_candidates')}</p>
          {[...(decision.meta?.candidates ?? [])].sort((a, b) => b.probability - a.probability).slice(0, 8).map((candidate, i) =>
            <div className="flex justify-between gap-3" key={i}>
              <span>{candidate.selected ? '✓ ' : ''}{actionLabel(candidate.action, t)}{candidate.continuation?.pai ? ` → ${candidate.continuation.pai}` : ''}</span>
              <span className="font-mono">{(candidate.probability * 100).toFixed(2)}%</span>
            </div>,
          )}
        </div>
      </div>}
    </details>
  )
}

export function LocalReviewPanel() {
  const { t } = useTranslation()
  const [params, setParams] = useSearchParams()
  const records = useHistoryStore((s) => s.records)
  const store = useLocalReviewStore()
  const [differencesOnly, setDifferencesOnly] = useState(false)
  const eligible = records.filter((record) => record.num_players === 4 && record.our_seat != null)
  const requested = params.get('game') ?? store.selectedId
  const selected = eligible.find((record) => record.id === requested) ?? eligible[0]
  const selectedId = selected?.id
  useEffect(() => { if (selectedId) void useLocalReviewStore.getState().open(selectedId) }, [selectedId])
  const result = store.selectedId === selected?.id ? store.result : null
  const identity = result?.decisions.find((decision) => decision.meta?.model_identity)?.meta?.model_identity
  return <Card>
    <CardHeader><CardTitle>{t('review.local_title')}</CardTitle></CardHeader>
    <CardContent className="space-y-5">
      <p className="text-sm text-muted-foreground">{t('review.local_description')}</p>
      <div className="flex flex-wrap gap-3">
        <select aria-label={t('review.games_title')} className="min-w-0 max-w-full flex-1 rounded-md border bg-background p-2 text-sm" value={selected?.id ?? ''} disabled={!!store.runningId} onChange={(event) => setParams({ game: event.target.value })}>
          {eligible.length === 0 && <option value="">{t('review.local_empty')}</option>}
          {eligible.map((record) => <option key={record.id} value={record.id}>{new Date(record.started_at).toLocaleString()} · {record.our_rank ?? '—'} · {record.id.slice(-6)}</option>)}
        </select>
        <Button disabled={!selected || !!store.runningId || store.loading} onClick={() => selected && void store.start(selected.id)}>
          {store.runningId ? <Loader2 className="h-4 w-4 animate-spin" /> : <SearchCheck className="h-4 w-4" />}
          {t(result ? 'review.local_rerun' : 'review.start_review')}
        </Button>
      </div>
      {store.runningId && <p role="status" className="text-sm">{t('review.local_running')}{store.progress ? ` · ${store.progress.processed}/${store.progress.total}` : ''}</p>}
      {store.loading && <Loader2 className="h-5 w-5 animate-spin" />}
      {store.error && <p role="alert" className="text-sm text-destructive break-words">{store.error}</p>}
      {result && <>
        <div className="space-y-1 text-sm">
          <p className="font-medium">{identity?.model_id ?? result.bot} · {t('review.local_summary', { decisions: result.decisions.length, matched: result.matched, compared: result.compared, rate: result.compared ? (100 * result.matched / result.compared).toFixed(1) : '—' })}</p>
          <p className="text-muted-foreground">{t('review.local_caveat')}</p>
          {identity?.actor_sha256 && <p className="break-all font-mono text-xs text-muted-foreground">SHA256: {identity.actor_sha256}</p>}
        </div>
        <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={differencesOnly} onChange={(event) => setDifferencesOnly(event.target.checked)} />{t('review.local_differences')}</label>
        <div className="space-y-2">{result.decisions.filter((decision) => !differencesOnly || decision.matches === false).map((decision) => <Decision key={`${result.created_at}:${decision.event_index}`} decision={decision} result={result} />)}</div>
      </>}
    </CardContent>
  </Card>
}
