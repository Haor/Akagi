import { useEffect } from 'react'
import { useSearchParams } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { Loader2, SearchCheck } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { LocalReviewViewer } from './LocalReviewViewer'
import { useHistoryStore } from '@/stores/historyStore'
import { useLocalReviewStore } from '@/stores/localReviewStore'


export function LocalReviewPanel() {
  const { t } = useTranslation()
  const [params, setParams] = useSearchParams()
  const records = useHistoryStore((s) => s.records)
  const store = useLocalReviewStore()
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
        <LocalReviewViewer key={`${result.history_id}:${result.created_at}`} result={result} />
      </>}
    </CardContent>
  </Card>
}
