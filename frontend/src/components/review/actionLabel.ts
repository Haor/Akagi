import { tileName } from '@/lib/tileName'
import type { ReviewAction } from '@/stores/localReviewStore'

export function actionLabel(action: ReviewAction | null, t: (key: string) => string, language: string): string {
  if (!action) return t('review.local_unknown')
  const label = t(`review.local_actions.${action.type}`)
  const tile = action.pai ?? action.dora_marker
  return [label, tile && tileName(tile, language), action.consumed?.map((tile) => tileName(tile, language)).join('、')].filter(Boolean).join(' ')
}
