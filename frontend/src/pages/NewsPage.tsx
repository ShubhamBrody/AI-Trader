import { useMutation, useQuery } from '@tanstack/react-query';
import { apiGet, apiPost } from '@/lib/api';
import { Card, CardBody, CardHeader, CardTitle } from '@/components/ui/Card';
import { Details } from '@/components/ui/Details';
import { Button } from '@/components/ui/Button';

type NewsItem = {
  id: string;
  title: string;
  source: string;
  region?: string | null;
  published_at?: string;
  timestamp?: string;
  url?: string | null;
  sentiment?: number | null;
  impact_score?: number | null;
  impact_label?: 'LOW' | 'MEDIUM' | 'HIGH' | string | null;
};

export function NewsPage() {
  const q = useQuery({
    queryKey: ['news-recent'],
    queryFn: () => apiGet<any>('/api/news/recent?limit=50&days=7'),
    retry: 0,
    refetchInterval: 30_000,
  });

  const refresh = useMutation({
    mutationFn: () => apiPost<any>('/api/news/refresh?days=7&per_feed=20'),
    onSuccess: () => q.refetch(),
  });

  const items = (q.data?.news as NewsItem[]) ?? [];

  return (
    <>
      <div className="flex items-end justify-between">
        <div>
          <div className="text-xl font-semibold">News</div>
          <div className="text-sm text-slate-400">Business headlines (current week)</div>
        </div>
        <Button variant="secondary" onClick={() => refresh.mutate()} disabled={refresh.isPending}>
          {refresh.isPending ? 'Refreshing…' : 'Refresh'}
        </Button>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Recent</CardTitle>
        </CardHeader>
        <CardBody>
          {q.isLoading ? (
            <div className="text-sm text-slate-400">Loading…</div>
          ) : q.isError ? (
            <div className="text-sm text-rose-300">Failed to load news</div>
          ) : items.length === 0 ? (
            <div className="text-sm text-slate-400">No news items found.</div>
          ) : (
            <div className="space-y-3">
              <div className="grid grid-cols-1 gap-3">
                {items.map((n) => {
                  const when = n.published_at ?? n.timestamp;
                  const dt = when ? new Date(when) : null;
                  const timeLabel = dt && !Number.isNaN(dt.getTime()) ? dt.toLocaleString() : '-';
                  const href = n.url ?? undefined;

                  const sentiment = typeof n.sentiment === 'number' ? n.sentiment : null;
                  const impactLabel = n.impact_label ?? null;
                  const impactScore = typeof n.impact_score === 'number' ? n.impact_score : null;
                  const sentimentText = sentiment === null ? '-' : `${sentiment >= 0 ? '+' : ''}${sentiment.toFixed(2)}`;
                  const impactText =
                    impactLabel && impactScore !== null
                      ? `${impactLabel} (${impactScore.toFixed(2)})`
                      : impactLabel
                        ? `${impactLabel}`
                        : '-';

                  const sentimentClass =
                    sentiment === null
                      ? 'text-slate-400'
                      : sentiment > 0.15
                        ? 'text-emerald-300'
                        : sentiment < -0.15
                          ? 'text-rose-300'
                          : 'text-slate-300';
                  const impactClass =
                    impactLabel === 'HIGH'
                      ? 'text-amber-300'
                      : impactLabel === 'MEDIUM'
                        ? 'text-sky-300'
                        : 'text-slate-300';

                  const content = (
                    <div className="rounded-xl border border-slate-800 bg-slate-950/40 p-4 hover:bg-slate-900/50">
                      <div className="flex items-start justify-between gap-3">
                        <div className="min-w-0">
                          <div className="truncate text-sm font-semibold text-slate-100" title={n.title}>
                            {n.title}
                          </div>
                          <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-slate-400">
                            <span>{n.source}</span>
                            {n.region ? (
                              <>
                                <span className="text-slate-600">·</span>
                                <span>{n.region}</span>
                              </>
                            ) : null}
                            <span className="text-slate-600">·</span>
                            <span>{timeLabel}</span>
                            <span className="text-slate-600">·</span>
                            <span className={impactClass} title="Estimated impact">
                              Impact: {impactText}
                            </span>
                            <span className="text-slate-600">·</span>
                            <span className={sentimentClass} title="Estimated sentiment (-1 to +1)">
                              Sentiment: {sentimentText}
                            </span>
                          </div>
                        </div>
                        <div className="shrink-0 text-xs text-slate-500">Open</div>
                      </div>
                    </div>
                  );

                  return href ? (
                    <a key={n.id} href={href} target="_blank" rel="noreferrer" className="block">
                      {content}
                    </a>
                  ) : (
                    <div key={n.id}>{content}</div>
                  );
                })}
              </div>
              <Details title="Raw" data={q.data} />
            </div>
          )}
        </CardBody>
      </Card>
    </>
  );
}
