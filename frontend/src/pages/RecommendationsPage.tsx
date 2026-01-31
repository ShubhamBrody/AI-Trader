import { useEffect, useMemo, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { apiGet, apiPost } from '@/lib/api';
import { Card, CardBody, CardHeader, CardTitle } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';

type RefreshProgress = {
  pct: number;
  processed?: number;
  total?: number;
  phase?: string;
  message?: string;
  updated_at_utc?: string;
};

export function RecommendationsPage() {
  const qc = useQueryClient();
  const [refreshJobId, setRefreshJobId] = useState<string | null>(null);
  const [refreshError, setRefreshError] = useState<string | null>(null);
  const [refreshProgress, setRefreshProgress] = useState<RefreshProgress | null>(null);

  const q = useQuery({
    queryKey: ['recommendations'],
    queryFn: () => apiGet<any>('/api/recommendations/top?n=10&min_confidence=0.6&max_risk=0.7'),
    retry: 0,
  });

  const items = (q.data?.recommendations as any[]) ?? [];
  const meta = q.data?.meta ?? {};
  const isStale = Boolean(meta?.is_stale);
  const servedFromBackup = Boolean(meta?.served_from_backup);
  const cacheDay = typeof meta?.cache_trading_day === 'string' ? meta.cache_trading_day : null;
  const createdAt = typeof meta?.created_at_utc === 'string' ? meta.created_at_utc : null;

  const wsUrl = useMemo(() => {
    const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
    return `${proto}://${window.location.host}/api/ws/recommendations`;
  }, []);

  useEffect(() => {
    const ws = new WebSocket(wsUrl);
    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg?.channel === 'recommendations' && msg?.type === 'progress') {
          if (msg?.payload?.job_id && String(msg.payload.job_id) === String(refreshJobId ?? '')) {
            const pct = Number(msg?.payload?.pct);
            if (!Number.isNaN(pct)) {
              setRefreshProgress({
                pct,
                processed: Number(msg?.payload?.processed),
                total: Number(msg?.payload?.total),
                phase: String(msg?.payload?.phase ?? ''),
                message: String(msg?.payload?.message ?? ''),
                updated_at_utc: String(msg?.payload?.updated_at_utc ?? ''),
              });
            }
          }
        }

        if (msg?.channel === 'recommendations' && (msg?.type === 'refreshed' || msg?.type === 'refresh_failed')) {
          if (msg?.type === 'refreshed') {
            setRefreshError(null);
            setRefreshJobId(null);
            setRefreshProgress(null);
            qc.invalidateQueries({ queryKey: ['recommendations'] });
          } else {
            setRefreshJobId(null);
            setRefreshProgress(null);
            setRefreshError(String(msg?.payload?.error ?? 'Refresh failed'));
          }
        }
      } catch {
        // ignore
      }
    };
    return () => {
      try {
        ws.close();
      } catch {
        // ignore
      }
    };
  }, [qc, refreshJobId, wsUrl]);

  // On page load, detect if a refresh is already running.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const cur = await apiGet<any>('/api/recommendations/refresh/current');
        if (cancelled) return;
        if (cur?.refreshing && cur?.job_id) {
          setRefreshJobId(String(cur.job_id));
          if (cur?.progress && typeof cur.progress?.pct !== 'undefined') {
            setRefreshProgress({
              pct: Number(cur.progress.pct ?? 0),
              processed: Number(cur.progress.processed ?? 0),
              total: Number(cur.progress.total ?? 0),
                phase: String(cur.progress.phase ?? ''),
              message: String(cur.progress.message ?? ''),
              updated_at_utc: String(cur.progress.updated_at_utc ?? ''),
            });
          }
        }
      } catch {
        // ignore
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // Heartbeat poll every second while refresh is running.
  useEffect(() => {
    if (!refreshJobId) return;
    let cancelled = false;
    const id = window.setInterval(async () => {
      try {
        const st = await apiGet<any>(`/api/recommendations/refresh/status?job_id=${encodeURIComponent(refreshJobId)}`);
        if (cancelled) return;
        if (st?.ok && st?.progress) {
          setRefreshProgress({
            pct: Number(st.progress.pct ?? 0),
            processed: Number(st.progress.processed ?? 0),
            total: Number(st.progress.total ?? 0),
            phase: String(st.progress.phase ?? ''),
            message: String(st.progress.message ?? ''),
            updated_at_utc: String(st.progress.updated_at_utc ?? ''),
          });
        }
        if (st?.ok && st?.status === 'completed') {
          setRefreshJobId(null);
          setRefreshProgress(null);
          setRefreshError(null);
          qc.invalidateQueries({ queryKey: ['recommendations'] });
        }
        if (st?.ok && st?.status === 'failed') {
          setRefreshJobId(null);
          setRefreshProgress(null);
          setRefreshError(String(st?.error ?? 'Refresh failed'));
        }
      } catch {
        // ignore
      }
    }, 1000);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [qc, refreshJobId]);

  async function onRefresh(force: boolean) {
    setRefreshError(null);
    setRefreshProgress({ pct: 0, message: 'starting…' });
    try {
      const res = await apiPost<any>(
        `/api/recommendations/refresh?consent=true&n=10&min_confidence=0.6&max_risk=0.7&universe_limit=200&universe_since_days=7`
      );
      if (res?.job_id) setRefreshJobId(String(res.job_id));
      if (force) {
        // Invalidate immediately too; WS will refresh again when done.
        qc.invalidateQueries({ queryKey: ['recommendations'] });
      }
    } catch (e: any) {
      setRefreshJobId(null);
      setRefreshError(String(e?.detail ?? 'Failed to start refresh'));
    }
  }

  return (
    <>
      <div>
        <div className="text-xl font-semibold">Recommendations</div>
        <div className="text-sm text-slate-400">Top opportunities for the day</div>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <Button size="sm" onClick={() => onRefresh(true)} disabled={Boolean(refreshJobId)}>
          {refreshJobId ? 'Refreshing…' : 'Refresh recommendations'}
        </Button>
        {refreshJobId ? (
          <div className="flex items-center gap-2">
            <div className="h-2 w-40 overflow-hidden rounded bg-slate-800">
              <div
                className="h-2 bg-sky-500"
                style={{ width: `${Math.max(0, Math.min(100, Number(refreshProgress?.pct ?? 0)))}%` }}
              />
            </div>
            <div className="text-xs text-slate-300">{Math.round(Number(refreshProgress?.pct ?? 0))}%</div>
            <div className="text-xs text-slate-500">
              {(refreshProgress?.phase ? `${refreshProgress.phase}` : 'working')}
              {typeof refreshProgress?.processed === 'number' && typeof refreshProgress?.total === 'number'
                ? ` (${refreshProgress.processed}/${refreshProgress.total})`
                : ''}
              {refreshProgress?.message ? ` — ${refreshProgress.message}` : ''}
            </div>
          </div>
        ) : null}
        {servedFromBackup && !refreshJobId ? (
          <div className="text-xs text-amber-300">Refreshing… showing backup cache.</div>
        ) : isStale ? (
          <div className="text-xs text-amber-300">
            Cache is stale{cacheDay ? ` (day ${cacheDay})` : ''}. Click refresh to update.
          </div>
        ) : createdAt ? (
          <div className="text-xs text-slate-500">Cached at {new Date(createdAt).toLocaleString()}</div>
        ) : null}
        {refreshError ? <div className="text-xs text-rose-300">{refreshError}</div> : null}
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Top 10</CardTitle>
        </CardHeader>
        <CardBody>
          {q.isLoading ? (
            <div className="text-sm text-slate-400">Loading…</div>
          ) : q.isError ? (
            <div className="text-sm text-rose-300">Failed to load recommendations</div>
          ) : items.length === 0 ? (
            <div className="text-sm text-slate-400">
              No recommendations yet. If this is the first run, precompute cache using `scripts/precompute_recommendations.py` or press Refresh.
            </div>
          ) : (
            <div className="overflow-auto">
              <table className="w-full text-sm">
                <thead className="text-left text-xs text-slate-400">
                  <tr>
                    <th className="py-2">Symbol</th>
                    <th>Action</th>
                    <th>Score</th>
                    <th>Confidence</th>
                    <th className="min-w-[260px]">Reasons</th>
                  </tr>
                </thead>
                <tbody>
                  {items.map((r) => (
                    <tr key={r.instrument_key} className="border-t border-slate-800">
                      <td className="py-2">
                        <div className="font-medium text-slate-100">{r.symbol}</div>
                        <div className="text-xs text-slate-500">{r.instrument_key}</div>
                      </td>
                      <td>
                        <Badge tone={r.predicted_action === 'BUY' ? 'good' : r.predicted_action === 'SELL' ? 'bad' : 'neutral'}>
                          {r.predicted_action}
                        </Badge>
                      </td>
                      <td className="text-slate-200">{Number(r.score).toFixed(2)}</td>
                      <td className="text-slate-200">{Math.round(Number(r.confidence) * 100)}%</td>
                      <td className="text-xs text-slate-300">
                        {(r.reasons ?? []).slice(0, 3).join(' • ')}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardBody>
      </Card>
    </>
  );
}
