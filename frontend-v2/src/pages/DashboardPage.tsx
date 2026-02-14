import { useQuery } from '@tanstack/react-query';
import { apiGet } from '@/lib/api';
import { Card, CardBody, CardHeader, CardTitle } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { KeyValueGrid } from '@/components/ui/KeyValue';
import { Details } from '@/components/ui/Details';

type Health = { status: string; gpu_available: boolean; gpu_name: string | null };

export function DashboardPage() {
  const health = useQuery({
    queryKey: ['health'],
    queryFn: () => apiGet<Health>('/health'),
  });

  const market = useQuery({
    queryKey: ['market-session'],
    queryFn: () => apiGet<any>('/api/market/status'),
    retry: 0,
  });

  const safety = useQuery({
    queryKey: ['safety'],
    queryFn: () => apiGet<any>('/api/safety/context'),
    retry: 0,
  });

  return (
    <>
      <Card>
        <CardBody className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div>
            <div className="text-lg font-semibold tracking-wide">Dashboard</div>
            <div className="text-sm text-slate-600 dark:text-slate-400">Operational status and market readiness</div>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            {health.data?.status ? (
              <Badge tone={health.data?.status === 'ok' ? 'good' : 'warn'}>Health {health.data?.status}</Badge>
            ) : health.isLoading ? (
              <Badge tone="neutral">Health…</Badge>
            ) : (
              <Badge tone="bad">Health error</Badge>
            )}
            {market.data?.session ? (
              <Badge tone={market.data?.can_trade ? 'good' : 'warn'}>
                Market {String(market.data?.session).toUpperCase()}
              </Badge>
            ) : market.isLoading ? (
              <Badge tone="neutral">Market…</Badge>
            ) : (
              <Badge tone="warn">Market n/a</Badge>
            )}
            {safety.data ? (
              <Badge tone={safety.data?.can_trade ? 'good' : 'warn'}>
                {safety.data?.can_trade ? 'Trading Enabled' : 'Trading Blocked'}
              </Badge>
            ) : safety.isLoading ? (
              <Badge tone="neutral">Safety…</Badge>
            ) : (
              <Badge tone="warn">Safety n/a</Badge>
            )}
          </div>
        </CardBody>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>System Snapshot</CardTitle>
        </CardHeader>
        <CardBody>
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
            <div className="rounded-2xl border border-slate-200 bg-white/70 p-4 dark:border-slate-800 dark:bg-slate-950/20">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-xs font-semibold text-slate-800 dark:text-slate-200">Backend</div>
                  <div className="text-[11px] text-slate-500">/health</div>
                </div>
                {health.isLoading ? (
                  <Badge tone="neutral">Loading…</Badge>
                ) : health.isError ? (
                  <Badge tone="bad">Offline</Badge>
                ) : (
                  <Badge tone={health.data?.status === 'ok' ? 'good' : 'warn'}>{health.data?.status}</Badge>
                )}
              </div>

              <div className="mt-3 grid grid-cols-2 gap-2">
                <div className="rounded-2xl border border-slate-200 bg-slate-50 p-3 dark:border-slate-800 dark:bg-slate-950/20">
                  <div className="text-[11px] text-slate-500">GPU</div>
                  <div className="mt-1 text-sm font-semibold text-slate-900 dark:text-slate-100">
                    {health.data?.gpu_available ? health.data.gpu_name ?? 'Available' : health.isError ? '—' : 'Not available'}
                  </div>
                </div>
                <div className="rounded-2xl border border-slate-200 bg-slate-50 p-3 dark:border-slate-800 dark:bg-slate-950/20">
                  <div className="text-[11px] text-slate-500">Mode</div>
                  <div className="mt-1 text-sm font-semibold text-slate-900 dark:text-slate-100">API</div>
                </div>
              </div>
            </div>

            <div className="rounded-2xl border border-slate-200 bg-white/70 p-4 dark:border-slate-800 dark:bg-slate-950/20">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-xs font-semibold text-slate-800 dark:text-slate-200">Market</div>
                  <div className="text-[11px] text-slate-500">/api/market/status</div>
                </div>
                {market.isLoading ? (
                  <Badge tone="neutral">Loading…</Badge>
                ) : market.isError ? (
                  <Badge tone="warn">Not available</Badge>
                ) : (
                  <Badge tone={market.data?.can_trade ? 'good' : 'warn'}>
                    {String(market.data?.session ?? '—').toUpperCase()}
                  </Badge>
                )}
              </div>

              <div className="mt-3">
                <KeyValueGrid
                  cols={1}
                  items={[
                    { label: 'IST Time', value: market.data?.ist_time },
                    { label: 'Trading Day', value: market.data?.trading_day },
                    { label: 'Can Trade', value: String(Boolean(market.data?.can_trade)) },
                  ]}
                />
              </div>
            </div>

            <div className="rounded-2xl border border-slate-200 bg-white/70 p-4 dark:border-slate-800 dark:bg-slate-950/20">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-xs font-semibold text-slate-800 dark:text-slate-200">Safety</div>
                  <div className="text-[11px] text-slate-500">/api/safety/context</div>
                </div>
                {safety.isLoading ? (
                  <Badge tone="neutral">Loading…</Badge>
                ) : safety.isError ? (
                  <Badge tone="warn">Not available</Badge>
                ) : (
                  <Badge tone={safety.data?.can_trade ? 'good' : 'warn'}>
                    {safety.data?.can_trade ? 'Enabled' : 'Blocked'}
                  </Badge>
                )}
              </div>

              <div className="mt-3 flex flex-wrap gap-2">
                <Badge tone={safety.data?.read_only ? 'warn' : 'good'}>
                  {safety.data?.read_only ? 'Read-only' : 'Writable'}
                </Badge>
                <Badge tone="neutral">Market {String(safety.data?.market ?? '—')}</Badge>
                <Badge tone="neutral">Session {String(safety.data?.session ?? '—')}</Badge>
              </div>

              <div className="mt-3">
                <Details title="Safety raw" data={safety.data} />
              </div>
            </div>
          </div>

          <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-2">
            <Details title="Market raw" data={market.data} />
            <Details title="Health raw" data={health.data} />
          </div>
        </CardBody>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Quick Actions</CardTitle>
        </CardHeader>
        <CardBody>
          <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
            <a
              className="group rounded-2xl border border-slate-200 bg-white/70 p-4 transition hover:bg-slate-50 dark:border-slate-800 dark:bg-slate-950/20 dark:hover:bg-slate-900/40"
              href="/candles"
            >
              <div className="text-xs text-slate-500">Charts</div>
              <div className="mt-1 text-sm font-semibold text-slate-900 dark:text-slate-100">Candles</div>
              <div className="mt-1 text-xs text-slate-600 dark:text-slate-400">Historical + WS updates</div>
              <div className="mt-3 text-[11px] text-sky-700/80 opacity-0 transition group-hover:opacity-100 dark:text-sky-200/80">Open →</div>
            </a>
            <a
              className="group rounded-2xl border border-slate-200 bg-white/70 p-4 transition hover:bg-slate-50 dark:border-slate-800 dark:bg-slate-950/20 dark:hover:bg-slate-900/40"
              href="/paper"
            >
              <div className="text-xs text-slate-500">Execution</div>
              <div className="mt-1 text-sm font-semibold text-slate-900 dark:text-slate-100">Paper Trading</div>
              <div className="mt-1 text-xs text-slate-600 dark:text-slate-400">Run trades safely</div>
              <div className="mt-3 text-[11px] text-sky-700/80 opacity-0 transition group-hover:opacity-100 dark:text-sky-200/80">Open →</div>
            </a>
            <a
              className="group rounded-2xl border border-slate-200 bg-white/70 p-4 transition hover:bg-slate-50 dark:border-slate-800 dark:bg-slate-950/20 dark:hover:bg-slate-900/40"
              href="/journal"
            >
              <div className="text-xs text-slate-500">Audit</div>
              <div className="mt-1 text-sm font-semibold text-slate-900 dark:text-slate-100">Journal</div>
              <div className="mt-1 text-xs text-slate-600 dark:text-slate-400">Exits + outcomes</div>
              <div className="mt-3 text-[11px] text-sky-700/80 opacity-0 transition group-hover:opacity-100 dark:text-sky-200/80">Open →</div>
            </a>
          </div>
        </CardBody>
      </Card>
    </>
  );
}
