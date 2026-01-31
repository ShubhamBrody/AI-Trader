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
      <div className="flex items-end justify-between">
        <div>
          <div className="text-xl font-semibold">Dashboard</div>
          <div className="text-sm text-slate-400">Operational status and market readiness</div>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <Card>
          <CardHeader>
            <CardTitle>Backend Health</CardTitle>
          </CardHeader>
          <CardBody className="space-y-2">
            {health.isLoading ? (
              <div className="text-sm text-slate-400">Loading…</div>
            ) : health.isError ? (
              <div className="text-sm text-rose-300">Failed to reach backend</div>
            ) : (
              <>
                <div className="flex items-center justify-between">
                  <div className="text-sm text-slate-300">Status</div>
                  <Badge tone={health.data?.status === 'ok' ? 'good' : 'warn'}>
                    {health.data?.status}
                  </Badge>
                </div>
                <div className="flex items-center justify-between">
                  <div className="text-sm text-slate-300">GPU</div>
                  <Badge tone={health.data?.gpu_available ? 'info' : 'neutral'}>
                    {health.data?.gpu_available ? health.data.gpu_name ?? 'Available' : 'Not available'}
                  </Badge>
                </div>
              </>
            )}
          </CardBody>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Market Session</CardTitle>
          </CardHeader>
          <CardBody className="space-y-2">
            {market.isLoading ? (
              <div className="text-sm text-slate-400">Loading…</div>
            ) : market.isError ? (
              <div className="text-sm text-amber-200">Endpoint not available</div>
            ) : (
              <>
                <KeyValueGrid
                  items={[
                    { label: 'Market', value: market.data?.market },
                    { label: 'Session', value: market.data?.session },
                    { label: 'IST Time', value: market.data?.ist_time },
                    { label: 'Trading Day', value: market.data?.trading_day },
                    { label: 'Can Trade', value: String(Boolean(market.data?.can_trade)) },
                  ]}
                />
                <div className="pt-2">
                  <Details title="Raw" data={market.data} />
                </div>
              </>
            )}
          </CardBody>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Safety Guard</CardTitle>
          </CardHeader>
          <CardBody className="space-y-2">
            {safety.isLoading ? (
              <div className="text-sm text-slate-400">Loading…</div>
            ) : safety.isError ? (
              <div className="text-sm text-amber-200">Endpoint not available</div>
            ) : (
              <>
                <div className="flex items-center gap-2">
                  <Badge tone={safety.data?.can_trade ? 'good' : 'warn'}>
                    {safety.data?.can_trade ? 'Trading Enabled' : 'Trading Blocked'}
                  </Badge>
                  <Badge tone={safety.data?.read_only ? 'warn' : 'good'}>
                    {safety.data?.read_only ? 'Read-only' : 'Writable'}
                  </Badge>
                </div>
                <div className="pt-2">
                  <KeyValueGrid
                    items={[
                      { label: 'Market', value: safety.data?.market },
                      { label: 'Session', value: safety.data?.session },
                      { label: 'Can Trade', value: String(Boolean(safety.data?.can_trade)) },
                      { label: 'Read Only', value: String(Boolean(safety.data?.read_only)) },
                    ]}
                  />
                </div>
                <div className="pt-2">
                  <Details title="Raw" data={safety.data} />
                </div>
              </>
            )}
          </CardBody>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Quick Links</CardTitle>
        </CardHeader>
        <CardBody className="grid grid-cols-1 gap-3 md:grid-cols-3">
          <a className="rounded-xl border border-slate-800 bg-slate-950/40 p-4 hover:bg-slate-900/60" href="/candles">
            <div className="text-sm font-semibold">Candles</div>
            <div className="text-xs text-slate-400">View chart + live candle stream</div>
          </a>
          <a className="rounded-xl border border-slate-800 bg-slate-950/40 p-4 hover:bg-slate-900/60" href="/paper">
            <div className="text-sm font-semibold">Paper Trading</div>
            <div className="text-xs text-slate-400">Execute and validate trades</div>
          </a>
          <a className="rounded-xl border border-slate-800 bg-slate-950/40 p-4 hover:bg-slate-900/60" href="/journal">
            <div className="text-sm font-semibold">Journal</div>
            <div className="text-xs text-slate-400">Audit trail for exits</div>
          </a>
        </CardBody>
      </Card>
    </>
  );
}
