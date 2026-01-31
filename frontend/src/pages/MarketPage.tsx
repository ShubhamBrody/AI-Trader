import { useQuery } from '@tanstack/react-query';
import { apiGet } from '@/lib/api';
import { Card, CardBody, CardHeader, CardTitle } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { KeyValueGrid } from '@/components/ui/KeyValue';
import { Details } from '@/components/ui/Details';

export function MarketPage() {
  const market = useQuery({
    queryKey: ['market-status'],
    queryFn: () => apiGet<any>('/api/market/status'),
    retry: 0,
    refetchInterval: 10_000,
  });

  const safety = useQuery({
    queryKey: ['safety-context'],
    queryFn: () => apiGet<any>('/api/safety/context'),
    retry: 0,
    refetchInterval: 10_000,
  });

  const canTrade = Boolean(market.data?.can_trade) && Boolean(safety.data?.can_trade);

  return (
    <>
      <div className="flex items-end justify-between">
        <div>
          <div className="text-xl font-semibold">Market</div>
          <div className="text-sm text-slate-400">Session status + safety gate</div>
        </div>
        <Badge tone={canTrade ? 'good' : 'warn'}>{canTrade ? 'Trading Enabled' : 'Trading Blocked'}</Badge>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>/api/market/status</CardTitle>
          </CardHeader>
          <CardBody>
            {market.isLoading ? (
              <div className="text-sm text-slate-400">Loading…</div>
            ) : market.isError ? (
              <div className="text-sm text-rose-300">Failed to load market status</div>
            ) : (
              <div className="space-y-3">
                <KeyValueGrid
                  items={[
                    { label: 'Market', value: market.data?.market },
                    { label: 'Session', value: market.data?.session },
                    { label: 'IST Time', value: market.data?.ist_time },
                    { label: 'Trading Day', value: market.data?.trading_day },
                    { label: 'Can Trade', value: String(Boolean(market.data?.can_trade)) },
                  ]}
                />
                <Details title="Raw" data={market.data} />
              </div>
            )}
          </CardBody>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>/api/safety/context</CardTitle>
          </CardHeader>
          <CardBody>
            {safety.isLoading ? (
              <div className="text-sm text-slate-400">Loading…</div>
            ) : safety.isError ? (
              <div className="text-sm text-rose-300">Failed to load safety context</div>
            ) : (
              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  <Badge tone={safety.data?.can_trade ? 'good' : 'warn'}>
                    {safety.data?.can_trade ? 'Trading Enabled' : 'Trading Blocked'}
                  </Badge>
                  <Badge tone={safety.data?.read_only ? 'warn' : 'good'}>
                    {safety.data?.read_only ? 'Read-only' : 'Writable'}
                  </Badge>
                </div>
                <KeyValueGrid
                  items={[
                    { label: 'Market', value: safety.data?.market },
                    { label: 'Session', value: safety.data?.session },
                    { label: 'Can Trade', value: String(Boolean(safety.data?.can_trade)) },
                    { label: 'Read Only', value: String(Boolean(safety.data?.read_only)) },
                  ]}
                />
                <Details title="Raw" data={safety.data} />
              </div>
            )}
          </CardBody>
        </Card>
      </div>
    </>
  );
}
