import { useEffect, useMemo, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { apiGet, apiPost } from '@/lib/api';
import { wsUrl } from '@/lib/ws';
import { Card, CardBody, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { KeyValueGrid } from '@/components/ui/KeyValue';

type HftEvent = {
  id?: number;
  ts?: number;
  channel?: string;
  type?: string;
  payload?: any;
};

export function HftIndexOptionsPage() {
  const qc = useQueryClient();
  const [events, setEvents] = useState<HftEvent[]>([]);

  const statusQ = useQuery({
    queryKey: ['hft-index-options-status'],
    queryFn: () => apiGet<any>('/api/hft/index-options/status'),
    retry: 0,
    refetchInterval: 5000,
  });

  const ws = useMemo(() => wsUrl('/api/ws/hft?history=50'), []);

  useEffect(() => {
    const sock = new WebSocket(ws);
    sock.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg?.channel === 'hft') {
          setEvents((prev) => {
            const next = [msg as HftEvent, ...prev];
            return next.slice(0, 200);
          });
        }
      } catch {
        // ignore
      }
    };
    return () => {
      try {
        sock.close();
      } catch {
        // ignore
      }
    };
  }, [ws]);

  const enabled = Boolean(statusQ.data?.enabled);
  const running = Boolean(statusQ.data?.running);
  const broker = String(statusQ.data?.broker ?? 'paper');
  const openTrades = statusQ.data?.open_trades;
  const tradesToday = statusQ.data?.trades_today;
  const pnlToday = statusQ.data?.pnl_today;

  async function act(path: string) {
    await apiPost<any>(path);
    qc.invalidateQueries({ queryKey: ['hft-index-options-status'] });
  }

  return (
    <>
      <div>
        <div className="text-xl font-semibold">HFT: Index Options</div>
        <div className="text-sm text-slate-400">
          Paper-first index options loop for NIFTY and SENSEX (1m/5m)
        </div>
      </div>

      <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Status</CardTitle>
          </CardHeader>
          <CardBody>
            <div className="flex flex-wrap items-center gap-2">
              <Badge tone={enabled ? 'good' : 'warn'}>{enabled ? 'Enabled' : 'Disabled'}</Badge>
              <Badge tone={running ? 'good' : 'neutral'}>{running ? 'Running' : 'Stopped'}</Badge>
              <Badge tone={broker === 'upstox' ? 'warn' : 'neutral'}>Broker: {broker}</Badge>
            </div>

            <div className="mt-4 flex flex-wrap gap-2">
              <Button
                disabled={!enabled || running}
                onClick={() => act('/api/hft/index-options/start')}
              >
                Start
              </Button>
              <Button disabled={!running} variant="secondary" onClick={() => act('/api/hft/index-options/stop')}>
                Stop
              </Button>
              <Button
                disabled={!enabled}
                variant="secondary"
                onClick={() => act('/api/hft/index-options/run-once')}
              >
                Run once
              </Button>
              <Button
                disabled={!enabled}
                variant="secondary"
                title="Paper only: bypass market-state gate for one cycle"
                onClick={() => act('/api/hft/index-options/run-once?force=true')}
              >
                Run once (force)
              </Button>
              <Button variant="danger" onClick={() => act('/api/hft/index-options/flatten')}>
                Flatten
              </Button>
            </div>

            {statusQ.isError ? (
              <div className="mt-3 text-sm text-rose-300">
                Failed to load: {String((statusQ.error as any)?.detail ?? 'error')}
              </div>
            ) : null}

            <div className="mt-4">
              <KeyValueGrid
                cols={2}
                items={[
                  { label: 'Open trades', value: openTrades },
                  { label: 'Trades today', value: tradesToday },
                  { label: 'PnL today (INR)', value: pnlToday },
                  { label: 'Started ts', value: statusQ.data?.started_ts },
                  { label: 'Last cycle ts', value: statusQ.data?.last_cycle_ts },
                ]}
              />
              <div className="mt-2 text-[11px] text-slate-500">
                “Run once (force)” only affects paper mode and does not enable live trading.
              </div>
            </div>

            <div className="mt-4 rounded-xl border border-slate-800 bg-slate-950/40 p-3">
              <div className="text-xs font-semibold text-slate-200">Calibration</div>
              <pre className="mt-2 overflow-auto text-[11px] text-slate-300">
                {JSON.stringify(statusQ.data?.calibration ?? {}, null, 2)}
              </pre>
            </div>
          </CardBody>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Last Decisions</CardTitle>
          </CardHeader>
          <CardBody>
            <pre className="max-h-[360px] overflow-auto text-[11px] text-slate-300">
              {JSON.stringify(statusQ.data?.last_decisions ?? [], null, 2)}
            </pre>
          </CardBody>
        </Card>

        <Card className="md:col-span-2">
          <CardHeader>
            <CardTitle>Live Events</CardTitle>
          </CardHeader>
          <CardBody>
            <div className="text-xs text-slate-400">WebSocket: /api/ws/hft</div>
            <pre className="mt-2 max-h-[360px] overflow-auto text-[11px] text-slate-300">
              {JSON.stringify(events, null, 2)}
            </pre>
          </CardBody>
        </Card>
      </div>
    </>
  );
}
