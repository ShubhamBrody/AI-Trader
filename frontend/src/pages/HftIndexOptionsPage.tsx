import { useEffect, useMemo, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { apiGet, apiPost } from '@/lib/api';
import { wsUrl } from '@/lib/ws';
import { Card, CardBody, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { KeyValueGrid } from '@/components/ui/KeyValue';
import { Select } from '@/components/ui/Select';
import { CandlestickChart, type CandleOverlays } from '@/components/charts/CandlestickChart';
import { useCandleStream } from '@/hooks/useCandleStream';

type HftEvent = {
  id?: number;
  ts?: number;
  channel?: string;
  type?: string;
  payload?: any;
};

type Underlying = 'NIFTY' | 'SENSEX';
type Mode = 'options' | 'futures';
type Interval = '1m' | '5m';

type Candle = {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
};

type OpenTrade = {
  id: number;
  instrument_key: string;
  side: string;
  qty: number;
  entry: number;
  stop: number;
  target: number;
  meta?: any;
};

type ChainRow = {
  strike: number;
  ce: null | { instrument_key: string; tradingsymbol?: string; ltp?: number | null };
  pe: null | { instrument_key: string; tradingsymbol?: string; ltp?: number | null };
};

function fmtPct01(v: any): string {
  const n = Number(v);
  if (!Number.isFinite(n)) return '—';
  return `${Math.round(n * 100)}%`;
}

function fmtNum(v: any, digits = 2): string {
  const n = Number(v);
  if (!Number.isFinite(n)) return '—';
  return n.toFixed(digits);
}

export function HftIndexOptionsPage() {
  const qc = useQueryClient();
  const [events, setEvents] = useState<HftEvent[]>([]);

  const [underlying, setUnderlying] = useState<Underlying>('NIFTY');
  const [mode, setMode] = useState<Mode>('options');
  const [interval, setInterval] = useState<Interval>('1m');
  const [selectedInstrumentKey, setSelectedInstrumentKey] = useState<string | null>(null);
  const [selectedTradeId, setSelectedTradeId] = useState<number | null>(null);

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

  const lastDecisions = (statusQ.data?.last_decisions ?? []) as any[];
  const latestDecision = useMemo(() => {
    if (!Array.isArray(lastDecisions) || lastDecisions.length === 0) return null;
    return lastDecisions[lastDecisions.length - 1] ?? null;
  }, [lastDecisions]);
  const aiGate = statusQ.data?.ai_gate as
    | { enabled?: boolean; min_confidence?: number; interval?: string; lookback_days?: number; horizon_steps?: number }
    | undefined;

  const trendConfluence = statusQ.data?.trend_confluence as
    | {
        enabled?: boolean;
        daily_days?: number;
        h4_days?: number;
        h1_days?: number;
        weights?: Record<string, number>;
        strategy?: string;
      }
    | undefined;

  const [brokerSwitching, setBrokerSwitching] = useState(false);
  const [brokerSwitchError, setBrokerSwitchError] = useState<string | null>(null);

  async function switchBroker(next: 'paper' | 'upstox') {
    setBrokerSwitchError(null);
    setBrokerSwitching(true);
    try {
      await apiPost<any>('/api/hft/index-options/broker', { broker: next });
      qc.invalidateQueries({ queryKey: ['hft-index-options-status'] });
      qc.invalidateQueries({ queryKey: ['hft-index-options-open-trades'] });
    } catch (e: any) {
      setBrokerSwitchError(String(e?.detail ?? e?.message ?? 'Failed to switch broker'));
    } finally {
      setBrokerSwitching(false);
    }
  }

  const chainQ = useQuery({
    queryKey: ['hft-index-options-chain', underlying],
    queryFn: () => apiGet<any>(`/api/hft/index-options/options-chain?underlying=${encodeURIComponent(underlying)}&count=7`),
    retry: 0,
    refetchInterval: 10_000,
    enabled: mode === 'options',
  });

  const futQ = useQuery({
    queryKey: ['hft-index-options-futures', underlying],
    queryFn: () => apiGet<any>(`/api/hft/index-options/futures?underlying=${encodeURIComponent(underlying)}`),
    retry: 0,
    refetchInterval: 10_000,
    enabled: mode === 'futures',
  });

  const openQ = useQuery({
    queryKey: ['hft-index-options-open-trades'],
    queryFn: () => apiGet<any>('/api/hft/index-options/open-trades?origin=hft_index_options&limit=200'),
    retry: 0,
    refetchInterval: 5000,
  });

  const trades: OpenTrade[] = (openQ.data?.trades ?? []) as OpenTrade[];
  const selectedTrade: OpenTrade | null = useMemo(() => {
    if (!selectedTradeId) return null;
    return trades.find((t) => t.id === selectedTradeId) ?? null;
  }, [selectedTradeId, trades]);

  // Auto-pick chart instrument based on mode.
  useEffect(() => {
    if (selectedTrade?.instrument_key) {
      setSelectedInstrumentKey(selectedTrade.instrument_key);
      return;
    }
    if (mode === 'futures') {
      const ik = futQ.data?.contract?.instrument_key;
      if (ik) setSelectedInstrumentKey(String(ik));
      return;
    }
    // Options: default to spot instrument_key from chain (useful even when chain has no prices)
    const spotKey = chainQ.data?.spot_instrument_key;
    if (spotKey) setSelectedInstrumentKey(String(spotKey));
  }, [mode, underlying, chainQ.data, futQ.data, selectedTrade]);

  const candlesQ = useQuery({
    queryKey: ['hft-index-options-candles', selectedInstrumentKey, interval],
    queryFn: () =>
      apiGet<Candle[]>(
        `/api/candles/historical?instrument_key=${encodeURIComponent(String(selectedInstrumentKey))}&interval=${encodeURIComponent(
          interval
        )}&limit=320`
      ),
    enabled: Boolean(selectedInstrumentKey),
    retry: 0,
    refetchInterval: 30_000,
  });

  const { lastCandle } = useCandleStream(Boolean(selectedInstrumentKey), {
    instrument_key: selectedInstrumentKey ?? undefined,
    interval,
    lookback_minutes: 30,
    poll_seconds: 2,
  });

  const series = useMemo(() => {
    const base = candlesQ.data ?? [];
    if (!selectedInstrumentKey) return base;
    if (lastCandle && lastCandle.instrument_key === selectedInstrumentKey && lastCandle.interval === interval) {
      const updated: Candle = lastCandle.candle;
      const copy = base.slice();
      const idx = copy.findIndex((c) => c.timestamp === updated.timestamp);
      if (idx >= 0) copy[idx] = updated;
      else copy.push(updated);
      return copy;
    }
    return base;
  }, [candlesQ.data, lastCandle, selectedInstrumentKey, interval]);

  const overlays: CandleOverlays | undefined = useMemo(() => {
    if (!selectedTrade) return undefined;
    return {
      entry: selectedTrade.entry,
      stop_loss: selectedTrade.stop,
      target: selectedTrade.target,
    };
  }, [selectedTrade]);

  async function updateRisk(tradeId: number, patch: { stop?: number; target?: number }) {
    await apiPost<any>(`/api/hft/index-options/trade/${tradeId}/risk`, patch);
    qc.invalidateQueries({ queryKey: ['hft-index-options-open-trades'] });
  }

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

            <div className="mt-3 flex flex-wrap items-center gap-2">
              <div className="text-xs text-slate-400">Trading mode</div>
              <div className="flex items-center gap-2">
                <Button
                  size="sm"
                  variant={broker === 'paper' ? 'primary' : 'secondary'}
                  disabled={brokerSwitching || running}
                  onClick={() => switchBroker('paper')}
                >
                  Paper
                </Button>
                <Button
                  size="sm"
                  variant={broker === 'upstox' ? 'primary' : 'secondary'}
                  disabled={brokerSwitching || running}
                  onClick={() => switchBroker('upstox')}
                    title="Requires Upstox login, SAFE_MODE=false, LIVE_TRADING_ENABLED=true"
                >
                  Live
                </Button>
              </div>
              {running ? <div className="text-[11px] text-slate-500">Stop loop to switch.</div> : null}
            </div>

            {brokerSwitchError ? (
              <div className="mt-2 text-sm text-rose-300">{brokerSwitchError}</div>
            ) : null}

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
                  { label: 'AI gate', value: aiGate?.enabled ? 'Enabled' : 'Disabled' },
                  { label: 'AI min conf', value: aiGate?.min_confidence != null ? Number(aiGate.min_confidence).toFixed(2) : '—' },
                  { label: 'AI interval', value: aiGate?.interval ?? '—' },
                  { label: 'AI lookback days', value: aiGate?.lookback_days ?? '—' },
                  { label: 'AI horizon steps', value: aiGate?.horizon_steps ?? '—' },
                  { label: 'Trend confluence', value: trendConfluence?.enabled ? 'Enabled' : 'Disabled' },
                  { label: 'Confluence 1D days', value: trendConfluence?.daily_days ?? '—' },
                  { label: 'Confluence 4H days', value: trendConfluence?.h4_days ?? '—' },
                  { label: 'Confluence 1H days', value: trendConfluence?.h1_days ?? '—' },
                  { label: 'Started ts', value: statusQ.data?.started_ts },
                  { label: 'Last cycle ts', value: statusQ.data?.last_cycle_ts },
                ]}
              />
              <div className="mt-2 text-[11px] text-slate-500">
                “Run once (force)” only affects paper mode and does not enable live trading.
              </div>
            </div>

            <div className="mt-4 rounded-xl border border-slate-200 bg-white/70 p-3 dark:border-slate-800 dark:bg-slate-950/40">
              <div className="text-xs font-semibold text-slate-700 dark:text-slate-200">Latest AI reasoning</div>

              {latestDecision ? (
                <div className="mt-2">
                  <KeyValueGrid
                    cols={2}
                    items={[
                      { label: 'Underlying', value: latestDecision?.underlying ?? '—' },
                      { label: 'Action', value: latestDecision?.action ?? '—' },
                      {
                        label: 'Result',
                        value: latestDecision?.placed?.ok
                          ? 'PLACED'
                          : latestDecision?.ok === false
                            ? 'REJECTED'
                            : '—',
                      },
                      { label: 'Confidence', value: fmtPct01(latestDecision?.confidence) },
                      { label: 'Min conf', value: latestDecision?.min_confidence != null ? fmtNum(latestDecision?.min_confidence, 2) : '—' },
                      { label: 'Reason', value: latestDecision?.reason ?? '—' },
                      { label: 'Policy', value: latestDecision?.policy?.reason ?? '—' },
                      { label: 'AI model', value: latestDecision?.ai?.model ?? '—' },
                      { label: 'AI conf', value: latestDecision?.ai?.confidence != null ? fmtPct01(latestDecision?.ai?.confidence) : '—' },
                      { label: 'AI unc', value: latestDecision?.ai?.uncertainty != null ? fmtNum(latestDecision?.ai?.uncertainty, 2) : '—' },
                      {
                        label: 'Confluence',
                        value:
                          latestDecision?.trend_confluence?.dir != null
                            ? `${String(latestDecision.trend_confluence.dir)} (${fmtPct01(latestDecision.trend_confluence.confidence)})`
                            : '—',
                      },
                      {
                        label: 'Risk mult',
                        value:
                          latestDecision?.trend_confluence_risk_multiplier != null
                            ? fmtNum(latestDecision?.trend_confluence_risk_multiplier, 2)
                            : '—',
                      },
                      { label: 'Qty', value: latestDecision?.qty ?? '—' },
                      { label: 'Entry', value: latestDecision?.option_entry != null ? fmtNum(latestDecision?.option_entry, 2) : '—' },
                      { label: 'Stop', value: latestDecision?.stop != null ? fmtNum(latestDecision?.stop, 2) : '—' },
                      { label: 'Target', value: latestDecision?.target != null ? fmtNum(latestDecision?.target, 2) : '—' },
                    ]}
                  />
                </div>
              ) : (
                <div className="mt-2 text-[11px] text-slate-500">No recent decisions.</div>
              )}

              <div className="mt-4 text-xs font-semibold text-slate-700 dark:text-slate-200">Last decisions (raw)</div>
              <pre className="mt-2 overflow-auto text-[11px] text-slate-700 dark:text-slate-300">
                {JSON.stringify(statusQ.data?.last_decisions ?? [], null, 2)}
              </pre>
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
            <CardTitle>Trading</CardTitle>
          </CardHeader>
          <CardBody>
            <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
              <div>
                <div className="mb-1 text-xs text-slate-400">Underlying</div>
                <Select value={underlying} onChange={(e) => setUnderlying(e.target.value as Underlying)}>
                  <option value="NIFTY">NIFTY</option>
                  <option value="SENSEX">SENSEX</option>
                </Select>
              </div>
              <div>
                <div className="mb-1 text-xs text-slate-400">Mode</div>
                <Select value={mode} onChange={(e) => setMode(e.target.value as Mode)}>
                  <option value="options">Options</option>
                  <option value="futures">Futures</option>
                </Select>
              </div>
              <div>
                <div className="mb-1 text-xs text-slate-400">Chart interval</div>
                <Select value={interval} onChange={(e) => setInterval(e.target.value as Interval)}>
                  <option value="1m">1m</option>
                  <option value="5m">5m</option>
                </Select>
              </div>
            </div>

            {mode === 'options' ? (
              <div className="mt-4">
                <div className="text-xs text-slate-400">Options chain (nearest expiry)</div>
                <div className="mt-2 overflow-auto rounded-xl border border-slate-800">
                  <table className="w-full text-left text-xs">
                    <thead className="bg-slate-950/40 text-slate-300">
                      <tr>
                        <th className="px-3 py-2">CE</th>
                        <th className="px-3 py-2">Strike</th>
                        <th className="px-3 py-2">PE</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800 text-slate-200">
                      {(chainQ.data?.rows as ChainRow[] | undefined)?.map((r) => (
                        <tr key={String(r.strike)}>
                          <td className="px-3 py-2">
                            {r.ce ? (
                              <button
                                className="text-sky-300 hover:text-sky-200"
                                onClick={() => setSelectedInstrumentKey(r.ce!.instrument_key)}
                                title={r.ce?.tradingsymbol ?? r.ce!.instrument_key}
                              >
                                {r.ce?.ltp ?? '-'}
                              </button>
                            ) : (
                              <span className="text-slate-600">-</span>
                            )}
                          </td>
                          <td className="px-3 py-2 text-slate-300">{r.strike}</td>
                          <td className="px-3 py-2">
                            {r.pe ? (
                              <button
                                className="text-sky-300 hover:text-sky-200"
                                onClick={() => setSelectedInstrumentKey(r.pe!.instrument_key)}
                                title={r.pe?.tradingsymbol ?? r.pe!.instrument_key}
                              >
                                {r.pe?.ltp ?? '-'}
                              </button>
                            ) : (
                              <span className="text-slate-600">-</span>
                            )}
                          </td>
                        </tr>
                      ))}
                      {!chainQ.data?.rows?.length ? (
                        <tr>
                          <td className="px-3 py-2 text-slate-500" colSpan={3}>
                            {chainQ.isLoading ? 'Loading…' : 'No chain (warm spot candles + ensure instruments imported).'}
                          </td>
                        </tr>
                      ) : null}
                    </tbody>
                  </table>
                </div>
                <div className="mt-2 text-[11px] text-slate-500">
                  Click CE/PE price to load chart for that contract.
                </div>
              </div>
            ) : (
              <div className="mt-4">
                <div className="text-xs text-slate-400">Nearest futures contract</div>
                <div className="mt-2 text-sm text-slate-200">
                  {futQ.data?.contract?.tradingsymbol ? (
                    <button
                      className="text-sky-300 hover:text-sky-200"
                      onClick={() => setSelectedInstrumentKey(String(futQ.data.contract.instrument_key))}
                      title={String(futQ.data.contract.instrument_key)}
                    >
                      {String(futQ.data.contract.tradingsymbol)}
                    </button>
                  ) : (
                    <span className="text-slate-500">No futures found (import instruments).</span>
                  )}
                </div>
              </div>
            )}
          </CardBody>
        </Card>

        <Card className="md:col-span-2">
          <CardHeader>
            <CardTitle>Chart + Positions</CardTitle>
          </CardHeader>
          <CardBody>
            <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
              <div className="lg:col-span-2">
                <div className="h-[420px] rounded-xl border border-slate-800 bg-slate-950/40">
                  {candlesQ.isLoading ? (
                    <div className="p-3 text-sm text-slate-400">Loading…</div>
                  ) : candlesQ.isError ? (
                    <div className="p-3 text-sm text-rose-300">Failed to load candles</div>
                  ) : (
                    <div className="h-full w-full p-2">
                      <CandlestickChart
                        candles={series}
                        overlays={overlays}
                        draggable={{ stop_loss: Boolean(selectedTrade), target: Boolean(selectedTrade) }}
                        onDragEnd={(key, price) => {
                          if (!selectedTrade) return;
                          if (key === 'stop_loss') updateRisk(selectedTrade.id, { stop: price });
                          if (key === 'target') updateRisk(selectedTrade.id, { target: price });
                        }}
                        watermark={selectedInstrumentKey ?? undefined}
                      />
                    </div>
                  )}
                </div>
                <div className="mt-2 text-[11px] text-slate-500">Drag SL/Target lines to update TP/SL.</div>
              </div>

              <div className="rounded-xl border border-slate-800 bg-slate-950/40 p-3">
                <div className="text-xs font-semibold text-slate-200">Open positions</div>
                <div className="mt-2 max-h-[420px] overflow-auto">
                  <table className="w-full text-left text-xs">
                    <thead className="text-slate-400">
                      <tr>
                        <th className="py-1">ID</th>
                        <th className="py-1">Side</th>
                        <th className="py-1">Entry</th>
                        <th className="py-1">SL</th>
                        <th className="py-1">TP</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800 text-slate-200">
                      {trades.map((t) => (
                        <tr
                          key={t.id}
                          className={t.id === selectedTradeId ? 'bg-slate-900/30' : ''}
                        >
                          <td className="py-2 pr-2">
                            <button className="text-sky-300 hover:text-sky-200" onClick={() => setSelectedTradeId(t.id)}>
                              {t.id}
                            </button>
                          </td>
                          <td className="py-2 pr-2">{t.side}</td>
                          <td className="py-2 pr-2">{Number(t.entry).toFixed(2)}</td>
                          <td className="py-2 pr-2">{Number(t.stop).toFixed(2)}</td>
                          <td className="py-2 pr-2">{Number(t.target).toFixed(2)}</td>
                        </tr>
                      ))}
                      {!trades.length ? (
                        <tr>
                          <td className="py-2 text-slate-500" colSpan={5}>
                            {openQ.isLoading ? 'Loading…' : 'No open positions.'}
                          </td>
                        </tr>
                      ) : null}
                    </tbody>
                  </table>
                </div>
                {selectedTrade ? (
                  <div className="mt-3 text-[11px] text-slate-500">Selected trade {selectedTrade.id}: {selectedTrade.instrument_key}</div>
                ) : (
                  <div className="mt-3 text-[11px] text-slate-500">Select a trade to show Entry/SL/TP on chart.</div>
                )}
              </div>
            </div>
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
