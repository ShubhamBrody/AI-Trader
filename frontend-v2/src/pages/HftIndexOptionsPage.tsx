import { useEffect, useMemo, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { apiGet, apiPost } from '@/lib/api';
import { parseWsEnvelope, wsUrl } from '@/lib/ws';
import { Card, CardBody, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { KeyValueGrid } from '@/components/ui/KeyValue';
import { Select } from '@/components/ui/Select';
import { Input } from '@/components/ui/Input';
import { CandlestickChart, type CandleOverlays } from '@/components/charts/CandlestickChart';
import { AiTraderPanel } from '@/components/ai/AiTraderPanel';
import { useCandleStream } from '@/hooks/useCandleStream';
import { useIntradayOverlaysStream } from '@/hooks/useIntradayOverlaysStream';

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

function toUtcSeconds(iso: string): number {
  return Math.floor(new Date(iso).getTime() / 1000);
}

function emaSeries(values: number[], period: number): number[] {
  if (!values.length) return [];
  const p = Math.max(1, Math.min(period, values.length));
  const alpha = 2 / (p + 1);
  let e = values[0];
  const out = [e];
  for (const v of values.slice(1)) {
    e = alpha * v + (1 - alpha) * e;
    out.push(e);
  }
  return out;
}

function vwapSeries(typical: number[], volumes: number[]): number[] {
  const n = Math.min(typical.length, volumes.length);
  let pv = 0;
  let vv = 0;
  const out: number[] = [];
  for (let i = 0; i < n; i++) {
    pv += typical[i] * (volumes[i] || 0);
    vv += volumes[i] || 0;
    out.push(vv > 0 ? pv / vv : typical[i]);
  }
  return out;
}

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

type WinnerStrategy = {
  id: string;
  label?: string;
  side?: 'buy' | 'sell' | 'neutral' | string;
  confidence?: number;
  source?: 'trade' | 'pattern' | 'candle_pattern' | string;
  pattern_hint?: { name?: string; side?: string; confidence?: number };
};

type CandlePatternRow = {
  name: string;
  family?: string;
  side?: 'buy' | 'sell' | 'neutral' | string;
  window?: number;
  start_ts?: number | null;
  end_ts?: number | null;
  confidence?: number;
  details?: any;
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

  const [useAdaptiveAlgo, setUseAdaptiveAlgo] = useState(true);
  const [indicatorPick, setIndicatorPick] = useState<string>('ema');
  const [emaPeriod, setEmaPeriod] = useState<number>(20);
  const [activeIndicators, setActiveIndicators] = useState<string[]>(['ema:20', 'ema:50', 'vwap']);

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
        const msg = parseWsEnvelope(JSON.parse(ev.data));
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
                  {selectedInstrumentKey ? (
                    <div className="mt-2 rounded-lg border border-slate-200 bg-white/70 p-2 dark:border-slate-800 dark:bg-slate-950/40">
                      <div className="mb-2 text-[11px] font-semibold text-slate-700 dark:text-slate-300">AI Trader</div>
                      <AiTraderPanel
                        instrumentKey={selectedInstrumentKey}
                        interval={interval}
                        lookbackMinutes={240}
                        horizonSteps={12}
                      />
                    </div>
                  ) : (
                    <div className="mt-2 text-[11px] text-slate-500">Pick an instrument to see AI Trader context.</div>
                  )}

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
    if (selectedTrade) {
      return {
        entry: selectedTrade.entry,
        stop_loss: selectedTrade.stop,
        target: selectedTrade.target,
      };
    }
    return undefined;
  }, [selectedTrade]);

  const overlaysWs = useIntradayOverlaysStream({
    enabled: Boolean(selectedInstrumentKey),
    instrument_key: selectedInstrumentKey ?? '',
    interval,
    lookback_minutes: 120,
  });

  const analysis = overlaysWs.last?.analysis;

  const adaptive = useQuery({
    queryKey: ['adaptive-algo-hft', selectedInstrumentKey, interval, useAdaptiveAlgo],
    queryFn: () =>
      apiGet<any>(
        `/api/adaptive-algo?instrument_key=${encodeURIComponent(String(selectedInstrumentKey))}&interval=${encodeURIComponent(
          interval
        )}&lookback=800&non_repainting=true&do_optimize=true`
      ),
    enabled: Boolean(selectedInstrumentKey) && useAdaptiveAlgo,
    retry: 0,
    refetchInterval: 30_000,
  });

  const adaptiveSignal = adaptive.data?.signal;
  const adaptiveSide = String(adaptiveSignal?.side ?? 'HOLD');
  const adaptiveStrength = typeof adaptiveSignal?.strength === 'number' ? Number(adaptiveSignal.strength) : null;
  const best = adaptive.data?.optimization?.best;
  const bestWin = typeof best?.win_rate === 'number' ? Number(best.win_rate) : null;
  const bestTrades = typeof best?.trades === 'number' ? Number(best.trades) : null;
  const trendRows: Array<{ interval: string; trend: string }> = (adaptive.data?.trends ?? []) as any;

  const lineOverlays = useMemo(() => {
    if (!activeIndicators.length) return [];
    const cs = series ?? [];
    if (!cs.length) return [];

    const times = cs.map((c) => toUtcSeconds(c.timestamp));
    const closes = cs.map((c) => Number(c.close));
    const vols = cs.map((c) => Number(c.volume ?? 0));
    const typical = cs.map((c) => (Number(c.high) + Number(c.low) + Number(c.close)) / 3);

    const enabled = new Set(activeIndicators);
    const out: any[] = [];

    const emaIds = activeIndicators.filter((id) => id.startsWith('ema:'));
    const emaPeriods = Array.from(
      new Set(
        emaIds
          .map((id) => {
            const n = Number(id.split(':')[1]);
            return Number.isFinite(n) ? Math.max(2, Math.min(500, Math.floor(n))) : null;
          })
          .filter((x): x is number => x != null)
      )
    );

    const emaColors = ['rgba(56,189,248,0.85)', 'rgba(251,146,60,0.85)'];
    emaPeriods.forEach((p, idx) => {
      const ema = emaSeries(closes, p);
      out.push({
        id: `ema:${p}`,
        title: `EMA ${p}`,
        color: emaColors[idx % emaColors.length],
        lineWidth: 2,
        data: times.map((t, i) => ({ time: t as any, value: ema[i] })),
      });
    });

    if (enabled.has('vwap')) {
      const vwap = vwapSeries(typical, vols);
      out.push({
        id: 'vwap',
        title: 'VWAP',
        color: 'rgba(148,163,184,0.75)',
        lineWidth: 2,
        lineStyle: 2,
        data: times.map((t, i) => ({ time: t as any, value: vwap[i] })),
      });
    }

    return out;
  }, [series, activeIndicators]);

  const pickedOverlays: CandleOverlays | undefined = useMemo(() => {
    if (overlays) {
      return {
        ...overlays,
        supports: analysis?.levels?.support ?? [],
        resistances: analysis?.levels?.resistance ?? [],
      };
    }
    if (analysis?.trade?.entry != null && analysis?.trade?.stop != null && analysis?.trade?.target != null) {
      return {
        supports: analysis?.levels?.support ?? [],
        resistances: analysis?.levels?.resistance ?? [],
        entry: Number(analysis.trade.entry),
        stop_loss: Number(analysis.trade.stop),
        target: Number(analysis.trade.target),
      };
    }
    if (analysis?.levels?.support || analysis?.levels?.resistance) {
      return {
        supports: analysis?.levels?.support ?? [],
        resistances: analysis?.levels?.resistance ?? [],
      };
    }
    return undefined;
  }, [overlays, analysis]);

  const tradeZoneSide = selectedTrade?.side ?? analysis?.trade?.side;

  const candlePatternRows: CandlePatternRow[] = (analysis?.candle_patterns ?? []) as any;
  const winner: WinnerStrategy | null = (analysis?.winner_strategy ?? null) as any;

  const winnerFallback: WinnerStrategy | null = (() => {
    if (winner) return winner;
    if (analysis?.trade) {
      return {
        id: String(analysis.trade.reason ?? 'trade_plan'),
        label: String(analysis.trade.reason ?? 'trade_plan').replaceAll('_', ' '),
        side: String(analysis.trade.side ?? 'neutral').toLowerCase(),
        confidence: Number(analysis.trade.confidence ?? 0),
        source: 'trade',
        pattern_hint: { name: analysis.trade.pattern_hint },
      };
    }
    const pats: any[] = (analysis?.patterns ?? []) as any;
    if (pats.length) {
      const bestP = pats.reduce((a, b) => (Number(b?.confidence ?? 0) > Number(a?.confidence ?? 0) ? b : a), pats[0]);
      return {
        id: String(bestP?.type ?? 'pattern'),
        label: String(bestP?.type ?? 'pattern').replaceAll('_', ' '),
        side: String(bestP?.side ?? 'neutral'),
        confidence: Number(bestP?.confidence ?? 0),
        source: 'pattern',
      };
    }
    if (candlePatternRows.length) {
      const bestC = candlePatternRows[0];
      return {
        id: `candle:${bestC.name}`,
        label: bestC.name,
        side: bestC.side ?? 'neutral',
        confidence: Number(bestC.confidence ?? 0),
        source: 'candle_pattern',
      };
    }
    return null;
  })();

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

            <div className="mt-4 rounded-xl border border-slate-200 bg-white/70 p-3 dark:border-slate-800 dark:bg-slate-950/40">
              <div className="text-xs font-semibold text-slate-700 dark:text-slate-200">Calibration</div>
              <pre className="mt-2 overflow-auto text-[11px] text-slate-700 dark:text-slate-300">
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
                <div className="mb-1 text-xs text-slate-600 dark:text-slate-400">Underlying</div>
                <Select value={underlying} onChange={(e) => setUnderlying(e.target.value as Underlying)}>
                  <option value="NIFTY">NIFTY</option>
                  <option value="SENSEX">SENSEX</option>
                </Select>
              </div>
              <div>
                <div className="mb-1 text-xs text-slate-600 dark:text-slate-400">Mode</div>
                <Select value={mode} onChange={(e) => setMode(e.target.value as Mode)}>
                  <option value="options">Options</option>
                  <option value="futures">Futures</option>
                </Select>
              </div>
              <div>
                <div className="mb-1 text-xs text-slate-600 dark:text-slate-400">Chart interval</div>
                <Select value={interval} onChange={(e) => setInterval(e.target.value as Interval)}>
                  <option value="1m">1m</option>
                  <option value="5m">5m</option>
                </Select>
              </div>
            </div>

            {mode === 'options' ? (
              <div className="mt-4">
                <div className="text-xs text-slate-600 dark:text-slate-400">Options chain (nearest expiry)</div>
                <div className="mt-2 overflow-auto rounded-xl border border-slate-200 dark:border-slate-800">
                  <table className="w-full text-left text-xs">
                    <thead className="bg-slate-50 text-slate-700 dark:bg-slate-950/40 dark:text-slate-300">
                      <tr>
                        <th className="px-3 py-2">CE</th>
                        <th className="px-3 py-2">Strike</th>
                        <th className="px-3 py-2">PE</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-200 text-slate-700 dark:divide-slate-800 dark:text-slate-200">
                      {(chainQ.data?.rows as ChainRow[] | undefined)?.map((r) => (
                        <tr key={String(r.strike)}>
                          <td className="px-3 py-2">
                            {r.ce ? (
                              <button
                                className="text-sky-700 hover:text-sky-800 dark:text-sky-300 dark:hover:text-sky-200"
                                onClick={() => setSelectedInstrumentKey(r.ce!.instrument_key)}
                                title={r.ce?.tradingsymbol ?? r.ce!.instrument_key}
                              >
                                {r.ce?.ltp ?? '-'}
                              </button>
                            ) : (
                              <span className="text-slate-600">-</span>
                            )}
                          </td>
                          <td className="px-3 py-2 text-slate-700 dark:text-slate-300">{r.strike}</td>
                          <td className="px-3 py-2">
                            {r.pe ? (
                              <button
                                className="text-sky-700 hover:text-sky-800 dark:text-sky-300 dark:hover:text-sky-200"
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
                <div className="mb-3 flex flex-wrap items-center gap-2">
                  <Button
                    variant={useAdaptiveAlgo ? 'primary' : 'secondary'}
                    size="sm"
                    onClick={() => setUseAdaptiveAlgo((v) => !v)}
                    disabled={!selectedInstrumentKey}
                    title="Adaptive Algo computes volume-force signals + optimized settings"
                  >
                    Adaptive Algo: {useAdaptiveAlgo ? 'ON' : 'OFF'}
                  </Button>
                  {useAdaptiveAlgo && adaptive.data?.ok && adaptiveSide !== 'HOLD' ? (
                    <Badge tone={adaptiveSide === 'BUY' ? 'good' : adaptiveSide === 'SELL' ? 'warn' : 'neutral'}>
                      Algo {adaptiveSide}
                      {adaptiveStrength != null ? ` · ${Math.round(adaptiveStrength * 100)}%` : ''}
                    </Badge>
                  ) : null}
                  {useAdaptiveAlgo && bestWin != null ? (
                    <Badge tone={bestWin >= 0.6 ? 'good' : bestWin <= 0.45 ? 'warn' : 'neutral'}>
                      Optimal {Math.round(bestWin * 100)}%
                      {bestTrades != null ? ` · trades ${bestTrades}` : ''}
                    </Badge>
                  ) : null}
                  {useAdaptiveAlgo && best?.period != null && best?.multiplier != null ? (
                    <Badge tone="neutral">p={best.period} · m={Number(best.multiplier).toFixed(1)}</Badge>
                  ) : null}
                  {useAdaptiveAlgo && trendRows?.length ? (
                    <div className="flex flex-wrap items-center gap-2">
                      <div className="text-xs text-slate-500">Trend</div>
                      {trendRows.slice(0, 5).map((r) => (
                        <Badge key={r.interval} tone={r.trend === 'UP' ? 'good' : r.trend === 'DOWN' ? 'warn' : 'neutral'}>
                          {r.interval} {r.trend}
                        </Badge>
                      ))}
                    </div>
                  ) : null}
                  <Badge tone={overlaysWs.state.connected ? 'good' : 'neutral'}>
                    AI {overlaysWs.state.connected ? 'Live' : 'Connecting…'}
                  </Badge>
                </div>

                <div className="mb-3 grid grid-cols-1 gap-3 md:grid-cols-4">
                  <div className="md:col-span-2">
                    <div className="mb-1 text-xs text-slate-400">Add indicator</div>
                    <div className="flex items-center gap-2">
                      <Select value={indicatorPick} onChange={(e) => setIndicatorPick(e.target.value)}>
                        <option value="ema">EMA (custom)</option>
                        <option value="vwap">VWAP</option>
                      </Select>
                      {indicatorPick === 'ema' ? (
                        <Input
                          type="number"
                          min={2}
                          max={500}
                          value={emaPeriod}
                          onChange={(e) => setEmaPeriod(Math.max(2, Math.min(500, Number(e.target.value) || 20)))}
                          title="EMA period"
                        />
                      ) : null}
                      <Button
                        size="sm"
                        variant="secondary"
                        onClick={() => {
                          const nextId =
                            indicatorPick === 'ema'
                              ? `ema:${Math.max(2, Math.min(500, Math.floor(emaPeriod || 20)))}`
                              : indicatorPick;
                          setActiveIndicators((cur) => (cur.includes(nextId) ? cur : [...cur, nextId]));
                        }}
                        title="Add selected indicator"
                      >
                        Add
                      </Button>
                    </div>
                  </div>
                  <div className="md:col-span-2">
                    <div className="mb-1 text-xs text-slate-600 dark:text-slate-400">Active indicators</div>
                    {activeIndicators.length ? (
                      <div className="flex flex-wrap items-center gap-2">
                        {activeIndicators.map((id) => (
                          <div
                            key={id}
                            className="flex items-center gap-2 rounded-lg border border-slate-200 bg-white/70 px-2 py-1 dark:border-slate-800 dark:bg-slate-900/30"
                          >
                            <div className="text-xs text-slate-700 dark:text-slate-200">
                              {id.startsWith('ema:') ? `EMA ${id.split(':')[1]}` : id === 'vwap' ? 'VWAP' : id}
                            </div>
                            <button
                              type="button"
                              className="rounded-md border border-slate-200 bg-slate-50 px-2 py-0.5 text-[11px] text-slate-700 hover:bg-slate-100 dark:border-slate-700 dark:bg-slate-800/60 dark:text-slate-200 dark:hover:bg-slate-800"
                              onClick={() => setActiveIndicators((cur) => cur.filter((x) => x !== id))}
                              title="Remove"
                            >
                              Delete
                            </button>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="text-xs text-slate-500">No indicators active.</div>
                    )}
                  </div>
                </div>

                <div className="h-[420px] rounded-xl border border-slate-200 bg-white/70 dark:border-slate-800 dark:bg-slate-950/40">
                  {candlesQ.isLoading ? (
                    <div className="p-3 text-sm text-slate-400">Loading…</div>
                  ) : candlesQ.isError ? (
                    <div className="p-3 text-sm text-rose-300">Failed to load candles</div>
                  ) : (
                    <div className="h-full w-full p-2">
                      <CandlestickChart
                        candles={series}
                        overlays={pickedOverlays}
                        lineOverlays={lineOverlays}
                        tradeZone={
                          pickedOverlays?.entry != null && pickedOverlays?.stop_loss != null && pickedOverlays?.target != null
                            ? { enabled: true, side: String(tradeZoneSide ?? '').toLowerCase() }
                            : { enabled: false }
                        }
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

              <div className="rounded-xl border border-slate-200 bg-white/70 p-3 dark:border-slate-800 dark:bg-slate-950/40">
                <div className="text-xs font-semibold text-slate-700 dark:text-slate-200">Open positions</div>
                <div className="mt-2 max-h-[420px] overflow-auto">
                  <table className="w-full text-left text-xs">
                    <thead className="text-slate-600 dark:text-slate-400">
                      <tr>
                        <th className="py-1">ID</th>
                        <th className="py-1">Side</th>
                        <th className="py-1">Entry</th>
                        <th className="py-1">SL</th>
                        <th className="py-1">TP</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-200 text-slate-700 dark:divide-slate-800 dark:text-slate-200">
                      {trades.map((t) => (
                        <tr
                          key={t.id}
                          className={t.id === selectedTradeId ? 'bg-slate-100 dark:bg-slate-900/30' : ''}
                        >
                          <td className="py-2 pr-2">
                            <button
                              className="text-sky-700 hover:text-sky-800 dark:text-sky-300 dark:hover:text-sky-200"
                              onClick={() => setSelectedTradeId(t.id)}
                            >
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

                <div className="mt-4 rounded-xl border border-slate-200 bg-white/70 p-3 dark:border-slate-800 dark:bg-slate-900/20">
                  <div className="text-xs font-semibold text-slate-700 dark:text-slate-200">Algo / Patterns</div>
                  {winnerFallback ? (
                    <div className="mt-2">
                      <div className="flex flex-wrap items-center gap-2">
                        <Badge tone={winnerFallback.side === 'buy' ? 'good' : winnerFallback.side === 'sell' ? 'warn' : 'neutral'}>
                          {String(winnerFallback.side ?? 'neutral').toUpperCase()}
                        </Badge>
                        <div className="text-sm font-semibold text-slate-900 dark:text-slate-100">{winnerFallback.label ?? winnerFallback.id}</div>
                        {winnerFallback.confidence != null ? (
                          <Badge tone={Number(winnerFallback.confidence) >= 0.7 ? 'good' : Number(winnerFallback.confidence) <= 0.45 ? 'warn' : 'neutral'}>
                            {Math.round(Number(winnerFallback.confidence) * 100)}%
                          </Badge>
                        ) : null}
                      </div>
                      <div className="mt-2 text-xs text-slate-600 dark:text-slate-400">Source: {winnerFallback.source ?? '—'}</div>
                      {winnerFallback.pattern_hint?.name ? (
                        <div className="mt-2 text-xs text-slate-600 dark:text-slate-400">Pattern hint: {winnerFallback.pattern_hint.name}</div>
                      ) : null}
                    </div>
                  ) : (
                    <div className="mt-2 text-sm text-slate-400">No setup detected.</div>
                  )}

                  {candlePatternRows.length ? (
                    <div className="mt-3">
                      <div className="text-xs text-slate-600 dark:text-slate-400">Candle patterns</div>
                      <div className="mt-2 max-h-[160px] overflow-auto rounded-lg border border-slate-200 dark:border-slate-800">
                        <table className="w-full text-left text-xs">
                          <thead className="bg-slate-50 text-slate-700 dark:bg-slate-950/40 dark:text-slate-300">
                            <tr>
                              <th className="px-2 py-1">Name</th>
                              <th className="px-2 py-1">Side</th>
                              <th className="px-2 py-1">Conf</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-slate-200 text-slate-700 dark:divide-slate-800 dark:text-slate-200">
                            {candlePatternRows.slice(0, 8).map((r) => (
                              <tr key={r.name}>
                                <td className="px-2 py-1">{r.name}</td>
                                <td className="px-2 py-1">{String(r.side ?? '—')}</td>
                                <td className="px-2 py-1">{r.confidence != null ? `${Math.round(Number(r.confidence) * 100)}%` : '—'}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  ) : null}
                </div>
              </div>
            </div>
          </CardBody>
        </Card>

        <Card className="md:col-span-2">
          <CardHeader>
            <CardTitle>Live Events</CardTitle>
          </CardHeader>
          <CardBody>
            <div className="text-xs text-slate-600 dark:text-slate-400">WebSocket: /api/ws/hft</div>
            <pre className="mt-2 max-h-[360px] overflow-auto text-[11px] text-slate-700 dark:text-slate-300">
              {JSON.stringify(events, null, 2)}
            </pre>
          </CardBody>
        </Card>
      </div>
    </>
  );
}
