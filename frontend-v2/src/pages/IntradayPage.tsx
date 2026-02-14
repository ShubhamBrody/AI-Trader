import { useMemo, useState } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import { apiGet, apiPost } from '@/lib/api';
import { Card, CardBody, CardHeader, CardTitle } from '@/components/ui/Card';
import { Input } from '@/components/ui/Input';
import { Select } from '@/components/ui/Select';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { CandlestickChart } from '@/components/charts/CandlestickChart';
import { AiTraderPanel } from '@/components/ai/AiTraderPanel';
import { useInstrumentResolve } from '@/hooks/useInstrumentResolve';
import { InstrumentSearch } from '@/components/instruments/InstrumentSearch';
import { Details } from '@/components/ui/Details';
import { useAppDispatch, useAppSelector } from '@/store/hooks';
import { setInstrumentInput, setInstrumentPick } from '@/store/selectionSlice';
import { useIntradayOverlaysStream } from '@/hooks/useIntradayOverlaysStream';

type Candle = {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
};

type Interval = '1m' | '3m' | '5m' | '15m' | '30m' | '1h';

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

export function IntradayPage() {
  const dispatch = useAppDispatch();
  const instrument = useAppSelector((s) => s.selection.instrument);

  const [interval, setInterval] = useState<Interval>('1m');
  const [lookbackMinutes, setLookbackMinutes] = useState(60);
  const [showOverlays, setShowOverlays] = useState(true);
  const [useAdaptiveAlgo, setUseAdaptiveAlgo] = useState(true);
  const [indicatorPick, setIndicatorPick] = useState<string>('ema');
  const [emaPeriod, setEmaPeriod] = useState<number>(20);
  const [activeIndicators, setActiveIndicators] = useState<string[]>(['ema:20', 'ema:50', 'vwap']);

  const instrumentQuery = instrument.instrument_key ?? instrument.input;
  const inst = useInstrumentResolve(instrumentQuery);
  const resolvedKey = instrument.instrument_key ?? inst.data?.instrument_key;

  const candles = useQuery({
    queryKey: ['intraday', resolvedKey ?? instrumentQuery, interval],
    queryFn: () =>
      apiGet<Candle[]>(
        `/api/intraday?instrument_key=${encodeURIComponent(resolvedKey ?? instrumentQuery)}&interval=${encodeURIComponent(interval)}`
      ),
    enabled: !inst.isError,
    retry: 0,
  });

  const adaptive = useQuery({
    queryKey: ['adaptive-algo-intraday', resolvedKey ?? instrumentQuery, interval, useAdaptiveAlgo],
    queryFn: () =>
      apiGet<any>(
        `/api/adaptive-algo?instrument_key=${encodeURIComponent(resolvedKey ?? instrumentQuery)}&interval=${encodeURIComponent(
          interval
        )}&lookback=800&non_repainting=true&do_optimize=true`
      ),
    enabled: !inst.isError && showOverlays && useAdaptiveAlgo,
    retry: 0,
  });

  const poll = useMutation({
    mutationFn: () =>
      apiPost<any>(
        `/api/intraday/poll?instrument_key=${encodeURIComponent(resolvedKey ?? instrumentQuery)}&interval=${encodeURIComponent(
          interval
        )}&lookback_minutes=${lookbackMinutes}`
      ),
    onSuccess: () => {
      candles.refetch();
    },
  });

  const overlaysWs = useIntradayOverlaysStream({
    enabled: showOverlays && !inst.isError,
    instrument_key: resolvedKey ?? instrumentQuery,
    interval,
    lookback_minutes: Math.max(60, lookbackMinutes),
  });

  const analysis = overlaysWs.last?.analysis;

  const trendRows: Array<{ interval: string; trend: string }> = (adaptive.data?.trends ?? []) as any;

  const lineOverlays = useMemo(() => {
    if (!activeIndicators.length) return [];
    const cs = candles.data ?? [];
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
  }, [candles.data, activeIndicators]);

  const nowMs = Date.now();
  const lastMsgAgeSec = overlaysWs.state.lastMessageAt ? Math.max(0, (nowMs - overlaysWs.state.lastMessageAt) / 1000) : null;
  const asofIso = analysis?.asof_ts ? new Date((analysis.asof_ts as number) * 1000).toISOString() : null;
  const asofAgeSec = analysis?.asof_ts ? Math.max(0, (nowMs - (analysis.asof_ts as number) * 1000) / 1000) : null;

  const fmtAge = (s: number | null) => {
    if (s == null) return '—';
    if (s < 90) return `${s.toFixed(1)}s`;
    const m = s / 60;
    if (m < 90) return `${m.toFixed(1)}m`;
    const h = m / 60;
    return `${h.toFixed(1)}h`;
  };

  const pollTone = poll.data?.status === 'polled' ? 'good' : poll.data?.status === 'blocked' ? 'warn' : 'neutral';

  const fmtTs = (ts: number | null | undefined) => {
    if (!ts) return '—';
    try {
      return new Date(ts * 1000).toLocaleString();
    } catch {
      return String(ts);
    }
  };

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
      const best = pats.reduce((a, b) => (Number(b?.confidence ?? 0) > Number(a?.confidence ?? 0) ? b : a), pats[0]);
      return {
        id: String(best?.type ?? 'pattern'),
        label: String(best?.type ?? 'pattern').replaceAll('_', ' '),
        side: String(best?.side ?? 'neutral'),
        confidence: Number(best?.confidence ?? 0),
        source: 'pattern',
      };
    }
    if (candlePatternRows.length) {
      const best = candlePatternRows[0];
      return {
        id: `candle:${best.name}`,
        label: best.name,
        side: best.side ?? 'neutral',
        confidence: Number(best.confidence ?? 0),
        source: 'candle_pattern',
      };
    }
    return null;
  })();

  return (
    <>
      <div>
        <div className="text-xl font-semibold">Intraday</div>
        <div className="text-sm text-slate-400">Poll live candles (market hours) and view series</div>
        {inst.data ? (
          <div className="mt-1 text-xs text-slate-400">
            {inst.data.canonical_symbol}
            {inst.data.name ? ` — ${inst.data.name}` : ''}
            <span className="text-slate-500"> · </span>
            <span className="text-slate-500">{inst.data.tradingsymbol}</span>
          </div>
        ) : inst.isError ? (
          <div className="mt-1 text-xs text-rose-300">Unknown instrument. Use a valid symbol or instrument key.</div>
        ) : null}
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Query</CardTitle>
        </CardHeader>
        <CardBody>
          <div className="grid grid-cols-1 gap-3 md:grid-cols-4">
            <div>
              <InstrumentSearch
                value={instrument.input}
                onChange={(v) => dispatch(setInstrumentInput(v))}
                onPick={(p) => dispatch(setInstrumentPick(p))}
                label="Stock"
              />
            </div>
            <div>
              <div className="mb-1 text-xs text-slate-400">Interval</div>
              <Select value={interval} onChange={(e) => setInterval(e.target.value as Interval)}>
                <option value="1m">1m</option>
                <option value="3m">3m</option>
                <option value="5m">5m</option>
                <option value="15m">15m</option>
                <option value="30m">30m</option>
                <option value="1h">1h</option>
              </Select>
            </div>
            <div>
              <div className="mb-1 text-xs text-slate-400">Lookback Minutes</div>
              <Input
                type="number"
                min={5}
                max={240}
                value={lookbackMinutes}
                onChange={(e) => setLookbackMinutes(Number(e.target.value))}
              />
            </div>
            <div className="flex items-end gap-2">
              <Button onClick={() => poll.mutate()} disabled={poll.isPending}>
                {poll.isPending ? 'Polling…' : 'Poll Now'}
              </Button>
              {poll.data?.status ? <Badge tone={pollTone as any}>{poll.data.status}</Badge> : null}
            </div>
          </div>
          {poll.data?.reason ? <div className="mt-3 text-sm text-amber-200">{poll.data.reason}</div> : null}
        </CardBody>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Chart</CardTitle>
        </CardHeader>
        <CardBody>
          <div className="mb-3 flex flex-wrap items-center gap-2">
            <Button variant={showOverlays ? 'primary' : 'secondary'} onClick={() => setShowOverlays((v) => !v)}>
              {showOverlays ? 'Overlays: ON' : 'Overlays: OFF'}
            </Button>
            <Button
              variant={useAdaptiveAlgo ? 'primary' : 'secondary'}
              onClick={() => setUseAdaptiveAlgo((v) => !v)}
              disabled={!showOverlays}
              title="Adaptive Algo computes volume-force signals + optimized settings"
            >
              Adaptive Algo: {useAdaptiveAlgo ? 'ON' : 'OFF'}
            </Button>
            {showOverlays ? (
              <Badge tone={overlaysWs.state.connected ? 'good' : 'neutral'}>
                {overlaysWs.state.connected ? 'AI Live' : 'AI Connecting…'}
              </Badge>
            ) : null}
            {showOverlays && useAdaptiveAlgo && adaptive.data?.ok && adaptive.data?.signal?.side ? (
              <Badge
                tone={
                  String(adaptive.data.signal.side) === 'BUY'
                    ? 'good'
                    : String(adaptive.data.signal.side) === 'SELL'
                      ? 'warn'
                      : 'neutral'
                }
              >
                Algo {String(adaptive.data.signal.side)}
              </Badge>
            ) : null}
            {showOverlays && useAdaptiveAlgo && adaptive.data?.optimization?.best?.win_rate != null ? (
              <Badge
                tone={
                  Number(adaptive.data.optimization.best.win_rate) >= 0.6
                    ? 'good'
                    : Number(adaptive.data.optimization.best.win_rate) <= 0.45
                      ? 'warn'
                      : 'neutral'
                }
              >
                Optimal {Math.round(Number(adaptive.data.optimization.best.win_rate) * 100)}%
              </Badge>
            ) : null}
            {showOverlays && useAdaptiveAlgo && adaptive.data?.optimization?.best?.period != null && adaptive.data?.optimization?.best?.multiplier != null ? (
              <Badge tone="neutral">
                p={Number(adaptive.data.optimization.best.period)} · m={Number(adaptive.data.optimization.best.multiplier).toFixed(1)}
              </Badge>
            ) : null}
            {showOverlays && overlaysWs.state.connected && lastMsgAgeSec != null ? (
              <Badge tone={lastMsgAgeSec <= 2 ? 'good' : lastMsgAgeSec <= 10 ? 'neutral' : 'warn'}>WS Δ {fmtAge(lastMsgAgeSec)}</Badge>
            ) : null}
            {showOverlays && analysis?.asof_ts ? (
              <Badge tone={asofAgeSec != null && asofAgeSec <= 90 ? 'good' : asofAgeSec != null && asofAgeSec <= 15 * 60 ? 'neutral' : 'warn'}>
                Candle Δ {fmtAge(asofAgeSec)}
              </Badge>
            ) : null}
            {showOverlays && analysis?.asof_ts ? <div className="text-xs text-slate-500">asof {asofIso}</div> : null}
            {analysis?.trade?.confidence != null ? (
              <Badge tone={analysis.trade.confidence >= 0.7 ? 'good' : analysis.trade.confidence <= 0.45 ? 'warn' : 'neutral'}>
                Conf {Math.round(analysis.trade.confidence * 100)}%
              </Badge>
            ) : null}
            {analysis?.trade?.risk_profile ? <Badge tone="neutral">{analysis.trade.risk_profile}</Badge> : null}
            {analysis?.trade?.reason ? <div className="text-xs text-slate-400">{analysis.trade.reason}</div> : null}

            {showOverlays && useAdaptiveAlgo && trendRows?.length ? (
              <div className="flex flex-wrap items-center gap-2">
                <div className="text-xs text-slate-500">Trend</div>
                {trendRows.slice(0, 5).map((r) => (
                  <Badge key={r.interval} tone={r.trend === 'UP' ? 'good' : r.trend === 'DOWN' ? 'warn' : 'neutral'}>
                    {r.interval} {r.trend}
                  </Badge>
                ))}
              </div>
            ) : null}
          </div>

          <div className="mb-3 grid grid-cols-1 gap-3 md:grid-cols-4">
            <div className="md:col-span-2">
              <div className="mb-1 text-xs text-slate-600 dark:text-slate-400">Add indicator</div>
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
                  variant="secondary"
                  onClick={() => {
                    const nextId = indicatorPick === 'ema' ? `ema:${Math.max(2, Math.min(500, Math.floor(emaPeriod || 20)))}` : indicatorPick;
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

          <div className="h-[420px]">
            {candles.isLoading ? (
              <div className="text-sm text-slate-400">Loading…</div>
            ) : inst.isError ? (
              <div className="text-sm text-rose-300">Invalid instrument</div>
            ) : candles.isError ? (
              <div className="text-sm text-rose-300">Failed to load intraday candles</div>
            ) : (
              (() => {
                const rawEntry = analysis?.trade?.entry;
                const rawStop = analysis?.trade?.stop;
                const rawTarget = analysis?.trade?.target;
                const side = String(analysis?.trade?.side ?? '').toLowerCase();

                const algoSig = adaptive.data?.signal;
                const algoSide = String(algoSig?.side ?? '').toLowerCase();
                const algoEntry = typeof algoSig?.entry === 'number' ? Number(algoSig.entry) : null;
                const algoStop = typeof algoSig?.stop_loss === 'number' ? Number(algoSig.stop_loss) : null;
                const algoTarget = typeof algoSig?.target === 'number' ? Number(algoSig.target) : null;

                const entry = typeof rawEntry === 'number' ? rawEntry : null;
                const stop = typeof rawStop === 'number' ? rawStop : null;
                const target = typeof rawTarget === 'number' ? rawTarget : null;

                // CandlestickChart zone overlay expects BUY-style ordering: SL < Entry < Target.
                // If the engine emits a SELL (short) plan, convert distances into a BUY-style plan
                // so the user still sees “when to buy / when to sell” zones.
                const normalized = (() => {
                  if (entry == null || stop == null || target == null) return { entry: rawEntry, stop_loss: rawStop, target: rawTarget };
                  if (side !== 'sell') return { entry, stop_loss: stop, target };

                  const dRisk = Math.abs(stop - entry);
                  const dReward = Math.abs(entry - target);
                  return {
                    entry,
                    stop_loss: entry - dReward,
                    target: entry + dRisk,
                  };
                })();

                const algoNormalized = (() => {
                  if (!useAdaptiveAlgo) return null;
                  if (!algoSide || algoSide === 'hold') return null;
                  if (algoEntry == null || algoStop == null || algoTarget == null) return null;

                  // Adaptive algo emits true BUY/SELL levels. Convert SELL to BUY-style zones.
                  if (algoSide !== 'sell') {
                    return { entry: algoEntry, stop_loss: algoStop, target: algoTarget };
                  }

                  const dRisk = Math.abs(algoStop - algoEntry);
                  const dReward = Math.abs(algoEntry - algoTarget);
                  return {
                    entry: algoEntry,
                    stop_loss: algoEntry - dReward,
                    target: algoEntry + dRisk,
                  };
                })();

                const picked = algoNormalized ?? normalized;

                return (
                  <CandlestickChart
                    candles={candles.data ?? []}
                    overlays={
                      showOverlays
                        ? {
                            supports: analysis?.levels?.support ?? [],
                            resistances: analysis?.levels?.resistance ?? [],
                            entry: picked.entry,
                            stop_loss: picked.stop_loss,
                            target: picked.target,
                          }
                        : undefined
                    }
                    lineOverlays={lineOverlays}
                    tradeZone={
                      showOverlays
                        ? {
                            enabled: true,
                            side: String(analysis?.trade?.side ?? '').toLowerCase(),
                          }
                        : { enabled: false }
                    }
                    watermark={inst.data?.canonical_symbol ?? resolvedKey ?? instrumentQuery}
                  />
                );
              })()
            )}
          </div>
        </CardBody>
      </Card>

      {resolvedKey ? (
        <Card>
          <CardHeader>
            <CardTitle>AI Trader</CardTitle>
          </CardHeader>
          <CardBody>
            <AiTraderPanel
              instrumentKey={resolvedKey}
              interval={interval}
              lookbackMinutes={lookbackMinutes}
              horizonSteps={12}
            />
          </CardBody>
        </Card>
      ) : null}

      <Card>
        <CardHeader>
          <CardTitle>Setups</CardTitle>
        </CardHeader>
        <CardBody>
          <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
            <div className="md:col-span-1 rounded-xl border border-slate-200 bg-white/70 p-3 dark:border-slate-800 dark:bg-slate-900/30">
              <div className="text-xs text-slate-600 dark:text-slate-400">Winner Strategy</div>
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
                    <div className="mt-2 text-xs text-slate-600 dark:text-slate-400">
                      Pattern hint: {winnerFallback.pattern_hint.name}
                      {winnerFallback.pattern_hint.confidence != null
                        ? ` (${Math.round(Number(winnerFallback.pattern_hint.confidence) * 100)}%)`
                        : ''}
                    </div>
                  ) : null}
                </div>
              ) : (
                <div className="mt-2 text-sm text-slate-400">No setup detected.</div>
              )}
            </div>

            <div className="md:col-span-2 rounded-xl border border-slate-200 bg-white/70 p-3 dark:border-slate-800 dark:bg-slate-900/30">
              <div className="flex items-center justify-between">
                <div className="text-xs text-slate-600 dark:text-slate-400">Candlestick Patterns (Windows)</div>
                {candlePatternRows.length ? <Badge tone="neutral">{candlePatternRows.length}</Badge> : null}
              </div>

              {candlePatternRows.length ? (
                <div className="mt-2 overflow-auto">
                  <table className="w-full border-separate border-spacing-0">
                    <thead>
                      <tr className="text-left text-[11px] text-slate-500">
                        <th className="py-1 pr-3">Start</th>
                        <th className="py-1 pr-3">End</th>
                        <th className="py-1 pr-3">Type</th>
                        <th className="py-1 pr-3">Side</th>
                        <th className="py-1 text-right">Conf</th>
                      </tr>
                    </thead>
                    <tbody>
                      {candlePatternRows.slice(0, 12).map((p, idx) => (
                        <tr key={`${p.name}-${idx}`} className="border-t border-slate-200 text-xs dark:border-slate-800/80">
                          <td className="py-2 pr-3 text-slate-600 dark:text-slate-300">{fmtTs(p.start_ts ?? null)}</td>
                          <td className="py-2 pr-3 text-slate-600 dark:text-slate-300">{fmtTs(p.end_ts ?? null)}</td>
                          <td className="py-2 pr-3 text-slate-900 dark:text-slate-100">
                            <span className="font-medium">{p.name}</span>
                            {p.window ? <span className="text-slate-500"> · w{p.window}</span> : null}
                          </td>
                          <td className="py-2 pr-3">
                            <Badge tone={p.side === 'buy' ? 'good' : p.side === 'sell' ? 'warn' : 'neutral'}>{String(p.side ?? '—')}</Badge>
                          </td>
                          <td className="py-2 text-right text-slate-700 dark:text-slate-200">
                            {p.confidence != null ? `${Math.round(Number(p.confidence) * 100)}%` : '—'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="mt-2 text-sm text-slate-400">No candlestick patterns detected in the current window.</div>
              )}
            </div>
          </div>
        </CardBody>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Overlays</CardTitle>
        </CardHeader>
        <CardBody>
          {!showOverlays ? (
            <div className="text-sm text-slate-400">Overlays are turned off.</div>
          ) : overlaysWs.state.error ? (
            <div className="text-sm text-rose-300">{overlaysWs.state.error}</div>
          ) : !analysis ? (
            <div className="text-sm text-slate-400">Waiting for live analysis…</div>
          ) : (
            <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
              <div className="rounded-xl border border-slate-200 bg-white/70 p-3 dark:border-slate-800 dark:bg-slate-900/30">
                <div className="text-xs text-slate-600 dark:text-slate-400">Trend</div>
                <div className="mt-1 text-sm">
                  <span className="font-medium">{analysis.trend?.dir ?? '—'}</span>
                  <span className="text-slate-500"> · </span>
                  <span className="text-slate-700 dark:text-slate-300">Strength {Math.round(((analysis.trend?.strength ?? 0) as number) * 100)}%</span>
                </div>
              </div>

              <div className="rounded-xl border border-slate-200 bg-white/70 p-3 dark:border-slate-800 dark:bg-slate-900/30">
                <div className="text-xs text-slate-600 dark:text-slate-400">Trade Plan</div>
                {analysis.trade ? (
                  <div className="mt-1 text-sm text-slate-800 dark:text-slate-200">
                    <div>
                      <span className="font-medium uppercase">{analysis.trade.side}</span>
                      <span className="text-slate-500"> · </span>
                      <span>Entry {analysis.trade.entry.toFixed(2)}</span>
                    </div>
                    <div className="text-xs text-slate-600 dark:text-slate-400">
                      SL {analysis.trade.stop.toFixed(2)} · Target {analysis.trade.target.toFixed(2)}
                      {analysis.trade.model_confidence != null ? ` · Model ${Math.round(analysis.trade.model_confidence * 100)}%` : ''}
                    </div>
                  </div>
                ) : (
                  <div className="mt-1 text-sm text-slate-400">No active setup.</div>
                )}
              </div>

              <div className="rounded-xl border border-slate-200 bg-white/70 p-3 dark:border-slate-800 dark:bg-slate-900/30">
                <div className="text-xs text-slate-600 dark:text-slate-400">AI</div>
                <div className="mt-1 text-sm text-slate-800 dark:text-slate-200">
                  <div>
                    {analysis.ai?.enabled ? 'Enabled' : 'Disabled'}
                    {analysis.ai?.enabled && analysis.ai?.p_up != null ? (
                      <span className="text-slate-500"> · p(up) {Math.round((analysis.ai.p_up as number) * 100)}%</span>
                    ) : null}
                  </div>
                  {analysis.ai?.enabled && analysis.ai?.model?.created_ts ? (
                    <div className="text-xs text-slate-600 dark:text-slate-400">Model ts {analysis.ai.model.created_ts}</div>
                  ) : (
                    <div className="text-xs text-slate-600 dark:text-slate-400">No model trained yet (math-only overlays).</div>
                  )}
                </div>
              </div>

              {analysis.patterns?.length ? (
                <div className="md:col-span-3 rounded-xl border border-slate-200 bg-white/70 p-3 dark:border-slate-800 dark:bg-slate-900/30">
                  <div className="text-xs text-slate-600 dark:text-slate-400">Patterns</div>
                  <div className="mt-1 flex flex-wrap gap-2">
                    {analysis.patterns.map((p, idx) => (
                      <Badge key={idx} tone={p.side === 'buy' ? 'good' : p.side === 'sell' ? 'warn' : 'neutral'}>
                        {p.type}
                        {p.confidence != null ? ` ${Math.round((p.confidence as number) * 100)}%` : ''}
                      </Badge>
                    ))}
                  </div>
                </div>
              ) : null}

              {analysis.candle_patterns?.length ? (
                <div className="md:col-span-3 rounded-xl border border-slate-200 bg-white/70 p-3 dark:border-slate-800 dark:bg-slate-900/30">
                  <div className="text-xs text-slate-600 dark:text-slate-400">Candlestick Patterns</div>
                  <div className="mt-1 flex flex-wrap gap-2">
                    {analysis.candle_patterns.slice(0, 8).map((p: any, idx: number) => (
                      <Badge key={idx} tone={p.side === 'buy' ? 'good' : p.side === 'sell' ? 'warn' : 'neutral'}>
                        {p.name}
                        {p.confidence != null ? ` ${Math.round((p.confidence as number) * 100)}%` : ''}
                        {p.details?.weight != null ? ` w${Number(p.details.weight).toFixed(2)}` : ''}
                      </Badge>
                    ))}
                  </div>
                </div>
              ) : null}
            </div>
          )}
        </CardBody>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Raw Data</CardTitle>
        </CardHeader>
        <CardBody>
          <Details title="Raw" data={candles.data ?? candles.error} />
        </CardBody>
      </Card>
    </>
  );
}
