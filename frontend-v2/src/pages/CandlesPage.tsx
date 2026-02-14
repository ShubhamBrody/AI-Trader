import { useMemo, useState } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import { apiGet, apiPost } from '@/lib/api';
import { Card, CardBody, CardHeader, CardTitle } from '@/components/ui/Card';
import { Input } from '@/components/ui/Input';
import { Select } from '@/components/ui/Select';
import { Badge } from '@/components/ui/Badge';
import { useCandleStream } from '@/hooks/useCandleStream';
import { Button } from '@/components/ui/Button';
import type { ApiError } from '@/lib/api';
import { useInstrumentResolve } from '@/hooks/useInstrumentResolve';
import { InstrumentSearch } from '@/components/instruments/InstrumentSearch';
import { Details } from '@/components/ui/Details';
import { CandlestickChart } from '@/components/charts/CandlestickChart';
import { AiTraderPanel } from '@/components/ai/AiTraderPanel';
import { useAppDispatch, useAppSelector } from '@/store/hooks';
import { setInstrumentInput, setInstrumentPick } from '@/store/selectionSlice';

type Candle = {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
};

type Interval = '1m' | '3m' | '5m' | '15m' | '30m' | '1h' | '1d';

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

function atrSeries(highs: number[], lows: number[], closes: number[], period: number): Array<number | null> {
  const n = Math.min(highs.length, lows.length, closes.length);
  if (n < 2) return Array.from({ length: n }, () => null);
  const p = Math.max(1, Math.min(period, n - 1));

  const tr: number[] = Array.from({ length: n }, () => 0);
  for (let i = 1; i < n; i++) {
    const h = highs[i];
    const l = lows[i];
    const pc = closes[i - 1];
    tr[i] = Math.max(h - l, Math.abs(h - pc), Math.abs(l - pc));
  }

  const out: Array<number | null> = Array.from({ length: n }, () => null);
  let s = 0;
  for (let i = 0; i < n; i++) {
    s += tr[i];
    if (i >= p) s -= tr[i - p];
    if (i >= p) out[i] = s / p;
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

export function CandlesPage() {
  const dispatch = useAppDispatch();
  const instrument = useAppSelector((s) => s.selection.instrument);

  const [interval, setInterval] = useState<Interval>('1m');
  const [limit, setLimit] = useState(200);
  const [bulkSessions, setBulkSessions] = useState(300);
  const [start, setStart] = useState('2026-01-01T09:15');
  const [end, setEnd] = useState('2026-01-30T15:29');
  const [useAdaptiveAlgo, setUseAdaptiveAlgo] = useState(true);
  const [indicatorPick, setIndicatorPick] = useState<string>('ema');
  const [emaPeriod, setEmaPeriod] = useState<number>(20);
  const [activeIndicators, setActiveIndicators] = useState<string[]>(['ema:20', 'ema:50', 'vwap']);

  const instrumentQuery = instrument.instrument_key ?? instrument.input;
  const inst = useInstrumentResolve(instrumentQuery);
  const resolvedKey = instrument.instrument_key ?? inst.data?.instrument_key;

  const toIso = (value: string) => (value ? new Date(value).toISOString() : '');

  const recommendedLimit = useMemo(() => {
    const startMs = new Date(start).getTime();
    const endMs = new Date(end).getTime();
    if (!Number.isFinite(startMs) || !Number.isFinite(endMs) || endMs <= startMs) return null;

    const intervalMinutes: Record<Interval, number> = {
      '1m': 1,
      '3m': 3,
      '5m': 5,
      '15m': 15,
      '30m': 30,
      '1h': 60,
      '1d': 24 * 60,
    };

    const mins = intervalMinutes[interval] ?? 1;
    const totalMinutes = Math.floor((endMs - startMs) / 60000);
    // Rough estimate; ignores non-trading hours/weekends.
    const estCandles = Math.max(1, Math.ceil(totalMinutes / mins));
    return Math.min(10000, estCandles);
  }, [start, end, interval]);

  const candles = useQuery({
    queryKey: ['candles', resolvedKey ?? instrumentQuery, interval, limit, start, end],
    queryFn: () =>
      apiGet<Candle[]>(
        `/api/candles/historical?instrument_key=${encodeURIComponent(resolvedKey ?? instrumentQuery)}&interval=${encodeURIComponent(
          interval
        )}&limit=${limit}&start=${encodeURIComponent(toIso(start))}&end=${encodeURIComponent(toIso(end))}`
      ),
    enabled: !inst.isError,
    retry: 0,
  });

  const load = useMutation({
    mutationFn: () =>
      apiPost<any>(
        `/api/candles/historical/load?instrument_key=${encodeURIComponent(resolvedKey ?? instrumentQuery)}&interval=${encodeURIComponent(
          interval
        )}&start=${encodeURIComponent(toIso(start))}&end=${encodeURIComponent(toIso(end))}`
      ),
    onSuccess: () => candles.refetch(),
  });

  const bulkLoad = useMutation({
    mutationFn: () =>
      apiPost<any>(
        `/api/candles/historical/bulk-load?instrument_key=${encodeURIComponent(
          resolvedKey ?? instrumentQuery
        )}&interval=${encodeURIComponent(interval)}&num_trading_sessions=${bulkSessions}`
      ),
    onSuccess: () => candles.refetch(),
  });

  const { state: streamState, lastCandle } = useCandleStream(true);

  const strategy = useQuery({
    queryKey: ['strategy-overlay', resolvedKey ?? instrumentQuery, interval, start, end, limit],
    queryFn: () =>
      apiGet<any>(
        `/api/strategy?instrument_key=${encodeURIComponent(resolvedKey ?? instrumentQuery)}&interval=${encodeURIComponent(
          interval
        )}&lookback=${Math.max(60, Math.min(limit, 5000))}&start=${encodeURIComponent(toIso(start))}&end=${encodeURIComponent(
          toIso(end)
        )}&long_only=1`
      ),
    enabled: !inst.isError,
    retry: 0,
  });

  const adaptive = useQuery({
    queryKey: ['adaptive-algo', resolvedKey ?? instrumentQuery, interval, start, end, limit, useAdaptiveAlgo],
    queryFn: () =>
      apiGet<any>(
        `/api/adaptive-algo?instrument_key=${encodeURIComponent(resolvedKey ?? instrumentQuery)}&interval=${encodeURIComponent(
          interval
        )}&lookback=${Math.max(120, Math.min(limit, 2000))}&start=${encodeURIComponent(toIso(start))}&end=${encodeURIComponent(
          toIso(end)
        )}&non_repainting=true&do_optimize=true`
      ),
    enabled: !inst.isError && useAdaptiveAlgo,
    retry: 0,
  });

  const entry = typeof strategy.data?.entry === 'number' ? Number(strategy.data.entry) : null;
  const stopLoss = typeof strategy.data?.stop_loss === 'number' ? Number(strategy.data.stop_loss) : null;
  const target = typeof strategy.data?.target === 'number' ? Number(strategy.data.target) : null;
  const rr =
    entry != null && stopLoss != null && target != null
      ? Math.abs(target - entry) / Math.max(Math.abs(entry - stopLoss), 1e-9)
      : null;

  const adaptiveSignal = adaptive.data?.signal;
  const adaptiveSide = String(adaptiveSignal?.side ?? 'HOLD');
  const adaptiveEntry = typeof adaptiveSignal?.entry === 'number' ? Number(adaptiveSignal.entry) : null;
  const adaptiveSL = typeof adaptiveSignal?.stop_loss === 'number' ? Number(adaptiveSignal.stop_loss) : null;
  const adaptiveTarget = typeof adaptiveSignal?.target === 'number' ? Number(adaptiveSignal.target) : null;
  const adaptiveStrength = typeof adaptiveSignal?.strength === 'number' ? Number(adaptiveSignal.strength) : null;
  const best = adaptive.data?.optimization?.best;
  const bestWin = typeof best?.win_rate === 'number' ? Number(best.win_rate) : null;
  const bestTrades = typeof best?.trades === 'number' ? Number(best.trades) : null;

  const trendRows: Array<{ interval: string; trend: string }> = (adaptive.data?.trends ?? []) as any;

  const series = useMemo(() => {
    const base = candles.data ?? [];
    // Optionally merge last candle update (if matches instrument/interval)
    if (lastCandle && lastCandle.instrument_key === (resolvedKey ?? instrumentQuery) && lastCandle.interval === interval) {
      const updated: Candle = lastCandle.candle;
      const copy = base.slice();
      const idx = copy.findIndex((c) => c.timestamp === updated.timestamp);
      if (idx >= 0) copy[idx] = updated;
      else copy.push(updated);
      return copy;
    }
    return base;
  }, [candles.data, instrumentQuery, resolvedKey, interval, lastCandle]);

  const chartData = useMemo(
    () => series,
    [series]
  );

  const lineOverlays = useMemo(() => {
    if (!activeIndicators.length) return [];
    const cs = chartData ?? [];
    if (!cs.length) return [];

    const times = cs.map((c) => toUtcSeconds(c.timestamp));
    const closes = cs.map((c) => Number(c.close));
    const highs = cs.map((c) => Number(c.high));
    const lows = cs.map((c) => Number(c.low));
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

    if (useAdaptiveAlgo && (enabled.has('fair') || enabled.has('bandU') || enabled.has('bandL'))) {
      const p = Number(best?.period ?? adaptive.data?.params?.period ?? 40);
      const m = Number(best?.multiplier ?? adaptive.data?.params?.multiplier ?? 2.0);
      const atrP = Number(adaptive.data?.params?.atr_period ?? 14);
      const mid = emaSeries(closes, p);
      const atr = atrSeries(highs, lows, closes, atrP);
      const upper: number[] = [];
      const lower: number[] = [];
      for (let i = 0; i < mid.length; i++) {
        const a = atr[i];
        upper.push(a == null ? mid[i] : mid[i] + m * a);
        lower.push(a == null ? mid[i] : mid[i] - m * a);
      }

      if (enabled.has('fair')) {
        out.push({
          id: 'fair',
          title: 'Fair Value',
          color: 'rgba(167,139,250,0.75)',
          lineWidth: 2,
          data: times.map((t, i) => ({ time: t as any, value: mid[i] })),
        });
      }
      if (enabled.has('bandU')) {
        out.push({
          id: 'bandU',
          title: 'Upper Band',
          color: 'rgba(34,197,94,0.45)',
          lineWidth: 1,
          lineStyle: 2,
          data: times.map((t, i) => ({ time: t as any, value: upper[i] })),
        });
      }
      if (enabled.has('bandL')) {
        out.push({
          id: 'bandL',
          title: 'Lower Band',
          color: 'rgba(34,197,94,0.45)',
          lineWidth: 1,
          lineStyle: 2,
          data: times.map((t, i) => ({ time: t as any, value: lower[i] })),
        });
      }
    }

    return out;
  }, [chartData, activeIndicators, useAdaptiveAlgo, best?.period, best?.multiplier, adaptive.data?.params?.period, adaptive.data?.params?.multiplier, adaptive.data?.params?.atr_period]);

  return (
    <>
      <div className="flex items-end justify-between">
        <div>
          <div className="text-xl font-semibold">Candles</div>
          <div className="text-sm text-slate-400">Historical chart + live updates (WS)</div>
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
        <div className="flex items-center gap-2">
          {entry != null && stopLoss != null && target != null ? (
            <Badge tone="info">
              Stock Plan · Entry {entry.toFixed(2)} · SL {stopLoss.toFixed(2)} · Target {target.toFixed(2)}
              {rr != null && Number.isFinite(rr) ? ` · RR ${rr.toFixed(2)}` : ''}
            </Badge>
          ) : null}
          {useAdaptiveAlgo && adaptive.data?.ok && adaptiveSide !== 'HOLD' && adaptiveEntry != null ? (
            <Badge tone={adaptiveSide === 'BUY' ? 'good' : adaptiveSide === 'SELL' ? 'warn' : 'neutral'}>
              Adaptive Algo · {adaptiveSide}
              {adaptiveStrength != null ? ` · ${Math.round(adaptiveStrength * 100)}%` : ''}
            </Badge>
          ) : null}
          <Badge tone={streamState.connected ? 'good' : 'warn'}>
            WS {streamState.connected ? 'Connected' : 'Disconnected'}
          </Badge>
          {streamState.lastMessageAt ? (
            <div className="text-xs text-slate-400">
              last msg {Math.round((Date.now() - streamState.lastMessageAt) / 1000)}s ago
            </div>
          ) : null}
        </div>
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
                <option value="1d">1d</option>
              </Select>
            </div>
            <div>
              <div className="mb-1 text-xs text-slate-400">Limit (candles)</div>
              <Input
                type="number"
                min={10}
                max={10000}
                value={limit}
                onChange={(e) => setLimit(parseInt(e.target.value || '200', 10))}
              />
              <div className="mt-1 text-[11px] text-slate-500">Max candles returned by the API.</div>
              {recommendedLimit && recommendedLimit > limit ? (
                <div className="mt-1 flex items-center justify-between gap-2 text-[11px] text-amber-200/80">
                  <div>
                    Selected range may contain ~{recommendedLimit.toLocaleString()} candles; increase limit to see the full window.
                  </div>
                  <button
                    type="button"
                    className="rounded-md border border-amber-200/30 bg-amber-200/10 px-2 py-1 text-[11px] text-amber-100 hover:bg-amber-200/15"
                    onClick={() => setLimit(recommendedLimit)}
                    title="Set limit to cover the selected window (approx)"
                  >
                    Set to {recommendedLimit}
                  </button>
                </div>
              ) : null}
            </div>
            <div className="flex items-end">
              <div className="text-xs text-slate-500">
                Data comes from `/api/candles/historical`.
              </div>
            </div>
          </div>

          <div className="mt-4 grid grid-cols-1 gap-3 md:grid-cols-4">
            <div>
              <div className="mb-1 text-xs text-slate-400">Start (local)</div>
              <Input type="datetime-local" value={start} onChange={(e) => setStart(e.target.value)} />
            </div>
            <div>
              <div className="mb-1 text-xs text-slate-400">End (local)</div>
              <Input type="datetime-local" value={end} onChange={(e) => setEnd(e.target.value)} />
            </div>
            <div>
              <div className="mb-1 text-xs text-slate-400">Bulk Sessions (days)</div>
              <Input
                type="number"
                min={1}
                max={1000}
                value={bulkSessions}
                onChange={(e) => setBulkSessions(Number(e.target.value))}
              />
              <div className="mt-1 text-[11px] text-slate-500">Fetch last N trading sessions into cache.</div>
            </div>
            <div className="flex items-end gap-2">
              <Button variant="secondary" onClick={() => load.mutate()} disabled={load.isPending}>
                {load.isPending ? 'Loading…' : 'Load Range'}
              </Button>
              <Button variant="secondary" onClick={() => bulkLoad.mutate()} disabled={bulkLoad.isPending}>
                {bulkLoad.isPending ? 'Bulk…' : 'Bulk Load'}
              </Button>
              <Button variant={useAdaptiveAlgo ? 'primary' : 'secondary'} onClick={() => setUseAdaptiveAlgo((v) => !v)}>
                Adaptive Algo: {useAdaptiveAlgo ? 'ON' : 'OFF'}
              </Button>
            </div>
          </div>

          <div className="mt-4 grid grid-cols-1 gap-3 md:grid-cols-4">
            <div className="md:col-span-2">
              <div className="mb-1 text-xs text-slate-600 dark:text-slate-400">Add indicator</div>
              <div className="flex items-center gap-2">
                <Select value={indicatorPick} onChange={(e) => setIndicatorPick(e.target.value)}>
                  <option value="ema">EMA (custom)</option>
                  <option value="vwap">VWAP</option>
                  <option value="fair" disabled={!useAdaptiveAlgo}>
                    Fair Value (Adaptive)
                  </option>
                  <option value="bandU" disabled={!useAdaptiveAlgo}>
                    Upper Band (Adaptive)
                  </option>
                  <option value="bandL" disabled={!useAdaptiveAlgo}>
                    Lower Band (Adaptive)
                  </option>
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
                >
                  Add
                </Button>
              </div>
              {!useAdaptiveAlgo ? (
                <div className="mt-1 text-[11px] text-slate-500">Enable Adaptive Algo to add Fair Value/Bands.</div>
              ) : null}
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
                        {id.startsWith('ema:')
                          ? `EMA ${id.split(':')[1]}`
                          : id === 'vwap'
                            ? 'VWAP'
                            : id === 'fair'
                              ? 'Fair Value'
                              : id === 'bandU'
                                ? 'Upper Band'
                                : id === 'bandL'
                                  ? 'Lower Band'
                                  : id}
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
          {(load.data || load.error || bulkLoad.data || bulkLoad.error) && (
            <div className="mt-4 grid grid-cols-1 gap-3 lg:grid-cols-2">
              <Details title="Load Range Response" data={load.data ?? load.error} />
              <Details title="Bulk Load Response" data={bulkLoad.data ?? bulkLoad.error} />
            </div>
          )}
        </CardBody>
      </Card>

      {useAdaptiveAlgo ? (
        <Card>
          <CardHeader>
            <CardTitle>Adaptive Algo</CardTitle>
          </CardHeader>
          <CardBody>
            {adaptive.isLoading ? (
              <div className="text-sm text-slate-400">Computing signal + optimizer…</div>
            ) : adaptive.isError ? (
              <div className="text-sm text-rose-300">Failed to compute adaptive signal</div>
            ) : (
              <div className="space-y-2">
                <div className="flex flex-wrap items-center gap-2">
                  <Badge tone={adaptiveSide === 'BUY' ? 'good' : adaptiveSide === 'SELL' ? 'warn' : 'neutral'}>{adaptiveSide}</Badge>
                  {adaptiveSignal?.reason ? <div className="text-xs text-slate-400">{String(adaptiveSignal.reason)}</div> : null}
                  {bestWin != null ? (
                    <Badge tone={bestWin >= 0.6 ? 'good' : bestWin <= 0.45 ? 'warn' : 'neutral'}>
                      Optimal win {Math.round(bestWin * 100)}%
                      {bestTrades != null ? ` · trades ${bestTrades}` : ''}
                    </Badge>
                  ) : null}
                  {best?.period != null && best?.multiplier != null ? (
                    <Badge tone="neutral">Optimal p={best.period} · m={Number(best.multiplier).toFixed(1)}</Badge>
                  ) : null}
                </div>

                {trendRows?.length ? (
                  <div className="flex flex-wrap items-center gap-2">
                    <div className="text-xs text-slate-500">Trend</div>
                    {trendRows.slice(0, 8).map((r) => (
                      <Badge
                        key={r.interval}
                        tone={r.trend === 'UP' ? 'good' : r.trend === 'DOWN' ? 'warn' : 'neutral'}
                      >
                        {r.interval} {r.trend}
                      </Badge>
                    ))}
                  </div>
                ) : null}
              </div>
            )}
          </CardBody>
        </Card>
      ) : null}

      <Card>
        <CardHeader>
          <CardTitle>Chart</CardTitle>
        </CardHeader>
        <CardBody className="h-[420px]">
          {candles.isLoading ? (
            <div className="text-sm text-slate-400">Loading…</div>
          ) : inst.isError ? (
            <div className="space-y-2">
              <div className="text-sm text-rose-300">Invalid instrument</div>
              <div className="text-xs text-slate-400">
                Enter a valid NSE EQ symbol (e.g. RELIANCE) or an Upstox `instrument_key` (e.g. NSE_EQ|INE002A01018).
              </div>
            </div>
          ) : candles.isError ? (
            <div className="space-y-2">
              <div className="text-sm text-rose-300">Failed to load candles</div>
              <div className="text-xs text-slate-400">
                {(() => {
                  const err = candles.error as ApiError | unknown;
                  if (typeof err === 'object' && err && 'detail' in err) {
                    return String((err as any).detail);
                  }
                  return 'Unknown error.';
                })()}
              </div>
              <div className="text-xs text-slate-500">
                If the market is closed and you have no cached candles, try “Bulk Load” for the last session.
              </div>
            </div>
          ) : (
            <CandlestickChart
              candles={chartData}
              overlays={{
                supports: strategy.data?.supports,
                resistances: strategy.data?.resistances,
                entry: useAdaptiveAlgo && adaptiveSide !== 'HOLD' ? adaptiveEntry ?? strategy.data?.entry : strategy.data?.entry,
                stop_loss: useAdaptiveAlgo && adaptiveSide !== 'HOLD' ? adaptiveSL ?? strategy.data?.stop_loss : strategy.data?.stop_loss,
                target: useAdaptiveAlgo && adaptiveSide !== 'HOLD' ? adaptiveTarget ?? strategy.data?.target : strategy.data?.target,
              }}
              lineOverlays={lineOverlays}
              tradeZone={{ enabled: true, side: 'buy' }}
              watermark={inst.data?.canonical_symbol ?? inst.data?.tradingsymbol}
            />
          )}
        </CardBody>
      </Card>

      {resolvedKey ? (
        <Card>
          <CardHeader>
            <CardTitle>AI Trader</CardTitle>
          </CardHeader>
          <CardBody>
            <AiTraderPanel instrumentKey={resolvedKey} interval={interval} lookbackDays={5} horizonSteps={12} />
          </CardBody>
        </Card>
      ) : null}

      <Card>
        <CardHeader>
          <CardTitle>Live Candle (last)</CardTitle>
        </CardHeader>
        <CardBody>
          <Details title="Raw" data={lastCandle} />
        </CardBody>
      </Card>
    </>
  );
}
