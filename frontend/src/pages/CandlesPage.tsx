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

export function CandlesPage() {
  const dispatch = useAppDispatch();
  const instrument = useAppSelector((s) => s.selection.instrument);

  const [interval, setInterval] = useState<Interval>('1m');
  const [limit, setLimit] = useState(200);
  const [bulkSessions, setBulkSessions] = useState(300);
  const [start, setStart] = useState('2026-01-01T09:15');
  const [end, setEnd] = useState('2026-01-30T15:29');

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
    queryKey: ['strategy-overlay', resolvedKey ?? instrumentQuery, interval],
    queryFn: () =>
      apiGet<any>(
        `/api/strategy?instrument_key=${encodeURIComponent(resolvedKey ?? instrumentQuery)}&interval=${encodeURIComponent(interval)}`
      ),
    enabled: !inst.isError,
    retry: 0,
  });

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
                entry: strategy.data?.entry,
                stop_loss: strategy.data?.stop_loss,
                target: strategy.data?.target,
              }}
              watermark={inst.data?.canonical_symbol ?? inst.data?.tradingsymbol}
            />
          )}
        </CardBody>
      </Card>

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
