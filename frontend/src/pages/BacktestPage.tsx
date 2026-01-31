import { useMemo, useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { apiPost } from '@/lib/api';
import { Card, CardBody, CardHeader, CardTitle } from '@/components/ui/Card';
import { Input } from '@/components/ui/Input';
import { Select } from '@/components/ui/Select';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { useInstrumentResolve } from '@/hooks/useInstrumentResolve';
import { InstrumentSearch } from '@/components/instruments/InstrumentSearch';
import { Details } from '@/components/ui/Details';
import { useAppDispatch, useAppSelector } from '@/store/hooks';
import { setInstrumentInput, setInstrumentPick } from '@/store/selectionSlice';

type Interval = '1m' | '3m' | '5m' | '15m' | '30m' | '1h' | '1d';

function toIsoFromDatetimeLocal(value: string) {
  // value like "2026-01-30T10:30"
  if (!value) return '';
  const d = new Date(value);
  return d.toISOString();
}

export function BacktestPage() {
  const dispatch = useAppDispatch();
  const instrument = useAppSelector((s) => s.selection.instrument);

  const [interval, setInterval] = useState<Interval>('15m');
  const [start, setStart] = useState('2026-01-01T09:15');
  const [end, setEnd] = useState('2026-01-30T15:29');

  const instrumentQuery = instrument.instrument_key ?? instrument.input;
  const inst = useInstrumentResolve(instrumentQuery);
  const resolvedKey = instrument.instrument_key ?? inst.data?.instrument_key;

  const run = useMutation({
    mutationFn: () =>
      apiPost<any>(
        `/api/backtest/run?instrument_key=${encodeURIComponent(resolvedKey ?? instrumentQuery)}&interval=${encodeURIComponent(
          interval
        )}&start=${encodeURIComponent(toIsoFromDatetimeLocal(start))}&end=${encodeURIComponent(
          toIsoFromDatetimeLocal(end)
        )}`
      ),
  });

  const tone = useMemo(() => {
    if (run.isPending) return 'info';
    if (run.isError) return 'bad';
    if (run.isSuccess) return 'good';
    return 'neutral';
  }, [run.isPending, run.isError, run.isSuccess]);

  return (
    <>
      <div>
        <div className="text-xl font-semibold">Backtest</div>
        <div className="text-sm text-slate-400">Runs `/api/backtest/run` and persists calibration</div>
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
          <CardTitle>Parameters</CardTitle>
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
              <div className="mb-1 text-xs text-slate-400">Start (local)</div>
              <Input type="datetime-local" value={start} onChange={(e) => setStart(e.target.value)} />
            </div>
            <div>
              <div className="mb-1 text-xs text-slate-400">End (local)</div>
              <Input type="datetime-local" value={end} onChange={(e) => setEnd(e.target.value)} />
            </div>
          </div>

          <div className="mt-4 flex items-center gap-3">
            <Button onClick={() => run.mutate()} disabled={run.isPending}>
              {run.isPending ? 'Running…' : 'Run Backtest'}
            </Button>
            <Badge tone={tone as any}>{run.isPending ? 'RUNNING' : run.isError ? 'FAILED' : run.isSuccess ? 'DONE' : 'IDLE'}</Badge>
          </div>

          <div className="mt-4">
            <Details title="Result" data={run.data ?? run.error} />
          </div>
        </CardBody>
      </Card>
    </>
  );
}
