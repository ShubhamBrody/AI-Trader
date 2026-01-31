import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { apiGet } from '@/lib/api';
import { Card, CardBody, CardHeader, CardTitle } from '@/components/ui/Card';
import { Input } from '@/components/ui/Input';
import { Select } from '@/components/ui/Select';
import { Badge } from '@/components/ui/Badge';
import { useInstrumentResolve } from '@/hooks/useInstrumentResolve';
import { InstrumentSearch } from '@/components/instruments/InstrumentSearch';
import { KeyValueGrid } from '@/components/ui/KeyValue';
import { Details } from '@/components/ui/Details';
import { useAppDispatch, useAppSelector } from '@/store/hooks';
import { setInstrumentInput, setInstrumentPick } from '@/store/selectionSlice';

type Interval = '1m' | '5m' | '15m' | '1h' | '1d';

export function StrategyPage() {
  const dispatch = useAppDispatch();
  const instrument = useAppSelector((s) => s.selection.instrument);

  const [interval, setInterval] = useState<Interval>('1m');
  const [accountBalance, setAccountBalance] = useState(100000);
  const [lotSize, setLotSize] = useState(1);

  const instrumentQuery = instrument.instrument_key ?? instrument.input;
  const inst = useInstrumentResolve(instrumentQuery);
  const resolvedKey = instrument.instrument_key ?? inst.data?.instrument_key;

  const strategy = useQuery({
    queryKey: ['strategy', resolvedKey ?? instrumentQuery, interval],
    queryFn: () =>
      apiGet<any>(
        `/api/strategy?instrument_key=${encodeURIComponent(resolvedKey ?? instrumentQuery)}&interval=${encodeURIComponent(interval)}`
      ),
    enabled: !inst.isError,
    retry: 0,
  });

  const decision = useQuery({
    queryKey: ['trade-decision', resolvedKey ?? instrumentQuery, interval, accountBalance, lotSize],
    queryFn: () =>
      apiGet<any>(
        `/api/strategy/decision?instrument_key=${encodeURIComponent(resolvedKey ?? instrumentQuery)}&interval=${encodeURIComponent(
          interval
        )}&account_balance=${accountBalance}&lot_size=${lotSize}`
      ),
    enabled: !inst.isError,
    retry: 0,
  });

  const signal = strategy.data;
  const actionTone = signal?.action === 'BUY' ? 'good' : signal?.action === 'SELL' ? 'bad' : 'neutral';

  return (
    <>
      <div>
        <div className="text-xl font-semibold">Strategy</div>
        <div className="text-sm text-slate-400">Deterministic technical engine + strategy types</div>
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

      <div className="relative z-30">
        <Card>
        <CardHeader>
          <CardTitle>Query</CardTitle>
        </CardHeader>
        <CardBody>
          <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
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
                <option value="5m">5m</option>
                <option value="15m">15m</option>
                <option value="1h">1h</option>
                <option value="1d">1d</option>
              </Select>
            </div>
            <div className="flex items-end gap-2">
              {signal?.action ? <Badge tone={actionTone}>{signal.action}</Badge> : null}
              {typeof signal?.confidence === 'number' ? (
                <Badge tone="info">conf {Math.round(signal.confidence * 100)}%</Badge>
              ) : null}
            </div>
          </div>

          <div className="mt-4 grid grid-cols-1 gap-3 md:grid-cols-3">
            <div>
              <div className="mb-1 text-xs text-slate-400">Account Balance</div>
              <Input
                type="number"
                value={accountBalance}
                onChange={(e) => setAccountBalance(Number(e.target.value))}
              />
            </div>
            <div>
              <div className="mb-1 text-xs text-slate-400">Lot Size</div>
              <Input type="number" value={lotSize} onChange={(e) => setLotSize(Number(e.target.value))} />
            </div>
            <div className="flex items-end text-xs text-slate-500">
              Uses `/api/strategy/decision`.
            </div>
          </div>
        </CardBody>
        </Card>
      </div>

      <div className="relative z-10 grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Signal</CardTitle>
          </CardHeader>
          <CardBody>
            {strategy.isLoading ? (
              <div className="text-sm text-slate-400">Loading…</div>
            ) : strategy.isError ? (
              <div className="text-sm text-rose-300">Failed to compute strategy</div>
            ) : (
              <div className="space-y-3">
                <KeyValueGrid
                  items={[
                    { label: 'Action', value: strategy.data?.action },
                    {
                      label: 'Confidence',
                      value:
                        typeof strategy.data?.confidence === 'number'
                          ? `${Math.round(strategy.data.confidence * 100)}%`
                          : null,
                    },
                    { label: 'Entry', value: strategy.data?.entry },
                    { label: 'Stop Loss', value: strategy.data?.stop_loss },
                    { label: 'Target', value: strategy.data?.target },
                    { label: 'Regime', value: strategy.data?.regime },
                  ]}
                  cols={2}
                />
                <Details title="Raw" data={strategy.data} />
              </div>
            )}
          </CardBody>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Trade Decision (Sizing)</CardTitle>
          </CardHeader>
          <CardBody>
            {decision.isLoading ? (
              <div className="text-sm text-slate-400">Loading…</div>
            ) : decision.isError ? (
              <div className="text-sm text-rose-300">Failed to compute trade decision</div>
            ) : (
              <div className="space-y-3">
                <KeyValueGrid
                  items={[
                    { label: 'Status', value: decision.data?.status },
                    { label: 'Reason', value: decision.data?.reason },
                    { label: 'Qty', value: decision.data?.quantity ?? decision.data?.qty },
                    { label: 'Side', value: decision.data?.side },
                    { label: 'Entry', value: decision.data?.entry_price ?? decision.data?.entry },
                    { label: 'Stop', value: decision.data?.stop_loss ?? decision.data?.stop },
                  ]}
                  cols={2}
                />
                <Details title="Raw" data={decision.data} />
              </div>
            )}
          </CardBody>
        </Card>
      </div>
    </>
  );
}
