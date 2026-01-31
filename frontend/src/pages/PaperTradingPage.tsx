import { useMemo, useState } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import { apiGet, apiPost } from '@/lib/api';
import { Card, CardBody, CardHeader, CardTitle } from '@/components/ui/Card';
import { Input } from '@/components/ui/Input';
import { Select } from '@/components/ui/Select';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { useInstrumentResolve } from '@/hooks/useInstrumentResolve';
import { InstrumentSearch } from '@/components/instruments/InstrumentSearch';
import { KeyValueGrid } from '@/components/ui/KeyValue';
import { Details } from '@/components/ui/Details';
import { InstrumentLabel } from '@/components/instruments/InstrumentLabel';
import { useAppDispatch, useAppSelector } from '@/store/hooks';
import { setInstrumentInput, setInstrumentPick } from '@/store/selectionSlice';

type Interval = '1m' | '5m' | '15m' | '1h' | '1d';

export function PaperTradingPage() {
  const dispatch = useAppDispatch();
  const instrument = useAppSelector((s) => s.selection.instrument);

  const [interval, setInterval] = useState<Interval>('1m');
  const [accountBalance, setAccountBalance] = useState(100000);
  const [lotSize, setLotSize] = useState(1);
  const [depositAmount, setDepositAmount] = useState(10000);
  const [withdrawAmount, setWithdrawAmount] = useState(1000);

  const instrumentQuery = instrument.instrument_key ?? instrument.input;
  const inst = useInstrumentResolve(instrumentQuery);
  const resolvedKey = instrument.instrument_key ?? inst.data?.instrument_key;

  const account = useQuery({
    queryKey: ['paper-account'],
    queryFn: () => apiGet<any>('/api/paper/account'),
    retry: 0,
  });

  const positions = useQuery({
    queryKey: ['paper-positions'],
    queryFn: () => apiGet<any>('/api/paper/positions'),
    retry: 0,
  });

  const deposit = useMutation({
    mutationFn: () => apiPost<any>(`/api/paper/deposit?amount=${depositAmount}`),
    onSuccess: () => {
      account.refetch();
    },
  });

  const withdraw = useMutation({
    mutationFn: () => apiPost<any>(`/api/paper/withdraw?amount=${withdrawAmount}`),
    onSuccess: () => {
      account.refetch();
    },
  });

  const execute = useMutation({
    mutationFn: () =>
      apiPost<any>(
        `/api/paper/execute?instrument_key=${encodeURIComponent(resolvedKey ?? instrumentQuery)}&interval=${encodeURIComponent(
          interval
        )}&account_balance=${accountBalance}&lot_size=${lotSize}`
      ),
    onSuccess: () => {
      account.refetch();
      positions.refetch();
    },
  });

  const tone = useMemo(() => {
    const status = execute.data?.status ?? execute.data?.result?.status;
    if (status === 'ACCEPTED' || status === 'OK') return 'good';
    if (status === 'REJECTED') return 'bad';
    return 'neutral';
  }, [execute.data]);

  return (
    <>
      <div>
        <div className="text-xl font-semibold">Paper Trading</div>
        <div className="text-sm text-slate-400">Strategy → Risk → Execution → Paper Broker</div>
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

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Execute</CardTitle>
          </CardHeader>
          <CardBody className="space-y-4">
            <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
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
              <div>
                <div className="mb-1 text-xs text-slate-400">Account Balance</div>
                <Input type="number" value={accountBalance} onChange={(e) => setAccountBalance(Number(e.target.value))} />
              </div>
              <div>
                <div className="mb-1 text-xs text-slate-400">Lot Size</div>
                <Input type="number" value={lotSize} onChange={(e) => setLotSize(Number(e.target.value))} />
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <Button
                variant="secondary"
                onClick={() => {
                  const cash = Number(account.data?.cash_balance);
                  if (!Number.isNaN(cash) && cash > 0) setAccountBalance(cash);
                }}
                type="button"
              >
                Use Paper Cash
              </Button>
              <div className="text-xs text-slate-500">
                Execution uses `account_balance` for sizing.
              </div>
            </div>

            <div className="flex items-center gap-3">
              <Button onClick={() => execute.mutate()} disabled={execute.isPending}>
                {execute.isPending ? 'Executing…' : 'Execute Paper Trade'}
              </Button>
              {execute.data?.status ? <Badge tone={tone as any}>{execute.data.status}</Badge> : null}
            </div>

            {execute.data?.status && execute.data?.reason ? (
              <div
                className={
                  execute.data.status === 'REJECTED' ? 'text-sm text-amber-200' : 'text-sm text-slate-300'
                }
              >
                Reason: {String(execute.data.reason)}
              </div>
            ) : null}

            <Details title="Execution Result" data={execute.data ?? execute.error} />
          </CardBody>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Account</CardTitle>
          </CardHeader>
          <CardBody>
            <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
              <div>
                <div className="mb-1 text-xs text-slate-400">Deposit</div>
                <div className="flex items-center gap-2">
                  <Input type="number" value={depositAmount} onChange={(e) => setDepositAmount(Number(e.target.value))} />
                  <Button variant="secondary" onClick={() => deposit.mutate()} disabled={deposit.isPending}>
                    Add
                  </Button>
                </div>
              </div>
              <div>
                <div className="mb-1 text-xs text-slate-400">Withdraw</div>
                <div className="flex items-center gap-2">
                  <Input type="number" value={withdrawAmount} onChange={(e) => setWithdrawAmount(Number(e.target.value))} />
                  <Button variant="secondary" onClick={() => withdraw.mutate()} disabled={withdraw.isPending}>
                    Take
                  </Button>
                </div>
              </div>
            </div>

            <div className="mt-4 space-y-3">
              <KeyValueGrid items={[{ label: 'Cash Balance', value: account.data?.cash_balance }]} cols={1} />
              <Details title="Raw" data={account.data ?? account.error} />
            </div>
          </CardBody>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Positions</CardTitle>
        </CardHeader>
        <CardBody>
          {positions.isLoading ? (
            <div className="text-sm text-slate-400">Loading…</div>
          ) : positions.isError ? (
            <div className="text-sm text-rose-300">Failed to load positions</div>
          ) : ((positions.data?.positions as any[]) ?? []).length === 0 ? (
            <div className="text-sm text-slate-400">No positions.</div>
          ) : (
            <div className="space-y-3">
              <div className="overflow-auto">
                <table className="w-full text-sm">
                  <thead className="text-left text-xs text-slate-400">
                    <tr>
                      <th className="py-2">Instrument</th>
                      <th>Qty</th>
                      <th>Avg</th>
                    </tr>
                  </thead>
                  <tbody>
                    {((positions.data?.positions as any[]) ?? []).map((p, idx) => (
                      <tr key={p.instrument_key ?? idx} className="border-t border-slate-800">
                        <td className="py-2">
                          <InstrumentLabel query={String(p.instrument_key)} />
                        </td>
                        <td className="text-slate-200">{p.quantity}</td>
                        <td className="text-slate-200">{p.average_price}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <Details title="Raw" data={positions.data} />
            </div>
          )}
        </CardBody>
      </Card>
    </>
  );
}
