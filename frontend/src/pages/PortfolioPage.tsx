import { useQuery } from '@tanstack/react-query';
import { apiGet } from '@/lib/api';
import { Card, CardBody, CardHeader, CardTitle } from '@/components/ui/Card';
import { KeyValueGrid } from '@/components/ui/KeyValue';
import { Details } from '@/components/ui/Details';
import { InstrumentLabel } from '@/components/instruments/InstrumentLabel';
import { clsx } from 'clsx';

function num(x: any): number | null {
  const v = Number(x);
  return Number.isFinite(v) ? v : null;
}

function pickNumber(obj: any, keys: string[]): number | null {
  for (const k of keys) {
    const v = num(obj?.[k]);
    if (v !== null) return v;
  }
  return null;
}

function fmt(n: any) {
  if (n === null || n === undefined || n === '') return '—';
  const v = Number(n);
  if (Number.isFinite(v)) return v.toLocaleString(undefined, { maximumFractionDigits: 2 });
  return String(n);
}

function holdingMetrics(h: any) {
  const qty = pickNumber(h, ['quantity', 'qty', 'net_quantity', 'net_qty']) ?? 0;
  const avg = pickNumber(h, ['average_price', 'avg_price', 'buy_avg_price']) ?? 0;
  const ltp = pickNumber(h, ['last_price', 'ltp', 'last_traded_price']) ?? null;
  const invested = qty > 0 ? qty * avg : null;
  const value = ltp !== null && qty > 0 ? qty * ltp : null;

  const rawPnl = pickNumber(h, ['pnl', 'p&l', 'unrealised_pnl', 'unrealized_pnl', 'unrealized', 'unrealised']);
  const pnl = rawPnl !== null ? rawPnl : invested !== null && value !== null ? value - invested : null;
  const retPct = invested && invested !== 0 && pnl !== null ? (pnl / invested) * 100 : null;

  // Day P&L: best-effort from common broker fields.
  const dayPnl = pickNumber(h, ['day_pnl', 'pnl_day', 'day_change', 'day_change_value']);
  const dayPct = pickNumber(h, ['day_change_percentage', 'day_change_pct', 'pnl_day_pct']);

  return { qty, avg, ltp, invested, value, pnl, retPct, dayPnl, dayPct };
}

export function PortfolioPage() {
  const balance = useQuery({
    queryKey: ['portfolio-balance'],
    queryFn: () => apiGet<any>('/api/portfolio/balance'),
    retry: 0,
    refetchInterval: 15000,
    refetchIntervalInBackground: true,
  });

  const holdings = useQuery({
    queryKey: ['portfolio-holdings'],
    queryFn: () => apiGet<any>('/api/portfolio/holdings'),
    retry: 0,
    refetchInterval: 15000,
    refetchIntervalInBackground: true,
  });

  const positions = useQuery({
    queryKey: ['portfolio-positions'],
    queryFn: () => apiGet<any>('/api/portfolio/positions'),
    retry: 0,
    refetchInterval: 15000,
    refetchIntervalInBackground: true,
  });

  const live = useQuery({
    queryKey: ['portfolio-live-summary'],
    queryFn: () => apiGet<any>('/api/portfolio/live/summary'),
    retry: 0,
    refetchInterval: 15000,
    refetchIntervalInBackground: true,
  });

  return (
    <>
      <div>
        <div className="text-xl font-semibold">Portfolio</div>
        <div className="text-sm text-slate-400">Live (Upstox) + Paper portfolio snapshot</div>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <Card>
          <CardHeader>
            <CardTitle>Balance</CardTitle>
          </CardHeader>
          <CardBody>
            {balance.isLoading ? (
              <div className="text-sm text-slate-400">Loading…</div>
            ) : balance.isError ? (
              <div className="text-sm text-rose-300">Failed to load balance</div>
            ) : (
              <div className="space-y-3">
                <KeyValueGrid
                  items={[
                    { label: 'Paper Balance', value: balance.data?.paper_balance },
                    { label: 'Live Balance', value: balance.data?.live_balance?.total },
                    { label: 'Total', value: balance.data?.total_balance },
                  ]}
                />
                <Details title="Raw" data={balance.data} />
              </div>
            )}
          </CardBody>
        </Card>

        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Live Equity Holdings (Upstox)</CardTitle>
          </CardHeader>
          <CardBody>
            {live.isLoading ? (
              <div className="text-sm text-slate-400">Loading…</div>
            ) : live.isError ? (
              <div className="text-sm text-rose-300">Failed to load live holdings</div>
            ) : ((live.data?.equity_holdings as any[]) ?? []).length === 0 ? (
              <div className="text-sm text-slate-400">No live equity holdings (or not logged in).</div>
            ) : (
              <div className="space-y-3">
                {(() => {
                  const raw = (live.data?.equity_holdings as any[]) ?? [];
                  const rows = raw
                    .map((h) => {
                      const m = holdingMetrics(h);
                      const key = String(h.instrument_token ?? h.instrument_key ?? h.tradingsymbol ?? h.trading_symbol ?? '');
                      return { h, m, key };
                    })
                    .sort((a, b) => {
                      const ap = a.m.pnl ?? 0;
                      const bp = b.m.pnl ?? 0;
                      return Math.abs(bp) - Math.abs(ap);
                    });

                  const totals = rows.reduce(
                    (acc, r) => {
                      if (r.m.invested !== null) acc.invested += r.m.invested;
                      if (r.m.value !== null) acc.value += r.m.value;
                      if (r.m.pnl !== null) acc.pnl += r.m.pnl;
                      if (r.m.dayPnl !== null) acc.dayPnl += r.m.dayPnl;
                      if (r.m.invested !== null) acc.investedKnown = true;
                      if (r.m.value !== null) acc.valueKnown = true;
                      if (r.m.pnl !== null) acc.pnlKnown = true;
                      if (r.m.dayPnl !== null) acc.dayPnlKnown = true;
                      return acc;
                    },
                    { invested: 0, value: 0, pnl: 0, dayPnl: 0, investedKnown: false, valueKnown: false, pnlKnown: false, dayPnlKnown: false }
                  );

                  const totalRetPct = totals.invested !== 0 && totals.pnlKnown ? (totals.pnl / totals.invested) * 100 : null;

                  return (
                    <>
                      <div className="grid grid-cols-2 gap-2">
                        <div className="rounded-xl border border-slate-800 bg-slate-950/40 p-3">
                          <div className="text-xs text-slate-400">Invested</div>
                          <div className="mt-1 text-lg font-semibold text-slate-100">
                            {totals.investedKnown ? fmt(totals.invested) : '—'}
                          </div>
                        </div>
                        <div className="rounded-xl border border-slate-800 bg-slate-950/40 p-3">
                          <div className="text-xs text-slate-400">Current</div>
                          <div className="mt-1 text-lg font-semibold text-slate-100">
                            {totals.valueKnown ? fmt(totals.value) : '—'}
                          </div>
                        </div>
                        <div className="rounded-xl border border-slate-800 bg-slate-950/40 p-3">
                          <div className="text-xs text-slate-400">Total P&L</div>
                          <div
                            className={clsx(
                              'mt-1 text-lg font-semibold',
                              totals.pnlKnown ? (totals.pnl >= 0 ? 'text-emerald-300' : 'text-rose-300') : 'text-slate-100'
                            )}
                            title={totalRetPct !== null ? `Return: ${totalRetPct.toFixed(2)}%` : undefined}
                          >
                            {totals.pnlKnown ? (totals.pnl >= 0 ? '+' : '') + fmt(totals.pnl) : '—'}
                          </div>
                          <div className="mt-0.5 text-xs text-slate-500">
                            Return {totalRetPct !== null ? `${totalRetPct >= 0 ? '+' : ''}${totalRetPct.toFixed(2)}%` : '—'}
                          </div>
                        </div>
                        <div className="rounded-xl border border-slate-800 bg-slate-950/40 p-3">
                          <div className="text-xs text-slate-400">1D P&L</div>
                          <div
                            className={clsx(
                              'mt-1 text-lg font-semibold',
                              totals.dayPnlKnown ? (totals.dayPnl >= 0 ? 'text-emerald-300' : 'text-rose-300') : 'text-slate-100'
                            )}
                          >
                            {totals.dayPnlKnown ? (totals.dayPnl >= 0 ? '+' : '') + fmt(totals.dayPnl) : '—'}
                          </div>
                          <div className="mt-0.5 text-xs text-slate-500">(best-effort from broker fields)</div>
                        </div>
                      </div>

                      <div className="max-h-[560px] overflow-y-auto overflow-x-hidden rounded-xl border border-slate-800">
                        <table className="w-full table-fixed text-sm">
                          <thead className="sticky top-0 bg-slate-950/90 text-left text-[11px] text-slate-400 backdrop-blur">
                            <tr>
                              <th className="w-[52%] px-3 py-2">Instrument</th>
                              <th className="w-[12%] py-2 pr-3 text-right">Qty</th>
                              <th className="hidden w-[12%] py-2 pr-3 text-right sm:table-cell">Avg</th>
                              <th className="hidden w-[12%] py-2 pr-3 text-right sm:table-cell">LTP</th>
                              <th className="w-[12%] py-2 pr-3 text-right">P&L</th>
                            </tr>
                          </thead>
                          <tbody>
                            {rows.map(({ h, m, key }, idx) => (
                              <tr key={h.instrument_token ?? h.instrument_key ?? key ?? idx} className="border-t border-slate-800">
                                <td className="px-3 py-2">
                                  <div className="min-w-0">
                                    <InstrumentLabel query={String(key)} />
                                    <div className="mt-0.5 flex flex-wrap items-center gap-x-3 gap-y-0.5 text-[11px] text-slate-500">
                                      <span>Inv {m.invested !== null ? fmt(m.invested) : '—'}</span>
                                      <span className="text-slate-700">·</span>
                                      <span>Val {m.value !== null ? fmt(m.value) : '—'}</span>
                                      <span className="text-slate-700">·</span>
                                      <span>
                                        Ret {m.retPct !== null ? `${m.retPct >= 0 ? '+' : ''}${m.retPct.toFixed(2)}%` : '—'}
                                      </span>
                                      {m.dayPnl !== null || m.dayPct !== null ? (
                                        <>
                                          <span className="text-slate-700">·</span>
                                          <span className={clsx((m.dayPnl ?? 0) >= 0 ? 'text-emerald-300/80' : 'text-rose-300/80')}>
                                            1D {m.dayPnl !== null ? (m.dayPnl >= 0 ? '+' : '') + fmt(m.dayPnl) : '—'}
                                            {m.dayPct !== null ? ` (${m.dayPct >= 0 ? '+' : ''}${m.dayPct.toFixed(2)}%)` : ''}
                                          </span>
                                        </>
                                      ) : null}
                                    </div>
                                  </div>
                                </td>
                                <td className="py-2 pr-3 text-right text-slate-200">{fmt(m.qty)}</td>
                                <td className="hidden py-2 pr-3 text-right text-slate-200 sm:table-cell">{fmt(m.avg)}</td>
                                <td className="hidden py-2 pr-3 text-right text-slate-200 sm:table-cell">{m.ltp !== null ? fmt(m.ltp) : '—'}</td>
                                <td
                                  className={clsx(
                                    'py-2 pr-3 text-right font-semibold',
                                    m.pnl === null ? 'text-slate-300' : m.pnl >= 0 ? 'text-emerald-300' : 'text-rose-300'
                                  )}
                                  title={m.retPct !== null ? `Return: ${m.retPct.toFixed(2)}%` : undefined}
                                >
                                  {m.pnl !== null ? (m.pnl >= 0 ? '+' : '') + fmt(m.pnl) : '—'}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </>
                  );
                })()}
                <Details title="Raw" data={live.data?.equity_holdings} />
              </div>
            )}
          </CardBody>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Live Mutual Funds (Upstox)</CardTitle>
          </CardHeader>
          <CardBody>
            {live.isLoading ? (
              <div className="text-sm text-slate-400">Loading…</div>
            ) : live.isError ? (
              <div className="text-sm text-rose-300">Failed to load MF holdings</div>
            ) : ((live.data?.mf_holdings as any[]) ?? []).length === 0 ? (
              <div className="text-sm text-slate-400">
                No MF holdings returned (MF API may be unavailable for this account).
              </div>
            ) : (
              <div className="space-y-3">
                <div className="overflow-auto">
                  <table className="w-full text-sm">
                    <thead className="text-left text-xs text-slate-400">
                      <tr>
                        <th className="py-2">Fund</th>
                        <th>Units</th>
                        <th>Value</th>
                      </tr>
                    </thead>
                    <tbody>
                      {((live.data?.mf_holdings as any[]) ?? []).map((m, idx) => (
                        <tr key={m.folio_number ?? m.scheme_code ?? m.isin ?? idx} className="border-t border-slate-800">
                          <td className="py-2 text-slate-200">{String(m.scheme_name ?? m.fund_name ?? m.name ?? m.scheme_code ?? 'MF')}</td>
                          <td className="text-slate-200">{m.units ?? m.unit_balance ?? m.quantity ?? '-'}</td>
                          <td className="text-slate-200">{m.current_value ?? m.value ?? m.amount ?? '-'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <Details title="Raw" data={live.data?.mf_holdings} />
              </div>
            )}
          </CardBody>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Paper Holdings</CardTitle>
          </CardHeader>
          <CardBody>
            {holdings.isLoading ? (
              <div className="text-sm text-slate-400">Loading…</div>
            ) : holdings.isError ? (
              <div className="text-sm text-rose-300">Failed to load holdings</div>
            ) : ((holdings.data?.paper_holdings as any[]) ?? []).length === 0 ? (
              <div className="text-sm text-slate-400">No holdings.</div>
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
                      {((holdings.data?.paper_holdings as any[]) ?? []).map((h, idx) => (
                        <tr key={h.instrument_key ?? idx} className="border-t border-slate-800">
                          <td className="py-2">
                            <InstrumentLabel query={String(h.instrument_key)} />
                          </td>
                          <td className="text-slate-200">{h.quantity}</td>
                          <td className="text-slate-200">{h.average_price}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <Details title="Raw" data={holdings.data} />
              </div>
            )}
          </CardBody>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Live Positions (Upstox)</CardTitle>
          </CardHeader>
          <CardBody>
            {live.isLoading ? (
              <div className="text-sm text-slate-400">Loading…</div>
            ) : live.isError ? (
              <div className="text-sm text-rose-300">Failed to load live positions</div>
            ) : ((live.data?.equity_positions as any[]) ?? []).length === 0 ? (
              <div className="text-sm text-slate-400">No live positions.</div>
            ) : (
              <div className="space-y-3">
                <div className="overflow-auto">
                  <table className="w-full text-sm">
                    <thead className="text-left text-xs text-slate-400">
                      <tr>
                        <th className="py-2">Instrument</th>
                        <th>Qty</th>
                        <th>Avg</th>
                        <th>LTP</th>
                        <th>P&L</th>
                      </tr>
                    </thead>
                    <tbody>
                      {((live.data?.equity_positions as any[]) ?? []).map((p, idx) => (
                        <tr key={p.instrument_token ?? p.instrument_key ?? idx} className="border-t border-slate-800">
                          <td className="py-2">
                            <InstrumentLabel query={String(p.instrument_token ?? p.instrument_key ?? p.tradingsymbol ?? p.trading_symbol ?? '')} />
                          </td>
                          <td className="text-slate-200">{p.quantity}</td>
                          <td className="text-slate-200">{p.average_price}</td>
                          <td className="text-slate-200">{p.last_price}</td>
                          <td className="text-slate-200">{p.pnl}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <Details title="Raw" data={live.data?.equity_positions} />
              </div>
            )}
          </CardBody>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Paper Positions</CardTitle>
          </CardHeader>
          <CardBody>
            {positions.isLoading ? (
              <div className="text-sm text-slate-400">Loading…</div>
            ) : positions.isError ? (
              <div className="text-sm text-rose-300">Failed to load paper positions</div>
            ) : ((positions.data?.paper_positions as any[]) ?? []).length === 0 ? (
              <div className="text-sm text-slate-400">No paper positions.</div>
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
                      {((positions.data?.paper_positions as any[]) ?? []).map((p, idx) => (
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
      </div>
    </>
  );
}
