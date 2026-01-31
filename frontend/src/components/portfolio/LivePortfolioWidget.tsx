import { useQuery } from '@tanstack/react-query';
import { apiGet } from '@/lib/api';
import { clsx } from 'clsx';

function fmt(n: any) {
  if (n === null || n === undefined || n === '') return '—';
  const num = Number(n);
  if (Number.isFinite(num)) return num.toLocaleString(undefined, { maximumFractionDigits: 2 });
  return String(n);
}

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
  // If only % is available, we still show it.
  const dayPnl = pickNumber(h, ['day_pnl', 'pnl_day', 'day_change', 'day_change_value']);
  const dayPct = pickNumber(h, ['day_change_percentage', 'day_change_pct', 'pnl_day_pct']);

  return { qty, avg, ltp, invested, value, pnl, retPct, dayPnl, dayPct };
}

export function LivePortfolioWidget() {
  const live = useQuery({
    queryKey: ['portfolio-live-summary-sidebar'],
    queryFn: () => apiGet<any>('/api/portfolio/live/summary'),
    retry: 0,
    refetchInterval: 8000,
    refetchIntervalInBackground: true,
  });

  const funds = live.data?.funds;
  const holdingsCount = (live.data?.equity_holdings ?? []).length;
  const positionsCount = (live.data?.equity_positions ?? []).length;
  const mfCount = (live.data?.mf_holdings ?? []).length;

  const holdings = (live.data?.equity_holdings as any[]) ?? [];
  const totals = holdings.reduce(
    (acc, h) => {
      const m = holdingMetrics(h);
      if (m.invested !== null) acc.invested += m.invested;
      if (m.value !== null) acc.value += m.value;
      if (m.pnl !== null) acc.pnl += m.pnl;
      if (m.dayPnl !== null) acc.dayPnl += m.dayPnl;
      if (m.invested !== null) acc.investedKnown = true;
      if (m.value !== null) acc.valueKnown = true;
      if (m.pnl !== null) acc.pnlKnown = true;
      if (m.dayPnl !== null) acc.dayPnlKnown = true;
      return acc;
    },
    { invested: 0, value: 0, pnl: 0, dayPnl: 0, investedKnown: false, valueKnown: false, pnlKnown: false, dayPnlKnown: false }
  );

  const totalRetPct = totals.invested !== 0 && totals.pnlKnown ? (totals.pnl / totals.invested) * 100 : null;

  const hasErrors = live.data?.errors && Object.keys(live.data.errors).length > 0;

  return (
    <div className="mt-4 rounded-xl border border-slate-800 bg-slate-950/40 p-3">
      <div className="flex items-center justify-between">
        <div className="text-xs font-semibold text-slate-200">Live Portfolio</div>
        <div
          className={clsx(
            'text-[11px]',
            live.isLoading
              ? 'text-slate-400'
              : live.isError
                ? 'text-rose-300'
                : hasErrors
                  ? 'text-amber-200'
                  : 'text-emerald-300'
          )}
        >
          {live.isLoading ? 'Loading…' : live.isError ? 'Error' : hasErrors ? 'Partial' : 'OK'}
        </div>
      </div>

      <div className="mt-2 grid grid-cols-2 gap-2 text-xs">
        <div className="col-span-2 rounded-lg border border-slate-800 bg-slate-900/30 px-2 py-2">
          <div className="flex items-center justify-between">
            <div className="text-[11px] text-slate-400">Equity P&L</div>
            <div
              className={clsx(
                'text-xs font-semibold',
                totals.pnlKnown ? (totals.pnl >= 0 ? 'text-emerald-300' : 'text-rose-300') : 'text-slate-300'
              )}
              title={totals.pnlKnown ? `Return: ${totalRetPct !== null ? totalRetPct.toFixed(2) + '%' : '—'}` : '—'}
            >
              {totals.pnlKnown ? (totals.pnl >= 0 ? '+' : '') + fmt(totals.pnl) : '—'}
            </div>
          </div>
          <div className="mt-1 flex items-center justify-between text-[11px]">
            <div className="text-slate-500">Invested</div>
            <div className="text-slate-300">{totals.investedKnown ? fmt(totals.invested) : '—'}</div>
          </div>
          <div className="mt-0.5 flex items-center justify-between text-[11px]">
            <div className="text-slate-500">Current</div>
            <div className="text-slate-300">{totals.valueKnown ? fmt(totals.value) : '—'}</div>
          </div>
          {totals.dayPnlKnown ? (
            <div className="mt-0.5 flex items-center justify-between text-[11px]">
              <div className="text-slate-500">1D P&L</div>
              <div className={clsx(totals.dayPnl >= 0 ? 'text-emerald-300' : 'text-rose-300')}>
                {(totals.dayPnl >= 0 ? '+' : '') + fmt(totals.dayPnl)}
              </div>
            </div>
          ) : null}
        </div>

        <div className="rounded-lg border border-slate-800 bg-slate-900/30 px-2 py-1.5">
          <div className="text-[11px] text-slate-400">Holdings</div>
          <div className="font-semibold text-slate-200">{fmt(holdingsCount)}</div>
        </div>
        <div className="rounded-lg border border-slate-800 bg-slate-900/30 px-2 py-1.5">
          <div className="text-[11px] text-slate-400">Positions</div>
          <div className="font-semibold text-slate-200">{fmt(positionsCount)}</div>
        </div>
        <div className="rounded-lg border border-slate-800 bg-slate-900/30 px-2 py-1.5">
          <div className="text-[11px] text-slate-400">MF</div>
          <div className="font-semibold text-slate-200">{fmt(mfCount)}</div>
        </div>
        <div className="rounded-lg border border-slate-800 bg-slate-900/30 px-2 py-1.5">
          <div className="text-[11px] text-slate-400">Avail. Margin</div>
          <div className="font-semibold text-slate-200">{fmt(funds?.equity_available_margin ?? funds?.equity_available_margin ?? funds?.equity_available ?? funds?.total)}</div>
        </div>
      </div>

      {live.isError ? (
        <div className="mt-2 text-[11px] text-slate-500">Login to Upstox to enable live data.</div>
      ) : null}
    </div>
  );
}
