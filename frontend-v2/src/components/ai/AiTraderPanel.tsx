import { useQuery } from '@tanstack/react-query';
import { apiGet, type ApiError } from '@/lib/api';
import { Badge } from '@/components/ui/Badge';
import type { TraderDecisionResponse } from '@/types/trader';

export function AiTraderPanel(props: {
  instrumentKey: string;
  interval: string;
  horizonSteps?: number;
  lookbackMinutes?: number;
  lookbackDays?: number;
}) {
  const { instrumentKey, interval, horizonSteps = 12, lookbackMinutes, lookbackDays } = props;

  const qs = new URLSearchParams({
    instrument_key: instrumentKey,
    interval,
    horizon_steps: String(horizonSteps),
  });
  if (typeof lookbackDays === 'number') qs.set('lookback_days', String(lookbackDays));
  if (typeof lookbackMinutes === 'number') qs.set('lookback_minutes', String(lookbackMinutes));

  const q = useQuery({
    queryKey: ['ai-trader', instrumentKey, interval, horizonSteps, lookbackMinutes ?? null, lookbackDays ?? null],
    queryFn: () => apiGet<TraderDecisionResponse>(`/api/trader/decision?${qs.toString()}`),
    enabled: Boolean(instrumentKey),
    refetchInterval: 10_000,
    retry: 0,
  });

  if (q.isLoading) {
    return <div className="text-sm text-slate-400">AI Trader: loading…</div>;
  }

  if (q.isError) {
    const err = q.error as ApiError | unknown;
    const detail = typeof err === 'object' && err && 'detail' in err ? String((err as any).detail) : 'Unknown error.';
    return <div className="text-sm text-rose-300">AI Trader error: {detail}</div>;
  }

  const d = q.data;
  const aiAction = String(d?.ai?.action ?? 'HOLD');
  const aiConf = d?.ai?.confidence;
  const aiUnc = d?.ai?.uncertainty;
  const finalAction = String(d?.decision?.action ?? aiAction);

  const positionSide = finalAction === 'BUY' ? 'LONG' : finalAction === 'SELL' ? 'SHORT' : 'FLAT';

  const tone = finalAction === 'BUY' ? 'good' : finalAction === 'SELL' ? 'warn' : 'neutral';
  const riskMult = d?.decision?.risk_multiplier;
  const riskFrac = d?.decision?.risk_fraction_suggested;

  const plan = d?.decision?.plan ?? null;
  const tdir = String((d as any)?.trend_confluence?.dir ?? '');
  const tconf = (d as any)?.trend_confluence?.confidence;

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-2">
        <Badge tone={tone as any}>{finalAction}</Badge>
        <Badge tone={finalAction === 'BUY' ? 'good' : finalAction === 'SELL' ? 'warn' : 'neutral'}>{positionSide}</Badge>
        {typeof aiConf === 'number' ? (
          <Badge tone={aiConf >= 0.7 ? 'good' : aiConf <= 0.45 ? 'warn' : 'neutral'}>AI conf {Math.round(aiConf * 100)}%</Badge>
        ) : null}
        {typeof aiUnc === 'number' ? (
          <Badge tone={aiUnc <= 0.35 ? 'good' : aiUnc >= 0.7 ? 'warn' : 'neutral'}>unc {Math.round(aiUnc * 100)}%</Badge>
        ) : null}
        {tdir ? (
          <Badge tone={tdir === 'up' ? 'good' : tdir === 'down' ? 'warn' : 'neutral'}>
            confluence {tdir}
            {typeof tconf === 'number' ? ` ${Math.round(Number(tconf) * 100)}%` : ''}
          </Badge>
        ) : null}
        {typeof riskMult === 'number' ? <Badge tone="neutral">risk× {riskMult.toFixed(2)}</Badge> : null}
        {typeof riskFrac === 'number' ? <Badge tone="neutral">risk {Math.round(riskFrac * 10000) / 100}%</Badge> : null}
        {d?.ai?.model ? <div className="text-xs text-slate-500">model {String(d.ai.model)}</div> : null}
      </div>

      {plan ? (
        <div className="flex flex-wrap items-center gap-2 text-xs text-slate-300">
          <div className="text-slate-500">Plan</div>
          <Badge tone={plan.side === 'buy' ? 'good' : plan.side === 'sell' ? 'warn' : 'neutral'}>{String(plan.side ?? '').toUpperCase()}</Badge>
          <div>entry {fmt(plan.entry)}</div>
          <div className="text-slate-500">·</div>
          <div>SL {fmt(plan.stop)}</div>
          <div className="text-slate-500">·</div>
          <div>TP {fmt(plan.target)}</div>
          {plan.source ? <div className="text-slate-500">· {plan.source}</div> : null}
        </div>
      ) : (
        <div className="text-xs text-slate-500">No trade plan (HOLD / not confirmed).</div>
      )}

      <div className="grid grid-cols-2 gap-2 text-xs text-slate-300 md:grid-cols-4">
        <div>
          <div className="text-slate-500">EMA20</div>
          <div>{fmt(d?.indicators?.ema20)}</div>
        </div>
        <div>
          <div className="text-slate-500">SMA20</div>
          <div>{fmt(d?.indicators?.sma20)}</div>
        </div>
        <div>
          <div className="text-slate-500">RSI14</div>
          <div>{fmt(d?.indicators?.rsi14)}</div>
        </div>
        <div>
          <div className="text-slate-500">MACD hist</div>
          <div>{fmt(d?.indicators?.macd?.hist)}</div>
        </div>
      </div>

      {Array.isArray(d?.decision?.reasons) && d!.decision!.reasons!.length ? (
        <div className="text-[11px] text-slate-500">
          {d!.decision!.reasons!.slice(0, 6).map((r, i) => (
            <div key={i}>• {String(r)}</div>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function fmt(v: unknown): string {
  if (typeof v !== 'number' || !isFinite(v)) return '—';
  if (Math.abs(v) >= 100) return v.toFixed(2);
  if (Math.abs(v) >= 10) return v.toFixed(3);
  return v.toFixed(4);
}
