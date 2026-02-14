import { useEffect, useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { apiGet } from '@/lib/api';
import { Card, CardBody, CardHeader, CardTitle } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { InstrumentSearch } from '@/components/instruments/InstrumentSearch';
import { useInstrumentResolve } from '@/hooks/useInstrumentResolve';
import { KeyValueGrid } from '@/components/ui/KeyValue';
import { Details } from '@/components/ui/Details';
import { useAppDispatch, useAppSelector } from '@/store/hooks';
import { setInstrumentInput, setInstrumentPick } from '@/store/selectionSlice';

type PredictSummaryEntry = {
  ts_ms: number;
  instrument_key: string;
  canonical_symbol?: string;
  name?: string | null;
  action?: string;
  overall_confidence?: number;
  uncertainty?: number;
  ensemble_agreement?: number;
  model_version?: string;
  data_quality_score?: number;
  predicted_ohlc?: { open?: number; high?: number; low?: number; close?: number };
  sentences: string[];
};

const SUMMARY_STORAGE_KEY = 'aitrader.ai_predict.summary.v1';
const ONE_DAY_MS = 24 * 60 * 60 * 1000;

function safeNum(v: any): number | null {
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function fmtPct01(v: any): string {
  const n = safeNum(v);
  if (n == null) return '—';
  return `${Math.round(n * 100)}%`;
}

function fmtNum(v: any, digits = 2): string {
  const n = safeNum(v);
  if (n == null) return '—';
  return n.toFixed(digits);
}

function marketOpenIstNow(now = new Date()): boolean {
  try {
    const fmt = new Intl.DateTimeFormat('en-GB', {
      timeZone: 'Asia/Kolkata',
      weekday: 'short',
      hour: '2-digit',
      minute: '2-digit',
      hour12: false,
    });
    const parts = fmt.formatToParts(now);
    const wd = (parts.find((p) => p.type === 'weekday')?.value ?? '').toLowerCase();
    const hh = Number(parts.find((p) => p.type === 'hour')?.value ?? '0');
    const mm = Number(parts.find((p) => p.type === 'minute')?.value ?? '0');
    const isWeekend = wd.startsWith('sat') || wd.startsWith('sun');
    if (isWeekend) return false;
    const mins = hh * 60 + mm;
    const open = 9 * 60 + 15;
    const close = 15 * 60 + 30;
    return mins >= open && mins <= close;
  } catch {
    return false;
  }
}

function loadSummaryHistory(): PredictSummaryEntry[] {
  try {
    const raw = window.localStorage.getItem(SUMMARY_STORAGE_KEY);
    const arr = raw ? (JSON.parse(raw) as any[]) : [];
    const now = Date.now();
    if (!Array.isArray(arr)) return [];
    return arr
      .filter((x) => x && typeof x.ts_ms === 'number' && now - x.ts_ms <= ONE_DAY_MS)
      .slice(-250);
  } catch {
    return [];
  }
}

function saveSummaryHistory(entries: PredictSummaryEntry[]) {
  try {
    window.localStorage.setItem(SUMMARY_STORAGE_KEY, JSON.stringify(entries.slice(-250)));
  } catch {
    // ignore
  }
}

function buildSummarySentences(args: {
  canonical_symbol?: string;
  name?: string | null;
  exchange?: string;
  segment?: string;
  action?: string;
  predicted_ohlc?: { open?: any; high?: any; low?: any; close?: any };
  overall_confidence?: any;
  uncertainty?: any;
  ensemble_agreement?: any;
  model_version?: string;
  data_quality_score?: any;
  market_open?: boolean;
}): string[] {
  const sym = args.canonical_symbol ?? 'this instrument';
  const nm = args.name ? String(args.name) : null;
  const ex = args.exchange ? String(args.exchange) : null;
  const seg = args.segment ? String(args.segment) : null;

  const action = args.action ? String(args.action).toUpperCase() : 'HOLD';
  const o = safeNum(args.predicted_ohlc?.open);
  const c = safeNum(args.predicted_ohlc?.close);
  const h = safeNum(args.predicted_ohlc?.high);
  const l = safeNum(args.predicted_ohlc?.low);

  const upDown = o != null && c != null ? (c > o ? 'up' : c < o ? 'down' : 'flat') : null;

  const confidence = fmtPct01(args.overall_confidence);
  const unc = args.uncertainty != null ? fmtNum(args.uncertainty, 3) : '—';
  const agreement = args.ensemble_agreement != null ? fmtPct01(args.ensemble_agreement) : '—';
  const dq = args.data_quality_score != null ? fmtNum(args.data_quality_score, 2) : '—';

  const model = args.model_version ? String(args.model_version) : 'unknown';
  const modelIsFallback = /rules|fallback/i.test(model);
  const marketOpen = Boolean(args.market_open);

  const s1 = nm ? `${sym} (${nm})${ex || seg ? ` on ${[ex, seg].filter(Boolean).join('/')}` : ''}.` : `${sym}${ex || seg ? ` on ${[ex, seg].filter(Boolean).join('/')}` : ''}.`;

  const s2 =
    upDown && o != null && c != null
      ? `The next-hour projection is ${upDown}: open ${o.toFixed(2)} → close ${c.toFixed(2)}${h != null && l != null ? ` (range ${l.toFixed(2)}–${h.toFixed(2)})` : ''}.`
      : `A next-hour OHLC projection is available, but direction couldn’t be derived reliably.`;

  const s3 = `Decision: ${action}. Confidence ${confidence}, uncertainty ${unc}${agreement !== '—' ? `, ensemble agreement ${agreement}` : ''}. Data quality ${dq}.`;

  const s4 = modelIsFallback
    ? `Model note: ${model} (rules/fallback), not a trained model output.`
    : `Model: ${model}.`;

  const s5 = marketOpen
    ? `Market status: looks OPEN (IST clock).`
    : `Market status: looks CLOSED (IST clock) — signals may be informational until open.`;

  const s6 =
    upDown && action !== 'HOLD'
      ? `Why: the projected move is ${upDown}, aligning with a ${action} bias, scaled by confidence/uncertainty.`
      : `Why: confidence/uncertainty and projection direction suggest staying FLAT right now.`;

  return [s1, s2, s3, s4, s5, s6];
}

export function AIPredictPage() {
  const dispatch = useAppDispatch();
  const instrument = useAppSelector((s) => s.selection.instrument);
  const instrumentQuery = instrument.instrument_key ?? instrument.input;
  const inst = useInstrumentResolve(instrumentQuery);
  const resolvedKey = instrument.instrument_key ?? inst.data?.instrument_key;

  const [summaryHistory, setSummaryHistory] = useState<PredictSummaryEntry[]>(() => loadSummaryHistory());

  const q = useQuery({
    queryKey: ['ai-predict', resolvedKey ?? instrumentQuery],
    queryFn: () => apiGet<any>(`/api/ai/predict?instrument_key=${encodeURIComponent(resolvedKey ?? instrumentQuery)}`),
    enabled: !inst.isError,
    retry: 0,
  });

  const action = q.data?.action as string | undefined;
  const tone = action === 'BUY' ? 'good' : action === 'SELL' ? 'bad' : 'neutral';
  const positionSide = action === 'BUY' ? 'LONG' : action === 'SELL' ? 'SHORT' : action ? 'FLAT' : undefined;

  const marketOpen = useMemo(() => marketOpenIstNow(), []);

  const latestSummary = useMemo(() => {
    if (!q.data || q.isError || q.isLoading) return null;
    const sentences = buildSummarySentences({
      canonical_symbol: inst.data?.canonical_symbol,
      name: inst.data?.name,
      exchange: inst.data?.exchange,
      segment: inst.data?.segment,
      action: q.data?.action,
      predicted_ohlc: q.data?.predicted_ohlc,
      overall_confidence: q.data?.overall_confidence,
      uncertainty: q.data?.uncertainty,
      ensemble_agreement: q.data?.ensemble_agreement,
      model_version: q.data?.model_version,
      data_quality_score: q.data?.data_quality_score,
      market_open: marketOpen,
    });

    return {
      ts_ms: Date.now(),
      instrument_key: String(q.data?.instrument_key ?? resolvedKey ?? instrumentQuery ?? ''),
      canonical_symbol: inst.data?.canonical_symbol,
      name: inst.data?.name,
      action: q.data?.action,
      overall_confidence: safeNum(q.data?.overall_confidence) ?? undefined,
      uncertainty: safeNum(q.data?.uncertainty) ?? undefined,
      ensemble_agreement: safeNum(q.data?.ensemble_agreement) ?? undefined,
      model_version: typeof q.data?.model_version === 'string' ? q.data.model_version : undefined,
      data_quality_score: safeNum(q.data?.data_quality_score) ?? undefined,
      predicted_ohlc: q.data?.predicted_ohlc,
      sentences,
    } satisfies PredictSummaryEntry;
  }, [q.data, q.isError, q.isLoading, inst.data, instrumentQuery, marketOpen, resolvedKey]);

  useEffect(() => {
    if (!latestSummary) return;
    const ts = (q.data?.timestamp as string | undefined) ?? '';
    if (!ts) return;

    setSummaryHistory((prev) => {
      const now = Date.now();
      const trimmed = (prev ?? []).filter((e) => e && typeof e.ts_ms === 'number' && now - e.ts_ms <= ONE_DAY_MS);
      const last = trimmed.length ? trimmed[trimmed.length - 1] : null;
      // De-dupe rapid re-renders for same instrument/action.
      if (
        last &&
        last.instrument_key === latestSummary.instrument_key &&
        last.action === latestSummary.action &&
        Math.abs(last.ts_ms - latestSummary.ts_ms) < 3000
      ) {
        return trimmed;
      }
      const next = [...trimmed, latestSummary].slice(-250);
      saveSummaryHistory(next);
      return next;
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [q.data?.timestamp]);

  return (
    <>
      <div>
        <div className="text-xl font-semibold">AI Predict</div>
        <div className="text-sm text-slate-400">Next-hour prediction + confidence + uncertainty</div>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Query</CardTitle>
        </CardHeader>
        <CardBody className="grid grid-cols-1 gap-3 md:grid-cols-3">
          <div>
            <InstrumentSearch
              value={instrument.input}
              onChange={(v) => dispatch(setInstrumentInput(v))}
              onPick={(p) => dispatch(setInstrumentPick(p))}
              label="Stock"
            />
          </div>
          <div className="flex items-end gap-2">
            {action ? <Badge tone={tone}>{action}</Badge> : null}
            {positionSide ? <Badge tone={tone}>{positionSide}</Badge> : null}
            {typeof q.data?.overall_confidence === 'number' ? (
              <Badge tone="info">conf {Math.round(q.data.overall_confidence * 100)}%</Badge>
            ) : null}
            {typeof q.data?.model_version === 'string' && q.data.model_version.length ? (
              <Badge tone="neutral">model {q.data.model_version}</Badge>
            ) : null}
          </div>
        </CardBody>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Summary (last 24h)</CardTitle>
        </CardHeader>
        <CardBody>
          <div className="flex flex-wrap items-center gap-2">
            {typeof q.data?.model_version === 'string' && q.data.model_version.length ? (
              <Badge tone="neutral">model {q.data.model_version}</Badge>
            ) : null}
            <Badge tone={marketOpen ? 'good' : 'warn'}>{marketOpen ? 'Market open (IST)' : 'Market closed (IST)'}</Badge>
            <Badge tone="neutral">retention 24h</Badge>
          </div>

          {latestSummary ? (
            <div className="mt-3 rounded-xl border border-slate-200 bg-white/70 p-3 text-sm text-slate-700 dark:border-slate-800 dark:bg-slate-950/40 dark:text-slate-300">
              <div className="space-y-1">
                {latestSummary.sentences.map((s, i) => (
                  <div key={i}>{s}</div>
                ))}
              </div>
            </div>
          ) : (
            <div className="mt-3 text-sm text-slate-400">Run a prediction to generate a summary.</div>
          )}

          <div className="mt-4">
            {summaryHistory.length ? (
              <div className="max-h-[320px] overflow-auto rounded-xl border border-slate-200 bg-white/70 p-3 dark:border-slate-800 dark:bg-slate-950/40">
                <div className="text-xs font-semibold text-slate-700 dark:text-slate-200">Timeline</div>
                <div className="mt-2 space-y-3">
                  {summaryHistory
                    .slice()
                    .reverse()
                    .map((e, idx) => (
                      <div key={`${e.ts_ms}-${idx}`} className="flex gap-3">
                        <div className="w-[86px] shrink-0 text-[11px] text-slate-500">
                          {new Date(e.ts_ms).toLocaleTimeString()}
                        </div>
                        <div className="flex-1 rounded-lg border border-slate-200 bg-white/60 p-2 dark:border-slate-800 dark:bg-slate-950/30">
                          <div className="flex flex-wrap items-center gap-2">
                            <Badge tone="neutral">{e.canonical_symbol ?? e.instrument_key}</Badge>
                            {e.action ? <Badge tone={e.action === 'BUY' ? 'good' : e.action === 'SELL' ? 'bad' : 'neutral'}>{e.action}</Badge> : null}
                            {e.overall_confidence != null ? <Badge tone="info">conf {fmtPct01(e.overall_confidence)}</Badge> : null}
                            {e.uncertainty != null ? <Badge tone="neutral">unc {fmtNum(e.uncertainty, 3)}</Badge> : null}
                          </div>
                          <div className="mt-2 text-[12px] text-slate-700 dark:text-slate-300">
                            {(e.sentences ?? []).slice(0, 2).join(' ')}
                          </div>
                        </div>
                      </div>
                    ))}
                </div>
              </div>
            ) : (
              <div className="text-sm text-slate-400">No timeline entries yet.</div>
            )}
          </div>
        </CardBody>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Result</CardTitle>
        </CardHeader>
        <CardBody>
          {q.isLoading ? (
            <div className="text-sm text-slate-400">Loading…</div>
          ) : inst.isError ? (
            <div className="text-sm text-rose-300">Unknown instrument</div>
          ) : q.isError ? (
            <div className="text-sm text-rose-300">Prediction failed</div>
          ) : (
            <div className="space-y-3">
              <div className="flex flex-wrap items-center gap-2">
                {inst.data ? (
                  <Badge tone="neutral">
                    {inst.data.canonical_symbol}{inst.data.name ? ` — ${inst.data.name}` : ''}
                  </Badge>
                ) : null}
                {typeof q.data?.uncertainty === 'number' ? (
                  <Badge tone="neutral">uncertainty {q.data.uncertainty.toFixed(3)}</Badge>
                ) : null}
                {typeof q.data?.ensemble_agreement === 'number' ? (
                  <Badge tone="neutral">agreement {Math.round(q.data.ensemble_agreement * 100)}%</Badge>
                ) : null}
              </div>

              <KeyValueGrid
                cols={3}
                items={[
                  { label: 'Predicted Open', value: q.data?.predicted_ohlc?.open },
                  { label: 'Predicted High', value: q.data?.predicted_ohlc?.high },
                  { label: 'Predicted Low', value: q.data?.predicted_ohlc?.low },
                  { label: 'Predicted Close', value: q.data?.predicted_ohlc?.close },
                  { label: 'Overall Conf.', value: q.data?.overall_confidence != null ? `${Math.round(q.data.overall_confidence * 100)}%` : null },
                  { label: 'Data Quality', value: q.data?.data_quality_score },
                ]}
              />

              {Array.isArray(q.data?.reasons) && q.data.reasons.length ? (
                <div className="rounded-xl border border-slate-200 bg-white/70 p-3 dark:border-slate-800 dark:bg-slate-950/40">
                  <div className="text-xs font-semibold text-slate-700 dark:text-slate-300">Reasons</div>
                  <div className="mt-2 flex flex-wrap gap-2">
                    {q.data.reasons.slice(0, 6).map((r: string, idx: number) => (
                      <Badge key={idx} tone="neutral">
                        {r}
                      </Badge>
                    ))}
                  </div>
                </div>
              ) : null}

              <Details title="Raw" data={q.data} />
            </div>
          )}
        </CardBody>
      </Card>
    </>
  );
}
