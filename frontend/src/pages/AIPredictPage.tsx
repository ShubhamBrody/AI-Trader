import { useState } from 'react';
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

export function AIPredictPage() {
  const dispatch = useAppDispatch();
  const instrument = useAppSelector((s) => s.selection.instrument);
  const instrumentQuery = instrument.instrument_key ?? instrument.input;
  const inst = useInstrumentResolve(instrumentQuery);
  const resolvedKey = instrument.instrument_key ?? inst.data?.instrument_key;

  const q = useQuery({
    queryKey: ['ai-predict', resolvedKey ?? instrumentQuery],
    queryFn: () => apiGet<any>(`/api/ai/predict?instrument_key=${encodeURIComponent(resolvedKey ?? instrumentQuery)}`),
    enabled: !inst.isError,
    retry: 0,
  });

  const action = q.data?.action as string | undefined;
  const tone = action === 'BUY' ? 'good' : action === 'SELL' ? 'bad' : 'neutral';

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
            {typeof q.data?.overall_confidence === 'number' ? (
              <Badge tone="info">conf {Math.round(q.data.overall_confidence * 100)}%</Badge>
            ) : null}
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
                <div className="rounded-xl border border-slate-800 bg-slate-950/40 p-3">
                  <div className="text-xs font-semibold text-slate-300">Reasons</div>
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
