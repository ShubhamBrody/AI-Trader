import { useQuery } from '@tanstack/react-query';
import { apiGet } from '@/lib/api';
import { Card, CardBody, CardHeader, CardTitle } from '@/components/ui/Card';
import { Details } from '@/components/ui/Details';
import { InstrumentLabel } from '@/components/instruments/InstrumentLabel';

export function JournalPage() {
  const q = useQuery({
    queryKey: ['paper-journal'],
    queryFn: () => apiGet<any>('/api/paper/journal'),
    retry: 0,
  });

  return (
    <>
      <div>
        <div className="text-xl font-semibold">Trade Journal</div>
        <div className="text-sm text-slate-600 dark:text-slate-400">Audit trail of closed trades (paper)</div>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Journal</CardTitle>
        </CardHeader>
        <CardBody>
          {q.isLoading ? (
            <div className="text-sm text-slate-400">Loading…</div>
          ) : q.isError ? (
            <div className="text-sm text-rose-300">Failed to load journal</div>
          ) : ((q.data?.trades as any[]) ?? []).length === 0 ? (
            <div className="text-sm text-slate-400">No trades yet.</div>
          ) : (
            <div className="space-y-3">
              <div className="overflow-auto">
                <table className="w-full text-sm">
                  <thead className="text-left text-xs text-slate-400">
                    <tr>
                      <th className="py-2">Instrument</th>
                      <th>Side</th>
                      <th>Qty</th>
                      <th>Entry</th>
                      <th>Exit</th>
                      <th>PnL</th>
                    </tr>
                  </thead>
                  <tbody>
                    {((q.data?.trades as any[]) ?? []).map((t, idx) => (
                      <tr key={t.id ?? idx} className="border-t border-slate-200 dark:border-slate-800">
                        <td className="py-2">
                          {t.instrument_key ? <InstrumentLabel query={String(t.instrument_key)} /> : <span className="text-slate-400">-</span>}
                        </td>
                        <td className="text-slate-700 dark:text-slate-200">{t.side ?? t.action ?? '-'}</td>
                        <td className="text-slate-700 dark:text-slate-200">{t.quantity ?? t.qty ?? '-'}</td>
                        <td className="text-slate-700 dark:text-slate-200">{t.entry_price ?? t.entry ?? '-'}</td>
                        <td className="text-slate-700 dark:text-slate-200">{t.exit_price ?? t.exit ?? '-'}</td>
                        <td className="text-slate-700 dark:text-slate-200">{t.pnl ?? t.p_and_l ?? '-'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <Details title="Raw" data={q.data} />
            </div>
          )}
        </CardBody>
      </Card>
    </>
  );
}
