import { useQuery } from '@tanstack/react-query';
import { apiGet } from '@/lib/api';

export type InstrumentInfo = {
  canonical_symbol: string;
  exchange: string;
  segment: string;
  isin: string;
  instrument_key: string;
  tradingsymbol: string;
  name?: string | null;
};

export function useInstrumentResolve(query: string) {
  const q = (query ?? '').trim();

  return useQuery({
    queryKey: ['instrument-resolve', q],
    queryFn: () => apiGet<InstrumentInfo>(`/api/instruments/resolve?query=${encodeURIComponent(q)}`),
    enabled: q.length > 0,
    retry: 0,
  });
}
