import { useQuery } from '@tanstack/react-query';
import { apiGet } from '@/lib/api';
import type { InstrumentInfo } from '@/hooks/useInstrumentResolve';

type SuggestResponse = {
  results: InstrumentInfo[];
  count: number;
};

export function useInstrumentSuggest(q: string, limit = 5) {
  const query = (q ?? '').trim();

  return useQuery({
    queryKey: ['instrument-suggest', query, limit],
    queryFn: () => apiGet<SuggestResponse>(`/api/instruments/suggest?q=${encodeURIComponent(query)}&limit=${limit}`),
    enabled: query.length >= 1,
    retry: 0,
  });
}
