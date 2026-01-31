import { useInstrumentResolve } from '@/hooks/useInstrumentResolve';

export function InstrumentLabel({
  query,
  fallback,
}: {
  query: string;
  fallback?: string;
}) {
  const inst = useInstrumentResolve(query);

  if (inst.data) {
    return (
      <div>
        <div className="font-medium text-slate-100">{inst.data.canonical_symbol}</div>
        <div className="text-xs text-slate-500">{inst.data.tradingsymbol}</div>
      </div>
    );
  }

  return <div className="text-sm text-slate-300">{fallback ?? query}</div>;
}
