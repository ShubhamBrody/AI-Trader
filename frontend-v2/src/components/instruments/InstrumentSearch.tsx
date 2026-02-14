import { useEffect, useMemo, useRef, useState } from 'react';
import { Input } from '@/components/ui/Input';
import { Badge } from '@/components/ui/Badge';
import { useDebouncedValue } from '@/hooks/useDebouncedValue';
import { useInstrumentSuggest } from '@/hooks/useInstrumentSuggest';

export type InstrumentPick = {
  canonical_symbol: string;
  tradingsymbol: string;
  instrument_key: string;
  name?: string | null;
};

export function InstrumentSearch({
  label = 'Stock',
  value,
  onChange,
  onPick,
  placeholder = 'Type a symbol (e.g. RELIANCE, TATASTEEL) or paste instrument_key',
}: {
  label?: string;
  value: string;
  onChange: (value: string) => void;
  onPick?: (pick: InstrumentPick) => void;
  placeholder?: string;
}) {
  const [open, setOpen] = useState(false);
  const debounced = useDebouncedValue(value, 500);
  const suggest = useInstrumentSuggest(debounced, 5);

  const items = (suggest.data?.results ?? []) as InstrumentPick[];

  const boxRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    function onDocClick(e: MouseEvent) {
      const el = boxRef.current;
      if (!el) return;
      if (!el.contains(e.target as any)) setOpen(false);
    }
    document.addEventListener('mousedown', onDocClick);
    return () => document.removeEventListener('mousedown', onDocClick);
  }, []);

  const status = useMemo(() => {
    if (!debounced.trim()) return null;
    if (suggest.isLoading) return 'searching';
    if (suggest.isError) return 'offline';
    return null;
  }, [debounced, suggest.isLoading, suggest.isError]);

  return (
    <div ref={boxRef} className="relative">
      <div className="mb-1 flex items-center justify-between">
        <div className="text-xs text-slate-500 dark:text-slate-400">{label}</div>
        {status === 'searching' ? <Badge tone="info">Searching</Badge> : null}
        {status === 'offline' ? <Badge tone="warn">No instruments</Badge> : null}
      </div>

      <Input
        value={value}
        placeholder={placeholder}
        onChange={(e) => {
          onChange(e.target.value);
          setOpen(true);
        }}
        onFocus={() => setOpen(true)}
        autoComplete="off"
      />

      {open && value.trim().length > 0 ? (
        <div className="absolute z-50 mt-2 w-full overflow-hidden rounded-xl border border-slate-200 bg-white shadow-xl dark:border-slate-800 dark:bg-slate-950/95">
          {suggest.isLoading ? (
            <div className="px-3 py-2 text-xs text-slate-500 dark:text-slate-400">Searching…</div>
          ) : items.length === 0 ? (
            <div className="px-3 py-2 text-xs text-slate-500 dark:text-slate-400">No matches.</div>
          ) : (
            <div className="max-h-[260px] overflow-auto">
              {items.map((it) => (
                <button
                  key={it.instrument_key}
                  type="button"
                  onClick={() => {
                    onPick?.(it);
                    onChange(it.canonical_symbol);
                    setOpen(false);
                  }}
                  className="flex w-full items-start justify-between gap-3 border-t border-slate-200 px-3 py-2 text-left hover:bg-slate-100 dark:border-slate-800 dark:hover:bg-slate-900/60"
                >
                  <div>
                    <div className="text-sm font-semibold text-slate-900 dark:text-slate-100">{it.canonical_symbol}</div>
                    <div className="text-xs text-slate-600 dark:text-slate-400">
                      {it.name ? it.name : it.tradingsymbol}
                      <span className="text-slate-600"> · </span>
                      <span className="text-slate-500">{it.tradingsymbol}</span>
                    </div>
                  </div>
                  <div className="text-[10px] text-slate-500">{it.instrument_key}</div>
                </button>
              ))}
            </div>
          )}
        </div>
      ) : null}
    </div>
  );
}
