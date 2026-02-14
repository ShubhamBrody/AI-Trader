function formatCompactIso(isoLike: string): string {
  // Example input: 2026-01-30T18:53:56.568259+05:30
  const m = isoLike.match(
    /^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})(?:\.\d+)?(?:(Z)|([+-]\d{2}:\d{2}))?$/
  );
  if (!m) return isoLike;
  const [, date, time, z, offset] = m;
  if (z) return `${date} ${time} UTC`;
  if (offset === '+05:30') return `${date} ${time} IST`;
  if (offset) return `${date} ${time} ${offset}`;
  return `${date} ${time}`;
}

function formatValue(value: any): { display: string; title?: string } {
  if (value === null || value === undefined || value === '') return { display: '-' };

  if (typeof value === 'number' && Number.isFinite(value)) {
    const formatted = Number.isInteger(value)
      ? value.toLocaleString(undefined)
      : value.toLocaleString(undefined, { maximumFractionDigits: 2 });
    return { display: formatted, title: String(value) };
  }

  const raw = String(value);
  const compact = formatCompactIso(raw);
  return { display: compact, title: raw };
}

export function KeyValue({ label, value }: { label: string; value: any }) {
  const { display, title } = formatValue(value);
  const isPlaceholder = display === '-';

  return (
    <div className="grid grid-cols-2 items-center gap-3 rounded-xl border border-slate-200 bg-white/70 px-3 py-2 dark:border-slate-800 dark:bg-slate-950/30">
      <div className="min-w-0 truncate text-xs text-slate-500 dark:text-slate-400" title={label}>
        {label}
      </div>
      <div
        className={
          isPlaceholder
            ? 'min-w-0 truncate text-right text-sm text-slate-500'
            : 'min-w-0 truncate text-right text-sm text-slate-900 dark:text-slate-100'
        }
        title={title}
      >
        {display}
      </div>
    </div>
  );
}

export function KeyValueGrid({
  items,
  cols = 2,
}: {
  items: Array<{ label: string; value: any }>;
  cols?: 1 | 2 | 3 | 4;
}) {
  const cls =
    cols === 1
      ? 'grid grid-cols-1 gap-2'
      : cols === 3
        ? 'grid grid-cols-1 gap-2 md:grid-cols-3'
        : cols === 4
          ? 'grid grid-cols-1 gap-2 md:grid-cols-4'
          : 'grid grid-cols-1 gap-2 md:grid-cols-2';

  return (
    <div className={cls}>
      {items.map((it) => (
        <KeyValue key={it.label} label={it.label} value={it.value} />
      ))}
    </div>
  );
}
