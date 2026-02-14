import { useState } from 'react';

export function Details({ title = 'Details', data }: { title?: string; data: any }) {
  const [open, setOpen] = useState(false);

  return (
    <div className="rounded-2xl border border-slate-200 bg-white/70 dark:border-slate-800 dark:bg-slate-950/30">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between px-4 py-2.5 text-left text-xs text-slate-700 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-900/40"
      >
        <span className="font-semibold">{title}</span>
        <span className="text-slate-500">{open ? 'Hide' : 'Show'}</span>
      </button>
      {open ? (
        <pre className="max-h-[360px] overflow-auto rounded-b-2xl bg-slate-50 p-4 text-xs text-slate-800 dark:bg-slate-950/50 dark:text-slate-200">
          {JSON.stringify(data, null, 2)}
        </pre>
      ) : null}
    </div>
  );
}
