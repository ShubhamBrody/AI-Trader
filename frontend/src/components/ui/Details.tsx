import { useState } from 'react';

export function Details({ title = 'Details', data }: { title?: string; data: any }) {
  const [open, setOpen] = useState(false);

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-950/40">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between px-3 py-2 text-left text-xs text-slate-300 hover:bg-slate-900/40"
      >
        <span className="font-semibold">{title}</span>
        <span className="text-slate-500">{open ? 'Hide' : 'Show'}</span>
      </button>
      {open ? (
        <pre className="max-h-[360px] overflow-auto rounded-b-xl bg-slate-950/60 p-3 text-xs text-slate-200">
          {JSON.stringify(data, null, 2)}
        </pre>
      ) : null}
    </div>
  );
}
