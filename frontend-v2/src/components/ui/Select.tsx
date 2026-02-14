import { clsx } from 'clsx';
import type { SelectHTMLAttributes } from 'react';

export function Select({ className, ...props }: SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select
      className={clsx(
        'h-10 w-full rounded-xl border border-slate-200 bg-white px-3 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-sky-500/30 '
          + '[html.accent-violet_&]:focus:ring-violet-500/30 [html.accent-emerald_&]:focus:ring-emerald-500/30 [html.accent-rose_&]:focus:ring-rose-500/30 [html.accent-amber_&]:focus:ring-amber-500/30 '
          + 'dark:border-slate-800 dark:bg-slate-950/40 dark:text-slate-100',
        className
      )}
      {...props}
    />
  );
}
