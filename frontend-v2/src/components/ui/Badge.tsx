import { clsx } from 'clsx';
import type { HTMLAttributes } from 'react';

type Props = HTMLAttributes<HTMLSpanElement> & {
  tone?: 'neutral' | 'good' | 'warn' | 'bad' | 'info';
};

export function Badge({ className, tone = 'neutral', ...props }: Props) {
  const tones: Record<NonNullable<Props['tone']>, string> = {
    neutral: 'bg-slate-100 text-slate-700 border-slate-200 dark:bg-slate-800 dark:text-slate-200 dark:border-slate-700',
    good: 'bg-emerald-500/15 text-emerald-700 border-emerald-500/30 dark:text-emerald-200',
    warn: 'bg-amber-500/15 text-amber-700 border-amber-500/30 dark:text-amber-200',
    bad: 'bg-rose-500/15 text-rose-700 border-rose-500/30 dark:text-rose-200',
    info:
      'bg-sky-500/15 text-sky-700 border-sky-500/30 dark:text-sky-200 '
      + '[html.accent-violet_&]:bg-violet-500/15 [html.accent-violet_&]:text-violet-700 [html.accent-violet_&]:border-violet-500/30 [html.dark.accent-violet_&]:text-violet-200 '
      + '[html.accent-emerald_&]:bg-emerald-500/15 [html.accent-emerald_&]:text-emerald-700 [html.accent-emerald_&]:border-emerald-500/30 [html.dark.accent-emerald_&]:text-emerald-200 '
      + '[html.accent-rose_&]:bg-rose-500/15 [html.accent-rose_&]:text-rose-700 [html.accent-rose_&]:border-rose-500/30 [html.dark.accent-rose_&]:text-rose-200 '
      + '[html.accent-amber_&]:bg-amber-500/15 [html.accent-amber_&]:text-amber-700 [html.accent-amber_&]:border-amber-500/30 [html.dark.accent-amber_&]:text-amber-200',
  };

  return (
    <span
      className={clsx(
        'inline-flex items-center rounded-xl border px-2.5 py-1 text-xs font-medium',
        tones[tone],
        className
      )}
      {...props}
    />
  );
}
