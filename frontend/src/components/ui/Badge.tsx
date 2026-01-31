import { clsx } from 'clsx';
import type { HTMLAttributes } from 'react';

type Props = HTMLAttributes<HTMLSpanElement> & {
  tone?: 'neutral' | 'good' | 'warn' | 'bad' | 'info';
};

export function Badge({ className, tone = 'neutral', ...props }: Props) {
  const tones: Record<NonNullable<Props['tone']>, string> = {
    neutral: 'bg-slate-800 text-slate-200 border-slate-700',
    good: 'bg-emerald-500/15 text-emerald-200 border-emerald-500/30',
    warn: 'bg-amber-500/15 text-amber-200 border-amber-500/30',
    bad: 'bg-rose-500/15 text-rose-200 border-rose-500/30',
    info: 'bg-sky-500/15 text-sky-200 border-sky-500/30',
  };

  return (
    <span
      className={clsx(
        'inline-flex items-center rounded-full border px-2.5 py-1 text-xs font-medium',
        tones[tone],
        className
      )}
      {...props}
    />
  );
}
