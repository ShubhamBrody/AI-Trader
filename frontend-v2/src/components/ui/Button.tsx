import { clsx } from 'clsx';
import type { ButtonHTMLAttributes } from 'react';

type Props = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger';
  size?: 'sm' | 'md';
};

export function Button({ className, variant = 'primary', size = 'md', ...props }: Props) {
  const base =
    'inline-flex items-center justify-center gap-2 rounded-xl font-medium transition focus:outline-none focus:ring-2 focus:ring-sky-500/30 disabled:pointer-events-none disabled:opacity-50 '
    +
    // Accent-aware focus ring (accent class lives on <html>)
    '[html.accent-violet_&]:focus:ring-violet-500/30 [html.accent-emerald_&]:focus:ring-emerald-500/30 [html.accent-rose_&]:focus:ring-rose-500/30 [html.accent-amber_&]:focus:ring-amber-500/30';

  const variants: Record<NonNullable<Props['variant']>, string> = {
    primary:
      'bg-sky-500 text-white hover:bg-sky-400 '
      +
      // Accent-aware primary button
      '[html.accent-violet_&]:bg-violet-500 [html.accent-violet_&]:hover:bg-violet-400 '
      +
      '[html.accent-emerald_&]:bg-emerald-600 [html.accent-emerald_&]:hover:bg-emerald-500 '
      +
      '[html.accent-rose_&]:bg-rose-600 [html.accent-rose_&]:hover:bg-rose-500 '
      +
      '[html.accent-amber_&]:bg-amber-600 [html.accent-amber_&]:hover:bg-amber-500',
    secondary:
      'border border-slate-200 bg-slate-50 text-slate-900 hover:bg-slate-100 dark:border-slate-800 dark:bg-slate-900/50 dark:text-slate-100 dark:hover:bg-slate-900/70',
    ghost:
      'bg-transparent text-slate-700 hover:bg-slate-200 dark:text-slate-100 dark:hover:bg-slate-900/60',
    danger: 'bg-rose-600 text-white hover:bg-rose-500',
  };

  const sizes: Record<NonNullable<Props['size']>, string> = {
    sm: 'h-9 px-3 text-sm',
    md: 'h-10 px-4 text-sm',
  };

  return (
    <button
      className={clsx(base, variants[variant], sizes[size], className)}
      {...props}
    />
  );
}
