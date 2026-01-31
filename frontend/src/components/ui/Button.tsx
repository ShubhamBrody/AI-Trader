import { clsx } from 'clsx';
import type { ButtonHTMLAttributes } from 'react';

type Props = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger';
  size?: 'sm' | 'md';
};

export function Button({ className, variant = 'primary', size = 'md', ...props }: Props) {
  const base =
    'inline-flex items-center justify-center gap-2 rounded-lg font-medium transition focus:outline-none focus:ring-2 focus:ring-sky-500/40 disabled:opacity-50 disabled:pointer-events-none';

  const variants: Record<NonNullable<Props['variant']>, string> = {
    primary: 'bg-sky-500 text-white hover:bg-sky-400',
    secondary:
      'bg-slate-800 text-slate-100 hover:bg-slate-700 border border-slate-700',
    ghost: 'bg-transparent hover:bg-slate-800/60 text-slate-100',
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
