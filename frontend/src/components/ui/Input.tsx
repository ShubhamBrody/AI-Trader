import { clsx } from 'clsx';
import type { InputHTMLAttributes } from 'react';

export function Input({ className, ...props }: InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      className={clsx(
        'h-10 w-full rounded-lg border border-slate-800 bg-slate-950/60 px-3 text-sm text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-sky-500/40',
        className
      )}
      {...props}
    />
  );
}
