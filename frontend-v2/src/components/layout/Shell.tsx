import { PropsWithChildren, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { NavLink } from 'react-router-dom';
import {
  Activity,
  BarChart3,
  BrainCircuit,
  CandlestickChart,
  Clock,
  FileText,
  LayoutDashboard,
  Newspaper,
  Shield,
  Wallet,
  Zap,
} from 'lucide-react';
import { clsx } from 'clsx';
import { Button } from '@/components/ui/Button';
import { getAccent, setAccent, toggleTheme, type Accent } from '@/theme';
import { Select } from '@/components/ui/Select';
import { apiGet } from '@/lib/api';
import { LivePortfolioWidget } from '@/components/portfolio/LivePortfolioWidget';

const nav = [
  { to: '/', label: 'Dashboard', icon: LayoutDashboard },
  { to: '/market', label: 'Market', icon: Clock },
  { to: '/candles', label: 'Candles', icon: CandlestickChart },
  { to: '/intraday', label: 'Intraday', icon: Activity },
  { to: '/strategy', label: 'Strategy', icon: Shield },
  { to: '/ai', label: 'AI Predict', icon: BrainCircuit },
  { to: '/recommendations', label: 'Recommendations', icon: BarChart3 },
  { to: '/backtest', label: 'Backtest', icon: BarChart3 },
  { to: '/news', label: 'News', icon: Newspaper },
  { to: '/portfolio', label: 'Portfolio', icon: Wallet },
  { to: '/paper', label: 'Paper Trading', icon: Activity },
  { to: '/journal', label: 'Journal', icon: FileText },
  { to: '/hft', label: 'HFT (Options)', icon: Zap },
];

export function Shell({ children }: PropsWithChildren) {
  const [accent, setAccentState] = useState<Accent>(getAccent());
  const auth = useQuery({
    queryKey: ['upstox-auth'],
    queryFn: () => apiGet<any>('/api/auth/upstox/status'),
    retry: 0,
  });

  return (
    <div className="min-h-screen">
      <header className="border-b border-slate-200 bg-slate-50/80 backdrop-blur dark:border-slate-900/60 dark:bg-slate-950/30">
        <div className="mx-auto max-w-screen-2xl px-3 py-4 sm:px-4">
          <div className="flex items-center justify-between gap-4">
            <div className="flex items-baseline gap-3">
              <div className="text-sm font-semibold tracking-wide">AITrader Studio</div>
              <div className="rounded-full border border-slate-200 bg-white/70 px-2.5 py-0.5 text-[11px] text-slate-600 dark:border-slate-800 dark:bg-slate-950/40 dark:text-slate-300">
                frontend-v2
              </div>
            </div>

            <div className="flex items-center gap-2">
              {!auth.data?.logged_in ? (
                <a
                  className={clsx(
                    'inline-flex h-9 items-center rounded-xl bg-sky-500/15 px-3 text-xs font-semibold text-sky-700 hover:bg-sky-500/20 dark:text-sky-200',
                    '[html.accent-violet_&]:bg-violet-500/15 [html.accent-violet_&]:text-violet-700 [html.accent-violet_&]:hover:bg-violet-500/20 [html.dark.accent-violet_&]:text-violet-200',
                    '[html.accent-emerald_&]:bg-emerald-500/15 [html.accent-emerald_&]:text-emerald-700 [html.accent-emerald_&]:hover:bg-emerald-500/20 [html.dark.accent-emerald_&]:text-emerald-200',
                    '[html.accent-rose_&]:bg-rose-500/15 [html.accent-rose_&]:text-rose-700 [html.accent-rose_&]:hover:bg-rose-500/20 [html.dark.accent-rose_&]:text-rose-200',
                    '[html.accent-amber_&]:bg-amber-500/15 [html.accent-amber_&]:text-amber-800 [html.accent-amber_&]:hover:bg-amber-500/20 [html.dark.accent-amber_&]:text-amber-200'
                  )}
                  href={`/api/auth/upstox/login?next=${encodeURIComponent(window.location.href)}`}
                >
                  Login
                </a>
              ) : (
                <button
                  type="button"
                  className="inline-flex h-9 items-center rounded-xl border border-slate-200 bg-white/70 px-3 text-xs font-semibold text-slate-700 hover:bg-slate-100 dark:border-slate-800 dark:bg-slate-950/30 dark:text-slate-200 dark:hover:bg-slate-900/60"
                  onClick={async () => {
                    await fetch('/api/auth/upstox/logout', { method: 'POST' });
                    auth.refetch();
                  }}
                >
                  Logout
                </button>
              )}

              <Select
                className="h-9 w-[120px] text-xs"
                value={accent}
                aria-label="Accent theme"
                onChange={(e) => {
                  const next = e.target.value as Accent;
                  setAccent(next);
                  setAccentState(next);
                }}
              >
                <option value="sky">Sky</option>
                <option value="violet">Violet</option>
                <option value="emerald">Emerald</option>
                <option value="rose">Rose</option>
                <option value="amber">Amber</option>
              </Select>

              <Button variant="ghost" size="sm" onClick={toggleTheme} title="Toggle theme">
                ☼
              </Button>
            </div>
          </div>

          <nav className="mt-4 flex flex-wrap gap-2">
            {nav.map((item) => {
              const Icon = item.icon;
              return (
                <NavLink
                  key={item.to}
                  to={item.to}
                  end={item.to === '/'}
                  className={({ isActive }) =>
                    clsx(
                      'inline-flex items-center gap-2 rounded-xl border px-3 py-2 text-xs font-semibold transition',
                      isActive
                        ? clsx(
                            'border-sky-500/25 bg-sky-500/15 text-sky-700 dark:text-sky-200',
                            '[html.accent-violet_&]:border-violet-500/25 [html.accent-violet_&]:bg-violet-500/15 [html.accent-violet_&]:text-violet-700 [html.dark.accent-violet_&]:text-violet-200',
                            '[html.accent-emerald_&]:border-emerald-500/25 [html.accent-emerald_&]:bg-emerald-500/15 [html.accent-emerald_&]:text-emerald-700 [html.dark.accent-emerald_&]:text-emerald-200',
                            '[html.accent-rose_&]:border-rose-500/25 [html.accent-rose_&]:bg-rose-500/15 [html.accent-rose_&]:text-rose-700 [html.dark.accent-rose_&]:text-rose-200',
                            '[html.accent-amber_&]:border-amber-500/25 [html.accent-amber_&]:bg-amber-500/15 [html.accent-amber_&]:text-amber-800 [html.dark.accent-amber_&]:text-amber-200'
                          )
                        : 'border-slate-200 bg-white/50 text-slate-700 hover:bg-slate-100 dark:border-slate-800 dark:bg-slate-950/20 dark:text-slate-200 dark:hover:bg-slate-900/60'
                    )
                  }
                >
                  <Icon size={16} className="opacity-90" />
                  <span>{item.label}</span>
                </NavLink>
              );
            })}
          </nav>
        </div>
      </header>

      <div className="mx-auto grid max-w-screen-2xl grid-cols-1 gap-6 px-3 py-6 sm:px-4 lg:grid-cols-[1fr_320px]">
        <main className="space-y-6">{children}</main>

        <aside className="space-y-4">
          <div className="rounded-2xl border border-slate-200 bg-slate-50/80 p-4 shadow-sm backdrop-blur dark:border-slate-800 dark:bg-slate-900/30 dark:shadow-soft">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-xs font-semibold text-slate-700 dark:text-slate-200">Upstox</div>
                <div className="text-[11px] text-slate-500">Broker session</div>
              </div>
              <div
                className={clsx(
                  'text-xs',
                  auth.data?.logged_in
                    ? 'text-emerald-600 dark:text-emerald-300'
                    : auth.isLoading
                      ? 'text-slate-400'
                      : 'text-amber-700 dark:text-amber-200'
                )}
              >
                {auth.isLoading ? 'Checking…' : auth.data?.logged_in ? 'Connected' : 'Not connected'}
              </div>
            </div>

            <div className="mt-3 flex items-center gap-2">
              {!auth.data?.logged_in ? (
                <a
                  className="inline-flex h-9 items-center rounded-xl bg-sky-500/15 px-3 text-xs font-semibold text-sky-700 hover:bg-sky-500/20 dark:text-sky-200"
                  href={`/api/auth/upstox/login?next=${encodeURIComponent(window.location.href)}`}
                >
                  Login
                </a>
              ) : (
                <button
                  type="button"
                  className="inline-flex h-9 items-center rounded-xl border border-slate-200 bg-white/70 px-3 text-xs font-semibold text-slate-700 hover:bg-slate-100 dark:border-slate-800 dark:bg-slate-950/30 dark:text-slate-200 dark:hover:bg-slate-900/60"
                  onClick={async () => {
                    await fetch('/api/auth/upstox/logout', { method: 'POST' });
                    auth.refetch();
                  }}
                >
                  Logout
                </button>
              )}
              <a
                className="text-xs text-slate-500 hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-200"
                href="https://account.upstox.com/developer/apps"
                target="_blank"
                rel="noreferrer"
              >
                Apps
              </a>
            </div>
          </div>

          <LivePortfolioWidget />

          <div className="rounded-2xl border border-slate-200 bg-slate-50/60 p-4 text-xs text-slate-600 dark:border-slate-800 dark:bg-slate-900/20 dark:text-slate-500">
            Backend:{' '}
            <span className="text-slate-900 dark:text-slate-300">/api/*</span> and{' '}
            <span className="text-slate-900 dark:text-slate-300">/health</span>
          </div>
        </aside>
      </div>
    </div>
  );
}
