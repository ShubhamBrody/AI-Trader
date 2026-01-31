import { PropsWithChildren } from 'react';
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
} from 'lucide-react';
import { clsx } from 'clsx';
import { Button } from '@/components/ui/Button';
import { toggleTheme } from '@/theme';
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
];

export function Shell({ children }: PropsWithChildren) {
  const auth = useQuery({
    queryKey: ['upstox-auth'],
    queryFn: () => apiGet<any>('/api/auth/upstox/status'),
    retry: 0,
  });

  return (
    <div className="min-h-screen">
      <div className="mx-auto grid max-w-7xl grid-cols-1 gap-6 px-4 py-6 md:grid-cols-[260px_1fr]">
        <aside className="rounded-2xl border border-slate-800 bg-slate-900/50 p-4 shadow-soft">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-sm font-semibold">AITrader V2</div>
              <div className="text-xs text-slate-400">Frontend Console</div>
            </div>
            <Button variant="ghost" size="sm" onClick={toggleTheme} title="Toggle theme">
              ☼
            </Button>
          </div>

          <nav className="mt-4 space-y-1">
            {nav.map((item) => {
              const Icon = item.icon;
              return (
                <NavLink
                  key={item.to}
                  to={item.to}
                  className={({ isActive }) =>
                    clsx(
                      'flex items-center gap-3 rounded-xl px-3 py-2 text-sm transition',
                      isActive
                        ? 'bg-sky-500/15 text-sky-200 border border-sky-500/20'
                        : 'text-slate-200 hover:bg-slate-800/60'
                    )
                  }
                  end={item.to === '/'}
                >
                  <Icon size={18} className="opacity-90" />
                  <span>{item.label}</span>
                </NavLink>
              );
            })}
          </nav>

          <div className="mt-4 rounded-xl border border-slate-800 bg-slate-950/40 p-3">
            <div className="flex items-center justify-between">
              <div className="text-xs font-semibold text-slate-200">Upstox</div>
              <div
                className={
                  auth.data?.logged_in
                    ? 'text-[11px] text-emerald-300'
                    : auth.isLoading
                      ? 'text-[11px] text-slate-400'
                      : 'text-[11px] text-amber-300'
                }
              >
                {auth.isLoading ? 'Checking…' : auth.data?.logged_in ? 'Connected' : 'Not connected'}
              </div>
            </div>

            <div className="mt-2 flex items-center gap-2">
              {!auth.data?.logged_in ? (
                <a
                  className="rounded-lg bg-sky-500/20 px-3 py-1.5 text-xs font-semibold text-sky-200 hover:bg-sky-500/30"
                  href={`/api/auth/upstox/login?next=${encodeURIComponent(window.location.href)}`}
                >
                  Login
                </a>
              ) : (
                <button
                  type="button"
                  className="rounded-lg bg-slate-800 px-3 py-1.5 text-xs font-semibold text-slate-200 hover:bg-slate-700"
                  onClick={async () => {
                    await fetch('/api/auth/upstox/logout', { method: 'POST' });
                    auth.refetch();
                  }}
                >
                  Logout
                </button>
              )}
              <a
                className="text-xs text-slate-400 hover:text-slate-200"
                href="https://account.upstox.com/developer/apps"
                target="_blank"
                rel="noreferrer"
              >
                Apps
              </a>
            </div>

            {!auth.data?.logged_in ? (
              <div className="mt-2 text-[11px] text-slate-500">
                Redirect URL must match the Upstox app setting.
              </div>
            ) : null}
          </div>

          <LivePortfolioWidget />

          <div className="mt-6 text-xs text-slate-500">
            Backend endpoints: <span className="text-slate-300">/api/*</span> and{' '}
            <span className="text-slate-300">/health</span>
          </div>
        </aside>

        <main className="space-y-6">{children}</main>
      </div>
    </div>
  );
}
